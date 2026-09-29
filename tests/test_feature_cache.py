import os

import numpy as np
import pytest
from helpers import assert_no_temp_files, make_disk_cache, track_keys

from scripts.data.feature_cache import (
    FeatureCache,
    cache_signature,
    frame_image_path,
    make_feature_key,
    open_run_feature_cache,
    parse_feature_key,
)

KEY_A = make_feature_key("v1", 0, (0.1, 0.2, 0.3, 0.4))
KEY_B = make_feature_key("v1", 2, (0.1, 0.2, 0.3, 0.4))


def test_key_roundtrip():
    video_id, frame_idx, bbox = parse_feature_key(KEY_A)
    assert video_id == "v1"
    assert frame_idx == 0
    assert bbox == pytest.approx((0.1, 0.2, 0.3, 0.4), abs=1e-6)


def test_key_depends_on_bbox_and_frame():
    assert KEY_A != KEY_B
    assert KEY_A != make_feature_key("v1", 0, (0.1, 0.2, 0.9, 0.4))
    assert KEY_A != make_feature_key("v2", 0, (0.1, 0.2, 0.3, 0.4))


def test_frame_image_path_is_zero_padded():
    assert frame_image_path("/frames", "v1", 7).endswith(os.path.join("v1", "img_00007.jpg"))


def test_ensure_extracts_each_key_once(annotations_df, in_memory_cache, fake_extract, cache_calls):
    keys = track_keys(annotations_df)
    rows = in_memory_cache.ensure(keys, fake_extract, verbose=False)

    assert len(cache_calls) == len(keys)
    assert rows.tolist() == sorted(rows.tolist()), "rows should follow request order"
    assert in_memory_cache.ensure(keys, fake_extract, verbose=False).tolist() == rows.tolist()
    assert len(cache_calls) == len(keys), "a warm cache must not call the extractor"


def test_missing_keys_are_recorded_as_invalid(annotations_df, in_memory_cache, fake_extract):
    keys = track_keys(annotations_df)
    missing = keys[1]  # frame 1 is absent from the fake filesystem

    def extract(key):
        return None if key == missing else np.arange(4, dtype=np.float32)

    rows = in_memory_cache.ensure(keys, extract, verbose=False)
    valid = in_memory_cache.is_valid(rows)

    assert valid.tolist() == [True, False, True, True, True, True]
    assert in_memory_cache.rows(rows)[1].tolist() == [0.0] * 4


def test_ensure_without_extractor_raises_on_cache_miss(in_memory_cache):
    with pytest.raises(ValueError, match="no feature_extractor"):
        in_memory_cache.ensure([KEY_A, KEY_B], None, verbose=False)


def test_ensure_with_no_extractable_features_raises(in_memory_cache):
    with pytest.raises(RuntimeError, match="produced nothing"):
        in_memory_cache.ensure([KEY_A], lambda key: None, verbose=False)


def test_cache_persists_and_reloads(annotations_df, tmp_path, weights_file, fake_extract):
    cache = make_disk_cache(tmp_path, weights_file)
    keys = track_keys(annotations_df)
    rows = cache.ensure(keys, fake_extract, verbose=False)
    features = cache.rows(rows).copy()
    assert cache.directory and os.path.isdir(cache.directory)
    assert_no_temp_files(cache.directory)

    reloaded = open_run_feature_cache(str(tmp_path), weights_file, layer_index=9)
    reloaded_rows = reloaded.ensure(keys, None, verbose=False)
    assert reloaded_rows.tolist() == rows.tolist()
    assert np.array_equal(reloaded.rows(reloaded_rows), features)
    assert reloaded.feature_dim == features.shape[1]


def test_reload_does_not_call_the_extractor(annotations_df, tmp_path, weights_file, cache_calls, fake_extract):
    cache = make_disk_cache(tmp_path, weights_file)
    cache.ensure(track_keys(annotations_df), fake_extract, verbose=False)
    calls_after_first = len(cache_calls)

    reloaded = open_run_feature_cache(str(tmp_path), weights_file, layer_index=9)
    reloaded.ensure(track_keys(annotations_df), fake_extract, verbose=False)
    assert len(cache_calls) == calls_after_first


def test_different_weights_get_different_caches(tmp_path, weights_file, annotations_df, fake_extract):
    first = make_disk_cache(tmp_path, weights_file, b"weights-v1")
    first.ensure(track_keys(annotations_df), fake_extract, verbose=False)

    second = make_disk_cache(tmp_path, weights_file, b"weights-v2")
    assert second.features_path != first.features_path
    assert second.size == 0, "new weights must not reuse the old cache"
    second.ensure(track_keys(annotations_df), fake_extract, verbose=False)
    assert second.size == first.size


def test_layer_index_separates_caches(tmp_path, weights_file):
    a = open_run_feature_cache(str(tmp_path), weights_file, layer_index=9)
    b = open_run_feature_cache(str(tmp_path), weights_file, layer_index=4)
    assert a.features_path != b.features_path


def test_corrupt_meta_is_ignored_rather_than_crashing(tmp_path, weights_file, annotations_df, fake_extract):
    cache = make_disk_cache(tmp_path, weights_file)
    cache.ensure(track_keys(annotations_df), fake_extract, verbose=False)
    with open(cache.meta_path, "w") as f:
        f.write("{not json")

    reopened = FeatureCache(cache.directory, cache.signature)
    assert reopened.load().size == 0


def test_signature_mismatch_is_rejected(tmp_path, weights_file, annotations_df, fake_extract):
    cache = make_disk_cache(tmp_path, weights_file)
    cache.ensure(track_keys(annotations_df), fake_extract, verbose=False)

    other_signature = dict(cache.signature)
    other_signature["input_size"] = 224
    other_signature["weights_sha256"] = "0" * 64
    other = FeatureCache(cache.directory, other_signature)
    assert other.load().size == 0


def test_in_memory_cache_writes_nothing(tmp_path):
    cache = FeatureCache(directory=None, signature=cache_signature("x"))
    cache.ensure([KEY_A], lambda key: np.ones(3, dtype=np.float32), verbose=False)
    cache.save()
    assert os.listdir(tmp_path) == []


def test_open_run_feature_cache_requires_existing_weights(tmp_path):
    with pytest.raises(FileNotFoundError):
        open_run_feature_cache(str(tmp_path), str(tmp_path / "missing.pt"))
