import numpy as np
import pytest
import torch
from helpers import fake_features, frame_exists, track_keys

from scripts.data.feature_cache import FeatureCache, cache_signature
from scripts.dataset.sequence_dataset import CowSequenceDataset, NormalisedSeqDataset

SEQ_LEN = 4
STRIDE = 2


def build_dataset(annotations_df, cache, seed=42, **kwargs):
    kwargs.setdefault("normal_action_ids", [0, 1, 2])
    return CowSequenceDataset(
        df=annotations_df,
        frames_dir=kwargs.pop("frames_dir", "/nonexistent"),
        feature_cache=cache,
        seq_len=SEQ_LEN,
        stride=STRIDE,
        device="cpu",
        seed=seed,
        verbose=False,
        **kwargs,
    )


def warm_cache(annotations_df, missing_frame=1):
    """Cache with every key present, except one whose frame is missing."""
    cache = FeatureCache(directory=None, signature=cache_signature("test"))

    def extract(key):
        _, frame_idx, _ = key.split("|")[0], int(key.split("|")[1]), None
        if frame_idx == missing_frame:
            return None
        return fake_features(key)

    cache.ensure(track_keys(annotations_df), extract, verbose=False)
    return cache


def test_windows_are_built_from_tracks(annotations_df, in_memory_cache):
    in_memory_cache.ensure(track_keys(annotations_df), fake_features, verbose=False)
    ds = build_dataset(annotations_df, in_memory_cache)

    assert len(ds) == 2, "6 frames - 4 window + stride 2 -> 2 windows"
    assert [start for _, start in ds.sequences] == [0, 2]
    assert ds.feature_dim == fake_features(track_keys(annotations_df)[0]).shape[0]


def test_normal_action_filter_is_applied(annotations_df, in_memory_cache):
    in_memory_cache.ensure(track_keys(annotations_df), fake_features, verbose=False)
    filtered = annotations_df.copy()
    filtered["action_id"] = 7
    ds = build_dataset(filtered, in_memory_cache, normal_action_ids=[0, 1, 2])
    assert len(ds) == 0


def test_item_shape_and_dtype(annotations_df, in_memory_cache):
    in_memory_cache.ensure(track_keys(annotations_df), fake_features, verbose=False)
    item = build_dataset(annotations_df, in_memory_cache)[0]
    assert item.shape == (SEQ_LEN, fake_features(track_keys(annotations_df)[0]).shape[0])
    assert item.dtype == torch.float32


def test_dataset_requires_a_cache_or_an_extractor(annotations_df):
    with pytest.raises(ValueError, match="feature_cache or a feature_extractor"):
        CowSequenceDataset(
            df=annotations_df, frames_dir="/nonexistent", feature_extractor=None,
            seq_len=SEQ_LEN, stride=STRIDE, normal_action_ids=[0, 1, 2],
            device="cpu", verbose=False,
        )


def test_reads_are_repeatable(annotations_df):
    """The gap-fill noise must not be resampled on every __getitem__."""
    cache = warm_cache(annotations_df)
    ds = build_dataset(annotations_df, cache)
    assert torch.equal(ds[0], ds[0])


def test_gap_fill_is_stable_across_dataset_instances(annotations_df):
    a = [build_dataset(annotations_df, warm_cache(annotations_df))[0] for _ in range(2)]
    b = build_dataset(annotations_df, warm_cache(annotations_df))[0]
    assert torch.equal(a[0], a[1])
    assert torch.equal(a[0], b)


def test_gap_fill_does_not_depend_on_access_order(annotations_df):
    forward = build_dataset(annotations_df, warm_cache(annotations_df))
    backward = build_dataset(annotations_df, warm_cache(annotations_df))
    values = [forward[i] for i in range(len(forward))]
    reversed_values = [backward[i] for i in reversed(range(len(backward)))][::-1]
    assert all(torch.equal(x, y) for x, y in zip(values, reversed_values))


def test_gap_fill_changes_with_the_seed(annotations_df):
    a = build_dataset(annotations_df, warm_cache(annotations_df), seed=42)[0]
    b = build_dataset(annotations_df, warm_cache(annotations_df), seed=7)[0]
    assert not torch.equal(a, b)


def test_gap_is_filled_from_the_previous_frame_plus_noise(annotations_df):
    """Frame 1 is missing, so position 1 should be close to position 0, not equal."""
    cache = warm_cache(annotations_df)
    ds = build_dataset(annotations_df, cache, noise_std=0.05)
    window = ds[0]

    assert not torch.equal(window[0], window[1])
    assert torch.allclose(window[0], window[1], atol=0.5), "fill must stay near the previous frame"
    assert not torch.equal(window[0], window[2]), "frame 2 is present and must be used as-is"


def test_present_frames_are_returned_verbatim(annotations_df):
    cache = warm_cache(annotations_df)
    ds = build_dataset(annotations_df, cache)
    keys = track_keys(annotations_df)
    expected = torch.tensor(np.stack([fake_features(k) for k in keys[2:6]]))
    assert torch.allclose(ds[1], expected)


def test_all_missing_window_falls_back_to_zeros(annotations_df):
    cache = warm_cache(annotations_df, missing_frame=None)
    for key in track_keys(annotations_df):
        cache._features[cache._index[key]] = 0.0
        cache._valid[cache._index[key]] = False

    ds = build_dataset(annotations_df, cache)
    assert ds.feature_dim == fake_features(track_keys(annotations_df)[0]).shape[0]
    assert torch.equal(ds[0], torch.zeros(SEQ_LEN, ds.feature_dim))


def test_normalised_dataset_applies_the_supplied_statistics(annotations_df):
    cache = warm_cache(annotations_df)
    ds = build_dataset(annotations_df, cache)
    raw = torch.stack([ds[i] for i in range(len(ds))])

    dim = raw.shape[-1]
    mean = raw.reshape(-1, dim).mean(0)
    std = raw.reshape(-1, dim).std(0) + 1e-8

    norm = NormalisedSeqDataset(
        mean=mean, std=std,
        df=annotations_df, frames_dir="/nonexistent", feature_cache=cache,
        seq_len=SEQ_LEN, stride=STRIDE, normal_action_ids=[0, 1, 2],
        device="cpu", seed=42, verbose=False,
    )
    assert torch.allclose(norm[0], (ds[0] - mean) / std, atol=1e-5)


def test_dataset_never_reads_the_frames_directory(annotations_df):
    """With a fully warm cache, extraction must not touch disk at all."""
    cache = warm_cache(annotations_df)
    ds = build_dataset(
        annotations_df, cache, frames_dir="/definitely/not/here",
    )
    for i in range(len(ds)):
        ds[i]
    assert not frame_exists("/definitely/not/here", "v1", 0)
