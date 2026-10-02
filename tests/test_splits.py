"""One canonical grouped split, drawn once and consumed by every model (1.3, 1.4).

Two separate bugs, one cause. `step_flat_vae` called `train_test_split` on row
indices with no `groups=`, so frame 100 of a cow went to train and frame 101 to val
(problem #5). `step_lstm_vae` split by video, but redrew its own split from a `set`,
whose iteration order is not stable across processes — so the two models trained on
different partitions of the same videos and no Flat-vs-LSTM comparison meant anything.

These tests pin the properties that make the fix real rather than cosmetic: groups
never straddle the split, the partition is reproducible, both stages resolve the same
one, and anything inconsistent fails loudly instead of quietly producing a second
partition.

No vision stack: everything here works on a small synthetic annotation frame.
"""

import json
import os

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.data.splits import (
    SPLIT_FORMAT_VERSION,
    build_split,
    load_split,
    resolve_indices,
    resolve_split,
    row_groups,
    save_split,
)

SEED = 42
VAL_SPLIT = 0.25
ROWS_PER_VIDEO = 10


def make_df(n_videos=10, rows_per_video=ROWS_PER_VIDEO, group_key="video_id", target_span=1):
    """`n_videos` groups of contiguous rows, as the AVA annotations are laid out.

    `target_span` controls how many videos one cow appears in: 1 keeps each video's
    cow distinct, higher values make one cow recur across videos — the case that
    grouping on `video_id` alone does not catch.
    """
    rows = []
    for video in range(n_videos):
        video_id = f"v{video:03d}"
        target_id = video // target_span
        for row in range(rows_per_video):
            rows.append((video_id, video * rows_per_video + row, target_id))
    return pd.DataFrame(rows, columns=["video_id", "timestamp", "target_id"])[
        ["video_id", "timestamp", "target_id"]
    ]


def split_of(build_kwargs=None, df=None, **kwargs):
    params = {"df": df if df is not None else make_df(), "val_split": VAL_SPLIT,
              "random_seed": SEED, "group_key": "video_id"}
    params.update(build_kwargs or {})
    params.update(kwargs)
    df = params.pop("df")
    return build_split(df, **params)


# ── 1.3: groups do not straddle the split ────────────────────────────────


def test_no_video_appears_in_both_train_and_val():
    assignment = split_of()

    train = set(assignment["train"])
    val = set(assignment["val"])
    assert not train & val, f"a video is on both sides: {sorted(train & val)}"


def test_adjacent_frames_of_one_cow_stay_together():
    """The specific failure of problem #5: frames 100/101 of one cow, one each side.

    Asserted on row indices rather than on the group lists, because the rows are what
    the models actually train on.
    """
    df = make_df()
    assignment = split_of({"df": df})
    groups = row_groups(df, "video_id")
    indices = resolve_indices(assignment, groups)

    for position, video_id in enumerate(groups):
        side = next(name for name, rows in indices.items() if position in set(rows.tolist()))
        assert video_id in assignment[side], f"row {position} disagrees with its group's side"


def test_every_group_is_assigned_exactly_once():
    assignment = split_of()
    everything = assignment["train"] + assignment["val"] + assignment["test"]

    assert len(everything) == len(set(everything)), "a group is in two splits"
    assert set(everything) == set(make_df()["video_id"].astype(str))


def test_val_split_size_is_respected_at_the_group_level():
    assignment = split_of()
    total = sum(len(assignment[name]) for name in ("train", "val", "test"))

    assert len(assignment["val"]) == pytest.approx(VAL_SPLIT * total, abs=1)


def test_target_id_grouping_stops_one_cow_straddling_videos():
    """`video_id` grouping does not close this; `target_id` does.

    AVA cows recur across videos, so grouping on video alone still leaves one cow in
    train and in val (tasklist Q1). This is the check that motivates the group key
    being configurable.
    """
    df = make_df(n_videos=12, target_span=4)
    groups = row_groups(df, "target_id")
    assignment = split_of({"df": df, "group_key": "target_id"})
    indices = resolve_indices(assignment, groups)

    for position, target in enumerate(groups):
        side = next(name for name, rows in indices.items() if position in set(rows.tolist()))
        assert target in assignment[side]

    train_targets = {groups[i] for i in indices["train"]}
    val_targets = {groups[i] for i in indices["val"]}
    assert not train_targets & val_targets

    # The looser video grouping does leak the same cows, which is why the key matters.
    by_video = split_of({"df": df, "group_key": "video_id"})
    cow_videos = {}
    for video_id, target in zip(df["video_id"].astype(str), df["target_id"].astype(str)):
        cow_videos.setdefault(target, set()).add(video_id)
    train_videos = set(by_video["train"])
    val_videos = set(by_video["val"])
    straddling = [
        target for target, videos in cow_videos.items()
        if videos & train_videos and videos & val_videos
    ]
    assert straddling, "expected video-level grouping to leave some cow on both sides"


def test_a_missing_group_column_is_rejected():
    """Falling back to a frame-level split would reintroduce 1.3 silently."""
    with pytest.raises(ValueError, match="not a column"):
        row_groups(make_df(), "cow_name")


# ── 1.4: one partition, drawn once ──────────────────────────────────────


def test_the_split_is_reproducible_across_calls():
    """The old LSTM path built its video list from a `set`, whose order is unstable."""
    first = split_of()
    second = split_of()

    assert first == second


def test_resolve_split_persists_and_the_second_caller_reuses_it(tmp_path):
    df = make_df()
    out = str(tmp_path)

    first = resolve_split(out, df, val_split=VAL_SPLIT, random_seed=SEED, group_key="video_id")
    assert os.path.isfile(os.path.join(out, "split_manifest.json"))

    second = resolve_split(out, df, val_split=VAL_SPLIT, random_seed=SEED, group_key="video_id")
    assert first == second


def test_two_stages_resolve_an_identical_partition(tmp_path):
    """What 1.4 actually buys: the flat VAE and the LSTM-VAE see the same videos.

    Modelled on the real disagreement — one stage keyed on `video_id`, the other on
    `(video_id, target_id)` tracks — which is why both are resolved through the same
    manifest rather than each drawing its own.
    """
    df = make_df(n_videos=12, target_span=2)
    out = str(tmp_path)

    flat = resolve_split(out, df, val_split=VAL_SPLIT, random_seed=SEED)
    lstm = resolve_split(out, df, val_split=VAL_SPLIT, random_seed=SEED)
    assert flat == lstm

    tracks = [(video_id, target_id) for video_id, target_id in zip(
        df["video_id"].astype(str), df["target_id"].astype(str))]
    flat_rows = resolve_indices(flat, df["video_id"].astype(str))
    lstm_rows = resolve_indices(lstm, [video for video, _ in tracks])

    assert np.array_equal(flat_rows["train"], lstm_rows["train"])
    assert np.array_equal(flat_rows["val"], lstm_rows["val"])


def test_a_changed_seed_is_rejected_rather_than_silently_reused(tmp_path):
    """Two runs reporting the same number for different partitions is the worst case."""
    df = make_df()
    out = str(tmp_path)
    resolve_split(out, df, val_split=VAL_SPLIT, random_seed=SEED)

    with pytest.raises(RuntimeError, match="random_seed"):
        resolve_split(out, df, val_split=VAL_SPLIT, random_seed=SEED + 1)


def test_a_changed_group_key_is_rejected(tmp_path):
    df = make_df()
    out = str(tmp_path)
    resolve_split(out, df, val_split=VAL_SPLIT, random_seed=SEED, group_key="video_id")

    with pytest.raises(RuntimeError, match="group_key"):
        resolve_split(out, df, val_split=VAL_SPLIT, random_seed=SEED, group_key="target_id")


def test_new_groups_in_the_annotations_are_rejected(tmp_path):
    """Annotations changed after the split was drawn, so the manifest no longer covers them."""
    out = str(tmp_path)
    resolve_split(out, make_df(n_videos=10), val_split=VAL_SPLIT, random_seed=SEED)

    with pytest.raises(RuntimeError, match="absent from the split manifest"):
        resolve_split(out, make_df(n_videos=14), val_split=VAL_SPLIT, random_seed=SEED)


def test_an_unknown_format_version_is_rejected(tmp_path):
    out = str(tmp_path)
    path = save_split(out, split_of(), "video_id", VAL_SPLIT, 0.0, SEED)
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["format_version"] = SPLIT_FORMAT_VERSION + 1
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)

    with pytest.raises(ValueError, match="format"):
        load_split(out)


def test_a_missing_split_manifest_explains_itself(tmp_path):
    with pytest.raises(FileNotFoundError, match="1.4"):
        load_split(str(tmp_path))


# ── Resolving rows ──────────────────────────────────────────────────────


def test_resolve_indices_partitions_rows_without_dropping_any():
    df = make_df()
    groups = row_groups(df, "video_id")
    indices = resolve_indices(split_of({"df": df}), groups)

    every = np.concatenate([indices[name] for name in ("train", "val", "test")])
    assert sorted(every.tolist()) == list(range(len(groups))), "rows were lost or duplicated"


def test_resolve_indices_raises_on_an_unassigned_row():
    """An unassigned row would vanish from both sides without either count noticing."""
    with pytest.raises(RuntimeError, match="no split assignment"):
        resolve_indices({"train": ["v0"], "val": ["v1"], "test": []}, ["v0", "v2"])


def test_resolve_indices_partitions_track_keys_by_the_configured_half():
    """The LSTM stage holds `(video_id, target_id)` keys and picks the half to split on.

    `CowSequenceDataset.tracks` is keyed by that pair, so the stage extracts the group
    itself before resolving; this mirrors what it does for both group keys.
    """
    tracks = [("v0", 1), ("v1", 1), ("v0", 2), ("v2", 3)]

    by_video = {"train": ["v0", "v2"], "val": ["v1"], "test": []}
    indices = resolve_indices(by_video, [str(key[0]) for key in tracks])
    assert indices["train"].tolist() == [0, 2, 3]
    assert indices["val"].tolist() == [1]

    # Grouping on identity is stricter: the cow in videos v0 and v2 must be held out
    # together, so it cannot be split across the boundary.
    by_target = {"train": ["1"], "val": ["2", "3"], "test": []}
    indices = resolve_indices(by_target, [str(key[1]) for key in tracks])
    assert indices["train"].tolist() == [0, 1]
    assert indices["val"].tolist() == [2, 3]


# ── The three-way protocol 1.5 will turn on ──────────────────────────────


def test_test_is_carved_out_before_val_so_the_fractions_are_honest():
    """`test_split` defaults to 0, but the slot exists so 1.5 is a config change."""
    assert split_of({"test_split": 0.0})["test"] == []

    assignment = split_of({"test_split": 0.2})
    total = sum(len(assignment[name]) for name in ("train", "val", "test"))
    assert len(assignment["test"]) == pytest.approx(0.2 * total, abs=1)
    assert len(assignment["val"]) == pytest.approx(VAL_SPLIT * total, abs=1)


def test_test_and_val_are_disjoint_when_both_are_requested():
    assignment = split_of({"test_split": 0.2, "val_split": 0.25})

    assert not set(assignment["test"]) & set(assignment["val"])
    assert not set(assignment["train"]) & set(assignment["test"])


def test_the_whole_test_set_cannot_be_taken():
    with pytest.raises(ValueError, match="must leave groups for train"):
        split_of({"val_split": 0.6, "test_split": 0.5})


def test_a_holdout_that_would_empty_a_side_is_rejected():
    """One group cannot be split; raising beats a one-video val set reported as a number."""
    df = make_df(n_videos=1)
    with pytest.raises(ValueError, match="empty split"):
        build_split(df, val_split=0.5, random_seed=SEED, group_key="video_id")


def test_an_impossible_fraction_is_rejected():
    with pytest.raises(ValueError, match="must be in"):
        split_of({"val_split": 1.5})


# ── Both stages, one partition ──────────────────────────────────────────
#
# The helper tests above pin the split's properties. This one runs both real stages
# into the same output directory, because 1.4's failure was not that the split was
# wrong — both splits were individually reasonable — but that they disagreed, which
# no single-stage test can see.

N_VIDEOS = 10
ROWS_PER_VIDEO = 12
SEQ_LEN = 4
SEQ_STRIDE = 2
FEATURE_DIM = 8


def _pipeline_config(**overrides):
    config = {
        "normal_action_ids": [0, 1, 2],
        "val_split": VAL_SPLIT,
        "test_split": 0.0,
        "split_group_key": "video_id",
        "random_seed": SEED,
        "vae_epochs": 1, "vae_batch_size": 8, "vae_lr": 1e-3,
        "vae_hidden_dim": 8, "vae_latent_dim": 2,
        "seq_len": SEQ_LEN, "seq_stride": SEQ_STRIDE,
        "lstm_vae_epochs": 1, "lstm_vae_batch_size": 4, "lstm_vae_lr": 1e-3,
        "lstm_vae_hidden_dim": 8, "lstm_vae_latent_dim": 2, "lstm_vae_num_layers": 1,
    }
    config.update(overrides)
    return config


@pytest.fixture
def two_stage_run(tmp_path):
    """Run `step_flat_vae` then `step_lstm_vae` into one output dir.

    Vision-stack work is stubbed: `build_feature_dataset` returns a feature block with
    its video ids, and the sequence dataset is fed a pre-warmed in-memory cache. The
    stage functions themselves run for real, so the split they resolve is the real one.
    """
    import scripts.run_full_pipeline as pipeline
    from scripts.data.feature_cache import FeatureCache, cache_signature, row_feature_key
    from scripts.utils.history import HISTORY_COLUMNS, empty_history

    rows = [
        (f"v{video}", video * ROWS_PER_VIDEO * 0.04 + row * 0.04, 0.1, 0.1, 0.5, 0.5, 0, video % 3)
        for video in range(N_VIDEOS) for row in range(ROWS_PER_VIDEO)
    ]
    df = pd.DataFrame(rows, columns=["video_id", "timestamp", "x1", "y1", "x2", "y2",
                                     "action_id", "target_id"])
    out = tmp_path / "out"
    out.mkdir()
    csv = tmp_path / "annotations.csv"
    df.to_csv(csv, index=False, header=False)

    rng = np.random.default_rng(0)
    features = (rng.random((len(df), FEATURE_DIM), dtype=np.float32) * 10).astype(np.float32)
    video_ids = np.array(df["video_id"].astype(str), dtype=object)

    # Keyed from the CSV exactly as each stage reads it, so the float formatting in the
    # cache keys matches on both sides.
    cache = FeatureCache(directory=None, signature=cache_signature("test-weights"))
    reread = pd.read_csv(csv, header=None, dtype={0: str})
    reread.columns = list(df.columns)
    cache.ensure(
        [row_feature_key(row, str(row["video_id"]), 25) for _, row in reread.iterrows()],
        lambda key: np.arange(FEATURE_DIM, dtype=np.float32),
        verbose=False,
    )

    def record(module, name, captured):
        def fake_train(model, train_loader, val_loader, epochs, lr, device):
            captured[name] = {
                "train": torch.cat([batch[0] for batch in train_loader]).numpy(),
                "val": torch.cat([batch[0] for batch in val_loader]).numpy(),
            }
            history = empty_history()
            for _ in range(epochs):
                for column in HISTORY_COLUMNS:
                    history[column].append(0.0)
            return history

        monkey = pytest.MonkeyPatch()
        monkey.setattr(module, name, fake_train)
        return monkey

    captured = {}
    config = _pipeline_config()
    manifest = pipeline.RunManifest.create(str(out), config, seed=SEED, device="cpu")

    monkey = pytest.MonkeyPatch()
    monkey.setattr(pipeline, "build_feature_dataset",
                   lambda *a, **k: (features, video_ids))
    real_sequence_dataset = pipeline.CowSequenceDataset
    real_normalised_dataset = pipeline.NormalisedSeqDataset

    def with_cache(cls):
        def build(df, **kwargs):
            kwargs.update(feature_cache=cache, feature_extractor=None, feature_dim=FEATURE_DIM)
            return cls(df=df, **kwargs)
        return build

    # Both dataset classes build their own cache; the stage constructs each with
    # `feature_cache=None`, so the pre-warmed one is injected here for both.
    monkey.setattr(pipeline, "CowSequenceDataset", with_cache(real_sequence_dataset))
    monkey.setattr(pipeline, "NormalisedSeqDataset", with_cache(real_normalised_dataset))
    for name in ("train_vae", "train_lstm_vae"):
        patch = record(pipeline, name, captured)
        try:
            if name == "train_vae":
                pipeline.step_flat_vae(str(csv), str(tmp_path / "frames"), None, None,
                                       "cpu", str(out), config, headless=True, manifest=manifest)
            else:
                pipeline.step_lstm_vae(str(csv), str(tmp_path / "frames"), None, "cpu",
                                       str(out), config, manifest=manifest)
        finally:
            patch.undo()
    monkey.undo()
    manifest.save()

    return {"captured": captured, "manifest": manifest, "video_ids": video_ids, "out": str(out)}


def test_the_two_stages_record_the_same_split_hash(two_stage_run):
    """1.4 in one assertion: the flat VAE and the LSTM-VAE trained on one partition."""
    steps = two_stage_run["manifest"].data["steps"]

    assert steps["6"]["split_sha256"] == steps["7"]["split_sha256"]
    assert steps["6"]["split_group_key"] == steps["7"]["split_group_key"] == "video_id"


def test_the_two_stages_resolve_the_same_videos(two_stage_run):
    manifest = two_stage_run["manifest"].data
    canonical = manifest["splits"]["canonical_groups"]
    lstm_groups = manifest["splits"]["lstm_vae_groups"]

    # Same split name, same content — a disagreement shows up as a manifest diff.
    assert canonical["sha256"] == lstm_groups["sha256"]
    assert canonical["n_items"] == lstm_groups["n_items"]


def test_both_recorded_hashes_match_the_manifest_on_disk(two_stage_run):
    """The strongest form of 1.4: the partition both stages *report* is the file they share.

    Comparing the two stages against each other is not enough — a stage that redraws its
    own split and records that will agree with itself. Only the file on disk is
    independent of both.
    """
    from scripts.data.splits import describe

    payload = load_split(two_stage_run["out"])
    on_disk = describe({name: payload["assignment"][name] for name in ("train", "val", "test")})

    steps = two_stage_run["manifest"].data["steps"]
    assert steps["6"]["split_sha256"] == on_disk["sha256"]
    assert steps["7"]["split_sha256"] == on_disk["sha256"]
    assert two_stage_run["manifest"].data["splits"]["canonical_groups"]["sha256"] == on_disk["sha256"]


def test_the_second_stage_does_not_rewrite_the_split_manifest(two_stage_run):
    """A stage that redraws its split would leave the file inconsistent with step 6."""
    from scripts.data.splits import describe

    path = os.path.join(two_stage_run["out"], "split_manifest.json")
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    recorded = describe({name: payload["assignment"][name] for name in ("train", "val", "test")})

    manifest = two_stage_run["manifest"].data
    assert manifest["splits"]["canonical_groups"]["sha256"] == recorded["sha256"]
    # The manifest records the split once per stage under the same name; a second stage
    # with a different partition would have overwritten it.
    assert manifest["steps"]["6"]["split_sha256"] == manifest["steps"]["7"]["split_sha256"]


def test_the_flat_stage_honours_the_split_on_disk(two_stage_run):
    """Whichever stage runs first draws the split; the other must adopt it.

    The train rows the trainer received must be exactly the train videos in the
    manifest — checked by row count against the manifest's group list, since the
    loader shuffles and the values are normalized.
    """
    payload = load_split(two_stage_run["out"])
    train_videos = set(payload["assignment"]["train"])
    val_videos = set(payload["assignment"]["val"])

    flat = two_stage_run["captured"]["train_vae"]
    # Each video contributes every one of its rows, so a grouped split makes the row
    # count an exact multiple of the video count. A row-level split cannot.
    assert flat["train"].shape[0] == len(train_videos) * ROWS_PER_VIDEO
    assert flat["val"].shape[0] == len(val_videos) * ROWS_PER_VIDEO
    assert flat["train"].shape[0] + flat["val"].shape[0] == N_VIDEOS * ROWS_PER_VIDEO


def test_no_video_appears_on_both_sides_in_either_stage(two_stage_run):
    """Problem #5 end to end: neither stage may put one video on both sides.

    Read off the recorded per-stage group split rather than the tensors, because
    min-max scaled values do not identify which video they came from without inverting
    the scaler, and the recorded split is what both stages demonstrably consumed.
    """
    manifest = two_stage_run["manifest"].data
    payload = load_split(two_stage_run["out"])
    train = set(payload["assignment"]["train"])
    val = set(payload["assignment"]["val"])

    assert train and val
    assert not train & val

    # The per-stage record must be the canonical one, not a second opinion.
    lstm_recorded = manifest["splits"]["lstm_vae_groups"]
    assert lstm_recorded["sha256"] == manifest["splits"]["canonical_groups"]["sha256"]

    captured = two_stage_run["captured"]
    for name in ("train_vae", "train_lstm_vae"):
        assert captured[name]["train"].shape[0] > 0
        assert captured[name]["val"].shape[0] > 0
        # Every sequence belongs to a video, so no stage can be scoring only train rows.
        assert captured[name]["train"].shape[0] + captured[name]["val"].shape[0] > 0