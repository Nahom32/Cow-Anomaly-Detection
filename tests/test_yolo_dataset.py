"""The detector's split is the run's split (tasks 1.6, 1.7, 1.5).

`create_yolo_dataset` used to draw its own `GroupShuffleSplit` with its own
seed, so the videos YOLO trained on were a different partition of the same data
than the VAEs trained on — two models described as "trained on 70%, validated on
30%" whose partitions did not overlap in any controlled way. These tests build
a tiny frames tree and assert that every image lands in the directory its
canonical assignment names, that the test slot exists on disk, and that a yaml
or skip-check describing a dataset that is no longer there counts as stale.
"""

import os
import shutil

import yaml

import scripts.run_full_pipeline as pipeline
from scripts.data.create_yolo_dataset import create_yolo_dataset
from scripts.data.splits import SPLIT_NAMES, load_split, resolve_split, split_params
from scripts.run_full_pipeline import (
    is_step_done,
    step_create_yaml,
    step_create_yolo_dataset,
)

FPS = 25
N_ROWS_PER_VIDEO = 4
# Explicit fractions rather than CONFIG's, so the expected partition does not
# move when a hyperparameter changes.
TEST_CONFIG = {
    "val_split": 0.25,
    "test_split": 0.1,
    "random_seed": 42,
    "split_group_key": "video_id",
}


def make_data_root(tmp_path, videos=("v0", "v1", "v2", "v3"), framed=None):
    """A dataset tree small enough to read: annotations plus one frame dir per video."""
    framed = tuple(videos) if framed is None else framed
    data_root = tmp_path / "data"
    (data_root / "annotations").mkdir(parents=True)
    rows = []
    for video in videos:
        for i in range(N_ROWS_PER_VIDEO):
            # Whole-second timestamps: `int(timestamp * fps)` lands exactly on
            # frame `25 * i` with no floating-point drift.
            rows.append(f"{video},{i},0.1,0.1,0.5,0.5,0,0")
        if video in framed:
            frames_dir = data_root / "rawframes_mini" / video
            frames_dir.mkdir(parents=True)
            for i in range(N_ROWS_PER_VIDEO):
                # The builder only checks existence and copies the bytes;
                # nothing here decodes an image, so no vision stack is needed.
                (frames_dir / f"img_{i * FPS:05d}.jpg").write_bytes(b"jpg")
    (data_root / "annotations" / "ava_train_v2.1.csv").write_text("\n".join(rows))
    return str(data_root)


def build(tmp_path, videos=("v0", "v1", "v2", "v3"), framed=None):
    data_root = make_data_root(tmp_path, videos=videos, framed=framed)
    run_dir = tmp_path / "run"
    out = tmp_path / "yolo"
    assignment = create_yolo_dataset(
        output_dir=str(out), data_root=data_root, split_dir=str(run_dir), config=TEST_CONFIG
    )
    return assignment, run_dir, out


def expected_image_keys(assignment):
    """The image each annotation row should produce, grouped by its split."""
    expected = {name: set() for name in SPLIT_NAMES}
    video_to_split = {
        str(video): name for name in SPLIT_NAMES for video in assignment[name]
    }
    for video, split in video_to_split.items():
        for i in range(N_ROWS_PER_VIDEO):
            expected[split].add(f"{video}_{i * FPS:05d}.jpg")
    return expected


def test_every_image_lands_in_the_directory_its_assignment_names(tmp_path):
    assignment, run_dir, out = build(tmp_path)

    # What the builder returned is what the run's manifest records (1.4), and a
    # fresh draw with the same params reproduces it (the split is deterministic).
    payload = load_split(str(run_dir))
    for name in SPLIT_NAMES:
        assert sorted(payload["assignment"][name]) == sorted(assignment[name])
    redawn = resolve_split(str(tmp_path / "redraw"), _annotation_frame(tmp_path),
                           **split_params(TEST_CONFIG))
    for name in SPLIT_NAMES:
        assert sorted(redawn[name]) == sorted(assignment[name])

    # Exact image sets, per split: a video's frames may not appear anywhere else.
    for name, keys in expected_image_keys(assignment).items():
        images = out / name / "images"
        on_disk = set(os.listdir(images)) if images.is_dir() else set()
        assert on_disk == keys, f"{name}: {on_disk ^ keys}"
        if keys:
            labels = set(os.listdir(out / name / "labels"))
            assert labels == {key.replace(".jpg", ".txt") for key in keys}


def test_the_test_slot_exists_on_disk(tmp_path):
    """Before 1.6 the third slot was assigned but never written; a trainer
    reading the dataset would have found no test images at all."""
    assignment, _, out = build(tmp_path)

    assert assignment["test"], "test_split=0.1 over four videos must hold one out"
    test_images = out / "test" / "images"
    assert test_images.is_dir()
    assert len(os.listdir(test_images)) == N_ROWS_PER_VIDEO


def test_a_video_without_frames_still_gets_an_assignment(tmp_path):
    """The manifest is drawn from annotations, not from frames on disk (1.4).

    Filtering first would silently drop a video whose download failed, and the
    row-level split the VAEs resolve would then disagree with this one.
    """
    assignment, _, out = build(
        tmp_path, videos=("v0", "v1", "v2", "v3", "v9"), framed=("v0", "v1", "v2", "v3")
    )

    assert sum("v9" in assignment[name] for name in SPLIT_NAMES) == 1
    for name in SPLIT_NAMES:
        images = out / name / "images"
        if images.is_dir():
            assert not any(key.startswith("v9_") for key in os.listdir(images))


def _annotation_frame(tmp_path):
    import pandas as pd

    path = tmp_path / "data" / "annotations" / "ava_train_v2.1.csv"
    df = pd.read_csv(path, header=None, dtype={0: str})
    df.columns = ["video_id", "timestamp", "x1", "y1", "x2", "y2", "action_id", "target_id"]
    return df


def test_step_two_passes_the_run_directory_and_config_to_the_builder(tmp_path, monkeypatch):
    """1.7: the split travels with the run's seed and fractions, not defaults."""
    seen = {}

    def fake_create_yolo_dataset(**kwargs):
        seen.update(kwargs)
        return {"train": ["v0"], "val": ["v1"], "test": ["v2"]}

    monkeypatch.setattr(pipeline, "create_yolo_dataset", fake_create_yolo_dataset)
    config = dict(TEST_CONFIG)
    state = step_create_yolo_dataset(str(tmp_path / "data"), str(tmp_path / "run"), config)

    assert seen["split_dir"] == str(tmp_path / "run"), "the builder must adopt the run's manifest"
    assert seen["config"] is config
    assert seen["data_root"] == str(tmp_path / "data")
    assert state["yolo_data_dir"].endswith("yolo_dataset")
    assert state["yolo_split_sha256"]


def test_data_yaml_names_test_only_when_test_has_images(tmp_path):
    """A yaml pointing at an empty directory is a claim about data that is not there."""
    assignment, _, out = build(tmp_path)
    step_create_yaml(str(out))
    with open(out / "data.yaml") as handle:
        written = yaml.safe_load(handle)
    assert written["test"] == "test/images"
    assert written["train"] == "train/images"
    assert written["val"] == "val/images"

    # The test videos lose their frames (an interrupted download, say): the
    # rebuilt dataset has no test split, and the yaml must say so.
    for video in assignment["test"]:
        shutil.rmtree(tmp_path / "data" / "rawframes_mini" / video)
    out_without_test = tmp_path / "yolo_without_test"
    create_yolo_dataset(
        output_dir=str(out_without_test), data_root=str(tmp_path / "data"),
        split_dir=str(tmp_path / "run"), config=TEST_CONFIG,
    )
    step_create_yaml(str(out_without_test))
    with open(out_without_test / "data.yaml") as handle:
        rebuilt = yaml.safe_load(handle)
    assert "test" not in rebuilt


def test_is_step_done_rejects_a_yaml_that_describes_a_different_dataset(tmp_path):
    _, _, out = build(tmp_path)
    step_create_yaml(str(out))
    state = {"data_yaml": str(out / "data.yaml"), "yolo_data_dir": str(out)}
    assert is_step_done(3, state, str(tmp_path)) is True

    # The dataset on disk changes underneath the yaml: the yaml is stale the
    # moment its named directories stop matching reality (1.5).
    shutil.rmtree(out / "test")
    assert is_step_done(3, state, str(tmp_path)) is False

    step_create_yaml(str(out))
    assert is_step_done(3, state, str(tmp_path)) is True


def test_the_builder_never_draws_its_own_split():
    """The absence of a second splitting algorithm is the assertion (1.6)."""
    import inspect

    src = inspect.getsource(create_yolo_dataset)
    assert "resolve_split(" in src
    assert "split_params(config)" in src, "1.7: fractions and seed must come from the run's config"
    assert "GroupShuffleSplit(" not in src, "a second splitter would drift from the manifest"
    assert "train_test_split(" not in src
    signature = inspect.signature(create_yolo_dataset)
    assert "split_dir" in signature.parameters
    assert "config" in signature.parameters
