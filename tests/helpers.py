"""Helpers shared by the tests. Not collected by pytest itself."""

import os

import numpy as np

from scripts.data.feature_cache import frame_image_path, make_feature_key, open_feature_cache
from scripts.dataset.sequence_dataset import frame_feature_key

FPS = 25
FEATURE_DIM = 6


def assert_no_temp_files(directory):
    leftovers = [f for f in os.listdir(directory) if f.endswith(".tmp")]
    assert not leftovers, f"temp files left behind: {leftovers}"


def track_keys(df, video_id="v1", fps=FPS):
    """The cache keys a `CowSequenceDataset` would request for `df`."""
    rows = [
        (row["timestamp"], row["x1"], row["y1"], row["x2"], row["y2"])
        for _, row in df.iterrows()
    ]
    return [frame_feature_key(row, video_id, fps) for row in rows]


def frame_exists(frames_dir, video_id, frame_idx):
    return os.path.isfile(frame_image_path(frames_dir, video_id, frame_idx))


def fake_features(key, dim=FEATURE_DIM):
    """Deterministic pseudo-features for a cache key, with no vision stack."""
    from scripts.data.feature_cache import parse_feature_key

    video_id, frame_idx, (x1, _, x2, _) = parse_feature_key(key)
    base = frame_idx + x1 + x2 + (len(video_id) * 0.01)
    return np.arange(dim, dtype=np.float32) + base


def make_disk_cache(tmp_path, weights_file, weights_bytes=b"fake-yolo-weights", layer_index=9):
    """A persistent cache bound to `weights_file`, keyed by its contents."""
    with open(weights_file, "wb") as f:
        f.write(weights_bytes)
    cache_dir = os.path.join(str(tmp_path), "feature_cache")
    return open_feature_cache(cache_dir, weights_file, layer_index=layer_index)


__all__ = [
    "FPS",
    "FEATURE_DIM",
    "assert_no_temp_files",
    "fake_features",
    "frame_exists",
    "make_disk_cache",
    "make_feature_key",
    "track_keys",
]
