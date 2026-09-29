"""Shared fixtures.

The tests deliberately avoid the vision stack (cv2 / torchvision / ultralytics):
feature extraction is injected as a callable so the caching, dataset and
determinism logic can be tested on a machine that only has torch and numpy.
"""

import pandas as pd
import pytest
from helpers import FEATURE_DIM, fake_features, track_keys

from scripts.data.feature_cache import FeatureCache, cache_signature

FRAME_STEP = 0.04  # 0.04s * 25fps = 1 frame per step


@pytest.fixture
def frames_dir(tmp_path):
    """A frames tree with one video, missing frame 1 (a real gap)."""
    video = tmp_path / "frames" / "v1"
    video.mkdir(parents=True)
    for idx in [0, 2, 3, 4, 5]:
        (video / f"img_{idx:05d}.jpg").write_bytes(b"")
    return str(tmp_path / "frames")


@pytest.fixture
def weights_file(tmp_path):
    path = tmp_path / "best.pt"
    path.write_bytes(b"fake-yolo-weights")
    return str(path)


@pytest.fixture
def annotations_df():
    """Six rows of one video/track; frame 1 is absent from disk."""
    rows = [
        ("v1", t * FRAME_STEP, 0.10 + t * 0.01, 0.1, 0.5, 0.5, 0, 0)
        for t in range(6)
    ]
    return pd.DataFrame(
        rows,
        columns=["video_id", "timestamp", "x1", "y1", "x2", "y2", "action_id", "target_id"],
    )


@pytest.fixture
def cache_calls():
    return []


@pytest.fixture
def fake_extract(cache_calls):
    """A deterministic stand-in for the YOLO backbone, recording every call."""

    def extract(key):
        cache_calls.append(key)
        return fake_features(key)

    extract.description = "fake"
    return extract


@pytest.fixture
def in_memory_cache():
    return FeatureCache(directory=None, signature=cache_signature("test-weights"))


@pytest.fixture
def keys(annotations_df):
    return track_keys(annotations_df)


@pytest.fixture
def feature_dim():
    return FEATURE_DIM
