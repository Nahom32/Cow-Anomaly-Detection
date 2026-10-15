"""The loader refuses to score with an unverifiable scaler (3.5.2).

A checkpoint now carries its normalizer, but sidecar files still exist. These
tests pin the contract: sidecars must match the embedded state, and a legacy
checkpoint with no embedded state is rejected rather than paired with whatever
sidecar happens to sit next to it.
"""

import numpy as np
import pytest
import torch

from scripts.checkpoint import load_checkpoint, save_model_checkpoint
from scripts.data.normalize import (
    MinMaxNormalizer,
    checkpoint_normalizer,
    load_checkpoint_normalizer,
    verify_normalizer_sidecars,
)


def _normalizer(seed=0):
    return MinMaxNormalizer().fit(np.random.default_rng(seed).random((30, 4)).astype(np.float32))


def _save(path, normalizer):
    save_model_checkpoint(
        path, model_state={"w": torch.zeros(2)}, normalizer=normalizer.state_dict(), config={}, seed=0,
    )
    return path


def _sidecar_files(tmp_path, normalizer):
    np.save(tmp_path / "min.npy", normalizer.min_)
    np.save(tmp_path / "max.npy", normalizer.max_)
    return {"min_": str(tmp_path / "min.npy"), "max_": str(tmp_path / "max.npy")}


def test_matching_sidecars_load(tmp_path):
    normalizer = _normalizer()
    path = _save(tmp_path / "m.pth", normalizer)
    restored = load_checkpoint_normalizer(path, sidecars=_sidecar_files(tmp_path, normalizer))
    np.testing.assert_array_equal(restored.min_, normalizer.min_)
    np.testing.assert_array_equal(restored.max_, normalizer.max_)


def test_sidecars_may_be_arrays(tmp_path):
    normalizer = _normalizer()
    path = _save(tmp_path / "m.pth", normalizer)
    restored = load_checkpoint_normalizer(
        path, sidecars={"min_": normalizer.min_, "max_": normalizer.max_}
    )
    assert restored.is_fitted


def test_mismatched_sidecar_value_is_loud(tmp_path):
    normalizer = _normalizer()
    path = _save(tmp_path / "m.pth", normalizer)
    wrong = str(tmp_path / "min.npy")
    np.save(wrong, normalizer.min_ + 1.0)
    with pytest.raises(ValueError, match="different runs"):
        load_checkpoint_normalizer(path, sidecars={"min_": wrong})


def test_mismatched_sidecar_shape_is_loud(tmp_path):
    normalizer = _normalizer()
    path = _save(tmp_path / "m.pth", normalizer)
    bad = str(tmp_path / "bad.npy")
    np.save(bad, np.zeros(7, dtype=np.float32))
    with pytest.raises(ValueError, match="mismatched scaler"):
        load_checkpoint_normalizer(path, sidecars={"min_": bad})


def test_unknown_sidecar_field_is_loud():
    normalizer = _normalizer()
    with pytest.raises(ValueError, match="no fitted field"):
        verify_normalizer_sidecars(normalizer, {"median_": np.zeros(4)})


def test_legacy_checkpoint_without_normalizer_is_rejected(tmp_path):
    path = tmp_path / "legacy.pth"
    save_model_checkpoint(path, model_state={"w": torch.zeros(2)}, config={}, seed=0)
    with pytest.raises(ValueError, match="predates 3.5"):
        load_checkpoint_normalizer(path)


def test_checkpoint_normalizer_reads_a_loaded_dict(tmp_path):
    normalizer = _normalizer()
    path = _save(tmp_path / "m.pth", normalizer)
    restored = checkpoint_normalizer(load_checkpoint(path))
    np.testing.assert_array_equal(restored.max_, normalizer.max_)


def test_checkpoint_normalizer_rejects_non_dict():
    with pytest.raises(TypeError, match="must be a dict"):
        checkpoint_normalizer(["not", "a", "checkpoint"])
