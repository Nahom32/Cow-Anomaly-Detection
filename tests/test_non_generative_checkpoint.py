"""Checkpoint envelope for fitted non-neural detectors (3.4).

The envelope is the non-torch counterpart of `scripts/checkpoint.py`: same
provenance fields, no optimizer/epoch. These tests pin the two properties 3.4.1
asks for — estimator identity is stored, and a missing/unsupported artifact
fails loudly instead of loading something plausible-but-wrong.
"""

import numpy as np
import pytest

from scripts.models.non_generative.base import (
    ARTIFACT_FORMAT_VERSION,
    FittedModel,
    load_artifact,
    load_detector,
    make_envelope,
    save_artifact,
)
from scripts.models.non_generative.mahalanobis import MahalanobisDetector


def _state():
    return {"mu": np.zeros(3), "covariance": np.eye(3)}


def test_envelope_requires_estimator_identity():
    with pytest.raises(ValueError, match="must name its estimator"):
        make_envelope(kind="mahalanobis", estimator=None, state=_state(), config={}, seed=1)


def test_envelope_carries_provenance(tmp_path):
    envelope = make_envelope(
        kind="mahalanobis",
        estimator="full",
        state=_state(),
        config={"cov_regularization": 1e-6},
        seed=3,
        data_fingerprint="deadbeef",
        fit_fingerprint="cafe",
    )
    assert envelope["format_version"] == ARTIFACT_FORMAT_VERSION
    assert envelope["git_sha"] is not None
    assert envelope["seed"] == 3
    assert envelope["data_fingerprint"] == "deadbeef"
    assert envelope["fit_fingerprint"] == "cafe"

    path = tmp_path / "artifact.pkl"
    save_artifact(path, envelope)
    loaded = load_artifact(path)
    assert loaded["estimator"] == "full"
    assert loaded["config"] == {"cov_regularization": 1e-6}
    np.testing.assert_array_equal(loaded["state"]["covariance"], np.eye(3))


def test_save_is_atomic_and_leaves_no_temp_file(tmp_path):
    path = tmp_path / "artifact.pkl"
    save_artifact(
        path, make_envelope(kind="mahalanobis", estimator="full", state=_state(), fit_fingerprint="cafe")
    )
    leftovers = [p.name for p in tmp_path.iterdir() if p.name != "artifact.pkl"]
    assert leftovers == []


def test_load_missing_file_is_loud(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_artifact(tmp_path / "nope.pkl")


def test_unsupported_format_version_is_rejected(tmp_path):
    envelope = make_envelope(kind="mahalanobis", estimator="full", state=_state())
    envelope["format_version"] = 999
    with pytest.raises(ValueError, match="format_version"):
        save_artifact(tmp_path / "artifact.pkl", envelope)


def test_missing_required_field_is_rejected(tmp_path):
    envelope = make_envelope(kind="mahalanobis", estimator="full", state=_state())
    del envelope["data_fingerprint"]
    with pytest.raises(ValueError, match="missing fields"):
        save_artifact(tmp_path / "artifact.pkl", envelope)


def test_unknown_kind_cannot_be_dispatched(tmp_path):
    path = tmp_path / "artifact.pkl"
    save_artifact(
        path, make_envelope(kind="no_such_model", estimator="full", state=_state(), fit_fingerprint="cafe")
    )
    with pytest.raises(ValueError, match="unknown detector kind"):
        load_detector(path)


def test_kind_mismatch_is_rejected(tmp_path):
    path = tmp_path / "artifact.pkl"
    save_artifact(path, make_envelope(kind="markov", estimator="full", state=_state(), fit_fingerprint="cafe"))
    with pytest.raises(ValueError, match="cannot load"):
        MahalanobisDetector.load(path)


def test_saving_an_unfitted_model_is_loud(tmp_path):
    model = MahalanobisDetector("full")
    with pytest.raises(RuntimeError, match="before fit"):
        model.save(tmp_path / "artifact.pkl")


def test_base_class_does_not_implement_the_contract():
    model = FittedModel()
    with pytest.raises(NotImplementedError):
        model.fit(np.zeros((2, 2)))
    with pytest.raises(NotImplementedError):
        model.score(np.zeros((2, 2)))
    with pytest.raises(NotImplementedError):
        model.state_dict()
