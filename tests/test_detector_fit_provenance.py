"""Fit provenance: fitted parameters come from the fit rows alone (4.1.2).

Phase 1's leak (1.8) was a scaler fitted on every row and then used as if it
described train only — undetectable after the fact, because `mu` fitted on all
rows looks exactly like `mu` fitted on train. `Detector.fit` fingerprints the
rows it is handed, builds state through `_fit_impl`, and `assert_fitted_on`
verifies the claim, so "fitted on train and only train" is checked rather than
assumed. Every registered detector is exercised, not just one, because a new
detector that re-derives statistics elsewhere would re-open the leak.
"""

import numpy as np
import pytest

from scripts.models.detector import Detector
from scripts.models.non_generative.base import _REGISTRY, make_envelope, save_artifact
from scripts.models.non_generative.euclidean import EuclideanDetector
from scripts.models.non_generative.mahalanobis import MahalanobisDetector
from scripts.utils.hashing import hash_array


def _features(n=60, d=4, seed=0):
    return np.random.default_rng(seed).normal(size=(n, d))


def _states(n_sequences=20, length=10, n_states=4, seed=0):
    return np.random.default_rng(seed).integers(0, n_states, size=(n_sequences, length))


FIT_DATA = {
    "euclidean": _features(),
    "isolation_forest": _features(),
    "mahalanobis": _features(),
    "markov": _states(),
    "pca": _features(),
}


def test_fit_records_the_rows_it_used():
    data = _features()
    model = EuclideanDetector()
    assert model.fit(data) is None, "fit returns nothing (4.1.1)"
    assert model.fit_fingerprint_ == hash_array(data)
    model.assert_fitted_on(data)


def test_fitted_state_is_rejected_for_a_different_train_split():
    train = _features(seed=0)
    stacked = np.vstack([train, _features(seed=1)])
    model = EuclideanDetector()
    model.fit(stacked)
    with pytest.raises(ValueError, match="different"):
        model.assert_fitted_on(train)


def test_a_failed_fit_leaves_no_provenance():
    model = EuclideanDetector()
    with pytest.raises(ValueError, match="empty split"):
        model.fit(np.zeros((0, 4)))
    with pytest.raises(RuntimeError, match="no fit provenance"):
        model.assert_fitted_on(np.zeros((0, 4)))


def test_a_model_rebuilt_from_state_has_no_fit_provenance():
    model = EuclideanDetector.from_state({"mu": np.zeros(3)}, config={}, estimator="euclidean")
    with pytest.raises(RuntimeError, match="fit provenance"):
        model.assert_fitted_on(_features(n=3, d=3))


def test_provenance_survives_the_artifact_round_trip(tmp_path):
    train = _features()
    other = _features(seed=1)
    model = MahalanobisDetector("full")
    model.fit(train)
    path = tmp_path / "artifact.pkl"
    model.save(path, config={}, seed=0)

    restored = MahalanobisDetector.load(path)
    restored.assert_fitted_on(train)
    with pytest.raises(ValueError, match="different"):
        restored.assert_fitted_on(other)


def test_saving_an_envelope_without_fit_provenance_is_rejected(tmp_path):
    envelope = make_envelope(kind="mahalanobis", estimator="full", state={"mu": np.zeros(3)}, fit_fingerprint=None)
    with pytest.raises(ValueError, match="fit rows"):
        save_artifact(tmp_path / "artifact.pkl", envelope)


def test_redefining_fit_is_refused():
    with pytest.raises(TypeError, match="_fit_impl"):

        class Sneaky(Detector):
            def fit(self, train_data):
                return None

            def score(self, data):
                return np.zeros(len(np.asarray(data)), dtype=np.float64)


@pytest.mark.parametrize("kind", sorted(_REGISTRY))
def test_every_registered_detector_fits_only_on_the_rows_it_is_given(kind):
    assert set(FIT_DATA) == set(_REGISTRY), "a detector was registered without provenance data (4.1.2)"
    train = FIT_DATA[kind]
    other = _features(seed=1) if train.ndim == 2 else _states(seed=1)

    model = _REGISTRY[kind]()
    model.fit(train)
    model.assert_fitted_on(train)
    with pytest.raises(ValueError, match="different"):
        model.assert_fitted_on(other)
