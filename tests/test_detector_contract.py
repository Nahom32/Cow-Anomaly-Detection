"""The one detector contract every model implements (4.1.1).

The Phase 4 harness drives every model through `fit`/`score`/`predict`
(`scripts/models/detector.py`), so the contract is checked against the registry
rather than against a hand-written list: a detector registered without a
`score`, or scoring in the wrong direction at a threshold, must fail here
instead of inverting a benchmark run.
"""

import numpy as np
import pytest

from scripts.models import non_generative
from scripts.models.detector import Detector
from scripts.models.non_generative.base import _REGISTRY, register


def _features(n=120, d=5, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, d))


def _states(n_sequences=20, length=12, n_states=4, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_states, size=(n_sequences, length))


# One fitting block per registered `kind`; `test_the_registry_has_fit_data_for_every_kind`
# fails if the registry grows without data for the new detector.
FIT_DATA = {
    "euclidean": _features(),
    "isolation_forest": _features(),
    "mahalanobis": _features(),
    "markov": _states(),
    "pca": _features(),
}

KINDS = sorted(_REGISTRY)


def test_the_package_exports_exactly_the_registered_detectors():
    exported = [getattr(non_generative, name) for name in non_generative.__all__]
    kinds = {cls.kind for cls in exported if isinstance(cls, type) and issubclass(cls, Detector) and cls.kind}
    assert kinds == set(_REGISTRY), "package exports and detector registry disagree"


def test_the_registry_has_fit_data_for_every_kind():
    assert set(FIT_DATA) == set(_REGISTRY), "a detector was registered without contract data (4.1.1)"


def test_the_contract_cannot_be_instantiated():
    with pytest.raises(TypeError, match="abstract"):
        Detector()


def test_a_detector_without_score_cannot_be_instantiated():
    class Incomplete(Detector):
        def _fit_impl(self, train_data):
            return None

    with pytest.raises(TypeError, match="score"):
        Incomplete()


def test_the_registry_rejects_classes_outside_the_contract():
    class NotADetector:
        kind = "not_a_detector"

    with pytest.raises(TypeError, match="Detector contract"):
        register(NotADetector)
    assert "not_a_detector" not in _REGISTRY


@pytest.mark.parametrize("kind", KINDS)
def test_registered_detector_satisfies_the_contract(kind):
    model = _REGISTRY[kind]()
    data = FIT_DATA[kind]

    assert isinstance(model, Detector)
    assert model.fit(data) is None, "fit returns nothing (4.1.1)"

    scores = np.asarray(model.score(data))
    assert scores.dtype.kind == "f", f"{kind}.score must return float scores, got {scores.dtype}"
    assert np.isfinite(scores).all(), f"{kind}.score returned a non-finite score"
    assert scores.shape[0] == data.shape[0], f"{kind}.score must return one score per observation"

    limit = float(np.median(scores))
    predicted = model.predict(data, limit)
    np.testing.assert_array_equal(predicted, (scores >= limit).astype(np.int8))
    assert predicted.dtype == np.int8
    # `score == threshold` counts as anomalous: the >= in the contract, pinned.
    assert model.predict(data, float(scores.min())).all()
    assert not model.predict(data, float(scores.max()) + 1.0).any()

    with pytest.raises(ValueError, match="finite"):
        model.predict(data, float("nan"))


@pytest.mark.parametrize("kind", KINDS)
def test_using_a_detector_before_fit_is_loud(kind):
    model = _REGISTRY[kind]()
    data = FIT_DATA[kind]
    with pytest.raises(RuntimeError, match="before fit"):
        model.score(data)
    with pytest.raises(RuntimeError, match="before fit"):
        model.predict(data, 0.0)
