"""Euclidean, PCA, Isolation Forest and Markov detectors (3.4.2).

Each is a fitted artifact, not a torch Module, so each must survive the shared
non-neural envelope (3.4) and reproduce its scores byte-for-byte after a reload.
"""

import numpy as np
import pytest

from scripts.models.non_generative import (
    EuclideanDetector,
    IsolationForestDetector,
    MahalanobisDetector,
    MarkovDetector,
    PCADetector,
    load_detector,
)


def _features(n=120, d=5, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, d))


def _states(n_sequences=20, length=12, n_states=4, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_states, size=(n_sequences, length))


ALL_DETECTORS = [
    pytest.param(lambda: EuclideanDetector(), _features(), id="euclidean"),
    pytest.param(lambda: PCADetector(n_components=3, seed=1), _features(), id="pca"),
    pytest.param(lambda: MahalanobisDetector("ledoit_wolf"), _features(), id="mahalanobis"),
    pytest.param(lambda: IsolationForestDetector(n_estimators=20, seed=1), _features(), id="isolation_forest"),
    pytest.param(lambda: MarkovDetector(), _states(), id="markov"),
]


@pytest.mark.parametrize("factory, data", ALL_DETECTORS)
def test_every_detector_round_trips_through_the_envelope(tmp_path, factory, data):
    model = factory().fit(data)
    before = model.score(data)
    path = tmp_path / "artifact.pkl"
    model.save(path, config={}, seed=1, data_fingerprint="fp")
    restored = load_detector(path)
    assert type(restored) is type(model)
    assert restored.estimator == model.estimator
    np.testing.assert_allclose(restored.score(data), before)


def test_euclidean_scores_distance_from_the_mean():
    data = _features()
    model = EuclideanDetector().fit(data)
    np.testing.assert_allclose(model.score(data.mean(axis=0).reshape(1, -1)), [0.0], atol=1e-12)
    assert model.state_dict()["mu"].shape == (5,)


def test_euclidean_empty_split_is_loud():
    with pytest.raises(ValueError, match="empty split"):
        EuclideanDetector().fit(np.zeros((0, 4)))


def test_pca_stores_components_and_explained_variance(tmp_path):
    model = PCADetector(n_components=3, seed=1).fit(_features())
    state = model.state_dict()
    assert state["components"].shape == (3, 5)
    assert state["explained_variance"].shape == (3,)
    assert np.all(np.diff(state["explained_variance"]) <= 0)

    path = tmp_path / "pca.pkl"
    model.save(path, config={"n_components": 3})
    assert PCADetector.load(path).n_components == 3


def test_pca_reconstruction_error_is_small_on_training_data():
    model = PCADetector(n_components=None, seed=0).fit(_features())
    outlier = model.score(np.full((1, 5), 50.0))
    assert outlier[0] > model.score(_features()).mean()


def test_pca_validates_component_count():
    with pytest.raises(ValueError, match="n_components"):
        PCADetector(n_components=0)
    data = _features(n=4, d=5)
    with pytest.raises(ValueError, match="n_components must be in"):
        PCADetector(n_components=10).fit(data)


def test_isolation_forest_flags_an_outlier():
    data = _features(seed=2)
    model = IsolationForestDetector(n_estimators=50, seed=2).fit(data)
    assert model.score(np.full((1, 5), 25.0))[0] > model.score(data).mean()
    assert "estimator" in model.state_dict()


def test_markov_transition_matrix_is_row_stochastic():
    model = MarkovDetector().fit(_states())
    rows = model.transition_matrix_.sum(axis=1)
    np.testing.assert_allclose(rows, np.ones_like(rows))
    assert model.prior_.sum() == pytest.approx(1.0)
    assert model.n_states == 4


def test_markov_scores_rare_transitions_higher():
    states = np.array([[0, 1, 0, 1, 0, 1, 0, 1]])  # 0<->1 common, never 3
    model = MarkovDetector(n_states=4, smoothing=1.0).fit(states)
    common = model.score(np.array([[0, 1]]))[0, 1]
    rare = model.score(np.array([[3, 2]]))[0, 1]
    assert rare > common


def test_markov_rejects_unknown_states():
    model = MarkovDetector(n_states=2).fit(np.array([[0, 1, 0, 1]]))
    with pytest.raises(ValueError, match="chain has 2 states"):
        model.score(np.array([[0, 5]]))


def test_markov_rejects_non_integer_states():
    with pytest.raises(ValueError, match="no states"):
        MarkovDetector().fit(np.array([]))
    with pytest.raises(ValueError, match="non-integral float"):
        MarkovDetector().fit(np.array([[0.0, 1.5]]))


def test_load_detector_dispatches_on_kind(tmp_path):
    cases = [
        (EuclideanDetector().fit(_features()), "euclidean"),
        (PCADetector(n_components=2).fit(_features()), "pca"),
        (IsolationForestDetector(n_estimators=10, seed=0).fit(_features()), "isolation_forest"),
        (MarkovDetector().fit(_states()), "markov"),
    ]
    for model, _kind in cases:
        path = tmp_path / f"{model.kind}.pkl"
        model.save(path, config={})
        assert type(load_detector(path)) is type(model)
