"""Mahalanobis detector + its covariance estimators (3.4.1)."""

import numpy as np
import pytest

from scripts.models.non_generative.base import load_artifact
from scripts.models.non_generative.mahalanobis import ESTIMATORS, MahalanobisDetector

ALL_ESTIMATORS = list(ESTIMATORS)


def _data(n=200, d=6, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d))
    # Correlate two columns so full/regularized covariances are not diagonal.
    x[:, 1] = 0.9 * x[:, 0] + 0.1 * x[:, 1]
    return x


def _probe(d=6):
    rng = np.random.default_rng(99)
    return rng.normal(size=(5, d))


@pytest.mark.parametrize("estimator", ALL_ESTIMATORS)
def test_round_trip_preserves_scores(tmp_path, estimator):
    n_components = 3 if estimator == "pca_mahalanobis" else None
    model = MahalanobisDetector(estimator, n_components=n_components, seed=1)
    model.fit(_data())
    before = model.score(_probe())

    path = tmp_path / "mahal.pkl"
    model.save(path, config={"cov_regularization": 1e-6, "n_components": n_components}, seed=1)
    restored = MahalanobisDetector.load(path)

    assert restored.estimator == estimator
    np.testing.assert_allclose(restored.score(_probe()), before)


def test_estimator_identity_is_stored_not_inferred(tmp_path):
    # `full` and `regularized` share a (d, d) covariance; only the envelope can
    # tell the two experiments apart (3.4.1).
    data = _data()
    full = MahalanobisDetector("full")
    regularized = MahalanobisDetector("regularized", regularization=1e-3)
    full.fit(data)
    regularized.fit(data)
    assert full.covariance_.shape == regularized.covariance_.shape

    full_path, reg_path = tmp_path / "full.pkl", tmp_path / "reg.pkl"
    full.save(full_path, config={"cov_regularization": 0.0})
    regularized.save(reg_path, config={"cov_regularization": 1e-3})

    assert load_artifact(full_path)["estimator"] == "full"
    assert load_artifact(reg_path)["estimator"] == "regularized"
    assert MahalanobisDetector.load(full_path).estimator == "full"
    assert MahalanobisDetector.load(reg_path).estimator == "regularized"


def test_diagonal_covariance_is_a_variance_vector():
    model = MahalanobisDetector("diagonal")
    model.fit(_data())
    assert model.covariance_.shape == (6,)
    assert model.state_dict()["covariance"].shape == (6,)


def test_pca_mahalanobis_persists_components_and_explained_variance(tmp_path):
    model = MahalanobisDetector("pca_mahalanobis", n_components=3, seed=1)
    model.fit(_data())
    state = model.state_dict()
    assert state["components"].shape == (3, 6)
    assert state["explained_variance"].shape == (3,)

    path = tmp_path / "pca_mahal.pkl"
    model.save(path, config={"n_components": 3, "seed": 1})
    envelope = load_artifact(path)
    assert "components" in envelope["state"]
    assert "explained_variance" in envelope["state"]
    assert MahalanobisDetector.load(path).estimator == "pca_mahalanobis"


def test_pca_mahalanobis_without_components_fails_loudly():
    state = {"mu": np.zeros(3), "covariance": np.eye(3)}
    with pytest.raises(ValueError, match="components"):
        MahalanobisDetector.from_state(state, config={}, estimator="pca_mahalanobis")


def test_shape_mismatch_against_estimator_fails_loudly():
    # A (d, d) covariance under the diagonal estimator is exactly the silent
    # mismatch the shape check exists to catch (3.6.1).
    state = {"mu": np.zeros(3), "covariance": np.eye(3)}
    with pytest.raises(ValueError, match="diagonal estimator requires"):
        MahalanobisDetector.from_state(state, config={}, estimator="diagonal")


def test_unknown_estimator_is_rejected():
    with pytest.raises(ValueError, match="unknown covariance estimator"):
        MahalanobisDetector("not_an_estimator")


def test_score_is_zero_at_the_training_mean():
    data = _data()
    model = MahalanobisDetector("full")
    model.fit(data)
    score = model.score(data.mean(axis=0).reshape(1, -1))
    assert score[0] == pytest.approx(0.0, abs=1e-10)


def test_fit_needs_at_least_two_rows():
    with pytest.raises(ValueError, match="at least 2 rows"):
        MahalanobisDetector("full").fit(np.zeros((1, 4)))
