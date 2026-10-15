"""The static-neural detectors: AE and VAE under the harness contract (4.3.1).

These are the first roster rows whose fitted state is a trained torch module, so
the tests pin the properties the harness depends on and the Phase 1 fixes made
load-bearing: the contract (fit/score/predict with higher-is-more-anomalous),
provenance, a train-only scaler, deterministic scoring, and (on data that is
separable by construction) a higher score for out-of-distribution rows.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="the neural detectors need torch")

import scripts.run_benchmark as rb  # noqa: E402
from scripts.models.detector import Detector  # noqa: E402
from scripts.models.generative import AutoencoderDetector, VAEDetector  # noqa: E402
from scripts.models.generative.base import as_rows, train_val_split  # noqa: E402

DIM = 6


def _cluster(n, seed, scale=0.05):
    return np.random.default_rng(seed).normal(loc=0.5, scale=scale, size=(n, DIM)).astype(np.float32)


def _detectors(seed=42, epochs=3):
    common = dict(seed=seed, epochs=epochs, batch_size=16, hidden_dim=16, latent_dim=4)
    return [AutoencoderDetector(**common), VAEDetector(**common)]


@pytest.mark.parametrize("factory", [AutoencoderDetector, VAEDetector])
def test_detector_satisfies_the_contract(factory):
    detector = factory(seed=42, epochs=1, batch_size=16, hidden_dim=16, latent_dim=4)
    train = _cluster(48, 1)
    detector.fit(train)

    scores = detector.score(_cluster(10, 2))
    assert scores.shape == (10,)
    assert np.isfinite(scores).all()
    assert (scores >= 0).all(), "squared reconstruction error is non-negative"

    decisions = detector.predict(_cluster(10, 2), threshold=float(np.median(scores)))
    assert decisions.dtype == np.int8
    assert set(decisions) <= {0, 1}


def test_a_neural_model_builds_through_the_benchmark_factory():
    for name in ("autoencoder", "vae"):
        assert isinstance(rb.build_model(name, 0), Detector)


def test_fit_provenance_is_recorded_and_checked():
    detector = AutoencoderDetector(seed=42, epochs=1, batch_size=16, hidden_dim=16, latent_dim=4)
    train = _cluster(40, 1)

    with pytest.raises(RuntimeError, match="provenance"):
        detector.assert_fitted_on(train)

    detector.fit(train)
    detector.assert_fitted_on(train)

    with pytest.raises(ValueError, match="different data"):
        detector.assert_fitted_on(_cluster(40, 99))


@pytest.mark.parametrize("factory", [AutoencoderDetector, VAEDetector])
def test_score_before_fit_is_loud(factory):
    with pytest.raises(RuntimeError, match="before fit"):
        factory(seed=42).score(_cluster(4, 3))


def test_empty_training_data_is_loud():
    with pytest.raises(ValueError, match="empty split"):
        AutoencoderDetector(seed=42, epochs=1).fit(np.empty((0, DIM), dtype=np.float32))


def test_the_scaler_is_fitted_on_the_training_rows_only():
    detector = AutoencoderDetector(seed=42, epochs=1, batch_size=16, hidden_dim=16, latent_dim=4)
    train = _cluster(40, 1)

    detector.fit(train)

    assert np.allclose(detector.normalizer_.min_, train.min(axis=0))
    assert np.allclose(detector.normalizer_.max_, train.max(axis=0))
    # A val row outside the train range maps outside [0, 1] rather than being
    # rescaled to hide it (1.1).
    outside = np.full((1, DIM), 10.0, dtype=np.float32)
    assert detector.normalizer_.transform(outside).max() > 1.0


def test_scoring_is_deterministic():
    detector = VAEDetector(seed=42, epochs=2, batch_size=16, hidden_dim=16, latent_dim=4)
    detector.fit(_cluster(48, 1))
    rows = _cluster(12, 5)

    assert np.array_equal(detector.score(rows), detector.score(rows))


def test_training_is_reproducible_for_a_seed():
    rows = _cluster(48, 1)
    first = VAEDetector(seed=7, epochs=3, batch_size=16, hidden_dim=16, latent_dim=4)
    second = VAEDetector(seed=7, epochs=3, batch_size=16, hidden_dim=16, latent_dim=4)

    first.fit(rows)
    second.fit(rows)

    assert np.allclose(first.score(rows), second.score(rows))


@pytest.mark.parametrize("factory", [AutoencoderDetector, VAEDetector])
def test_out_of_distribution_rows_score_higher(factory):
    detector = factory(seed=42, epochs=30, batch_size=32, hidden_dim=32, latent_dim=8, lr=1e-2)
    in_distribution = _cluster(128, 1, scale=0.03)
    detector.fit(in_distribution)

    far = np.full((16, DIM), 4.0, dtype=np.float32)
    assert detector.score(far).mean() > detector.score(in_distribution).mean()


def test_train_val_split_is_deterministic_and_disjoint():
    first = train_val_split(50, 0.2, seed=3)
    second = train_val_split(50, 0.2, seed=3)
    train_idx, val_idx = first

    assert np.array_equal(train_idx, second[0]) and np.array_equal(val_idx, second[1])
    assert len(val_idx) == 10
    assert set(train_idx).isdisjoint(val_idx)
    assert sorted(np.concatenate([train_idx, val_idx]).tolist()) == list(range(50))


def test_train_val_split_edge_cases():
    assert len(train_val_split(50, 0.0, seed=0)[1]) == 0
    assert len(train_val_split(1, 0.2, seed=0)[1]) == 0, "never take the only row for val"
    train_idx, val_idx = train_val_split(2, 0.9, seed=0)
    assert len(val_idx) == 1 and len(train_idx) == 1


def test_as_rows_rejects_bad_shapes():
    assert as_rows(np.zeros(DIM, dtype=np.float32)).shape == (1, DIM)
    with pytest.raises(ValueError, match="n_rows, n_features"):
        as_rows(np.zeros((2, 2, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="empty split"):
        as_rows(np.zeros((0, DIM), dtype=np.float32))
