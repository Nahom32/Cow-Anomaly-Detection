"""Per-split reconstruction scoring (task 1.5).

Three properties the reported numbers rest on: the score is a per-sample *mean*
(task 2.1, so it does not scale with feature dimension), it is computed through
the deterministic mu -> decode route rather than `forward` (whose sampled z
would make a rerun move the number), and an empty split is reported as a count
of zero rather than crashed on or silently summarised as something else.
"""

import numpy as np
import pytest
import torch

from scripts.evaluation.reconstruction import (
    SCORE_METRIC,
    reconstruction_score_report,
    row_reconstruction_errors,
    split_summaries,
    summarize,
    window_reconstruction_errors,
)
from scripts.models.lstm_vae import LSTMVAE
from scripts.models.vae import VAE


class _ZeroDecoder:
    """A stand-in whose reconstruction is always zero, so the expected error is
    known in closed form: the mean squared magnitude of the input itself."""

    def eval(self):
        return self

    def encode(self, x):
        return x, None

    def decode(self, mu, **kwargs):
        return torch.zeros_like(mu)


def test_flat_error_is_a_per_sample_mean_not_a_sum():
    """`reduction="sum"` in the training loss would make this scale with dim (2.1)."""
    rows = torch.arange(8, dtype=torch.float32).reshape(2, 4)

    errors = row_reconstruction_errors(_ZeroDecoder(), rows, "cpu")

    assert errors.shape == (2,)
    assert np.allclose(errors, rows.pow(2).mean(dim=1).numpy())
    # The distinction the assertion makes: sum would give 14, mean gives 3.5.
    assert not np.allclose(errors, rows.pow(2).sum(dim=1).numpy())


def test_window_error_reduces_over_both_time_and_feature_axes():
    windows = torch.full((3, 5, 4), 2.0)

    errors = window_reconstruction_errors(_ZeroDecoder(), windows, "cpu")

    assert errors.shape == (3,)
    assert np.allclose(errors, np.full(3, 4.0))


def test_scoring_an_empty_split_returns_an_empty_array():
    """`test_split = 0` leaves the test slot with no rows; that is not an error."""
    rows = np.zeros((0, 4), dtype=np.float32)
    assert row_reconstruction_errors(_ZeroDecoder(), rows, "cpu").shape == (0,)
    assert window_reconstruction_errors(_ZeroDecoder(), [], "cpu").shape == (0,)


def test_flat_scores_are_deterministic_across_rng_state():
    """`VAE.forward` samples z; scoring must not, or the number drifts on rerun."""
    vae = VAE(4, hidden_dim=8, latent_dim=2)
    rows = torch.randn(32, 4, generator=torch.Generator().manual_seed(0))

    torch.manual_seed(1)
    first = row_reconstruction_errors(vae, rows, "cpu")
    torch.manual_seed(999)
    second = row_reconstruction_errors(vae, rows, "cpu")

    assert np.array_equal(first, second)
    assert first.shape == (32,)
    assert (first >= 0.0).all()


def test_window_scores_are_deterministic_across_rng_state():
    vae = LSTMVAE(input_dim=4, hidden_dim=8, latent_dim=2, num_layers=1)
    windows = torch.randn(6, 5, 4, generator=torch.Generator().manual_seed(0))

    torch.manual_seed(1)
    first = window_reconstruction_errors(vae, windows, "cpu")
    torch.manual_seed(999)
    second = window_reconstruction_errors(vae, windows, "cpu")

    assert np.array_equal(first, second)
    assert first.shape == (6,)
    assert (first >= 0.0).all()


def test_summarize_reports_an_empty_split_as_count_zero():
    assert summarize(np.empty(0)) == {"n": 0}


def test_summarize_reports_the_distribution_it_was_given():
    stats = summarize(np.array([1.0, 2.0, 3.0]))
    assert stats["n"] == 3
    assert stats["mean"] == pytest.approx(2.0)
    assert stats["median"] == pytest.approx(2.0)
    assert stats["std"] == pytest.approx(np.std([1.0, 2.0, 3.0]))


def test_split_summaries_drop_empty_slots_but_keep_non_empty_ones():
    """`test_split = 0` means no test key; a zero-filled key would read as a
    score of zero — the difference between "not measured" and "perfect"."""
    summaries = split_summaries({
        "train": np.array([1.0, 2.0]),
        "val": np.array([3.0]),
        "test": np.empty(0),
    })

    assert set(summaries) == {"train", "val"}
    assert summaries["train"]["n"] == 2


def test_the_report_names_the_metric_the_model_and_the_fit_scope():
    report = reconstruction_score_report("flat_vae", {"train": [1.0], "test": [2.0, 4.0]})

    assert report["metric"] == SCORE_METRIC
    assert report["model"] == "flat_vae"
    assert report["normalizer_fit_on"] == "train"
    assert report["splits"]["test"]["n"] == 2
    assert report["splits"]["test"]["median"] == pytest.approx(3.0)
