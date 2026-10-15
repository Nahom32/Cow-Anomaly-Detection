"""Shared plumbing for the neural detectors (4.3.1).

The generative roster rows (the flat AE and VAE here; the LSTM-VAE and the
recurrent predictors in the next commit) differ in architecture and loss but
share the three things the Phase 1 fixes made non-negotiable:

- the scaler is fitted on the training rows inside `fit` and carried with the
  module (1.1 / 4.1.2), so a later score cannot silently pair weights with a
  different scaler;
- the module is trained on those rows only, with a deterministic train/val split
  taken from the same rows — the val slice monitors early stopping, it is not a
  second fit set;
- `score` is deterministic and higher-is-more-anomalous (2.6 / 4.1.1): the flat
  models score the fixed point `decode(encode(x).mu)` and the score is the
  squared L2 distance to the reconstruction, `||z - z_hat||^2`, matching
  `improvements_on_vad.md:175` (AE) and `:187` (VAE reconstruction). The existing
  per-split metric (`scripts/evaluation/reconstruction.py`) reports the
  per-element mean; the two order identically because the feature dimension is
  constant, so AUROC is unaffected — the benchmark uses the spec's squared L2.

A subclass supplies `_build_module`, `_train_module` and `_reconstruct`; the base
owns the scaler, the split, seeding, device placement and the score.
"""

import abc

import numpy as np

from scripts.data.normalize import MinMaxNormalizer
from scripts.models.detector import Detector
from scripts.utils.seeding import set_seed


def as_rows(data):
    """Coerce to a float32 `(n_rows, n_features)` matrix, loudly (a 1-D input is one row)."""
    matrix = np.asarray(data, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2:
        raise ValueError(f"expected a (n_rows, n_features) array, got shape {matrix.shape}")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"neural detector received an empty split, shape {matrix.shape}")
    return matrix


def as_windows(data, min_seq_len=1):
    """Coerce to a float32 `(n_windows, seq_len, n_features)` block, loudly.

    The temporal rows score one window per observation, so a 2-D array here is a
    mistaken provider, not something to reshape around. `min_seq_len` is the
    shortest window the model can use — the recurrent predictors need at least
    two steps (one to condition on, one to predict) and say so rather than
    silently training on a single step.
    """
    windows = np.asarray(data, dtype=np.float32)
    if windows.ndim != 3:
        raise ValueError(f"expected a (n_windows, seq_len, n_features) array, got shape {windows.shape}")
    if windows.shape[0] == 0 or windows.shape[2] == 0:
        raise ValueError(f"neural detector received an empty split, shape {windows.shape}")
    if windows.shape[1] < min_seq_len:
        raise ValueError(
            f"sequence length {windows.shape[1]} is shorter than the {min_seq_len} this model needs"
        )
    return windows


def train_val_split(n_rows, val_fraction, seed):
    """Deterministic `(train_idx, val_idx)` over `n_rows`, drawn from one seed.

    Derived from the cell seed rather than the global RNG so a cell's split does
    not depend on how many cells ran before it (AGENTS: seed per data item, not
    per access order). `val_fraction == 0` yields an empty val set; with at least
    two rows a requested fraction keeps at least one val row but never all of
    them, so training never gets an empty loader.
    """
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction!r}")
    if val_fraction > 0 and n_rows >= 2:
        n_val = min(max(1, int(round(n_rows * val_fraction))), n_rows - 1)
    else:
        n_val = 0
    order = np.random.default_rng(seed).permutation(n_rows)
    return order[n_val:], order[:n_val]


class TorchDetector(Detector):
    """A detector whose fitted state is a train-fitted scaler plus a trained module.

    `_prepare` is the one shape decision a subclass makes: the static rows take a
    `(n_rows, n_features)` matrix, the temporal rows a `(n_windows, seq_len,
    n_features)` block. The scaler is reduced over every axis except the feature
    axis in both cases, so a window and a frame of the same cow get the same
    statistics.
    """

    kind = None

    def __init__(self, *, seed=None, device="cpu", val_fraction=0.1):
        self.seed = seed
        self.device = device
        self.val_fraction = val_fraction
        self.normalizer_ = None
        self.module_ = None

    def _prepare(self, data):
        return as_rows(data)

    def _fit_impl(self, train_data):
        """Fit the scaler and train the module, both from `train_data` alone (4.1.2)."""
        prepared = self._prepare(train_data)
        if self.seed is not None:
            set_seed(self.seed)
        self.normalizer_ = MinMaxNormalizer().fit(prepared)
        scaled = self.normalizer_.transform(prepared)
        train_idx, val_idx = train_val_split(len(scaled), self.val_fraction, self.seed)
        self.module_ = self._build_module(scaled.shape[-1])
        self._train_module(scaled[train_idx], scaled[val_idx])
        self.module_.eval()

    def _require_fitted(self):
        if self.module_ is None or self.normalizer_ is None:
            raise RuntimeError(f"{type(self).__name__} used before fit; call fit() first")

    def score(self, data):
        """Squared reconstruction error per observation, `||z - z_hat||^2` (4.1.1).

        Reduced over every axis but the sample axis, so a flat row and a window
        both yield one score per observation; higher is anomalous.
        """
        self._require_fitted()
        scaled = self.normalizer_.transform(self._prepare(data))
        reconstruction = np.asarray(self._reconstruct(self.module_, scaled), dtype=np.float32)
        return np.sum((scaled - reconstruction) ** 2, axis=tuple(range(1, scaled.ndim)))

    @abc.abstractmethod
    def _build_module(self, input_dim):
        """Return a fresh, untrained module for `input_dim` features."""

    @abc.abstractmethod
    def _train_module(self, train_rows, val_rows):
        """Train `self.module_` on the scaled rows; `val_rows` may be empty."""

    @abc.abstractmethod
    def _reconstruct(self, module, scaled):
        """Deterministic reconstruction of `scaled`, same shape; no sampling (2.6)."""


class SequenceTorchDetector(TorchDetector):
    """A temporal detector: observations are `(n_windows, seq_len, n_features)` blocks."""

    min_seq_len = 1

    def _prepare(self, data):
        return as_windows(data, self.min_seq_len)
