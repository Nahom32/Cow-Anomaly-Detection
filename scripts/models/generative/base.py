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
    """A detector whose fitted state is a train-fitted scaler plus a trained module."""

    kind = None

    def __init__(self, *, seed=None, device="cpu", val_fraction=0.1):
        self.seed = seed
        self.device = device
        self.val_fraction = val_fraction
        self.normalizer_ = None
        self.module_ = None

    def _fit_impl(self, train_data):
        """Fit the scaler and train the module, both from `train_data` alone (4.1.2)."""
        rows = as_rows(train_data)
        if self.seed is not None:
            set_seed(self.seed)
        self.normalizer_ = MinMaxNormalizer().fit(rows)
        scaled = self.normalizer_.transform(rows)
        train_idx, val_idx = train_val_split(len(scaled), self.val_fraction, self.seed)
        self.module_ = self._build_module(scaled.shape[1])
        self._train_module(scaled[train_idx], scaled[val_idx])
        self.module_.eval()

    def _require_fitted(self):
        if self.module_ is None or self.normalizer_ is None:
            raise RuntimeError(f"{type(self).__name__} used before fit; call fit() first")

    def score(self, data):
        """Squared reconstruction error per row, `||z - z_hat||^2`; higher is anomalous (4.1.1)."""
        self._require_fitted()
        scaled = self.normalizer_.transform(as_rows(data))
        reconstruction = np.asarray(self._reconstruct(self.module_, scaled), dtype=np.float32)
        return np.sum((scaled - reconstruction) ** 2, axis=1)

    @abc.abstractmethod
    def _build_module(self, input_dim):
        """Return a fresh, untrained module for `input_dim` features."""

    @abc.abstractmethod
    def _train_module(self, train_rows, val_rows):
        """Train `self.module_` on the scaled rows; `val_rows` may be empty."""

    @abc.abstractmethod
    def _reconstruct(self, module, scaled):
        """Deterministic reconstruction of `scaled`, same shape; no sampling (2.6)."""
