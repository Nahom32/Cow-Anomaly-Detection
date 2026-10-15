"""The one detector contract every model in the benchmark implements (4.1.1).

Phase 4's harness iterates `model x anomaly_setting x split x seed x
context_length` and drives every entry the same way, so "adding a baseline"
must mean presenting these three methods, not writing bespoke plumbing:

- `fit(train_data) -> None` — estimate every fitted object (mu, Sigma, PCA
  components, transition matrix, sklearn estimator, and later the neural
  weights) from the training rows alone and return nothing.
- `score(data) -> np.ndarray` — the continuous anomaly score `A_t`: one value
  per observation (per timestep for a sequence model), always oriented so
  *higher = more anomalous*. Every threshold, AUROC and delta in Phase 4 reads
  that orientation, so a detector that reports lower-is-worse is inverting the
  experiment rather than expressing a preference.
- `predict(data, threshold) -> np.ndarray` — the score binarised at a
  threshold, `1` where the score is at or above it.

`predict` gets a default implementation because thresholding the score is the
only decision rule the harness needs; a model with a native decision rule can
override it. `fit` and `score` are abstract: a registered detector missing one
fails at class definition time, not halfway through a benchmark run.

The train-only half of the guarantee (`fit` must not see val/test rows, 1.8)
and the provenance recording live in 4.1.2. The neural models get their
adapters — `fit`/`score` over reconstruction error — in 4.3.1.
"""

import abc

import numpy as np


class Detector(abc.ABC):
    """See the module docstring for the contract; subclasses implement `fit`/`score`."""

    @abc.abstractmethod
    def fit(self, train_data):
        """Estimate every fitted parameter from `train_data` alone; returns nothing (4.1.1)."""

    @abc.abstractmethod
    def score(self, data):
        """Continuous anomaly score, one per observation; higher = more anomalous (4.1.1)."""

    def predict(self, data, threshold):
        """Binary decisions at `threshold`: `1` where the score is at or above it."""
        limit = float(threshold)
        if not np.isfinite(limit):
            raise ValueError(f"threshold must be a finite score cutoff, got {threshold!r}")
        return (np.asarray(self.score(data)) >= limit).astype(np.int8)
