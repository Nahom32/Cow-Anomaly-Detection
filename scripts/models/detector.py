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
  that orientation, so a detector reporting lower-is-worse would invert the
  experiment rather than express a preference.
- `predict(data, threshold) -> np.ndarray` — the score binarised at a
  threshold, `1` where the score is at or above it.

`predict` gets a default implementation because thresholding the score is the
only decision rule the harness needs; a model with a native decision rule can
override it. `score` (and `_fit_impl`) are abstract: a registered detector
missing one fails at class definition time, not mid-run.

`fit` is also the provenance choke point (4.1.2): subclasses implement
`_fit_impl`, never `fit`, so every fitted object is constructed inside this one
call from the rows it fingerprints. That makes "fitted on the training split
and only that" — the Phase 1 leak of 1.8, a statistic computed over every row
and then used as if it came from train — a verifiable claim via
`assert_fitted_on(train_data)` rather than a property you have to re-derive
from source order. The neural adapters, whose `fit` wraps a training loop over
reconstruction error, arrive in 4.3.1.
"""

import abc

import numpy as np

from scripts.utils.hashing import hash_array


class Detector(abc.ABC):
    """See the module docstring for the contract; subclasses implement `_fit_impl`/`score`."""

    fit_fingerprint_ = None

    def __init_subclass__(cls, **kwargs):
        """Redefining `fit` would silently skip the provenance record (4.1.2)."""
        super().__init_subclass__(**kwargs)
        if "fit" in cls.__dict__:
            raise TypeError(
                f"{cls.__name__} overrides fit(); implement _fit_impl instead, so every fitted object "
                "is built inside the single fit() choke point (4.1.2)"
            )

    def fit(self, train_data):
        """Fit from `train_data` alone, recording which rows produced the state (4.1.2).

        Subclasses never see this method — they implement `_fit_impl` — which
        is what keeps every fitted parameter constructed inside one call from
        the rows that call is given. The fingerprint is what
        `assert_fitted_on` checks. A fit that raises leaves no provenance: a
        half-updated state must not keep claiming rows it did not finish using.
        """
        self.fit_fingerprint_ = None
        fingerprint = hash_array(np.asarray(train_data))
        self._fit_impl(train_data)
        self.fit_fingerprint_ = fingerprint

    @abc.abstractmethod
    def _fit_impl(self, train_data):
        """Build every fitted parameter from `train_data`; called only by `fit` (4.1.2)."""

    @abc.abstractmethod
    def score(self, data):
        """Continuous anomaly score, one per observation; higher = more anomalous (4.1.1)."""

    def predict(self, data, threshold):
        """Binary decisions at `threshold`: `1` where the score is at or above it."""
        limit = float(threshold)
        if not np.isfinite(limit):
            raise ValueError(f"threshold must be a finite score cutoff, got {threshold!r}")
        return (np.asarray(self.score(data)) >= limit).astype(np.int8)

    def assert_fitted_on(self, train_data):
        """Raise unless this detector's state was fitted on exactly `train_data` (4.1.2).

        The Phase 1 leak is undetectable after the fact — `mu` fitted on every
        row looks exactly like `mu` fitted on train — so the claim is recorded
        at fit time and checked here instead.
        """
        if self.fit_fingerprint_ is None:
            raise RuntimeError(
                f"{type(self).__name__} has no fit provenance: fit() was never called, "
                "or its last fit did not complete (4.1.2)"
            )
        if hash_array(np.asarray(train_data)) != self.fit_fingerprint_:
            raise ValueError(
                f"{type(self).__name__} was fitted on different data than the given train_data "
                "(4.1.2); fitted state must describe the training split only"
            )
