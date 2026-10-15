"""Feature scaling fitted on the training split only.

`normalize_features` used to compute min/max over the entire feature matrix and
was called *before* `train_test_split`, so every validation sample was scaled
with statistics that included itself. The LSTM-VAE had the same shape of bug with
mean/std over every sequence in the run. The fix is not a different formula but a
different order of operations, which is only enforceable if the fitted state is an
object rather than a value computed inside the call: `transform` on an unfit
scaler raises instead of quietly deriving statistics from whatever it is handed.
"""

import numpy as np

EPS = 1e-8


class _FittedNormalizer:
    """Shared fit/transform plumbing: an explicit fitted state, or a loud failure."""

    kind = None
    state_fields = ()

    def __init__(self, eps=EPS):
        self.eps = eps
        for field in self.state_fields:
            setattr(self, field, None)

    @property
    def is_fitted(self):
        return all(getattr(self, field) is not None for field in self.state_fields)

    def state_dict(self):
        """The fitted state as plain arrays, ready to embed in a model checkpoint (3.5.1).

        `kind` is recorded so a loader can rebuild the right class without being
        told which scaler a run used; the field names are the same ones the
        `*_feature_min.npy` sidecars carry, so the two can be compared directly.
        """
        if not self.is_fitted:
            raise RuntimeError(f"{type(self).__name__}.state_dict called before fit")
        state = {"kind": self.kind, "eps": self.eps}
        for field in self.state_fields:
            state[field] = np.asarray(getattr(self, field))
        return state

    @classmethod
    def from_state(cls, state):
        """Rebuild a fitted normalizer from `state_dict` output, failing loudly."""
        if not isinstance(state, dict):
            raise TypeError(f"normalizer state must be a dict, got {type(state).__name__}")
        if state.get("kind") != cls.kind:
            raise ValueError(f"{cls.__name__} cannot load a {state.get('kind')!r} normalizer")
        normalizer = cls(eps=state.get("eps", EPS))
        for field in cls.state_fields:
            if field not in state:
                raise ValueError(f"normalizer state missing {field!r}")
            setattr(normalizer, field, np.asarray(state[field]))
        if not normalizer.is_fitted:
            raise ValueError("normalizer state is incomplete")
        return normalizer

    def _require_fitted(self, name):
        if not self.is_fitted:
            raise RuntimeError(
                f"{name}.transform called before fit. A scaler that derives its own statistics at "
                "transform time normalizes every sample using itself (1.1 / 1.2)."
            )

    def _check_dim(self, array, name):
        fitted_dim = getattr(self, self.state_fields[0]).shape[0]
        if array.shape[-1] != fitted_dim:
            raise ValueError(f"feature dim {array.shape[-1]} does not match the fitted dim {fitted_dim}")

    def _prepare_fit(self, features, name):
        flat = _as_matrix(features)
        if flat.shape[0] == 0:
            raise ValueError(f"{name}.fit received an empty split; there is nothing to fit on")
        return flat

    def fit_transform(self, features):
        return self.fit(features).transform(features)


class MinMaxNormalizer(_FittedNormalizer):
    """Per-feature min-max scaler: fit on train, apply to anything."""

    kind = "minmax"
    state_fields = ("min_", "max_")

    def fit(self, features):
        features = self._prepare_fit(features, "MinMaxNormalizer")
        self.min_ = features.min(axis=0)
        self.max_ = features.max(axis=0)
        return self

    def transform(self, features):
        self._require_fitted("MinMaxNormalizer")
        features = _as_array(features)
        self._check_dim(features, "MinMaxNormalizer")
        # `eps` keeps a constant training column (min == max) from dividing by zero.
        return (features - self.min_) / (self.max_ - self.min_ + self.eps)


class ZScoreNormalizer(_FittedNormalizer):
    """Per-feature standardiser: fit on train windows, apply to any window.

    Fitted over every axis except the feature axis, so a
    `(n_windows, seq_len, n_features)` block and a flat `(n_rows, n_features)`
    matrix give identical statistics — the LSTM-VAE's windows overlap, and
    reducing over the window axis as well is what makes its z-score differ from
    a flat one on the same features. `transform` keeps the input's shape.
    """

    kind = "zscore"
    state_fields = ("mean_", "std_")

    def fit(self, features):
        features = self._prepare_fit(features, "ZScoreNormalizer")
        self.mean_ = features.mean(axis=0)
        # `eps` is folded into the fitted std, not applied at transform time, so
        # the persisted `lstm_vae_feature_std.npy` is directly reusable at
        # scoring time (3.5.1).
        self.std_ = features.std(axis=0) + self.eps
        return self

    def transform(self, features):
        self._require_fitted("ZScoreNormalizer")
        features = _as_array(features)
        self._check_dim(features, "ZScoreNormalizer")
        return (features - self.mean_) / self.std_


def _as_array(features):
    array = np.asarray(features, dtype=np.float32)
    if array.ndim < 2:
        raise ValueError(f"expected a (..., n_features) array with at least 2 dimensions, got shape {array.shape}")
    return array


def _as_matrix(features):
    """`(..., n_features)` -> `(n_rows, n_features)`, so `fit` reduces over all
    leading axes. The flat VAE normalises a row-per-frame matrix and the
    LSTM-VAE a window-per-sequence block; both need the same per-feature
    statistics."""
    array = _as_array(features)
    return array.reshape(-1, array.shape[-1])


NORMALIZER_KINDS = {"minmax": MinMaxNormalizer, "zscore": ZScoreNormalizer}


def load_normalizer(state):
    """Rebuild the fitted normalizer a checkpoint embedded, dispatching on `kind` (3.5.1)."""
    if not isinstance(state, dict):
        raise TypeError(f"normalizer state must be a dict, got {type(state).__name__}")
    kind = state.get("kind")
    try:
        cls = NORMALIZER_KINDS[kind]
    except KeyError:
        raise ValueError(
            f"unknown normalizer kind {kind!r}; known: {sorted(NORMALIZER_KINDS)}"
        ) from None
    return cls.from_state(state)
