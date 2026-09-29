"""Feature scaling fitted on the training split only (task 1.1).

`normalize_features` used to compute min/max over the entire feature matrix and
was called *before* `train_test_split`, so every validation sample was scaled
with statistics that included itself. The fix is not a different formula but a
different order of operations, which is only enforceable if the fitted state is
an object rather than a value computed inside the call: `transform` on an unfit
scaler raises instead of quietly deriving min/max from whatever it is handed.

Task 1.2 is the same change for the LSTM-VAE z-score, which currently averages
over every sequence in the run.
"""

import numpy as np

EPS = 1e-8


class MinMaxNormalizer:
    """Per-feature min-max scaler: fit on train, apply to anything."""

    def __init__(self, eps=EPS):
        self.eps = eps
        self.min_ = None
        self.max_ = None

    @property
    def is_fitted(self):
        return self.min_ is not None and self.max_ is not None

    def fit(self, features):
        features = _as_matrix(features)
        if features.shape[0] == 0:
            raise ValueError("MinMaxNormalizer.fit received an empty split; there is nothing to fit on")
        self.min_ = features.min(axis=0)
        self.max_ = features.max(axis=0)
        return self

    def transform(self, features):
        if not self.is_fitted:
            raise RuntimeError(
                "MinMaxNormalizer.transform called before fit. A scaler that derives its own "
                "statistics at transform time normalizes every sample using itself (1.1)."
            )
        features = _as_matrix(features)
        if features.shape[1] != self.min_.shape[0]:
            raise ValueError(
                f"feature dim {features.shape[1]} does not match the fitted dim {self.min_.shape[0]}"
            )
        # `eps` keeps a constant training column (min == max) from dividing by zero.
        return (features - self.min_) / (self.max_ - self.min_ + self.eps)

    def fit_transform(self, features):
        return self.fit(features).transform(features)


def _as_matrix(features):
    array = np.asarray(features, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"expected a 2-D (n_samples, n_features) array, got shape {array.shape}")
    return array
