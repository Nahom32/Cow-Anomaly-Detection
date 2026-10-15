"""Euclidean novelty detector (3.4.2).

The covariance-free baseline: score the squared distance from the training mean.
It exists to answer whether covariance structure buys anything over a plain
centroid distance (`improvements_on_vad.md:279`), so its only fitted artifact is
`mu`.
"""

import numpy as np

from scripts.models.non_generative.base import FittedModel, as_matrix, register


@register
class EuclideanDetector(FittedModel):
    kind = "euclidean"
    estimator = "centroid"

    def __init__(self):
        self.mu_ = None

    def fit(self, features):
        matrix = as_matrix(features)
        if matrix.shape[0] == 0:
            raise ValueError("EuclideanDetector.fit received an empty split")
        self.mu_ = matrix.mean(axis=0)

    def score(self, features):
        if self.mu_ is None:
            raise RuntimeError("EuclideanDetector used before fit; call fit() or load a checkpoint")
        centered = as_matrix(features) - self.mu_
        return np.sum(centered**2, axis=1)

    def state_dict(self):
        if self.mu_ is None:
            raise RuntimeError("EuclideanDetector used before fit; nothing to serialize")
        return {"mu": np.asarray(self.mu_)}

    @classmethod
    def from_state(cls, state, *, config, estimator):
        if "mu" not in state:
            raise ValueError("euclidean artifact missing state entry: mu")
        model = cls()
        model.mu_ = np.asarray(state["mu"], dtype=np.float64)
        return model
