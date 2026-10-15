"""PCA reconstruction novelty detector (3.4.2).

Fit a linear subspace on the training features and score the squared
reconstruction error of each observation. The fitted artifact is the projection
itself — `components` plus the `explained_variance` that says how much of the
normal variation each direction actually carries — so the checkpoint can be
audited without re-fitting (`improvements_on_vad.md:282`).

This is the standalone PCA baseline. Mahalanobis with
`estimator="pca_mahalanobis"` is a different experiment (whitened distance in
the reduced space) and lives in `mahalanobis.py`.
"""

import numpy as np

from scripts.models.non_generative.base import FittedModel, as_matrix, register


@register
class PCADetector(FittedModel):
    kind = "pca"
    estimator = "svd"

    def __init__(self, n_components=None, seed=None):
        if n_components is not None and n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {n_components}")
        self.n_components = n_components
        self.seed = seed
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None

    def fit(self, features):
        from sklearn.decomposition import PCA

        matrix = as_matrix(features)
        max_components = min(matrix.shape)
        n = max_components if self.n_components is None else int(self.n_components)
        if n < 1 or n > max_components:
            raise ValueError(f"n_components must be in [1, {max_components}], got {self.n_components}")
        pca = PCA(n_components=n, random_state=self.seed).fit(matrix)
        self.mean_ = pca.mean_.astype(np.float64)
        self.components_ = pca.components_.astype(np.float64)
        self.explained_variance_ = pca.explained_variance_.astype(np.float64)

    def score(self, features):
        self._require_fitted()
        matrix = as_matrix(features)
        if matrix.shape[1] != self.components_.shape[1]:
            raise ValueError(
                f"feature dim {matrix.shape[1]} does not match the fitted dim {self.components_.shape[1]}"
            )
        centered = matrix - self.mean_
        reconstructed = centered @ self.components_.T @ self.components_
        return np.sum((centered - reconstructed) ** 2, axis=1)

    def state_dict(self):
        self._require_fitted()
        return {
            "mean": np.asarray(self.mean_),
            "components": np.asarray(self.components_),
            "explained_variance": np.asarray(self.explained_variance_),
        }

    @classmethod
    def from_state(cls, state, *, config, estimator):
        config = config or {}
        for key in ("mean", "components", "explained_variance"):
            if key not in state:
                raise ValueError(f"pca artifact missing state entry: {key}")
        model = cls(n_components=config.get("n_components"), seed=config.get("seed"))
        model.mean_ = np.asarray(state["mean"], dtype=np.float64)
        model.components_ = np.asarray(state["components"], dtype=np.float64)
        model.explained_variance_ = np.asarray(state["explained_variance"], dtype=np.float64)
        model.n_components = model.components_.shape[0]
        model._require_fitted()
        return model

    def _require_fitted(self):
        if self.components_ is None or self.mean_ is None:
            raise RuntimeError("PCADetector used before fit; call fit() or load a checkpoint")
        if self.components_.ndim != 2 or self.explained_variance_.shape != (self.components_.shape[0],):
            raise ValueError(
                "invalid PCA artifact: components must be (k, d) and explained_variance must have shape (k,)"
            )
