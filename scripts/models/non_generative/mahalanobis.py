"""Mahalanobis novelty detector (3.4.1, Phase 4 roster).

Scores the squared Mahalanobis distance of a feature vector from the training
mean:

    d(x) = (x - mu)^T Sigma^-1 (x - mu)

The point of the experiment is *which* Sigma. `improvements_on_vad.md:146-154`
asks the study to compare diagonal, full, regularized, shrinkage (Ledoit-Wolf)
and PCA-then-Mahalanobis estimators, so the estimator name is a first-class part
of the fitted artifact: `regularized` and `full` produce identically-shaped
`(d, d)` covariances and PCA+Mahalanobis carries components neither of the
others has. A checkpoint that stored only `mu` and `Sigma` could not say which
experiment it came from, which is exactly what 3.4.1 forbids.
"""

import numpy as np

from scripts.models.non_generative.base import FittedModel, as_matrix, register

ESTIMATORS = ("diagonal", "full", "regularized", "ledoit_wolf", "pca_mahalanobis")
DEFAULT_REGULARIZATION = 1e-6
_VARIANCE_EPS = 1e-8


@register
class MahalanobisDetector(FittedModel):
    kind = "mahalanobis"

    def __init__(self, estimator="full", *, regularization=DEFAULT_REGULARIZATION, n_components=None, seed=None):
        if estimator not in ESTIMATORS:
            raise ValueError(f"unknown covariance estimator {estimator!r}; choose one of {ESTIMATORS}")
        if estimator == "pca_mahalanobis" and n_components is not None and n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {n_components}")
        self.estimator = estimator
        self.regularization = float(regularization)
        self.n_components = n_components
        self.seed = seed
        self.mu_ = None
        self.covariance_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.pca_mean_ = None

    def _fit_impl(self, features):
        matrix = as_matrix(features)
        if matrix.shape[0] < 2:
            raise ValueError("MahalanobisDetector.fit needs at least 2 rows; nothing to estimate Sigma from")
        if self.estimator == "pca_mahalanobis":
            components, explained_variance, pca_mean = _fit_pca(matrix, self.n_components, self.seed)
            reduced = (matrix - pca_mean) @ components.T
            self.components_ = components
            self.explained_variance_ = explained_variance
            self.pca_mean_ = pca_mean
            self.mu_ = reduced.mean(axis=0)
            self.covariance_ = _fit_covariance(reduced, "regularized", self.regularization)
        else:
            self.mu_ = matrix.mean(axis=0)
            self.covariance_ = _fit_covariance(matrix, self.estimator, self.regularization)
        self._validate_state()

    def score(self, features):
        matrix = as_matrix(features)
        self._require_fitted()
        if self.estimator == "pca_mahalanobis":
            matrix = (matrix - self.pca_mean_) @ self.components_.T
        centered = matrix - self.mu_
        if self.estimator == "diagonal":
            return np.sum(centered**2 / self.covariance_, axis=1)
        solution = np.linalg.solve(self.covariance_, centered.T).T
        return np.sum(centered * solution, axis=1)

    def state_dict(self):
        self._require_fitted()
        state = {"mu": np.asarray(self.mu_), "covariance": np.asarray(self.covariance_)}
        if self.estimator == "pca_mahalanobis":
            state["components"] = np.asarray(self.components_)
            state["explained_variance"] = np.asarray(self.explained_variance_)
            state["pca_mean"] = np.asarray(self.pca_mean_)
        return state

    @classmethod
    def from_state(cls, state, *, config, estimator):
        config = config or {}
        model = cls(
            estimator=estimator,
            regularization=config.get("cov_regularization", DEFAULT_REGULARIZATION),
            n_components=config.get("n_components"),
            seed=config.get("seed"),
        )
        required = ("mu", "covariance")
        missing = [key for key in required if key not in state]
        if missing:
            raise ValueError(f"mahalanobis artifact missing state entries: {missing}")
        model.mu_ = np.asarray(state["mu"], dtype=np.float64)
        model.covariance_ = np.asarray(state["covariance"], dtype=np.float64)
        if estimator == "pca_mahalanobis":
            for key in ("components", "explained_variance", "pca_mean"):
                if key not in state:
                    raise ValueError(f"pca_mahalanobis artifact missing {key!r}")
            model.components_ = np.asarray(state["components"], dtype=np.float64)
            model.explained_variance_ = np.asarray(state["explained_variance"], dtype=np.float64)
            model.pca_mean_ = np.asarray(state["pca_mean"], dtype=np.float64)
        model._validate_state()
        return model

    def _require_fitted(self):
        if self.mu_ is None or self.covariance_ is None:
            raise RuntimeError("MahalanobisDetector used before fit; call fit() or load a checkpoint")

    def _validate_state(self):
        """Shape-check against the estimator name rather than trusting the arrays (3.6.1)."""
        mu = np.asarray(self.mu_)
        covariance = np.asarray(self.covariance_)
        if self.estimator == "pca_mahalanobis":
            components = np.asarray(self.components_)
            if components.ndim != 2:
                raise ValueError(f"components must be 2-D, got shape {components.shape}")
            n_components, dim = components.shape
            if np.asarray(self.pca_mean_).shape != (dim,):
                raise ValueError(f"pca_mean must have shape {(dim,)}, got {np.asarray(self.pca_mean_).shape}")
            if np.asarray(self.explained_variance_).shape != (n_components,):
                raise ValueError(
                    f"explained_variance must have shape {(n_components,)}, "
                    f"got {np.asarray(self.explained_variance_).shape}"
                )
            if mu.shape != (n_components,):
                raise ValueError(f"mu shape {mu.shape} must match reduced dim {(n_components,)}")
            if covariance.shape != (n_components, n_components):
                raise ValueError(
                    f"covariance shape {covariance.shape} must be {(n_components, n_components)} in the PCA subspace"
                )
            return
        dim = mu.shape[0]
        expected_covariance = (dim,) if self.estimator == "diagonal" else (dim, dim)
        if covariance.shape != expected_covariance:
            raise ValueError(
                f"{self.estimator} estimator requires covariance shape {expected_covariance}, got {covariance.shape}"
            )



def _fit_covariance(matrix, estimator, regularization):
    n_features = matrix.shape[1]
    if estimator == "diagonal":
        return matrix.var(axis=0) + _VARIANCE_EPS
    covariance = np.atleast_2d(np.cov(matrix, rowvar=False))
    if estimator == "full":
        return covariance
    if estimator == "regularized":
        scale = np.trace(covariance) / n_features
        return covariance + regularization * scale * np.eye(n_features)
    if estimator == "ledoit_wolf":
        from sklearn.covariance import ledoit_wolf

        shrunk, _ = ledoit_wolf(matrix)
        return np.atleast_2d(shrunk)
    raise ValueError(f"unknown covariance estimator {estimator!r}")


def _fit_pca(matrix, n_components, seed):
    from sklearn.decomposition import PCA

    max_components = min(matrix.shape)
    n = max_components if n_components is None else int(n_components)
    if n < 1 or n > max_components:
        raise ValueError(f"n_components must be in [1, {max_components}], got {n_components}")
    pca = PCA(n_components=n, random_state=seed)
    pca.fit(matrix)
    return pca.components_.astype(np.float64), pca.explained_variance_.astype(np.float64), pca.mean_.astype(np.float64)
