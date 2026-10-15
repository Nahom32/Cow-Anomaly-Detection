"""Isolation Forest novelty detector (3.4.2).

A non-parametric baseline that makes no Gaussian or subspace assumption. The
fitted artifact is the sklearn estimator itself, so it is stored whole inside
the shared envelope (there is no small set of arrays to reconstruct it from)
alongside the constructor params that identify it.

The sklearn import is local to `fit` so importing this package never drags the
model stack in until a forest is actually trained.
"""

from scripts.models.non_generative.base import FittedModel, as_matrix, register


@register
class IsolationForestDetector(FittedModel):
    kind = "isolation_forest"
    estimator = "sklearn_isolation_forest"

    def __init__(self, n_estimators=100, max_samples="auto", contamination="auto", seed=None):
        self.n_estimators = int(n_estimators)
        self.max_samples = max_samples
        self.contamination = contamination
        self.seed = seed
        self.estimator_ = None

    def fit(self, features):
        from sklearn.ensemble import IsolationForest

        matrix = as_matrix(features)
        if matrix.shape[0] == 0:
            raise ValueError("IsolationForestDetector.fit received an empty split")
        self.estimator_ = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            random_state=self.seed,
        ).fit(matrix)

    def score(self, features):
        self._require_fitted()
        # sklearn's score_samples is higher for inliers; negate so higher = more anomalous.
        return -self.estimator_.score_samples(as_matrix(features))

    def state_dict(self):
        self._require_fitted()
        return {
            "estimator": self.estimator_,
            "params": {
                "n_estimators": self.n_estimators,
                "max_samples": self.max_samples,
                "contamination": self.contamination,
                "seed": self.seed,
            },
        }

    @classmethod
    def from_state(cls, state, *, config, estimator):
        if "estimator" not in state:
            raise ValueError("isolation_forest artifact missing the fitted estimator")
        params = state.get("params", {})
        model = cls(
            n_estimators=params.get("n_estimators", 100),
            max_samples=params.get("max_samples", "auto"),
            contamination=params.get("contamination", "auto"),
            seed=params.get("seed"),
        )
        model.estimator_ = state["estimator"]
        model._require_fitted()
        return model

    def _require_fitted(self):
        if self.estimator_ is None:
            raise RuntimeError("IsolationForestDetector used before fit; call fit() or load a checkpoint")
