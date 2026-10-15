"""Non-generative novelty detectors (Phase 4 roster, checkpointed per 3.4).

Kept separate from the torch models in `scripts/models/` on purpose: these are
fitted statistical artifacts (a mean, a covariance, PCA components, a sklearn
estimator, a transition matrix), not Modules with optimizers and epochs, and
they carry no vision-stack dependency. `base.py` holds the checkpoint envelope
they all share; `scripts/models/detector.py` holds the fit/score/predict
contract they all satisfy (4.1.1).

Importing this package registers every detector, so `load_detector(path)` can
dispatch on the `kind` recorded in a checkpoint.
"""

from scripts.models.non_generative.base import (
    ARTIFACT_FORMAT_VERSION,
    FittedModel,
    as_matrix,
    load_artifact,
    load_detector,
    make_envelope,
    register,
    save_artifact,
)
from scripts.models.non_generative.euclidean import EuclideanDetector
from scripts.models.non_generative.isolation_forest import IsolationForestDetector
from scripts.models.non_generative.mahalanobis import MahalanobisDetector
from scripts.models.non_generative.markov import MarkovDetector
from scripts.models.non_generative.pca import PCADetector

__all__ = [
    "ARTIFACT_FORMAT_VERSION",
    "EuclideanDetector",
    "FittedModel",
    "IsolationForestDetector",
    "MahalanobisDetector",
    "MarkovDetector",
    "PCADetector",
    "as_matrix",
    "load_artifact",
    "load_detector",
    "make_envelope",
    "register",
    "save_artifact",
]
