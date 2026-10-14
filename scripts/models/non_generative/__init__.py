"""Non-generative novelty detectors (Phase 4 roster, checkpointed per 3.4).

Kept separate from the torch models in `scripts/models/` on purpose: these are
fitted statistical artifacts (a mean, a covariance, PCA components, a sklearn
estimator, a transition matrix), not Modules with optimizers and epochs, and
they carry no vision-stack dependency. `base.py` holds the checkpoint envelope
they all share.
"""

from scripts.models.non_generative.base import (
    ARTIFACT_FORMAT_VERSION,
    FittedModel,
    load_artifact,
    load_detector,
    make_envelope,
    register,
    save_artifact,
)
from scripts.models.non_generative.mahalanobis import MahalanobisDetector

__all__ = [
    "ARTIFACT_FORMAT_VERSION",
    "FittedModel",
    "MahalanobisDetector",
    "load_artifact",
    "load_detector",
    "make_envelope",
    "register",
    "save_artifact",
]
