"""Checkpoint envelope for fitted non-neural detectors (3.4).

The generative models are torch Modules and checkpoint through
`scripts/checkpoint.py` — optimizer, scheduler, epoch, best metric, RNG. The
detectors in this package have none of that: they are a handful of fitted
arrays plus the name of the method that produced them. They still need the same
provenance story (format version, resolved config, seed, git SHA, data
fingerprint) so Phase 4 can resume and audit them, so this is the non-neural
counterpart of the unified schema rather than a second convention.

The envelope is deliberately torch-free — `pickle` over
`scripts.utils.io.atomic_write_bytes` — so a Mahalanobis or Markov checkpoint can
be produced and inspected without importing torch (and without the vision stack).

`estimator` is a required, explicit field. A regularized and a full covariance
are both `(d, d)`; PCA+Mahalanobis carries components the plain form does not
use. Which one was fitted cannot be recovered from the arrays, so the loader
refuses an artifact that does not name it (3.4.1).
"""

import os
import pickle

from scripts.manifest import git_info
from scripts.models.detector import Detector
from scripts.utils.io import atomic_write_bytes

ARTIFACT_FORMAT_VERSION = 1
SUPPORTED_ARTIFACT_FORMAT_VERSIONS = (ARTIFACT_FORMAT_VERSION,)

_REQUIRED_FIELDS = (
    "format_version",
    "kind",
    "estimator",
    "state",
    "config",
    "seed",
    "git_sha",
    "data_fingerprint",
)

_REGISTRY = {}


def as_matrix(features):
    """Coerce to a float64 `(n_samples, n_features)` matrix, loudly.

    A 1-D input is a single sample; anything else that is not 2-D is a bug in the
    caller, not something to reshape around.
    """
    import numpy as np

    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2:
        raise ValueError(f"expected a (n_samples, n_features) array, got shape {matrix.shape}")
    if matrix.shape[1] == 0:
        raise ValueError("feature matrix has zero columns")
    return matrix


def register(cls):
    """Register a `Detector` subclass so `load_detector` can dispatch on `kind`."""
    if not issubclass(cls, Detector):
        raise TypeError(f"{cls.__name__} does not implement the Detector contract (fit/score/predict, 4.1.1)")
    if not cls.kind:
        raise ValueError(f"{cls.__name__} must set a non-empty `kind`")
    _REGISTRY[cls.kind] = cls
    return cls


def make_envelope(*, kind, estimator, state, config=None, seed=None, data_fingerprint=None, extra=None):
    if not estimator:
        raise ValueError(
            f"{kind!r} artifact must name its estimator. 'regularized' and 'full' share a Sigma shape "
            "and PCA+Mahalanobis adds components, so identity cannot be inferred from `state` (3.4.1)."
        )
    return {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "kind": kind,
        "estimator": estimator,
        "state": state,
        "config": dict(config) if config else {},
        "seed": int(seed) if seed is not None else None,
        "git_sha": git_info().get("sha"),
        "data_fingerprint": data_fingerprint,
        "extra": dict(extra) if extra else {},
    }


def _validate(envelope):
    if not isinstance(envelope, dict):
        raise TypeError(f"artifact envelope must be a dict, got {type(envelope).__name__}")
    missing = [field for field in _REQUIRED_FIELDS if field not in envelope]
    if missing:
        raise ValueError(f"artifact envelope missing fields: {missing}")
    if envelope["format_version"] not in SUPPORTED_ARTIFACT_FORMAT_VERSIONS:
        raise ValueError(
            f"unsupported artifact format_version {envelope['format_version']!r}; "
            f"supported: {SUPPORTED_ARTIFACT_FORMAT_VERSIONS}"
        )
    if not envelope["estimator"]:
        raise ValueError("artifact envelope must name its estimator (3.4.1)")
    return envelope


def save_artifact(path, envelope):
    _validate(envelope)
    atomic_write_bytes(path, pickle.dumps(envelope, protocol=pickle.HIGHEST_PROTOCOL))
    return path


def load_artifact(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"artifact checkpoint not found: {path}")
    with open(path, "rb") as f:
        envelope = pickle.load(f)
    return _validate(envelope)


def load_detector(path):
    """Load any registered detector by the `kind` recorded in its envelope."""
    envelope = load_artifact(path)
    try:
        cls = _REGISTRY[envelope["kind"]]
    except KeyError:
        raise ValueError(
            f"unknown detector kind {envelope['kind']!r}; registered: {sorted(_REGISTRY)}"
        ) from None
    return cls.from_envelope(envelope)


class FittedModel(Detector):
    """Base class for a detector whose entire fitted state is an explicit artifact.

    Subclasses set `kind`, implement `fit`/`score`/`state_dict`, and provide a
    `from_state` classmethod that rebuilds the object from a checkpointed state.
    `fit` is where every statistic is estimated and it returns nothing; callers
    fit on train only (1.8 / 4.1). `score` is higher-is-more-anomalous and
    `predict` comes from the `Detector` contract (4.1.1).
    """

    kind = None
    estimator = None

    def fit(self, data):
        raise NotImplementedError

    def score(self, data):
        raise NotImplementedError

    def state_dict(self):
        raise NotImplementedError

    @classmethod
    def from_state(cls, state, *, config, estimator):
        raise NotImplementedError

    @classmethod
    def from_envelope(cls, envelope):
        if not cls.kind or envelope["kind"] != cls.kind:
            raise ValueError(f"{cls.__name__} cannot load a {envelope['kind']!r} artifact")
        return cls.from_state(envelope["state"], config=envelope["config"], estimator=envelope["estimator"])

    @classmethod
    def load(cls, path):
        return cls.from_envelope(load_artifact(path))

    def save(self, path, *, config=None, seed=None, data_fingerprint=None, extra=None):
        return save_artifact(
            path,
            make_envelope(
                kind=self.kind,
                estimator=self.estimator,
                state=self.state_dict(),
                config=config,
                seed=seed,
                data_fingerprint=data_fingerprint,
                extra=extra,
            ),
        )
