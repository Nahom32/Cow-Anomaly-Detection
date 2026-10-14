"""Unified checkpoint format (3.1).

Carries model/optimizer/scheduler states, epoch/best metrics, resolved config,
seed, RNG states, git SHA, data fingerprint, and format version. Legacy
weights-only checkpoints (just a state_dict) are not loadable through this
API — they predate the schema and must be archived.
"""

import os
import random

import numpy as np
import torch

from scripts.manifest import git_info
from scripts.utils.hashing import hash_json
from scripts.utils.io import atomic_torch_save

CHECKPOINT_FORMAT_VERSION = 1

REQUIRED_FIELDS = (
    "format_version",
    "model_state",
    "optimizer_state",
    "scheduler_state",
    "epoch",
    "best_metric",
    "best_metric_epoch",
    "best_state",
    "config",
    "seed",
    "rng_state",
    "git_sha",
    "data_fingerprint",
)

SUPPORTED_FORMAT_VERSIONS = (CHECKPOINT_FORMAT_VERSION,)


def _git_sha(repo_root=None):
    info = git_info(repo_root=repo_root)
    return info.get("sha")


def rng_state():
    """Snapshot python/numpy/torch RNG state so a resumed run continues the stream."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": None,
    }
    if torch.cuda.is_available():
        try:
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
        except Exception:
            state["torch_cuda"] = None
    return state


def restore_rng_state(state):
    """Restore RNG state from a snapshot. Best-effort for CUDA."""
    if not isinstance(state, dict):
        return
    try:
        if "python" in state:
            random.setstate(state["python"])
    except Exception:
        pass
    try:
        if "numpy" in state:
            np.random.set_state(state["numpy"])
    except Exception:
        pass
    try:
        if "torch" in state:
            torch.set_rng_state(state["torch"])
    except Exception:
        pass
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        try:
            torch.cuda.set_rng_state_all(state["torch_cuda"])
        except Exception:
            pass


def data_fingerprint(manifest=None, arrays=None, splits=None, extras=None):
    """Derive a stable data fingerprint from manifest/arrays/splits."""
    parts = {}
    if manifest is not None:
        try:
            if getattr(manifest, "data", None):
                parts["manifest"] = {
                    "arrays": manifest.data.get("arrays", {}),
                    "splits": manifest.data.get("splits", {}),
                    "artifacts": {
                        k: v.get("sha256") for k, v in (manifest.data.get("artifacts", {}) or {}).items()
                        if v.get("sha256")
                    },
                }
        except Exception:
            parts["manifest"] = manifest.data if hasattr(manifest, "data") else str(manifest)
    if arrays is not None:
        parts["arrays"] = arrays
    if splits is not None:
        parts["splits"] = splits
    if extras is not None:
        parts["extras"] = extras
    return hash_json(parts) if parts else None


def _validate(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint must be a dict")
    missing = [f for f in REQUIRED_FIELDS if f not in checkpoint]
    if missing:
        raise ValueError(f"checkpoint missing required fields: {missing}")
    fmt = checkpoint.get("format_version")
    if fmt not in SUPPORTED_FORMAT_VERSIONS:
        raise ValueError(f"unsupported checkpoint format_version: {fmt}")
    return checkpoint


def build_checkpoint(
    *,
    model_state,
    optimizer_state=None,
    scheduler_state=None,
    epoch,
    best_metric=None,
    best_metric_epoch=None,
    best_state=None,
    config,
    seed,
    rng_state_snapshot=None,
    data_fingerprint=None,
    git_sha=None,
    repo_root=None,
    **extras,
):
    """Build the full checkpoint envelope with required fields."""
    if rng_state_snapshot is None:
        rng_state_snapshot = rng_state()
    if git_sha is None:
        git_sha = _git_sha(repo_root=repo_root)
    cp = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_state": model_state,
        "optimizer_state": optimizer_state,
        "scheduler_state": scheduler_state,
        "epoch": epoch,
        "best_metric": best_metric,
        "best_metric_epoch": best_metric_epoch,
        "best_state": best_state,
        "config": dict(config) if config is not None else {},
        "seed": int(seed) if seed is not None else None,
        "rng_state": rng_state_snapshot,
        "git_sha": git_sha,
        "data_fingerprint": data_fingerprint,
    }
    cp.update(extras)
    _validate(cp)
    return cp


def save_checkpoint(path, checkpoint):
    cp = _validate(checkpoint)
    atomic_torch_save(path, cp)
    return path


def load_checkpoint(path, map_location="cpu"):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # Older torch versions don't accept weights_only; try without it.
        checkpoint = torch.load(path, map_location=map_location)
    except Exception:
        raise
    # Detect legacy bare state_dict: if it's just a dict of tensors (or other values) with no format_version.
    if isinstance(checkpoint, dict) and "format_version" not in checkpoint and all(
        isinstance(k, str) or isinstance(k, tuple) and all(isinstance(x, str) for x in k)
        for k in checkpoint.keys()  # heuristic is loose; but bare state_dict keys are param names
    ):
        # More reliably: if keys look like model param names and required top-level missing -> legacy.
        # But easier: check for absence of REQUIRED_FIELDS subset.
        pass  # fall through to validation which will fail loudly
    _validate(checkpoint)
    return checkpoint


def checkpoint_format_version(obj):
    if isinstance(obj, dict) and "format_version" in obj:
        return obj.get("format_version")
    return None


def is_legacy_checkpoint(obj):
    """True if obj looks like a legacy weights-only checkpoint (bare state_dict)."""
    if not isinstance(obj, dict):
        return False
    if "format_version" in obj:
        return False
    # A bare state_dict has string keys and tensor values (typical). The new format has mixed types.
    # Heuristic: if it has model_state/best_state keys, it's the new format but missing format_version? Unlikely.
    # Legacy is usually just param->tensor mapping.
    # Check: if any key is in REQUIRED_FIELDS top-level and it's not the legacy case? No.
    # Conservative: if no REQUIRED_FIELDS present at top level, treat as legacy if it's non-empty dict of tensors/params.
    has_new_keys = any(k in obj for k in ("model_state", "optimizer_state", "epoch", "config"))
    if has_new_keys and "format_version" not in obj:
        return True  # malformed new-format-like but missing version
    # Pure legacy: dict of state_dict items
    return True
