"""Unified checkpoint format (3.1).

Carries model/optimizer/scheduler states, epoch/best metrics, resolved config,
seed, RNG states, git SHA, data fingerprint, and format version. Legacy
weights-only checkpoints (just a state_dict) are not loadable through this
API — they predate the schema and must be archived.
"""

import fnmatch
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
    _validate(checkpoint)
    atomic_torch_save(path, checkpoint)
    return path


def save_model_checkpoint(
    path,
    *,
    model_state,
    normalizer=None,
    config=None,
    seed=None,
    data_fingerprint=None,
    epoch=None,
    best_metric=None,
    best_metric_epoch=None,
    best_state=None,
    extra=None,
):
    """Write a loadable model checkpoint that carries its fitted normalizer (3.5.1).

    The final/best artifacts a scorer loads are the ones that must not be paired
    with a loose `*_feature_min.npy` sidecar, so this is the single writer for
    them: weights and `normalizer` state (a `state_dict`, not the object) go into
    one envelope and `load_checkpoint` returns both together. An optimizer and
    scheduler are deliberately absent — these are inference artifacts, not resume
    points.
    """
    checkpoint = build_checkpoint(
        model_state=model_state,
        optimizer_state=None,
        scheduler_state=None,
        epoch=0 if epoch is None else int(epoch),
        best_metric=best_metric,
        best_metric_epoch=best_metric_epoch,
        best_state=best_state,
        config=config or {},
        seed=seed,
        data_fingerprint=data_fingerprint,
        normalizer=normalizer,
        **(extra or {}),
    )
    save_checkpoint(path, checkpoint)
    return checkpoint


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


# ── Retention (3.2.2) ─────────────────────────────────────────────────

# Epoch checkpoints carry their epoch in the filename so retention can order them
# without loading each file. `best`/`final` deliberately do not: they are named
# explicitly and always retained, so they can never be pruned by accident.
EPOCH_CHECKPOINT_TEMPLATE = "{prefix}_epoch_{epoch:04d}.pth"
_EPOCH_MARKER = "_epoch_"


def epoch_checkpoint_name(prefix, epoch):
    """Canonical filename for an epoch checkpoint."""
    return EPOCH_CHECKPOINT_TEMPLATE.format(prefix=prefix, epoch=int(epoch))


def parse_epoch_checkpoint(filename):
    """The epoch encoded in an epoch-checkpoint filename, or None if not one.

    Only names produced by `epoch_checkpoint_name` parse. `best`/`model` and any
    unrelated `.pth` return None and are therefore outside retention's reach: a
    prune can only ever delete a file it can place on the epoch timeline.
    """
    stem = os.path.basename(str(filename))
    if not stem.endswith(".pth") or _EPOCH_MARKER not in stem:
        return None
    tail = stem[: -len(".pth")].rsplit(_EPOCH_MARKER, 1)[-1]
    return int(tail) if tail.isdigit() else None


def retention_plan(paths, keep_last=3, keep=()):
    """Partition `paths` into `(kept, removed)` under the retention policy (3.2.2).

    The policy is `best + final + the last N epochs`: everything named in `keep`
    (the best-val and final checkpoints, whose names carry no epoch) is retained,
    as are the `keep_last` highest-numbered epoch checkpoints. Only epoch
    checkpoints beyond that are removed; a path that does not parse as one is
    left alone, so a prune cannot delete a file it does not understand.
    """
    if keep_last < 0:
        raise ValueError(f"keep_last must be non-negative, got {keep_last}")
    keep_abs = {os.path.abspath(p) for p in keep}
    parsed = {p: parse_epoch_checkpoint(p) for p in paths}
    newest = sorted(
        ((epoch, path) for path, epoch in parsed.items() if epoch is not None),
        key=lambda item: item[0], reverse=True,
    )
    retain_abs = keep_abs | {os.path.abspath(path) for _, path in newest[:keep_last]}
    removed = sorted(
        path for path, epoch in parsed.items()
        if epoch is not None and os.path.abspath(path) not in retain_abs
    )
    removed_abs = {os.path.abspath(path) for path in removed}
    kept = sorted(path for path in paths if os.path.abspath(path) not in removed_abs)
    return kept, removed


def prune_checkpoints(paths, keep_last=3, keep=()):
    """Apply `retention_plan` to disk; returns the `(kept, removed)` partition."""
    kept, removed = retention_plan(paths, keep_last=keep_last, keep=keep)
    for path in removed:
        os.remove(path)
    return kept, removed


def prune_checkpoint_dir(directory, prefix, keep_last=3, keep=()):
    """Prune `{prefix}_*.pth` in one directory (3.2.2).

    Scoped by prefix because the flat and LSTM-VAE checkpoints share an output
    directory: an unscoped prune would let one model's retention policy delete
    the other's checkpoints.
    """
    pattern = f"{prefix}_*.pth"
    paths = [
        os.path.join(directory, name)
        for name in sorted(os.listdir(directory))
        if fnmatch.fnmatch(name, pattern)
    ]
    return prune_checkpoints(paths, keep_last=keep_last, keep=keep)


def record_and_prune_checkpoints(*, manifest, directory, prefix, keep_last, final=None, best=None, step=None):
    """Checksum the checkpoints a stage just wrote, then enforce retention.

    Recording happens before the prune so the manifest describes what was
    produced, not whatever survived. `manifest` may be None (a standalone caller
    without provenance), in which case only the prune runs.
    """
    if manifest is not None:
        for role, path in (("final", final), ("best", best)):
            if path is not None and os.path.isfile(path):
                manifest.record_checkpoint(f"{prefix}_{role}", path, role=role, step=step)
    keep = [path for path in (final, best) if path is not None]
    return prune_checkpoint_dir(directory, prefix, keep_last=keep_last, keep=keep)


# ── Resume (3.3.2) ────────────────────────────────────────────────────


def list_epoch_checkpoints(directory, prefix):
    """`(epoch, path)` for every epoch checkpoint of `prefix`, ascending by epoch.

    `directory` is scanned by filename, so an interrupted run's checkpoints can be
    ordered without loading any of them; only the names the retention policy
    recognises (`{prefix}_epoch_NNNN.pth`) are returned.
    """
    pattern = f"{prefix}_epoch_*.pth"
    if not os.path.isdir(directory):
        return []
    found = []
    for name in sorted(os.listdir(directory)):
        if not fnmatch.fnmatch(name, pattern):
            continue
        epoch = parse_epoch_checkpoint(name)
        if epoch is not None:
            found.append((epoch, os.path.join(directory, name)))
    return sorted(found)


def latest_epoch_checkpoint(directory, prefix, config=None):
    """The checkpoint to resume `prefix` from, or None (3.3.2).

    Returns the highest-epoch checkpoint. When `config` is given, checkpoints
    whose recorded config hash differs are skipped: continuing an interrupted run
    under a changed config would silently mix two experiments' weights, so the
    caller should fall back to a fresh start rather than trust the file. A
    mismatch is reported because the alternative — a silent fresh start — looks
    identical to a resume that found nothing.
    """
    candidates = list_epoch_checkpoints(directory, prefix)
    for _epoch, path in reversed(candidates):
        if config is None:
            return path
        recorded = load_checkpoint(path, map_location="cpu").get("config")
        if recorded is not None and hash_json(recorded) == hash_json(config):
            return path
    if config is not None and candidates:
        print(
            f"  {len(candidates)} epoch checkpoint(s) for {prefix!r} exist but none match the "
            "current config — starting fresh (3.3.2)"
        )
    return None


def clear_epoch_checkpoints(directory, prefix):
    """Delete every epoch checkpoint of `prefix` (a fresh run must not mix with a stale one).

    A fresh run starts at epoch 1 and would otherwise leave a previous run's
    higher-numbered epochs on disk, which retention would then treat as the
    newest and keep while deleting the current run's early epochs.
    """
    removed = [path for _, path in list_epoch_checkpoints(directory, prefix)]
    for path in removed:
        os.remove(path)
    return removed
