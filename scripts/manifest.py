"""Run manifest: what produced the numbers in an output directory.

`is_step_done` used to be existence-only, so a model trained before a config
change was silently reused and reported as current. This records enough of the
run's provenance — git SHA, config hash, seed, and content hashes of every
artifact that feeds a step — to tell whether a step's output still corresponds
to its inputs.

Phase 3.6.3 replaces the existence check outright; until then this is what
makes the staleness visible.
"""

import json
import os
import subprocess
import sys
import time

from scripts.utils.hashing import hash_array, hash_json, sha256_file
from scripts.utils.io import atomic_write_json

MANIFEST_FILENAME = "run_manifest.json"
MANIFEST_FORMAT_VERSION = 1


def git_info(repo_root=None):
    """Best-effort git provenance. Never raises — an unversioned run is legal."""
    def run(*args):
        try:
            return subprocess.run(
                ["git", *args],
                cwd=repo_root,
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    return {
        "sha": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
    }


class RunManifest:
    """The `run_manifest.json` of one output directory."""

    def __init__(self, output_dir, filename=MANIFEST_FILENAME):
        self.output_dir = output_dir
        self.path = os.path.join(output_dir, filename)
        self.data = {
            "format_version": MANIFEST_FORMAT_VERSION,
            "created_at": _timestamp(),
            "steps": {},
        }

    # ── construction ────────────────────────────────────────────────────

    @classmethod
    def create(cls, output_dir, config, seed, device=None, command=None, repo_root=None):
        manifest = cls(output_dir)
        manifest.begin_run(config, seed, device=device, command=command, repo_root=repo_root)
        return manifest

    def begin_run(self, config, seed, device=None, command=None, repo_root=None):
        """Start (or restart) a run in this output directory, keeping prior steps.

        The step entries are deliberately preserved: they record the config each
        artifact was actually produced under, which is what tells a resumable
        step apart from a stale one. Overwriting them with the current config
        would make a stale checkpoint look current.
        """
        prior = self.data.get("config_hash")
        if prior is not None and prior != hash_json(config):
            self.data["previous_run"] = {"config_hash": prior, "at": self.data.get("created_at")}
        self.data["git"] = git_info(repo_root)
        self.data["config"] = dict(config)
        self.data["config_hash"] = hash_json(config)
        self.data["seed"] = seed
        self.data["device"] = device
        self.data["command"] = list(command) if command else list(sys.argv)
        self.data.setdefault("created_at", _timestamp())
        self.data["resumed_at"] = _timestamp()
        return self

    @classmethod
    def load(cls, output_dir, filename=MANIFEST_FILENAME):
        manifest = cls(output_dir, filename)
        try:
            with open(manifest.path) as f:
                loaded = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(loaded, dict):
            return None
        manifest.data = loaded
        return manifest

    # ── recording ───────────────────────────────────────────────────────

    def record(self, key, value):
        self.data[key] = value
        return self

    def record_artifact(self, name, path):
        """Bind a file to the run by content hash, not by name."""
        if path is None or not os.path.isfile(path):
            self.data.setdefault("artifacts", {})[name] = {"path": path, "missing": True}
            return self
        self.data.setdefault("artifacts", {})[name] = {
            "path": os.path.abspath(path),
            "sha256": sha256_file(path),
            "bytes": os.path.getsize(path),
        }
        return self

    def record_array(self, name, array, **meta):
        entry = {"sha256": hash_array(array), "shape": list(array.shape)}
        entry.update(meta)
        self.data.setdefault("arrays", {})[name] = entry
        return self

    def record_split(self, name, assignment, ordered=False):
        """Record a `{split_name: [item, ...]}` mapping by hash.

        Membership splits hash order-insensitively by default: re-shuffling the
        id list must not invalidate a step. Pass ``ordered=True`` for a temporal
        split, where the order of the items *is* the information.
        """
        if not ordered:
            assignment = {k: sorted(v) for k, v in assignment.items()}
        self.data.setdefault("splits", {})[name] = {
            "sha256": hash_json(assignment),
            "ordered": ordered,
            "n_items": sum(len(v) for v in assignment.values()),
            "sizes": {k: len(v) for k, v in assignment.items()},
        }
        return self

    def record_step(self, step, **fields):
        entry = self.data["steps"].setdefault(str(step), {})
        entry.update(fields)
        entry["completed_at"] = _timestamp()
        return self

    def save(self):
        atomic_write_json(self.path, self.data)
        return self.path

    # ── queries ─────────────────────────────────────────────────────────

    @property
    def config_hash(self):
        return self.data.get("config_hash")

    def step(self, number):
        return self.data.get("steps", {}).get(str(number))

    def is_current_for(self, config, step=None):
        """Has this run's config changed since the manifest was written?

        With `step`, that step's own recorded config hash is checked instead,
        which is what matters when only part of a run is being reused.
        """
        target = self.config_hash
        if step is not None:
            entry = self.step(step)
            if not entry:
                return False
            target = entry.get("config_hash")
        if target is None:
            return False
        return target == hash_json(config)


def _timestamp():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")
