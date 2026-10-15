"""The Phase 4 benchmark runner: one row per cell of the experiment grid (4.2.1).

The grid is `model x novelty_setting x split x seed x context_length`, declared
in `CONFIG["benchmark"]` — the axes tasklist 4.2.1, Q3 and 4.4.1 (>=5 seeds per
cell) and 7.1 (the context-length sweep) all hang off. Each cell fits one
detector on the rows its novelty-setting provider hands it, scores the scored
split, and writes one JSON row — plus a regenerated `results.csv` projection of
every row — into a single `results/` directory, so any number in a later table
can be traced back to the run that produced it (4.7.4).

Two seams are deliberately stubs today:

- **Novelty-setting providers** are registered by the Phase 6 drivers (6.2
  leave-one-action-out, 6.4 unseen transitions, 6.5 temporal corruptions).
  Until a setting has a provider its cells raise, naming the task that will
  supply it. A runner that quietly scored a different setting than the row
  claims is exactly how a benchmark result becomes untraceable.
- **Metrics** come in 4.5 and need the ground-truth manifest (6.7), so a row
  records what was run, fitted and scored — never how well it did. Per 6.1,
  Phase 4 metrics do not start before the novelty definition is written.

Names say novelty, not anomaly (6.1): the settled framing is unseen-during-
training, with no clinical claim. The axis is still the one 4.2.1 calls
`anomaly_setting`.
"""

import argparse
import csv
import io
import itertools
import json
import os
import re
import time
from dataclasses import dataclass

import numpy as np

from scripts.config import CONFIG
from scripts.data.splits import SPLIT_NAMES
from scripts.manifest import git_info
from scripts.models.detector import Detector
from scripts.models.non_generative import (
    EuclideanDetector,
    IsolationForestDetector,
    MahalanobisDetector,
    MarkovDetector,
    PCADetector,
)
from scripts.utils.hashing import hash_array, hash_json
from scripts.utils.io import atomic_write_json, atomic_write_text
from scripts.utils.seeding import set_seed

# The novelty settings Q3 scopes the grid over, each bound to the tasklist task
# whose driver will register its provider. A missing provider is reported with
# that citation rather than a generic "not implemented".
NOVELTY_SETTINGS = {
    "leave_one_action_out": "6.2",
    "unseen_transition": "6.4",
    "temporal_corruption": "6.5",
}

# model id -> zero-argument-ish factory taking the cell seed. The `mahalanobis_*`
# ids are 3.4.1's covariance estimators under explicit names: the estimator is
# the experimental variable (3.4.1), so it is part of the model identity rather
# than a hidden constructor default. Which ids the grid runs is a CONFIG choice.
MODEL_FACTORIES = {
    "euclidean": lambda seed: EuclideanDetector(),
    "isolation_forest": lambda seed: IsolationForestDetector(seed=seed),
    "mahalanobis_diagonal": lambda seed: MahalanobisDetector("diagonal"),
    "mahalanobis_full": lambda seed: MahalanobisDetector("full"),
    "mahalanobis_regularized": lambda seed: MahalanobisDetector("regularized"),
    "mahalanobis_ledoit_wolf": lambda seed: MahalanobisDetector("ledoit_wolf"),
    "mahalanobis_pca": lambda seed: MahalanobisDetector("pca_mahalanobis", seed=seed),
    "markov": lambda seed: MarkovDetector(),
    "pca": lambda seed: PCADetector(seed=seed),
}

# Setting name -> provider. Populated by `register_setting`; empty until the
# Phase 6 drivers land, which is why the default grid is inspectable but not
# executable. Tests register their own providers here.
SETTING_PROVIDERS = {}

# The row schema. One JSON object per cell with exactly these keys, and the
# CSV columns in this order, so a row written today and a row written after 4.5
# adds metric columns land in predictable places.
RESULT_FIELDS = (
    "model",
    "novelty_setting",
    "split",
    "seed",
    "context_length",
    "n_train",
    "n_scored",
    "fit_fingerprint",
    "duration_sec",
    "git_sha",
    "config_hash",
    "created_at",
)

GRID_AXES = ("models", "novelty_settings", "splits", "seeds", "context_lengths")
_NAME_RE = re.compile(r"^[a-z0-9_]+$")


def register_setting(name, provider):
    """Attach a Phase 6 driver to a novelty setting (the 4.2.1 extension seam).

    The provider contract: `provider(cell, config)` returns a dict with
    `"train"` and `"score"` row arrays for that cell — the rows to fit on and
    the rows to score, already scoped by the setting (and by `cell.split`,
    `cell.seed`, `cell.context_length`). Ground-truth labels arrive with the
    manifest in 6.7; until then rows carry no metrics.
    """
    if name not in NOVELTY_SETTINGS:
        raise ValueError(f"unknown novelty setting {name!r}; known: {sorted(NOVELTY_SETTINGS)}")
    if name in SETTING_PROVIDERS:
        raise RuntimeError(f"novelty setting {name!r} already has a provider; providers are not replaced")
    SETTING_PROVIDERS[name] = provider


def build_model(name, seed):
    """Construct one benchmark model; unknown or non-Detector models fail loudly."""
    try:
        factory = MODEL_FACTORIES[name]
    except KeyError:
        raise ValueError(f"unknown benchmark model {name!r}; known: {sorted(MODEL_FACTORIES)}") from None
    model = factory(seed)
    if not isinstance(model, Detector):
        raise TypeError(f"benchmark model {name!r} produced {type(model).__name__}, not a Detector (4.1.1)")
    return model


@dataclass(frozen=True)
class BenchmarkCell:
    """One point of the grid; frozen so a cell stays a stable identity."""

    model: str
    novelty_setting: str
    split: str
    seed: int
    context_length: int

    def axes(self):
        return {
            "model": self.model,
            "novelty_setting": self.novelty_setting,
            "split": self.split,
            "seed": self.seed,
            "context_length": self.context_length,
        }

    def cell_id(self):
        """Filename-safe identity containing all five axes (no axis is implicit)."""
        return (
            f"{self.model}__{self.novelty_setting}__{self.split}"
            f"__s{self.seed}__k{self.context_length}"
        )


def _validate_axis(spec, axis, allowed=None, kind=int):
    values = spec.get(axis)
    if not isinstance(values, (list, tuple)) or len(values) == 0:
        raise ValueError(f"benchmark axis {axis!r} must be a non-empty list (4.2.1); got {values!r}")
    for value in values:
        if not isinstance(value, kind) or isinstance(value, bool):
            raise ValueError(f"benchmark axis {axis!r} must contain {kind.__name__}s; got {value!r}")
        if allowed is not None and value not in allowed:
            raise ValueError(f"unknown value {value!r} on benchmark axis {axis!r}; known: {sorted(allowed)}")
        if isinstance(value, str) and not _NAME_RE.match(value):
            raise ValueError(f"benchmark axis {axis!r} value {value!r} is not a plain [a-z0-9_] name")
    return list(values)


def benchmark_grid(config):
    """Every cell of `model x novelty_setting x split x seed x context_length`.

    Validation happens here, before anything executes: an unknown model or
    setting in the config is a typo that must not survive to row 170 of a
    225-cell run. Axes vary in declaration order, first slowest, so the grid
    order is stable across runs.
    """
    spec = config.get("benchmark")
    if not isinstance(spec, dict):
        raise ValueError("config has no 'benchmark' grid (4.2.1)")
    models = _validate_axis(spec, "models", allowed=MODEL_FACTORIES, kind=str)
    settings = _validate_axis(spec, "novelty_settings", allowed=NOVELTY_SETTINGS, kind=str)
    splits = _validate_axis(spec, "splits", allowed=SPLIT_NAMES, kind=str)
    seeds = _validate_axis(spec, "seeds", kind=int)
    context_lengths = _validate_axis(spec, "context_lengths", kind=int)

    for seed in seeds:
        if not 0 <= seed < 2**32:
            raise ValueError(f"benchmark seeds must fit numpy's seed range [0, 2**32); got {seed!r}")
    for context_length in context_lengths:
        if context_length < 1:
            raise ValueError(f"context lengths must be >= 1; got {context_length!r}")

    return [
        BenchmarkCell(model, setting, split, seed, context_length)
        for model, setting, split, seed, context_length in itertools.product(
            models, settings, splits, seeds, context_lengths
        )
    ]


def _rows(data, key):
    """One non-empty row block from a provider's return value."""
    if not isinstance(data, dict) or key not in data:
        raise ValueError(
            f"novelty provider must return a dict with 'train' and 'score' row arrays; missing {key!r}"
        )
    rows = np.asarray(data[key])
    if len(rows) == 0:
        raise ValueError(f"novelty provider returned no {key!r} rows; an empty split is an empty coverage cell")
    return rows


def run_cell(cell, config, results_dir=None):
    """Fit, score, and persist one row. Returns the row that was written.

    Fit provenance is asserted (4.1.2) and score integrity re-checked (shape
    one per scored row, all finite) because either failure would otherwise
    surface later as a plausible-but-wrong metric rather than at the run that
    produced it. A cell that fails raises: there is no status field yet —
    complete/missing/failed accounting is 4.2.3's job.
    """
    results_dir = results_dir or config["benchmark"]["results_dir"]
    set_seed(cell.seed)

    provider = SETTING_PROVIDERS.get(cell.novelty_setting)
    if provider is None:
        task = NOVELTY_SETTINGS[cell.novelty_setting]
        raise RuntimeError(
            f"no provider registered for novelty setting {cell.novelty_setting!r}; "
            f"its driver is task {task} of the tasklist (Phase 6)"
        )

    started = time.monotonic()
    data = provider(cell, config)
    train = _rows(data, "train")
    score = _rows(data, "score")

    model = build_model(cell.model, cell.seed)
    model.fit(train)
    model.assert_fitted_on(train)
    scores = np.asarray(model.score(score))
    if scores.shape[0] != len(score):
        raise ValueError(
            f"{cell.model}.score returned {scores.shape[0]} scores for {len(score)} scored rows"
        )
    if not np.isfinite(scores).all():
        raise ValueError(f"{cell.model}.score returned a non-finite score; metrics would silently propagate it")

    row = {
        "model": cell.model,
        "novelty_setting": cell.novelty_setting,
        "split": cell.split,
        "seed": cell.seed,
        "context_length": cell.context_length,
        "n_train": int(len(train)),
        "n_scored": int(len(score)),
        "fit_fingerprint": model.fit_fingerprint_,
        "duration_sec": round(time.monotonic() - started, 6),
        "git_sha": git_info().get("sha"),
        "config_hash": hash_json(config),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    unexpected = [key for key in row if key not in RESULT_FIELDS]
    if unexpected:
        raise ValueError(f"row fields {unexpected} are not declared in RESULT_FIELDS (4.2.1)")
    # `hash_array` is what `assert_fitted_on` checked against, so the recorded
    # fingerprint is the same value a later audit re-derives from the train rows.
    if row["fit_fingerprint"] != hash_array(train):
        raise RuntimeError("row's fit_fingerprint does not match the rows the model was fitted on (4.1.2)")

    atomic_write_json(os.path.join(results_dir, f"{cell.cell_id()}.json"), row)
    write_results_csv(results_dir)
    return row


def write_results_csv(results_dir):
    """Regenerate `results.csv` from every row JSON — a projection, not a second source of truth."""
    rows = []
    for name in sorted(os.listdir(results_dir)):
        if name.endswith(".json"):
            with open(os.path.join(results_dir, name), encoding="utf-8") as handle:
                rows.append(json.load(handle))
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(RESULT_FIELDS), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    path = os.path.join(results_dir, "results.csv")
    atomic_write_text(path, buffer.getvalue())
    return path


def run_benchmark(config, results_dir=None):
    """Execute every declared cell once; returns the rows written."""
    cells = benchmark_grid(config)
    results_dir = results_dir or config["benchmark"]["results_dir"]
    return [run_cell(cell, config, results_dir) for cell in cells]


def main(argv=None, config=None):
    parser = argparse.ArgumentParser(description="Run the Phase 4 benchmark grid (4.2.1)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the grid's cell ids without executing any of them",
    )
    args = parser.parse_args(argv)
    cfg = CONFIG if config is None else config
    cells = benchmark_grid(cfg)
    if args.dry_run:
        for cell in cells:
            print(cell.cell_id())
        print(f"{len(cells)} cells")
        return cells
    results_dir = cfg["benchmark"]["results_dir"]
    rows = run_benchmark(cfg)
    print(f"wrote {len(rows)} rows to {results_dir}")
    return rows

if __name__ == "__main__":
    main()
