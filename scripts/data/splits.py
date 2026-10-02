"""One canonical group-level train/val/test assignment, shared by every model.

Two separate leaks came from the split being computed independently at each call
site. `step_flat_vae` called `train_test_split` on row indices with no `groups=`,
so consecutive frames of one cow landed on both sides (problem #5). `step_lstm_vae`
split by video, on its own, from a `set` whose iteration order is not stable across
processes — so the two models were trained on different partitions of the same data
and no Flat-vs-LSTM comparison meant anything.

Both are the same problem: the split is a property of the *data*, not of one stage's
index array, so it is derived once here, from the annotation frame before any
feature extraction, persisted, and read back by every stage. Deriving it from `df`
rather than from the rows that survived feature extraction is what makes the stages
agree — a video whose crops are all missing still gets an assignment, so the LSTM
stage's shorter video list is a subset of the flat stage's rather than a different
partition.

The split unit is a `group_key` column, `"video_id"` by default. `"target_id"` is the
stricter unit — AVA cows recur across videos, so grouping on video still leaves one
cow in train and val — but its reliability is unverified (tasklist Q1), so it is a
config value to be switched once checked, not a hardcoded assumption.

Grouping is enforced structurally rather than by convention: `resolve_indices`
raises on a row whose group is absent from the manifest, so a stage cannot quietly
train on a subset no recorded split accounts for.
"""

import json
import os

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from scripts.utils.hashing import hash_json
from scripts.utils.io import atomic_write_json

SPLIT_FILENAME = "split_manifest.json"
SPLIT_FORMAT_VERSION = 1
SPLIT_NAMES = ("train", "val", "test")
GROUP_KEYS = ("video_id", "target_id")


def row_groups(df, group_key):
    """The group label of every annotation row, as strings.

    Video ids are read as strings and target ids are not, so a `target_id` of `1` and
    a `video_id` of `1` cannot silently become the same group.
    """
    if group_key not in df.columns:
        raise ValueError(
            f"split group key {group_key!r} is not a column of the annotations "
            f"(have: {list(df.columns)}). Grouping on an absent column would silently "
            "degrade to a frame-level split."
        )
    return df[group_key].astype(str).to_numpy()


def unique_groups(df, group_key):
    return sorted(set(row_groups(df, group_key).tolist()))


def _hold_out(population, fraction, random_seed):
    """`(kept, held_out)` group lists from a grouped shuffle.

    `GroupShuffleSplit` rather than a plain shuffle of the unique ids, so the
    grouping is explicit at the call site rather than implied by the array
    happening to be unique, and so a caller passing rows with a `groups=` array gets
    the same behaviour as one passing unique ids.
    """
    if fraction <= 0:
        return list(population), []
    splitter = GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=random_seed)
    try:
        kept_idx, held_idx = next(splitter.split(np.arange(len(population)), groups=population))
    except ValueError as exc:
        # sklearn's message is about `n_samples`; the actionable fact is that there
        # are too few groups to hold any out, which is what a reader needs.
        raise ValueError(
            f"a {fraction:g} hold-out over {len(population)} group(s) leaves an empty split; "
            "reduce the split fraction or add more groups"
        ) from exc
    kept = [population[i] for i in kept_idx]
    held = [population[i] for i in held_idx]
    if not kept or not held:
        raise ValueError(
            f"a {fraction:g} hold-out over {len(population)} groups leaves an empty split; "
            "reduce the split fraction or add more groups"
        )
    return kept, held


def _assert_disjoint(assignment):
    seen = {}
    for name in SPLIT_NAMES:
        for group in assignment[name]:
            if group in seen:
                raise RuntimeError(
                    f"group {group!r} is in both {seen[group]!r} and {name!r}; a group appearing "
                    "twice defeats the entire point of a grouped split"
                )
            seen[group] = name


def build_split(df, val_split=0.2, test_split=0.0, random_seed=42, group_key="video_id"):
    """`{split_name: [group, ...]}` from a two-stage grouped shuffle.

    Test is carved out first, so val is a fraction of what remains and the reported
    val fraction is not inflated by test rows. `test_split` defaults to 0 because
    1.5 has not created a test set yet; the key is still present, so 1.5 is a config
    change rather than a schema change.

    The two draws use offset seeds (`random_seed`, `random_seed + 1`) so the same
    group cannot land in test and val as an artefact of two identical draws.
    """
    if not 0.0 <= val_split < 1.0:
        raise ValueError(f"val_split must be in [0, 1), got {val_split}")
    if not 0.0 <= test_split < 1.0:
        raise ValueError(f"test_split must be in [0, 1), got {test_split}")
    if val_split + test_split >= 1.0:
        raise ValueError(
            f"val_split ({val_split}) + test_split ({test_split}) must leave groups for train"
        )

    remaining = unique_groups(df, group_key)
    if not remaining:
        raise ValueError("no groups to split: the annotations contain no rows for this group key")

    assignment = {name: [] for name in SPLIT_NAMES}
    if test_split > 0:
        remaining, assignment["test"] = _hold_out(remaining, test_split, random_seed)
    if val_split > 0:
        remaining, assignment["val"] = _hold_out(remaining, val_split, random_seed + 1)
    assignment["train"] = list(remaining)

    _assert_disjoint(assignment)
    return assignment


def save_split(output_dir, assignment, group_key, val_split, test_split, random_seed,
               filename=SPLIT_FILENAME):
    """Persist the assignment, so a later run reuses it instead of redrawing it."""
    payload = {
        "format_version": SPLIT_FORMAT_VERSION,
        "group_key": group_key,
        "val_split": val_split,
        "test_split": test_split,
        "random_seed": random_seed,
        "assignment": {name: sorted(assignment[name]) for name in SPLIT_NAMES},
    }
    path = os.path.join(output_dir, filename)
    atomic_write_json(path, payload)
    return path


def load_split(output_dir, filename=SPLIT_FILENAME):
    path = os.path.join(output_dir, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"no split manifest at {path}. It is written by the first stage that needs it; without "
            "it a stage must draw its own split, which is exactly 1.4's bug."
        )
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    version = payload.get("format_version")
    if version != SPLIT_FORMAT_VERSION:
        raise ValueError(
            f"split manifest at {path} is format {version}, this code writes {SPLIT_FORMAT_VERSION}. "
            "Delete the manifest to have it rebuilt rather than reading it on a guess."
        )
    missing = [name for name in SPLIT_NAMES if name not in payload.get("assignment", {})]
    if missing:
        raise ValueError(f"split manifest at {path} is missing the {missing} split(s)")
    return payload


def resolve_split(output_dir, df, val_split=0.2, test_split=0.0, random_seed=42,
                  group_key="video_id", filename=SPLIT_FILENAME):
    """Load the run's split, or build and persist it if this is the first stage.

    A manifest whose recorded group key, seed or fractions disagree with the config is
    a stale artifact, not something to honour quietly: the stages would each believe
    they share a partition while disagreeing about which one. That is why every field
    is compared, including `random_seed` — reusing a split drawn under a different
    seed is how two runs end up reporting the same number for different partitions.
    """
    path = os.path.join(output_dir, filename)
    if not os.path.isfile(path):
        assignment = build_split(
            df, val_split=val_split, test_split=test_split,
            random_seed=random_seed, group_key=group_key,
        )
        save_split(output_dir, assignment, group_key, val_split, test_split,
                   random_seed, filename=filename)
        return assignment

    payload = load_split(output_dir, filename=filename)
    requested = {
        "group_key": group_key,
        "val_split": val_split,
        "test_split": test_split,
        "random_seed": random_seed,
    }
    stale = {name: payload.get(name) for name, value in requested.items()
             if payload.get(name) != value}
    if stale:
        raise RuntimeError(
            f"the split manifest in {output_dir} was drawn under {stale} but this run asked for "
            f"{requested}. Delete the manifest to draw a new split, or restore the config that "
            "produced it; reusing this one puts the stages on different partitions."
        )

    assignment = {name: list(payload["assignment"][name]) for name in SPLIT_NAMES}
    unseen = sorted(set(unique_groups(df, group_key)) - _assigned_groups(assignment))
    if unseen:
        raise RuntimeError(
            f"{len(unseen)} group(s) in the annotations are absent from the split manifest in "
            f"{output_dir} (e.g. {unseen[:5]}). The annotations changed after the split was drawn, "
            "so the recorded partition does not cover this data; delete the manifest to redraw."
        )
    return assignment


def _assigned_groups(assignment):
    return set().union(*(set(assignment[name]) for name in SPLIT_NAMES)) if assignment else set()


def resolve_indices(assignment, groups):
    """Row indices per split name, for rows labelled by `groups`.

    Raises rather than dropping a row whose group is missing: an unassigned row would
    be absent from training and from validation without either count noticing, which
    is the failure mode 1.4 exists to remove.
    """
    labels = [str(group) for group in groups]
    membership = {name: set(assignment[name]) for name in SPLIT_NAMES}
    assigned = _assigned_groups(assignment)

    unknown = sorted({label for label in labels if label not in assigned})
    if unknown:
        raise RuntimeError(
            f"{len(unknown)} row(s) belong to group(s) with no split assignment "
            f"(e.g. {unknown[:5]}). Every row must be in exactly one split; draw the split from the "
            "same annotation frame the rows came from."
        )

    indices = {name: [] for name in SPLIT_NAMES}
    for position, label in enumerate(labels):
        for name in SPLIT_NAMES:
            if label in membership[name]:
                indices[name].append(position)
                break
    return {name: np.array(rows, dtype=np.int64) for name, rows in indices.items()}


def describe(assignment):
    """A compact summary for the run manifest and the console."""
    return {
        "sha256": hash_json({name: sorted(assignment[name]) for name in SPLIT_NAMES}),
        "n_groups": {name: len(assignment[name]) for name in SPLIT_NAMES},
    }