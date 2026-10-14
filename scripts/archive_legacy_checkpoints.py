"""Archive legacy weights-only checkpoints (predate 3.1 schema).

Legacy artifacts: bare torch state_dicts saved as .pth, without format_version.
They are scientifically void if trained under leaked split (as noted in tasklist).
This moves them to an archive directory with a sidecar .json note.
"""

import argparse
import json
import os
import shutil
from datetime import datetime

import torch

from scripts.utils.io import atomic_write_json


def detect_legacy(path):
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        try:
            obj = torch.load(path, map_location="cpu")
        except Exception:
            return False
    except Exception:
        return False
    # Legacy if it's a dict but has no format_version
    if isinstance(obj, dict) and "format_version" not in obj:
        # Heuristic: if it contains model_state/optimizer_state it's a malformed new one; treat as needing migration.
        # Pure legacy is typically just param name->tensor. But also could be full checkpoint without version (rare).
        # Presence of new keys without version means it's "legacy" relative to required schema.
        return True
    # If it's a state_dict saved directly, torch.load returns the dict; still no format_version.
    # Also could be non-dict? unlikely.
    return False


def archive_path(src, archive_root):
    rel = os.path.relpath(src, start=os.path.dirname(src) or ".")
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    dest_dir = os.path.join(archive_root, "legacy_checkpoints", ts)
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, os.path.basename(src))
    shutil.move(src, dest)
    note = {
        "archived_at": ts,
        "original_path": os.path.abspath(src),
        "reason": "Legacy weights-only checkpoint (no format_version); predates unified schema (3.1).",
        "note": "If trained under leaked split, scientifically void (tasklist 3.1.2). Not backfilled.",
    }
    atomic_write_json(os.path.join(dest_dir, "archive_note.json"), note)
    return dest


def main():
    p = argparse.ArgumentParser(description="Archive legacy checkpoints")
    p.add_argument("--path", action="append", help="Path to .pth file(s) to check/archive")
    p.add_argument("--root", default=".", help="Directory to scan recursively for *.pth")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--archive-root", default=".")
    args = p.parse_args()

    candidates = []
    if args.path:
        candidates.extend(args.path)
    else:
        for dirpath, _, filenames in os.walk(args.root):
            if "venv" in dirpath or "__pycache__" in dirpath:
                continue
            for fn in filenames:
                if fn.endswith(".pth"):
                    candidates.append(os.path.join(dirpath, fn))

    legacy = []
    for c in candidates:
        try:
            if detect_legacy(c):
                legacy.append(c)
        except Exception:
            pass

    print(f"Found {len(legacy)} legacy checkpoint(s):")
    for c in legacy:
        print(f"  {c}")
    if args.dry_run or not legacy:
        return
    for c in legacy:
        try:
            dest = archive_path(c, args.archive_root)
            print(f"Archived: {c} -> {dest}")
        except Exception as e:
            print(f"Failed to archive {c}: {e}")


if __name__ == "__main__":
    main()
