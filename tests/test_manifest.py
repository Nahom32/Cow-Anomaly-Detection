"""Run manifest: the record that makes a stale reuse detectable."""

import json
import os

import numpy as np
import pytest

from scripts.manifest import MANIFEST_FILENAME, RunManifest, git_info
from scripts.utils.hashing import hash_array, hash_json, sha256_file

CONFIG = {"random_seed": 42, "vae_epochs": 100, "val_split": 0.2}


@pytest.fixture
def manifest(tmp_path):
    return RunManifest.create(tmp_path, CONFIG, seed=42, device="cpu", command=["pytest"])


def test_create_records_provenance(manifest, tmp_path):
    manifest.save()
    data = json.loads((tmp_path / MANIFEST_FILENAME).read_text())
    assert data["config"] == CONFIG
    assert data["config_hash"] == hash_json(CONFIG)
    assert data["seed"] == 42
    assert data["device"] == "cpu"
    assert "created_at" in data
    assert set(data["git"]) == {"sha", "branch", "dirty"}


def test_config_hash_matches_the_config_dict(manifest):
    assert manifest.config_hash == hash_json(CONFIG)
    assert manifest.is_current_for(CONFIG)


def test_is_current_for_detects_a_config_change(manifest):
    changed = {**CONFIG, "vae_epochs": 101}
    assert not manifest.is_current_for(changed)
    assert not manifest.is_current_for({**CONFIG, "random_seed": 43})


def test_step_config_hash_detects_a_config_change(manifest):
    manifest.record_step(6, config_hash=hash_json(CONFIG))
    assert manifest.step(6)["config_hash"] == hash_json(CONFIG)
    assert manifest.is_current_for(CONFIG, step=6)
    assert not manifest.is_current_for({**CONFIG, "vae_epochs": 1}, step=6)


def test_missing_step_is_never_current(manifest):
    assert manifest.step(7) is None
    assert not manifest.is_current_for(CONFIG, step=7)


def test_record_artifact_binds_by_content_not_name(manifest, tmp_path):
    a = tmp_path / "a.bin"
    a.write_bytes(b"same")
    b = tmp_path / "b.bin"
    b.write_bytes(b"same")
    manifest.record_artifact("weights", str(a))
    manifest.record_artifact("copy", str(b))
    recorded = manifest.data["artifacts"]
    assert recorded["weights"]["sha256"] == recorded["copy"]["sha256"] == sha256_file(str(a))
    assert recorded["weights"]["bytes"] == 4
    assert os.path.isabs(recorded["weights"]["path"])


def test_record_artifact_marks_a_missing_file(manifest, tmp_path):
    manifest.record_artifact("weights", str(tmp_path / "nope.pt"))
    assert manifest.data["artifacts"]["weights"]["missing"] is True
    assert "sha256" not in manifest.data["artifacts"]["weights"]


def test_record_array_detects_value_changes(manifest):
    a = np.arange(12, dtype=np.float32).reshape(4, 3)
    manifest.record_array("features", a, n=4, dim=3)
    entry = manifest.data["arrays"]["features"]
    assert entry["sha256"] == hash_array(a)
    assert entry["shape"] == [4, 3]
    assert entry["n"] == 4 and entry["dim"] == 3
    manifest.record_array("other", a + 1)
    assert manifest.data["arrays"]["other"]["sha256"] != entry["sha256"]


def test_record_split_is_order_insensitive_by_default(manifest):
    manifest.record_split("videos", {"train": ["b", "a"], "val": ["c"]})
    entry = manifest.data["splits"]["videos"]
    assert entry["n_items"] == 3
    assert entry["sizes"] == {"train": 2, "val": 1}
    assert entry["ordered"] is False

    shuffled = RunManifest(manifest.output_dir)
    shuffled.record_split("videos", {"train": ["a", "b"], "val": ["c"]})
    assert shuffled.data["splits"]["videos"]["sha256"] == entry["sha256"]


def test_ordered_split_hashes_order(manifest):
    # A temporal split: the same items in a different order is a different split.
    manifest.record_split("time", {"train": ["a", "b"], "val": ["c"]}, ordered=True)
    forward = manifest.data["splits"]["time"]["sha256"]
    manifest.record_split("time", {"train": ["b", "a"], "val": ["c"]}, ordered=True)
    assert manifest.data["splits"]["time"]["sha256"] != forward
    assert manifest.data["splits"]["time"]["ordered"] is True


def test_split_hash_changes_when_a_video_moves(manifest):
    manifest.record_split("videos", {"train": ["a", "b"], "val": ["c"]})
    before = manifest.data["splits"]["videos"]["sha256"]
    manifest.record_split("videos", {"train": ["a"], "val": ["b", "c"]})
    assert manifest.data["splits"]["videos"]["sha256"] != before


def test_save_then_load_round_trips(manifest, tmp_path):
    manifest.record_step(6, config_hash=hash_json(CONFIG), n_features=100)
    manifest.save()
    reloaded = RunManifest.load(tmp_path)
    assert reloaded is not None
    assert reloaded.config_hash == manifest.config_hash
    assert reloaded.step(6)["n_features"] == 100
    assert reloaded.is_current_for(CONFIG, step=6)


def test_load_returns_none_for_a_missing_or_corrupt_manifest(tmp_path):
    assert RunManifest.load(tmp_path) is None
    (tmp_path / MANIFEST_FILENAME).write_text("{not json")
    assert RunManifest.load(tmp_path) is None
    (tmp_path / MANIFEST_FILENAME).write_text('["a list"]')
    assert RunManifest.load(tmp_path) is None


def test_save_is_atomic_and_leaves_no_temp_file(manifest, tmp_path):
    manifest.save()
    manifest.record("seed", 7)
    manifest.save()
    files = os.listdir(tmp_path)
    assert files == [MANIFEST_FILENAME]
    assert json.loads((tmp_path / MANIFEST_FILENAME).read_text())["seed"] == 7


def test_git_info_never_raises_outside_a_repo(tmp_path):
    info = git_info(repo_root=str(tmp_path))
    assert set(info) == {"sha", "branch", "dirty"}


def test_previous_run_is_carried_into_the_new_manifest(tmp_path):
    first = RunManifest.create(tmp_path, CONFIG, seed=42)
    first.record_step(6, config_hash=hash_json(CONFIG))
    first.save()

    second = RunManifest.create(tmp_path, {**CONFIG, "vae_epochs": 5}, seed=42)
    prior = RunManifest.load(tmp_path)
    second.record("previous_run", {"created_at": prior.data["created_at"], "config_hash": prior.config_hash})
    assert second.data["previous_run"]["config_hash"] != second.config_hash
