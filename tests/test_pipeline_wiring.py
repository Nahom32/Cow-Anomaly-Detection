"""Wiring checks for the pipeline entry points.

These assert which calls the entry points make, not what the models compute. The
orchestrator imports without cv2 / ultralytics (see the laziness test at the
bottom), so the whole file runs on a CPU-only box; the vision-stack smoke check
is the one test that skips when those packages are absent.
"""

import inspect
import subprocess
import sys

import pytest

from scripts.manifest import RunManifest
from scripts.run_full_pipeline import is_step_done, main, step_flat_vae, step_lstm_vae
from scripts.utils.hashing import hash_json


def source_of(func):
    return inspect.getsource(func)


def test_full_pipeline_seeds_before_doing_work():
    src = source_of(main)
    assert "set_seed(" in src
    # seeding must happen before the first pipeline step runs
    assert src.index("set_seed(") < src.index("state = load_state(")


def test_full_pipeline_exposes_seed_and_headless_flags():
    src = source_of(main)
    assert "--seed" in src
    assert "--headless" in src


def test_full_pipeline_writes_a_manifest():
    src = source_of(main)
    assert "RunManifest(" in src
    assert "begin_run(" in src
    assert "manifest.save()" in src


def test_full_pipeline_shares_one_feature_cache_between_vae_stages():
    src = source_of(main)
    assert "open_run_feature_cache" in src
    # both stages must receive the same cache object, not two fresh ones
    assert src.count("feature_cache=feature_cache") == 2


def test_full_pipeline_persists_lstm_history():
    assert "lstm_vae_history.csv" in source_of(step_lstm_vae)


def test_flat_vae_saves_its_plot_instead_of_only_showing_it():
    src = source_of(step_flat_vae)
    assert "save_path=" in src
    assert "headless=" in src


def test_flat_vae_splits_indices_so_the_split_can_be_hashed():
    src = source_of(step_flat_vae)
    assert "all_idx" in src
    assert "train_test_split(" in src
    assert "record_split(" in src


@pytest.mark.parametrize(
    "module",
    ["scripts.train_vae_pipeline", "scripts.train_lstm_vae_pipeline"],
)
def test_standalone_pipelines_seed_and_cache(module):
    mod = __import__(module, fromlist=["main"])
    src = source_of(mod.main)
    assert "set_seed(" in src
    assert "open_run_feature_cache" in src


def test_lstm_pipeline_persists_history():
    import scripts.train_lstm_vae_pipeline as mod

    assert "save_history(" in source_of(mod.main)


def test_standalone_pipelines_write_a_manifest():
    for module in ["scripts.train_vae_pipeline", "scripts.train_lstm_vae_pipeline"]:
        mod = __import__(module, fromlist=["main"])
        assert "RunManifest.create(" in source_of(mod.main)
        assert "manifest.save()" in source_of(mod.main)


def test_yolo_trainers_read_their_settings_from_config():
    from scripts.config import CONFIG
    from scripts.models.train_yolo_n import train_yolo

    src = source_of(train_yolo)
    assert 'CONFIG["yolo"]' in src
    assert "epochs=" not in src, "a hardcoded epoch count would silently win over CONFIG"
    assert "imgsz=" not in src
    assert CONFIG["yolo"]["epochs"] == 150


def test_is_step_done_rejects_a_stale_checkpoint(tmp_path):
    """A checkpoint on disk from a different config must not count as done."""
    config = {"random_seed": 42, "vae_epochs": 100}
    (tmp_path / "flat_vae_model.pth").write_bytes(b"weights")

    manifest = RunManifest.create(tmp_path, config, seed=42)
    assert is_step_done(6, {}, str(tmp_path), config, manifest) is False, (
        "a checkpoint with no manifest entry cannot be verified — retrain rather than trust it"
    )

    manifest.record_step(6, config_hash=hash_json(config))
    assert is_step_done(6, {}, str(tmp_path), config, manifest) is True

    other = RunManifest.create(tmp_path, {**config, "vae_epochs": 200}, seed=42)
    other.record_step(6, config_hash=hash_json({**config, "vae_epochs": 200}))
    assert is_step_done(6, {}, str(tmp_path), config, other) is False


def test_resuming_a_run_keeps_prior_step_hashes(tmp_path):
    """A second run must see the first run's step hashes, not an empty manifest."""
    config = {"random_seed": 42, "vae_epochs": 100}
    (tmp_path / "flat_vae_model.pth").write_bytes(b"weights")

    first = RunManifest.create(tmp_path, config, seed=42)
    first.record_step(6, config_hash=hash_json(config))
    first.save()

    second = RunManifest.load(tmp_path)
    second.begin_run(config, seed=42)
    assert is_step_done(6, {}, str(tmp_path), config, second) is True
    assert "previous_run" not in second.data

    changed = RunManifest.load(tmp_path)
    changed.begin_run({**config, "vae_epochs": 200}, seed=42)
    assert changed.data["previous_run"]["config_hash"] == hash_json(config)
    assert is_step_done(6, {}, str(tmp_path), {**config, "vae_epochs": 200}, changed) is False


def test_is_step_done_detects_retrained_yolo_weights(tmp_path, monkeypatch):
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"v1")
    manifest = RunManifest.create(tmp_path, {"random_seed": 42}, seed=42)
    manifest.record_artifact("yolo_weights", str(weights))
    monkeypatch.setattr("scripts.run_full_pipeline.find_best_pt", lambda: str(weights))
    assert is_step_done(4, {}, str(tmp_path), {"random_seed": 42}, manifest) is True

    weights.write_bytes(b"v2")
    assert is_step_done(4, {}, str(tmp_path), {"random_seed": 42}, manifest) is False


def test_is_step_done_ignores_the_manifest_when_no_config_is_given(tmp_path):
    """Backwards-compatible call form: existence only."""
    (tmp_path / "lstm_vae_model.pth").write_bytes(b"weights")
    assert is_step_done(7, {}, str(tmp_path)) is True


def test_importing_the_orchestrator_does_not_pull_in_the_vision_stack():
    """Keeps the CPU-only promise in AGENTS.md true for the whole pipeline."""
    code = (
        "import sys, scripts.run_full_pipeline as m;"
        "print(sorted(k for k in ('cv2', 'torchvision', 'ultralytics', 'kagglehub') if k in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]", f"eager vision imports: {out.stdout.strip()}"


def test_vision_stack_smoke_check():
    """Skipped without the heavy stack; run in CI where it is installed."""
    pytest.importorskip("cv2", reason="needs opencv")
    pytest.importorskip("ultralytics", reason="needs ultralytics")
    from scripts.models.feature_extractor import extract_cow_features

    assert callable(extract_cow_features)
    assert "input_size" in inspect.signature(extract_cow_features).parameters
