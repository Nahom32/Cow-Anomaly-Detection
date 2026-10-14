import os
import random

import numpy as np
import pytest
import torch

from scripts.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    REQUIRED_FIELDS,
    build_checkpoint,
    checkpoint_format_version,
    epoch_checkpoint_name,
    is_legacy_checkpoint,
    load_checkpoint,
    parse_epoch_checkpoint,
    prune_checkpoint_dir,
    prune_checkpoints,
    record_and_prune_checkpoints,
    restore_rng_state,
    retention_plan,
    rng_state,
    save_checkpoint,
)
from scripts.manifest import RunManifest
from scripts.utils.hashing import sha256_file


def test_rng_state_roundtrip():
    # Ensure clean state
    s0 = rng_state()
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    st = rng_state()
    # Advance all RNGs
    v1 = random.randint(0, 10**9)
    v2 = np.random.rand()
    v3 = torch.rand(1).item()
    # Restore and verify we get same next values
    restore_rng_state(st)
    assert random.randint(0, 10**9) == v1
    assert np.random.rand() == v2
    assert torch.rand(1).item() == v3
    # Restore original to avoid polluting test state
    restore_rng_state(s0)


def test_build_checkpoint_has_required_fields(tmp_path):
    model = torch.nn.Linear(4, 2)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1)
    cp = build_checkpoint(
        model_state=model.state_dict(),
        optimizer_state=opt.state_dict(),
        scheduler_state=sched.state_dict(),
        epoch=5,
        best_metric=0.1,
        best_metric_epoch=3,
        best_state=model.state_dict(),
        config={"seed": 42},
        seed=42,
        data_fingerprint="abc123",
    )
    for f in REQUIRED_FIELDS:
        assert f in cp
    assert cp["format_version"] == CHECKPOINT_FORMAT_VERSION
    assert checkpoint_format_version(cp) == CHECKPOINT_FORMAT_VERSION


def test_save_load_roundtrip(tmp_path):
    model = torch.nn.Linear(4, 2)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1)
    cp = build_checkpoint(
        model_state=model.state_dict(),
        optimizer_state=opt.state_dict(),
        scheduler_state=sched.state_dict(),
        epoch=2,
        best_metric=1.5,
        best_metric_epoch=1,
        best_state=model.state_dict(),
        config={"a": 1, "b": [2, 3]},
        seed=7,
        data_fingerprint="df123",
    )
    p = tmp_path / "ckpt.pt"
    save_checkpoint(p, cp)
    loaded = load_checkpoint(p)
    assert loaded["format_version"] == CHECKPOINT_FORMAT_VERSION
    assert loaded["epoch"] == 2
    assert loaded["seed"] == 7
    assert loaded["config"]["a"] == 1
    assert set(loaded["model_state"].keys()) == set(cp["model_state"].keys())


def test_load_detects_missing_format_version(tmp_path):
    model = torch.nn.Linear(2, 1)
    p = tmp_path / "legacy.pt"
    torch.save(model.state_dict(), p)
    with pytest.raises(ValueError):
        load_checkpoint(p)


def test_is_legacy_checkpoint(tmp_path):
    model = torch.nn.Linear(2, 1)
    legacy = model.state_dict()
    assert is_legacy_checkpoint(legacy)
    cp = build_checkpoint(
        model_state=model.state_dict(),
        optimizer_state=None,
        scheduler_state=None,
        epoch=1,
        best_metric=None,
        best_metric_epoch=None,
        best_state=None,
        config={},
        seed=0,
    )
    assert not is_legacy_checkpoint(cp)


# ── Retention (3.2.2) ─────────────────────────────────────────────────


def _write(path, data=b"x"):
    path.write_bytes(data)
    return str(path)


def test_epoch_checkpoint_name_roundtrips():
    name = epoch_checkpoint_name("flat_vae", 7)
    assert name == "flat_vae_epoch_0007.pth"
    assert parse_epoch_checkpoint(name) == 7


def test_parse_epoch_checkpoint_ignores_roles_and_foreign_files():
    # best/final carry no epoch marker, so retention can never place them on the
    # timeline and therefore can never delete them.
    assert parse_epoch_checkpoint("flat_vae_best.pth") is None
    assert parse_epoch_checkpoint("flat_vae_model.pth") is None
    assert parse_epoch_checkpoint("notes.txt") is None
    assert parse_epoch_checkpoint("flat_vae_epoch_abcd.pth") is None


def test_retention_keeps_best_final_and_the_last_n_epochs(tmp_path):
    epochs = [_write(tmp_path / epoch_checkpoint_name("m", e)) for e in range(1, 11)]
    best = _write(tmp_path / "m_best.pth")
    final = _write(tmp_path / "m_model.pth")

    kept, removed = retention_plan(epochs + [best, final], keep_last=3, keep=[best, final])

    assert {os.path.basename(p) for p in kept} == {
        "m_best.pth", "m_model.pth",
        "m_epoch_0008.pth", "m_epoch_0009.pth", "m_epoch_0010.pth",
    }
    assert {os.path.basename(p) for p in removed} == {
        epoch_checkpoint_name("m", e) for e in range(1, 8)
    }


def test_retention_keeps_every_epoch_when_fewer_than_the_window(tmp_path):
    epochs = [_write(tmp_path / epoch_checkpoint_name("m", e)) for e in (1, 2)]
    kept, removed = retention_plan(epochs, keep_last=3)
    assert removed == []
    assert len(kept) == 2


def test_retention_leaves_unrecognised_files_alone(tmp_path):
    # A prune must never delete a file it cannot place on the epoch timeline:
    # only the recognised epoch checkpoint is eligible, `m_notes.pth` is not.
    epoch = _write(tmp_path / epoch_checkpoint_name("m", 1))
    other = _write(tmp_path / "m_notes.pth")
    kept, removed = retention_plan([epoch, other], keep_last=0)
    assert removed == [epoch]
    assert other in kept


def test_retention_rejects_a_negative_window():
    with pytest.raises(ValueError, match="non-negative"):
        retention_plan([], keep_last=-1)


def test_prune_checkpoints_deletes_only_the_unretained(tmp_path):
    epochs = [_write(tmp_path / epoch_checkpoint_name("m", e)) for e in range(1, 6)]
    best = _write(tmp_path / "m_best.pth")
    final = _write(tmp_path / "m_model.pth")

    prune_checkpoints(epochs + [best, final], keep_last=2, keep=[best, final])

    assert set(os.listdir(tmp_path)) == {
        "m_best.pth", "m_model.pth", "m_epoch_0004.pth", "m_epoch_0005.pth",
    }


def test_prune_checkpoint_dir_is_prefix_scoped(tmp_path):
    # The flat and LSTM-VAE checkpoints share an output directory; one model's
    # retention policy must not touch the other's files.
    _write(tmp_path / "flat_vae_epoch_0001.pth")
    _write(tmp_path / "flat_vae_epoch_0002.pth")
    _write(tmp_path / "flat_vae_model.pth")
    _write(tmp_path / "lstm_vae_epoch_0001.pth")
    _write(tmp_path / "lstm_vae_epoch_0002.pth")

    prune_checkpoint_dir(str(tmp_path), "flat_vae", keep_last=1,
                         keep=[str(tmp_path / "flat_vae_model.pth")])

    remaining = set(os.listdir(tmp_path))
    assert "flat_vae_epoch_0001.pth" not in remaining
    assert {"flat_vae_epoch_0002.pth", "flat_vae_model.pth"} <= remaining
    assert {"lstm_vae_epoch_0001.pth", "lstm_vae_epoch_0002.pth"} <= remaining


def test_record_and_prune_records_hashes_and_prunes(tmp_path):
    final = _write(tmp_path / "m_model.pth", b"final")
    best = _write(tmp_path / "m_best.pth", b"best")
    for e in range(1, 6):
        _write(tmp_path / epoch_checkpoint_name("m", e))
    manifest = RunManifest.create(tmp_path, {"random_seed": 1}, seed=1)

    record_and_prune_checkpoints(
        manifest=manifest, directory=str(tmp_path), prefix="m",
        keep_last=2, final=final, best=best, step=6,
    )

    assert manifest.checkpoint("m_final")["sha256"] == sha256_file(final)
    assert manifest.checkpoint("m_final")["role"] == "final"
    assert manifest.checkpoint("m_best")["step"] == 6
    remaining = set(os.listdir(tmp_path))
    assert {"m_epoch_0004.pth", "m_epoch_0005.pth", "m_model.pth", "m_best.pth"} <= remaining
    assert "m_epoch_0001.pth" not in remaining


def test_record_and_prune_skips_a_missing_best(tmp_path):
    final = _write(tmp_path / "m_model.pth")
    manifest = RunManifest.create(tmp_path, {"random_seed": 1}, seed=1)

    record_and_prune_checkpoints(
        manifest=manifest, directory=str(tmp_path), prefix="m",
        keep_last=3, final=final, best=None, step=6,
    )

    assert manifest.checkpoint("m_best") is None
    assert manifest.checkpoint("m_final") is not None
