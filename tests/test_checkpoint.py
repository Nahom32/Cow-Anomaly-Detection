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
    load_model_state,
    migrate_checkpoint,
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
from scripts.utils.io import atomic_path, atomic_torch_save


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


# ── Validate on load (3.6.1) ──────────────────────────────────────────


def test_migrate_checkpoint_is_a_noop_at_the_current_version():
    cp = build_checkpoint(model_state={}, epoch=1, config={}, seed=0)
    assert migrate_checkpoint(cp) is cp


def test_migrate_checkpoint_runs_a_registered_migration(monkeypatch):
    from scripts import checkpoint as cp

    def upgrade_0_to_1(old):
        return {**old, "format_version": CHECKPOINT_FORMAT_VERSION}

    monkeypatch.setitem(cp._MIGRATIONS, 0, upgrade_0_to_1)
    migrated = migrate_checkpoint({"format_version": 0, "model_state": {}})
    assert migrated["format_version"] == CHECKPOINT_FORMAT_VERSION


def test_migrate_checkpoint_rejects_an_unknown_version():
    with pytest.raises(ValueError, match="no migration registered"):
        migrate_checkpoint({"format_version": 99})


def test_migrate_checkpoint_names_a_legacy_weights_only_file():
    model = torch.nn.Linear(2, 1)
    with pytest.raises(ValueError, match="legacy weights-only"):
        migrate_checkpoint(model.state_dict())


def test_load_rejects_a_legacy_weights_only_file(tmp_path):
    path = tmp_path / "legacy.pt"
    torch.save(torch.nn.Linear(2, 1).state_dict(), path)
    with pytest.raises(ValueError, match="legacy weights-only"):
        load_checkpoint(path)


def test_load_model_state_restores_matching_weights():
    source = torch.nn.Linear(4, 2)
    target = torch.nn.Linear(4, 2)
    with torch.no_grad():
        source.weight.fill_(3.0)
    load_model_state(target, source.state_dict())
    for name, tensor in source.state_dict().items():
        assert torch.equal(target.state_dict()[name], tensor)


def test_load_model_state_rejects_a_shape_mismatch():
    """3.6.1: a checkpoint for another width must fail, not load as a partial no-op."""
    source = torch.nn.Linear(4, 2)
    target = torch.nn.Linear(6, 2)
    with pytest.raises(ValueError, match="shape"):
        load_model_state(target, source.state_dict())


def test_load_model_state_rejects_unexpected_or_missing_keys():
    source = torch.nn.Linear(4, 2)
    target = torch.nn.Sequential(torch.nn.Linear(4, 2), torch.nn.Linear(2, 1))
    with pytest.raises(ValueError, match="does not match the model"):
        load_model_state(target, source.state_dict())


# ── Round-trip integrity and atomicity (3.6.2) ────────────────────────


def _populated_checkpoint():
    """A checkpoint with every required field plus the extras a resume reads."""
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    optimizer.zero_grad()
    model(torch.randn(3, 4)).sum().backward()
    optimizer.step()
    return build_checkpoint(
        model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict(),
        scheduler_state=None,
        epoch=4,
        best_metric=0.25,
        best_metric_epoch=2,
        best_state=model.state_dict(),
        config={"vae_epochs": 5, "nested": {"a": 1}},
        seed=11,
        data_fingerprint="fp",
        normalizer={"kind": "minmax", "min_": [0.0], "max_": [1.0]},
    )


def test_save_load_round_trips_every_field(tmp_path):
    """3.6.2(a): what goes into a checkpoint is what comes out — tensors included."""
    checkpoint = _populated_checkpoint()
    path = tmp_path / "c.pt"
    save_checkpoint(path, checkpoint)
    loaded = load_checkpoint(path)

    for key, tensor in checkpoint["model_state"].items():
        assert torch.equal(loaded["model_state"][key], tensor)
    for key, tensor in checkpoint["best_state"].items():
        assert torch.equal(loaded["best_state"][key], tensor)
    assert set(loaded["optimizer_state"]["state"]) == set(checkpoint["optimizer_state"]["state"])
    assert loaded["epoch"] == checkpoint["epoch"]
    assert loaded["best_metric"] == checkpoint["best_metric"]
    assert loaded["best_metric_epoch"] == checkpoint["best_metric_epoch"]
    assert loaded["seed"] == checkpoint["seed"]
    assert loaded["config"] == checkpoint["config"]
    assert loaded["data_fingerprint"] == checkpoint["data_fingerprint"]
    assert loaded["normalizer"] == checkpoint["normalizer"]


def test_atomic_torch_save_keeps_the_previous_file_on_an_interrupt(tmp_path, monkeypatch):
    """3.6.2(b): a crash mid-write must leave the old file intact and no temp behind.

    A truncated `.pth` still exists on disk, so existence-only step checks would
    report the step done; the temp-then-rename write is what makes the interrupted
    file either the whole old one or the whole new one.
    """
    path = tmp_path / "c.pt"
    atomic_torch_save(path, {"version": 1})
    before = path.read_bytes()

    def interrupted(obj, f, *args, **kwargs):
        with open(f, "wb") as handle:  # a partial payload in the temp file
            handle.write(b"half-written")
        raise RuntimeError("disk full")

    monkeypatch.setattr(torch, "save", interrupted)
    with pytest.raises(RuntimeError):
        atomic_torch_save(path, {"version": 2})

    assert path.read_bytes() == before, "the previous checkpoint must survive"
    assert sorted(os.listdir(tmp_path)) == [os.path.basename(path)], "no temp file may be left"


def test_atomic_write_does_not_create_a_file_it_could_not_finish(tmp_path):
    """A first-time write that dies leaves nothing behind to be mistaken for done."""
    path = tmp_path / "never.pt"
    with pytest.raises(RuntimeError):
        with atomic_path(path) as tmp:
            with open(tmp, "wb") as handle:
                handle.write(b"partial")
            raise RuntimeError("interrupted")

    assert not path.exists()
    assert list(tmp_path.iterdir()) == []


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
