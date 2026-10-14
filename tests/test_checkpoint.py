import json
import os
import random

import numpy as np
import torch

from scripts import checkpoint
from scripts.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    REQUIRED_FIELDS,
    build_checkpoint,
    checkpoint_format_version,
    data_fingerprint,
    is_legacy_checkpoint,
    load_checkpoint,
    restore_rng_state,
    rng_state,
    save_checkpoint,
)


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
    try:
        load_checkpoint(p)
        assert False, "should have failed"
    except ValueError:
        pass


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
