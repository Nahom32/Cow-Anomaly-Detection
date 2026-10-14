"""Resume continues an interrupted run and reproduces it (3.3.2, 3.3.3).

The failure this guards is not a crash: `--from-step` used to restart the epoch
counter at 1, so a 150-epoch run interrupted at 140 looked finished after ten
more epochs while its loss curve began a second time. The tests below pin the two
properties that make resume honest — it continues at the next epoch, and the
curve it produces is the curve the uninterrupted run would have produced.
"""

import os
import pickle

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from scripts.checkpoint import (
    build_checkpoint,
    clear_epoch_checkpoints,
    epoch_checkpoint_name,
    latest_epoch_checkpoint,
    list_epoch_checkpoints,
    load_checkpoint,
    save_checkpoint,
)
from scripts.models.training import EarlyStopping, make_scheduler, resume_training
from scripts.models.vae import VAE, train_vae
from scripts.utils.history import HISTORY_COLUMNS, empty_history
from scripts.utils.seeding import set_seed

SEED = 7
DIM = 8
EPOCHS = 5


def loaders(n=32):
    x = torch.randn(n, DIM, generator=torch.Generator().manual_seed(0))
    return DataLoader(TensorDataset(x), batch_size=8), DataLoader(TensorDataset(x), batch_size=8)


def fresh_vae():
    set_seed(SEED)
    return VAE(DIM, hidden_dim=16, latent_dim=4)


def run(model, out_dir, epochs=EPOCHS, keep_last=2, **kwargs):
    return train_vae(
        model, *loaders(), epochs=epochs, lr=1e-2, device="cpu", checkpoint={},
        seed=SEED, checkpoint_dir=str(out_dir), checkpoint_prefix="flat_vae",
        keep_last=keep_last, **kwargs,
    )


def test_resume_reproduces_the_uninterrupted_loss_curve(tmp_path):
    """3.3.3 / 3.6.2(c): two runs, one stopped and resumed, must be identical.

    The uninterrupted run trains all `EPOCHS`. The interrupted one trains two,
    dies, and resumes from the checkpoint; if the resume restored the weights but
    not the optimiser momentum, the RNG or the shuffle order, the tail would drift.
    """
    uninterrupted = run(fresh_vae(), tmp_path / "full", keep_last=10)

    first = run(fresh_vae(), tmp_path / "seg", epochs=2, keep_last=10)
    assert first["epoch"] == [1, 2]

    resume_from = latest_epoch_checkpoint(str(tmp_path / "seg"), "flat_vae")
    assert resume_from is not None
    resumed = run(
        VAE(DIM, hidden_dim=16, latent_dim=4), tmp_path / "seg",
        keep_last=10, resume_from=resume_from,
    )

    # The resumed history is rebuilt from the checkpoint, so it spans the whole run.
    assert resumed["epoch"] == list(range(1, EPOCHS + 1))
    for column in HISTORY_COLUMNS:
        assert len(resumed[column]) == EPOCHS
        assert all(
            abs(a - b) < 1e-6 for a, b in zip(resumed[column], uninterrupted[column])
        ), f"resumed {column} diverged from the uninterrupted curve"


def test_train_loop_writes_one_epoch_checkpoint_per_epoch_and_prunes(tmp_path):
    run(fresh_vae(), tmp_path / "out", keep_last=2)
    assert [epoch for epoch, _ in list_epoch_checkpoints(str(tmp_path / "out"), "flat_vae")] == [4, 5]


def test_a_fresh_run_clears_stale_epoch_checkpoints(tmp_path):
    """A new run's epochs are lower-numbered than the old run's survivors, so without
    the clear the prune would keep the stale high epochs and delete the live ones."""
    out = tmp_path / "out"
    out.mkdir()
    (out / epoch_checkpoint_name("flat_vae", 9)).write_bytes(b"stale")

    run(fresh_vae(), out, keep_last=2)

    assert [epoch for epoch, _ in list_epoch_checkpoints(str(out), "flat_vae")] == [4, 5]


def test_latest_epoch_checkpoint_skips_a_checkpoint_from_another_config(tmp_path):
    out = tmp_path / "out"
    out.mkdir()

    def write(epoch, config):
        save_checkpoint(
            out / epoch_checkpoint_name("flat_vae", epoch),
            build_checkpoint(
                model_state=VAE(DIM, hidden_dim=16, latent_dim=4).state_dict(),
                epoch=epoch, config=config, seed=SEED,
            ),
        )

    write(1, {"vae_epochs": 5})
    write(2, {"vae_epochs": 99})

    assert latest_epoch_checkpoint(str(out), "flat_vae") == str(out / epoch_checkpoint_name("flat_vae", 2))
    assert latest_epoch_checkpoint(str(out), "flat_vae", config={"vae_epochs": 5}) == \
        str(out / epoch_checkpoint_name("flat_vae", 1))
    assert latest_epoch_checkpoint(str(out), "flat_vae", config={"vae_epochs": 7}) is None


def test_clear_epoch_checkpoints_removes_only_its_prefix(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / epoch_checkpoint_name("flat_vae", 1)).write_bytes(b"a")
    (out / epoch_checkpoint_name("lstm_vae", 1)).write_bytes(b"b")

    removed = clear_epoch_checkpoints(str(out), "flat_vae")

    assert sorted(os.listdir(out)) == [epoch_checkpoint_name("lstm_vae", 1)]
    assert removed == [str(out / epoch_checkpoint_name("flat_vae", 1))]


def test_resume_training_restores_optimizer_early_stopping_and_next_epoch(tmp_path):
    model = VAE(DIM, hidden_dim=16, latent_dim=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0)
    scheduler = make_scheduler("reduce_on_plateau", optimizer, factor=0.5, patience=0)
    stopper = EarlyStopping(patience=3)

    # Create optimiser momentum and force one LR reduction, so both have state to lose.
    optimizer.zero_grad()
    recon, _, _ = model(torch.randn(2, DIM))
    recon.mean().backward()
    optimizer.step()
    scheduler.step(1.0)
    scheduler.step(2.0)
    assert optimizer.param_groups[0]["lr"] == 0.5
    stopper.update(1.0, 1, model)
    stopper.update(2.0, 2, model)
    assert stopper.bad_epochs == 1

    history = empty_history()
    for epoch in range(1, 4):
        for column in HISTORY_COLUMNS:
            history[column].append(float(epoch))
    path = tmp_path / epoch_checkpoint_name("flat_vae", 3)
    save_checkpoint(
        path,
        build_checkpoint(
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            scheduler_state=scheduler.state_dict(),
            epoch=3, best_metric=stopper.best, best_metric_epoch=stopper.best_epoch,
            best_state=stopper.best_state, config={}, seed=SEED,
            history=history, early_stopping=stopper.state_dict(),
        ),
    )

    resumed_model = VAE(DIM, hidden_dim=16, latent_dim=4)
    resumed_optimizer = torch.optim.Adam(resumed_model.parameters(), lr=1.0)
    resumed_scheduler = make_scheduler("reduce_on_plateau", resumed_model and resumed_model.parameters() and resumed_model.parameters() and resumed_optimizer)
    resumed_stopper = EarlyStopping(patience=3)

    restored_history, start_epoch = resume_training(
        path, resumed_model, resumed_optimizer, resumed_scheduler, resumed_stopper
    )

    assert start_epoch == 4, "resume must continue at the next epoch, not restart at 1"
    assert restored_history["epoch"] == [1, 2, 3]
    assert resumed_optimizer.param_groups[0]["lr"] == 0.5, "the reduced LR must survive"
    assert resumed_optimizer.state_dict()["state"].keys() == optimizer.state_dict()["state"].keys()
    assert resumed_stopper.bad_epochs == 1
    assert resumed_stopper.best == 1.0
    for name, tensor in model.state_dict().items():
        assert torch.equal(resumed_model.state_dict()[name], tensor)


def test_early_stopping_state_round_trips():
    model = VAE(DIM, hidden_dim=16, latent_dim=4)
    stopper = EarlyStopping(patience=3)
    stopper.update(1.0, 1, model)
    stopper.update(2.0, 2, model)

    restored = EarlyStopping(patience=3)
    restored.load_state_dict(stopper.state_dict())

    assert restored.best == 1.0
    assert restored.best_epoch == 1
    assert restored.bad_epochs == 1
    assert not restored.should_stop


def test_load_checkpoint_rejects_a_corrupted_file(tmp_path):
    """A truncated checkpoint is exactly the interrupted file we resume from, so it
    must be rejected loudly rather than loaded as an empty state dict."""
    path = tmp_path / "flat_vae_epoch_0001.pth"
    path.write_bytes(b"not a torch checkpoint")
    with pytest.raises((RuntimeError, EOFError, pickle.UnpicklingError)):
        load_checkpoint(path)
