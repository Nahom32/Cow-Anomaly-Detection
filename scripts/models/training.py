"""Shared training-loop mechanics for both VAEs.

The two loops must stay interchangeable for a Flat-vs-LSTM comparison, so the
parts that are not model-specific live here rather than being written twice and
drifting: the KL warm-up schedule (2.5), the optimisation pass with gradient
clipping, and best-weights tracking / early stopping (2.8). The flat VAE and the
LSTM-VAE rebalance KL against a per-element reconstruction the same way, and they
must use the same clip-then-step order for their gradients to be comparable.

`train_loop` is the one loop both models run (3.3.1); the model-specific pieces —
its loss and its deterministic evaluator — are passed in. `resume_training` is
the one way a loop resumes (3.3.2), so a flat-VAE run and an LSTM-VAE run cannot
diverge in what "resume" means.
"""

import os

import torch

from scripts.checkpoint import (
    build_checkpoint,
    clear_epoch_checkpoints,
    epoch_checkpoint_name,
    load_checkpoint,
    prune_checkpoint_dir,
    restore_rng_state,
    save_checkpoint,
)
from scripts.utils.history import empty_history
from scripts.utils.seeding import set_seed


def beta_for_epoch(epoch, beta, warmup_epochs):
    """Linearly ramp the KL weight from 0 to `beta` over `warmup_epochs` (2.5).

    `warmup_epochs <= 0` disables annealing and returns `beta` for every epoch.
    `epoch` is 1-based, so the ramp reaches `beta` exactly at the final warm-up
    epoch and stays there. KL is a sum over latent dimensions while recon is a
    per-element mean (2.1), so a plain `beta = 1` over-weights KL; annealing is
    how that imbalance is dialled in deliberately instead of inherited.
    """
    if warmup_epochs <= 0:
        return float(beta)
    return float(beta) * min(1.0, epoch / float(warmup_epochs))


def run_training_epoch(vae, loader, optimizer, device, loss_fn, beta, grad_clip=None):
    """One optimisation pass, shared by both VAEs (2.8).

    `loss_fn(recon_x, x, mu, logvar, beta)` is `vae_loss` or `lstm_vae_loss`;
    passing it in keeps this module free of a circular import while giving both
    models the same clip-then-step order, so their gradients are bounded alike.
    Returns per-batch means so the reported loss does not scale with batch count.
    """
    vae.train()
    total = recon = kl = 0.0
    for batch in loader:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(device)
        optimizer.zero_grad()
        recon_x, mu, logvar = vae(x)
        loss, recon_loss, kl_loss = loss_fn(recon_x, x, mu, logvar, beta)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(vae.parameters(), grad_clip)
        optimizer.step()
        total += loss.item()
        recon += recon_loss.item()
        kl += kl_loss.item()
    n_batches = len(loader)
    return total / n_batches, recon / n_batches, kl / n_batches


def count_parameters(model):
    """Trainable parameter count (2.11).

    The flat VAE is 111,424 parameters and the LSTM-VAE 321,856 — a 2.89x gap. A
    Flat-vs-LSTM comparison that does not report this is comparing two model sizes
    and calling the difference "temporal modelling", which is what the research
    question exists to rule out.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def match_parameter_budget(build_model, target_parameters, max_width=4096):
    """The model whose trainable size is closest to `target_parameters` (2.11).

    `build_model(width)` must return a fresh model at the given width. Parameter
    count is monotonic in width for the architectures here, so this widens until
    it reaches the target and keeps whichever of the last two widths is closer.
    Returns `(width, model, n_parameters)`.

    This is what turns "the LSTM-VAE is 2.9x the flat VAE" from a caveat into a
    controlled comparison (problem #13): the temporal model is sized down to the
    baseline's capacity, so a difference is not a difference in model size.
    """
    best = None
    for width in range(1, max_width + 1):
        model = build_model(width)
        n_parameters = count_parameters(model)
        if best is None or abs(n_parameters - target_parameters) < abs(best[2] - target_parameters):
            best = (width, model, n_parameters)
        if n_parameters >= target_parameters:
            break
    return best


def make_scheduler(name, optimizer, factor=0.5, patience=5):
    """Build the configured LR scheduler, or `None` (2.8).

    Only `reduce_on_plateau` is supported; it is stepped with the monitored val
    loss, which is why the loops call `scheduler.step(monitor)`.
    """
    if not name:
        return None
    if name == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=factor, patience=patience
        )
    raise ValueError(f"unknown lr scheduler: {name!r}")


class EarlyStopping:
    """Track the best-val weights and decide when to stop (2.8).

    Only meaningful because of 2.6 and 2.7: a deterministic val pass makes "no
    improvement for N epochs" signal rather than noise, and the best weights are
    already retained rather than discarded. `patience <= 0` records the best
    epoch but never stops, which is what the plain 2.7 behaviour needs.
    """

    def __init__(self, patience=0):
        self.patience = patience
        self.best = None
        self.best_epoch = None
        self.best_state = None
        self.bad_epochs = 0

    def update(self, metric, epoch, model):
        if self.best is None or metric < self.best:
            self.best = metric
            self.best_epoch = epoch
            self.best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            self.bad_epochs = 0
            return True
        self.bad_epochs += 1
        return False

    def state_dict(self):
        """The counter half of the tracker (3.3.2). `best_state` is stored separately.

        `bad_epochs` must survive a resume, otherwise early stopping restarts its
        patience count and a plateau already detected is forgotten.
        """
        return {"best": self.best, "best_epoch": self.best_epoch, "bad_epochs": self.bad_epochs}

    def load_state_dict(self, state):
        self.best = state.get("best")
        self.best_epoch = state.get("best_epoch")
        self.bad_epochs = state.get("bad_epochs", 0)

    @property
    def should_stop(self):
        return self.patience > 0 and self.bad_epochs >= self.patience


def _history_from_checkpoint(stored):
    """Rebuild a history dict from its checkpointed form, dropping unknown columns."""
    history = empty_history()
    if isinstance(stored, dict):
        for column in history:
            values = stored.get(column)
            if values:
                history[column] = list(values)
    return history


def resume_training(path, model, optimizer, scheduler, stopper, device="cpu"):
    """Restore a training loop's full state from a checkpoint (3.3.1, 3.3.2).

    Returns `(history, start_epoch)` where `start_epoch` is the first epoch still
    to run. Model, optimizer, scheduler, the early-stopping tracker, the history
    so far and the RNG state are all restored, which is what lets a resumed run
    reproduce the uninterrupted loss curve (3.3.3): the next epoch sees exactly
    the weights, the optimiser momentum and the random draws it would have seen
    had the process never died.
    """
    checkpoint = load_checkpoint(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    if checkpoint.get("optimizer_state") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler is not None and checkpoint.get("scheduler_state") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    if stopper is not None:
        stopper.load_state_dict(checkpoint.get("early_stopping") or {})
        if stopper.best_state is None:
            stopper.best_state = checkpoint.get("best_state")
    restore_rng_state(checkpoint.get("rng_state"))
    return _history_from_checkpoint(checkpoint.get("history")), int(checkpoint["epoch"]) + 1


def _save_epoch_checkpoint(directory, prefix, epoch, *, model, optimizer, scheduler,
                           stopper, history, config, seed, data_fingerprint, keep_last):
    """Write one epoch checkpoint in the unified schema, then enforce retention (3.2.2)."""
    path = os.path.join(directory, epoch_checkpoint_name(prefix, epoch))
    save_checkpoint(
        path,
        build_checkpoint(
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            scheduler_state=scheduler.state_dict() if scheduler is not None else None,
            epoch=epoch,
            best_metric=stopper.best,
            best_metric_epoch=stopper.best_epoch,
            best_state=stopper.best_state,
            config=config or {},
            seed=seed,
            data_fingerprint=data_fingerprint,
            # Extra fields a resume needs that the required schema does not name.
            history=history,
            early_stopping=stopper.state_dict(),
        ),
    )
    if keep_last is not None:
        prune_checkpoint_dir(directory, prefix, keep_last=keep_last)
    return path


def train_loop(
    model, train_loader, val_loader, *,
    epochs, lr, device, loss_fn, evaluate_fn,
    beta=1.0, kl_warmup_epochs=0,
    checkpoint=None,
    lr_scheduler=None, lr_scheduler_factor=0.5, lr_scheduler_patience=5,
    grad_clip=None, early_stopping=None,
    seed=None, config=None, data_fingerprint=None,
    checkpoint_dir=None, checkpoint_prefix=None, keep_last=None,
    resume_from=None,
):
    """The one training loop both VAEs run (3.3.1).

    Everything model-specific is injected: `loss_fn(recon, x, mu, logvar, beta)`
    and `evaluate_fn(model, val_loader, device, beta)`. The loop owns the
    optimizer, scheduler, best-weights tracker, history, per-epoch checkpointing
    and resume, so the flat VAE and the LSTM-VAE cannot drift in how they anneal
    KL, clip gradients, stop early, checkpoint, or resume.

    On `resume_from`, the checkpoint's state is restored and training continues at
    the next epoch; the caller decides *which* checkpoint to resume from (see
    `scripts.checkpoint.latest_epoch_checkpoint`). A fresh run clears any leftover
    epoch checkpoints so a shorter re-run cannot be pruned down to a stale run's
    higher-numbered epochs.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = make_scheduler(lr_scheduler, optimizer, lr_scheduler_factor, lr_scheduler_patience)
    history = empty_history()
    # Best-weights tracking (2.7) and the stop decision (2.8) live in one tracker:
    # the epoch early stopping would keep is the epoch whose weights are saved.
    stopper = early_stopping if early_stopping is not None else EarlyStopping(patience=0)

    start_epoch = 1
    if resume_from is not None:
        history, start_epoch = resume_training(
            resume_from, model, optimizer, scheduler, stopper, device=device
        )
        print(
            f"Resumed from {resume_from} at epoch {start_epoch} "
            f"(best val so far {stopper.best} at epoch {stopper.best_epoch}) (3.3.2)."
        )
    else:
        if checkpoint_dir and checkpoint_prefix:
            clear_epoch_checkpoints(checkpoint_dir, checkpoint_prefix)
        if seed is not None:
            set_seed(seed)

    for epoch in range(start_epoch, epochs + 1):
        epoch_beta = beta_for_epoch(epoch, beta, kl_warmup_epochs)
        train_loss, train_recon, train_kl = run_training_epoch(
            model, train_loader, optimizer, device, loss_fn, epoch_beta, grad_clip
        )

        history["epoch"].append(epoch)
        history["beta"].append(epoch_beta)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        history["train_loss"].append(train_loss)
        history["train_recon"].append(train_recon)
        history["train_kl"].append(train_kl)

        if val_loader:
            val_loss, val_recon, val_kl = evaluate_fn(model, val_loader, device, epoch_beta)
            history["val_loss"].append(val_loss)
            history["val_recon"].append(val_recon)
            history["val_kl"].append(val_kl)
            print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            monitor = val_loss
        else:
            print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f}")
            monitor = train_loss

        stopper.update(monitor, epoch, model)
        if scheduler is not None:
            scheduler.step(monitor)

        if checkpoint_dir and checkpoint_prefix:
            _save_epoch_checkpoint(
                checkpoint_dir, checkpoint_prefix, epoch,
                model=model, optimizer=optimizer, scheduler=scheduler, stopper=stopper,
                history=history, config=config, seed=seed,
                data_fingerprint=data_fingerprint, keep_last=keep_last,
            )

        if stopper.should_stop:
            print(f"Early stopping at epoch {epoch}; best was epoch {stopper.best_epoch} (2.8).")
            break

    if checkpoint is not None:
        checkpoint["best_state"] = stopper.best_state
        checkpoint["best_epoch"] = stopper.best_epoch
        checkpoint["best_metric"] = stopper.best
        checkpoint["final_epoch"] = history["epoch"][-1] if history["epoch"] else 0

    return history
