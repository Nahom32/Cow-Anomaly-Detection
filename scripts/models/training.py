"""Shared training-loop mechanics for both VAEs.

The two loops must stay interchangeable for a Flat-vs-LSTM comparison, so the
parts that are not model-specific live here rather than being written twice and
drifting: the KL warm-up schedule (2.5), the optimisation pass with gradient
clipping, and best-weights tracking / early stopping (2.8). The flat VAE and the
LSTM-VAE rebalance KL against a per-element reconstruction the same way, and they
must use the same clip-then-step order for their gradients to be comparable.
"""

import torch


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

    @property
    def should_stop(self):
        return self.patience > 0 and self.bad_epochs >= self.patience
