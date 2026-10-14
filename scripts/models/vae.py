import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.models.training import train_loop
from scripts.utils.plotting import setup_matplotlib, show_or_close


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """Reconstruction is a per-element MSE (2.1), KL a sum weighted by `beta` (2.5).

    `reduction="sum"` made the reported loss scale with `batch x dim`, so the flat
    VAE (`batch x 256`) and the LSTM-VAE (`batch x seq x 256`) reported units 16x
    apart — the whole source of the "~10x loss difference" (`problems_observed.md:45-63`).

    KL stays a sum over the latent dimensions while recon is a per-element mean, so
    its weight is inflated ~`latent_dim / feature_dim` relative to a per-element KL.
    `beta` makes that weight explicit and `train_vae` anneals it (2.5), instead of
    the effective constant 1.0 it silently used before.
    """
    recon_loss = F.mse_loss(recon_x, x, reduction="mean")
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss


def train_vae(
    vae, train_loader, val_loader, epochs, lr=1e-3, device="cuda",
    beta=1.0, kl_warmup_epochs=0, checkpoint=None,
    lr_scheduler=None, lr_scheduler_factor=0.5, lr_scheduler_patience=5,
    grad_clip=None, early_stopping=None,
    seed=None, config=None, data_fingerprint=None,
    checkpoint_dir=None, checkpoint_prefix="flat_vae", keep_last=None,
    resume_from=None,
):
    """Flat-VAE training: the shared loop with this model's loss and evaluator (3.3.1).

    The loop itself lives in `training.train_loop`; this wrapper exists only to
    bind `vae_loss`/`evaluate_vae`, so the flat VAE and the LSTM-VAE cannot drift
    in anneal order, grad clipping, early stopping, checkpointing or resume.
    """
    return train_loop(
        vae, train_loader, val_loader,
        epochs=epochs, lr=lr, device=device,
        loss_fn=vae_loss, evaluate_fn=evaluate_vae,
        beta=beta, kl_warmup_epochs=kl_warmup_epochs, checkpoint=checkpoint,
        lr_scheduler=lr_scheduler, lr_scheduler_factor=lr_scheduler_factor,
        lr_scheduler_patience=lr_scheduler_patience, grad_clip=grad_clip,
        early_stopping=early_stopping, seed=seed, config=config,
        data_fingerprint=data_fingerprint, checkpoint_dir=checkpoint_dir,
        checkpoint_prefix=checkpoint_prefix, keep_last=keep_last, resume_from=resume_from,
    )


def evaluate_vae(vae, loader, device, beta=1.0):
    """Deterministic validation pass (2.6).

    `VAE.forward` reparameterises, so scoring through it made `val_loss` a
    function of the RNG: the Monte-Carlo noise is visible as a 3.07-unit
    oscillation across the last 10 epochs of the real `flat_vae_history.csv`.
    Reconstructing from the posterior mean and using the analytic KL makes the
    number depend only on the weights, matching the fixed-point scoring in
    `scripts/evaluation/reconstruction.py`.
    """
    vae.eval()
    loss, recon, kl = 0.0, 0.0, 0.0
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
            mu, logvar = vae.encode(x)
            recon_x = vae.decode(mu)
            total, recon_loss, kl_loss = vae_loss(recon_x, x, mu, logvar, beta)
            loss += total.item()
            recon += recon_loss.item()
            kl += kl_loss.item()
    n_batches = len(loader)
    return loss / n_batches, recon / n_batches, kl / n_batches


def plot_history(history, save_path=None, headless=None):
    """Plot loss curves.

    Never blocks: the figure is written to `save_path` when given, shown without
    blocking when the backend is interactive, and always closed.
    """
    setup_matplotlib(headless)

    epochs = history["epoch"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train")
    if history.get("val_loss"):
        axes[0].plot(epochs, history["val_loss"], label="Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Total Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_recon"], label="Train Recon")
    if history.get("val_recon"):
        axes[1].plot(epochs, history["val_recon"], label="Val Recon")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Reconstruction Loss")
    axes[1].legend()

    axes[2].plot(epochs, history["train_kl"], label="Train KL")
    if history.get("val_kl"):
        axes[2].plot(epochs, history["val_kl"], label="Val KL")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("KL Divergence")
    axes[2].legend()

    fig.tight_layout()
    return show_or_close(fig, save_path=save_path)
