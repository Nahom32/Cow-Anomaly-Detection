"""Autoencoder novelty detector (4.3.1).

A plain MLP autoencoder; the anomaly score is the squared reconstruction error
`A_AE(z_t) = ||z_t - z_hat_t||^2` (`improvements_on_vad.md:175`). It reuses the
VAE training loop (`scripts.models.training.train_loop`) by presenting the same
`(recon, mu, logvar)` shape with the latent distribution unset and a loss that
carries no KL term — so early stopping, the LR schedule and gradient clipping
cannot drift between the generative rows.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.models.generative.base import TorchDetector, tensor_loader
from scripts.models.training import train_loop


class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """`(recon, None, None)` — loop-shaped, but an AE has no latent distribution."""
        return self.decoder(self.encoder(x)), None, None


def ae_loss(recon_x, x, mu, logvar, beta):
    """Reconstruction MSE with a zero KL, so `train_loop` can drive an AE unchanged.

    `mu`/`logvar` are unused: the AE has no posterior, and the KL the loop
    accumulates is genuinely zero, not dropped. `beta` is accepted only to match
    the `loss_fn(recon, x, mu, logvar, beta)` contract the shared loop uses.
    """
    recon = F.mse_loss(recon_x, x, reduction="mean")
    zero = torch.zeros((), device=recon.device)
    return recon, recon, zero


def evaluate_ae(model, loader, device, beta=0.0):
    """Deterministic validation pass, returning `(loss, recon, 0.0)` like `evaluate_vae`."""
    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
            recon, _, _ = model(x)
            total += F.mse_loss(recon, x, reduction="mean").item()
    n_batches = len(loader)
    return total / n_batches, total / n_batches, 0.0


class AutoencoderDetector(TorchDetector):
    kind = "autoencoder"

    def __init__(
        self, *, hidden_dim=128, latent_dim=32, epochs=50, batch_size=64, lr=1e-3,
        grad_clip=1.0, seed=None, device="cpu", val_fraction=0.1,
    ):
        super().__init__(seed=seed, device=device, val_fraction=val_fraction)
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.grad_clip = grad_clip

    def _build_module(self, input_dim):
        return Autoencoder(input_dim, hidden_dim=self.hidden_dim, latent_dim=self.latent_dim)

    def _train_module(self, train_rows, val_rows):
        train_loader = tensor_loader(train_rows, self.batch_size, shuffle=True)
        val_loader = tensor_loader(val_rows, self.batch_size) if len(val_rows) else None
        train_loop(
            self.module_, train_loader, val_loader,
            epochs=self.epochs, lr=self.lr, device=self.device,
            loss_fn=ae_loss, evaluate_fn=evaluate_ae, beta=0.0,
            grad_clip=self.grad_clip, seed=self.seed,
        )

    def _reconstruct(self, module, scaled):
        with torch.no_grad():
            recon, _, _ = module(torch.as_tensor(scaled, dtype=torch.float32).to(self.device))
        return recon.cpu().numpy()
