"""VAE novelty detector (4.3.1).

Wraps the existing flat `scripts.models.vae.VAE` and its `train_vae` loop as a
harness detector. The anomaly score is the reconstruction term
`A_VAE,rec(z_t) = ||z_t - z_hat_t||^2` (`improvements_on_vad.md:187`), computed
from the posterior mean rather than a sample so a rerun cannot move the score
(2.6 / 4.4.3). The document notes a latent-likelihood score as a separate,
deliberately-not-mixed-in comparison (`:190`), so only reconstruction is scored.
"""

import torch

from scripts.models.generative.base import TorchDetector, tensor_loader
from scripts.models.vae import VAE, train_vae


class VAEDetector(TorchDetector):
    kind = "vae"

    def __init__(
        self, *, hidden_dim=128, latent_dim=32, epochs=50, batch_size=64, lr=1e-3,
        beta=1.0, kl_warmup_epochs=0, grad_clip=1.0, seed=None, device="cpu", val_fraction=0.1,
    ):
        super().__init__(seed=seed, device=device, val_fraction=val_fraction)
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.beta = beta
        self.kl_warmup_epochs = kl_warmup_epochs
        self.grad_clip = grad_clip

    def _build_module(self, input_dim):
        return VAE(input_dim, hidden_dim=self.hidden_dim, latent_dim=self.latent_dim)

    def _train_module(self, train_rows, val_rows):
        train_loader = tensor_loader(train_rows, self.batch_size, shuffle=True)
        val_loader = tensor_loader(val_rows, self.batch_size) if len(val_rows) else None
        train_vae(
            self.module_, train_loader, val_loader,
            epochs=self.epochs, lr=self.lr, device=self.device,
            beta=self.beta, kl_warmup_epochs=self.kl_warmup_epochs,
            grad_clip=self.grad_clip, seed=self.seed,
        )

    def _reconstruct(self, module, scaled):
        with torch.no_grad():
            features = torch.as_tensor(scaled, dtype=torch.float32).to(self.device)
            mu, _ = module.encode(features)
            recon = module.decode(mu)
        return recon.cpu().numpy()
