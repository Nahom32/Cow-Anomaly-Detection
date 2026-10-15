"""LSTM-VAE novelty detector (4.3.1).

Wraps the existing `scripts.models.lstm_vae.LSTMVAE` and `train_lstm_vae` as a
harness detector. The score is the window reconstruction term computed from the
posterior mean, `||z_{t-k:t} - z_hat_{t-k:t}||^2`
(`improvements_on_vad.md:234`), so it is deterministic (2.6) and comparable with
the flat VAE's score on the same features. This is the roster's combined
temporal-and-generative row.
"""

import torch

from scripts.models.generative.base import SequenceTorchDetector, tensor_loader
from scripts.models.lstm_vae import LSTMVAE, train_lstm_vae


class LSTMVAEDetector(SequenceTorchDetector):
    kind = "lstm_vae"

    def __init__(
        self, *, hidden_dim=128, latent_dim=32, num_layers=1, epochs=50, batch_size=32,
        lr=1e-3, beta=1.0, kl_warmup_epochs=0, grad_clip=1.0, seed=None, device="cpu", val_fraction=0.1,
    ):
        super().__init__(seed=seed, device=device, val_fraction=val_fraction)
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.beta = beta
        self.kl_warmup_epochs = kl_warmup_epochs
        self.grad_clip = grad_clip

    def _build_module(self, input_dim):
        return LSTMVAE(
            input_dim, hidden_dim=self.hidden_dim, latent_dim=self.latent_dim, num_layers=self.num_layers
        )

    def _train_module(self, train_windows, val_windows):
        train_loader = tensor_loader(train_windows, self.batch_size, shuffle=True)
        val_loader = tensor_loader(val_windows, self.batch_size) if len(val_windows) else None
        train_lstm_vae(
            self.module_, train_loader, val_loader,
            epochs=self.epochs, lr=self.lr, device=self.device,
            beta=self.beta, kl_warmup_epochs=self.kl_warmup_epochs,
            grad_clip=self.grad_clip, seed=self.seed,
        )

    def _reconstruct(self, module, scaled):
        with torch.no_grad():
            windows = torch.as_tensor(scaled, dtype=torch.float32).to(self.device)
            mu, _ = module.encode(windows)
            recon = module.decode(mu, seq_len=windows.size(1))
        return recon.cpu().numpy()
