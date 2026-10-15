"""Recurrent next-step predictors as novelty detectors (4.3.1).

A GRU — and its LSTM twin, the architecture comparison `improvements_on_vad.md`
asks for at `:230` — models feature dynamics: given a window
`(z_{t-k}, ..., z_t)` it forecasts `z_{t+1}`, and the anomaly score is that
forecast's error `A(t) = ||z_{t+1} - z_hat_{t+1}||^2` (`:212-226`). This tests
whether predicting the next state is enough without a generative latent model.

Window contract: the last timestep of each window is the prediction target and
the preceding steps are the input, so a provider hands a `(n_windows, k+1, d)`
block and at least two steps are required. Training teacher-forces every
position (`z_{t+1}` from `z_0..z_t`); scoring uses only the window's final-step
forecast, which is the one the score formula names.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from scripts.models.generative.base import SequenceTorchDetector, as_windows
from scripts.models.training import train_loop

_CELLS = ("gru", "lstm")


class RecurrentPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=1, cell="gru"):
        super().__init__()
        rnn_cls = nn.GRU if cell == "gru" else nn.LSTM
        self.rnn = rnn_cls(input_dim, hidden_dim, num_layers, batch_first=True)
        # Sigmoid bounds the forecast to the min-max range the features live in,
        # matching both VAE decoders so the errors are on one scale (2.4).
        self.head = nn.Sequential(nn.Linear(hidden_dim, input_dim), nn.Sigmoid())

    def forward(self, x):
        """`(forecast, None, None)`; `forecast[:, t]` predicts `x[:, t+1]`."""
        hidden, _ = self.rnn(x)
        return self.head(hidden), None, None


def predictor_loss(forecast, x, mu, logvar, beta):
    """MSE of each position's forecast for the next step; no KL for a predictor."""
    error = F.mse_loss(forecast[:, :-1], x[:, 1:], reduction="mean")
    zero = torch.zeros((), device=error.device)
    return error, error, zero


def evaluate_predictor(model, loader, device, beta=0.0):
    """Deterministic validation pass, `(loss, loss, 0.0)` like the VAE evaluators."""
    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
            forecast, _, _ = model(x)
            total += F.mse_loss(forecast[:, :-1], x[:, 1:], reduction="mean").item()
    n_batches = len(loader)
    return total / n_batches, total / n_batches, 0.0


class RecurrentPredictorDetector(SequenceTorchDetector):
    min_seq_len = 2

    def __init__(
        self, *, cell="gru", hidden_dim=128, num_layers=1, epochs=50, batch_size=32,
        lr=1e-3, grad_clip=1.0, seed=None, device="cpu", val_fraction=0.1,
    ):
        if cell not in _CELLS:
            raise ValueError(f"predictor cell must be one of {_CELLS}, got {cell!r}")
        super().__init__(seed=seed, device=device, val_fraction=val_fraction)
        self.cell = cell
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.grad_clip = grad_clip

    def _build_module(self, input_dim):
        return RecurrentPredictor(
            input_dim, hidden_dim=self.hidden_dim, num_layers=self.num_layers, cell=self.cell
        )

    def _train_module(self, train_windows, val_windows):
        train_loader = _loader(train_windows, self.batch_size, shuffle=True)
        val_loader = _loader(val_windows, self.batch_size) if len(val_windows) else None
        train_loop(
            self.module_, train_loader, val_loader,
            epochs=self.epochs, lr=self.lr, device=self.device,
            loss_fn=predictor_loss, evaluate_fn=evaluate_predictor, beta=0.0,
            grad_clip=self.grad_clip, seed=self.seed,
        )

    def _reconstruct(self, module, scaled):
        """The full forecast tensor (same shape as the window); scoring uses the last step."""
        with torch.no_grad():
            forecast, _, _ = module(torch.as_tensor(scaled, dtype=torch.float32).to(self.device))
        return forecast.cpu().numpy()

    def score(self, data):
        """`||z_{t+1} - z_hat_{t+1}||^2` for the window's last step; higher is anomalous (4.1.1)."""
        self._require_fitted()
        scaled = self.normalizer_.transform(as_windows(data, self.min_seq_len))
        with torch.no_grad():
            forecast, _, _ = self.module_(torch.as_tensor(scaled, dtype=torch.float32).to(self.device))
        # `forecast[:, -2]` is the forecast for the last step `x[:, -1]`.
        predicted_last = forecast[:, -2].cpu().numpy()
        return np.sum((scaled[:, -1, :] - predicted_last) ** 2, axis=1)


def _loader(windows, batch_size, shuffle=False):
    return DataLoader(
        TensorDataset(torch.as_tensor(windows, dtype=torch.float32)), batch_size=batch_size, shuffle=shuffle
    )
