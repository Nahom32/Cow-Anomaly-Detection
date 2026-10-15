"""GRU + Mahalanobis novelty detector (4.3.1).

The roster's combined row: a GRU learns temporal dynamics by next-step
prediction (the same objective as the recurrent predictor), and its last hidden
state is used as the representation in which a Mahalanobis distance is measured.
This separates "the temporal representation" from "how tightly normal dynamics
cluster in it" — a static Mahalanobis on raw features (4.2.2) has no dynamics
to misjudge, and a raw predictor error reports a number rather than a
covariance-aware distance.

The Mahalanobis model is fitted on the hidden states of the *training windows
only*, inside `fit`, alongside the GRU — one train scope for both, per 4.1.2.
"""

import torch

from scripts.models.generative.base import SequenceTorchDetector, as_windows, tensor_loader
from scripts.models.generative.recurrent import (
    RecurrentPredictor,
    evaluate_predictor,
    predictor_loss,
)
from scripts.models.non_generative import MahalanobisDetector
from scripts.models.training import train_loop


class GRUMahalanobisDetector(SequenceTorchDetector):
    kind = "gru_mahalanobis"
    min_seq_len = 2

    def __init__(
        self, *, estimator="ledoit_wolf", hidden_dim=128, num_layers=1, epochs=50, batch_size=32,
        lr=1e-3, grad_clip=1.0, seed=None, device="cpu", val_fraction=0.1,
    ):
        super().__init__(seed=seed, device=device, val_fraction=val_fraction)
        self.estimator = estimator
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.grad_clip = grad_clip
        self.detector_ = None

    def _build_module(self, input_dim):
        return RecurrentPredictor(input_dim, hidden_dim=self.hidden_dim, num_layers=self.num_layers, cell="gru")

    def _train_module(self, train_windows, val_windows):
        train_loader = tensor_loader(train_windows, self.batch_size, shuffle=True)
        val_loader = tensor_loader(val_windows, self.batch_size) if len(val_windows) else None
        train_loop(
            self.module_, train_loader, val_loader,
            epochs=self.epochs, lr=self.lr, device=self.device,
            loss_fn=predictor_loss, evaluate_fn=evaluate_predictor, beta=0.0,
            grad_clip=self.grad_clip, seed=self.seed,
        )

    def _fit_impl(self, train_data):
        """Train the GRU, then fit the Mahalanobis model on its train hidden states."""
        super()._fit_impl(train_data)
        scaled = self.normalizer_.transform(as_windows(train_data, self.min_seq_len))
        self.detector_ = MahalanobisDetector(self.estimator, seed=self.seed)
        self.detector_.fit(self._embed(self.module_, scaled))

    def _embed(self, module, scaled):
        with torch.no_grad():
            hidden = module.encode(torch.as_tensor(scaled, dtype=torch.float32).to(self.device))
        return hidden.cpu().numpy()

    def _anomaly_scores(self, module, scaled):
        """Mahalanobis distance of each window's GRU embedding; higher is anomalous (4.1.1)."""
        return self.detector_.score(self._embed(module, scaled))
