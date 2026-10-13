import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.models.training import EarlyStopping, beta_for_epoch, make_scheduler, run_training_epoch
from scripts.utils.history import empty_history


class LSTMVAE(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128, latent_dim=32, num_layers=1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder_lstm = nn.LSTM(latent_dim, hidden_dim, num_layers, batch_first=True)
        # Ends in `Sigmoid` like the flat VAE (2.4): both models now min-max the
        # features to [0, 1], so both decoders reconstruct within the same range
        # and their reconstruction errors are on one scale.
        self.fc_out = nn.Sequential(nn.Linear(hidden_dim, input_dim), nn.Sigmoid())

    def encode(self, x):
        _, (h_n, _) = self.encoder_lstm(x)
        h = h_n[-1]
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, seq_len):
        z = z.unsqueeze(1).repeat(1, seq_len, 1)
        out, _ = self.decoder_lstm(z)
        out = self.fc_out(out)
        return out

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, x.size(1))
        return recon, mu, logvar


def lstm_vae_loss(recon, x, mu, logvar, beta=1.0):
    """Reconstruction is a per-element MSE (2.1), KL a sum weighted by `beta` (2.5).

    Same fix as the flat VAE's `vae_loss`: `reduction="sum"` summed over
    `batch x seq_len x 256` elements, which made the LSTM-VAE's reported loss scale
    with sequence length and read ~16x larger than the flat VAE's purely
    mechanically (`problems_observed.md:45-63`). KL stays a sum over the latent
    dimensions; `beta` is the explicit, annealed weight on it (2.5), matching the
    flat VAE so the two losses stay comparable.
    """
    recon_loss = F.mse_loss(recon, x, reduction="mean")
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss


def train_lstm_vae(
    vae, train_loader, val_loader, epochs, lr=1e-3, device="cuda",
    beta=1.0, kl_warmup_epochs=0, checkpoint=None,
    lr_scheduler=None, lr_scheduler_factor=0.5, lr_scheduler_patience=5,
    grad_clip=None, early_stopping=None,
):
    vae.to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)
    scheduler = make_scheduler(lr_scheduler, optimizer, lr_scheduler_factor, lr_scheduler_patience)
    history = empty_history()
    # Best-weights tracking (2.7) and the stop decision (2.8) live in one tracker.
    stopper = early_stopping if early_stopping is not None else EarlyStopping(patience=0)

    for epoch in range(1, epochs + 1):
        epoch_beta = beta_for_epoch(epoch, beta, kl_warmup_epochs)
        # Averaged over the batches drawn, not `len(loader.dataset)` (2.2). Both
        # loaders are built with `SubsetRandomSampler`, so `loader.dataset` is the
        # *full* dataset rather than the sampled subset.
        avg_train_loss, avg_train_recon, avg_train_kl = run_training_epoch(
            vae, train_loader, optimizer, device, lstm_vae_loss, epoch_beta, grad_clip
        )

        avg_val_loss, avg_val_recon, avg_val_kl = evaluate_lstm_vae(vae, val_loader, device, epoch_beta)

        history["epoch"].append(epoch)
        history["beta"].append(epoch_beta)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        history["train_loss"].append(avg_train_loss)
        history["train_recon"].append(avg_train_recon)
        history["train_kl"].append(avg_train_kl)
        history["val_loss"].append(avg_val_loss)
        history["val_recon"].append(avg_val_recon)
        history["val_kl"].append(avg_val_kl)
        print(f"Epoch {epoch:3d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        stopper.update(avg_val_loss, epoch, vae)
        if scheduler is not None:
            scheduler.step(avg_val_loss)
        if stopper.should_stop:
            print(f"Early stopping at epoch {epoch}; best was epoch {stopper.best_epoch} (2.8).")
            break

    if checkpoint is not None:
        checkpoint["best_state"] = stopper.best_state
        checkpoint["best_epoch"] = stopper.best_epoch
        checkpoint["best_metric"] = stopper.best
        checkpoint["final_epoch"] = history["epoch"][-1]

    return history


def evaluate_lstm_vae(vae, loader, device, beta=1.0):
    """Deterministic validation pass (2.6), the LSTM twin of `evaluate_vae`.

    `LSTMVAE.forward` reparameterises, so scoring through it made `val_loss` a
    function of the RNG. Reconstructing from the posterior mean and using the
    analytic KL makes the number depend only on the weights.
    """
    vae.eval()
    loss, recon, kl = 0.0, 0.0, 0.0
    with torch.no_grad():
        for batch in loader:
            x = batch.to(device) if not isinstance(batch, (list, tuple)) else batch[0].to(device)
            mu, logvar = vae.encode(x)
            recon_x = vae.decode(mu, seq_len=x.size(1))
            total, recon_loss, kl_loss = lstm_vae_loss(recon_x, x, mu, logvar, beta)
            loss += total.item()
            recon += recon_loss.item()
            kl += kl_loss.item()
    n_batches = len(loader)
    return loss / n_batches, recon / n_batches, kl / n_batches
