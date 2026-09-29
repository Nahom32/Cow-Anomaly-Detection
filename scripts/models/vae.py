import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from scripts.utils.history import empty_history
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


def vae_loss(recon_x, x, mu, logvar, reduction="sum"):
    recon_loss = F.mse_loss(recon_x, x, reduction=reduction)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + kl_loss
    return total_loss, recon_loss, kl_loss


def train_vae(vae, train_loader, val_loader, epochs, lr=1e-3, device="cuda"):
    vae.to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)

    history = empty_history()

    for epoch in range(1, epochs + 1):
        vae.train()
        train_loss, train_recon, train_kl = 0.0, 0.0, 0.0
        for batch in train_loader:
            x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = vae(x)
            loss, recon_loss, kl_loss = vae_loss(recon, x, mu, logvar, reduction="sum")
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_recon += recon_loss.item()
            train_kl += kl_loss.item()

        n_batches = len(train_loader)
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss / n_batches)
        history["train_recon"].append(train_recon / n_batches)
        history["train_kl"].append(train_kl / n_batches)

        if val_loader:
            vae.eval()
            val_loss, val_recon, val_kl = 0.0, 0.0, 0.0
            with torch.no_grad():
                for batch in val_loader:
                    x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
                    recon, mu, logvar = vae(x)
                    loss, recon_loss, kl_loss = vae_loss(recon, x, mu, logvar, reduction="sum")
                    val_loss += loss.item()
                    val_recon += recon_loss.item()
                    val_kl += kl_loss.item()
            n_val = len(val_loader)
            history["val_loss"].append(val_loss / n_val)
            history["val_recon"].append(val_recon / n_val)
            history["val_kl"].append(val_kl / n_val)
            print(f"Epoch {epoch:3d} | Train Loss: {history['train_loss'][-1]:.4f} | Val Loss: {history['val_loss'][-1]:.4f}")
        else:
            print(f"Epoch {epoch:3d} | Train Loss: {history['train_loss'][-1]:.4f}")

    return history


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
