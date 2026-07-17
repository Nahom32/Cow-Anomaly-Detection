import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.fc_out = nn.Linear(hidden_dim, input_dim)

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


def lstm_vae_loss(recon, x, mu, logvar, reduction="sum"):
    recon_loss = F.mse_loss(recon, x, reduction=reduction)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + kl_loss
    return total_loss, recon_loss, kl_loss


def train_lstm_vae(vae, train_loader, val_loader, epochs, lr=1e-3, device="cuda"):
    vae.to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)
    history = {"epoch": [], "train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        vae.train()
        total_loss = 0.0
        for batch in train_loader:
            x = batch.to(device) if not isinstance(batch, (list, tuple)) else batch[0].to(device)
            optimizer.zero_grad()
            recon, mu, logvar = vae(x)
            loss, _, _ = lstm_vae_loss(recon, x, mu, logvar, reduction="sum")
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader.dataset)

        vae.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch.to(device) if not isinstance(batch, (list, tuple)) else batch[0].to(device)
                recon, mu, logvar = vae(x)
                loss, _, _ = lstm_vae_loss(recon, x, mu, logvar, reduction="sum")
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader.dataset)

        history["epoch"].append(epoch)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        print(f"Epoch {epoch:3d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    return history
