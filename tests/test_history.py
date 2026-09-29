import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from scripts.models.lstm_vae import LSTMVAE, train_lstm_vae
from scripts.models.vae import VAE, train_vae
from scripts.utils.history import HISTORY_COLUMNS, history_to_frame, save_history

EPOCHS = 2
BATCH = 8


def flat_loaders(dim=16, n=32):
    x = torch.randn(n, dim)
    loader = DataLoader(TensorDataset(x), batch_size=BATCH)
    return loader, loader


def seq_loaders(dim=16, n=24, seq=4):
    x = torch.randn(n, seq, dim)
    loader = DataLoader(x, batch_size=BATCH)
    return loader, loader


@pytest.fixture
def flat_history():
    train_loader, val_loader = flat_loaders()
    return train_vae(VAE(16), train_loader, val_loader, epochs=EPOCHS, device="cpu")


@pytest.fixture
def lstm_history():
    train_loader, val_loader = seq_loaders()
    return train_lstm_vae(
        LSTMVAE(input_dim=16), train_loader, val_loader, epochs=EPOCHS, device="cpu"
    )


def test_both_models_emit_the_same_columns(flat_history, lstm_history):
    assert list(flat_history.keys()) == HISTORY_COLUMNS
    assert list(lstm_history.keys()) == HISTORY_COLUMNS


def test_history_has_one_row_per_epoch(flat_history, lstm_history):
    assert flat_history["epoch"] == list(range(1, EPOCHS + 1))
    assert lstm_history["epoch"] == list(range(1, EPOCHS + 1))


def test_lstm_history_keeps_the_recon_kl_split(lstm_history):
    """The old loop did `loss, _, _ = ...` and lost these entirely."""
    for split in ["train", "val"]:
        for component in ["loss", "recon", "kl"]:
            values = lstm_history[f"{split}_{component}"]
            assert len(values) == EPOCHS
            assert all(isinstance(v, float) for v in values)


def test_loss_equals_recon_plus_kl(lstm_history, flat_history):
    for history in (lstm_history, flat_history):
        for split in ["train", "val"]:
            for total, recon, kl in zip(
                history[f"{split}_loss"],
                history[f"{split}_recon"],
                history[f"{split}_kl"],
            ):
                assert total == pytest.approx(recon + kl, rel=1e-5)


def test_save_history_roundtrips(flat_history, tmp_path):
    path = save_history(flat_history, str(tmp_path / "sub" / "flat_vae_history.csv"))
    frame = pd.read_csv(path)
    assert frame.columns.tolist() == HISTORY_COLUMNS
    assert len(frame) == EPOCHS
    assert frame["epoch"].tolist() == flat_history["epoch"]


def test_save_history_creates_parent_directories(lstm_history, tmp_path):
    path = save_history(lstm_history, str(tmp_path / "a" / "b" / "lstm_vae_history.csv"))
    assert pd.read_csv(path).shape == (EPOCHS, len(HISTORY_COLUMNS))


def test_save_history_leaves_no_temp_files(flat_history, tmp_path):
    save_history(flat_history, str(tmp_path / "flat_vae_history.csv"))
    assert [f for f in tmp_path.iterdir() if f.name.endswith(".tmp")] == []


def test_history_to_frame_rejects_incomplete_history():
    with pytest.raises(ValueError, match="missing required columns"):
        history_to_frame({"epoch": [1], "train_loss": [1.0]})


def test_history_to_frame_uses_canonical_order(flat_history):
    shuffled = {k: flat_history[k] for k in reversed(HISTORY_COLUMNS)}
    assert history_to_frame(shuffled).columns.tolist() == HISTORY_COLUMNS
