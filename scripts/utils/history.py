"""Uniform training-history output.

Every model must emit the same columns, otherwise the Phase 4 harness cannot
aggregate across them. The flat VAE already produced train/val recon and KL
separately; the LSTM-VAE discarded them and never wrote a CSV at all.
"""

import pandas as pd

from scripts.utils.io import atomic_write_text

HISTORY_COLUMNS = [
    "epoch",
    # The effective KL weight for the epoch (2.5): with KL annealing this is not
    # `CONFIG["vae_beta"]`, and a loss curve is unreadable without it.
    "beta",
    # The learning rate actually used this epoch (2.8). A ReduceLROnPlateau run
    # that is logged without its LR cannot be told from one where the schedule
    # silently never fired.
    "lr",
    "train_loss",
    "train_recon",
    "train_kl",
    "val_loss",
    "val_recon",
    "val_kl",
]


def empty_history():
    return {column: [] for column in HISTORY_COLUMNS}


def history_to_frame(history):
    """History dict -> DataFrame, with the canonical column order."""
    columns = [c for c in HISTORY_COLUMNS if c in history]
    missing = [c for c in HISTORY_COLUMNS if c not in history]
    if missing:
        raise ValueError(f"history is missing required columns: {missing}")
    return pd.DataFrame({c: history[c] for c in columns})


def save_history(history, filename="training_history.csv"):
    """Write the history to CSV atomically, creating parent directories."""
    atomic_write_text(filename, history_to_frame(history).to_csv(index=False))
    return filename
