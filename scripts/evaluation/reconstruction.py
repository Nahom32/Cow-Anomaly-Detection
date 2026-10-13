"""Per-split reconstruction scores (task 1.5).

The protocol the project committed to is train -> val -> threshold selection ->
test: the test split exists to be scored on *once*, with nothing about the
model or the normalizer derived from it. Before 1.5 the pipeline trained on
train, monitored on val, and *reported* val — every number it produced had been
watched during training, so none of them was a held-out result.

These functions are deliberately not a benchmark: no labels, no threshold, no
AUROC (that is Phase 4.6's harness, and ground truth is problem #25). They are
the honest per-split reconstruction error the three-way split makes possible,
and the input the later metrics will be computed from.
"""

import numpy as np
import torch

SCORE_METRIC = "per_sample_mse"


def row_reconstruction_errors(vae, rows, device, batch_size=1024):
    """One MSE per row: the mean squared error between a row and its reconstruction.

    Deterministic on purpose. `VAE.forward` samples z from the posterior, so
    scoring through `forward` would make the reported test number a function of
    the RNG — a rerun could move it without anything about the data changing.
    Scoring `decode(encode(x).mu)` is the fixed-point reconstruction that Phase 2's
    deterministic-eval task (2.6) also uses for validation, so a split score and a
    reported `val_loss` cannot disagree because one sampled `z` and the other did not.

    Per-element mean, matching the training loss's `reduction="mean"` (task 2.1),
    so the number does not scale with the feature dimension.
    """
    rows = torch.as_tensor(rows, dtype=torch.float32)
    if len(rows) == 0:
        return np.empty(0, dtype=np.float64)
    vae.eval()
    errors = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size].to(device)
            mu, _ = vae.encode(batch)
            recon = vae.decode(mu)
            errors.append(((batch - recon) ** 2).mean(dim=1).cpu().numpy())
    return np.concatenate(errors)


def window_reconstruction_errors(vae, windows, device, batch_size=256):
    """One MSE per temporal window, reduced over both time and feature axes.

    Same determinism rule as the flat path: `LSTMVAE.forward` samples z, so the
    mu -> decode route is the only one whose output is reproducible.
    """
    windows = torch.as_tensor(windows, dtype=torch.float32)
    if len(windows) == 0:
        return np.empty(0, dtype=np.float64)
    vae.eval()
    errors = []
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            batch = windows[start : start + batch_size].to(device)
            mu, _ = vae.encode(batch)
            recon = vae.decode(mu, seq_len=batch.shape[1])
            errors.append(((batch - recon) ** 2).mean(dim=(1, 2)).cpu().numpy())
    return np.concatenate(errors)


def summarize(errors):
    """`{n, mean, median, std}` of a split's errors; `{"n": 0}` when empty.

    An empty split is reported, not raised on: with `test_split = 0` the test
    slot legitimately has no rows. The stages raise separately when the config
    *asked* for a test split and got no rows — that is an empty coverage cell,
    which must be loud (see `step_flat_vae`).
    """
    errors = np.asarray(errors)
    if errors.size == 0:
        return {"n": 0}
    return {
        "n": int(errors.size),
        "mean": float(errors.mean()),
        "median": float(np.median(errors)),
        "std": float(errors.std()),
    }


def split_summaries(errors_by_split):
    """Summarize every non-empty split, dropping empty ones."""
    return {name: summarize(errors) for name, errors in errors_by_split.items() if len(errors)}


def reconstruction_score_report(model_name, errors_by_split):
    """The payload written to `<model>_split_scores.json` by each VAE stage."""
    return {
        "metric": SCORE_METRIC,
        "model": model_name,
        "normalizer_fit_on": "train",
        "splits": split_summaries(errors_by_split),
    }
