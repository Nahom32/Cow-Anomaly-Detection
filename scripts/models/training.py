"""Shared training-loop mechanics for both VAEs.

The two loops must stay interchangeable for a Flat-vs-LSTM comparison, so the
parts that are not model-specific live here rather than being written twice and
drifting: the KL warm-up schedule (2.5). The flat VAE and the LSTM-VAE rebalance
KL against a per-element reconstruction the same way, so they share this value.
"""


def beta_for_epoch(epoch, beta, warmup_epochs):
    """Linearly ramp the KL weight from 0 to `beta` over `warmup_epochs` (2.5).

    `warmup_epochs <= 0` disables annealing and returns `beta` for every epoch.
    `epoch` is 1-based, so the ramp reaches `beta` exactly at the final warm-up
    epoch and stays there. KL is a sum over latent dimensions while recon is a
    per-element mean (2.1), so a plain `beta = 1` over-weights KL; annealing is
    how that imbalance is dialled in deliberately instead of inherited.
    """
    if warmup_epochs <= 0:
        return float(beta)
    return float(beta) * min(1.0, epoch / float(warmup_epochs))
