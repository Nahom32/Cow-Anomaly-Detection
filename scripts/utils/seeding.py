"""Global seeding for every entry point in the project.

`CONFIG["random_seed"]` was declared in `scripts/config.py` but never applied, so no
run in this repo was reproducible. Every `__main__` block now calls `set_seed()`
before touching a random number generator.
"""

import os
import random

import numpy as np
import torch

DEFAULT_SEED = 42


def set_seed(seed=DEFAULT_SEED, deterministic=False):
    """Seed python, numpy and torch from a single integer.

    Args:
        seed: the seed to apply.
        deterministic: when True, also force deterministic cuDNN kernels and
            `torch.use_deterministic_algorithms`. This slows some GPU ops down
            considerably, so it is opt-in rather than the default.

    Returns:
        The seed that was applied, so callers can log it.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except (AttributeError, RuntimeError):
            pass

    return seed


def seed_worker(worker_id):
    """`worker_init_fn` for DataLoaders with `num_workers > 0`.

    Without this every worker inherits the same numpy seed, so the per-worker
    numpy streams are identical and dataset augmentation repeats across workers.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
