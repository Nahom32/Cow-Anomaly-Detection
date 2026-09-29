import os
import subprocess
import sys

import numpy as np
import torch

from scripts.utils.seeding import DEFAULT_SEED, derive_seed, rng_for, seed_worker, set_seed

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def draw():
    import random

    return (
        random.random(),
        float(np.random.rand()),
        torch.randn(4).tolist(),
    )


def test_set_seed_makes_every_generator_repeat():
    set_seed(42)
    first = draw()
    set_seed(42)
    assert draw() == first


def test_different_seeds_diverge():
    set_seed(42)
    first = draw()
    set_seed(43)
    assert draw() != first


def test_set_seed_returns_and_sets_pythonhashseed():
    assert set_seed(7) == 7
    assert os.environ["PYTHONHASHSEED"] == "7"


def test_default_seed_matches_config():
    from scripts.config import CONFIG

    assert DEFAULT_SEED == CONFIG["random_seed"]


def test_derive_seed_is_stable_and_order_sensitive():
    assert derive_seed(42, "v1", 0, 0) == derive_seed(42, "v1", 0, 0)
    assert derive_seed(42, "v1", 0, 0) != derive_seed(42, "v1", 0, 1)
    assert derive_seed(42, "v1", 0) != derive_seed(42, "0v1", 0)


def test_derive_seed_survives_hash_randomisation():
    """python's str hash is salted per process; a hash()-based seed would not be."""
    code = (
        "import sys; sys.path.insert(0, %r);"
        "from scripts.utils.seeding import derive_seed;"
        "print(derive_seed(42, 'v1', 0, 0))" % REPO_ROOT
    )
    outputs = set()
    for salt in ["0", "1", "12345"]:
        env = dict(os.environ, PYTHONHASHSEED=salt)
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True
        )
        outputs.add(result.stdout.strip())
    assert len(outputs) == 1, f"seed changed with PYTHONHASHSEED: {outputs}"


def test_rng_for_is_reproducible_and_independent_of_call_order():
    a = rng_for(42, "v1", 0, 0).standard_normal(8)
    _ = rng_for(42, "v1", 9, 9).standard_normal(100)  # unrelated draws in between
    b = rng_for(42, "v1", 0, 0).standard_normal(8)
    assert np.array_equal(a, b)


def test_seed_worker_runs_without_error():
    set_seed(0)
    torch.manual_seed(1234)
    seed_worker(0)


def test_deterministic_flag_enables_cudnn_determinism():
    previous = torch.backends.cudnn.deterministic
    try:
        set_seed(42, deterministic=True)
        assert torch.backends.cudnn.deterministic is True
    finally:
        torch.backends.cudnn.deterministic = previous
