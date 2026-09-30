"""The normalizer must be fitted on train only (task 1.1).

The regression these guard is one line of arithmetic that was silently wrong:
`features.min(axis=0)` / `features.max(axis=0)` over the whole matrix, applied
before the split. Every val sample was then rescaled using its own extremes, so
the val distribution was a function of the val set and a val score could not be
compared with a train score. `1.8` generalises this to mean/std/PCA/covariance.
"""

import inspect
import sys

import numpy as np
import pandas as pd
import pytest
import torch

import scripts.run_full_pipeline as pipeline
from scripts.data.normalize import MinMaxNormalizer, ZScoreNormalizer
from scripts.run_full_pipeline import step_flat_vae
from scripts.train_vae_pipeline import main as train_vae_main
from scripts.utils.history import HISTORY_COLUMNS, empty_history

SEED = 42
N_FEATURES = 60
FEATURE_DIM = 4
VAL_SPLIT = 0.25
# `train_vae_pipeline.main` hardcodes its own 0.2 rather than reading CONFIG (9.1).
STANDALONE_VAL_SPLIT = 0.2


@pytest.fixture
def features():
    """A (60, 4) feature block. Deterministic, so the split is reproducible."""
    rng = np.random.default_rng(0)
    return (rng.random((N_FEATURES, FEATURE_DIM), dtype=np.float32) * 10).astype(np.float32)


def split(features, val_split=VAL_SPLIT, seed=SEED):
    """The same `train_test_split` the pipeline performs, on row indices."""
    from sklearn.model_selection import train_test_split

    idx = np.arange(features.shape[0])
    return train_test_split(idx, test_size=val_split, random_state=seed)


def fit_train_only(features, val_split=VAL_SPLIT, seed=SEED):
    """The pattern both entry points now use: split, then fit."""
    train_idx, val_idx = split(features, val_split, seed)
    normalizer = MinMaxNormalizer().fit(features[train_idx])
    return normalizer, normalizer.transform(features), train_idx, val_idx


def test_transform_before_fit_raises():
    """A scaler that computes its own statistics at transform time is the leak."""
    with pytest.raises(RuntimeError, match="before fit"):
        MinMaxNormalizer().transform(np.zeros((4, 3), dtype=np.float32))


def test_fitting_on_train_only_ignores_the_val_rows(features):
    """Perturbing val rows must not move a single normalisation statistic."""
    train_idx, val_idx = split(features)
    perturbed = features.copy()
    perturbed[val_idx] += 1000.0

    a = MinMaxNormalizer().fit(features[train_idx])
    b = MinMaxNormalizer().fit(perturbed[train_idx])

    assert np.array_equal(a.min_, b.min_)
    assert np.array_equal(a.max_, b.max_)


def test_val_rows_outside_the_train_range_are_not_rescaled_into_range(features):
    """The point of the fix: a novel val sample stays out of [0, 1]."""
    train_idx, val_idx = split(features)
    shifted = features.copy()
    shifted[val_idx] += 1000.0

    normalizer, transformed = fit_train_only(shifted)[0:2]
    assert transformed[val_idx].max() > 1.0, "out-of-range val data must stay out of range"
    # Train rows still fill [0, 1], so the VAE's sigmoid output is not starved.
    assert transformed[train_idx].min() == pytest.approx(0.0, abs=1e-6)
    assert transformed[train_idx].max() == pytest.approx(1.0, abs=1e-6)


def test_a_leaky_fit_would_have_hidden_the_outlier(features):
    """Guards the test above against silently regressing to the old behaviour."""
    train_idx, val_idx = split(features)
    shifted = features.copy()
    shifted[val_idx] += 1000.0

    leaked_min = shifted.min(axis=0)
    leaked_max = shifted.max(axis=0)
    leaked = (shifted - leaked_min) / (leaked_max - leaked_min + 1e-8)

    assert leaked[val_idx].max() == pytest.approx(1.0), "the old fit maps the outlier to 1.0"


def test_fitted_state_equals_the_train_extremes(features):
    normalizer, _, train_idx, _ = fit_train_only(features)
    assert np.array_equal(normalizer.min_, features[train_idx].min(axis=0))
    assert np.array_equal(normalizer.max_, features[train_idx].max(axis=0))


def test_a_constant_train_column_does_not_divide_by_zero():
    """eps keeps a dead feature column (min == max) from becoming inf/nan."""
    block = np.array([[1.0, 0.5], [2.0, 0.5], [3.0, 0.5]], dtype=np.float32)
    transformed = MinMaxNormalizer().fit_transform(block)

    assert np.isfinite(transformed).all()
    assert np.array_equal(transformed[:, 1], np.zeros(3, dtype=np.float32))


def test_feature_dim_mismatch_is_rejected(features):
    normalizer = MinMaxNormalizer().fit(features)
    with pytest.raises(ValueError, match="does not match the fitted dim"):
        normalizer.transform(features[:, :2])


def test_non_2d_input_is_rejected():
    with pytest.raises(ValueError, match="at least 2 dimensions"):
        MinMaxNormalizer().fit(np.zeros(5, dtype=np.float32))


def test_fitting_on_an_empty_split_is_rejected():
    with pytest.raises(ValueError, match="empty split"):
        MinMaxNormalizer().fit(np.zeros((0, 3), dtype=np.float32))


# ── Z-score, the LSTM-VAE path (1.2) ───────────────────────────────────────


@pytest.fixture
def windows():
    """A (12, 4, 3) block of overlapping windows, as the LSTM-VAE sees them."""
    rng = np.random.default_rng(1)
    return (rng.random((12, 4, 3), dtype=np.float32) * 5).astype(np.float32)


def test_zscore_transform_before_fit_raises():
    with pytest.raises(RuntimeError, match="before fit"):
        ZScoreNormalizer().transform(np.zeros((2, 3), dtype=np.float32))


def test_zscore_fitted_state_equals_the_train_window_statistics(windows):
    """Statistics reduce over the window axis too, not just the window axis's first level."""
    train = windows[:8]
    normalizer = ZScoreNormalizer().fit(train)
    flat = train.reshape(-1, train.shape[-1])

    assert np.allclose(normalizer.mean_, flat.mean(axis=0), atol=1e-6)
    assert np.allclose(normalizer.std_, flat.std(axis=0) + 1e-8, atol=1e-6)


def test_zscore_fit_reduces_over_every_non_feature_axis(windows):
    """`(n, seq_len, d)` and the same windows flattened must agree."""
    block = ZScoreNormalizer().fit(windows[:8])
    flat = ZScoreNormalizer().fit(windows[:8].reshape(-1, windows.shape[-1]))

    assert np.array_equal(block.mean_, flat.mean_)
    assert np.array_equal(block.std_, flat.std_)


def test_zscore_transform_reduces_over_every_non_feature_axis(windows):
    normalizer = ZScoreNormalizer().fit(windows[:8])
    block = normalizer.transform(windows)
    flat = normalizer.transform(windows.reshape(-1, windows.shape[-1]))

    assert block.shape == windows.shape
    assert np.allclose(block.reshape(-1, 3), flat, atol=1e-6)


def test_zscore_fitting_on_train_only_ignores_val_windows(windows):
    """What the fit sees is what it uses: the tail cannot reach the statistics.

    The guarantee is a property of *which rows are passed to `fit`*, which the
    call-site tests below check. Here the point is only that `fit` reduces over
    what it was handed and nothing else.
    """
    perturbed_tail = windows.copy()
    perturbed_tail[8:] += 1000.0

    a = ZScoreNormalizer().fit(windows[:8])
    b = ZScoreNormalizer().fit(perturbed_tail[:8])

    assert np.array_equal(a.mean_, b.mean_)
    assert np.array_equal(a.std_, b.std_)


def test_zscore_val_windows_outside_the_train_range_stay_out(windows):
    """A val window far from the train mean must not be pulled back to 0."""
    train = windows[:8]
    shifted = windows.copy()
    shifted[8:] += 1000.0

    transformed = ZScoreNormalizer().fit(train).transform(shifted)
    assert transformed[8:].max() > 100.0, "a shifted val window must stay far from the mean"
    assert np.abs(transformed[:8]).max() < 5.0


def test_zscore_eps_is_folded_into_the_fitted_std(windows):
    """The persisted std must be usable at scoring time as-is (3.5.1)."""
    train = windows[:8]
    normalizer = ZScoreNormalizer().fit(train)

    # `transform` divides by the stored std and adds nothing further, so a
    # checkpoint holding mean_/std_ reproduces these numbers exactly.
    flat = train.reshape(-1, train.shape[-1])
    expected = (flat - normalizer.mean_) / normalizer.std_
    assert np.array_equal(normalizer.transform(train).reshape(-1, 3), expected)


def test_zscore_eps_protects_a_constant_feature_column():
    """A dead column has std 0; `eps` keeps it finite instead of producing inf/nan."""
    block = np.zeros((3, 4, 2), dtype=np.float32)
    block[:, :, 0] = np.arange(4, dtype=np.float32)
    transformed = ZScoreNormalizer().fit_transform(block)

    assert np.isfinite(transformed).all()
    assert np.array_equal(transformed[..., 1], np.zeros((3, 4), dtype=np.float32))


def test_zscore_feature_dim_mismatch_is_rejected(windows):
    normalizer = ZScoreNormalizer().fit(windows)
    with pytest.raises(ValueError, match="does not match the fitted dim"):
        normalizer.transform(windows[:, :, :2])


def test_zscore_fitting_on_an_empty_split_is_rejected():
    with pytest.raises(ValueError, match="empty split"):
        ZScoreNormalizer().fit(np.zeros((0, 4, 3), dtype=np.float32))


def test_entry_points_split_before_fitting():
    """The order of the two statements is the whole bug; assert it in source."""
    for src in (inspect.getsource(step_flat_vae), inspect.getsource(train_vae_main)):
        assert "train_test_split(" in src
        assert "MinMaxNormalizer().fit(" in src
        assert src.index("train_test_split(") < src.index("MinMaxNormalizer().fit("), (
            "the normalizer must be fitted after the split, not before it"
        )


def test_entry_points_do_not_use_the_leaky_helper():
    """`normalize_features` derived min/max from whatever it was given."""
    for src in (inspect.getsource(step_flat_vae), inspect.getsource(train_vae_main)):
        assert "normalize_features(" not in src


# ── The real call site ────────────────────────────────────────────────────
#
# Source inspection is not enough: `.fit(features)` after a split still leaks if
# the wrong rows are passed, and it caught nothing when the fit was moved back
# onto the full matrix. These drive `step_flat_vae` itself with a stubbed
# extractor and a stubbed training loop, then perturb only the val rows and
# assert that nothing the trainer sees has changed. No vision stack involved.


@pytest.fixture
def annotations_csv(tmp_path):
    path = tmp_path / "annotations.csv"
    rows = [f"v{i // 10},{i * 0.04},0.1,0.1,0.5,0.5,0,{i % 3}" for i in range(N_FEATURES)]
    path.write_text("\n".join(rows))
    return str(path)


@pytest.fixture
def annotations_df():
    """`N_FEATURES` rows, so the stubbed extractor and the frame count agree."""
    return pd.DataFrame(
        {
            "video_id": [f"v{i // 10}" for i in range(N_FEATURES)],
            "timestamp": [i * 0.04 for i in range(N_FEATURES)],
            "x1": 0.1, "y1": 0.1, "x2": 0.5, "y2": 0.5,
            "action_id": 0, "target_id": 1,
        }
    )


class _NullHook:
    def remove(self):
        pass


def config(**overrides):
    base = {
        "normal_action_ids": [0, 1, 2],
        "val_split": VAL_SPLIT,
        "random_seed": SEED,
        "vae_epochs": 1,
        "vae_batch_size": 8,
        "vae_lr": 1e-3,
        "vae_hidden_dim": 8,
        "vae_latent_dim": 2,
    }
    base.update(overrides)
    return base


def _fake_train_loop(captured):
    """A `train_vae` that records exactly what the loop would have trained on."""
    def fake_train_vae(vae, train_loader, val_loader, epochs, lr, device):
        captured["train"] = torch.cat([batch[0] for batch in train_loader])
        captured["val"] = torch.cat([batch[0] for batch in val_loader])
        history = empty_history()
        for _ in range(epochs):
            for column in HISTORY_COLUMNS:
                history[column].append(0.0)
        return history

    return fake_train_vae


def run_step_flat_vae(tmp_path, feature_matrix, annotations_csv, **config_overrides):
    """Run step 6 for real, capturing exactly what the training loop receives."""
    captured = {}
    monkey = pytest.MonkeyPatch()
    monkey.setattr(pipeline, "build_feature_dataset", lambda *a, **k: feature_matrix)
    monkey.setattr(pipeline, "train_vae", _fake_train_loop(captured))
    monkey.setattr(pipeline, "plot_history", lambda *a, **k: None)
    try:
        step_flat_vae(
            annotations_csv, str(tmp_path / "frames"), None, None, "cpu",
            str(tmp_path / "out"), config(**config_overrides), headless=True,
        )
    finally:
        monkey.undo()

    saved_min = np.load(tmp_path / "out" / "flat_vae_feature_min.npy")
    saved_max = np.load(tmp_path / "out" / "flat_vae_feature_max.npy")
    return captured, saved_min, saved_max


def perturb_val_rows_only(matrix, val_split=VAL_SPLIT):
    """Blow up the val rows, leaving every train row bit-identical."""
    train_idx, val_idx = split(matrix, val_split=val_split)
    perturbed = matrix.copy()
    perturbed[val_idx] += 1000.0
    return perturbed, train_idx, val_idx


def test_step_flat_vae_normalizer_ignores_val_rows(tmp_path, annotations_csv, features):
    """The persisted min/max must be the train extremes, not the global ones."""
    _, saved_min, saved_max = run_step_flat_vae(tmp_path, features, annotations_csv)
    train_idx, _ = split(features)

    assert np.array_equal(saved_min, features[train_idx].min(axis=0))
    assert np.array_equal(saved_max, features[train_idx].max(axis=0))
    # The leak being removed: the old fit saved the global extremes.
    assert not np.array_equal(saved_min, features.min(axis=0))


def test_step_flat_vae_train_data_is_invariant_to_val_perturbation(
    tmp_path, annotations_csv, features
):
    """A val row that is arbitrarily large must not move a single train tensor."""
    a, _, _ = run_step_flat_vae(tmp_path / "a", features, annotations_csv)
    perturbed, _, _ = perturb_val_rows_only(features)
    b, _, _ = run_step_flat_vae(tmp_path / "b", perturbed, annotations_csv)

    assert torch.equal(a["train"], b["train"]), "val rows leaked into the training tensors"
    assert b["val"].max() > 1.0, "an out-of-range val sample must stay out of range, not be rescaled"


def test_step_flat_vae_saved_scaler_is_val_invariant(tmp_path, annotations_csv, features):
    a_min, a_max = run_step_flat_vae(tmp_path / "a", features, annotations_csv)[1:]
    perturbed, _, _ = perturb_val_rows_only(features)
    b_min, b_max = run_step_flat_vae(tmp_path / "b", perturbed, annotations_csv)[1:]

    assert np.array_equal(a_min, b_min)
    assert np.array_equal(a_max, b_max)


# ── The standalone entry point ────────────────────────────────────────────
#
# `train_vae_pipeline.py` is a second, independent copy of the flat-VAE path and
# had the identical leak. It hardcodes Colab paths (task 9.1) and writes to
# `OUTPUT_DIR = "."`, so the test chdirs into tmp_path and stubs the four
# vision-stack entry points.


def run_standalone_vae(monkeypatch, feature_matrix, annotations_df):
    """Run `train_vae_pipeline.main()`, capturing what its training loop sees."""
    captured = {}
    module = sys.modules["scripts.train_vae_pipeline"]

    monkeypatch.setattr(module, "build_feature_dataset", lambda *a, **k: feature_matrix)
    monkeypatch.setattr(module, "train_vae", _fake_train_loop(captured))
    monkeypatch.setattr(module, "plot_history", lambda *a, **k: None)
    monkeypatch.setattr(module, "create_feature_extractor", lambda *a, **k: (None, _NullHook()))
    monkeypatch.setattr(module, "open_run_feature_cache", lambda *a, **k: None)
    monkeypatch.setattr(module.pd, "read_csv", lambda *a, **k: annotations_df.copy())
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: False)
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    module.main()
    return captured, np.load("feature_min.npy"), np.load("feature_max.npy")


def test_standalone_vae_normalizer_ignores_val_rows(monkeypatch, tmp_path, annotations_df, features):
    """Same assertion as the orchestrator, on the second copy of this code path."""
    monkeypatch.chdir(tmp_path)
    _, saved_min, saved_max = run_standalone_vae(monkeypatch, features, annotations_df)
    train_idx, _ = split(features, val_split=STANDALONE_VAL_SPLIT)

    assert np.array_equal(saved_min, features[train_idx].min(axis=0))
    assert np.array_equal(saved_max, features[train_idx].max(axis=0))
    assert not np.array_equal(saved_min, features.min(axis=0)), "the global fit is the leak"


def test_standalone_vae_train_data_is_val_invariant(monkeypatch, tmp_path, annotations_df, features):
    monkeypatch.chdir(tmp_path)
    perturbed, _, _ = perturb_val_rows_only(features, val_split=STANDALONE_VAL_SPLIT)

    before = run_standalone_vae(monkeypatch, features, annotations_df)
    after = run_standalone_vae(monkeypatch, perturbed, annotations_df)

    assert torch.equal(before[0]["train"], after[0]["train"]), "val rows leaked into the train tensors"
    assert np.array_equal(before[1], after[1]) and np.array_equal(before[2], after[2])
