"""The normalizer must be fitted on train only (task 1.1).

The regression these guard is one line of arithmetic that was silently wrong:
`features.min(axis=0)` / `features.max(axis=0)` over the whole matrix, applied
before the split. Every val sample was then rescaled using its own extremes, so
the val distribution was a function of the val set and a val score could not be
compared with a train score. `1.8` generalises this to mean/std/PCA/covariance.
"""

import inspect
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import scripts.run_full_pipeline as pipeline
from scripts.config import CONFIG
from scripts.data.feature_cache import FeatureCache, cache_signature, row_feature_key
from scripts.data.normalize import MinMaxNormalizer, ZScoreNormalizer
from scripts.data.splits import resolve_indices, resolve_split
from scripts.run_full_pipeline import step_flat_vae, step_lstm_vae
from scripts.train_vae_pipeline import main as train_vae_main
from scripts.utils.history import HISTORY_COLUMNS, empty_history

SEED = 42
N_FEATURES = 60
FEATURE_DIM = 4
VAL_SPLIT = 0.25
# The test slot, opt-in per test; with six videos this holds exactly one out.
TEST_SPLIT = 0.1
# Both entry points read their fractions from CONFIG now (1.7), so the expected
# row sets must be drawn with the same numbers rather than with local copies.
STANDALONE_VAL_SPLIT = CONFIG["val_split"]
STANDALONE_TEST_SPLIT = CONFIG["test_split"]


def sorted_rows(tensor):
    """The rows of `tensor` in a canonical order.

    The train loader shuffles, so the row order a captured loader yields is a
    function of the RNG, not of the data. Comparing two runs' raw tensors tests the
    shuffle rather than the split; sorting by every column compares the row
    *multiset*, which is what "these rows are the training set" actually means.
    """
    rows = tensor.numpy()
    return rows[np.lexsort(rows.T[::-1])]


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
        assert "resolve_indices(" in src
        assert "MinMaxNormalizer().fit(" in src
        assert src.index("resolve_indices(") < src.index("MinMaxNormalizer().fit("), (
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
        "test_split": 0.0,
        "split_group_key": "video_id",
        "random_seed": SEED,
        "vae_epochs": 1,
        "vae_batch_size": 8,
        "vae_lr": 1e-3,
        "vae_hidden_dim": 8,
        "vae_latent_dim": 2,
        "vae_beta": 1.0,
        "vae_kl_warmup_epochs": 0,
        # Optimisation stabilisers (2.8): disabled in these tests so the stubbed
        # loops stay deterministic; they are exercised directly in test_history.
        "vae_lr_scheduler": None,
        "vae_lr_scheduler_factor": 0.5,
        "vae_lr_scheduler_patience": 5,
        "vae_grad_clip": None,
        "vae_early_stopping_patience": 0,
        # `step_lstm_vae` reads these; small enough to run in milliseconds.
        "seq_len": 4,
        "seq_stride": 2,
        "lstm_vae_epochs": 1,
        "lstm_vae_batch_size": 8,
        "lstm_vae_lr": 1e-3,
        "lstm_vae_hidden_dim": 8,
        "lstm_vae_latent_dim": 2,
        "lstm_vae_num_layers": 1,
        "lstm_vae_beta": 1.0,
        "lstm_vae_kl_warmup_epochs": 0,
        "lstm_vae_lr_scheduler": None,
        "lstm_vae_lr_scheduler_factor": 0.5,
        "lstm_vae_lr_scheduler_patience": 5,
        "lstm_vae_grad_clip": None,
        "lstm_vae_early_stopping_patience": 0,
        "lstm_vae_match_flat_parameters": False,
        # `step_flat_vae`/`step_lstm_vae` read the retention window (3.2.2).
        "checkpoint_keep_last": 3,
    }
    base.update(overrides)
    return base


def grouped_split(features, video_ids, output_dir, val_split=VAL_SPLIT, test_split=0.0,
                  seed=SEED, group_key="video_id"):
    """The split indices the pipeline's canonical split selects.

    Derived from the same `resolve_split`/`resolve_indices` the stage calls, rather
    than from an independent row shuffle, so these tests assert what the pipeline
    does instead of re-describing it. The 1.1 invariant they protect is unchanged:
    the fit sees train rows only.
    """
    df = pd.DataFrame({group_key: video_ids})
    assignment = resolve_split(
        output_dir, df, val_split=val_split, test_split=test_split,
        random_seed=seed, group_key=group_key,
    )
    return assignment, resolve_indices(assignment, video_ids)


def _fake_train_loop(captured):
    """A `train_vae` that records exactly what the loop would have trained on."""
    def fake_train_vae(vae, train_loader, val_loader, epochs, lr, device, **kwargs):
        captured["train"] = torch.cat([batch[0] for batch in train_loader])
        captured["val"] = torch.cat([batch[0] for batch in val_loader])
        history = empty_history()
        for _ in range(epochs):
            for column in HISTORY_COLUMNS:
                history[column].append(0.0)
        checkpoint = kwargs.get("checkpoint")
        if checkpoint is not None:
            checkpoint["best_state"] = {k: v.clone() for k, v in vae.state_dict().items()}
            checkpoint["best_epoch"] = 1
            checkpoint["best_metric"] = 0.0
            checkpoint["final_epoch"] = epochs
        return history

    return fake_train_vae


def video_ids_for(n_rows=N_FEATURES):
    """The video each stubbed feature row belongs to, matching `annotations_csv`."""
    return [f"v{i // 10}" for i in range(n_rows)]


def run_step_flat_vae(tmp_path, feature_matrix, annotations_csv, video_ids=None, **config_overrides):
    """Run step 6 for real, capturing exactly what the training loop receives."""
    if video_ids is None:
        video_ids = video_ids_for(feature_matrix.shape[0])
    captured = {}
    monkey = pytest.MonkeyPatch()
    # The real signature returns `(features, video_ids)`; the ids are what make the
    # split groupable, so the stub supplies them rather than hiding the coupling.
    monkey.setattr(
        pipeline, "build_feature_dataset",
        lambda *a, **k: (feature_matrix, np.array(video_ids, dtype=object)),
    )
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


def perturb_rows_only(matrix, split_name, val_split=VAL_SPLIT, test_split=0.0, seed=SEED,
                      output_dir=None):
    """Blow up the rows of one split, leaving every train row bit-identical."""
    video_ids = video_ids_for(matrix.shape[0])
    output_dir = output_dir or str(Path(tempfile.mkdtemp()) / "split")
    _, indices = grouped_split(
        matrix, video_ids, output_dir, val_split=val_split, test_split=test_split, seed=seed
    )
    held_out = indices[split_name]
    perturbed = matrix.copy()
    perturbed[held_out] += 1000.0
    return perturbed, indices["train"], held_out


def perturb_val_rows_only(matrix, val_split=VAL_SPLIT, test_split=0.0, seed=SEED, output_dir=None):
    """Blow up the val rows, leaving every train row bit-identical."""
    return perturb_rows_only(matrix, "val", val_split=val_split, test_split=test_split,
                             seed=seed, output_dir=output_dir)


def perturb_test_rows_only(matrix, val_split=VAL_SPLIT, test_split=TEST_SPLIT, seed=SEED,
                           output_dir=None):
    """Blow up the test rows: the slot no fit may ever read (1.5, 1.8)."""
    return perturb_rows_only(matrix, "test", val_split=val_split, test_split=test_split,
                             seed=seed, output_dir=output_dir)


def perturb_held_out_rows(matrix, val_split=VAL_SPLIT, test_split=0.0, seed=SEED,
                          output_dir=None):
    """Blow up everything outside the train split, in one perturbation."""
    video_ids = video_ids_for(matrix.shape[0])
    output_dir = output_dir or str(Path(tempfile.mkdtemp()) / "split")
    _, indices = grouped_split(
        matrix, video_ids, output_dir, val_split=val_split, test_split=test_split, seed=seed
    )
    held_out = np.concatenate([indices["val"], indices["test"]])
    perturbed = matrix.copy()
    perturbed[held_out] += 1000.0
    return perturbed, indices["train"], held_out


def test_step_flat_vae_normalizer_ignores_val_rows(tmp_path, annotations_csv, features):
    """The persisted min/max must be the train extremes, not the global ones."""
    _, saved_min, saved_max = run_step_flat_vae(tmp_path, features, annotations_csv)
    _, indices = grouped_split(features, video_ids_for(), str(tmp_path / "expected"))
    train_idx = indices["train"]

    assert np.array_equal(saved_min, features[train_idx].min(axis=0))
    assert np.array_equal(saved_max, features[train_idx].max(axis=0))


def test_step_flat_vae_train_data_is_invariant_to_val_perturbation(
    tmp_path, annotations_csv, features
):
    """A val row that is arbitrarily large must not move a single train tensor."""
    a, _, _ = run_step_flat_vae(tmp_path / "a", features, annotations_csv)
    perturbed, _, _ = perturb_val_rows_only(features)
    b, _, _ = run_step_flat_vae(tmp_path / "b", perturbed, annotations_csv)

    assert np.array_equal(sorted_rows(a["train"]), sorted_rows(b["train"])), \
        "val rows leaked into the training tensors"
    assert b["val"].max() > 1.0, "an out-of-range val sample must stay out of range, not be rescaled"


def test_step_flat_vae_saved_scaler_is_val_invariant(tmp_path, annotations_csv, features):
    a_min, a_max = run_step_flat_vae(tmp_path / "a", features, annotations_csv)[1:]
    perturbed, _, _ = perturb_val_rows_only(features)
    b_min, b_max = run_step_flat_vae(tmp_path / "b", perturbed, annotations_csv)[1:]

    assert np.array_equal(a_min, b_min)
    assert np.array_equal(a_max, b_max)


def test_step_flat_vae_keeps_one_video_entirely_on_one_side(tmp_path, annotations_csv):
    """1.3, asserted end to end rather than on the split helper.

    A row-level split put consecutive frames of one cow in both train and val, which
    is problem #5 and the reason the flat VAE scored better than it should have.

    Every row of a video carries the same value, so the set of distinct values a
    loader receives names exactly the videos on that side. Min-max scaling is
    injective, so distinct videos stay distinct after `transform` and no value
    decoding is needed — an earlier version encoded row indices in a column and read
    them back, which silently broke on the float32 rounding of the rescale.
    """
    video_ids = video_ids_for(N_FEATURES)
    value_of = {video_id: i for i, video_id in enumerate(video_ids)}
    per_video = np.array([value_of[v] for v in video_ids], dtype=np.float32)
    features = np.repeat(per_video[:, None], FEATURE_DIM, axis=1)

    captured, saved_min, saved_max = run_step_flat_vae(tmp_path, features, annotations_csv)

    def videos_in(tensor):
        # Invert the min-max affine map with the persisted scaler rather than
        # assuming the normalized values are the originals.
        original = tensor.numpy()[:, 0] * (saved_max[0] - saved_min[0]) + saved_min[0]
        return {video_ids[int(round(value))] for value in original}

    train_videos = videos_in(captured["train"])
    val_videos = videos_in(captured["val"])
    assert train_videos and val_videos, "both sides must be non-empty"
    assert not train_videos & val_videos, (
        f"videos on both sides of the split: {sorted(train_videos & val_videos)}"
    )


def test_step_flat_vae_writes_best_and_final_checkpoints(tmp_path, annotations_csv, features):
    """2.7: both the final-epoch and best-val checkpoints reach disk."""
    run_step_flat_vae(tmp_path, features, annotations_csv)
    out = tmp_path / "out"
    assert (out / "flat_vae_model.pth").is_file()
    assert (out / "flat_vae_best.pth").is_file()


def test_step_flat_vae_embeds_the_train_scaler_in_its_checkpoints(tmp_path, annotations_csv, features):
    """3.5.1: the final and best checkpoints carry the fitted min/max themselves."""
    from scripts.checkpoint import load_checkpoint
    from scripts.data.normalize import load_normalizer

    _, saved_min, saved_max = run_step_flat_vae(tmp_path, features, annotations_csv)
    out = tmp_path / "out"
    for name in ("flat_vae_model.pth", "flat_vae_best.pth"):
        embedded = load_normalizer(load_checkpoint(out / name)["normalizer"])
        np.testing.assert_array_equal(embedded.min_, saved_min)
        np.testing.assert_array_equal(embedded.max_, saved_max)


def test_step_flat_vae_checkpoint_verifies_against_its_sidecars(tmp_path, annotations_csv, features):
    """3.5.2: the loader accepts the stage's own checkpoint+sidecar pair."""
    from scripts.data.normalize import load_checkpoint_normalizer

    run_step_flat_vae(tmp_path, features, annotations_csv)
    out = tmp_path / "out"
    normalizer = load_checkpoint_normalizer(
        out / "flat_vae_model.pth",
        sidecars={
            "min_": str(out / "flat_vae_feature_min.npy"),
            "max_": str(out / "flat_vae_feature_max.npy"),
        },
    )
    np.testing.assert_array_equal(normalizer.min_, np.load(out / "flat_vae_feature_min.npy"))
    np.testing.assert_array_equal(normalizer.max_, np.load(out / "flat_vae_feature_max.npy"))


# ── The test split (1.5) ──────────────────────────────────────────────────
#
# With `test_split > 0` a third, never-trained-on partition exists. The fit must
# ignore it exactly as it ignores val — and unlike val, those rows are also read
# once at scoring time, so a silent fit on them would surface as a suspiciously
# good held-out number rather than as an error.


def test_step_flat_vae_saved_scaler_is_test_invariant(tmp_path, annotations_csv, features):
    """Perturbing the test rows must not move the fitted min/max."""
    a_min, a_max = run_step_flat_vae(
        tmp_path / "a", features, annotations_csv, test_split=TEST_SPLIT
    )[1:]
    perturbed, _, _ = perturb_test_rows_only(features, test_split=TEST_SPLIT)
    b_min, b_max = run_step_flat_vae(
        tmp_path / "b", perturbed, annotations_csv, test_split=TEST_SPLIT
    )[1:]

    assert np.array_equal(a_min, b_min)
    assert np.array_equal(a_max, b_max)


def test_step_flat_vae_train_data_is_invariant_to_test_perturbation(
    tmp_path, annotations_csv, features
):
    """A test row arbitrarily far from the rest must not move the training tensors."""
    a, _, _ = run_step_flat_vae(tmp_path / "a", features, annotations_csv, test_split=TEST_SPLIT)
    perturbed, _, test_idx = perturb_test_rows_only(features, test_split=TEST_SPLIT)
    b, _, _ = run_step_flat_vae(tmp_path / "b", perturbed, annotations_csv, test_split=TEST_SPLIT)

    assert test_idx.size > 0, "the test split must be non-empty for this to mean anything"
    assert np.array_equal(sorted_rows(a["train"]), sorted_rows(b["train"])), \
        "test rows leaked into the training tensors"
    # Control: the perturbation was confined to test rows, so val is untouched.
    assert torch.equal(a["val"], b["val"])


def test_a_test_split_with_no_feature_rows_is_rejected(tmp_path, annotations_csv, features):
    """An empty coverage cell must raise, not summarize (1.5)."""
    out = tmp_path / "out"
    out.mkdir()
    reread = pd.read_csv(annotations_csv, header=None, dtype={0: str})
    reread.columns = ["video_id", "timestamp", "x1", "y1", "x2", "y2", "action_id", "target_id"]
    # Draw the canonical split first so the stage re-reads the same one, then
    # strip the test video's feature rows — the coverage gap the guard exists for.
    assignment = resolve_split(
        str(out), reread, val_split=VAL_SPLIT, test_split=TEST_SPLIT,
        random_seed=SEED, group_key="video_id",
    )
    test_videos = set(assignment["test"])
    video_ids = video_ids_for()
    keep = [i for i, video_id in enumerate(video_ids) if video_id not in test_videos]

    with pytest.raises(RuntimeError, match="no feature rows"):
        run_step_flat_vae(
            tmp_path, features[keep], annotations_csv,
            video_ids=[video_ids[i] for i in keep],
            test_split=TEST_SPLIT,
        )


# ── The LSTM-VAE call site (1.2, 2.4) ─────────────────────────────────────
#
# The window twin of the flat tests: `min`/`max` must be computed from the
# training windows only, with the same `MinMaxNormalizer` the flat VAE uses (2.4).
# These run `step_lstm_vae` for real with a pre-warmed feature cache and a stubbed
# training loop, then perturb every held-out window and assert the persisted
# statistics do not move.


@pytest.fixture
def lstm_annotations_csv(tmp_path):
    """A constant `target_id`, so each video is exactly one 10-frame track."""
    path = tmp_path / "lstm_annotations.csv"
    rows = [f"v{i // 10},{i * 0.04},0.1,0.1,0.5,0.5,0,0" for i in range(N_FEATURES)]
    path.write_text("\n".join(rows))
    return str(path)


def _fake_lstm_train(captured):
    def fake_train_lstm_vae(vae, train_loader, val_loader, epochs, lr, device, **kwargs):
        def rows(loader):
            # TensorDataset yields `(tensor,)` batches, the sequence dataset
            # bare tensors; indexing a bare tensor would keep only its first row.
            batches = [batch[0] if isinstance(batch, (list, tuple)) else batch
                       for batch in loader]
            return torch.cat(batches)

        captured["train"] = rows(train_loader)
        captured["val"] = rows(val_loader)
        captured["val_loader"] = val_loader
        history = empty_history()
        for _ in range(epochs):
            for column in HISTORY_COLUMNS:
                history[column].append(0.0)
        checkpoint = kwargs.get("checkpoint")
        if checkpoint is not None:
            checkpoint["best_state"] = {k: v.clone() for k, v in vae.state_dict().items()}
            checkpoint["best_epoch"] = 1
            checkpoint["best_metric"] = 0.0
            checkpoint["final_epoch"] = epochs
        return history

    return fake_train_lstm_vae


def run_step_lstm_vae(tmp_path, feature_matrix, annotations_csv, **config_overrides):
    """Run step 7 for real, capturing exactly what the training loop receives."""
    captured = {}
    monkey = pytest.MonkeyPatch()

    # Cache keys built exactly as the sequence dataset builds them (fps=25),
    # one per annotation row, each carrying that row's feature vector.
    reread = pd.read_csv(annotations_csv, header=None, dtype={0: str})
    reread.columns = ["video_id", "timestamp", "x1", "y1", "x2", "y2", "action_id", "target_id"]
    keys = [row_feature_key(row, str(row["video_id"])) for _, row in reread.iterrows()]
    index_of = {key: i for i, key in enumerate(keys)}
    cache = FeatureCache(directory=None, signature=cache_signature("test-weights"))
    cache.ensure(keys, lambda key: feature_matrix[index_of[key]], verbose=False)

    monkey.setattr(pipeline, "train_lstm_vae", _fake_lstm_train(captured))
    # `input_dim` is derived from this FEATURE_DIM-wide cache (2.12), so the real
    # scoring pass runs without forcing the model width; the normalisation under
    # test does not depend on it.
    try:
        step_lstm_vae(
            annotations_csv, str(tmp_path / "frames"), None, "cpu",
            str(tmp_path / "out"), config(**config_overrides),
            feature_cache=cache,
        )
    finally:
        monkey.undo()

    min_ = np.load(tmp_path / "out" / "lstm_vae_feature_min.npy")
    max_ = np.load(tmp_path / "out" / "lstm_vae_feature_max.npy")
    return captured, min_, max_


def test_step_lstm_vae_statistics_are_held_out_invariant(tmp_path, lstm_annotations_csv, features):
    """1.2 end to end: no window outside train may move the fitted min/max."""
    _, min_a, max_a = run_step_lstm_vae(
        tmp_path / "a", features, lstm_annotations_csv, test_split=TEST_SPLIT
    )
    perturbed, _, held_out = perturb_held_out_rows(features, test_split=TEST_SPLIT)
    _, min_b, max_b = run_step_lstm_vae(
        tmp_path / "b", perturbed, lstm_annotations_csv, test_split=TEST_SPLIT
    )

    assert held_out.size > 0, "the held-out set must be non-empty for this to mean anything"
    assert np.array_equal(min_a, min_b)
    assert np.array_equal(max_a, max_b)


def test_step_lstm_vae_minmax_maps_train_into_unit_range(
    tmp_path, lstm_annotations_csv, features
):
    """2.4: the loaders receive `(x - min)/(max - min)`; a train fit sends *train* to [0, 1]."""
    captured, min_, max_ = run_step_lstm_vae(
        tmp_path, features, lstm_annotations_csv, test_split=TEST_SPLIT
    )
    train_flat = captured["train"].reshape(-1, FEATURE_DIM)

    assert min_.shape == (FEATURE_DIM,) and max_.shape == (FEATURE_DIM,)
    assert np.allclose(train_flat.amin(dim=0).numpy(), 0.0, atol=1e-6)
    assert np.allclose(train_flat.amax(dim=0).numpy(), 1.0, atol=1e-6)


def test_step_lstm_vae_held_out_windows_stay_out_of_range(
    tmp_path, lstm_annotations_csv, features
):
    """The shared scaler is train-fit, so an out-of-range held-out window stays out."""
    perturbed, _, _ = perturb_held_out_rows(features, test_split=TEST_SPLIT)
    captured, _, _ = run_step_lstm_vae(
        tmp_path, perturbed, lstm_annotations_csv, test_split=TEST_SPLIT
    )

    assert captured["val"].max() > 1.0, "an out-of-range val window must not be rescaled into [0, 1]"


def test_step_lstm_vae_writes_best_and_final_checkpoints(tmp_path, lstm_annotations_csv, features):
    """2.7 on the window path: both checkpoints reach disk."""
    run_step_lstm_vae(tmp_path, features, lstm_annotations_csv)
    out = tmp_path / "out"
    assert (out / "lstm_vae_model.pth").is_file()
    assert (out / "lstm_vae_best.pth").is_file()


def test_step_lstm_vae_validation_is_a_full_ordered_pass(tmp_path, lstm_annotations_csv, features):
    """2.9: a `SubsetRandomSampler` val loader re-shuffled every epoch, so `val_loss`
    carried a draw as well as the weights. It must now be a complete, ordered pass."""
    from torch.utils.data import Subset

    captured, _, _ = run_step_lstm_vae(tmp_path, features, lstm_annotations_csv)
    loader = captured["val_loader"]

    assert isinstance(loader.dataset, Subset), "val must be a Subset of the window dataset"
    assert isinstance(loader.sampler, torch.utils.data.SequentialSampler)

    def order():
        return torch.cat([b if not isinstance(b, (list, tuple)) else b[0] for b in loader])

    first, second = order(), order()
    assert len(first) == len(loader.dataset), "every val window must be visited exactly once"
    assert torch.equal(first, second), "a fixed pass must not re-shuffle between epochs"


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

    video_ids = video_ids_for(feature_matrix.shape[0])
    monkeypatch.setattr(
        module, "build_feature_dataset",
        lambda *a, **k: (feature_matrix, np.array(video_ids, dtype=object)),
    )
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
    _, indices = grouped_split(
        features, video_ids_for(), str(tmp_path / "expected"),
        val_split=STANDALONE_VAL_SPLIT, test_split=STANDALONE_TEST_SPLIT,
    )
    train_idx = indices["train"]

    assert np.array_equal(saved_min, features[train_idx].min(axis=0))
    assert np.array_equal(saved_max, features[train_idx].max(axis=0))


def test_standalone_vae_train_data_is_val_invariant(monkeypatch, tmp_path, annotations_df, features):
    monkeypatch.chdir(tmp_path)
    perturbed, _, _ = perturb_val_rows_only(
        features, val_split=STANDALONE_VAL_SPLIT, test_split=STANDALONE_TEST_SPLIT
    )

    before = run_standalone_vae(monkeypatch, features, annotations_df)
    after = run_standalone_vae(monkeypatch, perturbed, annotations_df)

    assert np.array_equal(sorted_rows(before[0]["train"]), sorted_rows(after[0]["train"])), \
        "val rows leaked into the train tensors"
    assert np.array_equal(before[1], after[1]) and np.array_equal(before[2], after[2])


def test_standalone_vae_keeps_one_video_entirely_on_one_side(monkeypatch, tmp_path, annotations_df):
    """1.3 on the standalone copy, which had its own independent split."""
    monkeypatch.chdir(tmp_path)
    video_ids = video_ids_for()
    value_of = {video_id: i for i, video_id in enumerate(video_ids)}
    per_video = np.array([value_of[v] for v in video_ids], dtype=np.float32)
    features = np.repeat(per_video[:, None], FEATURE_DIM, axis=1)

    captured, saved_min, saved_max = run_standalone_vae(monkeypatch, features, annotations_df)

    def videos_in(tensor):
        original = tensor.numpy()[:, 0] * (saved_max[0] - saved_min[0]) + saved_min[0]
        return {video_ids[int(round(value))] for value in original}

    train_videos, val_videos = videos_in(captured["train"]), videos_in(captured["val"])
    assert train_videos and val_videos
    assert not train_videos & val_videos, (
        f"videos on both sides of the split: {sorted(train_videos & val_videos)}"
    )
