"""The train-fitted scaler travels inside the model checkpoint (3.5.1).

`flat_vae_feature_min.npy`/`_max.npy` (and the LSTM-VAE pair) used to be the only
record of the normalization, so a checkpoint and a sidecar could be paired from
different runs. The fix is to serialize the normalizer with the weights it was
fitted for; these tests pin the round-trip and the embedding.
"""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from scripts.checkpoint import load_checkpoint, save_model_checkpoint
from scripts.data.normalize import (
    MinMaxNormalizer,
    ZScoreNormalizer,
    load_normalizer,
)
from scripts.models.vae import VAE, train_vae


def _features(n=60, d=4, seed=0):
    return np.random.default_rng(seed).random((n, d)).astype(np.float32)


def _loader(rows, batch_size=8, shuffle=False):
    return DataLoader(TensorDataset(rows), batch_size=batch_size, shuffle=shuffle)


@pytest.mark.parametrize("factory", [MinMaxNormalizer, ZScoreNormalizer])
def test_state_dict_round_trips(factory):
    normalizer = factory().fit(_features())
    restored = load_normalizer(normalizer.state_dict())

    assert type(restored) is type(normalizer)
    assert restored.is_fitted
    for field in normalizer.state_fields:
        np.testing.assert_array_equal(getattr(restored, field), getattr(normalizer, field))
    np.testing.assert_array_equal(restored.transform(_features()), normalizer.transform(_features()))


def test_state_dict_records_kind():
    assert MinMaxNormalizer().fit(_features()).state_dict()["kind"] == "minmax"
    assert ZScoreNormalizer().fit(_features()).state_dict()["kind"] == "zscore"


def test_state_dict_before_fit_is_loud():
    with pytest.raises(RuntimeError, match="before fit"):
        MinMaxNormalizer().state_dict()


def test_load_normalizer_rejects_unknown_kind():
    with pytest.raises(ValueError, match="unknown normalizer kind"):
        load_normalizer({"kind": "quantile"})


def test_from_state_rejects_missing_field():
    with pytest.raises(ValueError, match="missing 'max_'"):
        MinMaxNormalizer.from_state({"kind": "minmax", "min_": np.zeros(4)})


def test_from_state_rejects_wrong_class():
    with pytest.raises(ValueError, match="cannot load"):
        ZScoreNormalizer.from_state({"kind": "minmax", "min_": np.zeros(4), "max_": np.ones(4)})


def test_save_model_checkpoint_embeds_normalizer(tmp_path):
    normalizer = MinMaxNormalizer().fit(_features())
    model_state = VAE(4, hidden_dim=8, latent_dim=2).state_dict()
    path = tmp_path / "model.pth"

    save_model_checkpoint(
        path, model_state=model_state, normalizer=normalizer.state_dict(),
        config={"seed": 0}, seed=0, data_fingerprint="fp", epoch=3,
    )

    checkpoint = load_checkpoint(path)
    assert set(checkpoint["model_state"]) == set(model_state)
    assert checkpoint["epoch"] == 3
    restored = load_normalizer(checkpoint["normalizer"])
    np.testing.assert_array_equal(restored.min_, normalizer.min_)
    np.testing.assert_array_equal(restored.max_, normalizer.max_)


def test_save_model_checkpoint_without_normalizer_is_explicit(tmp_path):
    path = tmp_path / "model.pth"
    save_model_checkpoint(path, model_state={"w": torch.zeros(1)}, config={}, seed=0)
    assert load_checkpoint(path)["normalizer"] is None


def test_train_loop_epoch_checkpoint_carries_normalizer(tmp_path):
    normalizer = MinMaxNormalizer().fit(_features())
    vae = VAE(4, hidden_dim=8, latent_dim=2)

    train_vae(
        vae, _loader(torch.randn(16, 4), shuffle=True), _loader(torch.randn(8, 4)),
        epochs=1, lr=1e-3, device="cpu", checkpoint={}, seed=0, config={"key": "value"},
        checkpoint_dir=str(tmp_path), checkpoint_prefix="flat_vae", keep_last=3,
        normalizer=normalizer.state_dict(),
    )

    epoch_path = tmp_path / "flat_vae_epoch_0001.pth"
    assert epoch_path.is_file()
    embedded = load_normalizer(load_checkpoint(epoch_path)["normalizer"])
    np.testing.assert_array_equal(embedded.min_, normalizer.min_)
    np.testing.assert_array_equal(embedded.max_, normalizer.max_)
