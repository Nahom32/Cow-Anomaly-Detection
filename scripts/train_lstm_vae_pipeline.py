import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler

from scripts.data.feature_cache import open_run_feature_cache
from scripts.data.normalize import ZScoreNormalizer
from scripts.data.splits import describe, resolve_indices, resolve_split
from scripts.dataset.sequence_dataset import CowSequenceDataset, NormalisedSeqDataset
from scripts.manifest import RunManifest
from scripts.models.feature_extractor import create_feature_extractor
from scripts.models.lstm_vae import LSTMVAE, train_lstm_vae
from scripts.utils.hashing import hash_json
from scripts.utils.history import save_history
from scripts.utils.seeding import set_seed


def main():
    YOLO_MODEL_PATH = "/content/drive/MyDrive/cow_detector26n_results/weights/best.pt"
    FRAMES_DIR = "/root/.cache/kagglehub/datasets/fandaoerji/cbvd-5cow-behavior-video-dataset/versions/11/rawframes_mini"
    ANNOTATIONS_CSV = "/root/.cache/kagglehub/datasets/fandaoerji/cbvd-5cow-behavior-video-dataset/versions/11/annotations/ava_train_v2.1.csv"
    DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    NORMAL_ACTION_IDS = [0, 1, 2]
    SEQ_LEN = 16
    STRIDE = 8
    BATCH_SIZE = 32
    EPOCHS = 50
    LR = 1e-3
    OUTPUT_DIR = "."
    SEED = 42
    SPLIT_GROUP_KEY = "video_id"

    set_seed(SEED)
    print(f"Using device: {DEVICE}")

    config = {"model": "lstm_vae", "seed": SEED, "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR,
              "normal_action_ids": NORMAL_ACTION_IDS, "feature_layer": 9, "seq_len": SEQ_LEN,
              "stride": STRIDE, "val_split": 0.2, "test_split": 0.0,
              "split_group_key": SPLIT_GROUP_KEY, "yolo_weights": YOLO_MODEL_PATH}
    manifest = RunManifest.create(OUTPUT_DIR, config, seed=SEED, device=DEVICE)
    manifest.record_artifact("yolo_weights", YOLO_MODEL_PATH)
    manifest.record_artifact("annotations_csv", ANNOTATIONS_CSV)

    df = pd.read_csv(ANNOTATIONS_CSV, header=None, dtype={0: str})
    df.columns = ["video_id", "timestamp", "x1", "y1", "x2", "y2", "action_id", "target_id"]

    print("Loading YOLO feature extractor...")
    feature_extractor, hook = create_feature_extractor(YOLO_MODEL_PATH, layer_index=9, device=DEVICE)
    feature_cache = open_run_feature_cache(OUTPUT_DIR, YOLO_MODEL_PATH, layer_index=9)

    print("Building sequence dataset...")
    seq_dataset = CowSequenceDataset(
        df=df,
        frames_dir=FRAMES_DIR,
        feature_extractor=feature_extractor,
        feature_cache=feature_cache,
        seq_len=SEQ_LEN,
        stride=STRIDE,
        normal_action_ids=NORMAL_ACTION_IDS,
        device=DEVICE,
        seed=SEED,
    )
    print(f"Total sequences: {len(seq_dataset)}")

    # The canonical split (1.4), shared with the flat VAE through the run's split
    # manifest rather than redrawn here from an unordered set.
    assignment = resolve_split(
        OUTPUT_DIR, df, val_split=0.2, test_split=0.0,
        random_seed=SEED, group_key=SPLIT_GROUP_KEY,
    )
    print(
        f"Split by {SPLIT_GROUP_KEY}: "
        + ", ".join(f"{name}={len(assignment[name])}" for name in ("train", "val", "test"))
    )
    # Tracks are keyed `(video_id, target_id)`; which half is the split unit depends
    # on the configured group key.
    track_key_index = 0 if SPLIT_GROUP_KEY == "video_id" else 1
    sequence_groups = [str(key[track_key_index]) for key, _ in seq_dataset.sequences]
    sequence_split = resolve_indices(assignment, sequence_groups)
    train_indices = sequence_split["train"].tolist()
    val_indices = sequence_split["val"].tolist()

    # Statistics from the training sequences only (1.2). Fitting mean/std over every
    # sequence in the run let each val window contribute to the numbers it was then
    # normalized by.
    print("Computing normalisation statistics from the training sequences...")
    normalizer = ZScoreNormalizer().fit(seq_dataset.stack(train_indices))
    mean, std = normalizer.mean_, normalizer.std_

    norm_dataset = NormalisedSeqDataset(
        mean=mean,
        std=std,
        df=df,
        frames_dir=FRAMES_DIR,
        feature_extractor=feature_extractor,
        feature_cache=feature_cache,
        seq_len=SEQ_LEN,
        stride=STRIDE,
        normal_action_ids=NORMAL_ACTION_IDS,
        device=DEVICE,
        seed=SEED,
    )

    train_generator = torch.Generator().manual_seed(SEED)
    val_generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(norm_dataset, batch_size=BATCH_SIZE, sampler=SubsetRandomSampler(train_indices, generator=train_generator))
    val_loader = DataLoader(norm_dataset, batch_size=BATCH_SIZE, sampler=SubsetRandomSampler(val_indices, generator=val_generator))

    vae = LSTMVAE(input_dim=256, hidden_dim=128, latent_dim=32, num_layers=1)

    print("Starting LSTM-VAE training...")
    history = train_lstm_vae(vae, train_loader, val_loader, epochs=EPOCHS, lr=LR, device=DEVICE)

    torch.save(vae.state_dict(), os.path.join(OUTPUT_DIR, "lstm_vae_anomaly.pth"))
    save_history(history, os.path.join(OUTPUT_DIR, "lstm_vae_training_history.csv"))
    np.save(os.path.join(OUTPUT_DIR, "feature_mean.npy"), mean)
    np.save(os.path.join(OUTPUT_DIR, "feature_std.npy"), std)

    manifest.record_split("canonical_groups", {
        name: sorted(map(str, assignment[name])) for name in ("train", "val", "test")
    })
    manifest.record_split("lstm_vae_groups", {
        name: sorted({sequence_groups[i] for i in sequence_split[name]})
        for name in ("train", "val", "test")
    })
    manifest.record_array("lstm_vae_feature_mean", mean)
    manifest.record_array("lstm_vae_feature_std", std)
    manifest.record_step("lstm_vae", config_hash=hash_json(config), n_sequences=len(seq_dataset),
                         n_train=len(train_indices), n_val=len(val_indices),
                         n_videos=len(set(sequence_groups)),
                         split_group_key=SPLIT_GROUP_KEY,
                         split_sha256=describe(assignment)["sha256"],
                         normalizer_fit_on="train")
    manifest.save()
    print(f"Manifest: {manifest.path}")

    hook.remove()
    print("Done.")


if __name__ == "__main__":
    main()
