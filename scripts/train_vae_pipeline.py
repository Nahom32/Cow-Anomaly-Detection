import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from scripts.data.build_features import build_feature_dataset
from scripts.data.feature_cache import open_run_feature_cache
from scripts.data.normalize import MinMaxNormalizer
from scripts.data.splits import describe, resolve_indices, resolve_split
from scripts.manifest import RunManifest
from scripts.models.feature_extractor import create_feature_extractor
from scripts.models.vae import VAE, plot_history, train_vae
from scripts.utils.hashing import hash_json
from scripts.utils.history import save_history
from scripts.utils.seeding import set_seed


def main():
    YOLO_MODEL_PATH = "/content/drive/MyDrive/cow_detector26n_results/weights/best.pt"
    FRAMES_DIR = "/root/.cache/kagglehub/datasets/fandaoerji/cbvd-5cow-behavior-video-dataset/versions/11/rawframes_mini"
    ANNOTATIONS_CSV = "/root/.cache/kagglehub/datasets/fandaoerji/cbvd-5cow-behavior-video-dataset/versions/11/annotations/ava_train_v2.1.csv"
    NORMAL_ACTION_IDS = [0, 1, 2]
    DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    EPOCHS = 50
    BATCH_SIZE = 64
    LR = 1e-3
    OUTPUT_DIR = "."
    SEED = 42
    SPLIT_GROUP_KEY = "video_id"

    set_seed(SEED)
    print(f"Using device: {DEVICE}")

    config = {"model": "flat_vae", "seed": SEED, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
              "lr": LR, "normal_action_ids": NORMAL_ACTION_IDS, "feature_layer": 9,
              "val_split": 0.2, "test_split": 0.0, "split_group_key": SPLIT_GROUP_KEY,
              "yolo_weights": YOLO_MODEL_PATH}
    manifest = RunManifest.create(OUTPUT_DIR, config, seed=SEED, device=DEVICE)
    manifest.record_artifact("yolo_weights", YOLO_MODEL_PATH)
    manifest.record_artifact("annotations_csv", ANNOTATIONS_CSV)

    print("Loading annotations...")
    df = pd.read_csv(ANNOTATIONS_CSV, header=None, dtype={0: str})
    df.columns = ["video_id", "timestamp", "x1", "y1", "x2", "y2", "action_id", "target_id"]

    print("Creating YOLO feature extractor...")
    feature_extractor, hook = create_feature_extractor(YOLO_MODEL_PATH, layer_index=9, device=DEVICE)
    feature_cache = open_run_feature_cache(OUTPUT_DIR, YOLO_MODEL_PATH, layer_index=9)

    print("Extracting features from normal behaviour frames...")
    features, feature_video_ids = build_feature_dataset(
        df, FRAMES_DIR, feature_extractor,
        normal_action_ids=NORMAL_ACTION_IDS, device=DEVICE,
        feature_cache=feature_cache,
    )
    print(f"Extracted {features.shape[0]} feature vectors of dimension {features.shape[1]}")

    # The canonical split (1.3, 1.4): grouped, so consecutive frames of one cow
    # cannot land on both sides, and shared with the other stages via the manifest
    # rather than redrawn per entry point.
    print("Splitting by video, then fitting normalisation on the training rows only...")
    assignment = resolve_split(
        OUTPUT_DIR, df, val_split=0.2, test_split=0.0,
        random_seed=SEED, group_key=SPLIT_GROUP_KEY,
    )
    print(
        f"Split by {SPLIT_GROUP_KEY}: "
        + ", ".join(f"{name}={len(assignment[name])}" for name in ("train", "val", "test"))
    )
    row_split = resolve_indices(assignment, feature_video_ids)
    train_idx, val_idx = row_split["train"], row_split["val"]
    # Fit after the split: min/max over the whole matrix normalizes each val
    # sample with its own extremes (1.1).
    normalizer = MinMaxNormalizer().fit(features[train_idx])
    features_norm = normalizer.transform(features)
    X_train, X_val = features_norm[train_idx], features_norm[val_idx]
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32)),
        batch_size=BATCH_SIZE, shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.float32)),
        batch_size=BATCH_SIZE, shuffle=False,
    )

    input_dim = features.shape[1]
    vae = VAE(input_dim, hidden_dim=128, latent_dim=32)

    print("Starting VAE training...")
    history = train_vae(vae, train_loader, val_loader, epochs=EPOCHS, lr=LR, device=DEVICE)

    save_history(history, os.path.join(OUTPUT_DIR, "vae_training_history.csv"))
    torch.save(vae.state_dict(), os.path.join(OUTPUT_DIR, "vae_anomaly_model.pth"))
    np.save(os.path.join(OUTPUT_DIR, "feature_min.npy"), normalizer.min_)
    np.save(os.path.join(OUTPUT_DIR, "feature_max.npy"), normalizer.max_)

    plot_history(history, save_path=os.path.join(OUTPUT_DIR, "vae_training_history.png"))

    manifest.record_array("flat_vae_features", features, n=features.shape[0], dim=features.shape[1])
    manifest.record_array("flat_vae_features_norm", features_norm)
    manifest.record_array("flat_vae_train_rows", X_train)
    manifest.record_array("flat_vae_val_rows", X_val)
    manifest.record_array("flat_vae_feature_min", normalizer.min_)
    manifest.record_array("flat_vae_feature_max", normalizer.max_)
    manifest.record_split("flat_vae_rows", {"train": sorted(train_idx.tolist()), "val": sorted(val_idx.tolist())})
    manifest.record_split("canonical_groups", {
        name: sorted(map(str, assignment[name])) for name in ("train", "val", "test")
    })
    manifest.record_step("flat_vae", config_hash=hash_json(config), n_features=int(features.shape[0]),
                         n_train=int(X_train.shape[0]), n_val=int(X_val.shape[0]),
                         split_group_key=SPLIT_GROUP_KEY,
                         split_sha256=describe(assignment)["sha256"],
                         normalizer_fit_on="train")
    manifest.save()
    print(f"Manifest: {manifest.path}")

    hook.remove()
    print("Done.")


if __name__ == "__main__":
    main()
