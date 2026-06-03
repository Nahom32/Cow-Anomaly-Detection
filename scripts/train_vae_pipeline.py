import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from scripts.data.build_features import build_feature_dataset, normalize_features
from scripts.models.feature_extractor import create_feature_extractor
from scripts.models.vae import VAE, plot_history, save_history, train_vae


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

    print(f"Using device: {DEVICE}")

    print("Loading annotations...")
    df = pd.read_csv(ANNOTATIONS_CSV, header=None, dtype={0: str})
    df.columns = ["video_id", "timestamp", "x1", "y1", "x2", "y2", "action_id", "target_id"]

    print("Creating YOLO feature extractor...")
    feature_extractor, hook = create_feature_extractor(YOLO_MODEL_PATH, layer_index=9, device=DEVICE)

    print("Extracting features from normal behaviour frames...")
    features = build_feature_dataset(
        df, FRAMES_DIR, feature_extractor,
        normal_action_ids=NORMAL_ACTION_IDS, device=DEVICE,
    )
    print(f"Extracted {features.shape[0]} feature vectors of dimension {features.shape[1]}")

    print("Normalizing features...")
    features, min_val, max_val = normalize_features(features)

    X_train, X_val = train_test_split(features, test_size=0.2, random_state=42)
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32)),
        batch_size=BATCH_SIZE, shuffle=True,
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
    np.save(os.path.join(OUTPUT_DIR, "feature_min.npy"), min_val)
    np.save(os.path.join(OUTPUT_DIR, "feature_max.npy"), max_val)

    plot_history(history)

    hook.remove()
    print("Done.")


if __name__ == "__main__":
    main()
