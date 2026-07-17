import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, SubsetRandomSampler

from scripts.dataset.sequence_dataset import CowSequenceDataset, NormalisedSeqDataset
from scripts.models.feature_extractor import create_feature_extractor
from scripts.models.lstm_vae import LSTMVAE, train_lstm_vae


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

    print(f"Using device: {DEVICE}")

    df = pd.read_csv(ANNOTATIONS_CSV, header=None, dtype={0: str})
    df.columns = ["video_id", "timestamp", "x1", "y1", "x2", "y2", "action_id", "target_id"]

    print("Loading YOLO feature extractor...")
    feature_extractor, hook = create_feature_extractor(YOLO_MODEL_PATH, layer_index=9, device=DEVICE)

    print("Building sequence dataset...")
    seq_dataset = CowSequenceDataset(
        df=df,
        frames_dir=FRAMES_DIR,
        feature_extractor=feature_extractor,
        seq_len=SEQ_LEN,
        stride=STRIDE,
        normal_action_ids=NORMAL_ACTION_IDS,
        device=DEVICE,
    )
    print(f"Total sequences: {len(seq_dataset)}")

    print("Computing normalisation statistics...")
    all_feats = []
    for i in range(len(seq_dataset)):
        seq = seq_dataset[i]
        all_feats.append(seq.numpy())
    all_feats = np.concatenate(all_feats, axis=0)
    mean = all_feats.mean(axis=0)
    std = all_feats.std(axis=0) + 1e-8

    norm_dataset = NormalisedSeqDataset(
        mean=mean,
        std=std,
        df=df,
        frames_dir=FRAMES_DIR,
        feature_extractor=feature_extractor,
        seq_len=SEQ_LEN,
        stride=STRIDE,
        normal_action_ids=NORMAL_ACTION_IDS,
        device=DEVICE,
    )

    video_ids = list(set(key[0] for key in seq_dataset.tracks.keys()))
    train_vids, val_vids = train_test_split(video_ids, test_size=0.2, random_state=42)

    train_indices = [i for i, (key, _) in enumerate(seq_dataset.sequences) if key[0] in train_vids]
    val_indices = [i for i, (key, _) in enumerate(seq_dataset.sequences) if key[0] in val_vids]

    train_loader = DataLoader(norm_dataset, batch_size=BATCH_SIZE, sampler=SubsetRandomSampler(train_indices))
    val_loader = DataLoader(norm_dataset, batch_size=BATCH_SIZE, sampler=SubsetRandomSampler(val_indices))

    vae = LSTMVAE(input_dim=256, hidden_dim=128, latent_dim=32, num_layers=1)

    print("Starting LSTM-VAE training...")
    history = train_lstm_vae(vae, train_loader, val_loader, epochs=EPOCHS, lr=LR, device=DEVICE)

    torch.save(vae.state_dict(), os.path.join(OUTPUT_DIR, "lstm_vae_anomaly.pth"))
    np.save(os.path.join(OUTPUT_DIR, "feature_mean.npy"), mean)
    np.save(os.path.join(OUTPUT_DIR, "feature_std.npy"), std)

    hook.remove()
    print("Done.")


if __name__ == "__main__":
    main()
