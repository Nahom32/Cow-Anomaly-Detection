import os
import time

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, SubsetRandomSampler, TensorDataset

from scripts.config import CONFIG
from scripts.data.build_features import build_feature_dataset, normalize_features
from scripts.data.create_yolo_dataset import create_yolo_dataset
from scripts.data.download_dataset import download_dataset
from scripts.dataset.sequence_dataset import CowSequenceDataset, NormalisedSeqDataset
from scripts.models.feature_extractor import create_feature_extractor
from scripts.models.lstm_vae import LSTMVAE, train_lstm_vae
from scripts.models.train_yolo_n import train_yolo26n
from scripts.models.vae import VAE, plot_history, save_history, train_vae


DEFAULT_OUTPUT_DIR = "pipeline_output"


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def step_download_dataset(output_dir):
    print("=" * 60)
    print("STEP 1: Downloading dataset")
    print("=" * 60)
    data_root = download_dataset()

    annotations_csv = os.path.join(data_root, "annotations", "ava_train_v2.1.csv")
    frames_dir = os.path.join(data_root, "rawframes_mini")
    return data_root, annotations_csv, frames_dir


def step_create_yolo_dataset(data_root, output_dir):
    print("\n" + "=" * 60)
    print("STEP 2: Creating YOLO dataset")
    print("=" * 60)
    yolo_data_dir = os.path.join(output_dir, "yolo_dataset")
    create_yolo_dataset(output_dir=yolo_data_dir, data_root=data_root)
    return yolo_data_dir


def step_create_yaml(yolo_data_dir):
    print("\n" + "=" * 60)
    print("STEP 3: Creating YOLO data YAML")
    print("=" * 60)
    yaml_path = os.path.join(yolo_data_dir, "data.yaml")
    content = {
        "path": os.path.abspath(yolo_data_dir),
        "train": "train/images",
        "val": "val/images",
        "names": {0: "cow"},
    }
    with open(yaml_path, "w") as f:
        yaml.dump(content, f, default_flow_style=False)
    print(f"Written: {yaml_path}")
    return yaml_path


def step_train_yolo(data_yaml, output_dir, config):
    print("\n" + "=" * 60)
    print("STEP 4: Training YOLO26n")
    print("=" * 60)
    results = train_yolo26n(data_yaml=data_yaml)
    yolo_weights = os.path.join("cow_detector", "yolo26n_cbvd", "weights", "best.pt")
    print(f"YOLO weights saved at: {yolo_weights}")
    return yolo_weights


def step_feature_extractor(yolo_weights, device):
    print("\n" + "=" * 60)
    print("STEP 5: Creating YOLO feature extractor")
    print("=" * 60)
    feature_extractor, hook = create_feature_extractor(
        yolo_weights, layer_index=9, device=device
    )
    print("Feature extractor ready (SPPF layer 9)")
    return feature_extractor, hook


def step_flat_vae(annotations_csv, frames_dir, feature_extractor, hook, device, output_dir, config):
    print("\n" + "=" * 60)
    print("STEP 6: Training flat VAE")
    print("=" * 60)
    df = pd.read_csv(annotations_csv, header=None, dtype={0: str})
    df.columns = ["video_id", "timestamp", "x1", "y1", "x2", "y2", "action_id", "target_id"]

    print("Extracting frame-level features...")
    features = build_feature_dataset(
        df, frames_dir, feature_extractor,
        normal_action_ids=config["normal_action_ids"], device=device,
    )
    print(f"Extracted {features.shape[0]} features, dim={features.shape[1]}")

    features_norm, min_val, max_val = normalize_features(features)

    X_train, X_val = train_test_split(
        features_norm, test_size=config["val_split"], random_state=config["random_seed"]
    )
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32)),
        batch_size=config["vae_batch_size"], shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.float32)),
        batch_size=config["vae_batch_size"], shuffle=False,
    )

    vae = VAE(features.shape[1], hidden_dim=config["vae_hidden_dim"], latent_dim=config["vae_latent_dim"])
    history = train_vae(vae, train_loader, val_loader, epochs=config["vae_epochs"], lr=config["vae_lr"], device=device)

    save_history(history, os.path.join(output_dir, "flat_vae_history.csv"))
    torch.save(vae.state_dict(), os.path.join(output_dir, "flat_vae_model.pth"))
    np.save(os.path.join(output_dir, "flat_vae_feature_min.npy"), min_val)
    np.save(os.path.join(output_dir, "flat_vae_feature_max.npy"), max_val)
    plot_history(history)
    print("Flat VAE complete.")
    return features


def step_lstm_vae(annotations_csv, frames_dir, feature_extractor, device, output_dir, config):
    print("\n" + "=" * 60)
    print("STEP 7: Training LSTM-VAE")
    print("=" * 60)
    df = pd.read_csv(annotations_csv, header=None, dtype={0: str})
    df.columns = ["video_id", "timestamp", "x1", "y1", "x2", "y2", "action_id", "target_id"]

    print("Building sequence dataset...")
    seq_dataset = CowSequenceDataset(
        df=df,
        frames_dir=frames_dir,
        feature_extractor=feature_extractor,
        seq_len=config["seq_len"],
        stride=config["seq_stride"],
        normal_action_ids=config["normal_action_ids"],
        device=device,
    )
    print(f"Total sequences: {len(seq_dataset)}")

    print("Computing normalisation statistics...")
    all_feats = []
    for i in range(len(seq_dataset)):
        all_feats.append(seq_dataset[i].numpy())
    all_feats = np.concatenate(all_feats, axis=0)
    mean = all_feats.mean(axis=0)
    std = all_feats.std(axis=0) + 1e-8

    norm_dataset = NormalisedSeqDataset(
        mean=mean,
        std=std,
        df=df,
        frames_dir=frames_dir,
        feature_extractor=feature_extractor,
        seq_len=config["seq_len"],
        stride=config["seq_stride"],
        normal_action_ids=config["normal_action_ids"],
        device=device,
    )

    video_ids = list(set(key[0] for key in seq_dataset.tracks.keys()))
    train_vids, val_vids = train_test_split(
        video_ids, test_size=config["val_split"], random_state=config["random_seed"]
    )

    train_indices = [i for i, (key, _) in enumerate(seq_dataset.sequences) if key[0] in train_vids]
    val_indices = [i for i, (key, _) in enumerate(seq_dataset.sequences) if key[0] in val_vids]

    train_loader = DataLoader(norm_dataset, batch_size=config["lstm_vae_batch_size"], sampler=SubsetRandomSampler(train_indices))
    val_loader = DataLoader(norm_dataset, batch_size=config["lstm_vae_batch_size"], sampler=SubsetRandomSampler(val_indices))

    vae = LSTMVAE(
        input_dim=256,
        hidden_dim=config["lstm_vae_hidden_dim"],
        latent_dim=config["lstm_vae_latent_dim"],
        num_layers=config["lstm_vae_num_layers"],
    )

    history = train_lstm_vae(
        vae, train_loader, val_loader,
        epochs=config["lstm_vae_epochs"], lr=config["lstm_vae_lr"], device=device,
    )

    torch.save(vae.state_dict(), os.path.join(output_dir, "lstm_vae_model.pth"))
    np.save(os.path.join(output_dir, "lstm_vae_feature_mean.npy"), mean)
    np.save(os.path.join(output_dir, "lstm_vae_feature_std.npy"), std)
    print("LSTM-VAE complete.")


def main():
    output_dir = DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    device = get_device()
    print(f"Device: {device}")
    print(f"Output: {os.path.abspath(output_dir)}")
    config = CONFIG.copy()

    t0 = time.time()

    data_root, annotations_csv, frames_dir = step_download_dataset(output_dir)
    yolo_data_dir = step_create_yolo_dataset(data_root, output_dir)
    data_yaml = step_create_yaml(yolo_data_dir)
    yolo_weights = step_train_yolo(data_yaml, output_dir, config)
    feature_extractor, hook = step_feature_extractor(yolo_weights, device)

    step_flat_vae(annotations_csv, frames_dir, feature_extractor, hook, device, output_dir, config)
    step_lstm_vae(annotations_csv, frames_dir, feature_extractor, device, output_dir, config)

    hook.remove()

    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print(f"Total time: {elapsed / 60:.1f} min")
    print(f"Outputs saved to: {os.path.abspath(output_dir)}")
    print("=" * 60)
    print("\nArtifacts:")
    for f in sorted(os.listdir(output_dir)):
        size = os.path.getsize(os.path.join(output_dir, f))
        print(f"  {f:40s} {size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
