import argparse
import glob
import json
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
STATE_FILE = ".pipeline_state.json"

# Step names for display
STEP_NAMES = {
    1: "Download dataset",
    2: "Create YOLO dataset",
    3: "Create YOLO data YAML",
    4: "Train YOLO26n",
    5: "Create feature extractor",
    6: "Train flat VAE",
    7: "Train LSTM-VAE",
}


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_state(output_dir):
    path = os.path.join(output_dir, STATE_FILE)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_state(output_dir, state):
    path = os.path.join(output_dir, STATE_FILE)
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


def find_best_pt():
    # YOLO internally prepends runs/detect/ to the project path
    for project_dir in ["runs/detect/cow_detector", "cow_detector"]:
        exact = os.path.join(project_dir, "yolo26n_cbvd", "weights", "best.pt")
        if os.path.isfile(exact):
            return exact
    # Broad recursive fallback
    candidates = glob.glob("runs/detect/**/best.pt", recursive=True)
    if not candidates:
        candidates = glob.glob("**/best.pt", recursive=True)
    return candidates[0] if candidates else None


# ── Step runners ──────────────────────────────────────────────────────

def step_download_dataset(output_dir):
    print("=" * 60)
    print("STEP 1: Downloading dataset")
    print("=" * 60)
    data_root = download_dataset()
    annotations_csv = os.path.join(data_root, "annotations", "ava_train_v2.1.csv")
    frames_dir = os.path.join(data_root, "rawframes_mini")
    return {"data_root": data_root, "annotations_csv": annotations_csv, "frames_dir": frames_dir}


def step_create_yolo_dataset(data_root, output_dir):
    print("\n" + "=" * 60)
    print("STEP 2: Creating YOLO dataset")
    print("=" * 60)
    yolo_data_dir = os.path.join(output_dir, "yolo_dataset")
    create_yolo_dataset(output_dir=yolo_data_dir, data_root=data_root)
    return {"yolo_data_dir": yolo_data_dir}


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
    return {"data_yaml": yaml_path}


def step_train_yolo(data_yaml, output_dir, config):
    print("\n" + "=" * 60)
    print("STEP 4: Training YOLO26n")
    print("=" * 60)
    train_yolo26n(data_yaml=data_yaml)
    yolo_weights = find_best_pt()
    if not yolo_weights:
        raise FileNotFoundError(
            "YOLO training completed but could not find best.pt. "
            "Searched: runs/detect/cow_detector/, cow_detector/, and recursive glob."
        )
    print(f"YOLO weights saved at: {yolo_weights}")
    return {"yolo_weights": yolo_weights}


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


# ── Skip checks ───────────────────────────────────────────────────────

def is_step_done(step, state, output_dir):
    """Check whether a step's output already exists on disk."""
    if step == 1:
        return "data_root" in state and os.path.isdir(state["data_root"])
    if step == 2:
        d = state.get("yolo_data_dir", "")
        return os.path.isdir(os.path.join(d, "train", "images"))
    if step == 3:
        return os.path.isfile(state.get("data_yaml", ""))
    if step == 4:
        return find_best_pt() is not None
    if step == 6:
        return os.path.isfile(os.path.join(output_dir, "flat_vae_model.pth"))
    if step == 7:
        return os.path.isfile(os.path.join(output_dir, "lstm_vae_model.pth"))
    return False


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cow Anomaly Detection — Full Pipeline")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory (default: pipeline_output)")
    parser.add_argument("--from-step", type=int, default=1, choices=range(1, 8),
                        help="Force re-run from this step onwards (1-7), ignoring prior state")
    parser.add_argument("--force", action="store_true", help="Re-run all steps, ignoring prior state")
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    device = get_device()
    config = CONFIG.copy()

    print(f"Device:  {device}")
    print(f"Output:  {os.path.abspath(output_dir)}")

    state = load_state(output_dir)
    force_step = 1 if args.force else args.from_step

    if state:
        print(f"Prior state found: last completed step {state.get('last_step', '?')}")
    if force_step > 1:
        print(f"Forcing re-run from step {force_step}")

    t0 = time.time()

    # Step 1: Download
    if force_step <= 1 and is_step_done(1, state, output_dir):
        print(f"\n[SKIP] Step 1: Download dataset (already at {state['data_root']})")
    else:
        state.update(step_download_dataset(output_dir))
        state["last_step"] = 1
        save_state(output_dir, state)

    # Step 2: YOLO dataset
    if force_step <= 2 and is_step_done(2, state, output_dir):
        print(f"\n[SKIP] Step 2: YOLO dataset (already at {state['yolo_data_dir']})")
    else:
        state.update(step_create_yolo_dataset(state["data_root"], output_dir))
        state["last_step"] = 2
        save_state(output_dir, state)

    # Step 3: YAML
    if force_step <= 3 and is_step_done(3, state, output_dir):
        print(f"\n[SKIP] Step 3: data.yaml (already at {state['data_yaml']})")
    else:
        state.update(step_create_yaml(state["yolo_data_dir"]))
        state["last_step"] = 3
        save_state(output_dir, state)

    # Step 4: YOLO training
    if force_step <= 4 and is_step_done(4, state, output_dir):
        yolo_weights = find_best_pt()
        state["yolo_weights"] = yolo_weights
        print(f"\n[SKIP] Step 4: YOLO training (found {yolo_weights})")
    else:
        state.update(step_train_yolo(state["data_yaml"], output_dir, config))
        state["last_step"] = 4
        save_state(output_dir, state)

    # Step 5: Feature extractor (always runs — in-memory object)
    feature_extractor, hook = step_feature_extractor(state["yolo_weights"], device)

    # Step 6: Flat VAE
    if force_step <= 6 and is_step_done(6, state, output_dir):
        print(f"\n[SKIP] Step 6: flat VAE (found {os.path.join(output_dir, 'flat_vae_model.pth')})")
    else:
        step_flat_vae(
            state["annotations_csv"], state["frames_dir"],
            feature_extractor, hook, device, output_dir, config,
        )
        state["last_step"] = 6
        save_state(output_dir, state)

    # Step 7: LSTM-VAE
    if force_step <= 7 and is_step_done(7, state, output_dir):
        print(f"\n[SKIP] Step 7: LSTM-VAE (found {os.path.join(output_dir, 'lstm_vae_model.pth')})")
    else:
        step_lstm_vae(
            state["annotations_csv"], state["frames_dir"],
            feature_extractor, device, output_dir, config,
        )
        state["last_step"] = 7
        save_state(output_dir, state)

    hook.remove()

    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print(f"Total time: {elapsed / 60:.1f} min")
    print(f"Outputs saved to: {os.path.abspath(output_dir)}")
    print("=" * 60)
    print("\nArtifacts:")
    for f in sorted(os.listdir(output_dir)):
        if f.startswith("."):
            continue
        size = os.path.getsize(os.path.join(output_dir, f))
        print(f"  {f:40s} {size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
