import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, SubsetRandomSampler, TensorDataset

from scripts.config import CONFIG
from scripts.data.build_features import build_feature_dataset
from scripts.data.create_yolo_dataset import create_yolo_dataset
from scripts.data.download_dataset import download_dataset
from scripts.data.feature_cache import open_run_feature_cache
from scripts.data.normalize import MinMaxNormalizer, ZScoreNormalizer
from scripts.data.splits import describe, resolve_indices, resolve_split
from scripts.dataset.sequence_dataset import CowSequenceDataset, NormalisedSeqDataset
from scripts.manifest import RunManifest
from scripts.models.feature_extractor import create_feature_extractor
from scripts.models.lstm_vae import LSTMVAE, train_lstm_vae
from scripts.models.train_yolo_n import train_yolo26n
from scripts.models.vae import VAE, plot_history, train_vae
from scripts.utils.hashing import hash_json, sha256_file
from scripts.utils.history import save_history
from scripts.utils.plotting import setup_matplotlib
from scripts.utils.seeding import set_seed

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

def step_download_dataset(output_dir, manifest=None):
    print("=" * 60)
    print("STEP 1: Downloading dataset")
    print("=" * 60)
    data_root = download_dataset()
    annotations_csv = os.path.join(data_root, "annotations", "ava_train_v2.1.csv")
    frames_dir = os.path.join(data_root, "rawframes_mini")
    if manifest is not None:
        manifest.record("data_root", data_root)
        manifest.record_artifact("annotations_csv", annotations_csv)
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


def step_train_yolo(data_yaml, output_dir, config, manifest=None):
    print("\n" + "=" * 60)
    print("STEP 4: Training YOLO26n")
    print("=" * 60)
    train_yolo26n(data_yaml=data_yaml, seed=config["random_seed"])
    yolo_weights = find_best_pt()
    if not yolo_weights:
        raise FileNotFoundError(
            "YOLO training completed but could not find best.pt. "
            "Searched: runs/detect/cow_detector/, cow_detector/, and recursive glob."
        )
    print(f"YOLO weights saved at: {yolo_weights}")
    if manifest is not None:
        # Every downstream feature is a function of these weights; the cache
        # directory is named after this hash, so record it once here.
        manifest.record_artifact("yolo_weights", yolo_weights)
        manifest.record_step(4, config_hash=hash_json(config), weights_path=yolo_weights)
    return {"yolo_weights": yolo_weights}


def step_feature_extractor(yolo_weights, device, layer_index=9):
    print("\n" + "=" * 60)
    print("STEP 5: Creating YOLO feature extractor")
    print("=" * 60)
    feature_extractor, hook = create_feature_extractor(
        yolo_weights, layer_index=layer_index, device=device
    )
    print(f"Feature extractor ready (layer {layer_index})")
    return feature_extractor, hook


def step_flat_vae(annotations_csv, frames_dir, feature_extractor, hook, device, output_dir, config, feature_cache=None, headless=None, manifest=None):
    print("\n" + "=" * 60)
    print("STEP 6: Training flat VAE")
    print("=" * 60)
    df = pd.read_csv(annotations_csv, header=None, dtype={0: str})
    df.columns = ["video_id", "timestamp", "x1", "y1", "x2", "y2", "action_id", "target_id"]

    print("Extracting frame-level features...")
    features, feature_video_ids = build_feature_dataset(
        df, frames_dir, feature_extractor,
        normal_action_ids=config["normal_action_ids"], device=device,
        feature_cache=feature_cache,
    )
    print(f"Extracted {features.shape[0]} features, dim={features.shape[1]}")

    # The canonical split (1.3, 1.4). Read from the run's split manifest if a stage
    # already drew it, so both models train on the same partition; drawn from `df`
    # and persisted if this is the first stage to need it. Grouped, so consecutive
    # frames of one cow cannot straddle the split.
    assignment = resolve_split(
        output_dir, df,
        val_split=config["val_split"], test_split=config.get("test_split", 0.0),
        random_seed=config["random_seed"], group_key=config["split_group_key"],
    )
    print(
        f"Split by {config['split_group_key']}: "
        + ", ".join(f"{name}={len(assignment[name])}" for name in ("train", "val", "test"))
    )
    row_split = resolve_indices(assignment, feature_video_ids)
    train_idx, val_idx = row_split["train"], row_split["val"]

    # Fit the normalizer on the training rows only (1.1). Fitting on `features`
    # normalized every val sample using its own extremes, which is the leak
    # `1.8` asserts against. Val features outside the train range now map outside
    # [0, 1] rather than being rescaled to hide it.
    normalizer = MinMaxNormalizer().fit(features[train_idx])
    features_norm = normalizer.transform(features)
    X_train, X_val = features_norm[train_idx], features_norm[val_idx]
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32)),
        batch_size=config["vae_batch_size"], shuffle=True,
        generator=torch.Generator().manual_seed(config["random_seed"]),
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.float32)),
        batch_size=config["vae_batch_size"], shuffle=False,
    )

    vae = VAE(features.shape[1], hidden_dim=config["vae_hidden_dim"], latent_dim=config["vae_latent_dim"])
    history = train_vae(vae, train_loader, val_loader, epochs=config["vae_epochs"], lr=config["vae_lr"], device=device)

    save_history(history, os.path.join(output_dir, "flat_vae_history.csv"))
    torch.save(vae.state_dict(), os.path.join(output_dir, "flat_vae_model.pth"))
    np.save(os.path.join(output_dir, "flat_vae_feature_min.npy"), normalizer.min_)
    np.save(os.path.join(output_dir, "flat_vae_feature_max.npy"), normalizer.max_)
    plot_history(history, save_path=os.path.join(output_dir, "flat_vae_history.png"), headless=headless)
    if manifest is not None:
        manifest.record_array("flat_vae_features", features, n=features.shape[0], dim=features.shape[1])
        manifest.record_array("flat_vae_features_norm", features_norm)
        manifest.record_array("flat_vae_train_rows", features_norm[train_idx])
        manifest.record_array("flat_vae_val_rows", features_norm[val_idx])
        manifest.record_array("flat_vae_feature_min", normalizer.min_)
        manifest.record_array("flat_vae_feature_max", normalizer.max_)
        manifest.record_split("flat_vae_rows", {"train": sorted(train_idx.tolist()), "val": sorted(val_idx.tolist())})
        # The canonical manifest, recorded again under the name every stage uses, so
        # a resumed run can prove both models read the same partition rather than
        # merely having written files into the same directory.
        manifest.record_split("canonical_groups", {
            name: sorted(map(str, assignment[name])) for name in ("train", "val", "test")
        })
        manifest.record_step(
            6,
            config_hash=hash_json(config),
            split_group_key=config["split_group_key"],
            split_sha256=describe(assignment)["sha256"],
            n_features=int(features.shape[0]),
            n_train=int(X_train.shape[0]),
            n_val=int(X_val.shape[0]),
            normalizer_fit_on="train",
            weights_sha256=(manifest.data.get("artifacts", {}).get("yolo_weights", {}) or {}).get("sha256"),
        )
    print("Flat VAE complete.")


def step_lstm_vae(annotations_csv, frames_dir, feature_extractor, device, output_dir, config, feature_cache=None, manifest=None):
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
        feature_cache=feature_cache,
        seq_len=config["seq_len"],
        stride=config["seq_stride"],
        normal_action_ids=config["normal_action_ids"],
        device=device,
        seed=config["random_seed"],
    )
    print(f"Total sequences: {len(seq_dataset)}")

    # The same canonical split the flat VAE read (1.4). It used to be redrawn here
    # from `set(...)`, whose iteration order is not stable across processes, so the
    # two models were trained on different partitions of the same videos and no
    # Flat-vs-LSTM comparison meant anything.
    assignment = resolve_split(
        output_dir, df,
        val_split=config["val_split"], test_split=config.get("test_split", 0.0),
        random_seed=config["random_seed"], group_key=config["split_group_key"],
    )
    print(
        f"Split by {config['split_group_key']}: "
        + ", ".join(f"{name}={len(assignment[name])}" for name in ("train", "val", "test"))
    )
    # Tracks are keyed `(video_id, target_id)`, so which half of the key is the split
    # unit depends on the configured group key. `target_id` is the stricter unit: a
    # cow cannot reach val through a different video.
    track_key_index = 0 if config["split_group_key"] == "video_id" else 1
    sequence_groups = [str(key[track_key_index]) for key, _ in seq_dataset.sequences]
    sequence_split = resolve_indices(assignment, sequence_groups)
    train_indices = sequence_split["train"].tolist()
    val_indices = sequence_split["val"].tolist()

    # Compute the statistics from the training sequences only (1.2). Fitting mean/std
    # over every sequence in the run let each val window contribute to the numbers it
    # was then normalized by.
    print("Computing normalisation statistics from the training sequences...")
    normalizer = ZScoreNormalizer().fit(seq_dataset.stack(train_indices))
    mean, std = normalizer.mean_, normalizer.std_

    norm_dataset = NormalisedSeqDataset(
        mean=mean,
        std=std,
        df=df,
        frames_dir=frames_dir,
        feature_extractor=feature_extractor,
        feature_cache=feature_cache,
        seq_len=config["seq_len"],
        stride=config["seq_stride"],
        normal_action_ids=config["normal_action_ids"],
        device=device,
        seed=config["random_seed"],
    )

    train_loader = DataLoader(norm_dataset, batch_size=config["lstm_vae_batch_size"], sampler=SubsetRandomSampler(train_indices, generator=torch.Generator().manual_seed(config["random_seed"])))
    val_loader = DataLoader(norm_dataset, batch_size=config["lstm_vae_batch_size"], sampler=SubsetRandomSampler(val_indices, generator=torch.Generator().manual_seed(config["random_seed"])))

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
    save_history(history, os.path.join(output_dir, "lstm_vae_history.csv"))
    np.save(os.path.join(output_dir, "lstm_vae_feature_mean.npy"), mean)
    np.save(os.path.join(output_dir, "lstm_vae_feature_std.npy"), std)
    if manifest is not None:
        # Same name and same content as step 6's record, so a mismatch between the
        # two is a manifest diff rather than something a reader has to notice.
        manifest.record_split("canonical_groups", {
            name: sorted(map(str, assignment[name])) for name in ("train", "val", "test")
        })
        manifest.record_split("lstm_vae_groups", {
            name: sorted({sequence_groups[i] for i in sequence_split[name]})
            for name in ("train", "val", "test")
        })
        manifest.record_array("lstm_vae_feature_mean", mean)
        manifest.record_array("lstm_vae_feature_std", std)
        manifest.record_step(
            7,
            config_hash=hash_json(config),
            split_group_key=config["split_group_key"],
            split_sha256=describe(assignment)["sha256"],
            n_sequences=len(seq_dataset),
            n_train=len(train_indices),
            n_val=len(val_indices),
            n_videos=len(set(sequence_groups)),
            normalizer_fit_on="train",
            weights_sha256=(manifest.data.get("artifacts", {}).get("yolo_weights", {}) or {}).get("sha256"),
        )
    print("LSTM-VAE complete.")


# ── Skip checks ───────────────────────────────────────────────────────

def is_step_done(step, state, output_dir, config=None, manifest=None):
    """Check whether a step's output already exists on disk.

    Existence alone is not enough for the training steps: a checkpoint written
    before a config change is stale but still on disk, and reusing it reports a
    number that was never produced by the current config (task 3.6.3). When a
    manifest is available, require the recorded config hash to still match.
    """
    if step == 1:
        return "data_root" in state and os.path.isdir(state["data_root"])
    if step == 2:
        d = state.get("yolo_data_dir", "")
        return os.path.isdir(os.path.join(d, "train", "images"))
    if step == 3:
        return os.path.isfile(state.get("data_yaml", ""))
    if step == 4:
        found = find_best_pt()
        if not found:
            return False
        if manifest is not None:
            recorded = (manifest.data.get("artifacts", {}).get("yolo_weights") or {}).get("sha256")
            if recorded and sha256_file(found) != recorded:
                print(f"  {found} is not the weights this run trained (hash mismatch)")
                return False
        return True
    if step == 6:
        return _checkpoint_is_current(os.path.join(output_dir, "flat_vae_model.pth"), "flat VAE", 6, config, manifest)
    if step == 7:
        return _checkpoint_is_current(os.path.join(output_dir, "lstm_vae_model.pth"), "LSTM-VAE", 7, config, manifest)
    return False


def _checkpoint_is_current(path, label, step, config, manifest):
    """Is the checkpoint on disk the one this config would have produced?"""
    if not os.path.isfile(path):
        return False
    if config is None or manifest is None:
        return True
    if not manifest.is_current_for(config, step=step):
        if manifest.step(step) is None:
            print(
                f"  {path} has no manifest entry, so it cannot be verified against the "
                f"current config — retraining the {label} (3.6.3)"
            )
        else:
            print(f"  config changed since the {label} was trained — its checkpoint is stale")
        return False
    return True


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cow Anomaly Detection — Full Pipeline")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory (default: pipeline_output)")
    parser.add_argument("--from-step", type=int, default=1, choices=range(1, 8),
                        help="Force re-run from this step onwards (1-7), ignoring prior state")
    parser.add_argument("--force", action="store_true", help="Re-run all steps, ignoring prior state")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override CONFIG['random_seed']")
    parser.add_argument("--headless", action="store_true",
                        help="Force the non-interactive matplotlib backend (no plot window)")
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    device = get_device()
    config = CONFIG.copy()
    if args.seed is not None:
        config["random_seed"] = args.seed
    seed = set_seed(config["random_seed"])
    headless = setup_matplotlib(True if args.headless else None)

    print(f"Device:  {device}")
    print(f"Seed:    {seed}")
    print(f"Plots:   {headless}")
    print(f"Output:  {os.path.abspath(output_dir)}")

    state = load_state(output_dir)
    force_step = 1 if args.force else args.from_step

    # Reuse the manifest already in this output directory: its per-step config
    # hashes are what distinguish a resumable step from a stale one. A fresh
    # manifest would make every checkpoint look unverifiable and re-train all of them.
    manifest = RunManifest.load(output_dir) or RunManifest(output_dir)
    manifest.begin_run(
        config, seed=seed, device=device, command=sys.argv,
        repo_root=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    manifest.record("forced_from_step", force_step)
    manifest.save()

    if state:
        print(f"Prior state found: last completed step {state.get('last_step', '?')}")
    if force_step > 1:
        print(f"Forcing re-run from step {force_step}")

    t0 = time.time()

    # Step 1: Download
    if force_step <= 1 and is_step_done(1, state, output_dir):
        print(f"\n[SKIP] Step 1: Download dataset (already at {state['data_root']})")
    else:
        state.update(step_download_dataset(output_dir, manifest=manifest))
        state["last_step"] = 1
        save_state(output_dir, state)
        manifest.record_step(1, config_hash=hash_json(config), data_root=state["data_root"])
        manifest.save()

    # Step 2: YOLO dataset
    if force_step <= 2 and is_step_done(2, state, output_dir):
        print(f"\n[SKIP] Step 2: YOLO dataset (already at {state['yolo_data_dir']})")
    else:
        state.update(step_create_yolo_dataset(state["data_root"], output_dir))
        state["last_step"] = 2
        save_state(output_dir, state)
        manifest.record_step(2, config_hash=hash_json(config), yolo_data_dir=state["yolo_data_dir"])
        manifest.save()

    # Step 3: YAML
    if force_step <= 3 and is_step_done(3, state, output_dir):
        print(f"\n[SKIP] Step 3: data.yaml (already at {state['data_yaml']})")
    else:
        state.update(step_create_yaml(state["yolo_data_dir"]))
        state["last_step"] = 3
        save_state(output_dir, state)
        manifest.record_step(3, config_hash=hash_json(config), data_yaml=state["data_yaml"])
        manifest.save()

    # Step 4: YOLO training
    if force_step <= 4 and is_step_done(4, state, output_dir, config, manifest):
        yolo_weights = find_best_pt()
        state["yolo_weights"] = yolo_weights
        print(f"\n[SKIP] Step 4: YOLO training (found {yolo_weights})")
    else:
        state.update(step_train_yolo(state["data_yaml"], output_dir, config, manifest=manifest))
        state["last_step"] = 4
        save_state(output_dir, state)
        manifest.save()

    # Step 5: Feature extractor (always runs — in-memory object)
    feature_extractor, hook = step_feature_extractor(
        state["yolo_weights"], device, layer_index=config["feature_layer"]
    )
    manifest.record(
        "feature_extractor",
        {"layer": config["feature_layer"], "input_size": config["feature_input_size"],
         "weights": state["yolo_weights"]},
    )

    # One cache shared by both VAE stages, keyed by the weights hash, so each crop
    # is extracted once per run no matter how many datasets are built over it.
    feature_cache = open_run_feature_cache(
        output_dir, state["yolo_weights"], layer_index=config["feature_layer"],
        input_size=config["feature_input_size"],
    )

    # Step 6: Flat VAE
    if force_step <= 6 and is_step_done(6, state, output_dir, config, manifest):
        print(f"\n[SKIP] Step 6: flat VAE (found {os.path.join(output_dir, 'flat_vae_model.pth')})")
    else:
        step_flat_vae(
            state["annotations_csv"], state["frames_dir"],
            feature_extractor, hook, device, output_dir, config,
            feature_cache=feature_cache, headless=headless, manifest=manifest,
        )
        state["last_step"] = 6
        save_state(output_dir, state)
        manifest.save()

    # Step 7: LSTM-VAE
    if force_step <= 7 and is_step_done(7, state, output_dir, config, manifest):
        print(f"\n[SKIP] Step 7: LSTM-VAE (found {os.path.join(output_dir, 'lstm_vae_model.pth')})")
    else:
        step_lstm_vae(
            state["annotations_csv"], state["frames_dir"],
            feature_extractor, device, output_dir, config,
            feature_cache=feature_cache, manifest=manifest,
        )
        state["last_step"] = 7
        save_state(output_dir, state)
        manifest.save()

    hook.remove()

    elapsed = time.time() - t0
    manifest.record("total_seconds", round(elapsed, 3))
    manifest.save()
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
