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
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler, TensorDataset

from scripts.checkpoint import (
    data_fingerprint,
    latest_epoch_checkpoint,
    record_and_prune_checkpoints,
)
from scripts.config import CONFIG
from scripts.data.build_features import build_feature_dataset
from scripts.data.create_yolo_dataset import create_yolo_dataset
from scripts.data.download_dataset import download_dataset
from scripts.data.feature_cache import open_run_feature_cache
from scripts.data.normalize import MinMaxNormalizer
from scripts.data.splits import describe, resolve_indices, resolve_split, split_params
from scripts.dataset.sequence_dataset import CowSequenceDataset, NormalisedSeqDataset
from scripts.evaluation.reconstruction import (
    reconstruction_score_report,
    row_reconstruction_errors,
    window_reconstruction_errors,
)
from scripts.manifest import RunManifest
from scripts.models.feature_extractor import create_feature_extractor
from scripts.models.lstm_vae import LSTMVAE, train_lstm_vae
from scripts.models.train_yolo_n import train_yolo26n
from scripts.models.training import EarlyStopping, count_parameters, match_parameter_budget
from scripts.models.vae import VAE, plot_history, train_vae
from scripts.utils.hashing import hash_json, sha256_file
from scripts.utils.history import save_history
from scripts.utils.io import atomic_torch_save, atomic_write_json
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


def _save_best_checkpoint(checkpoint, path, label):
    """Write the best-val checkpoint a trainer reported, and say which epoch it was (2.7)."""
    best_state = (checkpoint or {}).get("best_state")
    if best_state is None:
        return False
    atomic_torch_save(path, best_state)
    print(
        f"  {label}: best val epoch {checkpoint.get('best_epoch')} "
        f"({checkpoint.get('best_metric'):.4f}); final epoch {checkpoint.get('final_epoch')}"
    )
    return True


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


def find_last_pt():
    """YOLO's rolling `last.pt` — the resume point for an interrupted run (3.3.2).

    Ultralytics writes it every epoch, so it is what makes a 150-epoch run
    interrupted at epoch 140 continue at 141 instead of restarting at 1.
    """
    for project_dir in ["runs/detect/cow_detector", "cow_detector"]:
        exact = os.path.join(project_dir, "yolo26n_cbvd", "weights", "last.pt")
        if os.path.isfile(exact):
            return exact
    candidates = glob.glob("runs/detect/**/last.pt", recursive=True)
    if not candidates:
        candidates = glob.glob("**/last.pt", recursive=True)
    return candidates[0] if candidates else None


def yolo_resume_point(data_yaml, config, last_pt=None):
    """The `last.pt` to resume YOLO from, or None (3.3.2).

    Ultralytics records the run's arguments in `args.yaml` beside `weights/`.
    Continuing a run whose epochs, seed or data differ would silently train a
    different experiment on the old run's optimiser state, so anything other than
    an exact match — including a missing or unreadable args file — means start fresh.
    """
    last_pt = last_pt or find_last_pt()
    if last_pt is None:
        return None
    args_path = os.path.join(os.path.dirname(os.path.dirname(last_pt)), "args.yaml")
    try:
        import yaml

        with open(args_path) as handle:
            args = yaml.safe_load(handle) or {}
        matches = (
            int(args.get("epochs", -1)) == int(config["yolo"]["epochs"])
            and int(args.get("seed", -1)) == int(config["random_seed"])
            and str(args.get("data", "")) == str(data_yaml)
        )
    except (OSError, ImportError, TypeError, ValueError, AttributeError):
        return None
    return last_pt if matches else None


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


def step_create_yolo_dataset(data_root, output_dir, config):
    print("\n" + "=" * 60)
    print("STEP 2: Creating YOLO dataset")
    print("=" * 60)
    yolo_data_dir = os.path.join(output_dir, "yolo_dataset")
    # The detector's split is the canonical manifest (1.6), written into the
    # run's output directory so steps 6 and 7 adopt exactly this partition
    # instead of resolving a second one. The config travels with it (1.7): the
    # seed and fractions it is drawn under are the run's, not module defaults.
    assignment = create_yolo_dataset(
        output_dir=yolo_data_dir, data_root=data_root,
        split_dir=output_dir, config=config,
    )
    return {
        "yolo_data_dir": yolo_data_dir,
        "yolo_split_sha256": describe(assignment)["sha256"],
    }


def _expected_data_yaml(yolo_data_dir):
    """The `data.yaml` for this dataset — written by step 3, compared by `is_step_done(3)`.

    The `test` key appears only when the test split actually has images: a yaml
    pointing at an empty directory is a claim about data that is not there, and
    a yaml written before 1.5 gave the run a test split must not count as done
    (it would silently leave the detector with no test partition on disk).
    """
    content = {
        "path": os.path.abspath(yolo_data_dir),
        "train": "train/images",
        "val": "val/images",
        "names": {0: "cow"},
    }
    test_images = os.path.join(yolo_data_dir, "test", "images")
    if os.path.isdir(test_images) and any(name.endswith(".jpg") for name in os.listdir(test_images)):
        content["test"] = "test/images"
    return content


def step_create_yaml(yolo_data_dir):
    print("\n" + "=" * 60)
    print("STEP 3: Creating YOLO data YAML")
    print("=" * 60)
    yaml_path = os.path.join(yolo_data_dir, "data.yaml")
    content = _expected_data_yaml(yolo_data_dir)
    with open(yaml_path, "w") as f:
        yaml.dump(content, f, default_flow_style=False)
    print(f"Written: {yaml_path}")
    return {"data_yaml": yaml_path}


def step_train_yolo(data_yaml, output_dir, config, manifest=None, resume=True):
    print("\n" + "=" * 60)
    print("STEP 4: Training YOLO26n")
    print("=" * 60)
    # Resume from Ultralytics' `last.pt` only when the interrupted run belongs to the
    # current config (3.3.2): a fresh run with no prior checkpoint behaves as before,
    # `--force` passes resume=False, and a mismatched config starts over rather than
    # continuing a different experiment.
    resume_from = yolo_resume_point(data_yaml, config) if resume else None
    if resume_from is not None:
        print(f"  resuming YOLO from {resume_from} (3.3.2)")
    train_yolo26n(data_yaml=data_yaml, seed=config["random_seed"], resume_from=resume_from)
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


def step_flat_vae(annotations_csv, frames_dir, feature_extractor, hook, device, output_dir, config, feature_cache=None, headless=None, manifest=None, resume=True):
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

    # The canonical split (1.3, 1.4, 1.5). Read from the run's split manifest if a
    # stage already drew it — including step 2, which now draws it for the detector
    # too — so every model trains on the same partition. Grouped, so consecutive
    # frames of one cow cannot straddle the split.
    assignment = resolve_split(output_dir, df, **split_params(config))
    print(
        f"Split by {config['split_group_key']}: "
        + ", ".join(f"{name}={len(assignment[name])}" for name in ("train", "val", "test"))
    )
    row_split = resolve_indices(assignment, feature_video_ids)
    train_idx, val_idx, test_idx = row_split["train"], row_split["val"], row_split["test"]
    if config["test_split"] > 0 and len(test_idx) == 0:
        raise RuntimeError(
            "test_split > 0 but no feature rows landed in the test split: those videos "
            "produced no normal-action crops, so the held-out number would be reported "
            "from an empty set. An empty coverage cell must be fixed, not summarized (1.5)."
        )

    # Fit the normalizer on the training rows only (1.1). Fitting on `features`
    # normalized every val sample using its own extremes, which is the leak
    # `1.8` asserts against. Val features outside the train range now map outside
    # [0, 1] rather than being rescaled to hide it.
    normalizer = MinMaxNormalizer().fit(features[train_idx])
    features_norm = normalizer.transform(features)
    X_train, X_val = features_norm[train_idx], features_norm[val_idx]
    # No private `generator`: the loop snapshots and restores the *global* RNG so a
    # resumed run reproduces the uninterrupted curve (3.3.3), and it can only do
    # that for a shuffle drawn from the global stream.
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32)),
        batch_size=config["vae_batch_size"], shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.float32)),
        batch_size=config["vae_batch_size"], shuffle=False,
    )

    vae = VAE(features.shape[1], hidden_dim=config["vae_hidden_dim"], latent_dim=config["vae_latent_dim"])
    checkpoint = {}
    # Resume an interrupted step 6 from its newest epoch checkpoint rather than
    # restarting at epoch 1 (3.3.2). `latest_epoch_checkpoint` returns None if the
    # checkpoints were written under a different config, so a changed run retrains.
    resume_from = latest_epoch_checkpoint(output_dir, "flat_vae", config=config) if resume else None
    history = train_vae(
        vae, train_loader, val_loader,
        epochs=config["vae_epochs"], lr=config["vae_lr"], device=device,
        beta=config["vae_beta"], kl_warmup_epochs=config["vae_kl_warmup_epochs"],
        checkpoint=checkpoint,
        lr_scheduler=config["vae_lr_scheduler"],
        lr_scheduler_factor=config["vae_lr_scheduler_factor"],
        lr_scheduler_patience=config["vae_lr_scheduler_patience"],
        grad_clip=config["vae_grad_clip"],
        early_stopping=EarlyStopping(patience=config["vae_early_stopping_patience"]),
        seed=config["random_seed"], config=config,
        data_fingerprint=data_fingerprint(manifest=manifest),
        checkpoint_dir=output_dir, checkpoint_prefix="flat_vae",
        keep_last=config["checkpoint_keep_last"], resume_from=resume_from,
    )

    save_history(history, os.path.join(output_dir, "flat_vae_history.csv"))
    # Both checkpoints are written and which one the scores below come from is
    # recorded. `flat_vae_model.pth` stays the final-epoch weights so existing
    # artifacts keep their meaning; `flat_vae_best.pth` is the best-val epoch (2.7).
    flat_final = os.path.join(output_dir, "flat_vae_model.pth")
    flat_best = os.path.join(output_dir, "flat_vae_best.pth")
    atomic_torch_save(flat_final, vae.state_dict())
    wrote_best = _save_best_checkpoint(checkpoint, flat_best, "flat VAE")
    # Checksum both and enforce retention (3.2.2, 3.2.3): the manifest binds each
    # checkpoint to its content, and any epoch checkpoints beyond the last
    # `checkpoint_keep_last` are dropped so a Phase 4 grid cannot fill the disk.
    record_and_prune_checkpoints(
        manifest=manifest, directory=output_dir, prefix="flat_vae",
        keep_last=config["checkpoint_keep_last"],
        final=flat_final, best=flat_best if wrote_best else None, step=6,
    )
    print(f"  flat VAE parameters: {count_parameters(vae):,} (recorded for the 2.11 comparison)")
    np.save(os.path.join(output_dir, "flat_vae_feature_min.npy"), normalizer.min_)
    np.save(os.path.join(output_dir, "flat_vae_feature_max.npy"), normalizer.max_)

    # Score all three splits (1.5). This is the only place test rows are read:
    # they are transformed with the *train* min/max and fed to the trained model
    # once, after training — never to `fit`, never to a loader. The report is the
    # three-way protocol's output: train (seen), val (monitored), test (held out).
    errors = {
        "train": row_reconstruction_errors(vae, X_train, device),
        "val": row_reconstruction_errors(vae, X_val, device),
        "test": row_reconstruction_errors(vae, features_norm[test_idx], device),
    }
    scores_path = os.path.join(output_dir, "flat_vae_split_scores.json")
    atomic_write_json(scores_path, reconstruction_score_report("flat_vae", errors))

    plot_history(history, save_path=os.path.join(output_dir, "flat_vae_history.png"), headless=headless)
    if manifest is not None:
        manifest.record_array("flat_vae_features", features, n=features.shape[0], dim=features.shape[1])
        manifest.record_array("flat_vae_features_norm", features_norm)
        manifest.record_array("flat_vae_train_rows", features_norm[train_idx])
        manifest.record_array("flat_vae_val_rows", features_norm[val_idx])
        manifest.record_array("flat_vae_feature_min", normalizer.min_)
        manifest.record_array("flat_vae_feature_max", normalizer.max_)
        manifest.record_split("flat_vae_rows", {
            "train": sorted(train_idx.tolist()),
            "val": sorted(val_idx.tolist()),
            "test": sorted(test_idx.tolist()),
        })
        manifest.record_artifact("flat_vae_split_scores", scores_path)
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
            n_test=int(test_idx.shape[0]),
            input_dim=int(features.shape[1]),
            n_parameters=count_parameters(vae),
            normalizer_fit_on="train",
            # 2.7: which epoch each checkpoint is, and which one the split scores use.
            best_epoch=checkpoint.get("best_epoch"),
            best_metric=checkpoint.get("best_metric"),
            final_epoch=checkpoint.get("final_epoch"),
            scored_checkpoint="final",
            weights_sha256=(manifest.data.get("artifacts", {}).get("yolo_weights", {}) or {}).get("sha256"),
        )
    print("Flat VAE complete.")


def step_lstm_vae(annotations_csv, frames_dir, feature_extractor, device, output_dir, config, feature_cache=None, manifest=None, resume=True):
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

    # The same canonical split every other stage read, including step 2 for the
    # detector (1.4, 1.6). It used to be redrawn here from `set(...)`, whose
    # iteration order is not stable across processes, so the two models were
    # trained on different partitions of the same videos and no Flat-vs-LSTM
    # comparison meant anything.
    assignment = resolve_split(output_dir, df, **split_params(config))
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
    test_indices = sequence_split["test"].tolist()
    if config["test_split"] > 0 and not test_indices:
        raise RuntimeError(
            "test_split > 0 but no sequences fall in the test split: those videos "
            "produced no usable windows, so the held-out number would be reported "
            "from an empty set. An empty coverage cell must be fixed, not summarized (1.5)."
        )

    # Compute the statistics from the training sequences only (1.2). Fitting min/max
    # over every sequence in the run let each val window contribute to the numbers it
    # was then normalized by. Same min-max scaler as the flat VAE (2.4), so both
    # models reconstruct [0, 1] features and their errors are on one scale.
    print("Computing normalisation statistics from the training sequences...")
    normalizer = MinMaxNormalizer().fit(seq_dataset.stack(train_indices))

    norm_dataset = NormalisedSeqDataset(
        normalizer=normalizer,
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

    train_loader = DataLoader(norm_dataset, batch_size=config["lstm_vae_batch_size"], sampler=SubsetRandomSampler(train_indices))
    # Validation is a full, ordered pass (2.9): a shuffled sampler made the val
    # batches differ every epoch, so `val_loss` depended on the draw as well as the
    # weights. `Subset` covers every val window exactly once, in split order.
    val_loader = DataLoader(Subset(norm_dataset, val_indices), batch_size=config["lstm_vae_batch_size"], shuffle=False)

    # `input_dim` is read from the extracted features rather than hardcoded to 256
    # (2.12): a different backbone or hooked layer produces a different width, and
    # a stale constant would fail at the first `nn.Linear` instead of adapting.
    input_dim = seq_dataset.feature_dim
    # The flat VAE's parameter count is the capacity the LSTM-VAE is measured
    # against (2.11). Derived from the same features and config step 6 used, so it
    # does not depend on step 6 having run in this invocation.
    flat_parameters = count_parameters(
        VAE(input_dim, hidden_dim=config["vae_hidden_dim"], latent_dim=config["vae_latent_dim"])
    )
    parameter_matched = bool(config["lstm_vae_match_flat_parameters"])
    lstm_hidden_dim = config["lstm_vae_hidden_dim"]
    if parameter_matched:
        lstm_hidden_dim, vae, _ = match_parameter_budget(
            lambda width: LSTMVAE(
                input_dim=input_dim, hidden_dim=width,
                latent_dim=config["lstm_vae_latent_dim"],
                num_layers=config["lstm_vae_num_layers"],
            ),
            flat_parameters,
        )
        print(
            f"  parameter match: LSTM hidden {config['lstm_vae_hidden_dim']} -> {lstm_hidden_dim} "
            f"targeting the flat VAE's {flat_parameters:,} params"
        )
    else:
        vae = LSTMVAE(
            input_dim=input_dim,
            hidden_dim=lstm_hidden_dim,
            latent_dim=config["lstm_vae_latent_dim"],
            num_layers=config["lstm_vae_num_layers"],
        )
    checkpoint = {}

    # Resume an interrupted step 7 from its newest epoch checkpoint (3.3.2).
    resume_from = latest_epoch_checkpoint(output_dir, "lstm_vae", config=config) if resume else None
    history = train_lstm_vae(
        vae, train_loader, val_loader,
        epochs=config["lstm_vae_epochs"], lr=config["lstm_vae_lr"], device=device,
        beta=config["lstm_vae_beta"], kl_warmup_epochs=config["lstm_vae_kl_warmup_epochs"],
        checkpoint=checkpoint,
        lr_scheduler=config["lstm_vae_lr_scheduler"],
        lr_scheduler_factor=config["lstm_vae_lr_scheduler_factor"],
        lr_scheduler_patience=config["lstm_vae_lr_scheduler_patience"],
        grad_clip=config["lstm_vae_grad_clip"],
        early_stopping=EarlyStopping(patience=config["lstm_vae_early_stopping_patience"]),
        seed=config["random_seed"], config=config,
        data_fingerprint=data_fingerprint(manifest=manifest),
        checkpoint_dir=output_dir, checkpoint_prefix="lstm_vae",
        keep_last=config["checkpoint_keep_last"], resume_from=resume_from,
    )

    # Final-epoch weights keep the existing artifact's meaning; the best-val epoch
    # is written alongside and named in the manifest (2.7).
    lstm_final = os.path.join(output_dir, "lstm_vae_model.pth")
    lstm_best = os.path.join(output_dir, "lstm_vae_best.pth")
    atomic_torch_save(lstm_final, vae.state_dict())
    wrote_best = _save_best_checkpoint(checkpoint, lstm_best, "LSTM-VAE")
    # Same checksum + retention treatment as the flat VAE (3.2.2, 3.2.3), scoped
    # to the `lstm_vae` prefix so the two models cannot prune each other.
    record_and_prune_checkpoints(
        manifest=manifest, directory=output_dir, prefix="lstm_vae",
        keep_last=config["checkpoint_keep_last"],
        final=lstm_final, best=lstm_best if wrote_best else None, step=7,
    )
    # 2.11: the LSTM-VAE is ~2.9x the flat VAE, so an unqualified win would be a
    # size win. The ratio is printed and recorded next to the flat count, and the
    # width actually used is recorded so a matched run is distinguishable from a
    # default-sized one.
    n_parameters = count_parameters(vae)
    ratio = n_parameters / flat_parameters
    match_note = "parameter-matched to the flat VAE" if parameter_matched else "default width"
    print(f"  LSTM-VAE parameters: {n_parameters:,} ({ratio:.2f}x the flat VAE's "
          f"{flat_parameters:,}; {match_note})")
    save_history(history, os.path.join(output_dir, "lstm_vae_history.csv"))
    np.save(os.path.join(output_dir, "lstm_vae_feature_min.npy"), normalizer.min_)
    np.save(os.path.join(output_dir, "lstm_vae_feature_max.npy"), normalizer.max_)

    # Score all three splits (1.5). `norm_dataset.stack` applies exactly the
    # train-fitted mean/std the loaders saw, and this is the only read of the
    # test windows: after training, once, with nothing fitted on them.
    errors = {
        name: window_reconstruction_errors(
            vae, norm_dataset.stack(indices) if indices else [], device
        )
        for name, indices in (
            ("train", train_indices),
            ("val", val_indices),
            ("test", test_indices),
        )
    }
    scores_path = os.path.join(output_dir, "lstm_vae_split_scores.json")
    atomic_write_json(scores_path, reconstruction_score_report("lstm_vae", errors))

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
        manifest.record_array("lstm_vae_feature_min", normalizer.min_)
        manifest.record_array("lstm_vae_feature_max", normalizer.max_)
        manifest.record_artifact("lstm_vae_split_scores", scores_path)
        manifest.record_step(
            7,
            config_hash=hash_json(config),
            split_group_key=config["split_group_key"],
            split_sha256=describe(assignment)["sha256"],
            n_sequences=len(seq_dataset),
            n_train=len(train_indices),
            n_val=len(val_indices),
            n_test=len(test_indices),
            n_videos=len(set(sequence_groups)),
            input_dim=int(input_dim),
            n_parameters=n_parameters,
            flat_n_parameters=flat_parameters,
            parameter_ratio_vs_flat=ratio,
            parameter_matched=parameter_matched,
            lstm_vae_hidden_dim=int(lstm_hidden_dim),
            normalizer_fit_on="train",
            # 2.7: checkpoint identity and which one the split scores use.
            best_epoch=checkpoint.get("best_epoch"),
            best_metric=checkpoint.get("best_metric"),
            final_epoch=checkpoint.get("final_epoch"),
            scored_checkpoint="final",
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
        if not os.path.isdir(os.path.join(d, "train", "images")):
            return False
        if config is None or manifest is None:
            return True
        # The dataset's *content* is its split (1.6), so the recorded split
        # params — not the directory's existence — decide whether it is current.
        # An existence check would hand a test_split=0.1 run the dataset drawn
        # under test_split=0, and the two partitions would then disagree while
        # both looking perfectly healthy.
        recorded = (manifest.step(2) or {}).get("split_params")
        if recorded != split_params(config):
            print(
                "  the YOLO dataset on disk was built under a different split than the "
                "current config asks for — rebuilding it (1.6)"
            )
            return False
        return True
    if step == 3:
        path = state.get("data_yaml", "")
        if not os.path.isfile(path):
            return False
        try:
            with open(path) as handle:
                written = yaml.safe_load(handle)
        except (OSError, yaml.YAMLError):
            return False
        # The yaml names the split directories, so it is stale the moment the
        # dataset gains or loses a test split (1.5). Compared as content rather
        # than by existence, for the same reason step 2 records its split.
        expected = _expected_data_yaml(state.get("yolo_data_dir", ""))
        if written != expected:
            print(f"  {path} no longer describes the dataset on disk — rewriting it")
            return False
        return True
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
    # `--force` means "start clean", so it also suppresses resume; every other run
    # continues an interrupted step from its last checkpoint (3.3.2).
    resume = not args.force

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
    if force_step <= 2 and is_step_done(2, state, output_dir, config, manifest):
        print(f"\n[SKIP] Step 2: YOLO dataset (already at {state['yolo_data_dir']})")
    else:
        state.update(step_create_yolo_dataset(state["data_root"], output_dir, config))
        state["last_step"] = 2
        save_state(output_dir, state)
        manifest.record_step(
            2,
            config_hash=hash_json(config),
            yolo_data_dir=state["yolo_data_dir"],
            split_params=split_params(config),
            split_sha256=state["yolo_split_sha256"],
        )
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
        state.update(step_train_yolo(state["data_yaml"], output_dir, config, manifest=manifest, resume=resume))
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
            feature_cache=feature_cache, headless=headless, manifest=manifest, resume=resume,
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
            feature_cache=feature_cache, manifest=manifest, resume=resume,
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
