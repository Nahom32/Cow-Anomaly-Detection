import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler

from scripts.checkpoint import data_fingerprint, latest_epoch_checkpoint, record_and_prune_checkpoints
from scripts.config import CONFIG
from scripts.data.feature_cache import open_run_feature_cache
from scripts.data.normalize import MinMaxNormalizer
from scripts.data.splits import describe, resolve_indices, resolve_split, split_params
from scripts.dataset.sequence_dataset import CowSequenceDataset, NormalisedSeqDataset
from scripts.manifest import RunManifest
from scripts.models.feature_extractor import create_feature_extractor
from scripts.models.lstm_vae import LSTMVAE, train_lstm_vae
from scripts.models.training import EarlyStopping, count_parameters, match_parameter_budget
from scripts.models.vae import VAE
from scripts.utils.hashing import hash_json
from scripts.utils.history import save_history
from scripts.utils.io import atomic_torch_save
from scripts.utils.seeding import set_seed


def main(resume=True):
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
    # From CONFIG, like every other entry point (1.4, 1.7): this script used to
    # hardcode its own fractions, so its "val" number came from a different
    # partition than the orchestrator's.
    SEED = CONFIG["random_seed"]
    SPLIT_GROUP_KEY = CONFIG["split_group_key"]

    set_seed(SEED)
    print(f"Using device: {DEVICE}")

    config = {"model": "lstm_vae", "seed": SEED, "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR,
              "normal_action_ids": NORMAL_ACTION_IDS, "feature_layer": 9, "seq_len": SEQ_LEN,
              "stride": STRIDE, "val_split": CONFIG["val_split"], "test_split": CONFIG["test_split"],
              "split_group_key": SPLIT_GROUP_KEY, "yolo_weights": YOLO_MODEL_PATH,
              "match_flat_parameters": CONFIG["lstm_vae_match_flat_parameters"]}
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
    assignment = resolve_split(OUTPUT_DIR, df, **split_params(CONFIG))
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

    # Statistics from the training sequences only (1.2). Fitting min/max over every
    # sequence in the run let each val window contribute to the numbers it was then
    # normalized by. Same min-max scaler as the flat VAE (2.4), so both models
    # reconstruct [0, 1] features and their errors are on one scale.
    print("Computing normalisation statistics from the training sequences...")
    normalizer = MinMaxNormalizer().fit(seq_dataset.stack(train_indices))

    norm_dataset = NormalisedSeqDataset(
        normalizer=normalizer,
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

    # No private generator on the train sampler: the shared loop restores the
    # global RNG on resume so the sampling order matches the uninterrupted run (3.3.3).
    train_loader = DataLoader(norm_dataset, batch_size=BATCH_SIZE, sampler=SubsetRandomSampler(train_indices))
    # Full, ordered validation pass (2.9); see step_lstm_vae.
    val_loader = DataLoader(Subset(norm_dataset, val_indices), batch_size=BATCH_SIZE, shuffle=False)

    # Width from the extracted features, not a hardcoded 256 (2.12). When the
    # config asks for it, the LSTM-VAE is sized to the flat VAE's parameter budget
    # so the comparison is not confounded by capacity (2.11).
    input_dim = seq_dataset.feature_dim
    flat_parameters = count_parameters(
        VAE(input_dim, hidden_dim=CONFIG["vae_hidden_dim"], latent_dim=CONFIG["vae_latent_dim"])
    )
    parameter_matched = bool(CONFIG["lstm_vae_match_flat_parameters"])
    if parameter_matched:
        lstm_hidden_dim, vae, _ = match_parameter_budget(
            lambda width: LSTMVAE(input_dim=input_dim, hidden_dim=width,
                                  latent_dim=CONFIG["lstm_vae_latent_dim"],
                                  num_layers=CONFIG["lstm_vae_num_layers"]),
            flat_parameters,
        )
        print(f"Parameter-matched LSTM-VAE: hidden width {lstm_hidden_dim} "
              f"for the flat VAE's {flat_parameters:,} params (2.11)")
    else:
        vae = LSTMVAE(input_dim=input_dim, hidden_dim=CONFIG["lstm_vae_hidden_dim"],
                      latent_dim=CONFIG["lstm_vae_latent_dim"],
                      num_layers=CONFIG["lstm_vae_num_layers"])

    print("Starting LSTM-VAE training...")
    checkpoint = {}
    # Continue an interrupted standalone run from its last epoch checkpoint (3.3.2).
    resume_from = latest_epoch_checkpoint(OUTPUT_DIR, "lstm_vae_anomaly", config=config) if resume else None
    history = train_lstm_vae(
        vae, train_loader, val_loader, epochs=EPOCHS, lr=LR, device=DEVICE,
        beta=CONFIG["lstm_vae_beta"], kl_warmup_epochs=CONFIG["lstm_vae_kl_warmup_epochs"],
        checkpoint=checkpoint,
        lr_scheduler=CONFIG["lstm_vae_lr_scheduler"],
        lr_scheduler_factor=CONFIG["lstm_vae_lr_scheduler_factor"],
        lr_scheduler_patience=CONFIG["lstm_vae_lr_scheduler_patience"],
        grad_clip=CONFIG["lstm_vae_grad_clip"],
        early_stopping=EarlyStopping(patience=CONFIG["lstm_vae_early_stopping_patience"]),
        seed=SEED, config=config, data_fingerprint=data_fingerprint(manifest=manifest),
        checkpoint_dir=OUTPUT_DIR, checkpoint_prefix="lstm_vae_anomaly",
        keep_last=CONFIG["checkpoint_keep_last"], resume_from=resume_from,
    )

    lstm_final = os.path.join(OUTPUT_DIR, "lstm_vae_anomaly.pth")
    lstm_best = os.path.join(OUTPUT_DIR, "lstm_vae_anomaly_best.pth")
    atomic_torch_save(lstm_final, vae.state_dict())
    wrote_best = checkpoint.get("best_state") is not None
    if wrote_best:
        atomic_torch_save(lstm_best, checkpoint["best_state"])
        print(f"Best val epoch {checkpoint['best_epoch']} ({checkpoint['best_metric']:.4f}); "
              f"final epoch {checkpoint['final_epoch']}. Scores use the final checkpoint (2.7).")
    # Checksum what was written and enforce the retention window (3.2.2, 3.2.3).
    record_and_prune_checkpoints(
        manifest=manifest, directory=OUTPUT_DIR, prefix="lstm_vae_anomaly",
        keep_last=CONFIG["checkpoint_keep_last"],
        final=lstm_final, best=lstm_best if wrote_best else None, step="lstm_vae",
    )
    save_history(history, os.path.join(OUTPUT_DIR, "lstm_vae_training_history.csv"))
    np.save(os.path.join(OUTPUT_DIR, "feature_min.npy"), normalizer.min_)
    np.save(os.path.join(OUTPUT_DIR, "feature_max.npy"), normalizer.max_)

    manifest.record_split("canonical_groups", {
        name: sorted(map(str, assignment[name])) for name in ("train", "val", "test")
    })
    manifest.record_split("lstm_vae_groups", {
        name: sorted({sequence_groups[i] for i in sequence_split[name]})
        for name in ("train", "val", "test")
    })
    manifest.record_array("lstm_vae_feature_min", normalizer.min_)
    manifest.record_array("lstm_vae_feature_max", normalizer.max_)
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
