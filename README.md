# Cow-Anomaly-Detection

Anomaly detection in cow behaviour using YOLO-based cow detection and Variational Autoencoders (VAE) for feature-level anomaly scoring.

## Overview

The pipeline extracts per-cow feature vectors from video frames using a YOLO26n backbone (hooked at the SPPF layer), then trains two types of VAEs on normal-behaviour features to learn a compact representation. Anomalous frames produce higher reconstruction error, enabling anomaly detection.

Two VAE variants are supported:
- **Flat VAE** — frame-level MLP that processes each frame independently
- **LSTM-VAE** — sequence model that captures temporal patterns across frames

## Project Structure

```
scripts/
├── config.py                        # All hyperparameters (single source of truth)
├── run_full_pipeline.py             # Orchestrator: download → YOLO → VAE → LSTM-VAE
├── run_full_pipeline.sh             # Shell wrapper (creates venv, installs deps, runs pipeline)
├── setup.py                         # Install all dependencies
├── manifest.py                      # run_manifest.json: git SHA, config hash, seed, artifact hashes
├── train_vae_pipeline.py            # Standalone flat VAE pipeline
├── train_lstm_vae_pipeline.py       # Standalone LSTM-VAE pipeline
├── data/
│   ├── download_dataset.py          # Download CBVD-5 dataset from Kaggle
│   ├── explore_dataset.py           # Dataset inspection and statistics
│   ├── create_yolo_dataset.py       # Convert AVA annotations → YOLO-format dataset
│   ├── build_features.py            # Extract frame-level features using YOLO backbone
│   ├── feature_cache.py             # On-disk feature cache keyed by the YOLO weights hash
│   └── normalize.py                 # MinMaxNormalizer: fit on train, transform anything
├── dataset/
│   └── sequence_dataset.py          # PyTorch Dataset for temporal sequences (LSTM-VAE input)
└── models/
    ├── feature_extractor.py         # YOLO forward hook at SPPF layer + cow crop feature extraction
    ├── vae.py                       # Flat VAE model, loss, training loop, plotting
    ├── lstm_vae.py                  # LSTM-VAE model, loss, training loop
    ├── train_yolo_n.py              # YOLO26n training (settings read from CONFIG['yolo'])
    └── train_yolo_m.py              # YOLO26m training (thin wrapper over the same settings)
```

## Quick Start

### Full pipeline (recommended)

```bash
# Shell wrapper handles venv creation and dependency installation
./scripts/run_full_pipeline.sh

# Or run directly if dependencies are already installed
python -m scripts.run_full_pipeline

# Override output directory
python -m scripts.run_full_pipeline --output-dir my_experiment
```

### Resuming a previous run

The pipeline writes a `.pipeline_state.json` to the output directory after each step, plus a `run_manifest.json` that records what produced the artifacts. On re-run, a step is skipped when its output exists **and** — for the two training steps — the manifest confirms it was produced by the current config. A checkpoint left over from a different config is reported as stale and retrained, rather than silently reported as a current result.

```bash
# Re-run — skips all completed steps automatically
python -m scripts.run_full_pipeline

# Force re-run from step 4 (YOLO training) onwards
python -m scripts.run_full_pipeline --from-step 4

# Ignore all prior state and re-run everything
python -m scripts.run_full_pipeline --force
```

| Flag | Effect |
|------|--------|
| `--output-dir DIR` | Output directory (default: `pipeline_output`) |
| `--from-step N` | Re-run from step N onwards, skip steps 1..N-1 if their outputs exist |
| `--force` | Re-run all steps, ignore all prior state |

### What the pipeline does

| Step | Description |
|------|-------------|
| 1 | Download CBVD-5 dataset from Kaggle |
| 2 | Build YOLO-format dataset (images + labels, train/val split by video ID) |
| 3 | Generate `data.yaml` for YOLO training |
| 4 | Train YOLO26n (150 epochs) → produces `best.pt` weights |
| 5 | Create feature extractor from trained YOLO backbone (SPPF layer 9) |
| 6 | Flat VAE: extract frame features → split → min-max normalise (train-fitted) → train → save model |
| 7 | LSTM-VAE: build temporal sequences → z-score normalise → train → save model |

### Pipeline outputs

All artifacts are saved to `pipeline_output/`:

| File | Description |
|------|-------------|
| `cow_detector/yolo26n_cbvd/weights/best.pt` | Trained YOLO26n weights |
| `feature_cache/` | Cached backbone features keyed by the YOLO weights hash (see below) |
| `flat_vae_model.pth` | Trained flat VAE state dict |
| `flat_vae_history.csv` | Training loss history |
| `flat_vae_feature_min.npy` | Per-feature min, fitted on the train split only |
| `flat_vae_feature_max.npy` | Per-feature max, fitted on the train split only |
| `lstm_vae_model.pth` | Trained LSTM-VAE state dict |
| `lstm_vae_history.csv` | Training loss history (recon and KL logged separately) |
| `lstm_vae_feature_mean.npy` | Mean values for feature normalisation |
| `lstm_vae_feature_std.npy` | Std values for feature normalisation |
| `run_manifest.json` | Provenance for the run: git SHA, config hash, seed, feature hashes |

> **Still leaking.** The flat VAE's min/max is now fitted on the training rows
> only. The LSTM-VAE's mean/std is still computed over *every* sequence in the
> run, and both models still split at frame/video level rather than by cow
> identity. See `tasklist.md` tasks 1.2–1.4. Numbers from this pipeline are not
> yet publishable.

### Run manifest

`run_manifest.json` is written to the output directory and updated after every
step, so an interrupted run still records what it completed. It answers the
question "which code, which config, which data produced these numbers?":

| Field | Contents |
|-------|----------|
| `git` | commit SHA, branch, and whether the tree was dirty |
| `config` / `config_hash` | the full resolved config and its hash |
| `seed`, `device`, `command` | how the run was invoked |
| `artifacts` | every input file, bound by SHA-256 rather than by name |
| `arrays` | SHA-256 of the extracted feature, normalisation and split arrays |
| `splits` | SHA-256 of each train/val assignment, plus its sizes |
| `steps` | per-step config hash and counts, kept across resumes |
| `previous_run` | the config hash of the run this one replaced |

Step entries deliberately survive a resume: a step's entry holds the config hash
it was *actually* trained under, which is what distinguishes a resumable step
from a stale one. A checkpoint with no entry at all (e.g. one produced before
the manifest existed) cannot be verified and is retrained.

### Feature cache

Extracting features runs the YOLO backbone once per crop. Features are therefore
cached to `feature_cache/` as a `(N, D)` array plus its key list and validity mask,
keyed by the SHA-256 of the YOLO weights and the hooked layer. Both VAE stages
share one cache, so re-running a pipeline invocation reuses existing features and
only extracts the crops it has not seen before. Deleting the directory forces a
full re-extraction; changing the weights creates a new cache automatically.

## Running individual steps

Each step can also be run independently:

```bash
# Download dataset only
python -m scripts.data.download_dataset

# Create YOLO-format dataset only
python -m scripts.data.create_yolo_dataset

# Train flat VAE only (requires pretrained YOLO weights)
python -m scripts.train_vae_pipeline

# Train LSTM-VAE only (requires pretrained YOLO weights)
python -m scripts.train_lstm_vae_pipeline

# Explore dataset statistics
python -m scripts.data.explore_dataset
```

Edit `scripts/config.py` to change hyperparameters before running any pipeline script.

## Configuration

All hyperparameters live in `scripts/config.py`:

```python
CONFIG = {
    # Dataset
    "normal_action_ids": [0, 1, 2],   # standing, walking, lying
    "val_split": 0.2,
    "random_seed": 42,

    # YOLO training
    "yolo_epochs": 150,
    "yolo_imgsz": 640,
    "yolo_batch": 16,

    # Flat VAE
    "vae_epochs": 50,
    "vae_batch_size": 64,
    "vae_lr": 1e-3,
    "vae_hidden_dim": 128,
    "vae_latent_dim": 32,

    # LSTM-VAE
    "lstm_vae_epochs": 50,
    "lstm_vae_batch_size": 32,
    "lstm_vae_lr": 1e-3,
    "lstm_vae_hidden_dim": 128,
    "lstm_vae_latent_dim": 32,
    "lstm_vae_num_layers": 1,
    "seq_len": 16,
    "seq_stride": 8,
}
```

## Development

```bash
python scripts/setup.py --dry-run   # show what would be installed
pip install -r requirements-dev.txt # runtime deps + pytest + ruff
```

```bash
ruff check scripts tests            # lint
pytest -q                           # ~4s, no vision stack required
pytest tests/test_pipeline_wiring.py  # entry-point wiring; skipped without cv2/ultralytics
```

The test suite deliberately avoids cv2 / torchvision / ultralytics so it runs on
a CPU-only machine: feature extraction is injected as a callable, and the caching,
dataset and determinism logic is exercised directly.

See `AGENTS.md` for the working conventions and the current state of the
project, and `tasklist.md` for the plan.

## Requirements

- Python 3.9+ (developed and tested on 3.9.6)
- GPU recommended (CUDA, MPS, or CPU fallback)
- ~11 GB disk for CBVD-5 dataset

`requirements.txt` is the single source of truth for dependencies and is what
`scripts/setup.py` installs. Version-verified packages are pinned with `==`; the
four that could not be verified locally (`torchvision`, `ultralytics`,
`opencv-python`, `kagglehub`) carry an upper bound instead.

```bash
python scripts/setup.py          # install everything
python scripts/setup.py --dry-run  # show what would be installed
```
