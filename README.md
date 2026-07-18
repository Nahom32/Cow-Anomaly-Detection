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
├── train_vae_pipeline.py            # Standalone flat VAE pipeline
├── train_lstm_vae_pipeline.py       # Standalone LSTM-VAE pipeline
├── data/
│   ├── download_dataset.py          # Download CBVD-5 dataset from Kaggle
│   ├── explore_dataset.py           # Dataset inspection and statistics
│   ├── create_yolo_dataset.py       # Convert AVA annotations → YOLO-format dataset
│   └── build_features.py            # Extract frame-level features using YOLO backbone
├── dataset/
│   ├── anomaly_dataset.py           # PyTorch Dataset for normal/anomaly video clips
│   └── sequence_dataset.py          # PyTorch Dataset for temporal sequences (LSTM-VAE input)
└── models/
    ├── feature_extractor.py         # YOLO forward hook at SPPF layer + cow crop feature extraction
    ├── vae.py                       # Flat VAE model, loss, training loop, plotting
    ├── lstm_vae.py                  # LSTM-VAE model, loss, training loop
    ├── train_yolo_n.py              # YOLO26n training configuration
    └── train_yolo_m.py              # YOLO26m training configuration
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

The pipeline writes a `.pipeline_state.json` to the output directory after each step. On re-run, completed steps are detected by their output files and skipped automatically.

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
| 6 | Flat VAE: extract frame features → min-max normalise → train → save model |
| 7 | LSTM-VAE: build temporal sequences → z-score normalise → train → save model |

### Pipeline outputs

All artifacts are saved to `pipeline_output/`:

| File | Description |
|------|-------------|
| `cow_detector/yolo26n_cbvd/weights/best.pt` | Trained YOLO26n weights |
| `flat_vae_model.pth` | Trained flat VAE state dict |
| `flat_vae_history.csv` | Training loss history |
| `flat_vae_feature_min.npy` | Min values for feature normalisation |
| `flat_vae_feature_max.npy` | Max values for feature normalisation |
| `lstm_vae_model.pth` | Trained LSTM-VAE state dict |
| `lstm_vae_feature_mean.npy` | Mean values for feature normalisation |
| `lstm_vae_feature_std.npy` | Std values for feature normalisation |

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

## Requirements

- Python 3.10+
- GPU recommended (CUDA, MPS, or CPU fallback)
- ~11 GB disk for CBVD-5 dataset

See `requirements.txt` or run `python scripts/setup.py`.
