# Cow-Anomaly-Detection

Anomaly detection in cow behaviour using YOLO-based cow detection and a Variational Autoencoder (VAE) for feature-level anomaly scoring.

## Project Structure

```
scripts/
├── data/
│   ├── download_dataset.py      # Download CBVD-5 dataset from Kaggle
│   ├── create_yolo_dataset.py   # Convert AVA annotations → YOLO-format dataset
│   └── build_features.py        # Extract VAE features from frames using YOLO backbone
├── models/
│   ├── vae.py                   # VAE model, loss, training loop, plotting
│   ├── feature_extractor.py     # YOLO forward hook + cow feature extraction
│   ├── train_yolo_n.py          # YOLOv26n training script
│   └── train_yolo_m.py          # YOLO medium training script
├── dataset/
│   └── anomaly_dataset.py       # PyTorch Dataset for normal/anomaly video clips
├── train_vae_pipeline.py        # End-to-end VAE training pipeline
└── setup.py                     # Install all dependencies
```

## VAE Anomaly Detection Pipeline

1. **Download the dataset** — `python scripts/data/download_dataset.py`
2. **Create YOLO dataset** — `python scripts/data/create_yolo_dataset.py`
3. **Train YOLO detector** — `python scripts/models/train_yolo_n.py`
4. **Train VAE** — update paths in `scripts/train_vae_pipeline.py` and run:
   ```bash
   python scripts/train_vae_pipeline.py
   ```

The pipeline:
- Loads a pretrained YOLO model and registers a forward hook at the SPPF layer (layer 9)
- Crops cow bounding boxes from each frame, resizes to 224×224, and extracts 256-d feature vectors
- Filters to "normal" behaviour classes only (action IDs 0, 1, 2)
- Min-max normalizes features and trains a VAE with a 32-d latent space
- Saves the trained model (`vae_anomaly_model.pth`), normalization stats, and training history

## Requirements

See `requirements.txt` or run `python scripts/setup.py`.
