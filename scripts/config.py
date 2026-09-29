CONFIG = {
    # Dataset
    "normal_action_ids": [0, 1, 2],
    "val_split": 0.2,
    "random_seed": 42,

    # YOLO training. Nested because there are two dozen of them; a config key
    # that no entry point reads is a lie, so the trainers take this dict as-is.
    "yolo": {
        "model": "yolo26n.pt",
        "epochs": 150,
        "imgsz": 640,
        "batch": 16,
        "patience": 50,
        "lr0": 0.01,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3,
        "warmup_momentum": 0.8,
        "box": 7.5,
        "cls": 0.5,
        "dfl": 1.5,
        # Augmentation
        "hsv_h": 0.02,
        "hsv_s": 0.8,
        "hsv_v": 0.4,
        "degrees": 10.0,
        "translate": 0.2,
        "scale": 0.5,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 1.0,
        # Runtime
        "optimizer": "auto",
        "device": 0,
        "workers": 4,
        "project": "cow_detector",
        "name": "yolo26n_cbvd",
        "exist_ok": True,
        "verbose": True,
    },

    # Feature extraction
    "feature_layer": 9,  # SPPF layer hooked for feature extraction
    "feature_input_size": 224,

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
