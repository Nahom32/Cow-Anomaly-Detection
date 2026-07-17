CONFIG = {
    # Dataset
    "normal_action_ids": [0, 1, 2],
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
