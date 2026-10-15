CONFIG = {
    # Dataset
    "normal_action_ids": [0, 1, 2],
    # Fractions of the *remaining* groups after each earlier hold-out: test is
    # carved first (1.5), then val, so the reported val fraction is not inflated
    # by test rows. With these values the partition is 72/18/10. Val is the
    # larger held-out set because it also picks thresholds later; test is scored
    # once and never fitted on.
    "val_split": 0.2,
    "test_split": 0.1,
    "random_seed": 42,

    # Split unit for the canonical train/val/test manifest (1.3, 1.4). Grouping on
    # `video_id` stops consecutive frames of one cow straddling the split; it does
    # not stop the same cow appearing in train and val via a different video, because
    # AVA cows recur across videos. `"target_id"` closes that gap but its reliability
    # is unverified (tasklist Q1), so it is a value to switch once checked rather
    # than an assumption to bake in.
    "split_group_key": "video_id",

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
    # KL weight and its linear warm-up (2.5). KL is a latent-dim sum while recon is
    # a per-element mean, so the effective weight is not 1 by default; `beta` makes
    # it explicit and `kl_warmup_epochs` ramps it in to avoid posterior collapse.
    "vae_beta": 1.0,
    "vae_kl_warmup_epochs": 0,
    # Optimisation stabilisers (2.8). `reduce_on_plateau` halves the LR once val
    # stalls for `patience` epochs; early stopping then ends the run after
    # `early_stopping_patience` stalled epochs (0 disables). Gradients are clipped
    # to this norm before each step; both loops clip in the same place so the two
    # models' updates stay comparable. The real logs showed LR changes at epochs
    # 15/25/30 with no record of why, and training that kept going long past the
    # best epoch (48) — these make both visible and bounded.
    "vae_lr_scheduler": "reduce_on_plateau",
    "vae_lr_scheduler_factor": 0.5,
    "vae_lr_scheduler_patience": 5,
    "vae_grad_clip": 1.0,
    "vae_early_stopping_patience": 10,

    # LSTM-VAE
    "lstm_vae_epochs": 50,
    "lstm_vae_batch_size": 32,
    "lstm_vae_lr": 1e-3,
    "lstm_vae_hidden_dim": 128,
    "lstm_vae_latent_dim": 32,
    "lstm_vae_num_layers": 1,
    "lstm_vae_beta": 1.0,
    "lstm_vae_kl_warmup_epochs": 0,
    "lstm_vae_lr_scheduler": "reduce_on_plateau",
    "lstm_vae_lr_scheduler_factor": 0.5,
    "lstm_vae_lr_scheduler_patience": 5,
    "lstm_vae_grad_clip": 1.0,
    "lstm_vae_early_stopping_patience": 10,
    # 2.11: the flat VAE (111,424 params) and the LSTM-VAE (321,856) differ 2.9x,
    # so an unqualified Flat-vs-LSTM result confounds temporal modelling with
    # capacity (problem #13). When true, the LSTM-VAE's hidden width is solved to
    # match the flat VAE's parameter budget and the achieved counts are recorded.
    # Off by default so the standard model is unchanged; a benchmark can run both
    # arms to show whether the comparison survives capacity-matching.
    "lstm_vae_match_flat_parameters": False,
    "seq_len": 16,
    "seq_stride": 8,

    # Phase 4 benchmark grid (4.2.1). One JSON row per cell of the cross product
    # model x novelty_setting x split x seed x context_length, into `results_dir`.
    # The three novelty settings are the ones tasklist Q3 scopes the grid over;
    # their providers land with the Phase 6 drivers (6.2/6.4/6.5), and cells whose
    # provider is missing fail loudly rather than scoring something else. `seeds`
    # is 4.4.1's >=5-seeds-per-cell axis; `context_lengths` is the axis 7.1 sweeps
    # over {1,3,5,10,20,40}; `splits` names the *scored* split (fit is always on
    # the provider's train rows). The grid is inspectable without executing:
    # `python -m scripts.run_benchmark --dry-run`.
    "benchmark": {
        "models": ["euclidean", "mahalanobis_full", "isolation_forest", "markov", "pca"],
        "novelty_settings": ["leave_one_action_out", "unseen_transition", "temporal_corruption"],
        "splits": ["train", "val", "test"],
        "seeds": [42, 43, 44, 45, 46],
        "context_lengths": [1],
        "results_dir": "pipeline_output/results",
    },

    # Checkpoint retention (3.2.2). Every run keeps its best-val and final
    # checkpoints plus the `keep_last` most recent per-epoch ones; older epoch
    # checkpoints are deleted after each stage. Disk cost becomes material once
    # Phase 4 runs the full model x seed x split grid, and a resume only ever
    # needs the most recent epochs plus the best weights.
    "checkpoint_keep_last": 3,
}
