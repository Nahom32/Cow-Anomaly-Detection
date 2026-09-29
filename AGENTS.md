# AGENTS.md

Working notes for coding agents and contributors on this repository.

## What this project is

Cattle behavioural **novelty** detection: YOLO detects cows, a hooked SPPF layer
produces a per-cow feature vector, and a generative model scores how far an
observation is from what was seen during training. The research question is
whether temporal + generative modelling is needed at all, or whether a simple
feature-space detector (Mahalanobis, Isolation Forest) does just as well.

The scientific plan lives in `tasklist.md` (Phases 0-9), derived from
`cow_anomaly_detection_problems_observed.md` (25 observed problems) and
`improvements_on_vad.md`. **Read the task list before changing anything** — the
phases are ordered by dependency, and several tasks exist specifically to stop a
later phase from re-introducing a bug an earlier one fixed.

## Current state

Phase 0 (foundations & reproducibility) is implemented. Phases 1+ are not. In
particular: **there is still no evaluation code, no anomaly-score function, no
test split and no train/val/test protocol**, and normalisation still leaks
statistics from validation into training (tasks 1.1, 1.2). No number produced by
this repo is publishable until Phase 1 lands.

## Commands

```bash
python scripts/setup.py            # install runtime deps from requirements.txt
python scripts/setup.py --dry-run  # show what would be installed

python -m scripts.run_full_pipeline                    # the whole pipeline
python -m scripts.run_full_pipeline --from-step 6      # re-run the VAE stages
python -m scripts.run_full_pipeline --force --headless --seed 7

ruff check scripts tests          # lint
pytest -q                         # tests (~4s, no vision stack needed)
pytest tests/test_pipeline_wiring.py  # needs cv2 + ultralytics, skipped otherwise
```

The full pipeline downloads ~11 GB from Kaggle and trains YOLO for up to 150
epochs. Do not run it as a way to check that a change works — use the tests.

## Layout

```
scripts/
├── config.py            every hyperparameter; nothing hardcodes a value
├── run_full_pipeline.py orchestrator; writes state + provenance to the output dir
├── utils/               cross-cutting helpers (seeding, io, hashing, history, plotting)
├── data/                dataset download, YOLO-format conversion, feature cache
├── dataset/             torch Datasets over cached features
└── models/              feature extractor, VAE, LSTM-VAE, YOLO trainers
tests/                   pytest suite; helpers.py holds shared fixtures/utilities
pipeline_output/         run artifacts (not committed)
```

## Conventions

- **Configuration is not optional.** If a behaviour is tunable, it goes in
  `CONFIG`. Hardcoded `256`, `9`, `0.2` and `150` are exactly what made 9.3 and
  2.12 necessary.
- **Seed every entry point.** Call `set_seed(CONFIG["random_seed"])` first. For
  anything that must be stable per *data item* rather than per run (window gap
  fill, augmentation), use `rng_for(...)` — a blake2b-derived generator, not the
  global RNG, which depends on access order.
- **Write artifacts atomically.** Use `scripts/utils/io.py`. A truncated
  `torch.save` output still exists on disk, and existence is currently how
  "this step is done" is decided.
- **Record provenance.** Bind every artifact to its inputs with
  `scripts/utils/hashing.py` (config hash, weights hash, feature-array hash) and
  write it to the run manifest.
- **Errors must be loud.** A wrong-but-plausible default — mismatched scaler,
  stale checkpoint, empty coverage cell — invalidates results silently. Raise
  instead of guessing.
- **Depend on a train-fitted artifact only.** Anything estimated from data
  (scaler, PCA, covariance, split) must be fitted inside the training scope and
  carried with the model. This is the Phase 1 leak; see 1.8 for the test that
  guards it.
- **Tests must not need the vision stack.** cv2/torchvision/ultralytics imports
  are lazy by design so the logic is testable on a CPU-only box.

## Style

Ruff with `E,F,I,B` at line-length 120 (`pyproject.toml`). No docstring on
obvious functions; docstrings explain *why* a non-obvious choice was made and
usually cite the task or problem number that forced it.
