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

Phase 0 (foundations & reproducibility) and all of Phase 1 (tasks 1.1–1.8) are
implemented: normalization is fitted on train only (min/max for the flat VAE,
mean/std for the LSTM-VAE), the split is grouped and canonical
(`scripts/data/splits.py` draws it once from the annotation frame, persists it
to `split_manifest.json`, and every stage resolves it through
`split_params(config)`), and the run is three-way — `CONFIG["test_split"] = 0.1`
— with both VAE stages scoring train/val/test per-sample reconstruction MSE
(`scripts/evaluation/reconstruction.py`) into `<model>_split_scores.json`. The
detector consumes the same manifest: `create_yolo_dataset` no longer draws its
own split, `data.yaml` names `test` only when test images exist, and steps 2/3
are skipped on *content* (recorded split params, yaml vs dataset), not
existence. Grouping is on `video_id`, which does not stop one cow spanning
train and val through different videos; `target_id` is available as the
stricter key and Q1 decides whether to switch. What is still missing is the
evaluation harness itself — anomaly-score functions, thresholds, AUROC — which
is Phase 4; until then the scores files are honest but unlabeled errors.
Ground truth is problem #25.

## Commands

```bash
python scripts/setup.py            # install runtime deps from requirements.txt
python scripts/setup.py --dry-run  # show what would be installed

python -m scripts.run_full_pipeline                    # the whole pipeline
python -m scripts.run_full_pipeline --from-step 6      # re-run the VAE stages
python -m scripts.run_full_pipeline --force --headless --seed 7

ruff check scripts tests          # lint
pytest -q                         # tests (~15s, no vision stack needed)
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
├── data/                dataset download, YOLO-format conversion, feature cache,
│                        train-fitted scalers, the canonical split
├── evaluation/          per-split anomaly scores (Phase 4 builds on this)
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
- **Record provenance.** `scripts/manifest.py` writes `run_manifest.json` into
  every output directory: git SHA, config hash, seed, and content hashes of
  every input file, feature array and split. A step is only skipped if that
  record says it was produced by the current config. Keep per-step config
  hashes across resumes — deleting them turns every checkpoint into "unverifiable".
- **Errors must be loud.** A wrong-but-plausible default — mismatched scaler,
  stale checkpoint, empty coverage cell — invalidates results silently. Raise
  instead of guessing.
- **Depend on a train-fitted artifact only.** Anything estimated from data
  (scaler, PCA, covariance, split) must be fitted inside the training scope and
  carried with the model. This is the Phase 1 leak; see 1.8 for the test that
  guards it. Represent it as an object with explicit `fit`/`transform`
  (`scripts/data/normalize.py`) rather than a function that returns transformed
  data plus two arrays — a function that derives its own statistics cannot
  express *which* rows it used, and silently normalizes each sample with itself.
  Fit the object on the train index, then transform everything.
- **Tests must not need the vision stack.** cv2/torchvision/ultralytics imports
  are lazy by design so the logic is testable on a CPU-only box. Leakage tests
  stub the extractor and run the real step function — a test that only inspects
  source order passes even when the wrong rows are passed to `fit`.

## Style

Ruff with `E,F,I,B` at line-length 120 (`pyproject.toml`). No docstring on
obvious functions; docstrings explain *why* a non-obvious choice was made and
usually cite the task or problem number that forced it.
