# Task List — Cattle Behavioral Anomaly Detection

Derived from `cow_anomaly_detection_problems_observed.md` (25 problems) and
`improvements_on_vad.md` (research plan), mapped onto the current state of `scripts/`.

**Baseline state of the repo (as read):** models train, but **no evaluation exists**. There is no
anomaly-score function, no AUROC/AUPRC, no threshold, no test set, and no baseline detector in
Python anywhere. Every "finding" so far (e.g. the ~10x loss gap between FlatVAE and LSTM-VAE) is an
artifact of `reduction="sum"` plus a frame-level split with pre-split normalization.

**Two cross-cutting infrastructure gaps** drive Phases 3 and 4:

- **Checkpointing is effectively absent.** The repo stores bare `state_dict`s
  (`flat_vae_model.pth`, `lstm_vae_model.pth`) with no optimizer state, no epoch, no config
  binding, no RNG state, and non-atomic writes. Training cannot be resumed, cannot be audited, and
  can silently reuse a stale model.
- **There is no benchmark harness.** Training code is bespoke per model and the results plumbing
  (metrics, thresholds, seed sweeps, aggregation, significance testing) does not exist. Nothing
  enforces that every cell of the experimental matrix gets evaluated.

**Research question (from `improvements_on_vad.md:30`):**

> Do cattle behavioral anomalies require temporal and generative modeling, or can substantially
> simpler feature-space anomaly detectors achieve comparable performance?

Every task below either (a) removes a confound that makes the answer unmeasurable, (b) makes results
reproducible and resumable, or (c) measures part of the answer.

---

## Status legend

| Mark | Meaning |
|---|---|
| `[ ]` | Not started |
| `[~]` | In progress |
| `[x]` | Done |

---

## Phase 0 — Foundations & reproducibility (blocks everything)

No result below is trustworthy until these land.

- [x] **0.1** Global seeding. `random_seed` is declared at `scripts/config.py:5` but never used —
      no `torch.manual_seed`, `np.random.seed`, or `random.seed` anywhere in `scripts/`. Add a
      single `set_seed(seed)` called at the top of every entry point.
- [x] **0.2** Seed the unseeded gap-fill noise in `scripts/dataset/sequence_dataset.py:99,108` and
      make it deterministic per `(video, track, window)` instead of resampled on every `__getitem__`.
- [x] **0.3** Feature caching. Features are re-extracted from disk on every access —
      `NormalisedSeqDataset` re-instantiates the whole parent (`sequence_dataset.py:117-126`), so
      YOLO runs at least twice per pipeline invocation and again every epoch. Cache the
      `(N, 256)` array + its metadata to `.npy` keyed by YOLO weights hash.
- [x] **0.4** Persist training history for **every** model. `flat_vae_history.csv` exists;
      **no `lstm_vae_history.csv` was ever written**, and `lstm_vae.py:65,77` discards the recon/KL
      split (`loss, _, _ = ...`). The Phase 4 harness requires uniform history output from all
      models.
- [x] **0.5** Make `plt.show()` non-blocking / headless-safe. `scripts/models/vae.py:139` is called
      from `run_full_pipeline.py:174` and will block any server or CI run.
- [x] **0.6** Pin `requirements.txt`. All 14 deps are unpinned. `scripts/setup.py:6-17` maintains a
      second, **divergent** list (omits `matplotlib`, `pyyaml`, `numpy`, `scikit-learn`) — make
      `setup.py` read `requirements.txt`.
- [x] **0.7** Drop or justify the dead deps: `pytorchvideo` and `decord` are installed but never
      imported.
- [x] **0.8** Fix Python version drift. README claims 3.10+; `create_yolo_dataset.py:18` and
      `explore_dataset.py:11` use PEP-604 `str | None` annotations that break on the local 3.9.6.
- [x] **0.9** Add a minimal test suite. There is currently **no** `tests/`, no CI, no lint/format
      config, no `AGENTS.md`. Start with leakage tests (Phase 1) and checkpoint round-trip tests
      (Phase 3) — those are the ones that silently invalidate results.
- [x] **0.10** Add a run manifest to every output dir: git SHA, config hash, seed, feature-array
      hash, split manifest hash. `is_step_done` (`run_full_pipeline.py:248-263`) is
      **existence-only**, so a model trained before a config change is silently reused. Resolved
      properly by 3.9.
- [x] **0.11** Scope `scripts/dataset/anomaly_dataset.py` — `CowDatasetHybrid` is **dead code**
      (never imported). It has a label off-by-one (`:26,29` uses `[c+1]`, inconsistent with the
      `[0,1,2]` used everywhere) and nondeterministic pseudo-anomaly sampling (`:89`).
      **Deleted** (never imported by any entry point; a copy survives in
      `notebooks/Cow_Localization_and_Anomaly_Detection.ipynb` for reference). Its
      `df.sample(1)` pairing was unfixable as written — it paired a normal clip with a
      *random* anomaly clip, so it was never a usable negative-pair dataset. Phase 2 needs
      evaluation-time pairing over cached features instead (see 2.1); do not resurrect this
      file, or the `[c+1]` label shift and the unseeded sample come back with it.

---

## Phase 1 — Eliminate data leakage (P0, blocks every claim)

Directly addresses problems #4, #5 and `improvements_on_vad.md:304-308`.

- [x] **1.1** Move normalization after the split. `normalize_features`
      (`scripts/data/build_features.py:50-54`) computed min/max over the **entire** feature matrix,
      and is called *before* `train_test_split` (`run_full_pipeline.py:153` → `:155`;
      `train_vae_pipeline.py:42` → `:44`). Every validation sample is currently normalized using
      statistics derived partly from itself. Fit on train only.
      **Done** — `normalize_features` deleted and replaced by
      `scripts/data.normalize.MinMaxNormalizer` (explicit `fit`/`transform`; `transform` on an
      unfit scaler raises rather than deriving min/max from whatever it is handed). Both entry
      points now split, then `.fit(features[train_idx])`, then transform. The persisted
      `flat_vae_feature_min/max.npy` are the **train** extremes and the manifest records
      `normalizer_fit_on="train"` plus the min/max array hashes. Guarded by
      `tests/test_normalization.py`, which drives `step_flat_vae` and `train_vae_pipeline.main` for
      real and asserts the train tensors and the saved scaler are bit-identical when only the val
      rows are perturbed by +1000. This is the min/max half of 1.8; the mean/std, PCA and
      covariance cases land with 1.2/4.3.
- [x] **1.2** Same fix for the LSTM-VAE z-score. Mean/std are computed over **all** sequences
      (`run_full_pipeline.py:198-203`, `train_lstm_vae_pipeline.py:47-54`) before the video split at
      `:217-223`. Fit on train sequences only. `NormalisedSeqDataset` already takes `mean`/`std` as
      constructor arguments, so this is a call-site fix plus the same train-only
      invariant the min-max path now has.
      **Done** — `ZScoreNormalizer` added to `scripts/data/normalize.py` (explicit `fit`/`transform`,
      folds `eps` into the fitted `std_` and preserves shape for `(n_windows, seq_len, d)`). Both
      `step_lstm_vae` and `train_lstm_vae_pipeline.main` now split by video **before** computing
      statistics, fit the normalizer on `seq_dataset.stack(train_indices)` only, and record
      `normalizer_fit_on="train"` in the manifest. `CowSequenceDataset.stack()` was added to read only
      the selected windows. The LSTM-VAE's z-score statistics are now train-fitted and never derived
      from the val set.
- [ ] **1.3** Replace the flat-VAE frame-level split with a grouped split. `run_full_pipeline.py:155-157`
      and `train_vae_pipeline.py:44` call `train_test_split` with **no `groups=` argument** —
      adjacent frames from the same cow land in both train and val (problem #5). Use
      `GroupShuffleSplit` on `video_id` (and `target_id` where reliable).
- [ ] **1.4** **Stop the two models from being trained on different partitions.** The flat VAE
      splits at frame level; the LSTM-VAE splits by video. Until these share one split, no
      Flat-vs-LSTM comparison means anything. Emit a single canonical split manifest
      (`{video_id -> train|val|test}`) and have both stages consume it.
- [ ] **1.5** Create a **test set**. Currently train/val only, and val doubles as the reported
      number. Introduce the three-way protocol from `problems_observed.md:339-349` and
      `improvements_on_vad.md:304-308`: train → val → **threshold selection** → test.
- [ ] **1.6** Propagate the detector-stage grouping. `create_yolo_dataset.py:70-73` is the **only**
      grouped split in the repo (verified: 392 train / 99 val videos, zero overlap) and it is not
      carried into the VAE stage.
- [ ] **1.7** Pass `CONFIG["random_seed"]` / `CONFIG["val_split"]` through. `step_create_yolo_dataset`
      (`run_full_pipeline.py:87-93`) ignores both and falls back to module defaults.
- [ ] **1.8** Regression test: assert that no statistic (min/max/mean/std/PCA/covariance) changes
      when val/test rows are perturbed. This is the single highest-value test in the project, and
      Phase 4 must not reintroduce the same leak.
      **Partially covered by 1.1** — `tests/test_normalization.py` already asserts the min/max case
      against the real `step_flat_vae` and `train_vae_pipeline.main` call sites. What remains is
      mean/std (1.2), PCA and covariance, which do not exist until Phase 4.3 adds those detectors,
      and the test-split case, which does not exist until 1.5.

---

## Phase 2 — Make FlatVAE and LSTM-VAE comparable (P0)

Addresses problems #2, #3, and the loss-scale issue at `problems_observed.md:45-63`.

- [ ] **2.1** Report **per-element MSE**, not a sum. Both losses use
      `F.mse_loss(..., reduction="sum")` (`vae.py:48`, `lstm_vae.py:47`), so magnitude scales with
      `batch x seq_len x 256`. The LSTM sums over `32x16x256 = 131,072` elements vs the flat VAE's
      `64x256 = 16,384` — a **16x** mechanical gap. This is the entire source of the "~10x loss
      difference". Keep `seq_len` fixed across all comparisons.
- [ ] **2.2** Fix the loss-averaging divisor bug in the LSTM-VAE. `lstm_vae.py:69,79` divides by
      `len(loader.dataset)`, but the loader is built with `SubsetRandomSampler`
      (`run_full_pipeline.py:226`) so `loader.dataset` is the **full** dataset, not the sampled
      subset. Meanwhile `vae.py:82` divides by `len(loader)` (batch count). The two pipelines report
      incomparable units — one is per-batch-sum, the other a wrongly-divided per-sample-sum.
- [ ] **2.3** Log recon and KL as **separate** series for both models (flat VAE already does;
      LSTM-VAE discards them).
- [ ] **2.4** Make output ranges consistent. The flat VAE ends in `nn.Sigmoid()` (`vae.py:25`);
      the LSTM-VAE's `fc_out` does not (`lstm_vae.py:19`). Their reconstruction errors are therefore
      not on the same scale and cannot be compared.
- [ ] **2.5** Add an explicit `beta` and KL annealing. The KL term is summed over 32 latent dims
      while the recon term sums over 256 elements (`vae.py:48-49`), so the effective KL weight is
      inflated relative to a per-element MSE. `beta` is hardcoded to 1.0 with no warm-up.
- [ ] **2.6** Make evaluation deterministic. `reparameterize` samples unconditionally
      (`vae.py:32-35`, `lstm_vae.py:28-31`), so `val_loss` carries Monte-Carlo noise — visible as a
      **3.07-unit oscillation** across the last 10 epochs of the real `flat_vae_history.csv`.
      Use the analytic KL and a fixed set of samples (or a mean) at eval.
- [ ] **2.7** Report **best** and **final** checkpoints and declare which one is compared (schema in
      Phase 3). The saved model is the epoch-50 checkpoint (`vae.py:107`); best val was epoch 48
      (197.90 vs 200.98 final).
- [ ] **2.8** Add LR scheduler, early stopping, and gradient clipping to both loops.
- [ ] **2.9** Remove `SubsetRandomSampler` from the **validation** loaders
      (`run_full_pipeline.py:226`, `train_lstm_vae_pipeline.py:75`) — validation is currently
      re-shuffled and stochastic every epoch.
- [ ] **2.10** Guard the empty-val case in the LSTM-VAE. `vae.py:88` has `if val_loader:`;
      `lstm_vae.py` does not.
- [ ] **2.11** Report a **parameter-matched** baseline. Flat VAE is 111,424 params, LSTM-VAE is
      321,856 (2.9x). Any observed difference is currently confounded with capacity (problem #13).
- [ ] **2.12** Derive `input_dim` from the feature array instead of hardcoding `256` in four places:
      `lstm_vae.py:7`, `run_full_pipeline.py:229`, `train_lstm_vae_pipeline.py:77`,
      `sequence_dataset.py:111`.

---

## Phase 3 — Checkpointing & artifact management

Nothing downstream is reproducible or resumable without this. Today a checkpoint is a bare
`state_dict` with no provenance, so there is no way to tell which config, seed, or data version
produced a given number.

### 3.1 Checkpoint schema

- [ ] **3.1.1** Define **one** checkpoint format for all models, carrying at minimum:
      `model_state`, `optimizer_state`, `scheduler_state`, `epoch`, `best_metric`,
      `best_metric_epoch`, resolved `config`, `seed`, RNG states (torch/numpy/python), `git_sha`,
      `data_fingerprint`, and a `format_version`.
- [ ] **3.1.2** Migrate the two existing artifacts (`flat_vae_model.pth` = 111,424 params,
      `lstm_vae_model.pth` = 321,856 params). They hold weights only, so they cannot be resumed —
      decide explicitly whether to backfill or archive them. **Note: both were trained under the
      leaked split, so they are scientifically void regardless and should be archived, not
      backfilled.**

### 3.2 Write safety & retention

- [ ] **3.2.1** **Atomic writes** — write to `*.tmp`, fsync, then `os.replace()`. Today a crash
      mid-`torch.save` leaves a truncated `.pth` that `is_step_done` (`run_full_pipeline.py:248-263`)
      still reports as a completed step.
- [ ] **3.2.2** Retention policy: keep best + last-N (N≈3) + final. Disk cost becomes material once
      Phase 4 runs ~12 models x 5 seeds x multiple splits.
- [ ] **3.2.3** Checksum each checkpoint and record it in the manifest (0.10) so silent corruption
      or substitution is detectable.

### 3.3 Shared API & resume

- [ ] **3.3.1** One `save_checkpoint` / `load_checkpoint` / `resume_training` implementation shared
      by every training loop. Today there are two bespoke, near-duplicate loops
      (`vae.py:54-107`, `lstm_vae.py:53-86`) with divergent accumulator conventions (see 2.2).
- [ ] **3.3.2** **Resume interrupted training mid-run.** `--from-step` currently restarts an epoch
      from zero; a 150-epoch YOLO run interrupted at epoch 140 begins again at epoch 1. The existing
      `runs/detect/cow_detector/yolo26n_cbvd/results.csv` stopped at 126 epochs via `patience=50`,
      which is early stopping, not interruption — do not confuse the two.
- [ ] **3.3.3** Resume must reproduce the uninterrupted loss curve. Test this explicitly (3.6.2).

### 3.4 Checkpoints for non-neural models

- [ ] **3.4.1** Mahalanobis needs `mu`, `Sigma`, **and the covariance-estimator name**
      (diagonal / full / regularized / Ledoit-Wolf / PCA+Mahalanobis) — the estimator identity is
      the experimental variable and must not be inferable from array shape.
- [ ] **3.4.2** Euclidean needs `mu`; PCA needs components + explained variance; Isolation Forest
      needs the fitted estimator; Markov needs the transition matrix `P(b_t|b_{t-1})`.
      These are fitted artifacts, not torch modules — wrap them in the same checkpoint envelope or
      Phase 4 cannot resume or audit them.

### 3.5 Bind preprocessing to the model

- [ ] **3.5.1** Store scaler/normalizer state **inside** the checkpoint.
      `flat_vae_feature_min.npy`/`_max.npy` and `lstm_vae_feature_mean.npy`/`_std.npy` are separate
      sidecars that can be mismatched against the weights they belong to — a silent correctness
      hazard at scoring time.
- [ ] **3.5.2** Provide a loader that verifies sidecar-vs-checkpoint agreement so legacy runs fail
      loudly rather than scoring with the wrong scaler.

### 3.6 Integrity & tests

- [ ] **3.6.1** Validate on load: shape check against `config`, `format_version` migration path,
      hard failure on mismatch rather than a silent `load_state_dict` no-op.
- [ ] **3.6.2** Tests: (a) save/load round-trip equality, (b) atomicity under simulated interrupt,
      (c) resume reproduces the uninterrupted curve, (d) sidecar-mismatch detection.
- [ ] **3.6.3** Rework `is_step_done` (`run_full_pipeline.py:248-263`) to validate a checkpoint's
      config hash + data fingerprint rather than mere file existence. This closes 0.10 and
      8.2. Note `find_best_pt()` (`:62-72`) falls back to a recursive `glob("**/best.pt")` from CWD
      and can select unrelated weights.

---

## Phase 4 — Unified benchmark harness (largest build)

The repo has **zero** scoring/evaluation code. Until this exists, no research question can be
answered. The goal is a model-agnostic harness that every detector plugs into, so adding a baseline
is a config change rather than bespoke plumbing.

### 4.1 Detector interface

- [ ] **4.1.1** Define one contract every model implements:
      `fit(train_data) -> None`, `score(data) -> np.ndarray` (continuous anomaly score `A_t`),
      `predict(data, threshold) -> np.ndarray`. Today each model has bespoke training code and
      **no** score function at all.
- [ ] **4.1.2** Enforce that **every** fitted object (scaler, PCA, `mu`, `Sigma`, weights) is
      constructed inside `fit()` from train data only. This is where the Phase 1 leak would come
      back — assert it in code (ties to 1.8).

### 4.2 Runner & coverage

- [ ] **4.2.1** `run_benchmark.py`: iterate the full grid
      `model x anomaly_setting x split x seed x context_length`, writing one JSON/CSV row per run
      into a single `results/` directory.
- [ ] **4.2.2** **Coverage matrix with enforcement.** Encode the experimental matrix from
      `improvements_on_vad.md:385-396` (problem #24) — a 10-model x 5-mechanism grid — as a declared
      spec, and make the benchmark **fail loudly on any empty cell**. Coverage is a checked
      invariant, not a documentation promise.
- [ ] **4.2.3** Record per-cell coverage status (complete / missing / failed) in the results index.

### 4.3 Full model roster

Every entry below is missing today except the two VAE training loops.

| Group | Models |
|---|---|
| Static | Euclidean, Mahalanobis (diag / full / regularized / Ledoit-Wolf / PCA+Mahal), Isolation Forest, AE, VAE |
| Temporal | Markov, GRU predictor, LSTM predictor, LSTM-VAE |
| Combined | GRU + Mahalanobis |
| Control | YOLO-representation → Mahalanobis (problem #11) |

- [ ] **4.3.1** Implement each row. Scores per `improvements_on_vad.md:125-158` (static),
      `:160-192` (AE/VAE), `:194-230` (Markov, GRU/LSTM predictors), `:232-256` (LSTM-VAE).
- [ ] **4.3.2** Log `cond(Sigma)` per Mahalanobis variant. Full covariance will be near-singular:
      256 dims vs ~1.5k independent training samples of heavily correlated SPPF features
      (problem #10).
- [ ] **4.3.3** Compute and report `Delta_Mah = AUROC_Mah - AUROC_Euc` with AUPRC and false-alert
      deltas (`improvements_on_vad.md:289-298`) as a first-class output of the harness.

### 4.4 Repetition & statistics

- [ ] **4.4.1** **>=5 seeds per cell**, report mean ± std. Single-seed deltas are uninterpretable at
      ~491 videos; the variance across seeds will likely exceed most effects being measured.
- [ ] **4.4.2** Paired significance testing across seeds — Wilcoxon signed-rank — plus **DeLong's
      test** for paired AUROC comparison (Mahalanobis vs Euclidean on identical data). Without
      this, "Mahalanobis is better" is not evaluable.
- [ ] **4.4.3** Verify determinism: re-running a cell reproduces the score vector exactly (or within
      a documented tolerance for stochastic models like the VAEs).

### 4.5 Metrics module

- [ ] **4.5.1** AUROC **and** AUPRC (problem #19 — AUROC alone hides false positives under class
      imbalance).
- [ ] **4.5.2** Precision / recall / F1 at a **validation-selected** threshold. Never select on the
      test set (problem #18). Record the chosen threshold and selection rule per run.
- [ ] **4.5.3** Operational metrics: **false alerts per hour** and **per camera-day**
      (`improvements_on_vad.md:317,323-327`). For a monitoring system this is more meaningful than
      reconstruction loss.
- [ ] **4.5.4** Detection delay and inference latency.

### 4.6 Event extraction & event-level metrics

- [ ] **4.6.1** Convert continuous scores into events via a **persistence rule** — entry threshold
      held for a minimum duration, lower recovery threshold (`improvements_on_vad.md:327`) —
      rather than treating every anomalous frame as an independent alert (problem #20).
- [ ] **4.6.2** Event-level metrics: false events/day, event detection rate, detection delay, event
      duration. Report alongside observation-level metrics, never instead of them.

### 4.7 Caching, aggregation, reporting

- [ ] **4.7.1** Persist per-model anomaly scores to disk so metrics, thresholds, plots, and the
      Phase 7 shuffle ablation can be recomputed **without retraining**.
- [ ] **4.7.2** Aggregation: a leaderboard table (one row per model, all metrics, all seeds) plus
      per-ablation tables.
- [ ] **4.7.3** Plots: accuracy-vs-latency Pareto front (feeds Phase 8), metric-vs-context-length
      curves (feeds Phase 7), score distributions per anomaly source.
- [ ] **4.7.4** Results index keyed by `(model, setting, split, seed)` so any reported number in the
      paper can be traced back to the exact run that produced it.

---

## Phase 5 — Representation and identity (problems #6, #7, #8, #11)

- [ ] **5.1** **Fix the feature extractor — it is not box-conditioned.** `feature_extractor.py:83`
      runs a full-image forward pass, grabs the **frame-global** SPPF map, and spatial-average-pools
      it. `extract_cow_features` resizes a crop to 224x224 but only to change input resolution —
      every cow in a frame yields the **identical** 256-dim vector. Train images average **7.83
      boxes/image** (max 24), so per-cow features are currently indistinguishable. ROI-align the
      feature map per bounding box, or crop-and-refeature.
- [ ] **5.2** **Add a real tracker.** There is no ByteTrack/IoU association anywhere — the "track"
      in `sequence_dataset.py:35-51` is the AVA **`target_id` label column**, which may be 0 or
      unreliable and can silently merge distinct cows into one track. Run ByteTrack (or Ultralytics
      tracking) over the detections to produce real `track_id`s (problem #7).
- [ ] **5.3** **Guard against the no-tracker case.** Without 5.2, `target_id` must be validated
      (is it unique per cow? how many distinct ids per video?) before it is trusted as identity.
- [ ] **5.4** Make windows strictly identity-aware: a sequence must never mix cows. If tracking is
      unavailable, **drop** cross-identity windows rather than concatenating them (problem #6).
- [ ] **5.5** Append **quality signals** to the feature vector: detector confidence, box
      area/aspect ratio, track persistence/age, occlusion proxy, blur estimate, consecutive-miss
      count. **Promoted to required.** Under the novelty framing, a degraded observation is
      *definitionally* novel — occlusion, blur, or a missed detection all produce unfamiliar vectors
      and the flagger fires. This is problem #8, and under a flagger framing it is the dominant
      false-alert source rather than a caveat. Fully domain-free: these describe the camera and the
      detector, not the cow. Paired with 6.6 (degradation as a positive control).
- [ ] **5.6** Handle the missing-frame fill deliberately. `sequence_dataset.py:99,108` fills gaps
      with `prev_feat + N(0, 0.05)` noise; `:111` falls back to `np.zeros(256)` — a large
      out-of-distribution spike that will read as maximally anomalous. Replace zeros with an explicit
      **missingness flag** feature so gaps are not scored as behavior.
- [ ] **5.7** Quantify the **detector's own contribution** (problem #11). The single most important
      benchmark row is `YOLO representation -> Mahalanobis` vs `YOLO representation -> LSTM-VAE`. If
      the simple version matches, the contribution is the representation, not the generative model.

---

## Phase 6 — Operationalize "novelty = unseen" (problem #25, `improvements_on_vad.md:329-353`)

**Framing decision (settled):** an anomaly in this study is **an observation the system has not seen
during training** — either an unseen behavior state or an unseen transition. There is no clinical or
veterinary claim, and none is needed. This is standard novelty/OOD detection, and it is the correct
framing for a methods paper.

Practical consequences:

- The paper never claims to detect abnormality, illness, or lameness. Behavioral precursors of
  disease may appear in the discussion as *motivation* ("such a detector could surface deviations
  an expert would later interpret"), never as a result.
- Vocabulary: **novelty score**, not anomaly score. Code identifiers included — rename
  `anomaly_dataset.py` -> `novelty_dataset.py` and `CONFIG["normal_action_ids"]` ->
  `CONFIG["in_distribution_action_ids"]` (`config.py:3`) so the code says what the paper says.
- **A flagger is only as good as its false-alert rate**, so 4.5.3 (false alerts per camera-day) is a
  headline metric under this framing, not a secondary one.
- Ground truth needs no domain annotation. It is generated by the experiment itself.

- [ ] **6.1** State the definition explicitly in one short section: anomaly := unseen during
      training, in two forms (unseen state, unseen transition). Parameterize the harness grid
      (4.2.1) over these two. Do **not** start Phase 4 metrics before this is written — the metrics
      are meaningless without it.
- [ ] **6.2** **Leave-one-action-out as the primary setting.** Hold out each action in turn, train on
      the rest, evaluate, and report the **full sweep** rather than one arbitrary partition. This is
      the key move: it removes the need to defend *why* a particular action was called unseen (a
      question no domain-free researcher can answer), by averaging over the choice instead. A method
      that performs consistently across all held-out actions is robust to the partition.
      `config.py:3` currently trains on a hardcoded `action_ids=[0,1,2]` — replace with a
      configurable LOO driver over all available actions.
- [ ] **6.3** **Control training coverage per held-out action.** A class held out with 0 training
      examples is not comparable to one held out with 9,000 — under the "unseen" definition, that
      difference *is* a confound, not a detail. The LOO sweep exposes it; record the retained-class
      counts per fold so variance across folds is interpretable rather than mysterious.
- [ ] **6.4** **Unseen transitions.** Individual frames are in-distribution; the *sequence* is
      unseen (problem #12). Requires the state-vs-transition distinction to be a first-class
      evaluation setting, and requires real cow identity (5.2) to be meaningful.
- [ ] **6.5** **Temporal corruption operators** as a controlled secondary setting: sequence reversal,
      frame skipping, temporal repetition, transition insertion, acceleration/deceleration
      (`improvements_on_vad.md:343-347`). These probe sensitivity to known temporal irregularities
      and require no ground-truth annotation.
- [ ] **6.6** **Degradation as a positive control.** Inject image-level degradations (occlusion,
      blur, small bbox, low confidence) and confirm the flagger fires — this measures the #8
      conflation directly and is entirely domain-free. If novel states and degraded observations are
      indistinguishable, that is a finding, not a bug.
- [ ] **6.7** Ship a **ground-truth manifest** (`video_id`, `track_id`, `t_start`, `t_end`,
      `novelty_type`, `fold`) so every evaluation reads from one auditable file, is reproducible, and
      is traceable through 4.7.4. It is generated by the LOO driver, not annotated by hand.
- [ ] **6.8** Report the novelty:normal ratio, retained-class counts, and per-fold results. Results
      from different folds are different experiments and must not share a table row without the fold
      identity attached.
- [ ] **6.9** Vocabulary guard: grep the paper, code, and config for "anomal", "abnormal", "sick",
      "disease" and ensure every occurrence is either the technical term "novelty/OOD detection" or
      explicitly labeled as out of scope.

---

## Phase 7 — Temporal ablations

- [ ] **7.1** Context-length sweep `k ∈ {1,3,5,10,20,40}` (`improvements_on_vad.md:260-272`,
      problem #15). Plot AUROC/AUPRC/latency vs `k` to determine whether anomalies need
      instantaneous state, short-term dynamics, or long context. Wire this into the harness grid
      (4.2.1) rather than as a separate script.
- [ ] **7.2** **Temporal-shuffle sanity check** (problem #16). Permute order within a sequence while
      preserving observations. If performance is unchanged, the model is not using temporal
      structure and the temporal claim is unsupported. Run for GRU predictor, LSTM predictor, and
      LSTM-VAE. Reuses cached scores (4.7.1) — no retraining needed.
- [ ] **7.3** **State vs transition separation** (problem #12). Report static and temporal detectors
      on the same anomalies, split by whether the individual frames are out-of-distribution or only
      the transitions are.
- [ ] **7.4** Sweep `seq_stride` alongside `seq_len` — currently fixed at 16/8 (`config.py:25-26`),
      and stride controls the effective overlap between windows.
- [ ] **7.5** Sweep the sampling rate (problem #21). Everything currently runs at native 25 FPS
      (`build_features.py:27`). Behavioral states change on much longer timescales; compare
      sampling rates on the performance/compute trade-off.

---

## Phase 8 — Computational analysis (`improvements_on_vad.md:369-381`)

- [ ] **8.1** Per-model report: parameter count, inference latency, throughput, GPU memory, training
      time, feature dimensionality, compute per observation. Emit this from the harness so it is
      collected for every cell automatically.
- [ ] **8.2** Produce the accuracy-vs-cost Pareto front and identify the **simplest adequate** model
      — the deliverable of problem #22, which explicitly separates research complexity from
      production complexity.

---

## Phase 9 — Engineering hygiene (do alongside, not after)

- [ ] **9.1** Remove hardcoded Colab/Linux paths. `train_vae_pipeline.py:15-17` and
      `train_lstm_vae_pipeline.py:15-17` hardcode `/content/drive/...` and
      `/root/.cache/kagglehub/...`. The committed `data.yaml` contains an absolute path from a
      different machine.
- [ ] **9.2** Fix `find_best_pt()` (`run_full_pipeline.py:62-72`). It falls back to a recursive
      `glob("**/best.pt")` from CWD, can pick up unrelated weights, and gates step-4 completion at
      `:258`. Subsumed by 3.6.3.
- [ ] **9.3** Wire `CONFIG` into the YOLO trainer. `config.py:8-10` (`yolo_epochs/imgsz/batch`) are
      **dead values** — `train_yolo_n.py:8-11` hardcodes 150/640/16 instead of reading them.
- [ ] **9.4** Remove `device=0` from `train_yolo_n.py:33` — it forces CUDA while every other module
      uses the MPS/CUDA/CPU `get_device()` path.
- [ ] **9.5** Fix `feature_extractor.py:83`'s bare `.squeeze()` (no `dim=`), which silently collapses
      unintended dimensions. Also: `create_feature_extractor` returns a closure over a **mutable**
      `features` list (`feature_extractor.py:72,80`) — not thread-safe — and the forward hook is
      never removed except at `run_full_pipeline.py:353`.
- [ ] **9.6** Drop the unused `hook` parameter from `step_flat_vae` (`run_full_pipeline.py:139`).
- [ ] **9.7** De-duplicate `NormalisedSeqDataset` re-instantiation (`sequence_dataset.py:117-126`),
      which doubles feature extraction per run.
- [ ] **9.8** Fix `create_yolo_dataset.py:35` (unused annotation on a `defaultdict`) and `:87`
      (`next(iter(box_set))[0]` re-derives the image path from an arbitrary set element).
- [ ] **9.9** Reconcile the notebooks with `scripts/`. All 6 are Colab exports with hardcoded paths;
      cells 26/28/30/31 of `Cow_Anomaly_Detection_Models.ipynb` are verbatim duplicates containing the
      *same* min-max-then-random-split leakage bug, which suggests the bug was copy-pasted rather
      than fixed. Either delete them or clearly mark them as archived.
- [ ] **9.10** Update `README.md` after Phases 1–4 and Phase 10. The pipeline table (steps 6/7) and
      the outputs table currently describe a 7-step train-only pipeline with no evaluation stage,
      and will be wrong once the benchmark harness and baselines land.

---

## Phase 10 — Pipeline orchestrator conformance

`scripts/run_full_pipeline.py` (370 lines) is the 7-step linear orchestrator. Every phase above
changes what it must do, and the file is structured so that each new step has to be wired in
**three** separate places. This phase conforms the orchestrator to the new architecture. It evolves
continuously rather than landing as one batch — items 10.1.1–10.1.2 should land early so the
remaining phases have somewhere to plug in.

### 10.1 Structure — replace the hand-unrolled step list

- [ ] **10.1.1** Replace the hand-unrolled `if/else` chain in `main()` (`:295-351`) with a **step
      registry** driven by `STEP_NAMES` (`:29-37`). Adding a step currently requires editing three
      places: the `STEP_NAMES` dict, `is_step_done` (`:248-263`), and the `main()` body. This is the
      root cause of the drift Phase 1/3/4 fixes would otherwise bake in.
- [ ] **10.1.2** Derive `--from-step` bounds from the registry. It is hardcoded as
      `choices=range(1, 8)` (`:271`), so it silently caps at 7 the moment an 8th step is added.
- [ ] **10.1.3** Extend `STEP_NAMES` to cover the new stages. Proposed target list:
      1 Download · 2 YOLO dataset · 3 data YAML · 4 Train YOLO · **5 Build split manifest** (Phase 1.4)
      · **6 Extract + cache features** (0.3 / 5.1 / 5.2) · **7 Build anomaly ground-truth manifests**
      (6.6) · **8 Fit detectors** (4.3) · **9 Run benchmark** (4.2) · **10 Aggregate + report** (4.7).
      Re-derive the numbering rather than assuming this one.
- [ ] **10.1.4** Collapse `step_flat_vae` (`:139-175`) and `step_lstm_vae` (`:178-243`) into a
      single **model-fit step that loops over the Phase 4.3 roster**. These two functions are
      model-specific; they cannot scale to 12 models and will duplicate once more per baseline.
- [ ] **10.1.5** Promote step 5 to a real artifact-producing step. It currently only builds an
      in-memory object and has no branch in `is_step_done`, so it silently re-runs every time; with
      the Phase 0.3 cache it becomes "extract + persist features" with a genuine skip predicate.

### 10.2 Step responsibility boundaries

- [ ] **10.2.1** Stop training steps from owning data preparation. `step_flat_vae` does
      CSV read → feature extraction → normalization → split → train → save in one function
      (`:143-173`). Extract data prep into a shared stage so every model in the roster consumes the
      same cached features and the same split manifest.
- [ ] **10.2.2** Deduplicate the annotation read. `pd.read_csv` with the identical 8-column schema
      appears at `:143` and `:182`, and again in `train_vae_pipeline.py:28` and
      `train_lstm_vae_pipeline.py`. Read once, pass the DataFrame.
- [ ] **10.2.3** Eliminate the double feature extraction between the two current model steps. The
      flat VAE extracts via `build_feature_dataset` (`:147`) and the LSTM-VAE builds
      `CowSequenceDataset` (`:186`), which runs YOLO over the same frames again. Resolved by the
      0.3 cache; the orchestrator must then not trigger it twice.
- [ ] **10.2.4** Place the split-manifest step **before** every model stage and have all of them
      consume it (Phase 1.4). Today the flat VAE and LSTM-VAE each compute their own split inline
      (`:155` vs `:217-223`) — which is precisely why they disagree.
- [ ] **10.2.5** Adopt a per-model, versioned artifact layout. Steps 6/7 write flat files into the
      output-dir root (`:171-173`, `:240-242`) with no model/config namespace, which will collide
      once 12 models x 5 seeds write to one directory.

### 10.3 Skip and resume semantics

- [ ] **10.3.1** Rewrite `is_step_done` (`:248-263`) to validate checkpoint provenance — config
      hash, data fingerprint, seed — per Phase 3.6.3, instead of bare file existence. Steps 6 and 7
      currently pass iff their `.pth` exists, which is how a stale model gets silently reused.
- [ ] **10.3.2** Make `save_state` (`:56-59`) atomic. It has the same torn-write hazard as 3.2.1 —
      and `.pipeline_state.json` is the file that decides what gets skipped on the next run.
- [ ] **10.3.3** Fix the `--from-step` help text (`:272`). It reads "ignoring prior state", but the
      logic (`if force_step <= N and is_step_done(N)`) does the opposite: it forces a re-run of
      step N onward while still skipping any later step whose output exists.
- [ ] **10.3.4** Record step outputs into `state`. `step_flat_vae` and `step_lstm_vae` return `None`,
      so `state["last_step"]` advances without recording what was produced; the artifacts are not
      traceable from the state file.
- [ ] **10.3.5** Thread the Phase 3.3.2 resume API through steps 4 and 8. `--from-step 4` currently
      restarts YOLO at epoch 1 regardless of how far the previous run got.
- [ ] **10.3.6** Validate that each step's declared outputs actually exist before advancing
      `last_step`, so a silently-failed step is not recorded as complete.

### 10.4 Resource handling

- [ ] **10.4.1** Wrap steps 5→end in `try/finally` so `hook.remove()` (`:353`) always runs. Any
      exception between `:329` and `:353` leaks the YOLO forward hook and keeps the module resident.
- [ ] **10.4.2** Make `plot_history` non-blocking (0.5). It is called at `:174` and will halt any
      headless run mid-pipeline.

### 10.5 CLI

- [ ] **10.5.1** Add a `--device` override. `get_device()` (`:40-45`) is unconditional, and
      `train_yolo_n.py:33` independently hardcodes `device=0`, so the two can disagree.
- [ ] **10.5.2** Add `--config PATH` to run from an explicit config file. Currently only the
      module-level `CONFIG` (`scripts/config.py`) is honored.
- [ ] **10.5.3** Persist the resolved config snapshot per run. `config = CONFIG.copy()` (`:279`) is
      a shallow copy that is never written to disk, so a run cannot be reproduced from its outputs.
- [ ] **10.5.4** Add `--list-steps` and `--dry-run`. The Phase 4 grid is ~12 models x 3 anomaly
      settings x 5 seeds; it must be inspectable before launch.
- [ ] **10.5.5** Add `--only-model NAME` / `--only-steps` for targeted reruns, so one baseline can
      be re-benchmarked without re-running the detector.

### 10.6 Reporting

- [ ] **10.6.1** Fix the artifact listing (`:361-366`). `os.path.getsize` is applied to directories
      as well as files; `pipeline_output/` already contains `yolo_dataset/` with 5,853 files, which
      will print as a meaningless few-KB size. Separate files from directories or recurse.
- [ ] **10.6.2** Print the Phase 4.7.4 results-index path and the 4.2.3 coverage summary at
      completion, so an incomplete or failed grid is visible immediately rather than at paper time.
- [ ] **10.6.3** Report the run manifest (0.10) at completion — git SHA, config hash, seed, feature
      hash — so every artifact in the directory is attributable.

### 10.7 Tests

- [ ] **10.7.1** Registry consistency test: every step in `STEP_NAMES` has a runner, a skip
      predicate, and an entry in the execution order. Catches the drift that produced the missing
      step-5 branch in `is_step_done`.
- [ ] **10.7.2** Smoke test: dry-run every step against a tiny config (small epochs, few videos) so
      the orchestrator is exercised without a 12-model grid.

---

## Paper-facing framing

- [ ] **P.1** Reframe from "a better VAE" to the mechanism question (problem #23). The abstract in
      `improvements_on_vad.md:10-20` already does this correctly — keep that framing and make sure
      the harness produces the evidence for it.
- [ ] **P.2** Report negative results explicitly. `improvements_on_vad.md:355-367` lists four
      findings that would each be publishable. Do not design the benchmark so one method must win.
- [ ] **P.3** State the novelty definition and its scope in the paper body, not a footnote (6.1).
      One short paragraph: anomaly := unseen during training; the detector is a novelty flagger; no
      clinical or veterinary claim is made, and disease precursors are future work.
- [ ] **P.4** Report significance, not just point estimates — every headline delta should carry the
      4.4.2 test result and the seed spread.
- [ ] **P.5** Lead with false alerts per camera-day (4.5.3) alongside AUROC. For a flagger, the
      operating point matters more than the ranking metric — an AUROC figure alone does not say
      whether the system is usable.

---

## Critical path

```text
Phase 0 (seeding, caching, manifest)
   -> Phase 1 (leakage + splits)      <- nothing is trustworthy without this
      -> Phase 2 (comparable losses)  <- nothing is comparable without this
         -> Phase 3 (checkpointing)   <- nothing is resumable or auditable without this
            -> Phase 4 (harness)      <- nothing can be measured without this
               -> Phase 6 (anomaly definition) -> Phase 7 (ablations) -> Phase 8 (compute)
```

Phase 5 (tracking + box-conditioned features) is a parallel track that should start early — item
**5.1** in particular, because the current frame-global pooling means the per-cow feature vector is
identical for every cow in a frame, which undermines the premise of the whole pipeline. Phase 9 is
continuous and independent.

Phase 4 depends on Phase 3: the harness must be able to resume and audit every cell it runs, and
the coverage matrix in 4.2.2 is only trustworthy if each cell's checkpoint records the config,
seed, and data fingerprint it was produced from.

Phase 10 spans the whole path rather than sitting at the end of it — the orchestrator has to absorb
each phase's outputs as they land. Its structural items (**10.1.1** registry, **10.1.2** step bounds)
should land before Phase 3/4 so those phases have a registration point rather than another
hardcoded block. The remaining conformance items close once Phases 1–4 are done, since the step list
cannot be finalized until the benchmark and report stages exist.

---

## Suggested first actions

1. **1.1 / 1.2 / 1.3 / 1.4** — the four leakage fixes. Cheap, mechanical, and invalidate every
   number produced so far.
2. **10.1.1 / 10.1.2** — the orchestrator step registry and derived `--from-step` bounds. Small,
   and it removes the three-place edit pattern that every later phase would otherwise inherit.
3. **3.1 / 3.2** — checkpoint schema and atomic writes. Small, and everything afterwards
   (including the existing artifacts, which are scientifically void) becomes discardable safely.
   Cheaper to do before the harness exists than after.
4. **4.1 / 4.2 / 4.3** — the detector interface, runner, and coverage matrix, with Euclidean +
   Mahalanobis as the first two rows. This is the smallest experiment that answers RQ3, and it
   immediately tests problem #11 (is the VAE doing anything, or just the YOLO representation?).
5. **5.1** — make the feature extractor box-conditioned. Until per-cow features actually differ, all
   downstream temporal results are about frame-level image statistics, not cow behavior.

---

## Open questions for the author

- **Q1** Is AVA `target_id` reliable enough to use as identity, or is 5.2 (real tracking) required
      before any temporal experiment is valid? This determines whether Phase 5 gates Phase 7.
- **Q2** ~~Are expert-annotated abnormal events obtainable?~~ **Resolved — no longer needed.** Under
      the settled framing (novelty = unseen), ground truth is generated by the leave-one-action-out
      driver (6.2), not annotated. No veterinary expertise is required, and none is claimed.
- **Q3** Which novelty setting is the **primary** result? Leave-one-action-out (6.2), unseen
      transitions (6.4), or corruption operators (6.5)? Each supports different claims, and the
      harness grid in 4.2.1 should be scoped
      accordingly.
- **Q4** The YOLO26n run early-stopped at 126/150 epochs (`runs/detect/cow_detector/yolo26n_cbvd/results.csv`,
      final mAP50 0.846, mAP50-95 0.430). Is the detector frozen for the paper, or will it be
      retrained? Every feature vector — and therefore every cached feature array keyed by weights
      hash (0.3) — depends on this choice.
- **Q5** How many seeds and benchmark cells are affordable? The Phase 4 grid is
      ~12 models x 3 anomaly settings x 5 seeds = ~180 runs, plus the `k` sweep. This determines
      whether the full matrix from problem #24 is practical or needs staged execution.
