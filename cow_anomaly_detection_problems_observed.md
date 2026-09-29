# Problems Observed in the Cow Behavioral Anomaly Detection Study

## Purpose

This document records the methodological and experimental problems identified so far in the current cow video anomaly-detection project and the proposed empirical paper.

The goal is **not** to assume that the proposed models are wrong. The goal is to identify the issues that must be controlled before conclusions such as "temporal modeling helps," "VAE is better," or "Mahalanobis distance improves detection" can be made.

## 1. The dataset is primarily a behavior-recognition dataset, not an anomaly-detection benchmark

CBVD-5 is designed around cow behavior recognition. It does not automatically provide a scientifically grounded definition of a **behavioral anomaly**.

Therefore:

> A model detecting an unusual behavior is not automatically detecting an abnormal or undesirable behavior.

For example, if `drinking` is excluded from training and detected as unusual, that demonstrates **held-out behavior detection**, not necessarily pathological anomaly detection.

The paper must explicitly define what counts as an anomaly. Possible evaluation settings include held-out behavior, synthetic temporal anomalies, abnormal transition sequences, expert-labeled abnormal behavior, or genuinely observed abnormal events. These should not be conflated.

## 2. The current FlatVAE/LSTM-VAE comparison does not establish that temporal modeling helps

A lower reconstruction loss for FlatVAE than LSTM-VAE does **not** demonstrate that FlatVAE is a better anomaly detector.

The relevant questions are whether temporal modeling improves AUROC, AUPRC, precision, recall, false alerts, or detection delay.

A model can have lower reconstruction error while producing worse separation between normal and anomalous observations.

### Required experiment

Compare:

```text
FlatVAE
vs
GRU predictor
vs
LSTM predictor
vs
LSTM-VAE
```

using the same evaluation protocol and anomaly definition.

## 3. The approximately 10x reconstruction-loss difference may partly be caused by loss reduction

The current VAE loss uses:

```python
F.mse_loss(recon_x, x, reduction='sum')
```

The magnitude therefore depends on the number of elements being summed. This is especially important when comparing frame-level and sequence-level models.

If the LSTM processes sequences of length `T`, total reconstruction error can naturally scale with `T`.

Therefore:

> A 10x difference in total reconstruction loss is not necessarily a 10x difference in reconstruction quality.

### Required fix

Report normalized reconstruction error, such as mean squared error per element, and keep sequence length constant when comparing models. Report the KL term separately.

## 4. Feature normalization leaks validation information

The current normalization computes:

```python
min_val = features.min(axis=0)
max_val = features.max(axis=0)
```

before splitting into training and validation data.

This means validation observations contribute to the normalization parameters.

That is data leakage.

### Correct procedure

```text
train videos
validation videos
test videos
```

Then fit normalization using **training data only** and apply it to validation/test data.

The same rule applies to means, covariance matrices, PCA, whitening, thresholds, and other learned preprocessing.

## 5. Frame-level splitting is inappropriate for temporal anomaly detection

Temporal observations are strongly correlated. Randomly splitting individual frames can place nearly identical neighboring frames in training and validation/test sets.

For example:

```text
frame 100 -> train
frame 101 -> validation
frame 102 -> train
frame 103 -> test
```

This can make evaluation unrealistically easy.

### Correct approach

Split by an independent unit such as video, cow, or recording session **before** constructing temporal windows.

## 6. Temporal windows must be identity-aware

A temporal sequence must represent the same cow over time.

The system should be:

```text
video
 ↓
YOLO
 ↓
tracking
 ↓
cow track
 ↓
temporal sequence
```

rather than constructing sequences from arbitrary feature vectors.

Otherwise a sequence could contain observations from multiple cows, which is not a meaningful behavioral trajectory.

## 7. Tracking is an important missing component

If temporal modeling is central to the paper, cow identity must be maintained. A lightweight tracker such as ByteTrack can provide:

```text
YOLO detections
      ↓
tracker
      ↓
track_id
      ↓
per-cow temporal sequence
```

Without tracking, it is difficult to establish that a temporal model is learning behavior rather than changes in detection composition.

## 8. Detection errors can be confused with behavioral anomalies

A bad feature vector can result from poor detection, partial crop, occlusion, motion blur, camera movement, small bounding box, missed detection, or tracker failure.

The anomaly detector could interpret this as:

> "The cow is behaving abnormally."

when the actual problem is:

> "The observation is unreliable."

The system should therefore distinguish **observation anomaly** from **behavioral anomaly** using quality signals such as detector confidence, bounding-box dimensions, track persistence, occlusion, blur, and missing observations.

## 9. A VAE may not be necessary at all

The current approach assumes:

```text
embedding
 ↓
VAE
 ↓
reconstruction error
 ↓
anomaly
```

But the same learned embedding may support a much simpler detector:

```text
embedding
 ↓
normal distribution
 ↓
Mahalanobis distance
 ↓
anomaly
```

or Isolation Forest.

Therefore the paper should not assume that generative modeling is necessary.

## 10. Mahalanobis distance introduces a covariance-estimation problem

Mahalanobis distance is:

\[
D_M^2(z)=(z-\mu)^T\Sigma^{-1}(z-\mu).
\]

It accounts for correlations between feature dimensions, unlike Euclidean distance.

However, estimating a full covariance matrix becomes difficult when feature dimension is large relative to the number of independent training samples or when features are highly correlated.

A naive matrix inverse can become unstable.

### Required comparison

Test:

- Euclidean distance;
- diagonal Mahalanobis;
- full covariance;
- regularized covariance;
- shrinkage covariance;
- PCA + Mahalanobis.

## 11. The YOLO representation itself may be doing most of the work

If the YOLO feature representation already separates behavioral states well, the anomaly detector may need to do very little.

The actual system could be:

```text
YOLO representation
        ↓
Mahalanobis
```

and perform nearly as well as:

```text
YOLO representation
        ↓
LSTM-VAE
```

If so, the important contribution is not the generative model. This is why simple baselines are necessary.

## 12. Static anomaly detection and temporal anomaly detection are different problems

A static detector asks:

> Is the current state unusual?

A temporal detector asks:

> Is this transition or next state unusual given the previous states?

For example, `standing -> lying` may be normal, while a repeated or unusual transition sequence may be anomalous. The individual frames may not be anomalous; the **sequence** may be.

The paper should distinguish **state anomaly** from **transition anomaly**.

## 13. Temporal modeling needs a strong non-generative baseline

If the only comparison is FlatVAE versus LSTM-VAE, the experiment cannot determine whether an improvement comes from temporal modeling, recurrence, generative modeling, or parameter count.

A better baseline is:

```text
z(t-k:t)
 ↓
GRU
 ↓
predict z(t+1)
 ↓
prediction error
```

This gives temporal modeling without a VAE. Comparing it with LSTM-VAE helps isolate the value of generative reconstruction.

## 14. A simple Markov model would be a valuable baseline

If behavior labels are available, estimate:

\[
P(b_t|b_{t-1}).
\]

Then use:

\[
A_t=-\log P(b_t|b_{t-1})
\]

as a temporal anomaly score.

This is computationally trivial. If a complicated LSTM-VAE cannot substantially outperform such a baseline, that is itself an important result.

## 15. Temporal context length has not been established

It is not known how much temporal context is actually required.

Test values such as:

```text
1
3
5
10
20
40
```

and evaluate AUROC, AUPRC, and latency versus context length.

This can determine whether anomalies require instantaneous state, short-term dynamics, or longer behavior context.

## 16. A temporal-shuffle experiment is important

A recurrent model can appear to be temporal without necessarily using meaningful temporal structure.

Take:

```text
z1 -> z2 -> z3 -> z4 -> z5
```

and shuffle the ordering while preserving the observations.

If performance remains nearly unchanged, the model may not be exploiting temporal order substantially.

This provides a useful sanity check for temporal-modeling claims.

## 17. Reconstruction error alone is not enough

A model can reconstruct an anomaly surprisingly well. Therefore high reconstruction error should not automatically be interpreted as high-quality anomaly detection.

The paper should evaluate discrimination directly.

## 18. Threshold selection can create misleading results

Anomaly detection produces a continuous score that must often be converted into normal/anomaly decisions.

If the threshold is selected using the test set, reported performance becomes optimistic.

### Correct protocol

```text
training
   ↓
validation
   ↓
threshold selection
   ↓
test
```

The test set should be reserved for final evaluation.

## 19. Class imbalance makes AUROC alone insufficient

Real anomaly detection will likely contain many more normal observations than anomalous observations.

AUROC can look strong even when a detector produces many false positives.

Therefore report:

- AUROC;
- AUPRC;
- precision;
- recall;
- false alerts/hour;
- false alerts/camera/day.

For a real monitoring system, false alerts per camera/day may be more operationally meaningful than reconstruction loss.

## 20. Frame-level metrics do not necessarily represent useful events

Suppose a cow produces an anomalous score for 100 consecutive frames. A frame-level evaluation counts 100 anomalous predictions, while operationally this may represent one event.

The paper should therefore evaluate both observation-level metrics and event-level metrics such as false events/day, event detection rate, detection delay, and event duration.

## 21. The 25 FPS sampling rate may be unnecessary

Behavioral changes such as lying, standing, rumination, and drinking occur over substantially longer timescales than individual video frames.

Running the anomaly model at every frame may waste computation.

A production experiment could compare several sampling rates and measure the performance/compute trade-off.

## 22. Research complexity and production complexity should be separated

The best research model does not necessarily need to be the production model.

The research study can compare:

```text
Euclidean
Mahalanobis
AE
VAE
GRU
LSTM
LSTM-VAE
```

while a production system might ultimately use:

```text
YOLO
 ↓
tracker
 ↓
compact embedding
 ↓
Mahalanobis / small temporal model
 ↓
temporal smoothing
 ↓
event detector
```

The paper should measure the performance/complexity trade-off rather than assuming the largest model is preferable.

## 23. The paper should not be framed simply as "a better VAE"

A weak framing would be:

> "We propose a new LSTM-VAE architecture for cow anomaly detection."

A stronger framing is:

> "We empirically investigate which modeling assumptions are necessary for cattle behavioral anomaly detection."

Then the result can be useful regardless of which method performs best.

For example, if Mahalanobis is close to GRU and VAE, that is itself a useful empirical result.

## 24. The central experimental matrix should isolate mechanisms

| Method | Static | Temporal | Generative | Covariance-aware |
|---|---:|---:|---:|---:|
| Euclidean | ✓ | | | |
| Mahalanobis | ✓ | | | ✓ |
| Isolation Forest | ✓ | | | |
| AE | ✓ | | ✓ | |
| VAE | ✓ | | ✓ | |
| Markov | | ✓ | | |
| GRU predictor | | ✓ | | |
| LSTM predictor | | ✓ | | |
| LSTM-VAE | | ✓ | ✓ | |
| GRU + Mahalanobis | | ✓ | | ✓ |

The purpose is to isolate the contribution of each mechanism.

## 25. The paper's most important unresolved issue is anomaly ground truth

Everything depends on what the experiment calls an anomaly.

The strongest version of the study would have:

```text
normal behavior
+
expert-labeled abnormal behavior
```

If that is unavailable, the paper must explicitly state that it evaluates controlled proxies such as novel behaviors, abnormal transitions, temporal corruptions, or synthetic anomalies.

It should not overclaim that these are equivalent to veterinary or real-world behavioral abnormalities.

# Proposed Central Research Question

The problems above suggest a stronger central question:

> **What level of modeling complexity is actually necessary for unsupervised cattle behavioral anomaly detection?**

With three primary subquestions:

1. **Does temporal modeling improve anomaly detection?**
2. **Does generative modeling provide additional benefit?**
3. **Does covariance-aware Mahalanobis scoring improve detection in learned feature space?**

The resulting study can compare **simple statistical methods -> generative methods -> temporal methods -> temporal-generative methods**, while controlling data leakage, identity, temporal context, anomaly definition, threshold selection, and computational cost.

The important result is not predetermined. The experiment is designed to find out which assumptions actually matter.
