# An Empirical Study of Static, Temporal, and Generative Methods
# for Unsupervised Cattle Behavioral Anomaly Detection

**Nahom Senay**  
Resonance Labs, Addis Ababa University  
September 23, 2026

## Abstract

Behavioral anomaly detection in cattle can be approached at several levels of complexity, ranging from distance-based detection in learned feature spaces to temporal prediction and generative reconstruction. It is not yet clear which of these mechanisms is necessary for reliable detection in a practical cattle monitoring system.

This paper proposes a controlled empirical study using object-level visual representations to investigate three questions:

1. Whether temporal modeling materially improves anomaly detection over static feature distributions.
2. Whether generative reconstruction provides an advantage over non-generative anomaly scoring.
3. Whether covariance-aware Mahalanobis distance improves detection compared with Euclidean distance and other lightweight baselines.

The study will compare statistical, autoencoding, variational, recurrent prediction, and recurrent generative approaches under controlled experimental conditions. Particular attention will be given to temporal context length, covariance estimation, temporal-order ablations, and computational cost.

The goal is not to propose another increasingly complex anomaly architecture, but to determine which mechanisms actually contribute to performance in cattle behavioral anomaly detection.

## 1. Introduction

Video-based monitoring provides a way to observe cattle continuously and may support early identification of unusual behavioral patterns. Existing video anomaly detection research contains a wide range of approaches based on reconstruction, prediction, memory mechanisms, object representations, and pseudo-anomaly generation [8, 7, 9]. However, improvements in benchmark performance are often accompanied by increasing architectural complexity, making it difficult to determine which components are actually responsible for the improvement.

This problem is particularly relevant for cattle monitoring. A practical system may already possess a strong object detector and therefore does not necessarily need to reconstruct raw video frames. Instead, the detector can provide an object-level representation of an individual cow, after which anomaly detection can be performed in feature space. This makes it possible to separate the contribution of representation learning, distribution modeling, temporal modeling, and generative reconstruction.

The central question of this study is therefore:

> Do cattle behavioral anomalies require temporal and generative modeling, or can substantially simpler feature-space anomaly detectors achieve comparable performance?

The question is decomposed into three research questions.

- **RQ1:** Does temporal modeling improve anomaly detection compared with static anomaly detection?
- **RQ2:** Does generative reconstruction improve anomaly detection compared with non-generative distribution-based methods?
- **RQ3:** What is the effect of Mahalanobis distance and covariance modeling on anomaly detection in learned cow representations?

The intended contribution is an empirical characterization of these design choices rather than a claim that one particular architecture is universally optimal.

## 2. Problem Formulation

Let a detected cow at time \(t\) be represented by a feature vector

\[
z_t \in \mathbb{R}^{d},
\]

obtained from an object-level visual encoder. The training data are assumed to contain predominantly normal behavior. The objective is to assign an anomaly score \(A_t\) to an observation or temporal segment.

The study considers four broad mechanisms:

1. Distance from the normal feature distribution.
2. Reconstruction of normal representations.
3. Prediction of future representations from temporal context.
4. Probabilistic or generative modeling of normal representations.

These mechanisms should not be treated as interchangeable. A static detector models whether the current state is unusual, whereas a temporal predictor can model whether the transition to the current or next state is unusual.

## 3. Research Hypotheses

The following hypotheses will be evaluated experimentally.

### 3.1 H1: Temporal Information

Temporal modeling improves detection when anomalous behavior is primarily expressed through unusual transitions rather than unusual instantaneous states.

Formally, a static detector primarily estimates deviations under a marginal distribution

\[
P(z_t),
\]

whereas a temporal model attempts to exploit

\[
P(z_t \mid z_{t-k}, \ldots, z_{t-1}).
\]

If anomalies are transition anomalies, the latter representation should provide additional information.

### 3.2 H2: Generative Modeling

Generative reconstruction is not necessarily required for effective anomaly detection when a learned feature representation already separates normal and unusual states sufficiently well.

This hypothesis will be tested by comparing autoencoder and variational autoencoder reconstruction scores with simpler statistical anomaly scores.

### 3.3 H3: Mahalanobis Distance

Covariance-aware distance can improve anomaly detection when normal feature dimensions are correlated and the covariance matrix can be estimated reliably.

For normal training representations, let

\[
\mu = E[z]
\]

and

\[
\Sigma = \operatorname{Cov}(z).
\]

The squared Mahalanobis distance is

\[
D_M^2(z) = (z-\mu)^	op \Sigma^{-1}(z-\mu).
\]

Unlike Euclidean distance, Mahalanobis distance accounts for the scale and correlation structure of the normal representation distribution [1].

## 4. Experimental Design

### 4.1 Representation

The first stage of the system detects and tracks individual cows. A visual feature vector is then extracted for each cow observation. The experiments should preserve cow identity and temporal order so that a temporal window never combines observations belonging to different animals.

The experimental pipeline is:

```text
video → cow detection → tracking → object feature → anomaly detector
```

The same feature representation should be used across the principal comparisons whenever possible. This prevents an improvement in the anomaly detector from being confounded with a change in the visual representation.

### 4.2 Static Baselines

The first group of experiments evaluates anomaly detection without temporal modeling.

#### 4.2.1 Euclidean Distance

Let \(\mu\) denote the mean normal feature vector. The baseline anomaly score is

\[
A_{\mathrm{Euc}}(z_t) = \|z_t-\mu\|_2^2.
\]

#### 4.2.2 Mahalanobis Distance

The covariance-aware baseline is

\[
A_{\mathrm{Mah}}(z_t)
= (z_t-\mu)^	op\Sigma^{-1}(z_t-\mu).
\]

Because covariance estimation becomes difficult in high-dimensional feature spaces, the study should compare several covariance estimators:

- Diagonal covariance.
- Full covariance.
- Regularized covariance.
- Shrinkage covariance.
- PCA followed by covariance estimation.

Shrinkage estimators such as the Ledoit–Wolf estimator are especially relevant when the feature dimension is large relative to the number of independent training samples [5].

#### 4.2.3 Isolation Forest

Isolation Forest provides a non-parametric tree-based baseline for detecting observations that are isolated from normal observations [4].

## 5. Generative Baselines

### 5.1 Autoencoder

An autoencoder maps

\[
z_t ightarrow h_t ightarrow \hat{z}_t
\]

and uses reconstruction error as an anomaly score:

\[
A_{\mathrm{AE}}(z_t) = \|z_t-\hat{z}_t\|_2^2.
\]

Autoencoder-based anomaly detection is a common reconstruction paradigm [6, 8].

### 5.2 Variational Autoencoder

The VAE introduces a probabilistic latent representation and optimizes a reconstruction term together with a Kullback–Leibler divergence term [2].

For anomaly detection, reconstruction error can be used as one score:

\[
A_{\mathrm{VAE,rec}}(z_t) = \|z_t-\hat{z}_t\|_2^2.
\]

Additional probabilistic scores may also be evaluated, but the comparison should explicitly distinguish reconstruction-based scoring from latent likelihood-based scoring.

The important empirical question is whether the VAE provides a measurable advantage over the simpler AE and statistical baselines.

## 6. Temporal Models

### 6.1 Markov Transition Baseline

If discrete behavior states are available, a simple transition model can estimate

\[
P(b_t \mid b_{t-1}).
\]

A transition anomaly score can then be defined as

\[
A_{\mathrm{Markov}}(t) = -\log P(b_t \mid b_{t-1}).
\]

This provides a very low-complexity temporal baseline.

### 6.2 Recurrent Predictor

A small GRU can model continuous feature dynamics:

\[
(z_{t-k}, \ldots, z_t) ightarrow \mathrm{GRU} ightarrow \hat{z}_{t+1}.
\]

The anomaly score is

\[
A_{\mathrm{GRU}}(t) = \|z_{t+1}-\hat{z}_{t+1}\|_2^2.
\]

This tests whether prediction of future feature states is sufficient without requiring a generative latent-variable model.

A corresponding LSTM predictor provides a recurrent architecture comparison [3].

## 7. Temporal Generative Models

The principal recurrent generative baseline is an LSTM-VAE. It receives a sequence

\[
z_{t-k:t}
\]

and learns a latent representation and reconstruction of the sequence.

This model tests whether combining temporal modeling and generative reconstruction provides a meaningful advantage over either mechanism alone.

The comparison is therefore:

| Model | Temporal | Generative |
|---|---:|---:|
| Euclidean | No | No |
| Mahalanobis | No | No |
| Isolation Forest | No | No |
| Autoencoder | No | Yes |
| VAE | No | Yes |
| Markov model | Yes | No |
| GRU predictor | Yes | No |
| LSTM predictor | Yes | No |
| LSTM-VAE | Yes | Yes |

## 8. Temporal Context Ablation

The amount of temporal context should be treated as an experimental variable. The study can compare window lengths such as

\[
k \in \{1,3,5,10,20,40\}.
\]

The resulting relationship

\[
\operatorname{performance}(k)
\]

can determine whether cattle behavioral anomalies require long temporal context or whether a small number of observations is sufficient.

A useful additional experiment is temporal shuffling. The temporal order of observations within a sequence is randomly permuted while preserving the individual observations. If a temporal model depends genuinely on temporal structure, its performance should degrade under this intervention.

## 9. Mahalanobis Ablation

The effect of Mahalanobis distance should be isolated from the effect of the visual representation.

The following comparisons are proposed:

1. Euclidean distance on the original feature.
2. Mahalanobis distance on the original feature.
3. Euclidean distance after PCA.
4. Mahalanobis distance after PCA.
5. Mahalanobis distance with diagonal covariance.
6. Mahalanobis distance with shrinkage covariance.

The central analysis is the difference between

\[
\Delta_{\mathrm{Mah}} =
\operatorname{AUROC}_{\mathrm{Mah}}
-
\operatorname{AUROC}_{\mathrm{Euc}},
\]

and the corresponding change in AUPRC and operational false-alert rate.

This determines whether covariance information provides practical value rather than merely theoretical sophistication.

## 10. Evaluation Protocol

### 10.1 Data Splitting

Train, validation, and test partitions should be separated by video and, where possible, by cow identity. Random frame-level splitting should be avoided because adjacent frames from the same sequence are strongly correlated and can produce leakage.

All feature normalization parameters, including means, variances, PCA components, and covariance estimates, must be fitted using training data only.

### 10.2 Detection Metrics

The primary metrics should include:

- AUROC.
- AUPRC.
- Precision and recall at a validation-selected threshold.
- False alerts per hour or per camera-day.
- Detection delay.
- Inference latency.

AUROC is useful for threshold-independent comparison, while AUPRC is particularly informative under class imbalance.

### 10.3 Operational Metrics

Because a deployed monitoring system must avoid alert fatigue, false alerts per unit time should be reported alongside classification metrics.

The anomaly score should also be converted into events using temporal persistence rather than treating every anomalous frame as an independent alert. For example, an event can require the score to remain above an entry threshold for a minimum duration and use a lower recovery threshold.

## 11. Defining Anomalies

A central methodological issue is that the CBVD-5 dataset is primarily a cow-behavior dataset rather than a dedicated real-world behavioral anomaly benchmark. Consequently, the experimental definition of “anomaly” must be made explicit.

Three evaluation settings are possible.

### 11.1 Held-Out Behavior

One behavior can be excluded from training and treated as an unseen state. This measures novel-behavior detection, but it should not automatically be interpreted as detection of pathological or undesirable behavior.

### 11.2 Synthetic Temporal Anomalies

Controlled temporal corruptions can be introduced, including:

- Sequence reversal.
- Frame skipping.
- Temporal repetition.
- Abnormal transition insertion.
- Temporal acceleration or deceleration.

Such experiments test sensitivity to known temporal irregularities, but the resulting anomalies are synthetic.

### 11.3 Expert-Annotated Abnormal Events

The strongest evaluation would use genuine abnormal behavioral events annotated by domain experts. If such annotations are unavailable, the paper should clearly distinguish controlled anomaly experiments from real-world abnormality detection.

## 12. Expected Scientific Findings

The study is deliberately designed so that no particular model needs to win. Several outcomes would be scientifically meaningful.

If Mahalanobis distance performs comparably to the VAE, this would suggest that the principal value lies in the learned visual representation and estimation of the normal distribution rather than in generative reconstruction.

If a GRU predictor substantially improves over static detectors, this would support the hypothesis that behavioral anomalies are partly transition anomalies.

If the GRU provides little improvement over Mahalanobis distance, this would suggest that instantaneous feature distributions contain much of the information needed for the evaluated anomaly definitions.

If the LSTM-VAE does not outperform simpler models, this would provide evidence against assuming that additional generative and recurrent complexity is automatically beneficial.

Conversely, if temporal generative models consistently outperform the simpler methods under genuine temporal anomaly settings, this would support the need for more expressive temporal modeling.

## 13. Computational Analysis

Accuracy should be considered together with computational cost. For each model, report:

- Parameter count.
- Inference latency.
- Throughput.
- GPU memory.
- Training time.
- Feature dimensionality.
- Computational cost per processed observation.

The practical objective is not to maximize architectural complexity but to identify the simplest model whose performance is adequate for the target application.

## 14. Proposed Experimental Matrix

| Method | Static | Temporal | Generative | Covariance |
|---|:---:|:---:|:---:|:---:|
| Euclidean | ✓ |  |  |  |
| Mahalanobis | ✓ |  |  | ✓ |
| Isolation Forest | ✓ |  |  |  |
| AE | ✓ |  | ✓ |  |
| VAE | ✓ |  | ✓ |  |
| Markov |  | ✓ |  |  |
| GRU predictor |  | ✓ |  |  |
| LSTM predictor |  | ✓ |  |  |
| LSTM-VAE |  | ✓ | ✓ |  |
| GRU + Mahalanobis |  | ✓ |  | ✓ |

## 15. Conclusion

This study frames cattle behavioral anomaly detection as a question of mechanism rather than architecture. The central experimental distinction is between unusual states, unusual transitions, and reconstruction failure. Mahalanobis distance provides a particularly useful low-complexity baseline because it tests whether covariance-aware modeling of normal feature representations is already sufficient.

The resulting benchmark can establish whether temporal modeling materially improves detection, whether generative reconstruction is necessary, and whether covariance-aware distance provides a measurable advantage. Such a study can also identify the computational cost associated with each mechanism, providing a principled basis for choosing between simple statistical detectors, recurrent predictors, and recurrent generative models.

## References

1. P. C. Mahalanobis. *On the generalized distance in statistics*. Proceedings of the National Institute of Sciences of India, 2(1):49–55, 1936.
2. D. P. Kingma and M. Welling. *Auto-encoding variational Bayes*. In International Conference on Learning Representations, 2014. https://arxiv.org/abs/1312.6114.
3. S. Hochreiter and J. Schmidhuber. *Long short-term memory*. Neural Computation, 9(8):1735–1780, 1997. doi:10.1162/neco.1997.9.8.1735.
4. F. T. Liu, K. M. Ting, and Z.-H. Zhou. *Isolation forest*. In 2008 Eighth IEEE International Conference on Data Mining, pp. 413–422, 2008. doi:10.1109/ICDM.2008.17.
5. O. Ledoit and M. Wolf. *A well-conditioned estimator for large-dimensional covariance matrices*. Journal of Multivariate Analysis, 88(2):365–411, 2004. doi:10.1016/S0047-259X(03)00096-4.
6. M. Sakurada and T. Yairi. *Anomaly detection using autoencoders with nonlinear dimensionality reduction*. In Proceedings of the MLSDA 2014 2nd Workshop on Machine Learning for Sensory Data Analysis, pp. 4–11, 2014. doi:10.1145/2689746.2689747.
7. R. Chalapathy and S. Chawla. *Deep learning for anomaly detection: A survey*. arXiv preprint arXiv:1901.03407, 2019. https://arxiv.org/abs/1901.03407.
8. G. Pang, C. Shen, L. Cao, and A. van den Hengel. *Deep learning for anomaly detection: A review*. ACM Computing Surveys, 54(2):1–38, 2021. doi:10.1145/3439950.
9. W. Luo, W. Liu, D. Lian, Junliang Tang, L. Duan, X. Peng, and S. Gao. *Video anomaly detection with sparse coding inspired deep neural networks*. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2017. https://openaccess.thecvf.com/content_ICCV_2017/html/Luo_A_Deep_Learning_ICCV_2017_paper.html.
