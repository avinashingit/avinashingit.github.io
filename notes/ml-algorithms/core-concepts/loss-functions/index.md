---
layout: note
title: "Loss Functions"
description: "where $d$ = distance between an embedding pair, $y=1$ if dissimilar, $m$ = margin."
note: true
note_collection: "ML algorithms"
note_section: "Core Concepts"
section_order: 1
note_order: 8
updated: 2026-06-29 22:24:23 -0700
keywords:
  - Training
  - Optimization
  - Deep Learning
  - Evaluation
  - Linear Models
math: true
mermaid: false
---
> A **loss** measures how wrong a single prediction is; the **cost** is the loss averaged over the dataset. Training = minimizing it via [Gradient Descent](/notes/ml-algorithms/core-concepts/gradient-descent/) / [backprop](/notes/ml-algorithms/core-concepts/back-propagation/). Picking the right loss encodes *what kind of mistakes you care about* and shapes the gradients the optimizer sees.

### How to choose (mental model)

- **Task type** — regression (continuous), classification (categorical), ranking/retrieval (relative order), generative (distribution matching).
- **Noise / outliers** — heavy-tailed errors → favor robust losses (MAE, Huber) over MSE.
- **Probabilistic grounding** — most losses are a **maximum-likelihood** (MLE) or **MAP** estimate under an assumed noise model: MSE ⇔ Gaussian noise, MAE ⇔ Laplace noise, CE ⇔ Bernoulli/Categorical.
- **Gradient health** — does the gradient vanish when confidently wrong? (see [Vanishing and Exploding Gradients](/notes/ml-algorithms/core-concepts/vanishing-and-exploding-gradients/)). Prefer losses whose gradient stays informative.

---

## Cheatsheet

| Loss | Task | Formula (per-sample) | Robust to outliers? |
|---|---|---|---|
| MSE / L2 | Regression | $(y-\hat y)^2$ | ✗ (penalizes² ) |
| MAE / L1 | Regression | $\lvert y-\hat y\rvert$ | ✓ |
| Huber | Regression | quadratic→linear | ✓ (tunable $\delta$) |
| Log-Cosh | Regression | $\log\cosh(y-\hat y)$ | ✓ (smooth) |
| Quantile / Pinball | Regression (intervals) | asymmetric L1 | ✓ |
| Cross-Entropy | Classification | $-\sum_k y_k\log\hat y_k$ | n/a |
| Hinge | Classification (SVM) | $\max(0,1-y\cdot s)$ | margin-based |
| Focal | Imbalanced classification | $-(1-\hat y)^\gamma\log\hat y$ | down-weights easy |
| KL Divergence | Distribution matching | $\sum p\log\frac{p}{q}$ | n/a |
| Contrastive / Triplet | Metric learning | margin on distances | n/a |
| InfoNCE / NT-Xent | Self-supervised | softmax over similarities | n/a |
| Dice / IoU | Segmentation | overlap ratio | handles imbalance |

---

## Regression Losses

### Mean Squared Error (MSE / L2)

$$L = \frac{1}{N}\sum_{i=1}^N (y_i-\hat y_i)^2$$

- **Intuition:** penalizes the *square* of the error, so large errors dominate. Minimizing MSE = predicting the **conditional mean** $\mathbb{E}[y\mid x]$. Equivalent to MLE under Gaussian noise.
- **Advantages:** smooth & convex (for linear models), differentiable everywhere, clean gradient $\propto (\hat y - y)$, single global minimum.
- **Disadvantages:** very sensitive to **outliers** (squared term blows up); the squared units aren't directly interpretable; can produce over-confident large gradients early in training.

### Mean Absolute Error (MAE / L1)

$$L = \frac{1}{N}\sum_{i=1}^N \lvert y_i-\hat y_i\rvert$$

- **Intuition:** penalizes error linearly. Minimizing MAE = predicting the **conditional median** ⇒ robust to outliers. MLE under Laplace noise.
- **Advantages:** robust to outliers; error is in the same units as the target (interpretable).
- **Disadvantages:** gradient is constant ($\pm1$) regardless of error size → slow/unstable convergence near the minimum; **non-differentiable at 0** (needs subgradient).

### Huber Loss (Smooth L1)

$$L_\delta = \begin{cases} \tfrac{1}{2}(y-\hat y)^2 & \lvert y-\hat y\rvert \le \delta \\[4pt] \delta\,\lvert y-\hat y\rvert - \tfrac{1}{2}\delta^2 & \text{otherwise} \end{cases}$$

- **Intuition:** the best of both — **quadratic** near zero (smooth gradients, fast convergence) and **linear** for large errors (robust). $\delta$ sets the transition point.
- **Advantages:** robust *and* differentiable; less outlier-sensitive than MSE, faster than MAE. (Used as "Smooth L1" in object-detection box regression.)
- **Disadvantages:** extra hyperparameter $\delta$ to tune; behavior depends on the scale of the targets.

### Log-Cosh

$$L = \sum_i \log\!\big(\cosh(y_i-\hat y_i)\big)$$

- **Intuition:** ≈ MSE for small errors, ≈ MAE (linear) for large ones — a *smooth* Huber with no hyperparameter and continuous second derivatives.
- **Advantages:** twice-differentiable (good for methods needing the Hessian, e.g. some boosters); robust-ish; no $\delta$ to tune.
- **Disadvantages:** slightly more expensive; still less robust than pure L1 for extreme outliers.

### Quantile / Pinball Loss

$$L_\tau = \sum_i \begin{cases} \tau\,(y_i-\hat y_i) & y_i \ge \hat y_i \\ (\tau-1)(y_i-\hat y_i) & y_i < \hat y_i \end{cases}$$

- **Intuition:** asymmetric L1 — penalizes under- vs over-prediction differently by quantile $\tau\in(0,1)$. $\tau=0.5$ recovers MAE (median); $\tau=0.9$ targets the 90th percentile.
- **Advantages:** enables **prediction intervals** / uncertainty without distributional assumptions; tune asymmetry to business cost (e.g. stockouts vs overstock).
- **Disadvantages:** train one model per quantile (or a multi-output head); quantile crossing can occur; non-smooth at 0.

---

## Classification Losses

### Cross-Entropy / Log Loss → see [Cross-Entropy Loss](/notes/ml-algorithms/core-concepts/cross-entropy-loss/)

$$L = -\sum_{k=1}^K y_k\log\hat y_k \quad\text{(BCE for }K=2\text{)}$$

- **Intuition:** distance between true and predicted distributions; equals **negative log-likelihood** of the correct class. Pairs with [Softmax](/notes/ml-algorithms/core-concepts/softmax/) (multiclass) or sigmoid (binary). See the full note: [Cross-Entropy Loss](/notes/ml-algorithms/core-concepts/cross-entropy-loss/).
- **Advantages:** strong, non-vanishing gradient $\hat y - y$ even when confidently wrong; probabilistically principled; default for classification.
- **Disadvantages:** sensitive to **class imbalance** and label noise; can be over-confident (mitigate with **label smoothing**); needs numerically-stable logit form.

### Hinge Loss (SVM / max-margin)

$$L = \max\big(0,\; 1 - y\cdot s\big), \quad y\in\{-1,+1\},\ s = \text{raw score}$$

- **Intuition:** zero loss once a point is on the correct side of the margin *by at least 1*; otherwise penalizes linearly. Drives **max-margin** separation (SVMs). Squared hinge penalizes margin violations quadratically.
- **Advantages:** encourages a robust decision margin; loss is exactly 0 for well-classified points → sparse support vectors.
- **Disadvantages:** **not** a probability (no calibrated confidence — see [Calibration](/notes/ml-algorithms/core-concepts/calibration/)); non-differentiable at the hinge; less common in deep nets than CE.

### Focal Loss

$$L = -\alpha\,(1-\hat y_t)^{\gamma}\,\log \hat y_t$$

- **Intuition:** a re-weighted cross-entropy. The modulating factor $(1-\hat y_t)^\gamma$ shrinks the loss for **easy, well-classified** examples ($\hat y_t\to1$), letting training focus on **hard** ones. Designed for extreme foreground/background imbalance (dense object detection, RetinaNet).
- **Advantages:** handles severe class imbalance without resampling; $\gamma$ (focusing) and $\alpha$ (class weight) tune the emphasis.
- **Disadvantages:** two extra hyperparameters; if $\gamma$ too high it ignores most data; can amplify the effect of noisy hard labels.

### KL Divergence

$$D_{KL}(p\parallel q) = \sum_i p_i\log\frac{p_i}{q_i}$$

- **Intuition:** "extra bits" to encode true $p$ using predicted $q$. Cross-entropy $=$ entropy $+$ KL, so minimizing CE = minimizing KL. Core to **knowledge distillation** (match teacher's soft targets), **VAEs** (regularize latent to a prior), and soft-label training.
- **Advantages:** principled distribution-matching; works with soft targets, not just one-hot.
- **Disadvantages:** **asymmetric** ($D_{KL}(p\|q)\neq D_{KL}(q\|p)$); undefined when $q_i=0$ but $p_i>0$; mode-seeking vs mean-seeking depending on direction.

---

## Metric Learning & Ranking Losses

### Contrastive Loss (pairs)

$$L = (1-y)\tfrac{1}{2}d^2 + y\,\tfrac{1}{2}\max(0, m-d)^2$$

where $d$ = distance between an embedding pair, $y=1$ if dissimilar, $m$ = margin.

- **Intuition:** pull similar pairs together, push dissimilar pairs apart until they're at least margin $m$ away. Foundation of Siamese networks.
- **Advantages:** learns an embedding space directly; great for verification / similarity (faces, signatures).
- **Disadvantages:** needs careful **pair mining**; margin $m$ is sensitive; only sees two points at a time.

### Triplet Loss

$$L = \max\big(0,\; d(a,p) - d(a,n) + m\big)$$

- **Intuition:** anchor $a$ should be closer to a positive $p$ than to a negative $n$ by margin $m$ — *relative* rather than absolute distances.
- **Advantages:** optimizes ranking of similarity directly; strong for face recognition (FaceNet) and retrieval.
- **Disadvantages:** triplet selection / **hard-negative mining** is critical and expensive; many triplets are "easy" and give zero gradient; can collapse without care.

### InfoNCE / NT-Xent (contrastive self-supervised)

$$L = -\log\frac{\exp(\text{sim}(z_i,z_j)/\tau)}{\sum_{k\neq i}\exp(\text{sim}(z_i,z_k)/\tau)}$$

- **Intuition:** a softmax over similarities — identify the one true positive among many negatives in the batch. Temperature $\tau$ sharpens the distribution. Powers SimCLR, MoCo, CLIP.
- **Advantages:** uses **many negatives at once** (more stable than triplet); no labels needed (self-supervised); scales with batch size.
- **Disadvantages:** demands **large batches / memory banks** for enough negatives; $\tau$ sensitive; representation collapse if positives/augmentations are poor.

### BPR (Bayesian Personalized Ranking)

$$L = -\sum \log \sigma\big(\hat x_{ui} - \hat x_{uj}\big)$$

- **Intuition:** for user $u$, an observed item $i$ should score higher than an unobserved item $j$ — a pairwise ranking objective for implicit-feedback recommenders.
- **Advantages:** optimizes ranking (AUC-like) directly from implicit signals; simple, effective for top-N recsys.
- **Disadvantages:** negative sampling quality matters; pairwise (not listwise) so misses higher-order list structure.

---

## Generative & Structured Losses

### Adversarial Loss (GANs)

$$\min_G\max_D\; \mathbb{E}_{x}[\log D(x)] + \mathbb{E}_{z}[\log(1-D(G(z)))]$$

- **Intuition:** a minimax game — discriminator $D$ learns real vs fake, generator $G$ learns to fool it. The "loss" is implicitly the divergence between real and generated distributions.
- **Advantages:** produces sharp, realistic samples; no explicit density needed.
- **Disadvantages:** **unstable** training, mode collapse, vanishing generator gradients; Wasserstein/​hinge GAN losses partly fix this.

### Dice Loss / IoU (segmentation)

$$L_{Dice} = 1 - \frac{2\sum p\,g + \epsilon}{\sum p + \sum g + \epsilon}$$

- **Intuition:** maximize overlap between predicted mask $p$ and ground truth $g$ (a soft F1). Directly optimizes the segmentation metric.
- **Advantages:** robust to severe **foreground/background imbalance** (small objects); aligned with the eval metric (IoU/Dice).
- **Disadvantages:** unstable gradients for tiny/empty masks; often combined with CE (e.g. `Dice + BCE`) for stability.

### CTC (Connectionist Temporal Classification)

- **Intuition:** loss for **unaligned** sequence prediction (speech→text, handwriting) — sums probability over all alignments of input frames to the target label sequence via a blank token.
- **Advantages:** no frame-level alignment labels needed.
- **Disadvantages:** assumes conditional independence per step; computationally heavier (forward-backward over alignments).

---

## Regularization terms (added to any loss)

The total objective is usually $L_{\text{data}} + \lambda\,R(\theta)$:

- **L2 (weight decay / Ridge):** $\lambda\sum\theta^2$ — shrinks weights smoothly, discourages large weights, improves generalization.
- **L1 (Lasso):** $\lambda\sum\lvert\theta\rvert$ — drives weights to exactly **0** → feature selection / sparsity.
- **Elastic Net:** mix of L1 + L2.

These don't fit the data; they constrain the hypothesis to fight overfitting (the bias–variance tradeoff).

---

## Common pitfalls

- **Scale matters:** MSE/Huber/log-cosh all depend on target scale → normalize targets.
- **Use logits, not probabilities,** in framework CE/BCE for numerical stability (log-sum-exp).
- **Imbalance:** plain CE skews toward the majority class → class weights, focal, or Dice.
- **Metric ≠ loss:** you optimize a differentiable surrogate (CE) but report a non-differentiable [metric](/notes/ml-algorithms/core-concepts/metrics/) (accuracy, F1, AUC) — watch the gap.
- **Outliers:** if errors are heavy-tailed, swap MSE → Huber/MAE before blaming the model.

---

### Related
- [Cross-Entropy Loss](/notes/ml-algorithms/core-concepts/cross-entropy-loss/) — the deep dive on CE / BCE / NLL
- [Softmax](/notes/ml-algorithms/core-concepts/softmax/) — produces the probabilities CE consumes
- [Activation Functions and Optimizers](/notes/ml-algorithms/core-concepts/activation-functions-and-optimizers/) — what minimizes these losses
- [Gradient Descent](/notes/ml-algorithms/core-concepts/gradient-descent/) · [Backpropagation](/notes/ml-algorithms/core-concepts/back-propagation/) — the optimization machinery
- [Vanishing and Exploding Gradients](/notes/ml-algorithms/core-concepts/vanishing-and-exploding-gradients/) — gradient health of a loss
- [Calibration](/notes/ml-algorithms/core-concepts/calibration/) · [Metrics](/notes/ml-algorithms/core-concepts/metrics/) — confidence vs the numbers you report
