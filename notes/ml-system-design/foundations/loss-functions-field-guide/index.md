---
layout: note
title: "Loss Functions Field Guide"
description: "Every architecture in this vault is \"encoder + loss.\" Interviewers care that you pick the loss that matches the decision the system makes. This note covers each loss, its formul…"
note: true
note_collection: "ML system design"
note_section: "Foundations"
section_order: 1
note_order: 5
updated: 2026-06-09 12:31:24 -0700
keywords:
  - Training
  - Embeddings
  - Fraud
  - Evaluation
  - Retrieval
math: true
mermaid: false
---
Every architecture in this vault is "encoder + loss." Interviewers care that you pick the loss that matches the *decision* the system makes. This note covers each loss, its formula, its gradient intuition, and when to use it.

## 1. Binary cross-entropy (BCE / log loss)

For label $y \in \{0,1\}$ and predicted probability $p = \sigma(s)$ (sigmoid of logit $s$):

$$\mathcal{L} = -[y \log p + (1-y)\log(1-p)]$$

Gradient w.r.t. the logit is beautifully simple: $\frac{\partial \mathcal{L}}{\partial s} = p - y$. The push is proportional to how wrong you are.

**Use when:** the output must be a *probability* that downstream systems consume — pCTR for an ad auction, P(fraud) gating enforcement, P(same owner) in AM-06 The Pairwise Matcher. BCE is a **proper scoring rule**: it is minimized by the true probability, so it trains naturally calibrated models (see [Class Imbalance, Calibration, and PU Learning](/notes/ml-system-design/foundations/class-imbalance-calibration-and-pu-learning/)).

## 2. Softmax cross-entropy
Multi-class version; see [Softmax, Sampled Softmax, and the logQ Correction](/notes/ml-system-design/foundations/softmax-sampled-softmax-and-the-logq-correction/).

## 3. Focal loss

BCE with a modulating factor that down-weights easy examples:

$$\mathcal{L} = -(1-p_t)^\gamma \log p_t, \quad p_t = \begin{cases} p & y=1 \\ 1-p & y=0\end{cases}$$

With $\gamma = 2$: an easy negative classified at $p_t = 0.99$ contributes $(0.01)^2 = 10^{-4}$ of its BCE loss — effectively muted; a hard example at $p_t = 0.3$ keeps ~half its loss. Invented for one-stage object detection ([RetinaNet](/notes/ml-system-design/vision-and-waymo/object-detection-rcnn-to-detr/)) where background boxes outnumber objects 1000:1 and would otherwise swamp the gradient. Also used in fraud (1:10⁴ imbalance). Caveat: focal loss **destroys calibration** — recalibrate afterwards.

## 4. Pairwise ranking losses (BPR, RankNet)

When the product decision is *ordering* (which item ranks above which), train on **pairs**: for a positive item $i$ and negative $j$ for the same user, demand $s_i > s_j$.

- **BPR (Bayesian Personalized Ranking):** $\mathcal{L} = -\log \sigma(s_i - s_j)$
- **RankNet:** same idea, cross-entropy on the pair-preference probability.
- **Margin/hinge version:** $\max(0, m - (s_i - s_j))$ — zero loss once the gap exceeds margin $m$.

**Use when:** implicit feedback (clicks) where absolute probabilities are meaningless but relative preference is observable. Limitation: pairwise losses don't care *where* in the list errors occur — swapping ranks 1↔2 costs the same as 99↔100.

## 5. Listwise losses (LambdaRank/LambdaMART intuition)

Fix the pairwise limitation by weighting each pair's gradient by how much swapping that pair would change a list metric like NDCG ($|\Delta \text{NDCG}|$). LambdaMART = these "lambda" gradients + gradient-boosted trees; for a decade the strongest learning-to-rank method and still a respected baseline at Google/Bing scale. Say "pointwise vs pairwise vs listwise" and you've covered the LTR taxonomy.

## 6. Triplet loss

For anchor $a$, positive $p$ (same identity), negative $n$ (different identity), with distance $d$:

$$\mathcal{L} = \max\big(0,\; d(a,p) - d(a,n) + m\big)$$

"Same-identity pairs must be at least margin $m$ closer than different-identity pairs." Made famous by **FaceNet** (face verification). The catch: with random negatives, most triplets quickly satisfy the margin and give zero gradient → you must do **(semi-)hard negative mining** (see [Negative Sampling and Hard Negatives](/notes/ml-system-design/foundations/negative-sampling-and-hard-negatives/)). Directly applicable to account matching: see AM-05 The Siamese Behavioral Encoder.

## 7. InfoNCE / NT-Xent (contrastive)

The modern default for representation learning; covered in depth in [Contrastive Learning - InfoNCE, Triplet, Siamese](/notes/ml-system-design/foundations/contrastive-learning-infonce-triplet-siamese/). Mathematically it's a sampled softmax where the "correct class" is the matching pair.

## 8. Regression losses
- **MSE** $\frac{1}{2}(y-\hat y)^2$: penalizes big errors quadratically; sensitive to outliers; implicitly Gaussian.
- **MAE / Huber**: robust alternatives; Huber = MSE near zero, MAE in the tails.
- **Quantile (pinball) loss**: $\max(q(y-\hat y), (q-1)(y-\hat y))$ trains the model to predict the $q$-th quantile — the backbone of probabilistic forecasting (predict p10/p50/p90 demand).

## 9. Weighted logistic regression for expected-value targets (YouTube trick)

To rank by *expected watch time* rather than click probability: train logistic regression on click labels but **weight positive examples by their watch time**. The resulting odds approximate expected watch time (for small click rates). This shows up verbatim in the [YouTube RecSys Case Study](/notes/ml-system-design/recommendation-systems/youtube-recsys-case-study/) and is a favorite interview detail: "how would you optimize watch time, not clicks?"

## 10. Multi-task combinations

Real rankers predict many heads (pClick, pLike, pReport...) each with its own BCE, summed with weights: $\mathcal{L} = \sum_t w_t \mathcal{L}_t$. The *final ranking score* combines the predicted probabilities with **business-policy weights** tuned by A/B testing — emphasize in interviews that this combination is a product decision, not a modeling one. Architecture for sharing: [Multi-Task Learning - Shared Bottom, MMoE, PLE](/notes/ml-system-design/recommendation-systems/multi-task-learning-shared-bottom-mmoe-ple/).

## Decision table

| System decision | Loss |
|---|---|
| Output feeds an auction / threshold / human review queue (needs real probability) | BCE (+ calibration check) |
| Order a list from implicit feedback | BPR / LambdaRank |
| Learn an embedding space for similarity search | InfoNCE or triplet (+ hard negatives) |
| Retrieve from a huge catalog | Sampled softmax + logQ |
| Detect rare class drowned in easy negatives | Focal loss (then recalibrate) |
| Probabilistic forecast | Quantile loss |
