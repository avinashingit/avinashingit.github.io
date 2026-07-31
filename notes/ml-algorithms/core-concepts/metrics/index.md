---
layout: note
title: "Metrics"
description: "Three layers of metrics — a strong answer touches all three:"
note: true
note_collection: "ML algorithms"
note_section: "Core Concepts"
section_order: 1
note_order: 9
updated: 2026-06-06 11:30:50 -0700
keywords:
  - Training
  - Optimization
  - Evaluation
  - Inference
  - Supervised Learning
math: true
mermaid: false
---
## Mental model

Three layers of metrics — a strong answer touches all three:

1. **Business metric** — what the company cares about (CTR, revenue, retention, abuse reports)
2. **Model metric (offline)** — measurable on a labeled dataset without shipping (AUC, NDCG, MSE)
3. **Guardrail metrics** — what must _not_ break (latency, fairness, downstream surface health)

---

## Foundational quantities (binary classification)

Confusion matrix terms (used everywhere downstream):

- **TP** — true positive (predicted positive, actually positive)
- **FP** — false positive (predicted positive, actually negative)
- **TN** — true negative (predicted negative, actually negative)
- **FN** — false negative (predicted negative, actually positive)

Core rates:

- **Precision** = TP / (TP + FP) _"Of the ones I flagged, how many were right?"_
- **Recall (TPR, Sensitivity)** = TP / (TP + FN) _"Of the ones I should have caught, how many did I catch?"_
- **FPR** = FP / (FP + TN) _"What fraction of negatives did I wrongly flag?"_
- **Specificity (TNR)** = TN / (FP + TN) = 1 − FPR
- **Accuracy** = (TP + TN) / (TP + FP + TN + FN) _Misleading on imbalanced data — avoid as a primary metric._

Trade-off intuition: **precision and recall pull against each other.** Lowering the decision threshold catches more positives (↑ recall) but flags more negatives (↓ precision). The right operating point depends on the **cost asymmetry** between FP and FN.

---

## Binary classification

Use for: fraud, spam, fake accounts, churn, conversion prediction.

### F1 and variants

- **F1** = 2 · (Precision · Recall) / (Precision + Recall) Harmonic mean. Use when FP and FN costs are roughly equal.
- **Fβ** = (1 + β²) · (Precision · Recall) / (β² · Precision + Recall) β > 1 weights recall more (e.g., F2); β < 1 weights precision more (e.g., F0.5).

### Probability / calibration metrics

- **Log loss (binary cross-entropy)** = −(1/N) · Σ [yᵢ · log(pᵢ) + (1 − yᵢ) · log(1 − pᵢ)] Penalizes confident wrong predictions hardest. Standard training loss.
- **Brier score** = (1/N) · Σ (pᵢ − yᵢ)² Calibration-sensitive. Lower is better.

### Threshold-free metrics

- **ROC-AUC** — area under the TPR vs FPR curve as the threshold sweeps. Probability that a random positive is ranked above a random negative. **Misleading on imbalanced data** — a model can score 0.99 ROC-AUC and still be useless when positives are 1-in-10,000.
- **PR-AUC** — area under the Precision vs Recall curve. **Preferred for imbalanced problems** (abuse, fraud, rare-event detection).

### Operating-point metrics (use when costs are asymmetric)

- **Precision @ fixed Recall** — "precision when we catch 80% of fakes"
- **Recall @ fixed FPR** — "recall when we wrongly flag <0.1% of reals"

### Business metrics (binary)

- Fraud $ prevented per week
- Abuse reports per 1K active users
- False-block appeal rate (guardrail)
- Downstream conversion rate

**Anti-pattern:** reporting accuracy on imbalanced data.

---

## Multi-class classification

Use for: topic classification, intent detection, content categorization.

- **Macro-F1** = (1/K) · Σ F1ₖ Equal weight per class. Exposes weak classes — use when tail performance matters.
- **Weighted-F1** = Σ (nₖ / N) · F1ₖ Weight by class frequency. Use when head performance matters more.
- **Micro-F1** — pool TP/FP/FN across classes, then compute F1. Equivalent to accuracy in single-label multi-class.
- **Top-k accuracy** = (1/N) · Σ 𝟙[yᵢ ∈ top-k predictions] Use when downstream UI shows k options.
- **Confusion matrix** — always inspect, don't just report aggregates.

---

## Ranking

Use for: search, feed, recommendations, PYMK, ads.

Setup: for each query/user, a list of items with relevance scores rel(i) (binary or graded).

### NDCG@k (workhorse)

- **DCG@k** = Σᵢ₌₁ᵏ (2^rel(i) − 1) / log₂(i + 1)
- **NDCG@k** = DCG@k / IDCG@k IDCG is DCG of the ideal ranking. Normalizes to [0, 1] so queries are comparable. Use when relevance is graded (0/1/2/3).

### Binary-relevance ranking metrics

- **Precision@k** = (relevant items in top k) / k
- **Recall@k** = (relevant items in top k) / (total relevant items)
- **MAP@k (Mean Average Precision)** = mean over queries of: AP@k = (1/R) · Σᵢ₌₁ᵏ Precision@i · 𝟙[item i is relevant] R = total relevant items. Rewards relevant items appearing earlier.
- **MRR** = (1/Q) · Σ (1 / rankᵢ) Reciprocal rank of the first relevant result. Use when only the first hit matters (one-shot queries, voice).
- **Hit rate@k** = fraction of queries with at least one relevant item in top k.

### Two-stage systems

- Evaluate **retrieval** with Recall@k (typically k = 100–1000): did the right item make it into the candidate set?
- Evaluate **ranking** with NDCG@k or CTR replay on the retrieved candidates.
- **Evaluation leak warning:** don't score the ranker on a candidate set the ranker helped generate.

### Business metrics (ranking)

CTR, conversion rate, dwell time, sessions per user, downstream engagement, accepted-invite rate.

---

## Regression

Use for: price prediction, ETA, LTV, demand forecasting.

- **MAE** = (1/N) · Σ |yᵢ − ŷᵢ| Robust to outliers, units match the target. Easy to explain.
- **MSE** = (1/N) · Σ (yᵢ − ŷᵢ)² Standard training loss. Penalizes big misses.
- **RMSE** = √MSE Same units as target. Use when big misses are disproportionately bad.
- **MAPE** = (100/N) · Σ |(yᵢ − ŷᵢ) / yᵢ| Scale-invariant. **Breaks near zero** (division by zero) — avoid when target can be small.
- **sMAPE** = (100/N) · Σ |yᵢ − ŷᵢ| / ((|yᵢ| + |ŷᵢ|) / 2) Symmetric variant, bounded [0, 200]. Better near-zero behavior.
- **R²** = 1 − (Σ (yᵢ − ŷᵢ)² / Σ (yᵢ − ȳ)²) Variance explained. Useful for comparing models on the same dataset; misleading across datasets.
- **Quantile loss (pinball loss)** = Σ max(τ(yᵢ − ŷᵢ), (τ − 1)(yᵢ − ŷᵢ)) Use when you need a _distribution_ (P50, P90), not a point estimate. Critical for ETA-type problems where the worst case matters.

### Business metrics (regression)

Revenue impact, on-time rate, inventory cost, forecast bias ($ over/under).

---

## Retrieval / embedding / nearest-neighbor

Use for: semantic search, item-to-item recs, candidate generation, dedup.

- **Recall@k** — primary metric. Did the relevant item land in top k?
- **MRR / NDCG** — secondary if ordering within the retrieved set matters.
- **Alignment** = E[‖f(x) − f(x⁺)‖²] over positive pairs Positive pairs should have similar embeddings (small distance).
- **Uniformity** = log E[exp(−t · ‖f(x) − f(y)‖²)] over random pairs Embeddings should spread on the unit sphere; lower = more uniform.
- **Hubness** — distribution of how often each item appears as someone's nearest neighbor. Heavy-tailed = bad (one embedding is everyone's neighbor).

---

## Generative / LLM

Use for: summarization, generation, RAG, chat.

### Automated (cheap, noisy)

- **BLEU** — n-gram precision against reference, with brevity penalty. Translation legacy.
- **ROUGE-N / ROUGE-L** — n-gram or longest-common-subsequence recall against reference. Summarization legacy.
- **BERTScore** — token-level cosine similarity between generated and reference embeddings.
- **Perplexity** = exp(− (1/N) · Σ log p(xᵢ | x<ᵢ)) Language-model-internal metric, not directly task-aligned.

### Modern eval

- **LLM-as-judge** — pairwise win rates against a baseline, scored by a strong reference model. Now standard.
- **Human eval** — gold standard. Win rate, helpfulness, faithfulness, harmlessness on a rubric.
- **Faithfulness / groundedness** — for RAG, did the answer come from the retrieved context?

### Business metrics (generative)

Task completion rate, retention, thumbs-up rate, edit distance from user (did they accept the output as-is).

---

## Clustering / unsupervised

Use for: segmentation, anomaly detection (when labels are scarce).

- **Silhouette score** = (b − a) / max(a, b), averaged over points a = mean intra-cluster distance, b = mean nearest-other-cluster distance. Range [−1, 1]; higher is better.
- **Davies-Bouldin** — average ratio of intra-cluster scatter to inter-cluster distance. Lower is better.
- **Calinski-Harabasz** — between-cluster variance / within-cluster variance. Higher is better.
- **Stability** — do clusters reproduce on resampled data?
- For **anomaly detection** with labels: PR-AUC against the labeled set. Without labels: expert review on top-scored anomalies.

---

## Time series forecasting

Use for: demand, traffic, KPI forecasting.

- **MAE / RMSE / MAPE / sMAPE** as in regression.
- **Forecast bias** = (1/N) · Σ (ŷᵢ − yᵢ) Systematic over- or under-prediction. Often more actionable than error magnitude.
- **MASE (Mean Absolute Scaled Error)** = MAE / MAE_naive Scaled against a naive baseline (e.g., last value). >1 means worse than naive.
- **Backtesting with rolling windows** — never random split for time series.

---

## Guardrail metrics (every problem)

Should appear in every A/B section:

- **Latency** — p50, p95, p99 at serving.
- **Coverage** — % of requests the model can score (vs. fallback).
- **Fairness / slice metrics** — performance broken out by demographic / geo / language / cohort.
- **Stability** — model output distribution day-over-day. Sudden drift = upstream broke.
- **Downstream surface metrics** — did fixing this break that? (Better PYMK that tanks feed engagement, etc.)

---

## Quick lookup by problem type

| Problem type                       | Primary offline                       | Watch out for                                   |
| ---------------------------------- | ------------------------------------- | ----------------------------------------------- |
| Binary classification (balanced)   | ROC-AUC, F1                           | —                                               |
| Binary classification (imbalanced) | PR-AUC, Recall@FPR                    | Don't use accuracy or ROC-AUC alone             |
| Multi-class                        | Macro-F1 (tail) or Weighted-F1 (head) | Inspect confusion matrix                        |
| Ranking (graded relevance)         | NDCG@k                                | Evaluation leak in two-stage systems            |
| Ranking (binary relevance)         | MAP@k, MRR                            | Choose by whether all-hits or first-hit matters |
| Retrieval                          | Recall@k                              | Hubness in embedding space                      |
| Regression                         | MAE or RMSE                           | MAPE breaks near zero                           |
| Forecasting                        | MASE, forecast bias                   | Rolling-window backtest, not random split       |
| Generative / LLM                   | LLM-as-judge, human eval              | BLEU/ROUGE are weak proxies                     |
| Clustering                         | Silhouette, stability                 | Metrics often disagree — sanity check visually  |
