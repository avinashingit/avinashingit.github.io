---
layout: note
title: "Class Imbalance, Calibration, and PU Learning"
description: "Three intertwined topics that dominate fraud/abuse interviews."
note: true
note_collection: "ML system design"
note_section: "Foundations"
section_order: 1
note_order: 2
math: true
mermaid: false
---
Three intertwined topics that dominate fraud/abuse interviews.

## 1. Class imbalance

Fraud base rates are 10⁻³–10⁻⁶. Naive training on 1:10⁵ data → the model predicts "never fraud," achieves 99.999% accuracy, and is useless. (Also why **accuracy is never the metric** — use PR-AUC, recall at fixed precision, or precision at a fixed review budget.)

**Standard playbook:**
1. **Downsample negatives** to something like 1:10 or 1:100 at training time. Keeps every rare positive, slashes compute. *But this distorts probabilities* (see calibration below).
2. **Upweight positives** in the loss (`pos_weight` in BCE) — gradient-equivalent to oversampling without duplicating data.
3. **Focal loss** when easy negatives swamp the gradient — see [Loss Functions Field Guide](/notes/ml-system-design/foundations/loss-functions-field-guide/).
4. Don't bother with SMOTE-style synthetic oversampling for deep models on industrial data; mention it only to dismiss it.

**Fixing probabilities after downsampling.** If you kept all positives and a fraction $\beta$ of negatives, the model's predicted odds are inflated by $1/\beta$. Closed-form correction of predicted probability $p'$ back to true probability $p$:

$$p = \frac{\beta p'}{\beta p' - p' + 1}$$

*Worked example:* train at 1:100 sampling ($\beta = 0.01$) on a true base rate of 0.1%. The model sees ~9% positives in training, so an "average" example gets $p' \approx 0.09$; the formula gives $p = \frac{0.01 \cdot 0.09}{0.01\cdot0.09 - 0.09 + 1} \approx 0.00099$ — the true 0.1%. Knowing this formula cold is a strong signal in ads/fraud interviews.

## 2. Calibration

A model is **calibrated** if among all events predicted at 0.8, ~80% actually happen. Ranking metrics (AUC) are invariant to any monotone distortion of scores — a model can rank perfectly and be wildly miscalibrated. Calibration matters whenever the score is *consumed as a probability*:
- **Ads:** bid = pCTR × value — 2× miscalibration = 2× misbid, real money.
- **Abuse enforcement:** "auto-ban above P(same owner) = 0.99" is meaningless if 0.99 really means 0.7. See AM-06 The Pairwise Matcher.
- **Thresholded review queues, risk scores, medical triage.**

**Measure:** reliability diagram (bin predictions, plot mean prediction vs empirical frequency) + ECE (expected calibration error). **Always check calibration per slice** (per country, per device type) — globally calibrated, per-slice biased is common and harmful.

**Fix (post-hoc, on a held-out calibration set):**
- **Platt scaling:** fit logistic regression $\sigma(as + b)$ on the model score $s$. Two parameters; good for small data.
- **Isotonic regression:** fit a monotone step function; more flexible, needs more data; the industry default.
- **Temperature scaling:** divide logits by learned $T$ (one parameter); the go-to for neural nets/softmax.

Causes of miscalibration: downsampling (above), focal loss, modern over-parameterized nets (overconfident), distribution shift after deployment (recalibrate on recent data, monitor ECE in prod).

## 3. PU learning (Positive–Unlabeled)

In abuse problems your "negatives" are really **unlabeled**: an account not flagged for ban evasion may be an undetected evader; a pair of accounts not labeled same-owner may simply never have been investigated. Treating unlabeled as negative:
- biases probabilities downward,
- punishes the model for correctly scoring undetected positives,
- and poisons [hard-negative mining](/notes/ml-system-design/foundations/negative-sampling-and-hard-negatives/) (the "hardest negatives" are disproportionately the undiscovered positives!).

**Framing:** let $c$ = label frequency = P(labeled | positive). Under the *Selected Completely At Random* (SCAR) assumption, a classifier trained on labeled-vs-unlabeled predicts $P(\text{labeled}|x) = c \cdot P(\text{positive}|x)$ — so dividing by an estimate of $c$ recovers the true positive probability (Elkan & Noto). Estimating $c$: average score over a held-out set of known positives.

**Practical techniques:**
- **Unbiased/non-negative PU risk estimators** (uPU/nnPU) — reweight the loss using an estimate of the class prior.
- **Spy / two-step:** plant known positives among the unlabeled; anything the first-pass model scores below the spies is treated as a *reliable negative*; retrain on positives vs reliable negatives.
- **In system design terms:** route a small random sample of "negatives" to human review to estimate the contamination rate and create gold negatives; treat reversed appeals as gold labels; report metrics with the contamination caveat. Saying "our negatives are unlabeled, so I'd estimate label frequency and use a PU-aware estimator, plus buy gold labels via sampled human review" is a senior-level answer.

Related: **label delay** (chargebacks arrive 30–90 days late; evasion confirmations take weeks) → train on data old enough for labels to mature, or model label maturation explicitly; **survivorship/feedback bias** (you only get labels where the previous system looked) → keep an exploration/holdout slice, see Monitoring, Drift, and Feedback Loops.
