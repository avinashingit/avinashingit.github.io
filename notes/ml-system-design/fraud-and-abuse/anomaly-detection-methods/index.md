---
layout: note
title: "Anomaly Detection Methods"
description: "When labels are scarce or the attack is novel, you detect deviation from normal instead of similarity to known-bad. Taxonomy by how much supervision you have:"
note: true
note_collection: "ML system design"
note_section: "Fraud and Abuse"
section_order: 5
note_order: 2
updated: 2026-06-09 12:44:02 -0700
keywords:
  - Fraud
  - Embeddings
  - Vision
  - Training
  - Retrieval
math: true
mermaid: false
---
When labels are scarce or the attack is novel, you detect *deviation from normal* instead of *similarity to known-bad*. Taxonomy by how much supervision you have:

## 1. Unsupervised / one-class

- **Isolation Forest (not deep — know it anyway):** build random trees that split on random features at random thresholds; anomalies isolate in **few splits** (short path length) because they sit in sparse regions. Fast, robust, the standard tabular baseline; an interviewer will respect "I'd start with isolation forest as a baseline."
- **Autoencoder reconstruction:** train encoder→bottleneck→decoder to reconstruct *normal* data; anomalies (off the learned manifold) reconstruct poorly → anomaly score = reconstruction error $\|x - \hat x\|$. VAE variant: use likelihood. Caveats: AEs can generalize to reconstruct anomalies too (too much capacity), and if training data is contaminated with anomalies they get learned as normal.
- **Deep SVDD:** train an encoder to map normal data inside a minimal hypersphere; score = distance from center. (Needs tricks to avoid the encoder collapsing everything to the center — no bias terms, fixed center.)
- **Density/distance in embedding space:** embed with any good encoder ([contrastive](/notes/ml-system-design/foundations/contrastive-learning-infonce-triplet-siamese/) or pretrained), score by kNN distance to training points — embarrassingly effective, easy to serve with [ANN](/notes/ml-system-design/foundations/approximate-nearest-neighbor-search/).

## 2. Sequence anomalies (the abuse-relevant one)

Model the **event stream**: train an autoregressive model (GRU or small causal [Transformer](/notes/ml-system-design/foundations/transformers-from-scratch/), exactly the [SASRec](/notes/ml-system-design/recommendation-systems/sequence-recommenders-din-sasrec-bert4rec/) recipe) to predict each next event given history; anomaly score = **perplexity** (how surprised the model is by what this account actually did). Catches: bot-like mechanical regularity (inter-event time entropy too LOW — bots are anomalously *predictable*), scripted action sequences, sudden behavior change inside one account (possible takeover or sale — see AM-10 Adversarial Dynamics and Edge Cases). Score per-event and aggregate over windows; condition on account age/segment so "new user" isn't auto-anomalous.

## 3. Weakly/semi-supervised middle ground
A few labels + many unlabeled: PU learning ([Class Imbalance, Calibration, and PU Learning](/notes/ml-system-design/foundations/class-imbalance-calibration-and-pu-learning/)), label propagation over a similarity graph, anomaly scores as *features* for a small supervised head.

## 4. Practical truths to say in interviews
1. **Anomalous ≠ abusive.** Power users, speedrunners, QA testers are anomalies. Anomaly scores should *route to investigation or feed a supervised layer*, rarely auto-enforce.
2. **Conditioning matters:** score relative to the right peer group (cohort by age/region/platform) or everything new is "anomalous."
3. **Threshold = review capacity:** set thresholds to fill the human-review budget, not at a magic score.
4. Anomaly detection is the **novel-attack safety net** around the supervised system — frame it as one layer of [defense in depth](/notes/ml-system-design/fraud-and-abuse/anatomy-of-a-fraud-detection-system/).
