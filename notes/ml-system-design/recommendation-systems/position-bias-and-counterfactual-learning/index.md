---
layout: note
title: "Position Bias and Counterfactual Learning"
description: "Training data for rankers comes from logged impressions of the previous ranker. Click probability factors (to first order) as:"
note: true
note_collection: "ML system design"
note_section: "Recommendation Systems"
section_order: 2
note_order: 4
math: true
mermaid: false
---
## The problem

Training data for rankers comes from **logged impressions of the previous ranker**. Click probability factors (to first order) as:

$$P(\text{click} \mid \text{item}, \text{pos}) = P(\text{examine} \mid \text{pos}) \cdot P(\text{click} \mid \text{item}, \text{examined})$$

Rank-1 items get clicked partly *because they're at rank 1* (examination probability ~0.7) while rank-10 items barely get seen (~0.1). Naive training on (features → click) teaches the model "whatever the old ranker put on top is good" — a **self-reinforcing feedback loop** that entrenches yesterday's ranking and suppresses everything else. Related selection bias: you only observe outcomes for items the old system *chose to show at all*.

## Fixes you should be able to explain

**1. Position-as-feature (shallow tower).** Train with position as an input through a separate small tower whose logit is *added* to the relevance logit; serve with the position input removed/neutralized. The relevance tower learns position-free quality. (YouTube 2019 — see [YouTube RecSys Case Study](/notes/ml-system-design/recommendation-systems/youtube-recsys-case-study/).) Simple, the default answer.

**2. Inverse Propensity Scoring (IPS).** Estimate examination propensities $p_{pos}$ (via randomization or models), weight each clicked example by $1/p_{pos}$ → unbiased estimate of position-free relevance, at the cost of variance (clip weights). "Unbiased learning-to-rank." How to get propensities without hurting users: **result randomization on a small traffic slice** (swap adjacent ranks — "interleaving-style perturbation") or intervention-free estimation from logs of multiple historical rankers.

**3. Click models.** Explicitly model examination (cascade model: user scans top-down, stops after a click). Mostly subsumed by 1–2 in modern practice; know the vocabulary.

**4. Skip-above heuristic for negatives.** If the user clicked rank 5, treat ranks 1–4 as *seen-and-skipped* (true negatives); don't use rank 50 as a negative at all. Connects to [Negative Sampling and Hard Negatives](/notes/ml-system-design/foundations/negative-sampling-and-hard-negatives/).

## The bigger umbrella: counterfactual / off-policy learning

The general question: *"What would engagement have been if we had shown something else?"* Logged bandit feedback only shows outcomes for chosen actions. Tools: IPS and doubly-robust estimators for **off-policy evaluation** (estimate a new policy's value from old logs before any A/B), exploration policies that log propensities by design (epsilon-greedy slots), and [bandits](/notes/ml-system-design/recommendation-systems/cold-start-and-exploration/). Dropping "I'd log propensities so we can do off-policy evaluation later" is a senior-level line.

## Where this bites in abuse detection

Enforcement creates the same loop: you only get confirmed-evasion labels where you *looked*. Models trained on review outcomes inherit the review-routing policy's biases → keep a random-audit slice (AM-10 Adversarial Dynamics and Edge Cases, [PU learning](/notes/ml-system-design/foundations/class-imbalance-calibration-and-pu-learning/)).
