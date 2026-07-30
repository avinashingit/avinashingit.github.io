---
layout: note
title: "Cold Start and Exploration"
description: "Every recsys interview ends with \"what about new users/items?\" Have a layered answer."
note: true
note_collection: "ML system design"
note_section: "Recommendation Systems"
section_order: 2
note_order: 1
updated: 2026-06-09 12:40:19 -0700
keywords:
  - Recommendation
  - Ranking
  - LLMs
  - Embeddings
  - Retrieval
math: true
mermaid: false
---
Every recsys interview ends with "what about new users/items?" Have a layered answer.

## New items (item cold start)

1. **Content-based representation:** the item tower uses text/image/audio encoders (title, description, thumbnail via [CLIP-style](/notes/ml-system-design/vision-and-waymo/clip-and-multimodal-models/) encoders, creator features) so a brand-new item embeds meaningfully with zero interactions. Train with **ID-embedding dropout** so the model doesn't lean entirely on the ID ([Two-Tower Retrieval Networks](/notes/ml-system-design/recommendation-systems/two-tower-retrieval-networks/)).
2. **Exploration traffic:** reserve a small slice of impressions for fresh items to *buy* interaction data. Allocate with bandits:
   - **Epsilon-greedy:** with prob ε show an explore item. Simple, wasteful.
   - **UCB:** score = predicted value + uncertainty bonus → explores what you know least.
   - **Thompson sampling:** maintain a posterior over each item's quality (e.g., Beta over CTR); sample from posteriors and rank by samples — exploration emerges from uncertainty automatically; converges to exploitation as posteriors sharpen. The interview-favorite.
3. **Graduation:** once an item has enough impressions, its learned ID embedding takes over.

## New users (user cold start)

1. Context features available from second zero: device, locale, time, acquisition channel, age band.
2. **Onboarding signals:** picked interests; first session's clicks are gold — use a [sequence encoder](/notes/ml-system-design/recommendation-systems/sequence-recommenders-din-sasrec-bert4rec/) that produces useful states from 3 events.
3. Popularity-by-segment priors; shrink personalization toward segment means until data accumulates (hierarchical/empirical-Bayes mindset).
4. **Meta-learning (mention only):** MAML-style "learn an initialization that adapts fast per user" — rarely productionized; fine to name and move past.

## The exploration–exploitation framing (say this)

Cold start is not a bug to patch; it's the **exploration side of a bandit**: short-term engagement loss purchased for information. The design questions are *budget* (what % of impressions), *placement* (dedicated slots so exploration doesn't pollute top ranks), and *measurement* (log propensities → enables [off-policy evaluation](/notes/ml-system-design/recommendation-systems/position-bias-and-counterfactual-learning/)).

## Cold start in account matching

A brand-new account has no behavior for the behavioral encoder — yet registration time is exactly when ban-evasion matching matters most. Answer: at t=0 rely on hard signals (device/IP/payment/email + registration telemetry), produce a *provisional* embedding from the first session, and re-score as events accumulate (sequential evidence accumulation — see AM-09 Enforcement, Serving, and Infra). The system design must explicitly include this **signal-maturity timeline**.
