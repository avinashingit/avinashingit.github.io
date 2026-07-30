---
layout: note
title: "Sequence Recommenders — DIN, SASRec, BERT4Rec"
description: "A user is not a bag of features; a user is a sequence of actions. Order and recency carry intent: someone who viewed {crib, stroller, bottle} yesterday is in a different state t…"
note: true
note_collection: "ML system design"
note_section: "Recommendation Systems"
section_order: 2
note_order: 5
math: true
mermaid: false
---
A user is not a bag of features; a user is a **sequence of actions**. Order and recency carry intent: someone who viewed {crib, stroller, bottle} yesterday is in a different state than their 5-year average profile suggests. Three canonical ways to use the sequence:

## 1. DIN — Deep Interest Network (Alibaba): target-aware attention *inside the ranker*

Problem with average-pooling a user's history into one vector: a user's history contains many interests; for scoring *this candidate*, only some history matters. DIN computes **attention where the candidate item is the query and history items are keys/values**:

$$\text{user vector for candidate } c = \sum_{h \in \text{history}} \alpha(e_h, e_c)\, e_h$$

with $\alpha$ a small MLP over $(e_h, e_c, e_h \odot e_c, e_h - e_c)$. Scoring a pet game? The weights light up past pet-game interactions and ignore the shooter phase. DIN is a *ranking-stage module* (it needs the candidate — can't be used in retrieval, which has no candidate yet). DIEN adds a GRU to model interest *evolution*; mention, don't dwell.

## 2. SASRec — Self-Attentive Sequential Recommendation: GPT for items

Treat the interaction sequence like a sentence; train a **causal (unidirectional) [Transformer](/notes/ml-system-design/foundations/transformers-from-scratch/)** to predict the next item at every position:

```
input:  [g1, g2, g3, g4]      (games played, in order, + position embeddings)
target: [g2, g3, g4, g5]      (next-item at each step; causal mask)
loss:   sampled softmax over the catalog per position
```

At serving, the hidden state at the last position is the **user state vector** → dot product with item embeddings → can be served via ANN (it *is* a two-tower whose user tower is a causal Transformer). 2 layers, d=64–256 is typical — these are small models. Captures short-term intent and session dynamics that static profiles miss.

## 3. BERT4Rec: bidirectional with masked-item prediction

Same idea, but BERT-style: randomly mask items in the sequence, predict them from **both sides**. More training signal per sequence and richer representations; but beware: at inference you only ever append a [MASK] at the end (you can't see the future), so the bidirectional advantage is mostly a training-regularization effect. Note for fairness: subsequent literature found tuned SASRec ≈ BERT4Rec; say "the choice matters less than negatives/loss details."

## 4. Long-sequence engineering (the production question)

Histories run to 10⁴–10⁵ events; attention is O(T²) ([Transformers from Scratch](/notes/ml-system-design/foundations/transformers-from-scratch/)). Standard answers:
- **Truncate** to last 100–1000 events (most value lives there).
- **Two-stage attention / SIM (search-based interest model):** first *retrieve* from the lifelong history the ~100 events most related to the candidate (by category or embedding ANN — yes, ANN over the user's own history), then run full attention on those. This is Alibaba's production answer for lifelong sequences.
- **Hierarchical:** encode sessions → sequence of session embeddings (PinnerFormer-style lifelong user embeddings at Pinterest).
- **Pre-aggregate** very old history into counters/cluster mixtures.

## 5. Direct relevance to account matching

An account's **event stream is the input to the behavioral encoder** in AM-05 The Siamese Behavioral Encoder — the SASRec recipe (typed event tokens + time-delta encodings + small causal Transformer) is exactly the encoder you'll propose, swapping the next-item objective for a [contrastive same-owner objective](/notes/ml-system-design/foundations/contrastive-learning-infonce-triplet-siamese/) (and optionally keeping next-event prediction as an auxiliary self-supervised loss).

## Interview cheat-row

| Question | Answer |
|---|---|
| "How do you use behavior history in *ranking*?" | DIN-style target attention over recent events |
| "How do you model short-term intent for *retrieval*?" | SASRec-style causal Transformer as the user tower |
| "Histories are 50k events long" | retrieve-then-attend (SIM) or hierarchical session encoding |
| "Cold users?" | sequence models degrade gracefully — short sequences still encode; back off to context features ([Cold Start and Exploration](/notes/ml-system-design/recommendation-systems/cold-start-and-exploration/)) |
