---
layout: note
title: "Bi-Encoder vs Cross-Encoder"
description: "The single most reusable architectural dichotomy in this vault. It is the same tradeoff as retrieval-vs-ranking in recsys, and Layer-1-vs-Layer-2 in account matching."
note: true
note_collection: "ML system design"
note_section: "Search and LLMs"
section_order: 3
note_order: 2
updated: 2026-06-09 12:41:40 -0700
keywords:
  - Retrieval
  - LLMs
  - Embeddings
  - Ranking
  - Recommendation
math: true
mermaid: false
---
The single most reusable architectural dichotomy in this vault. It is the same tradeoff as retrieval-vs-ranking in recsys, and Layer-1-vs-Layer-2 in account matching.

## Bi-encoder (dual encoder)
Encode query and document **separately**; score = dot/cosine of the two vectors.
- Documents pre-encodable offline → [ANN](/notes/ml-system-design/foundations/approximate-nearest-neighbor-search/) over millions/billions. O(1) encoder calls per query.
- Limitation: all interaction is compressed through one fixed-size vector *before* seeing the other side — fine-grained term-level interactions are lost.

## Cross-encoder
Concatenate the **pair** into one [Transformer](/notes/ml-system-design/foundations/transformers-from-scratch/): `[CLS] query [SEP] document` → every query token attends to every document token in every layer → [CLS] → score.
- Far more accurate: full token-level interaction.
- Cannot be indexed — the score exists only per-pair → O(candidates) full Transformer passes per query → usable only on ~10–500 candidates.

## The universal pattern

**Cheap separable model to retrieve, expensive joint model to decide.**

| Domain | Bi-encoder stage | Cross-encoder stage |
|---|---|---|
| Web search | dense retrieval | re-ranker |
| RecSys | two-tower retrieval | DCN/DLRM ranker (feature-cross models are "cross-encoders over features") |
| QA / RAG | passage retriever | answer re-ranker |
| **Account matching** | Siamese account encoder + ANN | pairwise matcher with cross features |
| Face/identity | embedding gallery search | pairwise verification model |

## Bridging the gap
- **Distillation:** train the bi-encoder to mimic cross-encoder scores (Model Compression - Distillation and Quantization) — standard, large quality gains.
- **ColBERT (late interaction):** keep per-token embeddings; score = sum of per-query-token max similarities ("MaxSim"). Near-cross-encoder quality, still indexable (at higher storage). Good name-drop.
