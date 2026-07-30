---
layout: note
title: "Embeddings from First Principles"
description: "Neural networks consume vectors of real numbers. But most industrial data is categorical: user IDs, item IDs, words, device models, country codes. How do you feed \"user84629173\"…"
note: true
note_collection: "ML system design"
note_section: "Foundations"
section_order: 1
note_order: 4
updated: 2026-06-09 12:29:07 -0700
keywords:
  - Embeddings
  - Training
  - Retrieval
  - Recommendation
  - Transformers
math: true
mermaid: false
---
## The problem embeddings solve

Neural networks consume vectors of real numbers. But most industrial data is **categorical**: user IDs, item IDs, words, device models, country codes. How do you feed "user_84629173" into a matrix multiplication?

**Naive answer: one-hot encoding.** Represent each of $N$ categories as an $N$-dimensional vector that is all zeros except a 1 at that category's index.

Problems:
1. **Size.** Roblox has ~10⁸ users → each user is a 100-million-dim vector. Untenable.
2. **No notion of similarity.** Every pair of one-hot vectors is equally far apart (dot product = 0). "Minecraft-style game A" and "Minecraft-style game B" are exactly as dissimilar as "Minecraft-style game A" and "fashion dress-up game". The model can't generalize across categories.

**The embedding answer:** map each category to a learned dense vector of dimension $d$ (typically 8–512), stored in an **embedding table** — a matrix $E \in \mathbb{R}^{N \times d}$. "Looking up" category $i$ means taking row $E_i$. Mathematically this is identical to multiplying the one-hot vector by $E$, but implemented as a table lookup (O(d), not O(Nd)).

The rows of $E$ are **parameters, learned by backprop** like any weight. During training, only the rows for categories present in the batch receive gradient — embedding training is naturally sparse.

## Why learned embeddings encode meaning

Embeddings get their meaning from the **loss function pulling co-occurring things together**. The classic intuition (word2vec): if "obby" and "parkour" appear in similar contexts, the model can only predict those contexts well if it gives both words similar vectors. Similarity in the embedding space comes to mean *behavioral substitutability*.

Concrete toy example. Suppose $d=2$ and after training a game-recommendation model we find:

| item | embedding |
|---|---|
| Adopt Me (pet game) | (0.9, 0.1) |
| Pet Simulator | (0.85, 0.2) |
| Phantom Forces (shooter) | (-0.7, 0.6) |
| Arsenal (shooter) | (-0.75, 0.5) |

Dot products: AdoptMe·PetSim = 0.785 (high), AdoptMe·Arsenal = -0.625 (low). Dimension 1 has *learned to mean* something like "casual-pet vs shooter" without anyone programming it. Real embeddings do this in 64–256 dimensions with thousands of soft, overlapping concepts.

## Measuring similarity

- **Dot product** $a \cdot b = \sum_i a_i b_i$: scale-sensitive — longer vectors score higher. In recsys, the norm of an item embedding often comes to encode *popularity* (popular items grow long so they score high against everyone). Sometimes desirable, sometimes a bias to control.
- **Cosine similarity** $\frac{a \cdot b}{\|a\|\|b\|}$: dot product after normalizing both to length 1; pure "angle" similarity, range [-1, 1]. Use when you want popularity factored out (e.g., matching two accounts by behavior — see AM-05 The Siamese Behavioral Encoder).
- **Euclidean distance** $\|a-b\|$: for unit vectors, monotonically related to cosine ($\|a-b\|^2 = 2 - 2\cos\theta$), so the choice between them mostly matters for unnormalized vectors.

## Practical engineering of embedding tables (interviewers probe this)

**The hashing trick.** You can't keep a row for 10⁸ users plus every new signup. Solution: hash the ID into a table of fixed size $M \ll N$: row = $\text{hash}(id) \bmod M$. Collisions force unrelated entities to share a vector — acceptable noise for tail entities. Refinement: **QR / multi-hash embeddings** — combine two smaller tables via two different hashes, e.g. $E_1[h_1(id)] + E_2[h_2(id)]$; two entities collide only if *both* hashes collide, which is rare, while memory is $O(2\sqrt{M})$-ish instead of $O(M)$.

**Frequency-aware sizing.** Give popular entities full embeddings and tail entities hashed/shared ones; or use mixed dimensions (big $d$ for head, small $d$ for tail).

**Sharding.** Industrial CTR models ([DLRM](/notes/ml-system-design/recommendation-systems/feature-interaction-models-wide-and-deep-deepfm-dcn-dlrm/)) have embedding tables totaling **terabytes** — they don't fit on one GPU. Tables are sharded across many GPUs/hosts (model parallelism for embeddings) while the dense MLP layers are replicated (data parallelism). The lookup results are exchanged via all-to-all communication. See Distributed Training.

**Serving.** Precomputed embeddings live in a key-value store or are baked into an [ANN index](/notes/ml-system-design/foundations/approximate-nearest-neighbor-search/). The pattern *"turn matching into embedding lookup + ANN search"* is the single most reused trick in ML system design.

## Where embeddings appear in this vault

- Retrieval: [Two-Tower Retrieval Networks](/notes/ml-system-design/recommendation-systems/two-tower-retrieval-networks/) — user and item embeddings, dot-product scoring.
- Sequence understanding: [Transformers from Scratch](/notes/ml-system-design/foundations/transformers-from-scratch/) — token embeddings + position embeddings.
- Graphs: [GCN, GraphSAGE, GAT](/notes/ml-system-design/graph-neural-networks/gcn-graphsage-gat/) — node embeddings computed from neighbors.
- Account matching: AM-05 The Siamese Behavioral Encoder — an embedding per account such that same-owner accounts are close.

Next: how do you *train* embeddings when there are 100M classes to predict? → [Softmax, Sampled Softmax, and the logQ Correction](/notes/ml-system-design/foundations/softmax-sampled-softmax-and-the-logq-correction/)
