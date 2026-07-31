---
layout: note
title: "Attention"
description: "Recall the seq2seq encoder-decoder for translation:"
note: true
note_collection: "ML algorithms"
note_section: "Deep Learning"
section_order: 4
note_order: 1
updated: 2026-06-06 11:30:50 -0700
keywords:
  - Deep Learning
  - Training
  - Transformers
  - Evaluation
  - Graphs
math: true
mermaid: false
---
> Attention lets a model **focus on relevant parts of the input** when producing each output, instead of compressing everything into a single fixed-size vector. It's the foundation of Transformers and modern NLP.

---

## Table of Contents

- [1. The Problem Attention Solves](#1-the-problem-attention-solves)
- [2. The Core Idea — Soft Lookup](#2-the-core-idea-soft-lookup)
- [3. The Three Roles — Query, Key, Value](#3-the-three-roles-query-key-value)
- [4. The Math — Scaled Dot-Product Attention](#4-the-math-scaled-dot-product-attention)
- [5. A Worked Example](#5-a-worked-example)
- [6. Types of Attention](#6-types-of-attention)
- [7. Self-Attention](#7-self-attention)
- [8. Multi-Head Attention](#8-multi-head-attention)
- [9. Masked Attention](#9-masked-attention)
- [10. Complexity](#10-complexity)
- [11. Strengths and Weaknesses](#11-strengths-and-weaknesses)
- [12. Common Pitfalls](#12-common-pitfalls)

---

## 1. The Problem Attention Solves

Recall the seq2seq encoder-decoder for translation:

```
Encoder: reads source sentence → final hidden state c (context vector)
Decoder: starts from c → generates target sentence one word at a time
```

**Bottleneck problem:** The entire source sentence — possibly 50 words — must be squeezed into one fixed-size vector $\mathbf{c}$. For long inputs, information is inevitably lost.

> **Example:** Translation pain point Translating a 100-word English paragraph to French via a single 512-dim vector? The early words get diluted by the time the decoder needs them.

**Attention's idea:** Instead of using one fixed $\mathbf{c}$ for the whole output, let the decoder **look back at all encoder hidden states**, and dynamically choose which ones are relevant at each output step.

---

## 2. The Core Idea — Soft Lookup

Attention is essentially a **differentiable, soft dictionary lookup**.

Imagine a Python dict:

```python
d = {"red": "color", "dog": "animal", "three": "number"}
value = d["dog"]   # hard lookup: returns "animal"
```

A hard lookup is **not differentiable** — you can't backprop through "exact match." Attention is the **soft version**:

```
Given a query, compare it to all keys.
Compute a similarity score for each key.
Take a weighted average of the corresponding values.
```

The weights are determined by similarity — high-similarity keys contribute more. This is **fully differentiable**.

---

## 3. The Three Roles — Query, Key, Value

Attention introduces three vectors per token:

|Vector|Role|Analogy|
|---|---|---|
|**Query** ($\mathbf{q}$)|What we're looking for|Search query|
|**Key** ($\mathbf{k}$)|What each item advertises|Search index entry|
|**Value** ($\mathbf{v}$)|The actual content to retrieve|Search result|

For each query, we:

1. Compare it to every key → similarity scores
2. Normalize scores to weights (sum to 1) → softmax
3. Take weighted sum of values → output

---

## 4. The Math — Scaled Dot-Product Attention

Given:

- Query matrix $\mathbf{Q} \in \mathbb{R}^{n_q \times d_k}$ (each row is a query)
- Key matrix $\mathbf{K} \in \mathbb{R}^{n_k \times d_k}$
- Value matrix $\mathbf{V} \in \mathbb{R}^{n_k \times d_v}$

Attention is computed as:

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right) \mathbf{V}$$

### Breaking it down

**Step 1: Compute raw scores via dot product.**

$$\mathbf{S} = \mathbf{Q}\mathbf{K}^\top \quad \in \mathbb{R}^{n_q \times n_k}$$

Element $S_{ij}$ = similarity between query $i$ and key $j$. Dot product is high when vectors point similar directions.

**Step 2: Scale by $\sqrt{d_k}$.**

For large $d_k$, dot products grow large in magnitude, pushing softmax into saturated regions (where gradients vanish). Scaling by $\sqrt{d_k}$ keeps variance ~1, regardless of dimension.

> **Tip:** Why $\sqrt{d_k}$ specifically? If $\mathbf{q}, \mathbf{k}$ have entries with mean 0 and variance 1, then $\mathbf{q} \cdot \mathbf{k}$ has variance $d_k$. Dividing by $\sqrt{d_k}$ restores variance 1.

**Step 3: Apply softmax (row-wise).**

$$\mathbf{A} = \text{softmax}(\mathbf{S}/\sqrt{d_k})$$

Each row sums to 1. Row $i$ is a probability distribution over keys — how much attention query $i$ pays to each key.

**Step 4: Weighted sum of values.**

$$\mathbf{O} = \mathbf{A}\mathbf{V} \quad \in \mathbb{R}^{n_q \times d_v}$$

Row $i$ of $\mathbf{O}$ is the attention output for query $i$ — a weighted combination of value vectors, weighted by attention.

---

## 5. A Worked Example

Suppose we have 3 keys/values (encoder positions) and 1 query (decoder position), with $d_k = 4$:

$$\mathbf{q} = [1, 0, 1, 0]$$

$$\mathbf{K} = \begin{bmatrix} 1 & 0 & 1 & 0 \ 0 & 1 & 0 & 1 \ 1 & 1 & 0 & 0 \end{bmatrix}, \quad \mathbf{V} = \begin{bmatrix} 10 \ 20 \ 30 \end{bmatrix}$$

**Scores** ($\mathbf{q} \cdot \mathbf{k}_i$):

- $\mathbf{q} \cdot \mathbf{k}_1 = 1 \cdot 1 + 0 + 1 \cdot 1 + 0 = 2$
- $\mathbf{q} \cdot \mathbf{k}_2 = 0 + 0 + 0 + 0 = 0$
- $\mathbf{q} \cdot \mathbf{k}_3 = 1 + 0 + 0 + 0 = 1$

**Scaled** by $\sqrt{4} = 2$: $[1, 0, 0.5]$.

**Softmax:** $[e^1, e^0, e^{0.5}] / (e^1 + e^0 + e^{0.5}) \approx [0.506, 0.186, 0.307]$.

**Output:** $0.506 \cdot 10 + 0.186 \cdot 20 + 0.307 \cdot 30 \approx 18$.

So the query "looked mostly at key 1, a bit at key 3, and least at key 2," producing a value blended accordingly.

---

## 6. Types of Attention

### Bahdanau (Additive) Attention — 2014

The original, for RNN-based seq2seq. Uses a small feedforward network to score:

$$\text{score}(\mathbf{q}, \mathbf{k}) = \mathbf{v}^\top \tanh(\mathbf{W}_q \mathbf{q} + \mathbf{W}_k \mathbf{k})$$

Slower (per-pair feedforward) but doesn't need scaling.

### Luong (Multiplicative) Attention — 2015

$$\text{score}(\mathbf{q}, \mathbf{k}) = \mathbf{q}^\top \mathbf{W} \mathbf{k}$$

Faster (matmul). Common variants: dot, general, concat.

### Scaled Dot-Product Attention — 2017

The Transformer's version:

$$\text{score}(\mathbf{q}, \mathbf{k}) = \frac{\mathbf{q}^\top \mathbf{k}}{\sqrt{d_k}}$$

Fast, parallelizable, well-behaved gradients.

---

## 7. Self-Attention

In **self-attention**, queries, keys, and values all come from the **same sequence**.

For a sequence $\mathbf{X} \in \mathbb{R}^{T \times d}$ (T tokens, each $d$-dimensional):

$$\mathbf{Q} = \mathbf{X}\mathbf{W}_Q, \quad \mathbf{K} = \mathbf{X}\mathbf{W}_K, \quad \mathbf{V} = \mathbf{X}\mathbf{W}_V$$

where $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V$ are learned projection matrices.

Then:

$$\mathbf{O} = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}$$

Each output token is a **weighted combination of all input tokens** — the model decides for each position which other positions are most relevant.

> **Important:** Why self-attention is powerful
> 
> - **Every token can directly attend to every other token in one step** (no sequential bottleneck like RNNs).
> - Computation is **fully parallelizable** across positions.
> - Captures **long-range dependencies** without gradient pathologies.

### Cross-Attention vs Self-Attention

|Type|$\mathbf{Q}$ from|$\mathbf{K}, \mathbf{V}$ from|
|---|---|---|
|**Self-attention**|Same sequence|Same sequence|
|**Cross-attention**|One sequence (e.g., decoder)|Another sequence (e.g., encoder)|

Cross-attention is what lets the decoder look at the encoder in seq2seq.

---

## 8. Multi-Head Attention

Instead of one attention computation, run $h$ in parallel — each with different learned projections — and concatenate.

For $h$ heads, each with dim $d_k = d/h$:

$$\text{head}_i = \text{Attention}(\mathbf{X}\mathbf{W}_Q^i, \mathbf{X}\mathbf{W}_K^i, \mathbf{X}\mathbf{W}_V^i)$$

$$\text{MultiHead}(\mathbf{X}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\mathbf{W}_O$$

### Why multi-head?

Each head can specialize in different relationships:

- Head 1 might attend to syntactic dependencies (subject ↔ verb).
- Head 2 might attend to semantic relations (pronoun ↔ antecedent).
- Head 3 might attend to nearby positions.

Single-head attention can only learn one "mode" of attention; multi-head lets the model attend in multiple ways simultaneously.

> **Tip:** Total params unchanged With $h$ heads of dim $d/h$ each, total compute and params are similar to one full-dim head — but the model is **strictly more expressive**.

---

## 9. Masked Attention

For **autoregressive** tasks (language modeling, generation), each position should only attend to **previous** positions, not future ones (you can't peek at words that haven't been generated yet).

**Mask the score matrix** by setting future positions to $-\infty$ before softmax:

$$\mathbf{S}_{ij} = \begin{cases} \mathbf{q}_i \cdot \mathbf{k}_j / \sqrt{d_k} & \text{if } j \leq i \ -\infty & \text{if } j > i \end{cases}$$

After softmax, $-\infty$ → 0, so future positions get zero attention.

This is called **causal masking** and is essential for GPT-style models.

---

## 10. Complexity

- **Self-attention:** $O(T^2 \cdot d)$ — every token compares to every other token.
- **RNN:** $O(T \cdot d^2)$ — sequential through time.

For short sequences, attention is faster (parallelizable). For very long sequences (T >> d), attention becomes expensive — hence interest in **linear attention**, **sparse attention**, **FlashAttention**, etc.

|Operation|Complexity|
|---|---|
|Self-attention|$O(T^2 d)$|
|RNN|$O(T d^2)$|
|1D Convolution|$O(k T d^2)$|
|Feedforward|$O(T d^2)$|

---

## 11. Strengths and Weaknesses

### Strengths

- Captures long-range dependencies in **one step**
- Fully **parallelizable** (unlike RNNs)
- **Interpretable** — attention weights show what the model focused on
- No vanishing gradient issues across positions
- Flexible — works for any modality (text, vision, audio)

### Weaknesses

- **$O(T^2)$ memory and compute** — expensive for very long sequences
- **No positional information by default** — needs positional encodings
- Attention weights are sometimes **misleading** as explanations (an active area of research)
- Can attend to spurious correlations if not regularized

---

## 12. Common Pitfalls

- **Forgetting positional encodings** — pure self-attention is permutation-equivariant; without positional info, "dog bites man" = "man bites dog."
- **Not scaling by $\sqrt{d_k}$** — softmax saturates, gradients vanish.
- **No mask in autoregressive training** — model cheats by peeking at future tokens.
- **Wrong mask direction** — easy to flip causal mask.
- **Treating attention as causal explanation** — high attention ≠ high causal influence on output.
- **Quadratic memory blowup** on long sequences — use FlashAttention or chunking.

---
