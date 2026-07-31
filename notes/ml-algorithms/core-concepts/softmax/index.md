---
layout: note
title: "Softmax"
description: "For logits $\\mathbf{z} = (z1, \\dots, zK)$:"
note: true
note_collection: "ML algorithms"
note_section: "Core Concepts"
section_order: 1
note_order: 11
updated: 2026-06-06 11:37:32 -0700
keywords:
  - Training
  - Optimization
  - Linear Models
  - Probability
  - Supervised Learning
math: true
mermaid: false
---
> Turns a vector of raw scores (**logits**) into a **probability distribution** — all outputs in $(0,1)$ and summing to 1. The standard output activation for multi-class classification.

### Definition

For logits $\mathbf{z} = (z_1, \dots, z_K)$:

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

Each output is positive and they sum to 1 — a valid distribution over $K$ classes.

### Numerical Stability — always subtract the max

$e^{z_i}$ overflows for large logits. Softmax is shift-invariant, so subtract $\max_j z_j$ first:

$$\text{softmax}(z_i) = \frac{e^{z_i - \max_j z_j}}{\sum_k e^{z_k - \max_j z_j}}$$

This changes nothing mathematically but keeps exponents $\le 0$.

### Key Properties

- **Shift-invariant:** $\text{softmax}(\mathbf{z} + c) = \text{softmax}(\mathbf{z})$ — only *differences* between logits matter.
- **Not scale-invariant:** multiplying logits by a **temperature** $T$ sharpens ($T<1$) or softens ($T>1$) the distribution: $\text{softmax}(z_i / T)$.
- **Generalizes the [sigmoid](/notes/ml-algorithms/supervised-learning/logistic-regression/):** for $K=2$, softmax reduces to the sigmoid.

### Why it pairs with Cross-Entropy

When softmax is followed by [Cross-Entropy Loss](/notes/ml-algorithms/core-concepts/cross-entropy-loss/), the gradient w.r.t. the logits collapses to the beautifully simple:

$$\frac{\partial L}{\partial z_i} = \hat{y}_i - y_i \quad (\text{predicted} - \text{true})$$

This clean gradient is why frameworks fuse them (e.g. PyTorch's `CrossEntropyLoss` takes **raw logits**, not softmax outputs — applying softmax twice is a common bug).

### Softmax vs Sigmoid

| | Softmax | Sigmoid |
|---|---|---|
| Output | Distribution over $K$ classes (sums to 1) | Independent prob per output |
| Use | **Multi-class** (one label) | **Binary** or **multi-label** |

### Related
- [Cross-Entropy Loss](/notes/ml-algorithms/core-concepts/cross-entropy-loss/) — its natural loss partner
- [Logistic Regression](/notes/ml-algorithms/supervised-learning/logistic-regression/) — softmax = multi-class generalization of sigmoid
- [Attention](/notes/ml-algorithms/deep-learning/attention/) — softmax normalizes attention weights
- [Neural Networks](/notes/ml-algorithms/deep-learning/neural-networks/) · [Activation Functions and Optimizers](/notes/ml-algorithms/core-concepts/activation-functions-and-optimizers/)
