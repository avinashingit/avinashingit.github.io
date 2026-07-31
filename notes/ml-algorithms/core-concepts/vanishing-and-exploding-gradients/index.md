---
layout: note
title: "Vanishing and Exploding Gradients"
description: "The gradient at layer $\\ell$ is a product of terms across all later layers:"
note: true
note_collection: "ML algorithms"
note_section: "Core Concepts"
section_order: 1
note_order: 12
updated: 2026-06-06 11:38:30 -0700
keywords:
  - Optimization
  - Training
  - Deep Learning
  - Transformers
math: true
mermaid: false
---
> In deep networks, [backprop](/notes/ml-algorithms/core-concepts/back-propagation/) multiplies many Jacobians together. If their magnitudes are consistently $<1$ the gradient **vanishes**; if $>1$ it **explodes** — either way, early layers fail to train.

### The Mechanism

The gradient at layer $\ell$ is a product of terms across all later layers:

$$\frac{\partial L}{\partial \theta^{(\ell)}} \propto \prod_{k=\ell}^{L} W^{(k)} \cdot \text{diag}\big(\phi'(z^{(k)})\big)$$

Multiplying many factors:
- each $< 1$ → product shrinks exponentially → **vanishing** (early layers get ~zero gradient, stop learning).
- each $> 1$ → product grows exponentially → **exploding** (NaNs, divergence).

This is most acute in **deep nets** and **RNNs** over long sequences (see [Recurrent Neural Networks](/notes/ml-algorithms/deep-learning/recurrent-neural-networks/)).

### Symptoms

| | Vanishing | Exploding |
|---|---|---|
| Loss | Plateaus early, early layers frozen | Spikes, NaN/Inf |
| Gradient norms | → 0 in early layers | → very large |
| Common with | Sigmoid/tanh, deep stacks, long RNNs | Poor init, high LR, RNNs |

### Fixes

**For vanishing:**
- **ReLU-family activations** (non-saturating) instead of sigmoid/tanh — see [Activation Functions and Optimizers](/notes/ml-algorithms/core-concepts/activation-functions-and-optimizers/).
- **Residual / skip connections** (`Residual Connections`) — give gradients a highway (ResNets, [Transformers](/notes/ml-algorithms/deep-learning/transformers/)).
- **[Batch Normalization](/notes/ml-algorithms/core-concepts/batch-normalization/)** / LayerNorm.
- **Careful weight init** (`Weight Initialization` — Xavier/Glorot, He).
- **LSTM/GRU** gating for sequences ([Recurrent Neural Networks](/notes/ml-algorithms/deep-learning/recurrent-neural-networks/)).

**For exploding:**
- **Gradient clipping** (clip by norm/value) — the standard fix for RNNs.
- Lower learning rate; proper init; normalization layers.

### Why it shaped modern architectures

The vanishing-gradient problem is *the* reason plain deep MLPs/RNNs were historically hard to train — and why ResNets, LSTMs, normalization, and attention/[Transformers](/notes/ml-algorithms/deep-learning/transformers/) (which shorten gradient paths) exist.

### Related
- [Backpropagation](/notes/ml-algorithms/core-concepts/back-propagation/) — where the chained products arise
- [Recurrent Neural Networks](/notes/ml-algorithms/deep-learning/recurrent-neural-networks/) — classic victim; BPTT over long sequences
- [Batch Normalization](/notes/ml-algorithms/core-concepts/batch-normalization/) · [Activation Functions and Optimizers](/notes/ml-algorithms/core-concepts/activation-functions-and-optimizers/) · [Gradient Descent](/notes/ml-algorithms/core-concepts/gradient-descent/)
