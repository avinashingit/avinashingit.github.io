---
layout: note
title: "Gradient Descent"
description: "Plain SGD struggles with ravines and saddle points. Improvements:"
note: true
note_collection: "ML algorithms"
note_section: "Core Concepts"
section_order: 1
note_order: 7
updated: 2026-06-06 11:38:01 -0700
keywords:
  - Optimization
  - Training
  - Deep Learning
  - Linear Models
  - Supervised Learning
math: true
mermaid: false
---
> The optimization workhorse of ML. Iteratively step **downhill** along the negative gradient of the loss to find parameters that minimize it.

### The Update Rule

$$\theta \leftarrow \theta - \eta\, \nabla_\theta L(\theta)$$

- $\nabla_\theta L$ — gradient (direction of steepest *ascent*), computed via [Backpropagation](/notes/ml-algorithms/core-concepts/back-propagation/).
- $\eta$ — **learning rate**, the step size.

### Batch vs Stochastic vs Mini-batch

| Variant | Gradient computed on | Trade-off |
|---|---|---|
| **Batch GD** | Entire dataset | Stable but slow; full pass per step |
| **Stochastic GD (SGD)** | One sample | Noisy, fast, can escape shallow minima |
| **Mini-batch** | A batch (32–512) | **The default** — vectorized + stable |

### The Learning Rate

- **Too high** → overshoot, diverge, loss explodes.
- **Too low** → painfully slow, may stall in plateaus.
- **Schedules** (see [Activation Functions and Optimizers](/notes/ml-algorithms/core-concepts/activation-functions-and-optimizers/)): step decay, cosine annealing, warmup. `Learning Rate Schedules`.

### Momentum & Adaptive Methods

Plain SGD struggles with ravines and saddle points. Improvements:

- **Momentum:** accumulate a velocity $v \leftarrow \beta v + \nabla L$, step with $v$ — dampens oscillation, accelerates along consistent directions.
- **Nesterov:** look-ahead momentum.
- **AdaGrad / RMSProp:** per-parameter adaptive learning rates.
- **Adam / AdamW:** momentum + RMSProp + bias correction — the de-facto default for deep nets. (Details in [Activation Functions and Optimizers](/notes/ml-algorithms/core-concepts/activation-functions-and-optimizers/).)

### The Loss Landscape

- **Convex** (e.g. [Linear Regression](/notes/ml-algorithms/supervised-learning/linear-regression/), [Logistic Regression](/notes/ml-algorithms/supervised-learning/logistic-regression/)): one global minimum — GD provably converges.
- **Non-convex** (deep nets): many local minima/saddle points, but in high dimensions most are near-equivalent; saddles, not bad local minima, are the real obstacle.

### Common Pitfalls

1. Forgetting to **shuffle** data between epochs (biases mini-batches).
2. Not **scaling features** — elongated contours make GD zig-zag.
3. [Vanishing and Exploding Gradients](/notes/ml-algorithms/core-concepts/vanishing-and-exploding-gradients/) in deep nets stall or destabilize updates.

### Related
- [Backpropagation](/notes/ml-algorithms/core-concepts/back-propagation/) — computes the gradients GD consumes
- [Activation Functions and Optimizers](/notes/ml-algorithms/core-concepts/activation-functions-and-optimizers/) — Adam, RMSProp, schedules
- [Vanishing and Exploding Gradients](/notes/ml-algorithms/core-concepts/vanishing-and-exploding-gradients/) · [Linear Regression](/notes/ml-algorithms/supervised-learning/linear-regression/) · [Logistic Regression](/notes/ml-algorithms/supervised-learning/logistic-regression/)
