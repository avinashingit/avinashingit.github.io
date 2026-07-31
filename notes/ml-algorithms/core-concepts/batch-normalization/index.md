---
layout: note
title: "Batch Normalization"
description: "For a mini-batch $\\mathcal{B} = \\{x1, \\dots, xm\\}$ at some layer:"
note: true
note_collection: "ML algorithms"
note_section: "Core Concepts"
section_order: 1
note_order: 3
updated: 2026-06-06 11:38:16 -0700
keywords:
  - Optimization
  - Training
  - Inference
  - Deep Learning
  - Transformers
math: true
mermaid: false
---
> Normalizes each layer's pre-activations **per mini-batch** to zero mean and unit variance, then rescales with learned parameters. Stabilizes and accelerates training of deep networks.

### The Algorithm

For a mini-batch $\mathcal{B} = \{x_1, \dots, x_m\}$ at some layer:

$$\mu_\mathcal{B} = \frac{1}{m}\sum_i x_i, \qquad \sigma_\mathcal{B}^2 = \frac{1}{m}\sum_i (x_i - \mu_\mathcal{B})^2$$

$$\hat{x}_i = \frac{x_i - \mu_\mathcal{B}}{\sqrt{\sigma_\mathcal{B}^2 + \epsilon}}, \qquad y_i = \gamma\, \hat{x}_i + \beta$$

- $\gamma, \beta$ are **learned** scale/shift parameters — so the network can undo normalization if it helps (even recover the identity).
- $\epsilon$ is a small constant for numerical stability.

### Why it helps

- **Smoother optimization landscape** — the modern explanation (Santurkar et al.); allows higher learning rates.
- Reduces sensitivity to **weight initialization**.
- Mild **regularization** effect (batch statistics add noise), often reducing the need for dropout.
- Historically motivated as reducing "internal covariate shift" (now considered an incomplete explanation).

### Train vs Inference (critical detail)

- **Training:** normalize using the **current batch's** mean/variance.
- **Inference:** use **running (moving-average) statistics** accumulated during training — predictions must not depend on other samples in the batch.

> This is exactly why you call `model.eval()` in PyTorch — it switches BN to running stats (and disables dropout).

### Gotchas

1. **Small batch sizes** → noisy statistics → unstable. Use **GroupNorm / LayerNorm** instead.
2. Placement (before vs after activation) is debated; original paper put it before.
3. Interacts with weight decay on $\gamma, \beta$.

### Normalization Variants

| Variant | Normalizes over | Typical use |
|---|---|---|
| **BatchNorm** | Batch dim (per feature) | CNNs ([Convolutional Neural Networks](/notes/ml-algorithms/deep-learning/convolutional-neural-networks/)) |
| **LayerNorm** | Feature dim (per sample) | [Transformers](/notes/ml-algorithms/deep-learning/transformers/), RNNs |
| **GroupNorm** | Groups of channels | Small-batch vision |
| **InstanceNorm** | Per sample, per channel | Style transfer |

### Related
- [Vanishing and Exploding Gradients](/notes/ml-algorithms/core-concepts/vanishing-and-exploding-gradients/) — BN mitigates these
- [Neural Networks](/notes/ml-algorithms/deep-learning/neural-networks/) · [Convolutional Neural Networks](/notes/ml-algorithms/deep-learning/convolutional-neural-networks/) · [Transformers](/notes/ml-algorithms/deep-learning/transformers/) (LayerNorm)
- [Activation Functions and Optimizers](/notes/ml-algorithms/core-concepts/activation-functions-and-optimizers/) · [Backpropagation](/notes/ml-algorithms/core-concepts/back-propagation/)
