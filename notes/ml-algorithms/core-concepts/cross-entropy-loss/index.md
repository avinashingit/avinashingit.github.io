---
layout: note
title: "Cross-Entropy Loss"
description: "Cross-entropy $H(p, q) = -\\sumi pi \\log qi$ is the expected number of bits to encode events from the true distribution $p$ using a code optimized for the predicted $q$. Minimizi…"
note: true
note_collection: "ML algorithms"
note_section: "Core Concepts"
section_order: 1
note_order: 6
updated: 2026-06-06 11:37:46 -0700
keywords:
  - Training
  - Optimization
  - Probability
  - Linear Models
  - Supervised Learning
math: true
mermaid: false
---
> Measures the distance between the **true distribution** and the **predicted distribution**. The default loss for classification — it heavily penalizes confident wrong predictions.

### Information-Theoretic Intuition

Cross-entropy $H(p, q) = -\sum_i p_i \log q_i$ is the expected number of bits to encode events from the true distribution $p$ using a code optimized for the predicted $q$. Minimizing it pushes $q \to p$. It equals entropy plus the **KL divergence** $H(p,q) = H(p) + D_{KL}(p \parallel q)$, so minimizing CE = minimizing KL from truth.

### Binary Cross-Entropy (BCE)

For a true label $y \in \{0,1\}$ and predicted probability $\hat{y}$ (from a [sigmoid](/notes/ml-algorithms/supervised-learning/logistic-regression/)):

$$L = -\big[\,y \log \hat{y} + (1-y)\log(1-\hat{y})\,\big]$$

### Categorical Cross-Entropy

For one-hot $\mathbf{y}$ over $K$ classes and predicted $\hat{\mathbf{y}}$ (from [Softmax](/notes/ml-algorithms/core-concepts/softmax/)):

$$L = -\sum_{k=1}^{K} y_k \log \hat{y}_k = -\log \hat{y}_{\text{true class}}$$

Because $\mathbf{y}$ is one-hot, it reduces to the **negative log-probability of the correct class** (a.k.a. NLL).

### Why not MSE for classification?

- **Gradient quality:** MSE + sigmoid gives gradients that vanish when the model is confidently wrong (the sigmoid saturates). CE's gradient stays strong.
- **Clean gradient:** softmax + CE yields $\frac{\partial L}{\partial z_i} = \hat{y}_i - y_i$ — see [Softmax](/notes/ml-algorithms/core-concepts/softmax/).
- **Probabilistic grounding:** CE = maximum likelihood for a Bernoulli/categorical model.

### Practical Notes

- Use **logits**, not probabilities, in framework losses (`CrossEntropyLoss`, `BCEWithLogitsLogits`) for numerical stability (log-sum-exp trick).
- **Class imbalance:** weight the loss per class, or use **focal loss** to down-weight easy examples.
- **Label smoothing:** replace hard 0/1 targets with $\epsilon$-softened ones to reduce overconfidence.

### Related
- [Softmax](/notes/ml-algorithms/core-concepts/softmax/) — produces the probabilities CE consumes
- [Logistic Regression](/notes/ml-algorithms/supervised-learning/logistic-regression/) — BCE is its training objective
- [Metrics](/notes/ml-algorithms/core-concepts/metrics/) · [Neural Networks](/notes/ml-algorithms/deep-learning/neural-networks/) · [Backpropagation](/notes/ml-algorithms/core-concepts/back-propagation/)
