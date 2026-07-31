---
layout: note
title: "Support Vector Machines"
description: "A supervised algorithm for classification (and, via SVR, regression). Among all hyperplanes that separate two classes, the SVM picks the one that maximizes the margin — the dist…"
note: true
note_collection: "ML algorithms"
note_section: "Supervised Learning"
section_order: 2
note_order: 8
updated: 2026-06-06 11:35:54 -0700
keywords:
  - Supervised Learning
  - Linear Models
  - Evaluation
  - Optimization
  - Training
math: true
mermaid: false
---
> An SVM finds the **hyperplane that separates classes with the largest possible margin**. Only the closest points — the **support vectors** — determine the boundary; everything else is irrelevant.

### What is an SVM

A supervised algorithm for **classification** (and, via SVR, regression). Among all hyperplanes that separate two classes, the SVM picks the one that **maximizes the margin** — the distance to the nearest training point of either class. Maximizing the margin is a form of structural risk minimization, which tends to generalize well.

### The Margin and the Optimization Problem

A hyperplane is $\mathbf{w}^\top \mathbf{x} + b = 0$. With labels $y_i \in \{-1, +1\}$, the geometric margin is $\frac{2}{\lVert \mathbf{w} \rVert}$. The **hard-margin** problem is:

$$\min_{\mathbf{w}, b} \tfrac{1}{2}\lVert \mathbf{w}\rVert^2 \quad \text{s.t.}\quad y_i(\mathbf{w}^\top \mathbf{x}_i + b) \ge 1 \;\; \forall i$$

The points where the constraint is tight ($y_i(\mathbf{w}^\top\mathbf{x}_i+b)=1$) are the **support vectors**.

### Soft Margin — Handling Non-Separable Data

Real data isn't perfectly separable. Introduce slack variables $\xi_i \ge 0$ and a penalty $C$:

$$\min_{\mathbf{w}, b, \xi} \tfrac{1}{2}\lVert\mathbf{w}\rVert^2 + C\sum_i \xi_i \quad \text{s.t.}\quad y_i(\mathbf{w}^\top\mathbf{x}_i + b) \ge 1 - \xi_i$$

- **Large `C`** → penalize misclassification heavily → narrow margin, risk of overfitting (low bias, high variance).
- **Small `C`** → tolerate violations → wide margin, more regularized (high bias, low variance).

This is equivalent to minimizing **hinge loss** $\max(0, 1 - y_i f(\mathbf{x}_i))$ plus L2 regularization.

### The Kernel Trick

To separate non-linear data, map inputs into a higher-dimensional space $\phi(\mathbf{x})$. The dual formulation depends only on **dot products** $\phi(\mathbf{x}_i)^\top \phi(\mathbf{x}_j)$, which a **kernel** $K(\mathbf{x}_i, \mathbf{x}_j)$ computes without ever materializing $\phi$.

| Kernel | Formula | Use case |
|---|---|---|
| Linear | $\mathbf{x}_i^\top \mathbf{x}_j$ | High-dim sparse data (text) |
| Polynomial | $(\gamma\,\mathbf{x}_i^\top\mathbf{x}_j + r)^d$ | Interaction features |
| RBF (Gaussian) | $\exp(-\gamma\lVert \mathbf{x}_i - \mathbf{x}_j\rVert^2)$ | Default non-linear choice |
| Sigmoid | $\tanh(\gamma\,\mathbf{x}_i^\top\mathbf{x}_j + r)$ | Neural-net-like |

`gamma` controls RBF reach: high `gamma` = tight, wiggly boundaries (overfit); low `gamma` = smooth.

### Strengths

1. **Effective in high dimensions**, even when features > samples.
2. **Memory-efficient** — only support vectors are stored.
3. **Versatile** via kernels for non-linear boundaries.
4. **Strong theoretical guarantees** (max-margin generalization).

### Weaknesses

1. **Poor scaling** — training is roughly $O(n^2)$ to $O(n^3)$; impractical for very large datasets.
2. **No native probability output** (needs Platt scaling / calibration).
3. **Sensitive to feature scaling** — always standardize.
4. **Kernel & `C`/`gamma` tuning** is fiddly and crucial.
5. Less interpretable than linear/tree models.

### When to Use

Small-to-medium datasets, high-dimensional features (text classification), clear margin of separation. For large tabular data, prefer [Gradient Boosted Trees](/notes/ml-algorithms/supervised-learning/gradient-boosted-trees/) or [Random Forests](/notes/ml-algorithms/supervised-learning/random-forests/).

### Related
- [Logistic Regression](/notes/ml-algorithms/supervised-learning/logistic-regression/) — also a linear classifier; SVM maximizes margin, LR maximizes likelihood
- [Decision Trees](/notes/ml-algorithms/supervised-learning/decision-trees/) · [Random Forests](/notes/ml-algorithms/supervised-learning/random-forests/) · [Gradient Boosted Trees](/notes/ml-algorithms/supervised-learning/gradient-boosted-trees/)
- [kNN](/notes/ml-algorithms/supervised-learning/knn/) · [Metrics](/notes/ml-algorithms/core-concepts/metrics/) · [Bias-Variance Tradeoff](/notes/ml-algorithms/core-concepts/common-concepts/)
