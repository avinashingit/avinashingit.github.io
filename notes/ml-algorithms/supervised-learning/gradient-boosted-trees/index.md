---
layout: note
title: "Gradient Boosted Trees"
description: "An ensemble method that combines many weak learners (shallow decision trees) into a strong one. Unlike Random Forests which builds trees independently and in parallel (bagging),…"
note: true
note_collection: "ML algorithms"
note_section: "Supervised Learning"
section_order: 2
note_order: 2
updated: 2026-06-06 11:36:15 -0700
keywords:
  - Supervised Learning
  - Trees
  - Optimization
  - Training
  - Deep Learning
math: true
mermaid: false
---
> Build trees **sequentially**, where each new tree corrects the errors of the ensemble so far by fitting the **negative gradient of the loss**. The state of the art for tabular data.

### What is Gradient Boosting

An **ensemble** method that combines many **weak learners** (shallow decision trees) into a strong one. Unlike [Random Forests](/notes/ml-algorithms/supervised-learning/random-forests/) which builds trees **independently and in parallel** (bagging), boosting builds them **sequentially**, each one focused on what the previous ones got wrong.

$$F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \eta\, h_m(\mathbf{x})$$

where $h_m$ is the new tree, and $\eta$ is the **learning rate** (shrinkage).

### The Core Algorithm

1. Initialize with a constant prediction $F_0(\mathbf{x})$ (e.g. the mean for regression, log-odds for classification).
2. For $m = 1 \dots M$:
   - Compute **pseudo-residuals** = negative gradient of the loss w.r.t. current predictions: $r_{im} = -\left[\frac{\partial L(y_i, F(\mathbf{x}_i))}{\partial F(\mathbf{x}_i)}\right]_{F=F_{m-1}}$
   - Fit a tree $h_m$ to these residuals.
   - Update: $F_m = F_{m-1} + \eta\, h_m$.

For **squared error**, the negative gradient *is* the ordinary residual $(y_i - F(\mathbf{x}_i))$ — so boosting literally fits the leftover error. For other losses (log-loss, etc.) it generalizes to the gradient, hence "gradient" boosting.

### Regularization (why modern GBT wins)

- **Learning rate / shrinkage** ($\eta$): small values (0.01–0.1) + more trees → better generalization.
- **Number of trees** (`n_estimators`): tuned with **early stopping** on a validation set.
- **Tree constraints**: `max_depth`, `min_child_weight`, `min_samples_leaf`.
- **Subsampling**: row sampling (stochastic gradient boosting) + column sampling per tree/split.
- **L1/L2 penalties** on leaf weights (XGBoost).

### The Popular Implementations

| Library | Distinguishing feature |
|---|---|
| **XGBoost** | Regularized objective, 2nd-order (Newton) boosting, sparsity-aware |
| **LightGBM** | Leaf-wise growth + histogram binning → very fast on large data |
| **CatBoost** | Native categorical handling, ordered boosting (less target leakage) |

### Strengths

1. **Best-in-class accuracy on tabular data.**
2. Handles mixed feature types, missing values (XGBoost/LightGBM).
3. No scaling required; captures non-linear interactions.
4. Built-in feature importance.

### Weaknesses

1. **Sequential → harder to parallelize** than RF; slower to train.
2. **Sensitive to hyperparameters**; can overfit without early stopping.
3. Less interpretable; more tuning effort than Random Forests.
4. Can struggle with very high-dimensional sparse data vs linear models.

### Related
- [Decision Trees](/notes/ml-algorithms/supervised-learning/decision-trees/) — the weak learner being boosted
- [Random Forests](/notes/ml-algorithms/supervised-learning/random-forests/) — the bagging counterpart
- [Backpropagation](/notes/ml-algorithms/core-concepts/back-propagation/) · [Activation Functions and Optimizers](/notes/ml-algorithms/core-concepts/activation-functions-and-optimizers/) — the gradient-descent idea applied to trees
- [Metrics](/notes/ml-algorithms/core-concepts/metrics/) · [Bias-Variance Tradeoff](/notes/ml-algorithms/core-concepts/common-concepts/)
