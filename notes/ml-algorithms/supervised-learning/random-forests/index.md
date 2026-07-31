---
layout: note
title: "Random Forests"
description: "Random Forest is an ensemble learning algorithm that combines many decision trees into a single, more robust model. Each tree is trained on a different random subset of the data…"
note: true
note_collection: "ML algorithms"
note_section: "Supervised Learning"
section_order: 2
note_order: 7
updated: 2026-06-06 11:30:50 -0700
keywords:
  - Supervised Learning
  - Trees
  - Training
  - Evaluation
  - Probability
math: true
mermaid: false
---
### What are Random Forests

**Random Forest** is an **ensemble learning algorithm** that combines many decision trees into a single, more robust model. Each tree is trained on a different random subset of the data and features, and the final prediction is made by **averaging** (regression) or **majority voting** (classification) across all trees.

**Coined by:** Leo Breiman (2001).

#### Core Idea — Two Sources of Randomness:

1. **Bagging (Bootstrap Aggregating):** Each tree is trained on a random bootstrap sample of the data.
2. **Random Feature Selection:** At each split, only a random subset of features is considered.

These two together = **Random Forest**. The goal is to create **decorrelated, diverse trees** whose individual errors cancel out when averaged.

### Why Random Forest? — Motivation

#### The Problem with Single Trees:

- **High variance** — small changes in training data → drastically different trees.
- **Overfitting** — deep trees memorize noise.
- **Greedy** — locally optimal splits, not globally optimal.

#### How Random Forest Fixes It:

- **Averaging many trees reduces variance** (mathematical fact — averaging reduces variance of estimators).
- **Randomness ensures trees are different** — they make different errors that cancel out.
- **No need for pruning** — overfitting is controlled by ensemble averaging.

#### Result:

- High accuracy
- Robustness to noise and outliers
- Reduces overfitting dramatically
- Works well out-of-the-box with minimal tuning

### Random Forest = Bagging + Random Feature Selection

#### The Problem with Pure Bagging:

If we just bag decision trees, all trees tend to make the same splits (because the same "best" features dominate). This makes them **highly correlated** — and as the formula above shows, correlation limits the benefit of averaging.

#### The Random Forest Trick:

At **each split**, instead of considering all _n_ features, randomly select a subset of features and only consider splits on those.

This forces trees to use different features, making them **more diverse and decorrelated** → bigger variance reduction from averaging.

#### How Many Features to Sample?

A hyperparameter, denoted _m_try_ or `max_features`:

- **Classification (rule of thumb):** n\sqrt{n} n​ — square root of total features.
- **Regression (rule of thumb):** n/3n/3 n/3.
- Can be tuned via cross-validation.
- **`max_features=n` would reduce RF back to bagging.**

### The Random Forest Algorithm — Step by Step

**Training:**

1. For b=1 to B (number of trees): a. Draw a bootstrap sample of size *m* from training data. b. Grow a decision tree on this sample:
    - At each node, randomly select $m_{try}$ features.
    - Find the best split among those features.
    - Split the node.
    - Recurse until stopping criteria met. c. **Trees are typically grown deep with no pruning** — overfitting is controlled by averaging.
2. Output the ensemble of B trees.

**Prediction:**

- **Classification:** Each tree votes for a class; the **majority class wins**. Or, for probabilities, average the probability outputs across trees.
- **Regression:** **Average the predictions** of all trees.

#### Hyperparameters to Tune:

- **`n_estimators` (B):** Number of trees. More is generally better, but with diminishing returns. Common: 100–1000.
- **`max_features` (m_try):** Features per split. $\sqrt{(n)}$ or $\frac{n}{3}$  
- **`max_depth`:** Tree depth. Often left unlimited.
- **`min_samples_split` / `min_samples_leaf`:** Minimum samples for splitting / in a leaf.
- **`bootstrap`:** Whether to use bootstrap sampling (default True).
- **`oob_score`:** Use OOB samples for validation (we'll cover this).

### Out-of-Bag (OOB) Evaluation — A Free Validation Set

**Out-of-Bag (OOB) samples** are the ~36.8% of training data not included in a particular tree's bootstrap sample.

#### The OOB Trick:

For each training sample $x_i$:

- Find all trees where xix_i xi​ was NOT in the bootstrap sample (OOB trees for $x_i$​).
- Predict $x_i$ using only these trees.
- Compare to true label.

The average error over all samples is the **OOB error**.

#### Why This is Powerful:

- **No need for a separate validation set or cross-validation.**
- OOB error is an unbiased estimate of generalization error (similar to leave-one-out CV).
- Saves computation — built-in validation during training.

### Why Random Forest Works — The Math Intuition

#### Bias-Variance Decomposition:

Total error = Bias² + Variance + Irreducible error.

- **Single deep tree:** Low bias, **very high variance**.
- **Random Forest:** Same low bias (each tree is unbiased on its bootstrap sample), **dramatically reduced variance** through averaging.

#### Two-Level Variance Reduction:

1. **Bagging** averages over different bootstrap samples → reduces variance due to data sampling.
2. **Random feature selection** decorrelates trees → reduces residual correlation → enables more variance reduction from averaging.

#### Why More Trees Is Always Safe:

Adding more trees:

- **Never increases overfitting** (unlike adding depth to a single tree).
- Reduces variance monotonically.
- Eventually plateaus — beyond a certain point, additional trees don't help much.

---

### 8. Feature Importance in Random Forests

#### Two Main Methods:

#### (A) Mean Decrease in Impurity (MDI / Gini Importance):

For each feature, sum the impurity reduction (Gini or entropy) over all nodes that split on it, weighted by sample count, then average across all trees.

$$\text{Importance}(f) = \frac{1}{B}\sum_{b=1}^{B}\;\sum_{\text{nodes splitting on } f} \frac{n_t}{n}\, \Delta\text{impurity}$$
**Pros:** Fast (computed during training). **Cons:**

- Biased toward **high-cardinality features** (continuous or many-category).
- Biased toward features with many split opportunities.
- Can be unreliable with correlated features.

#### (B) Permutation Importance (Mean Decrease in Accuracy):

1. Compute baseline model accuracy on a validation/OOB set.
2. For each feature: randomly shuffle its values (breaking the relationship with target).
3. Measure the drop in accuracy.
4. Average over multiple shuffles.

**Pros:** More reliable, model-agnostic. **Cons:** Slower; correlated features can share importance ambiguously.

### Strengths of Random Forest

1. **High accuracy** — competitive with state-of-the-art models on tabular data.
2. **Robust to outliers and noise.**
3. **Handles missing values** (in some implementations) and mixed data types.
4. **No feature scaling needed** — splits are threshold-based.
5. **Out-of-Bag error** = free validation.
6. **Parallelizable** — trees are independent and can be trained in parallel.
7. **Provides feature importance.**
8. **Reduces overfitting** dramatically vs single trees.
9. **Few hyperparameters** to tune, works well with defaults.

---

### Weaknesses of Random Forest

1. **Less interpretable than a single tree** — you can't easily visualize 500 trees.
2. **Slower at prediction time** — must traverse all trees (though parallelizable).
3. **Memory-intensive** for large datasets and many trees.
4. **Doesn't handle very high-dimensional sparse data well** — gradient boosting / linear models often do better here (e.g., text).
5. **Can't extrapolate** for regression.
6. **Biased feature importance** (MDI is biased toward high-cardinality).
7. **Underperforms on data with strong linear relationships** — linear models would be better.
8. **Less accurate than gradient boosting (XGBoost, LightGBM)** on many tabular problems.
