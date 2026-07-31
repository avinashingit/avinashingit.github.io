---
layout: note
title: "kNN"
description: "K-Nearest Neighbors is a non-parametric, instance-based supervised learning algorithm used for both classification and regression. It makes predictions based on the K closest tr…"
note: true
note_collection: "ML algorithms"
note_section: "Supervised Learning"
section_order: 2
note_order: 3
updated: 2026-06-06 11:30:50 -0700
keywords:
  - Supervised Learning
  - Embeddings
  - Evaluation
  - Retrieval
  - Training
math: true
mermaid: false
---
**K-Nearest Neighbors** is a **non-parametric, instance-based** supervised learning algorithm used for both **classification and regression**. It makes predictions based on the **K closest training examples** to a query point in feature space.

The core idea is disarmingly simple: *"Tell me your neighbors, and I'll tell you who you are."*

## How KNN Works

To predict the label for a new point $x_{\text{query}}$:

1. **Compute distances** from $x_{\text{query}}$ to every point in the training set.
2. **Find the $K$ nearest neighbors** — the $K$ training points with the smallest distances.
3. **Aggregate their labels**:
	- **Classification** → **majority vote** among the $K$ neighbors.
	- **Regression** → **mean (or weighted mean)** of the $K$ neighbors' target values.
4. **Return the prediction**.

That's it. There is no "training" in the traditional sense — KNN just **memorizes the training data**.

## Lazy Learning vs Eager Learning

KNN is the prototypical **lazy learner**:

| Aspect | Eager Learners (e.g., Logistic Reg, Trees, NN) | Lazy Learners (KNN) |
|---|---|---|
| **Training phase** | Builds a model (fits parameters) | Just stores the data |
| **Training cost** | Expensive | $O(1)$ — trivial |
| **Prediction cost** | Cheap (one forward pass) | Expensive ($O(n)$ per query) |
| **Memory** | Stores parameters only | Stores **entire training set** |
| **Model representation** | Explicit (weights, splits, etc.) | Implicit (the data *is* the model) |

> **Note:** Why "non-parametric"?
> KNN has **no fixed number of parameters** that summarize the data. The number of "parameters" grows with the dataset itself — every training example is effectively a parameter. This makes KNN extremely **flexible** but also **memory-heavy**.

## Distance Metrics

The choice of distance function defines what "near" means.

### Euclidean Distance (L2)

$$d(x, x') = \sqrt{\sum_{j=1}^{n}(x_j - x'_j)^2}$$

Straight-line distance. Works well for continuous, comparably-scaled features. **Default choice.**

### Manhattan Distance (L1, taxicab)

$$d(x, x') = \sum_{j=1}^{n}|x_j - x'_j|$$

Sum of absolute differences. More robust to outliers; useful in high dimensions.

### Minkowski Distance (general L$p$)

$$d(x, x') = \left(\sum_{j=1}^{n}|x_j - x'_j|^p\right)^{1/p}$$

Generalizes Euclidean ($p = 2$) and Manhattan ($p = 1$).

### Cosine Similarity (angle-based)

$$\text{sim}(x, x') = \frac{x \cdot x'}{\|x\|\, \|x'\|}, \quad d = 1 - \text{sim}$$

Measures angle, not magnitude. Standard for **text** (TF-IDF vectors, embeddings) and **high-dimensional sparse data**.

### Hamming Distance

Count of disagreeing positions. Used for **categorical / binary** features.

### Feature Scaling Is Critical

> **Warning:** Always scale before KNN
> KNN is **extremely sensitive to feature scales** because distance is dominated by features with larger numeric ranges. A feature measured in dollars (range: 0–100,000) will completely overwhelm a feature measured in years (range: 0–100), regardless of which is actually informative.
> **Always standardize or normalize** features before using KNN.

## Choosing K

$K$ is the central hyperparameter — it controls the **bias-variance tradeoff**:

| $K$ | Behavior | Bias / Variance |
|---|---|---|
| **$K = 1$** | Each query takes the label of its single nearest neighbor | **Low bias, high variance** — overfits, noisy boundary |
| **Small $K$** | Boundary follows local structure | Flexible, can overfit |
| **Large $K$** | Boundary smooths out, averages over many neighbors | **High bias, low variance** — underfits |
| **$K = n$** | Predicts the global majority class / overall mean | Equivalent to predicting the **most common label** for every query |

### Practical Heuristics

- Use **cross-validation** to choose $K$.
- A common rule of thumb: $K \approx \sqrt{n}$.
- For **binary classification**, prefer an **odd $K$** to avoid ties in voting.
- Plot **validation error vs $K$** and look for the "elbow."

## Weighted KNN

Instead of treating all $K$ neighbors equally, weight them by **proximity** — closer neighbors influence the prediction more:

$$\hat{y} = \frac{\sum_{i=1}^{K} w_i \, y_i}{\sum_{i=1}^{K} w_i}, \quad w_i = \frac{1}{d(x_{\text{query}}, x_i) + \epsilon}$$

This often improves performance, especially for larger $K$, because distant neighbors contribute less. Scikit-learn exposes this via `weights="distance"`.

## Decision Boundary

The KNN decision boundary is **piecewise linear** and can be **arbitrarily complex** — it's defined implicitly by the training points and the value of $K$.

- **$K = 1$**: boundary is the **Voronoi diagram** of the training set (highly jagged).
- **Larger $K$**: boundary becomes **smoother** as local noise averages out.

KNN can approximate **any decision boundary** given enough data — it's a **universal approximator** in the limit.

## Advantages & Disadvantages

> **Check:** Advantages
> - **Conceptually simple** — easy to understand and explain.
> - **No training phase** — instant model "training."
> - **No assumptions** about data distribution (non-parametric).
> - **Naturally handles multi-class** classification (no special tricks needed).
> - **Adapts to local structure** — works well when decision boundaries are irregular.
> - **Universal approximator** — given enough data, can fit any boundary.

> **Warning:** Disadvantages
> - **Slow at prediction time** — $O(n \cdot d)$ per query, where $d$ is the feature dimension.
> - **Memory-heavy** — must store the entire training set.
> - **Sensitive to feature scaling** — must normalize.
> - **Sensitive to irrelevant features** — distance gets polluted by noise dimensions.
> - **Suffers in high dimensions** — see *curse of dimensionality* below.
> - **Sensitive to imbalanced data** — majority class dominates votes.
> - **No interpretability** — there's no model to inspect.

## The Curse of Dimensionality

KNN's biggest theoretical weakness. As the number of features $n$ grows:

- **Distances concentrate**: in high dimensions, the ratio of the nearest to the farthest neighbor approaches 1. Every point looks roughly equidistant from every other point.
- **Volume explodes**: to maintain the same data density, you need exponentially more points as $n$ grows.
- **Notion of "nearest" loses meaning**: with too many dimensions, "nearest" becomes essentially random.

### Mitigations

- **Dimensionality reduction** (PCA, t-SNE, UMAP, autoencoders) before applying KNN.
- **Feature selection** to keep only informative dimensions.
- **Learned distance metrics** (e.g., metric learning, embeddings).

## Speeding Up KNN

Naive KNN is $O(n)$ per query — fine for small datasets, painful for large ones. Common accelerations:

| Method | Idea | Complexity |
|---|---|---|
| **KD-Tree** | Recursively partition space along axes | $O(\log n)$ per query in **low** dimensions; degenerates to $O(n)$ when $d \gtrsim 20$ |
| **Ball Tree** | Partition into nested hyperspheres | Better than KD-tree in moderate dimensions |
| **Locality-Sensitive Hashing (LSH)** | Hash similar points to the same bucket | **Approximate** but very fast in high dimensions |
| **HNSW / Annoy / FAISS** | Graph-based or quantization-based approximate nearest neighbor (ANN) | Sub-linear; standard for **production** vector search at scale |

> **Note:** Modern relevance
> KNN is the foundation of **vector search** in modern AI systems — **semantic search**, **retrieval-augmented generation (RAG)**, **recommendation systems**, and **embedding-based retrieval** all boil down to approximate KNN over learned embeddings. Libraries like FAISS, ScaNN, and Pinecone exist specifically to make this fast.

## KNN for Regression

The same algorithm, but instead of majority voting, **average the target values** of the $K$ neighbors:

$$\hat{y}_{\text{query}} = \frac{1}{K}\sum_{i=1}^{K} y_i$$

Or, with distance weighting:

$$\hat{y}_{\text{query}} = \frac{\sum_{i=1}^{K} w_i \, y_i}{\sum_{i=1}^{K} w_i}$$

KNN regression produces a **piecewise constant** prediction surface (or piecewise smooth with distance weighting).

## When to Use KNN

> **Tip:** KNN is a good fit when…
> - The dataset is **small to medium** (thousands, not millions).
> - The **decision boundary is irregular** or unknown.
> - **Local structure** matters more than global trends.
> - You want a **strong baseline** with minimal assumptions.
> - You're building a **recommendation** or **retrieval** system (with ANN methods).

> **Warning:** Avoid KNN when…
> - The dataset is **very large** (millions of points) without ANN acceleration.
> - The feature space is **high-dimensional** without reduction.
> - **Inference latency** matters and you can't precompute.
> - **Interpretability** of decisions is required.
