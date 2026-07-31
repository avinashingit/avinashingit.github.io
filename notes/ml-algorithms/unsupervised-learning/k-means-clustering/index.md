---
layout: note
title: "k-Means Clustering"
description: "Before diving into K-Means, understand the paradigm shift:"
note: true
note_collection: "ML algorithms"
note_section: "Unsupervised Learning"
section_order: 3
note_order: 3
updated: 2026-06-06 11:30:50 -0700
keywords:
  - Clustering
  - Unsupervised Learning
  - Evaluation
  - Linear Models
  - Supervised Learning
math: true
mermaid: false
---
## Unsupervised Learning — Quick Foundation

Before diving into K-Means, understand the paradigm shift:

| Aspect | Supervised | Unsupervised |
|---|---|---|
| **Input** | $(X, y)$ — features + labels | $X$ only — features |
| **Goal** | Predict $y$ from $X$ | Find structure in $X$ |
| **Examples** | Classification, regression | Clustering, dim reduction, density estimation |
| **Evaluation** | Accuracy, MSE on labeled test set | Harder — no ground truth |
| **Algorithms** | LR, GBDT, NNs | K-Means, DBSCAN, PCA, GMM |

**Common unsupervised tasks:**

- **Clustering** — group similar samples (K-Means, DBSCAN, hierarchical).
- **Dimensionality reduction** — compress to fewer features (PCA, t-SNE, UMAP).
- **Density estimation** — model $P(x)$ (GMM, KDE).
- **Anomaly detection** — find unusual samples (Isolation Forest, One-Class SVM).
- **Association rule mining** — find patterns (Apriori).

> **Warning:** Why unsupervised learning is hard
> - **No ground truth** — how do you know if a clustering is "right"?
> - **Evaluation is subjective** — depends on downstream use.
> - **Hyperparameter selection is trickier** — no labels to validate against.

## What is K-Means?

K-Means is a **partitional clustering algorithm** that divides $n$ samples into $K$ non-overlapping clusters. Each sample belongs to **exactly one cluster** — the one whose centroid (mean) is closest.

**Core idea:**

- Each cluster is represented by its **centroid** — the mean of all points in the cluster.
- Each point is assigned to the **nearest centroid** (Euclidean distance).
- Centroids are updated to be the **means** of their assigned points.
- This process **iterates until convergence**.

**"K-Means" comes from:**
- $K$: number of clusters (a hyperparameter).
- **Means**: each cluster is represented by the mean of its points.

## The K-Means Algorithm — Step by Step

**Input**: Dataset $X = \{x_1, x_2, \dots, x_n\}$, number of clusters $K$.
**Output**: $K$ centroids and cluster assignments.

**Step 1 — Initialize**: choose $K$ initial centroids $\mu_1, \mu_2, \dots, \mu_K$.

**Step 2 — Assignment**: assign each point to the nearest centroid:

$$c_i = \arg\min_{k} \|x_i - \mu_k\|^2$$

**Step 3 — Update**: recompute centroids as the mean of assigned points:

$$\mu_k = \frac{1}{|C_k|}\sum_{i \in C_k} x_i$$

**Step 4 — Repeat** steps 2–3 until convergence (centroids stop moving, or max iterations reached).

> **Note:** Convergence
> - The algorithm is **guaranteed to converge** — each step decreases or maintains the objective.
> - However, it may converge to a **local minimum**, not the global one.
> - Convergence usually happens in **10–30 iterations** for typical data.

## The Objective Function — What K-Means Optimizes

K-Means minimizes the **within-cluster sum of squares (WCSS)**, also called **inertia**:

$$J = \sum_{k=1}^K \sum_{x_i \in C_k} \|x_i - \mu_k\|^2$$

- **Lower $J$** → tighter, more compact clusters.
- $J = 0$ means every point is a centroid (only when $K = n$).

### EM Perspective

K-Means is a special case of **Expectation-Maximization (EM)**:

- **E-step**: assign points to clusters (computes "expected" labels given current centroids).
- **M-step**: update centroids to maximize fit (compute means given assignments).

Each step **decreases $J$**, so the algorithm converges. But to a **local minimum**, since the objective is non-convex in cluster assignments.

### Mathematical Note

The update step (centroid = mean of points) is **provably optimal** because the mean minimizes squared distances:

$$\mu^* = \arg\min_\mu \sum_i \|x_i - \mu\|^2 \Rightarrow \mu^* = \frac{1}{n}\sum_i x_i$$

This is why K-Means uses **Euclidean distance** and **means** specifically — they're mathematically coupled.

## Initialization — A Critical Step

The initial centroids significantly affect the final clustering. **Bad initialization → bad clusters.**

### Random Initialization (Forgy Method)

Randomly pick $K$ data points as initial centroids. Simple, but can yield poor results and is highly sensitive to seed.

### K-Means++ (The Standard Today)

A smarter initialization scheme (Arthur & Vassilvitskii, 2007) that drastically improves clustering quality.

**Algorithm:**

1. Pick the **first centroid** uniformly at random from the data.
2. For each subsequent centroid:
	- Compute $D(x)$ = distance from each point to the **nearest already-chosen centroid**.
	- Pick the next centroid with probability **proportional to $D(x)^2$**.
3. Continue until $K$ centroids are chosen.
4. Run standard K-Means from these centroids.

> **Check:** Why K-Means++ works
> - **Spreads initial centroids far apart** — closer to the global optimum.
> - **Theoretical guarantee**: expected error is $O(\log K)$ times the optimal.
> - In practice, dramatically **faster convergence** and **better solutions**.
> - Used in **scikit-learn by default** (`init='k-means++'`).

### Multiple Initializations

Run K-Means multiple times with different initializations and pick the result with **lowest inertia**. In scikit-learn: `n_init=10` (default) means try 10 different starts.

## Choosing K — The Big Question

$K$ is a hyperparameter that must be specified upfront — but how do you choose it?

### The Elbow Method

Plot **inertia vs $K$**. Look for an "elbow" where the rate of decrease sharply slows.

Adding more clusters always decreases inertia, but at some point the gain becomes **marginal**. The elbow is where you get **diminishing returns**.

> **Warning:** Issue: often subjective — sometimes there's no clear elbow.

### Silhouette Score

For each point $i$:

$$s_i = \frac{b_i - a_i}{\max(a_i, b_i)}$$

Where:
- $a_i$ = average distance from $i$ to other points in its **own cluster** (cohesion).
- $b_i$ = average distance from $i$ to points in the **nearest other cluster** (separation).

**Range: $[-1, 1]$**

| Score | Interpretation |
|---|---|
| Near **+1** | Well-clustered |
| Near **0** | On the boundary between clusters |
| Near **−1** | Probably in the wrong cluster |

Pick $K$ with the **highest average silhouette score**.

### Other Methods

- **Gap Statistic** — compare inertia to a random uniform reference distribution; pick $K$ where the gap is largest.
- **Davies-Bouldin Index** — ratio of within-cluster scatter to between-cluster separation; lower is better.
- **Domain knowledge** — often the most practical; you know how many clusters make sense (e.g., 5 customer segments).

### Practical Workflow

1. Try several $K$ values (e.g., 2–10).
2. Plot **elbow + silhouette**.
3. Look for consensus.
4. Apply domain knowledge.

> **Note:** Sometimes there's no single "right" $K$ — just useful ones.

## Distance Metrics in K-Means

K-Means is defined for **Euclidean (L2) distance** — the centroid (mean) is provably optimal under squared Euclidean distance. Using other distances breaks the mathematical guarantee that the mean minimizes the objective.

### Alternatives for Other Distances

| Algorithm | Center type | Distance | Notes |
|---|---|---|---|
| **K-Medoids (PAM)** | Actual data points (medoids) | Any | More flexible, slower |
| **K-Medians** | Median of points | Manhattan | Robust to outliers |
| **Spherical K-Means** | Normalized centroids | Cosine | Standard for text/embeddings |

> **Tip:** Cosine similarity trick
> If using K-Means with cosine similarity, **normalize features to unit length first** — then Euclidean distance on unit vectors is equivalent to cosine.

## Feature Scaling — Critical!

K-Means is distance-based, so **feature scale matters enormously**.

**Example without scaling:**
- Feature 1: age (0–100)
- Feature 2: income (0–1,000,000)

Income **completely dominates** the distance computation — clusters form based on income alone.

**Always standardize features before K-Means:**

- **Z-score**: $x' = \frac{x - \mu}{\sigma}$ (mean 0, std 1).
- **Min-max**: $x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$ (range $[0, 1]$).

> Forgetting to scale features is one of the **most common mistakes** with K-Means. Always scale first.

## K-Means Assumptions (and When They Break)

| Assumption | What breaks it |
|---|---|
| Clusters are **spherical and isotropic** | Elongated, banana-shaped, or irregular clusters |
| Clusters are **roughly the same size** | Imbalanced cluster sizes — gets split or merged incorrectly |
| Clusters have **similar density** | Dense cluster next to a sparse one — misallocation |
| Data is **continuous and numerical** | Categorical features (need encoding or K-Modes) |
| **Euclidean distance is meaningful** | High-dim, sparse, or non-numerical data |
| **$K$ is known in advance** | Requires choosing $K$ — often the hardest part |

### When K-Means Fails

- **Non-convex clusters** (e.g., two interlocked half-moons).
- Clusters of **very different sizes or densities**.
- **High-dimensional data** (curse of dimensionality).
- **Noisy data** with many outliers.

> **Tip:** Alternatives when K-Means fails
> - **DBSCAN** — density-based; handles arbitrary shapes and outliers.
> - **Spectral Clustering** — graph-based; handles non-convex clusters.
> - **Hierarchical Clustering** — no need to specify $K$ upfront.
> - **Gaussian Mixture Models (GMM)** — soft probabilistic cluster assignments; handles elliptical clusters.

# K-Means Stopping Criteria — Deep Dive

K-Means is iterative — it keeps alternating between assignment and update steps. Without a stopping rule:

- The algorithm could run **forever** (in theory).
- Or waste compute on **tiny improvements** that don't matter.
- Or stop **too early** before reaching a good solution.

A good stopping criterion balances **quality**, **efficiency**, and **robustness**. In practice, K-Means usually converges in **10–30 iterations** for typical data, so stopping criteria mostly serve as **safeguards** rather than strict optimality controls.

## The Main Stopping Criteria

Production implementations like scikit-learn typically use **multiple criteria simultaneously** — stopping as soon as any is satisfied.

### (A) Centroids Stop Moving (Convergence)

The classic criterion: stop when centroids **stabilize** between iterations:

$$\max_{k} \|\mu_k^{(t)} - \mu_k^{(t-1)}\| < \text{tol}$$

Or with sum of squared shifts:

$$\sum_{k=1}^K \|\mu_k^{(t)} - \mu_k^{(t-1)}\|^2 < \text{tol}$$

If centroids barely moved this iteration, **further iterations won't change much**.

- Threshold `tol`: small positive number (e.g., $10^{-4}$).
- In scikit-learn: the `tol` parameter (default `1e-4`); checks **Frobenius norm** of centroid shift.

### (B) Cluster Assignments Stop Changing

Stop when **no points change clusters** between iterations.

If every point stays in the same cluster, centroids will compute the same means → algorithm has converged **exactly**.

- This is the **strictest** convergence criterion — guarantees fixed-point convergence.
- In practice, always satisfied when (A) is satisfied with small enough tolerance.
- Often combined with (A) for safety.

> **Warning:** Edge case: ties
> With ties (a point equidistant from multiple centroids), assignments may **oscillate**. Tie-breaking rules (e.g., pick lowest cluster ID) prevent infinite loops.

### (C) Inertia Stops Decreasing

Stop when the objective function (WCSS) decrease becomes **negligible**:

$$\frac{J^{(t-1)} - J^{(t)}}{J^{(t-1)}} < \text{tol}$$

Or absolute change:

$$|J^{(t-1)} - J^{(t)}| < \text{tol}$$

- Since K-Means **monotonically decreases** inertia, the change is always non-negative.
- Easy to compute — already calculated each iteration.
- Provides a **smooth, principled** stopping signal.

### (D) Maximum Iterations

A **hard cap** on iterations regardless of convergence.

- **Safety net** — guarantees the algorithm terminates.
- **Resource control** — caps compute cost.
- **Production reliability** — prevents pathological cases from hanging.

Typical default: 100–300 iterations (scikit-learn default: `max_iter=300`).

> **Warning:** If `max_iter` is hit, something is wrong — bad initialization, weird data — and you should investigate.

### (E) Wall-Clock Time Limit (Production)

In some production settings, cap by **time** rather than iterations (e.g., "run for at most 30 seconds"). Less common but pragmatic when integrated into latency-sensitive pipelines.

## How They're Combined in Practice

Real implementations run multiple criteria simultaneously and stop as soon as **any** is satisfied:

```python
while iteration < max_iter:
    new_centroids = update_step(...)
    if shift(new_centroids, old_centroids) < tol:  # Criterion A
        break
    if assignments_unchanged():                     # Criterion B
        break
    old_centroids = new_centroids
    iteration += 1
# Hit max_iter (Criterion D) → exit loop
```

**Scikit-learn specifically:**
- Uses **Frobenius norm** of centroid shift `< tol` (criterion A) as primary.
- Also checks if inertia stops decreasing (criterion C).
- Has hard `max_iter` cap (criterion D).

## Why K-Means Is Guaranteed to Converge

K-Means optimizes:

$$J = \sum_{k=1}^K \sum_{x_i \in C_k} \|x_i - \mu_k\|^2$$

**Each step decreases or maintains $J$:**

- **Assignment step**: each point goes to its nearest centroid. Moving a point closer always **reduces $J$**.
- **Update step**: setting $\mu_k$ = mean of $C_k$ is provably the value that **minimizes** $\sum_{i \in C_k}\|x_i - \mu_k\|^2$ for fixed $C_k$. So the update can only **decrease or maintain $J$**.

Therefore: $J$ is **monotonically non-increasing**.

**Why it must terminate:**

- $J \geq 0$ — sum of squared distances is non-negative.
- $J$ is non-increasing each iteration.
- There are **finitely many possible cluster assignments** ($K^n$ to be precise — finite for any finite dataset).
- If $J$ ever doesn't strictly decrease, the assignments must **repeat** — and once they repeat, the centroids repeat, and the algorithm has **converged**.

> **Note:** Conclusion
> K-Means must terminate in a **finite number of iterations**, even without explicit stopping criteria. The stopping criteria exist for **efficiency and robustness**, not for theoretical convergence guarantees.
