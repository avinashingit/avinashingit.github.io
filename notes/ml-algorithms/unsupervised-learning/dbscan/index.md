---
layout: note
title: "DBScan"
description: "A cluster is a region of high point density. DBSCAN grows clusters from dense \"core\" points and leaves low-density points as noise. It finds arbitrarily shaped clusters and is r…"
note: true
note_collection: "ML algorithms"
note_section: "Unsupervised Learning"
section_order: 3
note_order: 1
updated: 2026-06-06 11:36:30 -0700
keywords:
  - Clustering
  - Unsupervised Learning
  - Retrieval
  - Evaluation
math: true
mermaid: false
---
> **Density-Based Spatial Clustering of Applications with Noise.** Clusters are **dense regions** separated by sparse ones. Unlike [k-Means Clustering](/notes/ml-algorithms/unsupervised-learning/k-means-clustering/), you don't pick the number of clusters — and it labels outliers as noise.

### Core Idea

A cluster is a region of **high point density**. DBSCAN grows clusters from dense "core" points and leaves low-density points as **noise**. It finds **arbitrarily shaped** clusters and is robust to outliers.

### The Two Parameters

- **`eps` (ε):** the radius of a neighborhood around a point.
- **`min_samples` (MinPts):** minimum points within ε for a point to be "dense."

### Point Types

| Type | Definition |
|---|---|
| **Core point** | Has ≥ `min_samples` points within ε (including itself) |
| **Border point** | Within ε of a core point, but not itself a core point |
| **Noise point** | Neither core nor border — an outlier |

### The Algorithm

1. Pick an unvisited point $p$.
2. Find all points within ε of $p$ (its ε-neighborhood).
3. If $p$ is a **core point**, start a new cluster and **expand** it: recursively add all density-reachable points (core points pull in their neighbors).
4. If $p$ is not core, tentatively label it **noise** (it may later become a border point of another cluster).
5. Repeat until all points are visited.

### Choosing `eps` — the k-distance plot

Compute each point's distance to its $k$-th nearest neighbor ($k = $ `min_samples`), sort descending, and plot. The **"elbow"/knee** of this curve is a good `eps`. Rule of thumb: `min_samples` ≈ $2 \times$ dimensions.

### DBSCAN vs K-Means

| | [k-Means Clustering](/notes/ml-algorithms/unsupervised-learning/k-means-clustering/) | DBSCAN |
|---|---|---|
| # clusters | Must specify `k` | Discovered automatically |
| Cluster shape | Spherical / convex | Arbitrary shape |
| Outliers | Forced into a cluster | Labeled as noise |
| Parameters | `k` | `eps`, `min_samples` |
| Varying density | OK | **Struggles** |

### Strengths

1. **No need to pre-specify the number of clusters.**
2. Finds **arbitrarily shaped** clusters.
3. **Robust to outliers** (explicit noise label).
4. Only two intuitive parameters.

### Weaknesses

1. **Struggles with varying density** clusters (one `eps` doesn't fit all) → see **HDBSCAN**.
2. **Curse of dimensionality** — distance becomes meaningless in high dimensions.
3. Sensitive to `eps` / `min_samples` choice.
4. Border points can be assigned non-deterministically depending on processing order.

### Related
- [k-Means Clustering](/notes/ml-algorithms/unsupervised-learning/k-means-clustering/) — the centroid-based alternative
- Hierarchical Clustering — another non-parametric clustering approach
- [Metrics](/notes/ml-algorithms/core-concepts/metrics/) — clustering evaluation (silhouette, etc.)
