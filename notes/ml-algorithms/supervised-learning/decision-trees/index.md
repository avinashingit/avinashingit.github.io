---
layout: note
title: "Decision Trees"
description: "A Decision Tree is a supervised learning algorithm used for both classification and regression. It splits data into subsets based on feature values, forming a tree-like structur…"
note: true
note_collection: "ML algorithms"
note_section: "Supervised Learning"
section_order: 2
note_order: 1
updated: 2026-06-06 11:30:50 -0700
keywords:
  - Supervised Learning
  - Trees
  - Deep Learning
  - Linear Models
  - Clustering
math: true
mermaid: false
---
### What are Decision Trees

A **Decision Tree** is a supervised learning algorithm used for both **classification and regression**. It splits data into subsets based on feature values, forming a **tree-like structure of decisions**.

**Structure**

- **Root Node** — the topmost node; contains the entire dataset.
- **Internal Nodes** — represent decisions based on a feature (e.g., *"Age > 30?"*).
- **Branches** — outcomes of decisions (Yes / No paths).
- **Leaf Nodes (Terminal Nodes)** — final predictions:
	- **Class label** for classification.
	- **Numeric value** for regression.

**Visual**

            [Age > 30?]
            /        \
          Yes          No
          /              \
    [Income > 50K?]    [Student?]
     /      \           /     \
    Yes     No        Yes      No
    |       |          |        |
    Buy    Don't       Buy     Don't

**How It Makes Predictions**

1. Start at the **root**.
2. At each **internal node**, check the feature condition.
3. Go down the **branch** matching the answer.
4. When you reach a **leaf**, output its prediction.

It's essentially a series of **if-else rules learned automatically** from data — which is exactly why trees feel so intuitive compared to algorithms like neural networks or SVMs.

---

**Why Decision Trees?**

> **Check:** Advantages
> - **Highly interpretable** — easy to visualize and explain to non-technical stakeholders.
> - Handles both **numerical and categorical features** natively (depending on implementation).
> - **No feature scaling** required — splits are based on thresholds, not distances. No need to standardize or normalize.
> - **Non-linear relationships** captured naturally — no need for polynomial features or kernels.
> - **Handles missing values** in many implementations (e.g., surrogate splits in CART).
> - **Captures feature interactions** automatically — every path from root to leaf is an interaction.
> - **Fast inference** — prediction is just a series of comparisons (O(depth)).

> **Warning:** Disadvantages
> - **Prone to overfitting**, especially when grown deep — a tree can memorize training data exactly.
> - **High variance** — small changes in the data can produce a very different tree. (This is what motivates **bagging** and **Random Forests**.)
> - **Biased toward features with more levels** — features with many possible split points have more chances to look "good" by chance.
> - **Greedy algorithm** — picks the locally optimal split at each node, which doesn't guarantee a globally optimal tree.
> - **Poor extrapolation** — can't predict values **outside the training range** in regression (leaves output constants).
> - **Axis-aligned splits only** — struggles with diagonal decision boundaries unless features are engineered.

> **Tip:** When are decision trees the right choice?
> Single decision trees are rarely state-of-the-art on their own, but they shine when:
> - **Interpretability is critical** (medical, legal, regulated industries).
> - You need a **fast, simple baseline** before reaching for heavier models.
> - You're using them as **building blocks** for ensemble methods — **Random Forests** (bagging) and **Gradient Boosted Trees** (XGBoost, LightGBM, CatBoost) routinely win on tabular data.

- - - 
### How Does a Decision Tree Learn? — The Core Algorithm

Decision trees are built **top-down** using a **greedy, recursive approach** called **Recursive Binary Splitting**:
#### Pseudo-algorithm:

1. Start with all data at the root.
2. For each feature, find the best split point that maximizes a chosen criterion.
3. Choose the feature and threshold that gives the **best overall split**.
4. Split the data into two (or more) subsets.
5. Repeat steps 2-4 recursively on each subset.
6. Stop when a **stopping criterion** is met (depth, min samples, purity, etc.).
7. Assign a prediction to each leaf:
    - **Classification:** Majority class.
    - **Regression:** Mean of target values.

The big question: **how do we decide what makes a "best" split?** That's where splitting criteria come in.

- - -
**Splitting Criteria — The Heart of Decision Trees**

The goal of any split: make resulting subsets more **"pure"** — i.e., more **homogeneous** in terms of the target variable.

- For **classification**, "pure" means most samples belong to the same class.
- For **regression**, "pure" means values are close to each other (low variance).

Three main criteria to know.

---
### Splitting Criteria — The Heart of Decision Trees

**(A) Gini Impurity — For Classification**

Gini Impurity measures the **probability of incorrectly classifying** a randomly chosen sample if it were labeled randomly according to the class distribution in the node.

$$\text{Gini}(t) = 1 - \sum_{k=1}^{K} p_k^2$$

Where:
- $K$ = number of classes
- $p_k$ = proportion of class $k$ in node $t$

**Range**: $[0,\, 1 - 1/K]$

- $0$ → **pure node** (all samples same class).
- Maximum → classes equally distributed.

**Examples (binary case)**

| Node composition | Calculation | Gini |
|---|---|---|
| [10 pos, 0 neg] | $1 - (1^2 + 0^2)$ | **0** ✅ pure |
| [5 pos, 5 neg] | $1 - (0.5^2 + 0.5^2)$ | **0.5** (maximally impure) |
| [8 pos, 2 neg] | $1 - (0.8^2 + 0.2^2)$ | **0.32** |

**Gini for a split** — compute the weighted Gini of the children:

$$\text{Gini}_{\text{split}} = \frac{n_L}{n} \text{Gini}(L) + \frac{n_R}{n} \text{Gini}(R)$$

We pick the split that **minimizes the weighted Gini**.

> **Note:** Intuition
> Gini = probability that **two random samples** from the node have **different classes**. Lower = more homogeneous.

---

**(B) Entropy & Information Gain — For Classification**

Entropy comes from **information theory** and measures the **impurity / disorder** of a node.

$$\text{Entropy}(t) = -\sum_{k=1}^{K} p_k \log_2(p_k)$$

Where $p_k$ is the proportion of class $k$ in node $t$.

**Range**: $[0,\, \log_2(K)]$

- $0$ → pure node.
- $\log_2(K)$ → classes equally distributed (maximum disorder).

**Examples (binary case)**

| Node composition | Calculation | Entropy |
|---|---|---|
| [10, 0] | $-1\log_2(1) - 0\log_2(0)$ | **0** ✅ (convention: $0\log 0 = 0$) |
| [5, 5] | $-0.5\log_2(0.5) \times 2$ | **1** (max for binary) |
| [8, 2] | $-0.8\log_2(0.8) - 0.2\log_2(0.2)$ | **≈ 0.72** |

**Information Gain** — the reduction in entropy after a split:

$$\text{IG} = \text{Entropy(parent)} - \sum_{c \in \text{children}} \frac{n_c}{n}\, \text{Entropy}(c)$$

We pick the split that **maximizes Information Gain** — equivalently, the split with the **lowest weighted entropy** in children.

> **Note:** Intuition
> Entropy = average number of **bits** needed to encode the class of a sample. A pure node needs **0 bits**; a balanced binary node needs **1 bit**. Information Gain tells us how many bits the split "saves."

---

**(C) Variance Reduction — For Regression**

For regression trees, the target is **continuous**, not categorical, so Gini and entropy don't apply. Instead, we minimize **variance** (or sum of squared errors / MSE).

**Variance of a node**

$$\text{Var}(t) = \frac{1}{n_t}\sum_{i \in t}(y_i - \bar{y}_t)^2$$

Where $\bar{y}_t$ is the mean of $y$ in node $t$.

**Variance Reduction (split criterion)**

$$\text{Variance Reduction} = \text{Var(parent)} - \left[\frac{n_L}{n}\text{Var}(L) + \frac{n_R}{n}\text{Var}(R)\right]$$

We pick the split that **maximizes variance reduction** (or equivalently, **minimizes the weighted variance** of children).

**Equivalent formulations in libraries**

- **MSE** (squared error) — standard default.
- **MAE** (Friedman MSE) — more robust to outliers.
- **Poisson** — for count data.

> **Tip:** Leaf prediction in regression
> The prediction at a regression leaf is simply the **mean** of target values in that leaf. (For MAE-based trees, it's the **median**.)

---

**Gini vs Entropy — Which to Use?**

| Aspect | Gini Impurity | Entropy |
|---|---|---|
| **Formula** | $1 - \sum p_k^2$ | $-\sum p_k \log_2 p_k$ |
| **Computational cost** | **Faster** (no log) | Slower (uses log) |
| **Range (binary)** | $[0,\, 0.5]$ | $[0,\, 1]$ |
| **Behavior** | Slightly biased toward **larger partitions** | Slightly more **balanced** splits |
| **Default in scikit-learn** | ✅ Gini | Optional (`criterion="entropy"`) |

> **Note:** In practice
> They usually produce **very similar trees**. Gini is **faster** and is the default in scikit-learn. Entropy has a stronger **theoretical (information-theoretic) basis**. Choose based on speed and convention — performance differences are typically tiny.

> **Tip:** Useful approximation
> For binary classification near the boundary:
> $$\text{Entropy} \approx 2 \times \text{Gini}$$
> This is why they tend to rank splits in the same order, even though their absolute values differ.

- - - 
### Stopping Criteria & Tree Pruning

A tree grown to full depth will overfit (every leaf has 1 sample → 100% training accuracy, terrible generalization). We need to stop or prune.

#### Pre-Pruning (Stopping Early):

Apply during tree construction:

- **`max_depth`** — limit the depth of the tree.
- **`min_samples_split`** — minimum samples required to split a node.
- **`min_samples_leaf`** — minimum samples in a leaf.
- **`max_leaf_nodes`** — total leaves allowed.
- **`min_impurity_decrease`** — split only if it improves the criterion by this amount.

#### Post-Pruning:

Grow the full tree, then prune subtrees that don't generalize well.

**Cost-Complexity Pruning**

$$R_\alpha(T) = R(T) + \alpha |T|$$

This is the **cost-complexity criterion** used to prune decision trees (introduced in the CART algorithm by Breiman et al.).

**Breaking down the terms**

| Symbol | Meaning |
|---|---|
| $T$ | A subtree (candidate pruned version of the full tree) |
| $R(T)$ | **Training error / risk** of the tree — total impurity or misclassification cost across all leaves |
| $\lvert T \rvert$ | **Number of leaf nodes** in $T$ (a measure of tree complexity) |
| $\alpha \geq 0$ | **Complexity parameter** — penalty per leaf |
| $R_\alpha(T)$ | **Cost-complexity** of $T$ — the regularized objective we minimize |

**The intuition**

This is just **regularized risk**, with the same structure as Ridge / Lasso:

$$\underbrace{R(T)}_{\text{fit to data}} + \underbrace{\alpha |T|}_{\text{complexity penalty}}$$

- A **bigger tree** fits training data better → lower $R(T)$.
- But a bigger tree has more leaves → higher $\alpha |T|$.
- We trade off **accuracy vs simplicity**, controlled by $\alpha$.

**The role of $\alpha$**

| $\alpha$ | Effect | Result |
|---|---|---|
| $\alpha = 0$ | No penalty for complexity | **Full tree** (overfit) |
| Small $\alpha$ | Mild penalty | Large tree, lightly pruned |
| Large $\alpha$ | Heavy penalty | Small tree, aggressively pruned |
| $\alpha \to \infty$ | Penalty dominates | **Root only** (underfit) |

**How pruning actually works (Minimal Cost-Complexity Pruning)**

1. **Grow** the tree fully (or to some stopping criterion) → call it $T_0$.
2. For each internal node, compute the **"effective $\alpha$"** — the value of $\alpha$ at which collapsing that node into a leaf would leave $R_\alpha$ unchanged:
$$\alpha_{\text{eff}}(t) = \frac{R(t) - R(T_t)}{|T_t| - 1}$$
where $T_t$ is the subtree rooted at $t$, and $R(t)$ is the error if $t$ were a leaf.
3. **Collapse** the node with the smallest $\alpha_{\text{eff}}$ — it's the "cheapest" to prune.
4. Repeat → produces a **nested sequence** of trees $T_0 \supset T_1 \supset \dots \supset \{\text{root}\}$.
5. **Pick the best tree** in this sequence using **cross-validation**.

> **Note:** Why this matters
> Pruning solves decision trees' biggest weakness: **overfitting**. A fully grown tree memorizes the training set; pruning trades a small amount of training accuracy for substantially better generalization.

> **Tip:** In scikit-learn
> `DecisionTreeClassifier` and `DecisionTreeRegressor` expose this directly via the **`ccp_alpha`** parameter. Use `cost_complexity_pruning_path()` to get the full sequence of $\alpha$ values and corresponding trees, then pick $\alpha$ via cross-validation.

> **Warning:** Pre-pruning vs post-pruning
> - **Pre-pruning** (early stopping): limit `max_depth`, `min_samples_leaf`, `min_samples_split` during growth. Fast but **greedy** — may stop before discovering useful deeper splits.
> - **Post-pruning** (cost-complexity, above): grow fully, then cut back. **More principled** but more expensive. CART uses post-pruning.

- - - 
### Feature Importance in Decision Trees

Trees naturally provide a **feature importance score** — a measure of how much each feature contributes to reducing impurity across the tree.

**Definition (Mean Decrease in Impurity, MDI)**

$$\text{Importance}(f) = \sum_{\text{nodes splitting on } f} \frac{n_t}{n} \cdot \Delta\text{impurity}$$

Where:
- The sum runs over **all internal nodes** that split on feature $f$.
- $n_t$ = number of samples reaching node $t$.
- $n$ = total number of samples.
- $\Delta\text{impurity}$ = reduction in impurity (Gini, entropy, or variance) achieved by the split.

In words: **sum the impurity reductions across every split that uses feature $f$, weighted by how many samples pass through each node.**

**Intuition**

A feature is important if:
1. It's used **frequently** as a split, AND
2. Its splits produce **large impurity reductions**, AND
3. Those splits happen **high up the tree** (where more samples are affected — $n_t$ is larger).

Importances are typically **normalized to sum to 1**, so you can read them as relative contributions.

**Caveats — Why MDI Is Often Misleading**

> **Warning:** Known biases of impurity-based importance
> - **Cardinality bias**: features with **many unique values** (continuous variables, high-cardinality categoricals) have more potential split points and look artificially important.
> - **Correlated features**: if two features are correlated, the tree picks one **arbitrarily**, giving it all the importance while the other looks useless.
> - **Doesn't capture interactions well**: a feature only "useful in combination" gets a lower score than its true contribution.
> - **Training-set only**: MDI is computed on training data — features that overfit can score highly without actually generalizing.

**Better Alternatives**

| Method | What it measures | Pros | Cons |
|---|---|---|---|
| **MDI (default)** | Total impurity reduction | Free — comes with the tree | Biased (see above) |
| **Permutation importance** | Drop in **validation accuracy** when feature is shuffled | Model-agnostic; uses **held-out data**; unbiased | Slower; struggles with correlated features (shuffling one doesn't help if a correlated one is still informative) |
| **SHAP values** | Per-prediction contribution based on Shapley values from game theory | **Local + global** importance; handles interactions; theoretically principled | Computationally expensive (fast for tree models via TreeSHAP) |
| **Drop-column importance** | Retrain without the feature, measure performance drop | Most faithful measure | **Very** expensive — retrain once per feature |

> **Tip:** Practical recommendation
> - For **quick exploration** during model building: MDI is fine.
> - For **reporting, decisions, or publication**: use **permutation importance** or **SHAP** (`shap.TreeExplainer` is fast for tree models).
> - When features are **highly correlated**: consider **grouped permutation importance** or cluster correlated features before measuring importance.

> **Note:** Importance ≠ causality
> Feature importance tells you what the **model uses**, not what **causes** the target. A feature can be highly important because it's a **proxy** for the true cause, or because of **leakage**. Always sanity-check important features against domain knowledge.
- - - 
### Why Trees Are Often Combined into Ensembles

A single tree has **high variance** — different training sets produce very different trees. To reduce variance, we combine many trees:

- **Bagging / Random Forest:** Train many trees on bootstrapped samples + random feature subsets. Average their predictions.
- **Boosting (AdaBoost, Gradient Boosting, XGBoost):** Train trees sequentially, each correcting the errors of the previous.

Ensembles drastically reduce overfitting and are some of the most powerful ML models. We'll cover them later.
