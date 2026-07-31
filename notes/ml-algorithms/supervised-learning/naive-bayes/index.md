---
layout: note
title: "Naive Bayes"
description: "Naive Bayes is a family of probabilistic classifiers based on Bayes' Theorem with a \"naive\" assumption: all features are conditionally independent given the class."
note: true
note_collection: "ML algorithms"
note_section: "Supervised Learning"
section_order: 2
note_order: 6
updated: 2026-06-06 11:30:50 -0700
keywords:
  - Supervised Learning
  - Probability
  - Evaluation
  - Inference
  - Graphs
math: true
mermaid: false
---
**Naive Bayes** is a family of probabilistic classifiers based on **Bayes' Theorem** with a "naive" assumption: **all features are conditionally independent given the class**.

It is:

- **Supervised** — for classification.
- **Generative** — models the joint distribution $P(X, y)$, not just $P(y \mid X)$.
- **Probabilistic** — outputs class probabilities.
- **Parametric** — has a fixed number of parameters per class.

## Core Idea

Given features $x = (x_1, x_2, \dots, x_n)$, compute the probability of each class and pick the most likely one:

$$\hat{y} = \arg\max_{c} P(y = c \mid x)$$

Using **Bayes' Theorem**, this becomes feasible to compute even when direct estimation of $P(y \mid x)$ is hard.

## Bayes' Theorem — The Foundation

$$P(y \mid x) = \frac{P(x \mid y) \cdot P(y)}{P(x)}$$

### Components

| Term | Name | Meaning |
|---|---|---|
| $P(y \mid x)$ | **Posterior** | Probability of class $y$ given features $x$ (**what we want**) |
| $P(x \mid y)$ | **Likelihood** | Probability of observing features $x$ if class is $y$ |
| $P(y)$ | **Prior** | Overall probability of class $y$ before seeing $x$ |
| $P(x)$ | **Evidence** | Overall probability of observing features $x$ |

> **Note:** Intuition
> **Posterior ∝ Likelihood × Prior.** Update your belief about the class after seeing features.

### Classic Example — Medical Test

- Prior: $P(\text{disease}) = 0.001$ (1 in 1000 have it).
- Likelihood: $P(+ \mid \text{disease}) = 0.99$ (99% sensitive).
- $P(+ \mid \text{no disease}) = 0.05$ (5% false positive rate).

What is $P(\text{disease} \mid +)$?

$$P(\text{disease} \mid +) = \frac{0.99 \times 0.001}{0.99 \times 0.001 + 0.05 \times 0.999} \approx 0.019$$

## The "Naive" Assumption — Conditional Independence

Computing $P(x_1, x_2, \dots, x_n \mid y)$ directly requires modeling all interactions between features for each class — **exponentially many parameters**. Infeasible.

### The Trick

Assume features are **conditionally independent given the class**:

$$P(x_1, x_2, \dots, x_n \mid y) = \prod_{i=1}^{n} P(x_i \mid y)$$

This factorizes the joint likelihood into a product of marginal likelihoods — **drastically reducing parameters from exponential to linear**.

### Is This Assumption Realistic?

**No, almost never.** Features in real data are usually correlated:

- In text: "machine" and "learning" co-occur — not independent.
- In medicine: blood pressure and weight are correlated.

### Why Does It Work Anyway?

> **Tip:** The classic Domingos & Pazzani (1997) result
> Despite violating its core assumption, Naive Bayes often performs surprisingly well. Reasons:
> - For classification, we only need the **right ranking** of class probabilities — not exact values.
> - Errors in modeling individual feature dependencies often **cancel out**.
> - With enough data, the model captures the **dominant signal**.
> - NB is **biased but has very low variance** — robust on small datasets.

## The Naive Bayes Classifier — The Full Formula

Combining Bayes' theorem with the independence assumption:

$$P(y \mid x_1, \dots, x_n) = \frac{P(y) \prod_{i=1}^{n} P(x_i \mid y)}{P(x_1, \dots, x_n)}$$

For classification, the denominator $P(x_1, \dots, x_n)$ is the **same for all classes**, so we ignore it:

$$\hat{y} = \arg\max_{c} P(y = c) \prod_{i=1}^{n} P(x_i \mid y = c)$$

### In Log Space (numerical stability)

$$\hat{y} = \arg\max_{c} \left[\log P(y = c) + \sum_{i=1}^{n} \log P(x_i \mid y = c)\right]$$

> **Warning:** Why log space?
> Multiplying many small probabilities causes **underflow** (numbers too small for the computer). Taking logs converts products to sums — numerically stable. **Always implement Naive Bayes in log space.**

## Variants of Naive Bayes

Different variants make different assumptions about the form of $P(x_i \mid y)$.

### Gaussian Naive Bayes

Assumes **continuous features** follow a Gaussian distribution within each class:

$$P(x_i \mid y = c) = \frac{1}{\sqrt{2\pi\sigma_{c,i}^2}}\exp\left(-\frac{(x_i - \mu_{c,i})^2}{2\sigma_{c,i}^2}\right)$$

**Training**: for each class $c$ and feature $i$, compute the mean $\mu_{c,i}$ and variance $\sigma_{c,i}^2$ of $x_i$ among samples of class $c$.

**Use case**: continuous features (height, weight, sensor readings).

### Multinomial Naive Bayes

Designed for **discrete count data** (e.g., word counts in documents):

$$P(x_i \mid y) = \frac{\text{count}(i, y) + \alpha}{\text{total count in } y + \alpha \cdot |V|}$$

Where $\alpha$ is the smoothing parameter and $|V|$ is the vocabulary size.

**Use case**: text classification (spam detection, topic classification, sentiment analysis). Models documents as a **bag of words** — count of each word matters, order doesn't.

### Bernoulli Naive Bayes

For **binary features** (feature is either present or absent):

$$P(x_i \mid y) = P(i \mid y)^{x_i} (1 - P(i \mid y))^{1 - x_i}$$

**Use case**: text classification when you care only about **whether** a word appears, not **how many times**.

### Categorical Naive Bayes

For features that take a **fixed number of discrete categories**. Each feature has a categorical distribution per class.

**Use case**: survey data, demographic features (color = red/green/blue).

### Complement Naive Bayes

Variant of Multinomial NB designed for **imbalanced text data**. Uses statistics from the **complement** (other classes) rather than the class itself for parameter estimation. Often outperforms standard Multinomial NB on imbalanced data.

## Laplace (Additive) Smoothing — The Zero-Probability Problem

### The Problem

Suppose the word **"blockchain"** never appears in spam emails in training. Then $P(\text{blockchain} \mid \text{spam}) = 0$. If a test email contains "blockchain", the entire product becomes:

$$P(\text{spam}) \cdot \prod P(\text{word}_i \mid \text{spam}) = P(\text{spam}) \cdot \ldots \cdot 0 \cdot \ldots = 0$$

**A single unseen feature kills the prediction** for that class.

### The Solution — Laplace Smoothing

Add a small constant $\alpha$ to all counts:

$$P(x_i \mid y) = \frac{\text{count}(x_i, y) + \alpha}{\text{count}(y) + \alpha \cdot |V|}$$

- $\alpha = 1$ → **Laplace smoothing** (add-one smoothing).
- $\alpha < 1$ → **Lidstone smoothing**.

### Effect

- Never assigns **zero probability** to unseen features.
- Slightly shrinks probabilities **toward uniform**.
- Tunable hyperparameter (default $\alpha = 1$ usually works well).

## Naive Bayes for Text Classification — The Killer Application

Naive Bayes has historically dominated text classification, especially spam filtering.

### Setup

- **Documents**: vectors of word counts (or binary indicators).
- **Vocabulary**: set of all words in training data.
- **Bag-of-words assumption**: word order doesn't matter; only counts do.

### Training (Multinomial NB)

1. Compute the **prior**: $P(y = \text{spam}) = \frac{\#\,\text{spam}}{\#\,\text{total}}$.
2. For each word $w$ and class $c$:
$$P(w \mid c) = \frac{\text{count}(w, c) + 1}{\text{total count in } c + |V|}$$

### Prediction

For a new document $d$ with words $w_1, \dots, w_n$:

$$\text{score}(c) = \log P(c) + \sum_{i=1}^{n} \log P(w_i \mid c)$$

Pick the class with the **highest score**.

### Why It Works So Well for Text

> **Check:** Strengths on text data
> - **High-dimensional sparse data** — works well where many models struggle.
> - **Fast training and prediction** — linear in size of data.
> - **Naturally handles many classes** — multi-class is trivial.
> - **Robust to irrelevant features** — they contribute roughly equally to all classes.
> - **Strong baseline** — often surprisingly hard to beat with more complex models.

**Used in production**: spam filters (Gmail's early system), document categorization, news classification, sentiment analysis baselines.

## Generative vs Discriminative Models

### Discriminative Models (e.g., Logistic Regression)

- Model $P(y \mid x)$ **directly**.
- Learn the **decision boundary** between classes.
- Don't model how data is generated.

### Generative Models (e.g., Naive Bayes)

- Model the **joint distribution** $P(x, y) = P(x \mid y) P(y)$.
- Can **generate new data** by sampling from the modeled distribution.
- Use Bayes' theorem to derive $P(y \mid x)$.

### Comparison

| Aspect | Generative (NB) | Discriminative (LR) |
|---|---|---|
| **Models** | $P(x, y)$ | $P(y \mid x)$ |
| **Training** | Estimate parameters of class-conditional distributions | Optimize $P(y \mid x)$ directly (e.g., via gradient descent) |
| **Data efficiency** | Better with **small data** | Better with **large data** |
| **Asymptotic accuracy** | Often **lower** | Often **higher** |
| **Can generate samples** | ✅ Yes | ❌ No |
| **Robust to missing features** | ✅ Yes | ❌ No |

> **Note:** Classic paper — Ng & Jordan (2002)
> Generative models **converge faster** but **plateau at higher error** than discriminative models. With **small data**, Naive Bayes beats Logistic Regression; with **large data**, LR wins.

## Strengths of Naive Bayes

> **Check:** Advantages
> - **Very fast** training and prediction — linear in features and samples.
> - **Scales to high dimensions** — works well even with millions of features.
> - **Handles many classes** naturally.
> - **Robust to irrelevant features.**
> - **Small data efficient** — needs less data than discriminative models.
> - **Probabilistic output** — gives class probabilities, not just predictions.
> - **Easy to implement** — simple counting + smoothing.
> - **Strong baseline** for text classification.
> - **Online learning friendly** — easy to update with new data.
> - **No hyperparameter tuning (mostly)** — defaults work well.

## Weaknesses of Naive Bayes

> **Warning:** Disadvantages
> - **Independence assumption is unrealistic** — correlated features hurt accuracy.
> - **Probability estimates are poorly calibrated** — predicted probabilities can be extreme (close to 0 or 1).
> - **Sensitive to data quality** — needs good feature engineering.
> - **Performance plateaus** — discriminative models eventually outperform.
> - **Zero-frequency problem** — must use smoothing.
> - **Numerical instability** without log-space computation.
> - **Continuous features require distributional assumptions** (Gaussian NB assumes normality).
> - **Class imbalance** can bias the prior heavily.
> - **Categorical features with many levels** can be problematic.

- - - 
# Likelihood for Continuous Variables — Deep Dive

This exposes a subtle but crucial concept that often trips people up. The trick is that **continuous variables don't have probabilities at single points**, so "likelihood" works fundamentally differently from discrete variables.

## The Core Problem — Why It's Tricky

For **discrete variables** (e.g., word counts), likelihood is straightforward:

$$P(X = x \mid y) = \text{count or probability of that exact value}$$

You can literally count how often $x$ appears for class $y$ in training data.

For **continuous variables** (e.g., height = 175.234 cm), the **probability of any exact value is zero**:

$$P(X = 175.234) = 0$$

### Why?

A continuous variable can take **infinitely many possible values** within any range. This is a fundamental property of continuous distributions — they're measured by **densities over intervals**, not probabilities at points.

> **Important:** So how do we use $P(x \mid y)$ in Naive Bayes if it's always zero?
> We don't use probabilities — we use **probability densities**.

## Probability Mass vs Probability Density

### Probability Mass Function (PMF) — Discrete

$$P(X = x) = \text{actual probability at } x$$

- **Sums to 1** over all possible values.
- $P(X = x) \in [0, 1]$.
- Direct interpretation as probability.

### Probability Density Function (PDF) — Continuous

$$f(x) = \text{density at } x$$

- **Integrates to 1** over all possible values.
- $f(x) \geq 0$ but **can be greater than 1** (it's a density, not a probability).
- Probability is the **area under the curve** over an interval:

$$P(a \leq X \leq b) = \int_a^b f(x) \, dx$$

For continuous variables, a **single point has zero width**, so the probability there is zero. An **interval has non-zero width**, so the probability is non-zero.

## The Trick — Use Density as "Likelihood"

In probabilistic models with continuous features, we use the **PDF value as a proxy for likelihood** — even though it's technically a density, not a probability:

$$\text{Likelihood}(x \mid y) \approx f(x \mid y)$$

### Why This Works

For classification, we compare likelihoods **across classes**:

$$\hat{y} = \arg\max_c P(y = c) \cdot f(x \mid y = c)$$

We're taking **ratios and comparisons** — the absolute value of the density doesn't matter, only the relative ordering. The "infinitesimal width" cancels out across classes.

### The Approximation

For a small region $\Delta x$ around point $x$:

$$P(x - \Delta x/2 \leq X \leq x + \Delta x/2 \mid y) \approx f(x \mid y) \cdot \Delta x$$

In classification:

$$\hat{y} = \arg\max_c P(c) \cdot f(x \mid c) \cdot \Delta x = \arg\max_c P(c) \cdot f(x \mid c)$$

> **Note:** The $\Delta x$ cancels
> $\Delta x$ is the same across all classes — it cancels out. So we can ignore the subtlety and just use $f(x \mid y)$ directly.

## The Most Common Approach — Gaussian Likelihood

The dominant method assumes each continuous feature follows a **Gaussian (Normal) distribution** within each class.

### Gaussian PDF

$$f(x \mid y = c) = \frac{1}{\sqrt{2\pi\sigma_c^2}} \exp\left(-\frac{(x - \mu_c)^2}{2\sigma_c^2}\right)$$

Where:
- $\mu_c$ = mean of feature in class $c$
- $\sigma_c^2$ = variance of feature in class $c$

### Training

For each class $c$ and each feature $i$:

1. **Filter** training data to samples of class $c$.
2. **Compute the mean**: $\mu_{c,i} = \frac{1}{n_c}\sum_{j \in c} x_{j,i}$
3. **Compute the variance**: $\sigma_{c,i}^2 = \frac{1}{n_c}\sum_{j \in c} (x_{j,i} - \mu_{c,i})^2$

### Prediction

For a new sample $x = (x_1, x_2, \dots, x_n)$:

$$\hat{y} = \arg\max_c P(c) \prod_{i=1}^n f(x_i \mid y = c)$$

Where $f(x_i \mid y = c)$ is the Gaussian density evaluated at $x_i$ with the trained $\mu_{c,i}, \sigma_{c,i}^2$.

### Concrete Example

Suppose feature = "height in cm" and we're classifying gender.

- **Males**: $\mu_M = 175,\ \sigma_M = 7$
- **Females**: $\mu_F = 162,\ \sigma_F = 6$

A new person has height 170 cm. Compute:

$$f(170 \mid M) = \frac{1}{\sqrt{2\pi \cdot 49}}\exp\left(-\frac{(170-175)^2}{2 \cdot 49}\right) \approx 0.050$$

$$f(170 \mid F) = \frac{1}{\sqrt{2\pi \cdot 36}}\exp\left(-\frac{(170-162)^2}{2 \cdot 36}\right) \approx 0.028$$

Combined with equal priors, 170 cm is **more "likely" under the male distribution** → classify as male.

> **Warning:** These aren't probabilities
> The values 0.050 and 0.028 are **densities, not probabilities**. But for comparing classes, that's all we need.

## Why Gaussian? — Justification

> **Check:** Why Gaussian is the popular default
> - **Central Limit Theorem** — many real-world features arise from sums of small independent effects, leading to approximately Gaussian distributions (heights, weights, measurement errors).
> - **Only two parameters** — mean and variance, easy to estimate.
> - **Closed-form PDF** — fast to compute.
> - **Well-understood mathematics** — calculus, optimization, conjugate priors all work cleanly.
> - **Maximum entropy** for fixed mean and variance — the "least committal" continuous distribution.

But this is a **strong assumption**. If features aren't Gaussian, accuracy suffers.

## What If Features Aren't Gaussian?

### (A) Use a Different Parametric Distribution

Pick a distribution that better matches your data:

| Distribution | When to use |
|---|---|
| **Exponential** | Waiting times, positive skewed data |
| **Log-normal** | When $\log(x)$ is Gaussian (income, file sizes) |
| **Gamma / Beta** | Positive bounded continuous data |
| **Student's t** | Heavy-tailed data |

Process is the same — estimate parameters per class, use PDF value as likelihood.

### (B) Transform the Feature

Make the data more Gaussian-like:

- **Log transform** — for right-skewed data.
- **Box-Cox transform** — finds optimal power transformation.
- **Quantile transform** — maps any distribution to Gaussian via quantiles.

After transformation, Gaussian NB works well.

### (C) Discretize the Feature

Bin the continuous feature into discrete intervals, then use Multinomial or Categorical NB:

- **Equal-width bins** — divide range into equal intervals.
- **Equal-frequency bins** (quantile binning) — each bin has the same number of samples.
- **Decision-tree-based binning** — optimal split points learned from data.

Loses information but avoids distributional assumptions.

### (D) Kernel Density Estimation (KDE)

**Non-parametric** — estimate the density empirically without assuming any distribution:

$$f(x \mid y) = \frac{1}{n_y \cdot h}\sum_{i=1}^{n_y} K\left(\frac{x - x_i}{h}\right)$$

Where:
- $K$ is a kernel function (typically Gaussian).
- $h$ is the bandwidth (smoothing parameter).
- Sum is over all training samples in class $y$.

> **Check:** KDE pros
> - **Flexible**, no distributional assumption.

> **Warning:** KDE cons
> - **Slow** (must evaluate over all training points).
> - Needs **bandwidth tuning**.
> - **Suffers in high dimensions.**

Used in **Kernel Naive Bayes**, available in some libraries.

### (E) Mixture Models

Model each class's feature distribution as a **mixture of Gaussians** (or other distributions):

$$f(x \mid y) = \sum_{k=1}^K \pi_{y,k} \cdot \mathcal{N}(x \mid \mu_{y,k}, \sigma_{y,k}^2)$$

Captures **multimodal distributions**. Fit via the **EM algorithm**. More flexible than a single Gaussian.

## The Same Logic Applies in Other Models

The "density as likelihood" idea isn't unique to Naive Bayes — it's how continuous likelihoods work throughout ML.

| Model | How density-as-likelihood appears |
|---|---|
| **Linear Regression (MLE)** | Gaussian noise $\epsilon \sim \mathcal{N}(0, \sigma^2)$; likelihood is the Gaussian PDF evaluated at $y - x^T\beta$ |
| **Gaussian Mixture Models (GMM)** | Likelihood is a weighted sum of Gaussian PDFs |
| **Variational Autoencoders (VAEs)** | Gaussian likelihoods model reconstruction loss |
| **Bayesian inference** | Posterior involves continuous likelihoods |
| **Hidden Markov Models** (continuous observations) | Gaussian emission probabilities |

> **Tip:** The big takeaway
> Wherever you have **continuous data and a probabilistic model**, "likelihood" means **PDF value** — not probability.

## Mathematical Subtlety — Is It Really "Likelihood"?

Technically, in statistics:

- **Likelihood** refers to $P(\text{data} \mid \theta)$ viewed as a function of **parameters $\theta$ for fixed data**.
- For **discrete** data, it's a probability.
- For **continuous** data, it's a density.

The term "likelihood" is deliberately used in **both cases** — the meaning is *"how plausible is this data under this model?"* — but the underlying mathematical object differs.

### Why It's OK

Likelihoods aren't probabilities in the strict sense; they're **comparative measures**. We compare likelihoods across hypotheses (or classes), so the **absolute scale doesn't matter** — only ratios do.

### Maximum Likelihood Estimation (MLE)

Whether discrete or continuous, MLE works the same way:

$$\hat{\theta} = \arg\max_\theta L(\theta \mid \text{data}) = \arg\max_\theta \prod_i f(x_i \mid \theta)$$

For continuous data, this $f$ is a **density**. We still maximize it. The mathematical machinery (gradients, optimization) is **identical**.

## Common Pitfalls

> **Warning:** Watch out for
> - **Confusing density with probability** — densities can be > 1, probabilities can't.
> - **Forgetting the Gaussian assumption** in Gaussian NB — check if your data is actually Gaussian.
> - **Treating likelihood values as probabilities** — they're not, but they work for comparison.
> - **Numerical instability** — densities of multivariate Gaussians can be astronomically small or large; use **log-densities** in computation.
> - **Variance estimation issues** — if $\sigma^2 \to 0$ (feature is constant in a class), the Gaussian PDF becomes degenerate. Add small regularization: $\sigma^2 \to \sigma^2 + \epsilon$.
> - **High dimensions** — products of many densities can underflow; **always work in log space**.

## Quick Summary

| Type | Likelihood is… |
|---|---|
| **Discrete features** | **Probability mass** (count-based, sums to 1) |
| **Continuous features** | **Probability density** value (PDF evaluation) |

The mechanics of using likelihood in classification (and Bayes' rule) are **identical in both cases** — multiply prior by likelihood, normalize, pick the max. The underlying math is just slightly different.

**Most common method for continuous data**: assume Gaussian → estimate $\mu, \sigma^2$ per class per feature → evaluate Gaussian PDF as likelihood.

**Alternatives when Gaussian doesn't fit**: other parametric distributions, feature transformation, discretization, KDE, or mixture models.
