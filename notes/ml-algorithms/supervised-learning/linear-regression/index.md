---
layout: note
title: "Linear Regression"
description: "Linear regression is a supervised learning algorithm used for regression problems (predicting continuous values). It models the relationship between a dependent variable y and o…"
note: true
note_collection: "ML algorithms"
note_section: "Supervised Learning"
section_order: 2
note_order: 4
updated: 2026-06-06 11:30:50 -0700
keywords:
  - Supervised Learning
  - Linear Models
  - Probability
  - Training
  - Evaluation
math: true
mermaid: false
---
### What is Linear Regression

Linear regression is a supervised learning algorithm used for regression problems (predicting continuous values). It models the relationship between a dependent variable y and one or more **independent** variables X by fitting a linear equation to observed data.

$$y = \beta_0 + \beta_1 x + \epsilon$$
Multiple linear regression (n features):

$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n$$
Vector/Matrix Form:

$$\hat{y} = X \beta$$
The goal is to learn $\beta$ that best fits the data.

### The hypothesis function

The model’s prediction for a single example is:
$$h_{\beta}(x) = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n$$
### The loss function

For a single training example, the squared error is $L^{(i)} = (y^{(i)} - \hat{y}^{(i)})^2$ . For the whole dataset the cost function is $$J(\beta) = \frac{1}{2m} \sum_{i=1}^{m} (h_{\beta}(x^{(i)}) - y^{(i)})^2$$
**Notes**

1. $\frac{1}{2}$ is just a convention to cancel out the square when derivative is taken.
2. $\frac{1}{m}$ averages out over the entire dataset making the gradient independent of the dataset size.
3. Why Squared Error
	1. Differentiable everywhere, unlike absolute error (derivative doesn’t exist at 0, when prediction is equal to truth)
	2. Convex function - One Global Minima, no local minimum
	3. **Maximum Likelihood Justification**: Under gaussian noise assumption, minimizing MSE = MLE

### Solving Linear Regression - Two Approaches

#### Normal Equation (Closed Form Solution)

We want to minimize $J(\beta)$. Set the gradient to zero and solve analytically.

Cost function in the matrix form:

$$ J(\beta) = \frac{1}{2m} (X\beta - y) ^ T (X\beta - y)$$
Expand:

$$ J(\beta) = \frac{1}{2m} (\beta^T X^T X \beta - 2 \beta^T X^T y + y^T y)$$
Take gradient w.r.t $\beta$

$$\nabla_{\beta} J = \frac{1}{m} (X^T X \beta - X^T y)$$
Set to zero:

$$ \beta = (X^T X)^{-1} X^T y$$
This is the **Normal Equation**

#### Gradient Descent

It is an iterative optimization algorithm where you start with random $\beta$ and then repeatedly move in direction of the steepest descent.

The core update rule,
$$\beta_j = \beta_j - \alpha \frac{d J(\beta)}{d\beta_j}$$
##### Deriving the Gradient:

$$J(\beta) = \frac{1}{2m} \sum_{i=1}^{m} (h_{\beta}(x^{(i)}) - y^{(i)})^2$$
Take partial derivate w.r.t. $\beta_j$

$$\frac{d J(\beta)}{d\beta_j} = \frac{1}{m} \sum_{i=1}^{m} (h_{\beta}(x^i) - y^i). x_j^i$$
So the update rule becomes,
$$\beta_j = \beta_j - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_{\beta}(x^i) - y^i). x_j^i$$
In vectorized form it is,
$$\beta = \beta - \alpha \frac{1}{m} X^T(X\beta - y)$$
**Algorithm Steps**

1. Initialize $\beta$ to random values or all zeroes
2. Compute predictions $\hat{y} = X \beta$
3. Compute the gradient
4. Update $\beta$ using the rule above
5. Repeat until convergence (cost stops decreasing significantly or max iterations reached)

**Variants of Gradient Descent**

| Variant                               | Examples per Update                                  | Update Rule                                                       | ✅ Pros                                                                                        | ❌ Cons                                                                             |
| ------------------------------------- | ---------------------------------------------------- | ----------------------------------------------------------------- | --------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| **Batch Gradient Descent (BGD)**      | All $m$ examples (one epoch = one update)            | $\beta := \beta - \alpha \cdot \frac{1}{m} X^T(X\beta - y)$       | Stable, smooth convergence; deterministic                                                     | Slow on large datasets; doesn't fit if data exceeds memory                         |
| **Stochastic Gradient Descent (SGD)** | One randomly chosen example                          | $\beta := \beta - \alpha (h_\beta(x^{(i)}) - y^{(i)}) x^{(i)}$    | Very fast updates; works for online learning; can escape shallow local minima                 | Noisy convergence (zig-zags); never quite settles; loss bounces around the minimum |
| **Mini-Batch Gradient Descent**       | A batch of $b$ examples (typically 32, 64, 128, 256) | $\beta := \beta - \alpha \cdot \frac{1}{b} X_b^T(X_b\beta - y_b)$ | Balances speed and stability; leverages GPU vectorization; smoother than SGD, faster than BGD | Requires tuning batch size; small noise still present                              |
 
 **Real-world examples**

- **Linear regression on 5,000 housing prices** → Batch GD. Closed-form solution is even better, but batch GD is fine.
- **Training ResNet on ImageNet (1.2M images)** → Mini-batch, typically 256 across multiple GPUs.
- **Fine-tuning a language model** → Mini-batch, often 8–64 depending on model size and GPU memory.
- **Recommendation system updating from live user clicks** → SGD or small mini-batches, since data is streaming.
- **Logistic regression for a Kaggle tabular dataset (50K rows)** → Mini-batch with b=128b = 128 b=128, or just use scikit-learn's L-BFGS.

**Notes**

1. Feature scaling matters for gradient descent
2. Learning rate choosing is critical, small one - slow learning, large one - overshoot the minimum or oscillate
3. Learning rate scheduler may help for large number of iterations.

**Assumptions of Linear Regression**

1. Relationship between features and target is linear
2. Each observation’s error is independent of others
3. Homoscedasticity: The variance of the residuals should be the same across all values of X. If variance increases or decreases, it's called **heteroscedasticity**. 
4. Residuals should be normally distributed
5. Features should not be highly correlated with others

### Probabilistic Interpretation — Why MSE = MLE

**Assumption**

We assume the data is generated by a linear model with Gaussian noise:

$$y^{(i)} = \beta^T x^{(i)} + \epsilon^{(i)}, \quad \epsilon^{(i)} \sim \mathcal{N}(0, \sigma^2) \text{ i.i.d.}$$

This implies the conditional distribution of $y^{(i)}$ given $x^{(i)}$:

$$y^{(i)} \mid x^{(i)};\, \beta \sim \mathcal{N}(\beta^T x^{(i)},\, \sigma^2)$$

**Likelihood of the dataset**

$$L(\beta) = \prod_{i=1}^{m} \frac{1}{\sqrt{2\pi}\,\sigma} \exp\left(-\frac{(y^{(i)} - \beta^T x^{(i)})^2}{2\sigma^2}\right)$$

**Log-likelihood**

Taking the log turns the product into a sum:

$$\log L(\beta) = -m \log(\sqrt{2\pi}\,\sigma) \;-\; \frac{1}{2\sigma^2} \sum_{i=1}^{m} \left(y^{(i)} - \beta^T x^{(i)}\right)^2$$

**Key insight**

> **Important:** OLS = MLE under Gaussian noise
> Maximizing $\log L(\beta)$ over $\beta$ is equivalent to **minimizing**:
> $$\sum_{i=1}^{m} \left(y^{(i)} - \beta^T x^{(i)}\right)^2$$
> which is exactly the **sum of squared errors**.

The first term $-m\log(\sqrt{2\pi}\,\sigma)$ doesn't depend on $\beta$, so it drops out. The $\frac{1}{2\sigma^2}$ factor is a positive constant, so maximizing the negative of the sum is the same as minimizing the sum itself.

**Why this matters**

- **MSE isn't arbitrary** — it falls out naturally from assuming Gaussian noise.
- This is the **deep probabilistic justification** for squared error loss.
- Changing the noise assumption changes the loss:
	- Laplace noise → **L1 / absolute error**
	- Bernoulli (for classification) → **cross-entropy / log loss**
	- Poisson → **Poisson deviance**

### Evaluation Metrics for Regression

**(a) Mean Squared Error (MSE)**

$$\text{MSE} = \frac{1}{m}\sum_{i=1}^{m}(y_i - \hat{y}_i)^2$$

- Units: **squared** (e.g., dollars²).
- Penalizes **large errors heavily** due to the squaring.
- Same quantity that's minimized in OLS — convenient for optimization but harder to interpret.

**(b) Root Mean Squared Error (RMSE)**

$$\text{RMSE} = \sqrt{\text{MSE}}$$

- Same **units as the target** variable.
- More interpretable than MSE.
- Still penalizes large errors more than small ones.

**(c) Mean Absolute Error (MAE)**

$$\text{MAE} = \frac{1}{m}\sum_{i=1}^{m}|y_i - \hat{y}_i|$$

- **Robust to outliers**; less influenced by extreme values.
- Treats all errors linearly (a 10-unit error is 10× a 1-unit error, not 100×).
- Not differentiable at zero — slightly harder to optimize.

**(d) R² (Coefficient of Determination)**

$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$

- **Proportion of variance** in $y$ explained by the model.
- Ranges from $-\infty$ to $1$ (negative if the model is worse than predicting the mean).
- $R^2 = 0.85$ → "Model explains **85% of the variance**."

> **Warning:** Issue
> $R^2$ **always increases** when you add more features, even useless ones. Don't use raw $R^2$ to compare models with different numbers of features.

**(e) Adjusted R²**

$$\text{Adj } R^2 = 1 - \frac{(1-R^2)(m-1)}{m - n - 1}$$

Where $m$ = number of samples, $n$ = number of features.

- **Penalizes** the addition of useless features.
- Use this for **model comparison** when feature counts differ.
- Can decrease if a new feature doesn't improve the model enough to justify the added complexity.

**(f) Mean Absolute Percentage Error (MAPE)**

$$\text{MAPE} = \frac{100\%}{m}\sum_{i=1}^{m}\left|\frac{y_i - \hat{y}_i}{y_i}\right|$$

- Easy to **interpret as a percentage**.
- **Fails when actual values are zero** (division by zero).
- Asymmetric — penalizes over-predictions and under-predictions differently.

---

### Regularization

When linear regression **overfits**, add a penalty term to the cost function to constrain the coefficients.

**Ridge Regression (L2 Regularization)**

$$J(\beta) = \frac{1}{2m}\sum_{i=1}^{m}(y_i - \hat{y}_i)^2 + \lambda\sum_{j=1}^{n}\beta_j^2$$

- Shrinks coefficients **toward zero** but **never exactly to zero**.
- Helps with **multicollinearity** (correlated features).
- **Closed-form solution**: $\beta = (X^TX + \lambda I)^{-1}X^Ty$.
- The $\lambda I$ term makes $X^TX$ invertible even when features are collinear.

**Lasso Regression (L1 Regularization)**

$$J(\beta) = \frac{1}{2m}\sum_{i=1}^{m}(y_i - \hat{y}_i)^2 + \lambda\sum_{j=1}^{n}|\beta_j|$$

- Shrinks some coefficients to **exactly zero** → performs **feature selection**.
- **No closed-form solution** — use coordinate descent or subgradient methods.
- Useful when you suspect only a few features are truly relevant.

**Elastic Net**

$$J(\beta) = \frac{1}{2m}\sum_{i=1}^{m}(y_i - \hat{y}_i)^2 + \lambda_1\sum_{j=1}^{n}|\beta_j| + \lambda_2\sum_{j=1}^{n}\beta_j^2$$

- **Combines L1 and L2** penalties.
- Useful when you have **many correlated features** — Lasso alone arbitrarily picks one from each correlated group; Elastic Net keeps them together.

> **Tip:** Choosing λ
> $\lambda$ (regularization strength) is a **hyperparameter** — tune with **cross-validation**. Larger $\lambda$ = more shrinkage = simpler model but more bias.
