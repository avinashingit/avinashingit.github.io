---
layout: note
title: "Logistic Regression"
description: "Logistic regression is a supervised learning algorithm used for binary classification (and extendable to multi-class). It predicts the probability that an input belongs to a par…"
note: true
note_collection: "ML algorithms"
note_section: "Supervised Learning"
section_order: 2
note_order: 5
updated: 2026-06-06 11:30:50 -0700
keywords:
  - Supervised Learning
  - Linear Models
  - Probability
  - Deep Learning
  - Evaluation
math: true
mermaid: false
---
### What is Logistic Regression?

**Logistic regression** is a **supervised learning algorithm** used for **binary classification** (and extendable to multi-class). It predicts the **probability** that an input belongs to a particular class.

It outputs a probability between 0 and 1, and we classify based on a threshold (usually 0.5).

#### Why Not Use Linear Regression for Classification?

Naively, you might try linear regression on a 0/1 target. But:

1. **Predictions aren't bounded** — linear regression can output -3.5 or +47, which makes no sense as a probability.
2. **Sensitive to outliers** — a single extreme point can shift the decision boundary dramatically.
3. **Assumes linear relationship with target** — but probabilities should saturate (can't exceed 1 or go below 0).
4. **Violates regression assumptions** — residuals aren't normal or homoscedastic with binary targets.

We need a function that maps any real number to (0, 1). Enter the **sigmoid**.

- - -
### The Sigmoid Function

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

This is a monotonic function, always increasing and is bounded (0, 1). It also has a beautiful derivative $\sigma’ (z) = \sigma (z) (1 - \sigma (z))$ 

- - -
### The Logistic Regression Model

Given features $x$ and parameters $\beta$:

**Step 1: Linear combination**

$$z = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n = \beta^T x$$

This is the same linear score as in linear regression — a weighted sum of features.

**Step 2: Apply sigmoid to get a probability**

$$P(y = 1 \mid x; \beta) = h_\beta(x) = \sigma(\beta^T x) = \frac{1}{1 + e^{-\beta^T x}}$$

The sigmoid squashes $z \in (-\infty, \infty)$ into $(0, 1)$, so the output can be interpreted as a probability.

**Step 3: Complementary probability**

Since the two outcomes must sum to 1:

$$P(y = 0 \mid x; \beta) = 1 - h_\beta(x)$$

**The Compact Bernoulli Form**

The two cases can be combined into a single expression:

$$P(y \mid x; \beta) = h_\beta(x)^y \cdot \bigl(1 - h_\beta(x)\bigr)^{1-y}$$

This works because the exponents act as **switches**:

- When $y = 1$: $\;P = h_\beta(x)^1 \cdot (1 - h_\beta(x))^0 = h_\beta(x)$
- When $y = 0$: $\;P = h_\beta(x)^0 \cdot (1 - h_\beta(x))^1 = 1 - h_\beta(x)$

> **Note:** Why this compact form matters
> Writing the Bernoulli PMF this way lets us take a **product over all training examples** and then a **log** to get the log-likelihood as a clean sum. That sum, negated, is the **binary cross-entropy loss** — the standard objective for logistic regression.

We can’t use MSE for logistic regression as the sigmoid function in the output makes the function non-convex and make it have many local minima. Also vanishing gradients could be a problem.

- - - 
### The Loss Function - Binary Cross Entropy Loss

**Deriving the Loss via Maximum Likelihood**

We assume each $y^{(i)}$ is a **Bernoulli random variable** with success probability $h_\beta(x^{(i)})$:

$$y^{(i)} \sim \text{Bernoulli}\bigl(h_\beta(x^{(i)})\bigr)$$

**Likelihood of one observation**

$$P(y^{(i)} \mid x^{(i)}; \beta) = h_\beta(x^{(i)})^{y^{(i)}} \bigl(1 - h_\beta(x^{(i)})\bigr)^{1 - y^{(i)}}$$

**Likelihood of the whole dataset** (assuming independence across examples):

$$L(\beta) = \prod_{i=1}^{m} h_\beta(x^{(i)})^{y^{(i)}} \bigl(1 - h_\beta(x^{(i)})\bigr)^{1 - y^{(i)}}$$

**Take the log** (turns the product into a sum):

$$\log L(\beta) = \sum_{i=1}^{m} \left[y^{(i)} \log h_\beta(x^{(i)}) + (1 - y^{(i)}) \log\bigl(1 - h_\beta(x^{(i)})\bigr)\right]$$

**From maximization to minimization**

We *maximize* the log-likelihood, which is equivalent to *minimizing* the **negative** log-likelihood. Dividing by $m$ for averaging:

$$\boxed{\,J(\beta) = -\frac{1}{m}\sum_{i=1}^{m}\left[y^{(i)} \log h_\beta(x^{(i)}) + (1 - y^{(i)}) \log\bigl(1 - h_\beta(x^{(i)})\bigr)\right]\,}$$

This is the **Binary Cross-Entropy Loss**, also called **Log Loss**.

**Understanding It Intuitively**

For a single example (with $\hat{y} = h_\beta(x)$):

$$\text{Loss} = -\bigl[y \log(\hat{y}) + (1 - y) \log(1 - \hat{y})\bigr]$$

The two terms act as switches based on the true label:

- **When $y = 1$**: loss $= -\log(\hat{y})$
	- If $\hat{y} \to 1$: loss $\to 0$ ✅ (confident and correct)
	- If $\hat{y} \to 0$: loss $\to \infty$ ❌ (confident and wrong → huge penalty)
- **When $y = 0$**: loss $= -\log(1 - \hat{y})$
	- If $\hat{y} \to 0$: loss $\to 0$ ✅
	- If $\hat{y} \to 1$: loss $\to \infty$ ❌

> **Note:** Why log loss is the "right" loss for classification
> Just as MSE falls out of MLE under **Gaussian** noise (for regression), binary cross-entropy falls out of MLE under a **Bernoulli** likelihood (for binary classification). The loss isn't arbitrary — it's the **negative log-likelihood of the data under the assumed probability model**.

- - - 
### Gradient Descent

**Gradient Descent for Logistic Regression**

We need to compute $\frac{\partial J}{\partial \beta_j}$ to run gradient descent.

**Deriving the Gradient**

Start with the cost:

$$J(\beta) = -\frac{1}{m}\sum_{i=1}^{m}\left[y^{(i)} \log \sigma(z^{(i)}) + (1 - y^{(i)}) \log\bigl(1 - \sigma(z^{(i)})\bigr)\right]$$

where $z^{(i)} = \beta^T x^{(i)}$.

Applying the chain rule, with the key sigmoid identity:

$$\sigma'(z) = \sigma(z)\bigl(1 - \sigma(z)\bigr)$$

$$\boxed{\,\frac{\partial J}{\partial \beta_j} = \frac{1}{m}\sum_{i=1}^{m}\bigl(h_\beta(x^{(i)}) - y^{(i)}\bigr)\, x_j^{(i)}\,}$$

**🤯 The Beautiful Result**

This is **exactly the same form** as the linear regression gradient!

**Update rule (scalar form)**

$$\beta_j := \beta_j - \alpha \cdot \frac{1}{m}\sum_{i=1}^{m}\bigl(h_\beta(x^{(i)}) - y^{(i)}\bigr)\, x_j^{(i)}$$

**Update rule (vectorized form)**

$$\beta := \beta - \frac{\alpha}{m} X^T \bigl(h_\beta(X) - y\bigr)$$

**The only difference from linear regression**

The hypothesis function:

|  | Linear Regression | Logistic Regression |
|---|---|---|
| $h_\beta(x)$ | $\beta^T x$ | $\sigma(\beta^T x) = \dfrac{1}{1 + e^{-\beta^T x}}$ |
| Output range | $(-\infty, \infty)$ | $(0, 1)$ |
| Interpretation | Predicted value | Predicted probability |

> **Note:** Why the gradients look identical
> Both losses arise as **negative log-likelihoods** of exponential-family distributions (Gaussian for linear regression, Bernoulli for logistic regression). For any **generalized linear model (GLM)** with a canonical link function, the gradient takes the form **(prediction − target) × feature**. The cancellation isn't coincidence — it's a structural property of GLMs.

> **Tip:** Practical implication
> You can reuse the **exact same gradient descent code** for both models. Just swap the hypothesis function from $\beta^T x$ to $\sigma(\beta^T x)$ — everything else (loop, update, vectorization) stays identical.

- - - 
### Why Logistic Regression

Logistic Regression models the log-odds of the positive class as a linear function of the parameters.

Odds is the ratio of probability of an event to its complement.

$$odds = \frac{P(y=1)}{P(y=0)} = \frac{p}{1-p}$$
Log-odds (Logit):
$$ logit(p) = log(\frac{p}{1-p}) $$
If $p=\sigma(\beta^T x) = \frac{1}{1 + e^{-\beta^Tx}}$, then $\log(\frac{p}{1-p}) = \beta^Tx)$. 
- - - 
### Multi Class Classification

Two main approaches for extending logistic regression to $K > 2$ classes.

**(a) One-vs-Rest (OvR / One-vs-All)**

For $K$ classes, train $K$ **binary classifiers**, each separating one class from all others. Predict the class with the **highest probability**.

$$\hat{y} = \arg\max_{k \in \{1, \dots, K\}} h_{\beta_k}(x)$$

- ✅ **Simple** — works with any binary classifier (logistic regression, SVM, etc.).
- ✅ Easy to parallelize (each binary problem is independent).
- ❌ Probabilities **don't naturally sum to 1** — they're computed independently.
- ❌ Can produce **ambiguous predictions** when multiple classifiers fire confidently.
- ❌ **Class imbalance**: each binary problem is imbalanced by construction (1 class vs $K-1$).

**(b) Softmax Regression (Multinomial Logistic Regression)**

Generalize the sigmoid to the **softmax** function. For each class $k$:

$$P(y = k \mid x) = \frac{e^{\beta_k^T x}}{\sum_{j=1}^{K} e^{\beta_j^T x}}$$

Each class gets its own weight vector $\beta_k$. The denominator normalizes across all classes so the outputs form a **proper probability distribution**.

- ✅ Output is a **proper probability distribution** (non-negative, sums to 1).
- ✅ Trained **jointly** — classes compete with each other directly.
- ✅ Reduces to **standard logistic regression** when $K = 2$.
- ❌ Slightly more complex to implement than OvR.

**Loss: Categorical Cross-Entropy**

The binary cross-entropy generalizes to the multi-class case:

$$J(\beta) = -\frac{1}{m}\sum_{i=1}^{m}\sum_{k=1}^{K} \mathbb{1}\{y^{(i)} = k\} \, \log P(y^{(i)} = k \mid x^{(i)})$$

The indicator $\mathbb{1}\{y^{(i)} = k\}$ acts as a **one-hot selector** — only the log-probability of the true class contributes to the loss for each example. Effectively:

$$J(\beta) = -\frac{1}{m}\sum_{i=1}^{m} \log P(y^{(i)} = y_{\text{true}}^{(i)} \mid x^{(i)})$$

**Comparison at a glance**

| Property | One-vs-Rest | Softmax |
|---|---|---|
| Number of models | $K$ binary | 1 joint model |
| Outputs sum to 1? | ❌ No | ✅ Yes |
| Classes compete? | ❌ No | ✅ Yes |
| Reduces to logistic regression for $K=2$? | ⚠️ Effectively | ✅ Exactly |
| Common in deep learning? | ❌ Rarely | ✅ Standard |

> **Note:** Where you'll see softmax
> Softmax is the **standard final layer** in neural network classifiers — from MNIST digit classification to ImageNet to language model token prediction. The "logits" you hear about in deep learning are exactly the $\beta_k^T x$ values **before** the softmax is applied.

> **Tip:** Numerical stability
> Computing $e^{\beta_k^T x}$ directly can overflow for large logits. The standard fix is to subtract the max logit before exponentiating:
> $$P(y = k \mid x) = \frac{e^{\beta_k^T x - c}}{\sum_j e^{\beta_j^T x - c}}, \quad c = \max_j \beta_j^T x$$
> The result is mathematically identical but numerically safe. Every deep learning library does this internally.

- - - 
**Notes**

1. Regularization is similar to linear regression
2. Multi Class Classification can be done as one vs all or softmax regression

- - - 
### Evaluation

**Evaluation Metrics for Classification**

Accuracy alone is **misleading**, especially with imbalanced data. You need the full toolkit.

**(a) Confusion Matrix**

|  | Predicted 0 | Predicted 1 |
|---|---|---|
| **Actual 0** | TN (True Negative) | FP (False Positive) |
| **Actual 1** | FN (False Negative) | TP (True Positive) |

Everything else is derived from these four counts.

**(b) Core Metrics**

**Accuracy**

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

Fraction correct. **Misleading on imbalanced data** — a model predicting "not fraud" for every transaction gets 99% accuracy on a 1% fraud dataset.

**Precision**

$$\text{Precision} = \frac{TP}{TP + FP}$$

Of those **predicted positive**, how many are actually positive? → *"Quality of positive predictions."*

**Recall (Sensitivity, TPR)**

$$\text{Recall} = \frac{TP}{TP + FN}$$

Of all **actual positives**, how many did we catch? → *"Coverage of positives."*

**F1 Score**

$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

**Harmonic mean** of precision and recall — penalizes extreme imbalance between the two (a model with precision 1.0 and recall 0.01 has $F_1 \approx 0.02$, not 0.5).

**Specificity (TNR)**

$$\text{Specificity} = \frac{TN}{TN + FP}$$

Of all **actual negatives**, how many were correctly rejected?

**(c) ROC Curve & AUC**

- **ROC curve**: plot **TPR vs FPR** at various classification thresholds.
- **AUC**: area under the ROC curve.
	- AUC = 1.0 → perfect classifier
	- AUC = 0.5 → random guessing
	- AUC < 0.5 → worse than random (flip predictions!)
- **Interpretation**: probability that the model ranks a random positive **higher** than a random negative.

**(d) Precision-Recall Curve**

Plot **precision vs recall** at various thresholds. **Better than ROC for imbalanced datasets** because ROC can look deceptively good when negatives vastly outnumber positives (FPR stays low simply because the denominator is huge).

**(e) Log Loss**

$$\text{Log Loss} = -\frac{1}{m}\sum_{i=1}^{m}\bigl[y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)})\bigr]$$

Directly measures the **calibration of probabilities**, not just classification accuracy. A model that's "right but not confident" scores worse than one that's "right and confident."

---

**The Precision-Recall Tradeoff**

The classification threshold controls the tradeoff:

- **Lower threshold** → more positives predicted → **recall ↑, precision ↓**
- **Higher threshold** → fewer positives predicted → **recall ↓, precision ↑**

Choosing the metric depends on the **cost of errors**:

| Problem | Which matters most? | Why |
|---|---|---|
| Spam detection | **Precision** | Don't flag good emails as spam |
| Cancer screening | **Recall** | Don't miss sick patients |
| Fraud detection | **Both** (F1, PR-AUC) | Missing fraud is costly; flagging good users is costly |
| Search ranking | **Precision @ k** | Top results must be relevant |
| Imbalanced data generally | **F1, PR-AUC** | Accuracy is meaningless |

---

**13. Handling Class Imbalance**

When one class is much rarer than the other (e.g., **fraud detection**: 1% fraud, 99% legitimate), naive training will produce a model that mostly ignores the minority class.

**(a) Resampling**

- **Oversampling the minority class** — duplicate or synthesize minority examples.
	- **SMOTE** (Synthetic Minority Over-sampling Technique): generates new synthetic minority points by **interpolating between existing minority neighbors** in feature space.
	- ❌ Risk: can amplify noise; synthetic points may not be realistic.
- **Undersampling the majority class** — randomly drop majority examples.
	- ❌ Risk: throws away potentially useful information.

**(b) Class Weights**

Most libraries (`sklearn`, `PyTorch`, `TensorFlow`) let you weight the loss **inversely to class frequency**:

$$J(\beta) = -\frac{1}{m}\sum_{i=1}^{m} w_{y^{(i)}}\bigl[y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)})\bigr]$$

Common choice: $w_k = \frac{m}{K \cdot m_k}$, where $m_k$ is the count of class $k$. No data duplication needed — same effect as oversampling, much cheaper.

**(c) Threshold Tuning**

Don't blindly use 0.5 as the threshold. Pick the threshold that **optimizes your business metric** (F1, recall at fixed precision, expected cost, etc.) using the **Precision-Recall curve** on a validation set.

**(d) Use Appropriate Metrics**

Use **F1, PR-AUC, balanced accuracy, or class-specific metrics** — not raw accuracy.

**(e) Anomaly Detection**

For **extreme imbalances** (e.g., 0.01% positive), treat it as **anomaly detection** rather than classification: Isolation Forest, One-Class SVM, autoencoders, etc.

**(f) Focal Loss**

Introduced in the **RetinaNet** paper (Lin et al., 2017) for dense object detection, where background examples vastly outnumber object examples. Focal loss reshapes cross-entropy to **down-weight easy examples** so the model focuses on hard, misclassified ones.

Standard binary cross-entropy for one example:

$$\text{CE}(p_t) = -\log(p_t)$$

where $p_t$ is the model's predicted probability for the **true class**:

$$p_t = \begin{cases} \hat{y} & \text{if } y = 1 \\ 1 - \hat{y} & \text{if } y = 0 \end{cases}$$

**Focal loss** adds a modulating factor $(1 - p_t)^\gamma$:

$$\boxed{\,\text{FL}(p_t) = -\alpha_t \,(1 - p_t)^\gamma \, \log(p_t)\,}$$

- $\gamma \geq 0$ is the **focusing parameter** (typically $\gamma = 2$).
- $\alpha_t \in [0, 1]$ is an optional **class-weighting** factor (works like class weights).

**How it works**

| Example type | $p_t$ | $(1 - p_t)^\gamma$ with $\gamma = 2$ | Effect |
|---|---|---|---|
| Easy, correctly classified | 0.9 | $0.01$ | Loss reduced **100×** |
| Moderately classified | 0.5 | $0.25$ | Loss reduced 4× |
| Hard / misclassified | 0.1 | $0.81$ | Loss barely changed |

The model effectively **stops wasting capacity on examples it already gets right**, and instead concentrates gradient on the hard cases.

**When to use it**

- Severe class imbalance (especially in **dense prediction** tasks like object detection or segmentation).
- When **class weights alone aren't enough** — they balance the classes but don't distinguish easy from hard examples.
- When the **majority class is also easy** (the typical pattern), so most of the loss is being driven by examples the model already handles well.

> **Note:** Focal loss vs class weights
> **Class weights** rescale by *class identity*. **Focal loss** rescales by *example difficulty*. They're complementary — you can (and typically do) use both together via the $\alpha_t$ term.

> **Tip:** Reduces to cross-entropy
> When $\gamma = 0$, focal loss is **identical to weighted binary cross-entropy**. The $\gamma$ knob lets you smoothly interpolate from "treat all examples equally" to "focus heavily on hard ones."
