---
layout: note
title: "Neural Networks"
description: "A neural network is a parameterized function $f\\theta: \\mathbb{R}^d \\rightarrow \\mathbb{R}^k$ that maps inputs to outputs by composing many simple nonlinear transformations. The…"
note: true
note_collection: "ML algorithms"
note_section: "Deep Learning"
section_order: 4
note_order: 3
updated: 2026-06-10 21:19:19 -0700
keywords:
  - Deep Learning
  - Training
  - Optimization
  - Clustering
  - Evaluation
math: true
mermaid: false
---
## Table of Contents

- [1. What is a Neural Network?](#1-what-is-a-neural-network)
- [2. The Building Block — A Single Neuron](#2-the-building-block-a-single-neuron)
- [3. Multi-Layer Perceptron (MLP)](#3-multi-layer-perceptron-mlp)
- [4. Activation Functions](#4-activation-functions)
- [5. Loss Functions](#5-loss-functions)
- [6. Forward Propagation](#6-forward-propagation)
- [7. Backpropagation — Full Derivation](#7-backpropagation-full-derivation)
- [8. Backpropagation — Indexing Demystified](#8-backpropagation-indexing-demystified)
- [9. Output Layer Derivation — Why δ = ŷ − y](#9-output-layer-derivation-why-y)
- [10. Gradient Descent and Optimizers](#10-gradient-descent-and-optimizers)
- [11. Weight Initialization](#11-weight-initialization)
- [12. Regularization Techniques](#12-regularization-techniques)
- [13. BatchNorm vs LayerNorm — Detailed](#13-batchnorm-vs-layernorm-detailed)
- [14. Vanishing & Exploding Gradients](#14-vanishing-and-exploding-gradients)
- [15. Hyperparameters](#15-hyperparameters)
- [16. Universal Approximation Theorem](#16-universal-approximation-theorem)
- [17. Strengths and Weaknesses](#17-strengths-and-weaknesses)
- [18. When to Use vs Not Use](#18-when-to-use-vs-not-use)
- [19. NN vs Other Algorithms](#19-nn-vs-other-algorithms)
- [21. Production Considerations](#21-production-considerations)

---

## 1. What is a Neural Network?

A **neural network** is a parameterized function $f_\theta: \mathbb{R}^d \rightarrow \mathbb{R}^k$ that maps inputs to outputs by composing many simple nonlinear transformations. The "neural" terminology comes from loose biological inspiration, but mathematically it's just **function composition with learnable parameters**.

The core idea: any sufficiently complex function can be approximated by stacking simple building blocks (affine transformations + nonlinearities). This is the **Universal Approximation Theorem** — even a single hidden layer with enough neurons can approximate any continuous function on a compact domain (though "enough" may be exponentially large).

---

## 2. The Building Block — A Single Neuron

A single neuron computes:

$$z = \mathbf{w}^\top \mathbf{x} + b, \quad a = \sigma(z)$$

where:

- $\mathbf{x} \in \mathbb{R}^d$ is the input vector
- $\mathbf{w} \in \mathbb{R}^d$ is the weight vector (learnable)
- $b \in \mathbb{R}$ is the bias (learnable)
- $\sigma(\cdot)$ is a nonlinear activation function
- $z$ is the **pre-activation**, $a$ is the **post-activation** (or just "activation")

**Why nonlinearity?** Without $\sigma$, composing layers gives:

$$\mathbf{W}_2(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2 = (\mathbf{W}_2 \mathbf{W}_1)\mathbf{x} + (\mathbf{W}_2 \mathbf{b}_1 + \mathbf{b}_2)$$

This is just another affine transformation. **Without nonlinearity, any deep network collapses to a linear model.** Nonlinearity is what gives depth its power.

---

## 3. Multi-Layer Perceptron (MLP)

For a network with $L$ layers, define recursively:

$$\mathbf{z}^{(\ell)} = \mathbf{W}^{(\ell)} \mathbf{a}^{(\ell-1)} + \mathbf{b}^{(\ell)}$$ $$\mathbf{a}^{(\ell)} = \sigma^{(\ell)}(\mathbf{z}^{(\ell)})$$

with $\mathbf{a}^{(0)} = \mathbf{x}$ (the input) and $\hat{\mathbf{y}} = \mathbf{a}^{(L)}$ (the output).

**Shapes:**

- $\mathbf{W}^{(\ell)} \in \mathbb{R}^{n_\ell \times n_{\ell-1}}$
- $\mathbf{b}^{(\ell)} \in \mathbb{R}^{n_\ell}$
- $\mathbf{a}^{(\ell)}, \mathbf{z}^{(\ell)} \in \mathbb{R}^{n_\ell}$

**Total parameter count** for a layer: $n_\ell \cdot n_{\ell-1} + n_\ell$.

### Worked Example: MNIST MLP

For an MLP with input dim 784 (MNIST), hidden layers [256, 128], output dim 10:

- Layer 1: $784 \times 256 + 256 = 200{,}960$ params
- Layer 2: $256 \times 128 + 128 = 32{,}896$ params
- Layer 3: $128 \times 10 + 10 = 1{,}290$ params
- **Total: 235,146 parameters**

---

## 4. Activation Functions

|Activation|Formula|Range|Derivative|Pros|Cons|
|---|---|---|---|---|---|
|**Sigmoid**|$\frac{1}{1+e^{-z}}$|$(0,1)$|$\sigma(z)(1-\sigma(z))$|Smooth, probabilistic interpretation|Vanishing gradients, not zero-centered, saturates|
|**Tanh**|$\frac{e^z - e^{-z}}{e^z + e^{-z}}$|$(-1,1)$|$1-\tanh^2(z)$|Zero-centered|Still saturates → vanishing gradients|
|**ReLU**|$\max(0,z)$|$[0,\infty)$|$\mathbb{1}[z>0]$|Fast, no saturation for $z>0$, sparse|"Dying ReLU" (neurons stuck at 0)|
|**Leaky ReLU**|$\max(\alpha z, z)$, $\alpha \approx 0.01$|$\mathbb{R}$|$\alpha$ or $1$|Fixes dying ReLU|Extra hyperparameter|
|**PReLU**|$\max(\alpha z, z)$, $\alpha$ learned|$\mathbb{R}$|—|Learns slope|More params, can overfit|
|**ELU**|$z$ if $z>0$, else $\alpha(e^z-1)$|$(-\alpha, \infty)$|—|Smooth, zero-mean activations|Slower (exp)|
|**GELU**|$z \cdot \Phi(z)$|$\mathbb{R}$|—|Smooth, used in Transformers|Slower|
|**Swish/SiLU**|$z \cdot \sigma(z)$|$\mathbb{R}$|—|Smooth, often beats ReLU|Slightly slower|
|**Softmax**|$\frac{e^{z_i}}{\sum_j e^{z_j}}$|$(0,1)$, sums to 1|—|Probabilistic output for multi-class|Only for output layer|

> **Tip:** Intuition on vanishing gradients with sigmoid $\sigma'(z) \leq 0.25$, so backpropagating through $L$ sigmoid layers multiplies gradients by at most $0.25^L$ — exponentially small. ReLU avoids this for $z > 0$ since $\text{ReLU}'(z) = 1$.

---

## 5. Loss Functions

The loss measures how wrong predictions are. Choice depends on task.

### Regression — Mean Squared Error (MSE)

$$\mathcal{L}_{\text{MSE}} = \frac{1}{N}\sum_{i=1}^N (\hat{y}_i - y_i)^2$$

Derived from Gaussian likelihood (assumes Gaussian noise).

### Binary Classification — Binary Cross-Entropy (BCE)

$$\mathcal{L}_{\text{BCE}} = -\frac{1}{N}\sum_{i=1}^N \left[y_i \log \hat{y}_i + (1-y_i)\log(1-\hat{y}_i)\right]$$

### Multi-class — Categorical Cross-Entropy (CCE)

$$\mathcal{L}_{\text{CCE}} = -\frac{1}{N}\sum_{i=1}^N \sum_{c=1}^C y_{i,c} \log \hat{y}_{i,c}$$

All these are **maximum likelihood estimators** under different output distributions (Gaussian, Bernoulli, Categorical).

> **Important:** Why cross-entropy instead of MSE for classification? With sigmoid + MSE, the gradient is $(\hat{y}-y) \cdot \sigma'(z)$ — when prediction is very wrong, $\sigma'(z)$ is tiny, so learning is slow. With sigmoid + BCE, the gradient is simply $(\hat{y}-y)$ — proportional to the error, no saturation problem.

---

## 6. Forward Propagation

Given input $\mathbf{x}$, compute layer by layer:

```
a⁽⁰⁾ = x
for ℓ = 1 to L:
    z⁽ℓ⁾ = W⁽ℓ⁾ a⁽ℓ⁻¹⁾ + b⁽ℓ⁾
    a⁽ℓ⁾ = σ(z⁽ℓ⁾)
ŷ = a⁽ᴸ⁾
L = loss(ŷ, y)
```

Cache all $\mathbf{z}^{(\ell)}, \mathbf{a}^{(\ell)}$ — needed for backprop.

---

## 7. Backpropagation — Full Derivation

Backprop is just **the chain rule applied efficiently** with dynamic programming. Let $\mathcal{L}$ be the scalar loss.

Define the **error signal** at layer $\ell$:

$$\boldsymbol{\delta}^{(\ell)} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(\ell)}}$$

### Step 1: Output Layer Error

For softmax + cross-entropy (a common case):

$$\boldsymbol{\delta}^{(L)} = \hat{\mathbf{y}} - \mathbf{y}$$

This clean form is _why_ softmax+CCE pair is standard. (Full derivation in [9. Output Layer Derivation — Why δ = ŷ − y](#9-output-layer-derivation-why-y).)

### Step 2: Backpropagate the Error

Using chain rule:

$$\boldsymbol{\delta}^{(\ell)} = \left( \mathbf{W}^{(\ell+1)\top} \boldsymbol{\delta}^{(\ell+1)} \right) \odot \sigma'(\mathbf{z}^{(\ell)})$$

where $\odot$ is element-wise (Hadamard) product.

> **Tip:** Intuition Errors at layer $\ell+1$ flow back through the transposed weights, then get gated by how active that neuron was (the derivative of activation).

### Step 3: Parameter Gradients

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(\ell)}} = \boldsymbol{\delta}^{(\ell)} \mathbf{a}^{(\ell-1)\top}$$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(\ell)}} = \boldsymbol{\delta}^{(\ell)}$$

### Why is backprop O(params) and not exponential?

Naive symbolic differentiation would re-derive partial derivatives many times. Backprop reuses intermediate values — it's just **reverse-mode automatic differentiation**, with cost ~2× the forward pass.

---

## 8. Backpropagation — Indexing Demystified

The indexing is the most confusing part of backprop. Let's be super explicit about what every index means.

### 8.1 What the Indices Mean

|Letter|What it indexes|Example|
|---|---|---|
|$\ell$|**Layer**|$1, 2, 3$|
|$i$|**Neuron in the current layer** (layer $\ell$)|depends on $\ell$|
|$j$|**Neuron in the previous layer** (layer $\ell - 1$)|depends on $\ell$|
|$k$|**Neuron in the next layer** (layer $\ell + 1$)|depends on $\ell$|

**Key mnemonic:**

```
layer ℓ-1          layer ℓ            layer ℓ+1
 (prev)           (current)            (next)
   j      →→→        i        →→→        k
```

- $j$ feeds **into** layer $\ell$
- $i$ is the neuron we're focusing on **in** layer $\ell$
- $k$ is in the layer that $i$ feeds **into**

### 8.2 What a Single Neuron Does — With Indices

Neuron $i$ in layer $\ell$ computes:

$$z_i^{(\ell)} = \underbrace{\sum_j W_{ij}^{(\ell)} , a_j^{(\ell-1)}}_{\text{sum over previous-layer neurons}} + b_i^{(\ell)}$$

$$a_i^{(\ell)} = \sigma(z_i^{(\ell)})$$

**The summation is over $j$ — every neuron in the previous layer.**

### 8.3 What does $W_{ij}^{(\ell)}$ mean exactly?

$W_{ij}^{(\ell)}$ is **the weight from neuron $j$ (in layer $\ell-1$) to neuron $i$ (in layer $\ell$)**.

**Read indices right-to-left:** "from $j$ to $i$" — destination first, source second.

```
neuron j (layer ℓ-1)   --[ weight W_{ij}^{(ℓ)} ]-->   neuron i (layer ℓ)
```

This convention makes the matrix form $\mathbf{z}^{(\ell)} = \mathbf{W}^{(\ell)} \mathbf{a}^{(\ell-1)}$ work: rows indexed by destination ($i$), columns by source ($j$).

### 8.4 The Big Idea of Backprop

We want $\dfrac{\partial \mathcal{L}}{\partial W_{ij}^{(\ell)}}$ for every weight. Backprop's trick: compute one intermediate quantity per _neuron_:

$$\delta_i^{(\ell)} ;\stackrel{\text{def}}{=}; \frac{\partial \mathcal{L}}{\partial z_i^{(\ell)}}$$

Once we have $\delta_i^{(\ell)}$ for every neuron, getting the gradient of any weight is a single multiplication.

### 8.5 Deriving $\delta_i^{(\ell)}$ — Where Do $k$ and the Sum Come From?

When you change $z_i^{(\ell)}$, you change $a_i^{(\ell)}$, which affects **every** $z_k^{(\ell+1)}$ in the next layer — because neuron $i$'s output is an input to all neurons in the next layer.

**That's why there's a sum over $k$ — $k$ ranges over the neurons in layer $\ell+1$.**

By the multivariable chain rule:

$$\frac{\partial \mathcal{L}}{\partial z_i^{(\ell)}} = \sum_{k} \frac{\partial \mathcal{L}}{\partial z_k^{(\ell+1)}} \cdot \frac{\partial z_k^{(\ell+1)}}{\partial z_i^{(\ell)}}$$

The first factor is $\delta_k^{(\ell+1)}$ — already computed. That's the dynamic programming!

$$\delta_i^{(\ell)} = \sum_{k} \delta_k^{(\ell+1)} \cdot \frac{\partial z_k^{(\ell+1)}}{\partial z_i^{(\ell)}}$$

### 8.6 Computing $\dfrac{\partial z_k^{(\ell+1)}}{\partial z_i^{(\ell)}}$

By definition:

$$z_k^{(\ell+1)} = \sum_{j'} W_{kj'}^{(\ell+1)} , a_{j'}^{(\ell)} + b_k^{(\ell+1)}$$

Now $a_{j'}^{(\ell)} = \sigma(z_{j'}^{(\ell)})$. When we take $\partial / \partial z_i^{(\ell)}$, only the $j' = i$ term survives:

$$\frac{\partial z_k^{(\ell+1)}}{\partial z_i^{(\ell)}} = W_{ki}^{(\ell+1)} \cdot \sigma'(z_i^{(\ell)})$$

Substituting back:

$$\boxed{\delta_i^{(\ell)} = \sigma'(z_i^{(\ell)}) \sum_{k} W_{ki}^{(\ell+1)} , \delta_k^{(\ell+1)}}$$

### 8.7 Reading the Formula

|Piece|Meaning|
|---|---|
|$\delta_i^{(\ell)}$|Error signal at neuron $i$ in layer $\ell$|
|$\sum_k$|Sum over all neurons $k$ in the **next** layer|
|$\delta_k^{(\ell+1)}$|Error signal at next layer (already computed)|
|$W_{ki}^{(\ell+1)}$|Weight from $i$ (current) to $k$ (next)|
|$\sigma'(z_i^{(\ell)})$|Slope of activation at neuron $i$'s pre-activation|

> **Quote:** Plain English "Neuron $i$'s error = (how active it was, in derivative sense) × (sum of errors at all neurons it feeds into, each weighted by how strongly it influences them)."

### 8.8 Concrete Worked Example

Let $\ell = 1$ (4 neurons), $\ell + 1 = 2$ (2 neurons). So $i \in {1, 2, 3, 4}$ and $k \in {1, 2}$.

Given:

- $\delta_1^{(2)} = 0.3$, $\delta_2^{(2)} = -0.1$
- $\mathbf{W}^{(2)} = \begin{bmatrix} 0.5 & -0.2 & 0.1 & 0.4 \ 0.3 & 0.6 & -0.4 & 0.2 \end{bmatrix}$
- ReLU with all $z_i^{(1)} > 0$, so $\sigma'(z_i^{(1)}) = 1$

Compute:

$$\delta_i^{(1)} = \sigma'(z_i^{(1)}) \sum_{k=1}^{2} W_{ki}^{(2)} \cdot \delta_k^{(2)}$$

- $\delta_1^{(1)} = (0.5)(0.3) + (0.3)(-0.1) = 0.12$
- $\delta_2^{(1)} = (-0.2)(0.3) + (0.6)(-0.1) = -0.12$
- $\delta_3^{(1)} = (0.1)(0.3) + (-0.4)(-0.1) = 0.07$
- $\delta_4^{(1)} = (0.4)(0.3) + (0.2)(-0.1) = 0.10$

So $\boldsymbol{\delta}^{(1)} = [0.12, -0.12, 0.07, 0.10]$.

**Notice:** for each $i$, we summed over the _column_ $i$ of $\mathbf{W}^{(2)}$ — this is what the transpose in the compact matrix form does.

### 8.9 Gradient of Weights — Where Does $j$ Come Back?

Once we have $\delta_i^{(\ell)}$:

$$\frac{\partial \mathcal{L}}{\partial W_{ij}^{(\ell)}} = \delta_i^{(\ell)} \cdot a_j^{(\ell-1)}$$

**Derivation:** $z_i^{(\ell)} = \sum_{j'} W_{ij'}^{(\ell)} a_{j'}^{(\ell-1)} + b_i^{(\ell)}$, so $\partial z_i^{(\ell)} / \partial W_{ij}^{(\ell)} = a_j^{(\ell-1)}$. Chain rule:

$$\frac{\partial \mathcal{L}}{\partial W_{ij}^{(\ell)}} = \delta_i^{(\ell)} \cdot a_j^{(\ell-1)}$$

> **Quote:** Plain English "Gradient of weight from $j$ to $i$ = (error at destination $i$) × (activation of source $j$)."

If the source neuron was inactive ($a_j^{(\ell-1)} = 0$), the weight doesn't change. **No summation here** — single weight, single source-destination pair.

### 8.10 Summary Table — When Do We Sum?

|Quantity|Formula|Sum over?|Why?|
|---|---|---|---|
|$z_i^{(\ell)}$ (forward)|$\sum_j W_{ij}^{(\ell)} a_j^{(\ell-1)} + b_i^{(\ell)}$|$j$ = previous-layer|Aggregating inputs|
|$\delta_i^{(\ell)}$ (backward)|$\sigma'(z_i^{(\ell)}) \sum_k W_{ki}^{(\ell+1)} \delta_k^{(\ell+1)}$|$k$ = next-layer|Multivariable chain rule|
|$\partial \mathcal{L} / \partial W_{ij}^{(\ell)}$|$\delta_i^{(\ell)} \cdot a_j^{(\ell-1)}$|**No sum**|Single weight|
|$\partial \mathcal{L} / \partial b_i^{(\ell)}$|$\delta_i^{(\ell)}$|**No sum**|Single bias|

### 8.11 The Three Indices in One Picture

```
       layer ℓ-1        layer ℓ          layer ℓ+1
                          
       j = 1 ────W_{ij}^(ℓ)────► i = 1 ────W_{ki}^(ℓ+1)────► k = 1
       j = 2 ────────────────►  i = 2  ────────────────►     k = 2
       j = 3 ────────────────►  i = 3  ────────────────►     ...
       ...                     ...

   FORWARD: sum over j (inputs to i)
       z_i^(ℓ) = Σ_j W_{ij}^(ℓ) a_j^(ℓ-1) + b_i^(ℓ)

   BACKWARD: sum over k (outputs from i)
       δ_i^(ℓ) = σ'(z_i^(ℓ)) Σ_k W_{ki}^(ℓ+1) δ_k^(ℓ+1)
```

> **Important:** The forward sum goes over **inputs** ($j$); the backward sum goes over **outputs** ($k$). This symmetry is the whole structure of backprop.

### 8.12 Backprop Algorithm Summarized

```
a⁽⁰⁾ = x
for ℓ = 1, 2, ..., L:
    z⁽ℓ⁾ = W⁽ℓ⁾ a⁽ℓ⁻¹⁾ + b⁽ℓ⁾
    a⁽ℓ⁾ = σ(z⁽ℓ⁾)
ŷ = a⁽ᴸ⁾
compute L(ŷ, y)

# Backward pass
δ⁽ᴸ⁾ = ŷ - y                              # output layer
for ℓ = L, L-1, ..., 1:
    ∂L/∂W⁽ℓ⁾ = δ⁽ℓ⁾ (a⁽ℓ⁻¹⁾)ᵀ            # outer product
    ∂L/∂b⁽ℓ⁾ = δ⁽ℓ⁾
    if ℓ > 1:
        δ⁽ℓ⁻¹⁾ = (W⁽ℓ⁾ᵀ δ⁽ℓ⁾) ⊙ σ'(z⁽ℓ⁻¹⁾)
```

---

## 9. Output Layer Derivation — Why δ = ŷ − y

For sigmoid + BCE (binary classification):

- $\hat{y} = \sigma(z^{(L)})$
- $\mathcal{L} = -[y \log \hat{y} + (1-y)\log(1-\hat{y})]$

By chain rule: $\delta^{(L)} = \dfrac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \dfrac{\partial \hat{y}}{\partial z^{(L)}}$.

### Step 1: $\dfrac{\partial \mathcal{L}}{\partial \hat{y}}$

$$\frac{\partial \mathcal{L}}{\partial \hat{y}} = -\frac{y}{\hat{y}} + \frac{1-y}{1-\hat{y}} = \frac{-y(1-\hat{y}) + (1-y)\hat{y}}{\hat{y}(1-\hat{y})} = \frac{\hat{y} - y}{\hat{y}(1-\hat{y})}$$

### Step 2: $\dfrac{\partial \hat{y}}{\partial z^{(L)}}$

Sigmoid identity: $\sigma'(z) = \sigma(z)(1-\sigma(z))$, so:

$$\frac{\partial \hat{y}}{\partial z^{(L)}} = \hat{y}(1-\hat{y})$$

### Step 3: Magical Cancellation

$$\delta^{(L)} = \frac{\hat{y} - y}{\hat{y}(1-\hat{y})} \cdot \hat{y}(1-\hat{y}) = \boxed{\hat{y} - y}$$

### Why this matters

With **sigmoid + MSE**, $\delta^{(L)} = (\hat{y} - y) \cdot \hat{y}(1-\hat{y})$ — saturates when prediction is very wrong. With **sigmoid + BCE**, $\delta^{(L)} = \hat{y} - y$ — proportional to error.

### Softmax + CCE

For multi-class, same clean result: $\dfrac{\partial \mathcal{L}}{\partial z_j} = \hat{y}_j - y_j$.

**Proof:** Use $\dfrac{\partial \hat{y}_i}{\partial z_j} = \hat{y}_i(\delta_{ij} - \hat{y}_j)$ (where $\delta_{ij}$ is Kronecker delta), then:

$$\frac{\partial \mathcal{L}}{\partial z_j} = \sum_i \left(-\frac{y_i}{\hat{y}_i}\right) \hat{y}_i(\delta_{ij} - \hat{y}_j) = -y_j + \hat{y}_j \sum_i y_i = \hat{y}_j - y_j$$

(Since $\mathbf{y}$ is one-hot, $\sum_i y_i = 1$.)

### Summary Table

|Pairing|$\delta^{(L)}$|Saturates?|
|---|---|---|
|Sigmoid + BCE|$\hat{y} - y$|**No**|
|Softmax + CCE|$\hat{\mathbf{y}} - \mathbf{y}$|**No**|
|Sigmoid + MSE|$(\hat{y}-y)\hat{y}(1-\hat{y})$|**Yes** — vanishes when very wrong|
|Linear + MSE|$\hat{y} - y$|No (no activation to saturate)|

---

## 10. Gradient Descent and Optimizers

### Vanilla Gradient Descent

$$\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}(\theta)$$

|Variant|Batch Size|Pros|Cons|
|---|---|---|---|
|**Batch GD**|$N$ (full)|Stable, true gradient|Slow, memory-heavy|
|**SGD**|1|Fast, noisy escapes local minima|Very noisy|
|**Mini-batch SGD**|32–512|Balance + GPU-friendly|Need to tune batch size|

### Momentum

$$\mathbf{v}_t = \beta \mathbf{v}_{t-1} + \nabla_\theta \mathcal{L}, \quad \theta \leftarrow \theta - \eta \mathbf{v}_t$$

Accumulates velocity; smooths out oscillations in ravines. Typical $\beta = 0.9$.

### Nesterov Accelerated Gradient (NAG)

Looks ahead: computes gradient at $\theta - \eta\beta \mathbf{v}_{t-1}$ instead of $\theta$. Slightly better convergence.

### AdaGrad

$$\mathbf{G}_t = \mathbf{G}_{t-1} + (\nabla \mathcal{L})^2, \quad \theta \leftarrow \theta - \frac{\eta}{\sqrt{\mathbf{G}_t + \epsilon}} \nabla \mathcal{L}$$

Per-parameter learning rate. Issue: $\mathbf{G}_t$ grows unboundedly → learning rate decays to 0.

### RMSProp

$$\mathbf{G}_t = \beta \mathbf{G}_{t-1} + (1-\beta)(\nabla \mathcal{L})^2$$

Fixes AdaGrad with exponential moving average.

### Adam

Combines momentum + RMSProp:

$$\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1-\beta_1)\nabla\mathcal{L}$$ $$\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1-\beta_2)(\nabla\mathcal{L})^2$$ $$\hat{\mathbf{m}}_t = \mathbf{m}_t / (1-\beta_1^t), \quad \hat{\mathbf{v}}_t = \mathbf{v}_t / (1-\beta_2^t)$$ $$\theta \leftarrow \theta - \eta \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}$$

Bias correction $1-\beta^t$ is crucial in early steps when $\mathbf{m}, \mathbf{v}$ are biased toward zero.

**Defaults:** $\beta_1=0.9, \beta_2=0.999, \epsilon=10^{-8}$, $\eta=10^{-3}$.

### AdamW

Decouples weight decay from gradient update. Fixes a subtle bug in Adam where L2 regularization interacts incorrectly with adaptive learning rates.

> **Tip:** Use AdamW, not Adam with `weight_decay`, in modern practice.

---

## 11. Weight Initialization

### Xavier/Glorot Initialization (for tanh/sigmoid)

$$W_{ij} \sim \mathcal{N}\left(0, \frac{2}{n_{\text{in}} + n_{\text{out}}}\right)$$

**Derivation:** We want $\text{Var}(\mathbf{z}^{(\ell)}) \approx \text{Var}(\mathbf{a}^{(\ell-1)})$. For $z = \sum_i w_i a_i$:

$$\text{Var}(z) = n_{\text{in}} \cdot \text{Var}(w) \cdot \text{Var}(a)$$

To keep variances equal: $\text{Var}(w) = 1/n_{\text{in}}$. Symmetric argument for backward pass gives $1/n_{\text{out}}$. Glorot averages them.

### He Initialization (for ReLU)

$$W_{ij} \sim \mathcal{N}\left(0, \frac{2}{n_{\text{in}}}\right)$$

ReLU kills half the activations on average, so we need 2× variance to compensate.

> **Warning:** Never initialize all weights to zero Every neuron in a layer computes the same thing, gets the same gradient, stays identical. This is the **symmetry problem**.

**Rule of thumb:**

- He init for ReLU and variants
- Glorot for tanh/sigmoid

---

## 12. Regularization Techniques

### L2 (Weight Decay)

$$\mathcal{L}_{\text{total}} = \mathcal{L} + \frac{\lambda}{2}|\theta|^2$$

Gradient becomes $\nabla\mathcal{L} + \lambda \theta$ — shrinks weights toward zero each step.

### L1 (Sparsity)

$$\mathcal{L}_{\text{total}} = \mathcal{L} + \lambda |\theta|_1$$

Encourages exact zeros (feature selection).

### Dropout

At training time, randomly zero out each activation with probability $p$:

$$\tilde{a}_i = \frac{a_i \cdot m_i}{1-p}, \quad m_i \sim \text{Bernoulli}(1-p)$$

The $1/(1-p)$ scaling ("inverted dropout") keeps expected activation magnitude unchanged — no rescaling needed at inference.

**Typical:** $p = 0.5$ for hidden, $0.1$–$0.2$ for input.

> **Tip:** Why dropout works Acts as ensemble of exponentially many sub-networks; prevents co-adaptation of neurons.

**At inference: dropout is OFF** (use full network).

### Early Stopping

Monitor validation loss; stop when it stops improving. Implicit regularization — limits effective capacity.

### Data Augmentation

Synthetic variations (rotation, crop, flip, noise) — best regularizer when applicable.

---

## 13. BatchNorm vs LayerNorm — Detailed

### 13.1 The Core Problem Both Solve

When training deep networks, the distribution of inputs to each layer changes as earlier layers update. **Solution:** Normalize activations to a stable distribution before passing them on.

The general normalization step:

$$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}, \quad y = \gamma \hat{x} + \beta$$

where $\gamma, \beta$ are learnable parameters. $\epsilon \sim 10^{-5}$ for numerical stability.

> **Important:** The only difference between BN, LN, etc. is: **over what dimensions do we compute $\mu, \sigma^2$?**

### 13.2 Setup: Activation Matrix

For batch size $m = 4$, features $d = 3$:

$$\mathbf{A} = \begin{bmatrix} a_{11} & a_{12} & a_{13} \ a_{21} & a_{22} & a_{23} \ a_{31} & a_{32} & a_{33} \ a_{41} & a_{42} & a_{43} \end{bmatrix} \begin{matrix} \leftarrow \text{sample 1} \ \leftarrow \text{sample 2} \ \leftarrow \text{sample 3} \ \leftarrow \text{sample 4} \end{matrix}$$

Rows are samples; columns are features.

### 13.3 BatchNorm — Normalize Across the Batch

For feature $j$:

$$\mu_j = \frac{1}{m}\sum_{i=1}^m a_{ij}, \quad \sigma_j^2 = \frac{1}{m}\sum_{i=1}^m (a_{ij} - \mu_j)^2$$

$$\hat{a}_{ij} = \frac{a_{ij} - \mu_j}{\sqrt{\sigma_j^2 + \epsilon}}, \quad y_{ij} = \gamma_j \hat{a}_{ij} + \beta_j$$

**Visually:** Compute $\mu, \sigma^2$ down each **column** of $\mathbf{A}$.

```
        feat1   feat2   feat3
sample1   ↓       ↓       ↓
sample2   ↓       ↓       ↓
sample3   ↓       ↓       ↓
sample4   ↓       ↓       ↓
        mean,   mean,   mean,
        var     var     var
```

**Key points:**

- $\mu_j, \sigma_j^2$ are per feature
- $\gamma_j, \beta_j$ are per feature ($2d$ learnable params)
- **Statistics depend on the batch**

**Inference:** Use running averages of $\mu, \sigma^2$ collected during training.

```python
# Training:
running_mean = momentum * running_mean + (1 - momentum) * batch_mean
running_var  = momentum * running_var  + (1 - momentum) * batch_var

# Inference: use running_mean, running_var (frozen)
```

### 13.4 LayerNorm — Normalize Across Features

For sample $i$:

$$\mu_i = \frac{1}{d}\sum_{j=1}^d a_{ij}, \quad \sigma_i^2 = \frac{1}{d}\sum_{j=1}^d (a_{ij} - \mu_i)^2$$

$$\hat{a}_{ij} = \frac{a_{ij} - \mu_i}{\sqrt{\sigma_i^2 + \epsilon}}, \quad y_{ij} = \gamma_j \hat{a}_{ij} + \beta_j$$

**Visually:** Compute $\mu, \sigma^2$ across each **row** of $\mathbf{A}$.

```
        feat1   feat2   feat3
sample1 → → →                    mean, var (sample 1)
sample2 → → →                    mean, var (sample 2)
sample3 → → →                    mean, var (sample 3)
sample4 → → →                    mean, var (sample 4)
```

**Key points:**

- $\mu_i, \sigma_i^2$ are per sample
- $\gamma_j, \beta_j$ are still per feature
- **Statistics independent of batch**
- **No running averages.** Train and inference identical.

### 13.5 Side-by-Side Comparison

|Aspect|BatchNorm|LayerNorm|
|---|---|---|
|**Normalize over**|Batch dim (across samples)|Feature dim (across features)|
|**One $\mu, \sigma^2$ per**|Feature|Sample|
|**Depends on batch**|Yes|No|
|**Train ≠ inference?**|Yes (running stats at inference)|No|
|**Works with batch size 1?**|No|Yes|
|**Sequence length issues**|Yes|No|
|**Best for**|CNNs, large batches|Transformers, RNNs|
|**Sensitivity to batch size**|Very sensitive|Insensitive|

### 13.6 Concrete Numerical Example

Take:

$$\mathbf{A} = \begin{bmatrix} 1 & 2 & 3 \ 4 & 5 & 6 \ 7 & 8 & 9 \ 10 & 11 & 12 \end{bmatrix}$$

**BatchNorm** (per column):

- Feature 1: $\mu = 5.5$, $\sigma^2 = 11.25$
- Feature 2: $\mu = 6.5$, $\sigma^2 = 11.25$
- Feature 3: $\mu = 7.5$, $\sigma^2 = 11.25$

$$\hat{\mathbf{A}}_{\text{BN}} = \begin{bmatrix} -1.34 & -1.34 & -1.34 \ -0.45 & -0.45 & -0.45 \ 0.45 & 0.45 & 0.45 \ 1.34 & 1.34 & 1.34 \end{bmatrix}$$

**LayerNorm** (per row):

- Sample 1: $\mu = 2$, $\sigma^2 = 2/3$
- Sample 2: $\mu = 5$, $\sigma^2 = 2/3$
- ...

$$\hat{\mathbf{A}}_{\text{LN}} = \begin{bmatrix} -1.22 & 0 & 1.22 \ -1.22 & 0 & 1.22 \ -1.22 & 0 & 1.22 \ -1.22 & 0 & 1.22 \end{bmatrix}$$

> **Important:** They do fundamentally different things, even though the formula looks the same!

### 13.7 When to Use Which?

**BatchNorm shines when:**

- Large batch sizes (≥ 32)
- Images / CNNs (each spatial pixel ≈ a sample)
- Distributions stable across samples

**BatchNorm fails when:**

- Tiny batch sizes
- Sequence models with variable lengths
- Online/streaming inference with batch size 1
- Test-time distribution shift

**LayerNorm shines when:**

- Transformers / NLP
- RNNs
- Small/unstable batches
- Reinforcement learning

### 13.8 Other Normalizations

|Method|Normalizes Over|Best For|
|---|---|---|
|**BatchNorm**|Batch dim per feature|CNNs, large batches|
|**LayerNorm**|Features per sample|RNNs, Transformers|
|**InstanceNorm**|Spatial dims per sample per channel|Style transfer|
|**GroupNorm**|Groups of channels per sample|Small batches, detection|

### 13.9 Backprop Through BatchNorm

BN is differentiable but tricky because normalization depends on the whole batch. Given upstream gradient $\frac{\partial \mathcal{L}}{\partial y_{ij}}$:

$$\frac{\partial \mathcal{L}}{\partial \gamma_j} = \sum_i \frac{\partial \mathcal{L}}{\partial y_{ij}} \hat{a}_{ij}, \quad \frac{\partial \mathcal{L}}{\partial \beta_j} = \sum_i \frac{\partial \mathcal{L}}{\partial y_{ij}}$$

$\partial \mathcal{L} / \partial a_{ij}$ involves three terms (through $\hat{a}$, $\mu$, $\sigma^2$). Frameworks handle this automatically.

> **Tip:** Key conceptual point A single sample's gradient depends on all other samples in the batch through $\mu, \sigma^2$. This is BN's blessing (regularization effect) and curse (small batches → bad gradients).

### 13.10 Common Pitfalls

- **BN with batch size 1:** $\sigma^2 = 0$ → division by ε. Use LayerNorm/GroupNorm.
- **Forgetting `model.eval()`:** PyTorch BN keeps using batch stats. Catastrophic for small inference batches.
- **BN before or after activation?** Original paper: before. Modern: often after. Empirically similar.
- **Pre-LN vs Post-LN in Transformers:** Pre-LN trains more stably; Post-LN (original) needs warmup.
- **Distributed training:** Use SyncBN to synchronize stats across GPUs.
- **Fine-tuning with BN:** Frozen BN must use running stats.

### 13.11 Quick Mental Model

> **Quote:** **BatchNorm: "Look at how this feature varies across the batch, and normalize that."** **LayerNorm: "Look at how this sample's features vary, and normalize that."**

---

## 14. Vanishing & Exploding Gradients

**Vanishing:** Products of small derivatives ($<1$) shrink gradients exponentially in deep networks. Early layers don't learn.

**Exploding:** Gradients grow exponentially → NaN losses, unstable training.

**Solutions:**

- ReLU and variants (derivative = 1 for $z>0$)
- Proper initialization (He, Xavier)
- BatchNorm/LayerNorm
- **Residual connections** (ResNet): $\mathbf{a}^{(\ell)} = \sigma(\mathbf{z}^{(\ell)}) + \mathbf{a}^{(\ell-1)}$ — gradient has an identity path
- Gradient clipping: clip $|\nabla| \leq c$ (critical for RNNs)
- LSTM/GRU gating for sequential models

---

## 15. Hyperparameters

|Hyperparameter|Typical Range|Notes|
|---|---|---|
|Learning rate|$10^{-5}$ to $10^{-1}$|Most important; use LR finder/scheduling|
|Batch size|32–512|Larger = more stable, may generalize worse|
|Number of layers|2–100+|More = more capacity but harder to train|
|Hidden units/layer|64–4096|Often powers of 2|
|Optimizer|SGD+momentum, AdamW|AdamW for default, SGD+momentum for best generalization|
|Weight decay|$10^{-5}$ to $10^{-2}$||
|Dropout rate|0.1–0.5||
|Activation|ReLU/GELU||

### Learning Rate Schedules

- **Step decay:** Divide by 10 every $K$ epochs
- **Exponential:** $\eta_t = \eta_0 \cdot \gamma^t$
- **Cosine annealing:** $\eta_t = \eta_{\min} + \frac{1}{2}(\eta_0-\eta_{\min})(1+\cos(\pi t/T))$
- **Warmup:** Linearly ramp up $\eta$ for first $K$ steps (critical for Transformers)
- **One-cycle:** Warmup then anneal down

---

## 16. Universal Approximation Theorem

**Statement (Hornik, Cybenko, 1989):** A feedforward network with a single hidden layer containing finitely many neurons and a non-polynomial activation can approximate any continuous function on a compact subset of $\mathbb{R}^n$ to arbitrary accuracy.

**Caveats:**

- "Finitely many" could be **exponentially large**
- Doesn't say it's learnable via gradient descent
- Doesn't say it generalizes
- **Depth helps:** deep networks express certain functions exponentially more compactly than shallow ones

---

## 17. Strengths and Weaknesses

### Strengths

- Universal function approximators
- End-to-end learning — no manual feature engineering
- Scale extremely well with data and compute
- Transfer learning — pretrained features transfer across tasks
- Flexible architectures for any modality (CNN, RNN, Transformer, GNN)

### Weaknesses

- **Data hungry**
- **Compute hungry** — GPUs/TPUs essential
- **Black box** — limited interpretability
- **Hyperparameter sensitive**
- **Brittle** — adversarial examples, distribution shift
- **No uncertainty estimates by default**
- **Don't dominate on tabular data** — GBDTs usually win

---

## 18. When to Use vs Not Use

### Use NNs when:

- Unstructured data (images, audio, text)
- Large datasets (10k+ examples)
- Complex hierarchical features
- Transfer learning available
- End-to-end differentiable pipelines

### Don't use NNs when:

- Small datasets (< 1k samples)
- Tabular data — GBDTs typically beat NNs
- Interpretability critical
- Tight latency/memory constraints
- Need calibrated uncertainty

---

## 19. NN vs Other Algorithms

|Aspect|Neural Networks|GBDTs (XGBoost)|SVM|Logistic Regression|
|---|---|---|---|---|
|Data type|Unstructured best|Tabular best|Medium tabular|Tabular, linear|
|Sample efficiency|Low|High|Medium|Very high|
|Feature engineering|Implicit|Some manual|Heavy|Heavy|
|Interpretability|Low|Medium|Low|High|
|Training time|High|Medium|High for large $N$|Very low|
|Inference time|Medium-High|Low-Medium|Slow ($O(SV)$)|Very fast|
|Hyperparameter sensitivity|High|Medium|Medium|Low|
|Handles missing values|No (must impute)|Yes (natively)|No|No|

---

## 21. Production Considerations

- **Quantization** (FP32 → INT8) — ~4× memory reduction
- **Pruning** — remove low-magnitude weights (often >50% removable)
- **Knowledge distillation** — train small student from large teacher
- **ONNX/TorchScript/TensorRT** for deployment
- **Monitoring** — input drift, output drift, latency, throughput
- **Reproducibility** — fix seeds for `numpy`, `torch`, `random`; deterministic ops
- **Mixed precision (FP16/BF16)** — 2× speedup, minimal accuracy loss
- **Gradient accumulation** — simulate large batches on small GPUs

---

## Related Notes

- [Linear Regression](/notes/ml-algorithms/supervised-learning/linear-regression/)
- [Logistic Regression](/notes/ml-algorithms/supervised-learning/logistic-regression/)
- Bias-Variance Tradeoff
- CNNs (next)
- RNNs and LSTMs (upcoming)
- [Transformers](/notes/ml-algorithms/deep-learning/transformers/) (upcoming)
