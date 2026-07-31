---
layout: note
title: "Back Propagation"
description: "Backprop is just the chain rule applied systematically over a Computational Graph. Everything below builds from one neuron up to the vectorized, minibatched form."
note: true
note_collection: "ML algorithms"
note_section: "Core Concepts"
section_order: 1
note_order: 2
updated: 2026-06-06 11:30:50 -0700
keywords:
  - Optimization
  - Training
  - Deep Learning
  - Graphs
  - Probability
math: true
mermaid: false
---
> **Abstract:** One breath Run the **forward** pass, caching every $z^{(l)}$ and $a^{(l)}$. Compute the **output delta** from the loss. **Recurse** the delta backward through each layer with $W^{(l+1)\top}$ and $\odot,\sigma'(z^{(l)})$. Read off each $\partial L/\partial W^{(l)}$ and $\partial L/\partial b^{(l)}$ from the deltas and cached activations. Hand the gradients to the optimizer: $W^{(l)} \leftarrow W^{(l)} - \eta,\partial L/\partial W^{(l)}$.

Backprop is just the **chain rule** applied systematically over a Computational Graph. Everything below builds from one neuron up to the vectorized, minibatched form.

---

## 1. Mental model

The question backprop answers: _how much does each parameter nudge the loss?_ You can't differentiate $L$ w.r.t. an early weight directly — it's buried many operations deep — so you **decompose the path** and multiply the local derivatives along it. The trick that makes it efficient is computing each shared piece **once** and reusing it.

---

## 2. Single neuron — worked example

Concrete numbers: $x = 2$, $w = 1.5$, $b = -1$, target $y = 1$, sigmoid activation, squared-error loss.

```
 w=1.5 ─┐
 x=2  ──┼─→  z = wx+b ──→  a = σ(z) ──→  L = ½(a−y)²
 b=−1 ─┘        =2            =0.881         =0.007
                                              ↑
                                            y=1
```

### Forward pass

$$ \begin{aligned} z &= wx + b = (1.5)(2) + (-1) = 2 \ a &= \sigma(z) = \frac{1}{1+e^{-2}} = 0.881 \ L &= \tfrac{1}{2}(a - y)^2 = \tfrac{1}{2}(0.881 - 1)^2 = 0.007 \end{aligned} $$

### Backward pass — the chain rule

$$ \frac{\partial L}{\partial w} = \underbrace{(a-y)}_{\partial L/\partial a}; \underbrace{\sigma'(z)}_{\partial a/\partial z}; \underbrace{x}_{\partial z/\partial w} $$

Each factor is a **local gradient**, computable from the cached forward values:

$$ \begin{aligned} \frac{\partial L}{\partial a} &= a - y = -0.119 \ \frac{\partial a}{\partial z} &= \sigma'(z) = a(1-a) = (0.881)(0.119) = 0.105 \ \frac{\partial z}{\partial w} &= x = 2 \end{aligned} $$

Multiplying out:

$$ \frac{\partial L}{\partial w} = (-0.119)(0.105)(2) = -0.025 \qquad \frac{\partial L}{\partial b} = (-0.119)(0.105)(1) = -0.0125 $$

> **Note:** Sigmoid's nice property $\sigma'(z) = a(1-a)$ — the derivative is expressible in terms of the activation's own output, so no extra computation is needed once you've cached $a$.

---

## 3. Three-layer network

"3 layers" here means **3 layers of weights**: input → hidden 1 → hidden 2 → output. The input is not counted.

### Forward

$$ x \to \underbrace{z_1 = w_1 x + b_1}_{} \to a_1=\sigma(z_1) \to z_2 = w_2 a_1 + b_2 \to a_2=\sigma(z_2) \to z_3 = w_3 a_2 + b_3 \to a_3=\sigma(z_3) \to L=\tfrac12(a_3-y)^2 $$

$a_3$ is the prediction $\hat{y}$. Cache every $z_l$ and $a_l$.

### The chain just gets longer

$$ \frac{\partial L}{\partial w_1} = (a_3-y),\sigma'(z_3),w_3,\sigma'(z_2),w_2,\sigma'(z_1),x $$

### Deltas — the recursion that _is_ backprop

Define the delta as the gradient of the loss w.r.t. a layer's **pre-activation**, $\delta_l = \partial L/\partial z_l$:

$$ \begin{aligned} \delta_3 &= (a_3 - y),\sigma'(z_3) && \text{output layer} \ \delta_2 &= \delta_3, w_3, \sigma'(z_2) && \text{carry } \delta_3 \text{ back through } w_3 \ \delta_1 &= \delta_2, w_2, \sigma'(z_1) && \text{carry } \delta_2 \text{ back through } w_2 \end{aligned} $$

Each delta is _the next delta, pulled back through the weight, times this layer's local activation derivative._

### Parameter gradients (one-liners once you have the deltas)

$$ \begin{array}{lll} \dfrac{\partial L}{\partial w_3} = \delta_3, a_2 &\quad& \dfrac{\partial L}{\partial b_3} = \delta_3 \[6pt] \dfrac{\partial L}{\partial w_2} = \delta_2, a_1 &\quad& \dfrac{\partial L}{\partial b_2} = \delta_2 \[6pt] \dfrac{\partial L}{\partial w_1} = \delta_1, x &\quad& \dfrac{\partial L}{\partial b_1} = \delta_1 \end{array} $$

The gradient w.r.t. a weight is always **delta at this layer × activation coming in**. That pattern survives into the vectorized form.

> **Warning:** Vanishing gradients (a favorite follow-up) For sigmoid, $\sigma'(z) = a(1-a) \le 0.25$. In $\partial L/\partial w_1$ you multiply several such terms together (plus the weights), so the gradient reaching early layers can shrink toward zero. This is why deep sigmoid nets train poorly — and why ReLU, Residual Connections, and [Batch Normalization](/notes/ml-algorithms/core-concepts/batch-normalization/) exist. See [Vanishing and Exploding Gradients](/notes/ml-algorithms/core-concepts/vanishing-and-exploding-gradients/).

---

## 4. Vectorized notation

Real layers have many neurons: $w_l$ becomes a matrix $W^{(l)}$, scalars become vectors, and the elementwise activation derivative becomes a Hadamard product $\odot$.

### Forward ($l = 1,2,3$, with $a^{(0)} = x$)

$$ \begin{aligned} z^{(l)} &= W^{(l)} a^{(l-1)} + b^{(l)} \ a^{(l)} &= \sigma(z^{(l)}) \ L &= \tfrac{1}{2}\lVert a^{(3)} - y\rVert^2 \end{aligned} $$

### The four equations of backprop

> **Example:** Memorize these four $$ \begin{aligned} \delta^{(3)} &= (a^{(3)} - y)\odot\sigma'(z^{(3)}) && \text{(1) output delta}\ \delta^{(l)} &= \left(W^{(l+1)\top}\delta^{(l+1)}\right)\odot\sigma'(z^{(l)}) && \text{(2) recurse backward}\ \frac{\partial L}{\partial W^{(l)}} &= \delta^{(l)}\left(a^{(l-1)}\right)^{\top} && \text{(3) weight gradient}\ \frac{\partial L}{\partial b^{(l)}} &= \delta^{(l)} && \text{(4) bias gradient} \end{aligned} $$

The two transposes are the part to **understand**, not memorize:

- In (2), forward you multiply by $W^{(l+1)}$ to go _up_ a layer; going _back_ you multiply by $W^{(l+1)\top}$ — the gradient flows through the same connections in reverse.
- In (3), the outer product $\delta^{(l)}(a^{(l-1)})^{\top}$ is the matrix version of "delta × incoming activation."

### Dimension check (the fastest sanity test)

With layer $l$ having $n_l$ neurons, $W^{(l)}$ is $n_l \times n_{l-1}$:

|Quantity|Shape|
|---|---|
|$a^{(l)},\ z^{(l)},\ b^{(l)},\ \delta^{(l)}$|$n_l \times 1$|
|$W^{(l+1)\top}\delta^{(l+1)}$|$(n_l \times n_{l+1})(n_{l+1}\times 1) \to n_l \times 1$ ✓ matches $z^{(l)}$|
|$\delta^{(l)}(a^{(l-1)})^{\top}$|$(n_l \times 1)(1 \times n_{l-1}) \to n_l \times n_{l-1}$ ✓ matches $W^{(l)}$|

> **Question:** Blanking on whether it's $W^\top\delta$ or $\delta W^\top$? Write the shapes and let them dictate the only arrangement that yields an $n_l \times 1$ vector. The dimensions remove the guesswork entirely.

### Minibatch of $m$ examples

Stack examples as **columns**, so $A^{(l)}$ and $\Delta^{(l)}$ are $n_l \times m$. Only the parameter-gradient equations change — you average over the batch:

$$ \frac{\partial L}{\partial W^{(l)}} = \frac{1}{m},\Delta^{(l)}\left(A^{(l-1)}\right)^{\top} \qquad \frac{\partial L}{\partial b^{(l)}} = \frac{1}{m}\sum_{j} \delta^{(l)}_j $$

The matrix product $\Delta^{(l)}(A^{(l-1)})^{\top}$ quietly performs the sum-over-examples for the weights for free; the bias has no incoming activation to multiply, so you sum its deltas explicitly (a row-sum of $\Delta^{(l)}$ over the $m$ columns).

---

## 5. NumPy implementation

> **Info:** Convention Examples are stacked as **columns** so the code maps 1:1 to the four equations: `Z = W @ A_prev + b`. Frameworks use "batch-first" rows (`X` shaped `(m, features)`, `Z = A_prev @ W + b`) — that's just this code with everything transposed.

```python
"""Minibatch backpropagation for a 3-layer neural network, in pure NumPy."""

import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    s = sigmoid(z)
    return s * (1.0 - s)

class ThreeLayerNet:
    def __init__(self, sizes, seed=0):
        # sizes = [n_in, n_hidden1, n_hidden2, n_out]  -> exactly 3 weight layers
        assert len(sizes) == 4, "expect [n_in, n_h1, n_h2, n_out]"
        self.sizes = sizes
        rng = np.random.default_rng(seed)
        self.W, self.b = {}, {}
        for l in range(1, 4):
            n_out, n_in = sizes[l], sizes[l - 1]
            # He-style init keeps activations from saturating early
            self.W[l] = rng.standard_normal((n_out, n_in)) * np.sqrt(2.0 / n_in)
            self.b[l] = np.zeros((n_out, 1))

    def forward(self, X):
        """Returns prediction and a cache of (Z, A) needed for the backward pass."""
        A = {0: X}
        Z = {}
        for l in range(1, 4):
            Z[l] = self.W[l] @ A[l - 1] + self.b[l]
            A[l] = sigmoid(Z[l])
        return A[3], (Z, A)

    def backward(self, Y, cache):
        """Returns gradients dW, db for every layer. m = batch size = #columns."""
        Z, A = cache
        m = Y.shape[1]
        dW, db = {}, {}

        # (1) output-layer delta
        dZ = (A[3] - Y) * sigmoid_prime(Z[3])

        for l in range(3, 0, -1):
            # (3) and (4): gradients for this layer's parameters
            dW[l] = (1.0 / m) * (dZ @ A[l - 1].T)
            db[l] = (1.0 / m) * np.sum(dZ, axis=1, keepdims=True)
            # (2) propagate the delta back to the previous layer (skip when l == 1)
            if l > 1:
                dA_prev = self.W[l].T @ dZ
                dZ = dA_prev * sigmoid_prime(Z[l - 1])
        return dW, db

    def step(self, dW, db, lr):
        for l in range(1, 4):
            self.W[l] -= lr * dW[l]
            self.b[l] -= lr * db[l]

    @staticmethod
    def loss(pred, Y):
        m = Y.shape[1]
        return float(np.sum((pred - Y) ** 2) / (2 * m))

    def fit(self, X, Y, epochs=200, batch_size=32, lr=1.0, seed=0):
        rng = np.random.default_rng(seed)
        m = X.shape[1]
        for epoch in range(epochs):
            perm = rng.permutation(m)               # shuffle each epoch
            Xs, Ys = X[:, perm], Y[:, perm]
            for start in range(0, m, batch_size):    # iterate over minibatches
                end = start + batch_size
                Xb, Yb = Xs[:, start:end], Ys[:, start:end]
                pred, cache = self.forward(Xb)
                dW, db = self.backward(Yb, cache)
                self.step(dW, db, lr)
            if (epoch + 1) % 40 == 0 or epoch == 0:
                full_pred, _ = self.forward(X)
                acc = np.mean((full_pred > 0.5) == (Y > 0.5))
                print(f"epoch {epoch + 1:4d} | loss {self.loss(full_pred, Y):.4f} "
                      f"| acc {acc:.3f}")

def make_xor_data(n=800, noise=0.15, seed=1):
    """Noisy XOR: label = 1 when the two coords have opposite signs."""
    rng = np.random.default_rng(seed)
    Xrows = rng.uniform(-1, 1, size=(n, 2))
    labels = ((Xrows[:, 0] > 0) ^ (Xrows[:, 1] > 0)).astype(float)
    Xrows += rng.normal(0, noise, size=Xrows.shape)
    X = Xrows.T                       # (2, n)  features x examples
    Y = labels.reshape(1, n)          # (1, n)
    return X, Y

if __name__ == "__main__":
    X, Y = make_xor_data()
    net = ThreeLayerNet([2, 16, 16, 1], seed=0)   # 3 weight layers
    net.fit(X, Y, epochs=200, batch_size=32, lr=1.0)
```

**Sample run** (loss falling + accuracy rising confirms gradients flow correctly):

```
epoch    1 | loss 0.1252 | acc 0.552
epoch   40 | loss 0.0936 | acc 0.807
epoch   80 | loss 0.0609 | acc 0.825
epoch  120 | loss 0.0480 | acc 0.865
epoch  160 | loss 0.0408 | acc 0.890
epoch  200 | loss 0.0384 | acc 0.894
```

### Reading the code

- The backward pass is **one loop counting down** `l = 3, 2, 1`. `dW[l]`/`db[l]` come from the _current_ `dZ`; then `dZ` is overwritten with the previous layer's delta via `W[l].T @ dZ` times `sigmoid_prime(Z[l-1])`. The single reused `dZ` is the "compute delta once, reuse it" point in code.
- `1/m` is what makes it a **minibatch** gradient. `dZ @ A[l-1].T` sums the per-example outer products across the batch automatically; `np.sum(dZ, axis=1)` does the explicit bias sum.
- `forward` caches every `Z`/`A` because `backward` needs `sigmoid_prime(Z[l])` and the incoming `A[l-1]` — the memory-vs-compute tradeoff.

---

## Related

- Computational Graph
- [Gradient Descent](/notes/ml-algorithms/core-concepts/gradient-descent/)
- Chain Rule
- [Vanishing and Exploding Gradients](/notes/ml-algorithms/core-concepts/vanishing-and-exploding-gradients/)
- Activation Functions
- [Softmax](/notes/ml-algorithms/core-concepts/softmax/) · [Cross-Entropy Loss](/notes/ml-algorithms/core-concepts/cross-entropy-loss/)
