---
layout: note
title: "Normalization and Activations in LLMs"
description: "Modern LLMs stabilize training by normalizing activations inside the residual branch (pre-norm) and almost always use RMSNorm — a cheaper LayerNorm variant that drops mean-cente…"
note: true
note_collection: "ML algorithms"
note_section: "Language Models"
section_order: 6
note_order: 9
updated: 2026-06-07 03:55:24 -0700
keywords:
  - LLMs
  - Transformers
  - Optimization
  - Training
  - Deep Learning
math: true
mermaid: true
---
> The two "plumbing" choices — where/how you normalize activations and which nonlinearity the feed-forward block uses — that make deep transformers trainable and efficient. Related: [Transformers](/notes/ml-algorithms/deep-learning/transformers/) · [LLM Architecture](/notes/ml-algorithms/language-models/llm-architecture/) · [Activation Functions and Optimizers](/notes/ml-algorithms/core-concepts/activation-functions-and-optimizers/) · [Batch Normalization](/notes/ml-algorithms/core-concepts/batch-normalization/)

## TL;DR

Modern LLMs stabilize training by normalizing activations *inside* the residual branch (**pre-norm**) and almost always use **RMSNorm** — a cheaper LayerNorm variant that drops mean-centering and bias. In the feed-forward network (FFN), the field moved from ReLU to the smooth **GELU**, and now to gated units like **SwiGLU** (a Swish-gated linear unit). Together these give stable optimization of 100+ layer stacks and a few points of perplexity for almost no extra parameters.

## Why it matters

A transformer is a tall stack of residual blocks. Each block adds attention and FFN outputs back into a shared residual stream. Without normalization, activation magnitudes drift and compound across dozens of layers, the loss landscape becomes ill-conditioned, gradients explode or vanish, and training diverges. **Normalization controls the scale and distribution of activations at each layer**, keeping the signal in a regime where gradient descent ([Gradient Descent](/notes/ml-algorithms/core-concepts/gradient-descent/)) behaves well. It is what lets you scale depth.

The **activation function** in the FFN is where the model spends most of its parameters and a large share of its nonlinear "thinking" capacity. The FFN is two linear maps with a nonlinearity between them; the choice of nonlinearity directly affects how expressive and trainable that block is. Picking a better activation is a near-free quality win, so it is a heavily tuned design knob.

Both choices are about **trainability and efficiency at scale** rather than headline features — but they are the difference between a model that converges and one that does not.

## How it works

### LayerNorm (recap and contrast with BatchNorm)

Given an activation vector $x \in \mathbb{R}^d$ for a single token, LayerNorm normalizes across the **feature dimension**:

$$\mu = \frac{1}{d}\sum_{i=1}^{d} x_i, \qquad \sigma^2 = \frac{1}{d}\sum_{i=1}^{d}(x_i - \mu)^2$$

$$\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

where $\gamma, \beta \in \mathbb{R}^d$ are learned gain and bias, $\epsilon$ is a small constant (e.g. $10^{-5}$) for numerical stability, and $\odot$ is elementwise multiply. Crucially, statistics are computed **per token over its own features** — no dependence on other tokens or the batch.

This is the key contrast with **BatchNorm** (see [Batch Normalization](/notes/ml-algorithms/core-concepts/batch-normalization/)), which normalizes each feature across the *batch* dimension. BatchNorm is a poor fit for transformers: sequence lengths vary, batch statistics are noisy with small/variable batches, and it couples examples together — bad for autoregressive decoding where you generate one token at a time. LayerNorm's per-token statistics are batch-independent, so train and inference behave identically and a single token can be normalized on its own.

### RMSNorm

**RMSNorm** (Root-Mean-Square Norm) observes that the re-centering (subtracting $\mu$) and the bias $\beta$ contribute little, so it drops both and only rescales by the root-mean-square:

$$\text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}, \qquad \text{RMSNorm}(x) = \gamma \odot \frac{x}{\text{RMS}(x)}$$

Only the gain $\gamma$ is learned. This removes the mean/variance subtraction (fewer reductions, no centering), making it cheaper and lower-latency while matching LayerNorm quality. It is the default in **Llama, T5, Mistral, Gemma, Qwen** and most modern open models.

### Pre-Norm vs Post-Norm

The other question is *where* the norm sits relative to the residual add. The original "Attention Is All You Need" transformer used **post-norm**: normalize *after* adding the sublayer output back to the residual stream. Modern LLMs use **pre-norm**: normalize the input to each sublayer, *inside* the residual branch, leaving a clean identity path.

$$\text{Post-Norm: } x_{l+1} = \text{Norm}\big(x_l + \text{Sublayer}(x_l)\big)$$
$$\text{Pre-Norm: } x_{l+1} = x_l + \text{Sublayer}\big(\text{Norm}(x_l)\big)$$

In pre-norm the residual stream is never squashed by a norm, so gradients flow straight from the loss to every layer through the identity path. This is dramatically more stable for deep stacks: you can train 50–100+ layers without warmup tricks or learning-rate babysitting. Post-norm can reach slightly better final quality when it trains, but is fragile at depth. Many recent models (e.g. Gemma 2) even add an extra norm on the *output* of each branch ("sandwich"/double norm) for extra stability.

<pre class="mermaid">
flowchart TD
  subgraph PRE[&quot;Pre-Norm block (modern)&quot;]
    A1[&quot;residual x_l&quot;] --&gt; N1[&quot;RMSNorm&quot;]
    N1 --&gt; S1[&quot;Attention / FFN&quot;]
    A1 --&gt; ADD1[&quot;+&quot;]
    S1 --&gt; ADD1
    ADD1 --&gt; O1[&quot;x_l+1 (clean residual path)&quot;]
  end
  subgraph POST[&quot;Post-Norm block (original)&quot;]
    A2[&quot;residual x_l&quot;] --&gt; S2[&quot;Attention / FFN&quot;]
    A2 --&gt; ADD2[&quot;+&quot;]
    S2 --&gt; ADD2
    ADD2 --&gt; N2[&quot;LayerNorm&quot;]
    N2 --&gt; O2[&quot;x_l+1 (norm in main path)&quot;]
  end
</pre>
### Activations in the FFN

The FFN (a.k.a. MLP) applied per token is, in its classic form:

$$\text{FFN}(x) = W_2\,\phi(W_1 x + b_1) + b_2$$

with $W_1 \in \mathbb{R}^{d_{ff}\times d}$ expanding to a hidden width $d_{ff}$, $\phi$ a nonlinearity, and $W_2$ projecting back. The progression of $\phi$:

- **ReLU** $\;\phi(z)=\max(0,z)$ — original transformer; simple but hard zero for negatives kills gradient there.
- **GELU** (Gaussian Error Linear Unit) — smooth gating by the Gaussian CDF $\Phi$:
$$\text{GELU}(z) = z \cdot \Phi(z) \approx 0.5\,z\left(1 + \tanh\!\big[\sqrt{2/\pi}\,(z + 0.044715 z^3)\big]\right)$$
Used in BERT and GPT-2/3. The smoothness gives better gradients than ReLU.

- **GLU family / SwiGLU** — a **Gated Linear Unit** splits the projection into two: a *content* path and a *gate* path, and multiplies them elementwise. With the **Swish/SiLU** gate $\text{Swish}(z)=z\cdot\sigma(z)$ ($\sigma$ = sigmoid), this is **SwiGLU**:
$$\text{SwiGLU}(x) = \big(\text{Swish}(W_g x)\big) \odot (W_u x), \qquad \text{FFN}_{\text{SwiGLU}}(x) = W_2\big[\text{Swish}(W_g x) \odot (W_u x)\big]$$

The gate lets the network *learn what to pass through* per channel, a multiplicative interaction that a single nonlinearity cannot express. SwiGLU consistently lowers perplexity and is used in **Llama, PaLM, Mistral, Qwen**.

### FFN expansion ratio

The classic FFN uses $d_{ff} \approx 4d$. SwiGLU adds a third weight matrix ($W_g, W_u, W_2$ vs $W_1, W_2$), so to keep parameter count constant the hidden width is shrunk to about $d_{ff} \approx \tfrac{8}{3}d \approx 2.67d$. That is why Llama reports FFN sizes like 11008 for $d=4096$ ($\approx 2.7\times$): same params, better activation.

## Variants / Trade-offs

| Choice | Formula core | Learned params | Cost | Used in | When to use |
|---|---|---|---|---|---|
| **BatchNorm** | normalize over batch | $\gamma,\beta$ | sync stats | CNNs ([Convolutional Neural Networks](/notes/ml-algorithms/deep-learning/convolutional-neural-networks/)) | not for transformers |
| **LayerNorm** | center + scale per token | $\gamma,\beta$ | 2 reductions | GPT-2/3, BERT | safe default, classic |
| **RMSNorm** | scale-only per token | $\gamma$ | 1 reduction, no centering | Llama, T5, Mistral | modern default; cheaper |
| **Post-Norm** | norm after residual add | — | — | original transformer | shallow / max quality |
| **Pre-Norm** | norm inside residual branch | — | — | all modern LLMs | deep stacks, stable training |
| **ReLU** | $\max(0,z)$ | — | cheapest | early models | legacy / latency-critical |
| **GELU** | $z\,\Phi(z)$ | — | cheap | BERT, GPT-2/3 | strong non-gated default |
| **SwiGLU** | $\text{Swish}(W_gx)\odot W_ux$ | extra $W_g$ | +1 matmul, $\tfrac83 d$ width | Llama, PaLM, Mistral | best quality/param today |

## Practical considerations

- **Defaults today:** pre-norm + RMSNorm + SwiGLU FFN at $\approx\tfrac83 d$ width is the de facto modern recipe (Llama-style).
- **Numerical precision:** compute norms in **FP32** even in mixed/BF16 training — the sum of squares can overflow or lose precision in FP16/BF16. The reduction is a known target for fused kernels (FlashAttention-adjacent) since it is memory-bound.
- **Keep $\epsilon$ consistent** between train and inference; mismatched $\epsilon$ silently degrades a loaded checkpoint.
- **No bias is the trend:** many models drop biases on FFN/attention projections and norms entirely — fewer params, negligible quality loss, simpler.
- **Stability at extreme depth:** very large models may add QK-norm (normalizing query/key) or sandwich norms to tame attention-logit blow-up; pre-norm alone can still grow the residual stream's scale across depth.
- **SwiGLU bookkeeping:** remember the $3$-matrix structure when counting params or sizing the hidden dim, or you will overshoot the parameter budget by ~50%.

## Related

- [LLM Architecture](/notes/ml-algorithms/language-models/llm-architecture/) · [Transformers](/notes/ml-algorithms/deep-learning/transformers/) · [Attention](/notes/ml-algorithms/deep-learning/attention/)
- [Activation Functions and Optimizers](/notes/ml-algorithms/core-concepts/activation-functions-and-optimizers/) · [Batch Normalization](/notes/ml-algorithms/core-concepts/batch-normalization/) · [Gradient Descent](/notes/ml-algorithms/core-concepts/gradient-descent/)
- [Positional Encodings](/notes/ml-algorithms/language-models/positional-encodings/) · [Attention Variants and Efficiency](/notes/ml-algorithms/language-models/attention-variants-and-efficiency/) · [Mixture of Experts](/notes/ml-algorithms/language-models/mixture-of-experts/)
