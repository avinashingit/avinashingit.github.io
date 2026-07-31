---
layout: note
title: "Positional Encodings"
description: "Self-attention treats its input as a set, not a sequence — by itself it cannot tell \"dog bites man\" from \"man bites dog.\" Positional encodings fix this by adding or modifying to…"
note: true
note_collection: "ML algorithms"
note_section: "Language Models"
section_order: 6
note_order: 11
updated: 2026-06-07 03:55:08 -0700
keywords:
  - Transformers
  - LLMs
  - Embeddings
  - Training
  - Evaluation
math: true
mermaid: true
---
> Mechanisms that inject word-order information into transformers, whose self-attention is otherwise permutation-invariant. Related: [Attention](/notes/ml-algorithms/deep-learning/attention/), [Transformers](/notes/ml-algorithms/deep-learning/transformers/), [LLM Architecture](/notes/ml-algorithms/language-models/llm-architecture/), [Long Context](/notes/ml-algorithms/language-models/long-context/)

## TL;DR
Self-attention treats its input as a *set*, not a *sequence* — by itself it cannot tell "dog bites man" from "man bites dog." Positional encodings fix this by adding or modifying token representations so the model knows *where* each token sits. The original Transformer used fixed **sinusoidal** encodings; BERT/GPT-2 used **learned absolute** embeddings; modern LLMs (Llama, Mistral, Qwen) use **RoPE** (Rotary Position Embedding), which rotates query/key vectors so attention scores depend on *relative* position and extrapolate to longer contexts. **ALiBi** is an alternative that biases attention by token distance.

## Why it matters
Attention computes a weighted sum where the weight between token $i$ and token $j$ depends only on their content vectors via $q_i \cdot k_j$ — not on their indices. Permute the input tokens and you get the same set of outputs, just permuted. So a transformer with *no* positional signal is **permutation-invariant**: it literally cannot distinguish word orders, which destroys language understanding.

Positional encoding is the layer that makes a transformer sequence-aware. It is foundational architecture (sits right after [Tokenization](/notes/ml-algorithms/language-models/tokenization/) and embedding lookup, before the first attention block), and the *choice* of scheme directly governs how far the model can extrapolate beyond its training context — the central concern in [Long Context](/notes/ml-algorithms/language-models/long-context/) work. Getting this right is why a model trained on 4K tokens can sometimes serve 32K, and why some cannot.

## How it works

### 1. Absolute sinusoidal (original Transformer, 2017)
A fixed, parameter-free encoding added to token embeddings. For position $pos$ and embedding dimension index $i$ (with model width $d_{\text{model}}$):

$$
PE_{(pos,\,2i)} = \sin\!\left(\frac{pos}{10000^{\,2i/d_{\text{model}}}}\right), \qquad
PE_{(pos,\,2i+1)} = \cos\!\left(\frac{pos}{10000^{\,2i/d_{\text{model}}}}\right)
$$

Each dimension is a sinusoid of a different wavelength (geometric progression from $2\pi$ to $\sim 10000 \cdot 2\pi$). Intuition: like a binary clock, low dimensions tick fast, high dimensions tick slow, giving every position a unique fingerprint. A useful property: $PE_{pos+k}$ is a **linear function** of $PE_{pos}$, so the model can in principle learn to attend by relative offset. The encoding is summed into the input: $x_{\text{in}} = \text{Embed}(token) + PE_{pos}$.

### 2. Learned absolute (BERT, GPT-2)
Replace the fixed sinusoids with a trainable embedding table indexed by position: one learned vector per slot $0 \dots L_{\max}-1$. Simple and often slightly better in-distribution, but the table has a **hard maximum length** $L_{\max}$ (e.g., 512 for BERT, 1024 for GPT-2). Positions beyond $L_{\max}$ have no embedding, so the model cannot process longer sequences without retraining/resizing — no extrapolation.

### 3. Relative position ideas
Instead of "token is at absolute index 5," encode "token is 3 to the left of the query." Schemes like Shaw et al. (2018) and T5's relative bias add a learned scalar/vector keyed by the *offset* $i-j$ directly into the attention logits. This matches the inductive bias of language (meaning depends on relative arrangement) and generalizes better across positions, but the classic forms add parameters and per-layer compute.

### 4. RoPE — Rotary Position Embedding (the modern default)
RoPE encodes position by **rotating** the query and key vectors by an angle proportional to their absolute position, *before* the dot product. Pair up the $d$ dimensions into $d/2$ 2-D planes; plane $m$ rotates at frequency $\theta_m = 10000^{-2m/d}$. For a vector at position $p$, rotate plane $m$ by angle $p\,\theta_m$:

$$
\begin{pmatrix} x'_{2m} \\ x'_{2m+1} \end{pmatrix}
=
\begin{pmatrix} \cos p\theta_m & -\sin p\theta_m \\ \sin p\theta_m & \cos p\theta_m \end{pmatrix}
\begin{pmatrix} x_{2m} \\ x_{2m+1} \end{pmatrix}
$$

The magic: when you take the dot product of a rotated query at position $m$ with a rotated key at position $n$, the absolute angles **cancel** and only the difference $(m-n)$ survives:

$$
\langle R_m\,q,\; R_n\,k \rangle = g(q, k, \,m-n)
$$

So attention becomes a function of **relative** distance even though we applied an *absolute* rotation per token. RoPE is applied inside each attention head to $q$ and $k$ only (not to values), adds **zero learned parameters**, and because rotation is well-defined for *any* position, it extrapolates more gracefully than learned tables.

<pre class="mermaid">
flowchart LR
  TOK[&quot;Token embeddings&quot;] --&gt; QK[&quot;Compute Q and K per head&quot;]
  QK --&gt; ROT[&quot;Rotate Q,K by angle = position x frequency&quot;]
  ROT --&gt; DOT[&quot;Dot product Q.K&quot;]
  DOT --&gt; REL[&quot;Score depends on relative offset (m - n)&quot;]
  REL --&gt; SM[&quot;Softmax then weight Values&quot;]
</pre>
### 5. ALiBi — Attention with Linear Biases
ALiBi adds **no** positional vectors at all. Instead it adds a *penalty* to each attention logit proportional to the query–key distance, before [Softmax](/notes/ml-algorithms/core-concepts/softmax/):

$$
\text{score}(i,j) = q_i \cdot k_j - m \cdot |i - j|
$$

where $m$ is a fixed, head-specific slope (geometrically spaced across heads). Nearby tokens get little penalty; far tokens get strongly down-weighted. Because the bias is a simple linear function defined for any distance, ALiBi **extrapolates to lengths far beyond training** essentially for free — its headline strength.

## Variants / Trade-offs

| Method | Type | Extra params | Extrapolation | Used by | Notes |
|---|---|---|---|---|---|
| Sinusoidal | Absolute (added) | None | Weak | Orig. Transformer | Fixed sin/cos; elegant but rarely SOTA today |
| Learned absolute | Absolute (added) | $L_{\max}\!\times\!d$ | None (hard cap) | BERT, GPT-2 | Simple, good in-distribution; cannot exceed $L_{\max}$ |
| Relative (T5, Shaw) | Relative bias | Some | Good | T5 | Adds bias keyed by offset $i-j$ |
| **RoPE** | Rotary (on Q,K) | **None** | Good, scalable | Llama, Mistral, Qwen, GPT-NeoX | Modern default; relative via absolute rotation |
| **ALiBi** | Distance bias | None | **Excellent** | BLOOM, MPT | Cheapest long-context; slight quality trade-off |

**When to use which:** new LLM from scratch → RoPE (best quality + tooling, scalable context). Need aggressive train-short / serve-long extrapolation cheaply → ALiBi. Encoder-only / fixed length → learned absolute is fine.

## Context-length extension (RoPE scaling)
A model trained with RoPE at context $L_{\text{train}}$ degrades past it because rotation angles for unseen positions are out of distribution. Three practical fixes stretch the *same weights* to longer contexts:

- **Position Interpolation (linear / RoPE scaling):** divide positions by a factor $s = L_{\text{new}}/L_{\text{train}}$ so position $L_{\text{new}}$ maps back to a seen angle. Simple; usually needs a short fine-tune. Crushes high-frequency detail.
- **NTK-aware interpolation:** instead of scaling all frequencies equally, change the RoPE base $\theta$ so high-frequency (local) dimensions are barely touched and low-frequency (global) ones are stretched. Often works *without* fine-tuning.
- **YaRN (Yet another RoPE extensioN):** combines NTK-style per-frequency scaling with an attention-temperature correction; the strongest of the three, extending e.g. 4K → 128K with minimal fine-tuning. Used in many modern long-context releases.

These are the practical bridge between architecture and [Long Context](/notes/ml-algorithms/language-models/long-context/) serving.

## Practical considerations
- **RoPE base ($\theta$) matters:** long-context models raise the base (e.g., $10000 \to 1\text{M}$) so low frequencies span the larger window — a config knob you will see in `config.json` (`rope_theta`).
- **RoPE composes with efficiency tricks:** it lives inside the attention head, so it works cleanly with GQA (Grouped-Query Attention) and FlashAttention-2 — see [Attention Variants and Efficiency](/notes/ml-algorithms/language-models/attention-variants-and-efficiency/) and [KV Cache and Inference Optimization](/notes/ml-algorithms/language-models/kv-cache-and-inference-optimization/).
- **Where it's applied:** sinusoidal/learned encodings are added once at the input; RoPE/ALiBi act at *every* attention layer on $q,k$ (and ALiBi as a logit bias). Don't double-apply.
- **Eval the extension:** after RoPE scaling, validate with long-context probes (needle-in-a-haystack, perplexity at length). Naive linear interpolation often hurts short-context quality — measure both.
- **Defaults today (2025–26):** decoder-only LLMs ship RoPE almost universally; long-context variants use NTK/YaRN scaling.

## Related
- Foundational: [Transformers](/notes/ml-algorithms/deep-learning/transformers/) · [Attention](/notes/ml-algorithms/deep-learning/attention/) · [Softmax](/notes/ml-algorithms/core-concepts/softmax/) · [LLM Architecture](/notes/ml-algorithms/language-models/llm-architecture/)
- Siblings: [Long Context](/notes/ml-algorithms/language-models/long-context/) · [Attention Variants and Efficiency](/notes/ml-algorithms/language-models/attention-variants-and-efficiency/) · [KV Cache and Inference Optimization](/notes/ml-algorithms/language-models/kv-cache-and-inference-optimization/) · [Tokenization](/notes/ml-algorithms/language-models/tokenization/)
