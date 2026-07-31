---
layout: note
title: "LLM Architecture"
description: "A modern LLM (GPT, Llama, Claude-style) is a decoder-only transformer: tokens are embedded, position information is added, and the sequence passes through $N$ identical decoder…"
note: true
note_collection: "ML algorithms"
note_section: "Language Models"
section_order: 6
note_order: 5
updated: 2026-06-07 03:55:11 -0700
keywords:
  - LLMs
  - Transformers
  - Embeddings
  - Evaluation
  - Graphs
math: true
mermaid: true
---
> The decoder-only transformer that powers modern LLMs: stacked masked self-attention + FFN blocks predicting the next token. Related: [Transformers](/notes/ml-algorithms/deep-learning/transformers/), [Attention](/notes/ml-algorithms/deep-learning/attention/), [Positional Encodings](/notes/ml-algorithms/language-models/positional-encodings/), [Softmax](/notes/ml-algorithms/core-concepts/softmax/)

## TL;DR

A modern LLM (GPT, Llama, Claude-style) is a **decoder-only transformer**: tokens are embedded, position information is added, and the sequence passes through $N$ identical decoder blocks — each a **masked (causal) multi-head self-attention** sublayer followed by a **position-wise feed-forward network (FFN)**, both wrapped in residual connections with pre-normalization. A final norm and a linear **LM head** (usually weight-tied to the embeddings) produce logits, and [Softmax](/notes/ml-algorithms/core-concepts/softmax/) turns them into a next-token distribution. The model is trained to predict the next token and generates **autoregressively** — one token at a time, feeding each output back as input.

## Why it matters

## How it works

### The forward stack

For input tokens $x_1,\dots,x_T$ (integer IDs from [Tokenization](/notes/ml-algorithms/language-models/tokenization/)):

1. **Token embedding.** A lookup table $E \in \mathbb{R}^{V \times d}$ maps each ID to a $d$-dimensional vector ($V$ = vocab size, $d$ = model/hidden dimension). Output: $H^{(0)} \in \mathbb{R}^{T \times d}$.
2. **Positional information.** Transformers are permutation-invariant, so position must be injected — either added to the embeddings (learned/sinusoidal) or applied inside attention (RoPE — rotary position embeddings, now dominant). See [Positional Encodings](/notes/ml-algorithms/language-models/positional-encodings/).
3. **$N$ decoder blocks.** Each updates the hidden states: $H^{(\ell)} = \text{Block}_\ell(H^{(\ell-1)})$.
4. **Final norm.** A last LayerNorm/RMSNorm stabilizes the output (see [Normalization and Activations in LLMs](/notes/ml-algorithms/language-models/normalization-and-activations-in-llms/)).
5. **LM head.** A linear map to vocab logits $z \in \mathbb{R}^{T \times V}$. Commonly **weight-tied**: the head reuses $E^\top$, saving $V\cdot d$ parameters and improving quality.
6. **Softmax.** $p(\text{next token}) = \text{softmax}(z_t)$ over the vocabulary (see [Softmax](/notes/ml-algorithms/core-concepts/softmax/)).

### Inside a decoder block (pre-norm)

With pre-norm (the modern default — more stable than the original post-norm):

$$
\begin{aligned}
A &= H + \text{MHA}\big(\text{Norm}(H)\big) \\
H' &= A + \text{FFN}\big(\text{Norm}(A)\big)
\end{aligned}
$$

- **MHA = masked multi-head self-attention.** Project $\text{Norm}(H)$ into queries $Q$, keys $K$, values $V$ across $h$ heads, then per head:
$$
\text{Attn}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V
$$
where $d_k = d/h$ and $M$ is the **causal mask**: $M_{ij} = 0$ if $j \le i$ and $M_{ij} = -\infty$ if $j > i$. The $-\infty$ entries become $0$ after softmax, so token $i$ attends only to itself and earlier tokens — never the future. See [Attention](/notes/ml-algorithms/deep-learning/attention/).
- **FFN.** A two-layer MLP applied per position, typically expanding to $4d$ (or using a gated variant like SwiGLU with $\approx \frac{8}{3}d$): $\text{FFN}(x) = W_2\,\phi(W_1 x)$. This is where most non-attention parameters live.
- **Residual connections** ($H + \dots$) give gradients a clean path and let the network refine a "residual stream" incrementally.

### Why causal masking

The task is next-token prediction. If position $i$ could see token $i{+}1$, predicting $i{+}1$ would be trivial (look it up) — the model would learn nothing and would be useless at generation time, where the future doesn't exist yet. The mask enforces that the prediction at every position is a *causal* function of the past only. This is what lets training compute a loss at all $T$ positions in **one parallel forward pass**, while inference stays consistent with that training objective.

### Autoregressive generation

Generation is a loop: feed the prompt, take the distribution at the last position, **sample/argmax** a token, append it, and repeat. The expensive part is recomputation — which is why production serving keeps a **KV cache** (see [KV Cache and Inference Optimization](/notes/ml-algorithms/language-models/kv-cache-and-inference-optimization/)) so prior keys/values are reused and each step is $O(T)$ instead of $O(T^2)$. Sampling strategy (greedy, top-k, top-p, temperature) is covered in [Decoding Strategies](/notes/ml-algorithms/language-models/decoding-strategies/).

<pre class="mermaid">
flowchart TD
  subgraph STACK[&quot;Decoder-only LLM (overall stack)&quot;]
    T[&quot;Token IDs&quot;] --&gt; EMB[&quot;Token embedding E&quot;]
    EMB --&gt; POS[&quot;Add positional info&quot;]
    POS --&gt; B1[&quot;Decoder block 1&quot;]
    B1 --&gt; BD[&quot;... N decoder blocks ...&quot;]
    BD --&gt; BN[&quot;Decoder block N&quot;]
    BN --&gt; FN[&quot;Final norm&quot;]
    FN --&gt; HEAD[&quot;LM head (tied to E)&quot;]
    HEAD --&gt; SM[&quot;Softmax over vocab&quot;]
    SM --&gt; NXT[&quot;Next-token distribution&quot;]
  end
  NXT -. &quot;sample and append (autoregressive)&quot; .-&gt; T
</pre>
<pre class="mermaid">
flowchart TD
  subgraph BLK[&quot;Inside one decoder block (pre-norm)&quot;]
    IN[&quot;Hidden states H&quot;] --&gt; N1[&quot;Norm&quot;]
    N1 --&gt; MHA[&quot;Masked multi-head self-attention&quot;]
    IN --&gt; R1[&quot;Residual add&quot;]
    MHA --&gt; R1
    R1 --&gt; N2[&quot;Norm&quot;]
    N2 --&gt; FFN[&quot;Position-wise FFN&quot;]
    R1 --&gt; R2[&quot;Residual add&quot;]
    FFN --&gt; R2
    R2 --&gt; OUT[&quot;Updated hidden states&quot;]
  end
</pre>
## Transformer families and trade-offs

The original 2017 transformer had both an encoder and a decoder. Three families specialized from it:

| Family | Examples | Attention | Best at | Why / limitation |
|---|---|---|---|---|
| **Encoder-only** | BERT, RoBERTa | Bidirectional (no mask) | Understanding: classification, NER, embeddings | Sees full context both ways; cannot generate text autoregressively |
| **Decoder-only** | GPT, Llama, Claude, Mistral | Causal (masked) | Generation, chat, in-context learning, reasoning | Only sees the past; ideal for next-token prediction; quadratic attention |
| **Encoder-decoder** | T5, BART, original NMT | Bidirectional encoder + causal decoder w/ cross-attention | Seq2seq: translation, summarization with a fixed input | More params/complexity; rigid input/output split |

**Why decoder-only won for general LLMs.** (1) The next-token objective is a *universal* task — translation, Q&A, summarization, and reasoning all become "continue this text," so one model handles everything via prompting (**in-context learning**). (2) It is the simplest architecture (no separate encoder, no cross-attention) and scales cleanly — and the [Scaling Laws](/notes/ml-algorithms/language-models/scaling-laws/) that drove progress were measured on decoder-only LMs. (3) Training is maximally efficient: every token in a document is a supervised label in a single pass. Encoder-decoder still wins when there is a clean fixed input to encode bidirectionally (e.g., a sentence to translate), but for open-ended generation the decoder-only stack is simpler and more general.

### Rough parameter-count math

For hidden dim $d$, $N$ layers, FFN expansion $4d$, vocab $V$:

- **Embeddings:** $V \cdot d$ (shared with the LM head if tied).
- **Per layer — attention:** four $d \times d$ projections ($Q,K,V,O$) $= 4d^2$.
- **Per layer — FFN:** $W_1 \in \mathbb{R}^{4d \times d}$, $W_2 \in \mathbb{R}^{d \times 4d}$ $= 8d^2$.
- **Per layer total:** $\approx 12d^2$ (norm/bias terms are negligible).
- **Total:** $P \approx 12 N d^2 + V d$.

Sanity check on GPT-3 ($N{=}96$, $d{=}12288$): $12 \cdot 96 \cdot 12288^2 \approx 1.74\times10^{11}$ — about 174B, near the stated 175B once embeddings are added. The FFN ($8d^2$) holds **two-thirds** of each layer's weights, which is why Mixture-of-Experts targets the FFN to grow capacity cheaply (see [Mixture of Experts](/notes/ml-algorithms/language-models/mixture-of-experts/)).

## Practical considerations

- **Context is quadratic.** Attention is $O(T^2 d)$ in compute and the KV cache grows $O(T)$ in memory per layer — the main reason long context is hard (see [Long Context](/notes/ml-algorithms/language-models/long-context/), [Attention Variants and Efficiency](/notes/ml-algorithms/language-models/attention-variants-and-efficiency/)).
- **Modern defaults differ from the 2017 paper.** Pre-norm + **RMSNorm**, **RoPE** instead of additive positions, **SwiGLU** FFNs, no biases, and **GQA** (grouped-query attention — fewer KV heads to shrink the cache) are standard in Llama/Mistral-class models.
- **Weight tying** is near-universal in open models; it saves $V \cdot d$ params (large when $V \approx 128$k) and regularizes the output.
- **Memory at inference is dominated by the KV cache, not weights**, for long sequences — driving GQA, paged attention, and quantization (see [KV Cache and Inference Optimization](/notes/ml-algorithms/language-models/kv-cache-and-inference-optimization/), [Quantization for LLMs](/notes/ml-algorithms/language-models/quantization-for-llms/)).
- **The residual stream is $d$-wide throughout.** Every sublayer reads from and writes to the same $d$-dimensional stream — a useful frame for interpretability and for reasoning about where information flows.
- **Numerical stability:** FP8/BF16 training, attention in higher precision, and FlashAttention-2 (an IO-aware exact attention kernel) to avoid materializing the $T \times T$ matrix.

## Related

- Components: [Attention](/notes/ml-algorithms/deep-learning/attention/) · [Positional Encodings](/notes/ml-algorithms/language-models/positional-encodings/) · [Normalization and Activations in LLMs](/notes/ml-algorithms/language-models/normalization-and-activations-in-llms/) · [Softmax](/notes/ml-algorithms/core-concepts/softmax/) · [Tokenization](/notes/ml-algorithms/language-models/tokenization/) · [Transformers](/notes/ml-algorithms/deep-learning/transformers/)
- Scaling & variants: [Attention Variants and Efficiency](/notes/ml-algorithms/language-models/attention-variants-and-efficiency/) · [Mixture of Experts](/notes/ml-algorithms/language-models/mixture-of-experts/) · [Scaling Laws](/notes/ml-algorithms/language-models/scaling-laws/)
- Training & inference: [Pretraining and Language Modeling](/notes/ml-algorithms/language-models/pretraining-and-language-modeling/) · [KV Cache and Inference Optimization](/notes/ml-algorithms/language-models/kv-cache-and-inference-optimization/) · [Decoding Strategies](/notes/ml-algorithms/language-models/decoding-strategies/) · [Long Context](/notes/ml-algorithms/language-models/long-context/)
