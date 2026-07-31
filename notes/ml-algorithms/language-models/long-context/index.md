---
layout: note
title: "Long Context"
description: "A model's context window is the maximum number of tokens it can process in one forward pass. It grew from ~2k (GPT-3) to 128k–1M+ in modern models. Long context is hard because…"
note: true
note_collection: "ML algorithms"
note_section: "Language Models"
section_order: 6
note_order: 7
updated: 2026-06-07 03:59:06 -0700
keywords:
  - LLMs
  - Transformers
  - Inference
  - Optimization
  - Retrieval
math: true
mermaid: true
---
> The set of techniques that let an LLM attend to very long input sequences (128k–1M+ tokens) despite the quadratic cost of attention. Related: [Attention Variants and Efficiency](/notes/ml-algorithms/language-models/attention-variants-and-efficiency/), [KV Cache and Inference Optimization](/notes/ml-algorithms/language-models/kv-cache-and-inference-optimization/), [Positional Encodings](/notes/ml-algorithms/language-models/positional-encodings/), [Retrieval-Augmented Generation](/notes/ml-algorithms/language-models/retrieval-augmented-generation/)

## TL;DR

A model's **context window** is the maximum number of tokens it can process in one forward pass. It grew from ~2k (GPT-3) to 128k–1M+ in modern models. Long context is hard because self-attention is $O(n^2)$ in compute and the **KV-cache** grows linearly in memory with sequence length $n$. We extend context with positional tricks (RoPE scaling, NTK-aware, YaRN) plus continued training, and make it affordable with efficient attention (FlashAttention, sliding-window/sparse) and smaller KV (GQA/MQA). Even when it "fits," models suffer **lost in the middle** — they use the start and end of the prompt far better than the middle.

## Why it matters

The context window is the model's working memory: everything it can "see" — system prompt, conversation history, retrieved documents, code files, the user's question — must fit inside it. A 2k window can hold roughly 3 pages; a 1M window can hold a small codebase or a full book. Longer context unlocks whole-document QA, multi-file code reasoning, long agentic trajectories, and many-shot in-context learning without fine-tuning.

But context isn't free. Doubling the input more than doubles attention cost and linearly inflates the memory you must hold during generation. So "long context" is really two problems stitched together: (1) can the model *attend* correctly over long sequences (a positional/architecture question), and (2) can you *afford* to serve it (a systems question). It also competes directly with [Retrieval-Augmented Generation](/notes/ml-algorithms/language-models/retrieval-augmented-generation/): do you stuff everything into the prompt, or retrieve only the relevant chunks?

## How it works

### Why it's hard: $O(n^2)$ compute and linear KV memory

Self-attention compares every token to every other token. For sequence length $n$ and head dimension $d$, scores $QK^\top$ form an $n \times n$ matrix:

$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right)V \quad\Rightarrow\quad O(n^2 d) \text{ compute}, \; O(n^2) \text{ attention memory}$$

Going from 4k to 128k tokens is a 32× length increase but a ~1000× increase in raw attention FLOPs. During autoregressive generation we also cache the keys and values of every past token (the **KV-cache**) so we don't recompute them. KV memory grows linearly:

$$\text{KV bytes} = 2 \times n \times L \times H_{kv} \times d \times \text{bytes/elt}$$

where $L$ = layers, $H_{kv}$ = key/value heads, $d$ = head dim. At 128k tokens this is often *tens of GB* — frequently larger than the model weights themselves. See [KV Cache and Inference Optimization](/notes/ml-algorithms/language-models/kv-cache-and-inference-optimization/).

### Context-length extension

Most models are trained at a modest length (say 4k–8k) and then *extended*. The bottleneck is usually positional encoding: a model that never saw position 50,000 generalizes poorly to it. With **RoPE** (Rotary Position Embedding, see [Positional Encodings](/notes/ml-algorithms/language-models/positional-encodings/)), each position rotates query/key vectors by an angle proportional to position, so extension means rescaling those angles:

| Technique | Idea | Cost / quality |
|---|---|---|
| **Position Interpolation (PI)** | Linearly squeeze positions so $n_{new}$ maps into the trained range | Cheap; some fine-tuning needed; degrades high-freq detail |
| **NTK-aware scaling** | Scale RoPE base $\theta$ so high frequencies are preserved, low ones stretched | Often works *training-free* for moderate extension |
| **YaRN** | Per-frequency interpolation + attention temperature; best quality | Needs a little continued training; widely used |
| **Continued long-context training** | Keep training on real long documents at the target length | Most robust; expensive (data + compute) |

In practice teams combine a positional method with **continued pretraining** on long sequences (and long-context instruction data) so the model actually *learns* to use far-away tokens, not just tolerate the positions.

### Architectural help (making long context affordable)

<pre class="mermaid">
flowchart TD
    N1[&quot;Long input (n tokens)&quot;] --&gt; N2[&quot;FlashAttention (exact, IO-aware)&quot;]
    N1 --&gt; N3[&quot;Sliding-window / sparse attention&quot;]
    N1 --&gt; N4[&quot;GQA / MQA (fewer KV heads)&quot;]
    N2 --&gt; N5[&quot;Same result, less memory traffic&quot;]
    N3 --&gt; N6[&quot;Each token attends to a local band&quot;]
    N4 --&gt; N7[&quot;Smaller KV-cache&quot;]
    N5 --&gt; N8[&quot;Affordable long-context inference&quot;]
    N6 --&gt; N8
    N7 --&gt; N8
</pre>
- **FlashAttention** (and FlashAttention-2): computes *exact* attention but tiles the computation in SRAM so it never materializes the full $n \times n$ matrix in slow memory. Same math, far less memory traffic — a key enabler of long context. See [Attention Variants and Efficiency](/notes/ml-algorithms/language-models/attention-variants-and-efficiency/).
- **Sliding-window / sparse attention**: each token attends only to a local band (e.g. 4k neighbors) instead of all $n$, turning the cost roughly linear; combined with a few global tokens it preserves most quality.
- **GQA / MQA** (Grouped-Query / Multi-Query Attention): share key/value heads across query heads, shrinking $H_{kv}$ and therefore the KV-cache — the single biggest lever for long-context serving memory. See [KV Cache and Inference Optimization](/notes/ml-algorithms/language-models/kv-cache-and-inference-optimization/).

### Lost in the middle

Even with a perfectly fitting window, accuracy is **U-shaped** in the position of the relevant fact: models recall information at the **start** and **end** of the context well and the **middle** poorly. Causes include positional biases (recency from causal attention; primacy from training distribution) and dilution of attention mass over many tokens. Practical implication: put the most important instructions and evidence at the top and bottom, not buried in the middle.

<pre class="mermaid">
flowchart LR
    A[&quot;Start of context&quot;] --&gt;|&quot;high recall&quot;| Q[&quot;Answer quality&quot;]
    M[&quot;Middle of context&quot;] --&gt;|&quot;low recall (U-shaped dip)&quot;| Q
    E[&quot;End of context&quot;] --&gt;|&quot;high recall&quot;| Q
</pre>
## Variants / Trade-offs: Long Context vs RAG

The central design choice is whether to **stuff** everything into the prompt or **retrieve** only what's needed.

| Dimension | Long context (stuff it all) | [Retrieval-Augmented Generation](/notes/ml-algorithms/language-models/retrieval-augmented-generation/) (retrieve chunks) |
|---|---|---|
| **Cost** | Pay for *every* token each call; can be huge | Pay only for top-$k$ retrieved chunks |
| **Latency** | Grows with input length (prefill is $O(n^2)$) | Low — short prompt after retrieval |
| **Recall vs precision** | High recall (nothing left out) but "lost in the middle" hurts | High precision if retriever is good; misses what it fails to fetch |
| **Freshness** | Limited by what you paste in | Easy to update the index without retraining |
| **Engineering** | Simple — no retriever, no chunking | Needs embeddings, vector store, chunking, reranking |
| **Failure mode** | Distraction by irrelevant context; quadratic cost | Bad retrieval = missing evidence = confident wrong answer |

**Rule of thumb:** RAG for large, changing knowledge bases (cheaper, scalable, fresh); long context for self-contained tasks where you genuinely need *all* the tokens at once (a single long contract, a whole code module). Hybrids are common: retrieve to fill a long window with the *best* chunks. Prompt caching makes a large, static prefix cheap to reuse across calls.

## Practical considerations

- **Prefill vs decode.** Long prompts dominate the *prefill* (processing the input) — that's the $O(n^2)$ part and the latency you feel as "time to first token." Generation (decode) is memory-bandwidth-bound on the KV-cache.
- **KV-cache is the real constraint.** At long context, KV memory often exceeds weights. Mitigate with GQA/MQA, KV quantization (e.g. INT8/INT4 KV), and paged attention (vLLM) to avoid fragmentation.
- **Advertised vs effective context.** A model may *accept* 1M tokens but degrade well before that. Always validate on *your* task length, not the spec sheet.
- **Place key info at the edges.** Because of lost-in-the-middle, put critical instructions/evidence near the start and end; consider reordering or reranking retrieved chunks.
- **Cost discipline.** Token cost scales with input length on every call — a 200k-token prompt repeated per turn is expensive. Use prompt/KV caching for stable prefixes and trim history.
- **FP8/INT4 weights** plus FlashAttention are standard for serving long context efficiently; see [Quantization for LLMs](/notes/ml-algorithms/language-models/quantization-for-llms/).

## Related

- [Positional Encodings](/notes/ml-algorithms/language-models/positional-encodings/) — RoPE and the scaling methods used to extend context
- [Attention Variants and Efficiency](/notes/ml-algorithms/language-models/attention-variants-and-efficiency/) — FlashAttention, sliding-window/sparse attention
- [KV Cache and Inference Optimization](/notes/ml-algorithms/language-models/kv-cache-and-inference-optimization/) — GQA/MQA, KV quantization, paged attention
- [Retrieval-Augmented Generation](/notes/ml-algorithms/language-models/retrieval-augmented-generation/) — the main alternative/complement to long context
- [Quantization for LLMs](/notes/ml-algorithms/language-models/quantization-for-llms/) — FP8/INT4 weights and KV quantization for serving
- [Decoding Strategies](/notes/ml-algorithms/language-models/decoding-strategies/) · [Attention](/notes/ml-algorithms/deep-learning/attention/) · [Transformers](/notes/ml-algorithms/deep-learning/transformers/)
