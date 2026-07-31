---
layout: note
title: "Attention Variants and Efficiency"
description: "Standard multi-head attention (MHA) costs $O(n^2 d)$ compute and, at inference, a KV cache that grows with the number of heads. Multi-Query Attention (MQA) lets all query heads…"
note: true
note_collection: "ML algorithms"
note_section: "Language Models"
section_order: 6
note_order: 1
updated: 2026-06-07 03:55:14 -0700
keywords:
  - Transformers
  - LLMs
  - Inference
  - Optimization
  - Training
math: true
mermaid: true
---
> Techniques that make self-attention cheaper in compute and memory — by sharing K/V heads (MQA/GQA), tiling the kernel (FlashAttention), or sparsifying the pattern (sliding window) — without (mostly) changing the model's expressiveness. Related: [Attention](/notes/ml-algorithms/deep-learning/attention/), [KV Cache and Inference Optimization](/notes/ml-algorithms/language-models/kv-cache-and-inference-optimization/), [Long Context](/notes/ml-algorithms/language-models/long-context/), [Transformers](/notes/ml-algorithms/deep-learning/transformers/)

## TL;DR

Standard multi-head attention (MHA) costs $O(n^2 d)$ compute and, at inference, a KV cache that grows with the number of heads. **Multi-Query Attention (MQA)** lets all query heads share one K/V head, **Grouped-Query Attention (GQA)** lets groups of query heads share K/V heads — both shrink the KV cache and speed up decoding with little quality loss (GQA is the modern default: Llama-2/3, Mistral). **FlashAttention** is an IO-aware *exact* kernel that tiles attention in fast SRAM and never materializes the $n \times n$ matrix, cutting memory from $O(n^2)$ to $O(n)$ and running much faster. **Sparse/sliding-window** attention drops the cost to $O(n)$ by attending only locally, trading some global context.

## Why it matters

Attention is the bottleneck of the Transformer at scale. Two distinct costs hurt in different regimes:

- **Training / prefill (compute-bound):** the $n \times n$ score matrix means quadratic FLOPs and, naively, quadratic memory in sequence length $n$. This is what makes 32k+ context expensive.
- **Decoding / serving (memory-bandwidth-bound):** during autoregressive generation, each new token attends to all previous keys/values, so we cache them (the **KV cache**). The cache size scales with `layers × heads × head_dim × seq_len × batch`. For long sequences and high batch sizes this dominates GPU memory and, because every decode step must *read the whole cache*, it dominates latency too. See [KV Cache and Inference Optimization](/notes/ml-algorithms/language-models/kv-cache-and-inference-optimization/).

## How it works

**Baseline MHA.** For $h$ heads, project the input into per-head queries, keys, values $Q_i, K_i, V_i \in \mathbb{R}^{n \times d_k}$ and compute

$$\text{Attn}(Q_i,K_i,V_i) = \text{softmax}\!\left(\frac{Q_i K_i^\top}{\sqrt{d_k}}\right) V_i$$

where $n$ = sequence length, $d_k$ = per-head dimension, $h$ = number of heads. The KV cache must store $K_i, V_i$ for **all $h$ heads** — that is the memory MQA/GQA target.

**MQA / GQA — sharing K/V heads.** Keep $h$ *query* heads but reduce the number of *key/value* heads to $g$:

- MHA: $g = h$ (every query head has its own K/V).
- GQA: $1 < g < h$ (each K/V head is shared by $h/g$ query heads).
- MQA: $g = 1$ (all query heads share a single K/V head).

Fewer K/V heads means a proportionally smaller KV cache and fewer bytes to read per decode step, so decoding is faster and memory-cheaper. MQA can cause a small quality drop and sometimes training instability; GQA recovers nearly all the quality while still giving most of the savings. In practice GQA models are often **uptrained** — converted from an MHA checkpoint by mean-pooling K/V heads into groups, then fine-tuning briefly.

<pre class="mermaid">
flowchart LR
  subgraph MHA[&quot;MHA (g = h)&quot;]
    Q1[&quot;Q1&quot;] --&gt; K1[&quot;K/V 1&quot;]
    Q2[&quot;Q2&quot;] --&gt; K2[&quot;K/V 2&quot;]
    Q3[&quot;Q3&quot;] --&gt; K3[&quot;K/V 3&quot;]
    Q4[&quot;Q4&quot;] --&gt; K4[&quot;K/V 4&quot;]
  end
  subgraph GQA[&quot;GQA (g groups)&quot;]
    GQ1[&quot;Q1&quot;] --&gt; GK1[&quot;K/V A&quot;]
    GQ2[&quot;Q2&quot;] --&gt; GK1
    GQ3[&quot;Q3&quot;] --&gt; GK2[&quot;K/V B&quot;]
    GQ4[&quot;Q4&quot;] --&gt; GK2
  end
  subgraph MQA[&quot;MQA (g = 1)&quot;]
    MQ1[&quot;Q1&quot;] --&gt; MK1[&quot;K/V shared&quot;]
    MQ2[&quot;Q2&quot;] --&gt; MK1
    MQ3[&quot;Q3&quot;] --&gt; MK1
    MQ4[&quot;Q4&quot;] --&gt; MK1
  end
</pre>
**FlashAttention — IO-aware exact attention.** The naive kernel writes the full $n \times n$ scores to slow high-bandwidth memory (HBM), then reads them back for softmax — the cost is dominated by these memory round-trips, not the math. FlashAttention instead **tiles** Q, K, V into blocks that fit in on-chip SRAM, and uses the *online softmax* trick (running max and running sum) to compute the exact softmax-weighted output block-by-block without ever storing the full matrix. Results:

- Memory drops from $O(n^2)$ to $O(n)$ (only block-sized buffers + the output live in SRAM/registers).
- Far fewer HBM reads/writes → 2–4× wall-clock speedup; the answer is **mathematically identical** to standard attention (it is *exact*, not approximate).
- Backward pass recomputes attention on the fly instead of storing it, saving activation memory.
- **FlashAttention-2** improves GPU work partitioning/occupancy; **FlashAttention-3** exploits Hopper (H100) FP8 and asynchrony for further gains.

FlashAttention is orthogonal to MQA/GQA — production stacks use both together.

**Sparse / local / sliding-window attention.** Instead of attending to all $n$ tokens, each token attends only to a fixed window of $w$ neighbors, giving $O(n \cdot w)$ ≈ $O(n)$ cost. With $L$ stacked layers the *effective* receptive field grows to ~$L \cdot w$ tokens, so information still propagates globally, just indirectly. **Longformer** combines a sliding window with a few global tokens; **Mistral 7B** uses a 4096-token sliding window. **Linear / approximate attention** (Performer, Linformer, linear-attention/SSM-style models) reformulates softmax as a kernel feature map or low-rank projection to reach $O(n)$, but these are approximations and have historically lagged exact attention on quality.

## Variants / Trade-offs

| Variant | Query heads | K/V heads | KV-cache size | Compute | Exact? | Quality | Used in |
|---|---|---|---|---|---|---|---|
| **MHA** | $h$ | $h$ | $1\times$ (baseline) | $O(n^2 d)$ | yes | best | GPT-3, original Transformer |
| **MQA** | $h$ | $1$ | $\sim 1/h$ | $O(n^2 d)$ | yes | slight drop | PaLM, Falcon |
| **GQA** | $h$ | $g\ (1<g<h)$ | $\sim g/h$ | $O(n^2 d)$ | yes | ≈ MHA | Llama-2/3, Mistral, Gemma |
| **FlashAttention** | $h$ | $h$ | unchanged | $O(n^2 d)$ FLOPs, $O(n)$ **memory** | yes | identical | nearly all modern training/inference |
| **Sliding window** | $h$ | $h$ | $\sim w/n$ (windowed) | $O(n\,w)$ | no (local) | task-dependent | Mistral, Longformer |
| **Linear/approx** | $h$ | $h$ | $O(1)$ state | $O(n)$ | no (approx) | often lags | Performer, Linformer, RWKV |

**When to use which:** GQA + FlashAttention is the standard pairing for general LLMs today. Add sliding-window when you need long context cheaply and can tolerate weaker long-range precision. Reach for linear/approximate attention only for very long sequences where exact attention is infeasible and quality is secondary.

## Practical considerations

- **GQA is the de facto default.** Llama-2 70B and Llama-3 use GQA (e.g., 8 K/V heads for 64 query heads), giving a large KV-cache reduction with negligible perplexity change.
- **KV cache dominates serving memory.** MQA/GQA's cache savings let you raise batch size / context length on the same GPU. They compose with KV-cache **quantization** (FP8/INT8 K/V) and paged caches (PagedAttention / vLLM) — see [KV Cache and Inference Optimization](/notes/ml-algorithms/language-models/kv-cache-and-inference-optimization/) and LLM Serving Platform.
- **FlashAttention is essentially free quality-wise** (exact) so there's rarely a reason not to use it; main caveats are hardware/kernel support and head-dim limits. It's built into PyTorch SDPA, xFormers, vLLM, etc.
- **Sliding window needs care at inference:** the KV cache can be a rolling buffer of size $w$, but tokens beyond the window are truly invisible to a given layer — verify your task tolerates that.
- **Quality testing:** always evaluate long-context and retrieval tasks after switching to sparse/approximate attention; perplexity alone can hide "needle-in-a-haystack" failures.

## Related

- [Attention](/notes/ml-algorithms/deep-learning/attention/) · [Transformers](/notes/ml-algorithms/deep-learning/transformers/) · [Softmax](/notes/ml-algorithms/core-concepts/softmax/)
- [KV Cache and Inference Optimization](/notes/ml-algorithms/language-models/kv-cache-and-inference-optimization/) · [Long Context](/notes/ml-algorithms/language-models/long-context/) · LLM Serving Platform
- [LLM Architecture](/notes/ml-algorithms/language-models/llm-architecture/) · [Positional Encodings](/notes/ml-algorithms/language-models/positional-encodings/) · [Quantization for LLMs](/notes/ml-algorithms/language-models/quantization-for-llms/) · [Speculative Decoding and Distillation](/notes/ml-algorithms/language-models/speculative-decoding-and-distillation/)
