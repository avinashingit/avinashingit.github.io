---
layout: note
title: "KV Cache and Inference Optimization"
description: "When an LLM generates text one token at a time, each new token must attend to every previous token. Naively you would re-run the whole prompt through attention at every step, wa…"
note: true
note_collection: "ML algorithms"
note_section: "Language Models"
section_order: 6
note_order: 4
updated: 2026-06-07 03:58:39 -0700
keywords:
  - LLMs
  - Transformers
  - Inference
  - Optimization
  - Training
math: true
mermaid: true
---
> The KV cache stores past Keys and Values so autoregressive decoding attends to history without recomputing it, turning per-token cost from quadratic to linear — and reshaping how LLMs are served. Related: [Attention](/notes/ml-algorithms/deep-learning/attention/) · [Attention Variants and Efficiency](/notes/ml-algorithms/language-models/attention-variants-and-efficiency/) · LLM Serving Platform · [Long Context](/notes/ml-algorithms/language-models/long-context/)

## TL;DR

When an LLM generates text one token at a time, each new token must attend to **every previous token**. Naively you would re-run the whole prompt through attention at every step, wasting $O(n^2)$ work. The **KV cache** fixes this: for each layer you store the Key (K) and Value (V) tensors of all past tokens, so a new token only computes its own Query (Q), K, and V and attends against the cache. This splits inference into a parallel, compute-bound **prefill** phase and a sequential, memory-bandwidth-bound **decode** phase. The cache itself becomes the dominant memory cost at scale, which motivates **PagedAttention** (vLLM), **continuous batching**, **GQA/MQA**, and KV-cache quantization.

## Why it matters

Inference, not training, is where most production GPU dollars are spent — a model is trained once but serves billions of tokens. Decoding is inherently sequential: token $t+1$ depends on token $t$. The attention math says token $t$ must look at the K and V of all tokens $1..t-1$. Without caching, generating an $n$-token sequence costs roughly $\sum_{t=1}^{n} O(t \cdot d) = O(n^2 d)$ — you redo the same projections and dot-products over and over.

The KV cache is the single most important inference optimization: it stores K and V once per token and reuses them, so each step does only $O(t \cdot d)$ attention work instead of recomputing the whole prefix. But that storage is not free — it grows linearly with sequence length and batch size, and for long context or high concurrency it, not the model weights, becomes the bottleneck. Understanding the cache is therefore the gateway to understanding **throughput**, **latency**, and **why GPUs sit idle** in real serving systems.

## How it works

Recall self-attention (see [Attention](/notes/ml-algorithms/deep-learning/attention/)). Each token's hidden state $x$ is projected into $Q = xW_Q$, $K = xW_K$, $V = xW_V$, and the output is

$$\text{Attn}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

where $d_k$ is the per-head dimension. The key observation: **K and V for past tokens never change** once computed (the prompt and already-generated tokens are fixed). So we cache them per layer and per head.

**Two phases.**

- **Prefill**: process the entire prompt of length $p$ in one parallel forward pass. This computes K and V for all $p$ tokens at every layer and writes them to the cache. Prefill is **compute-bound** — large matrix multiplies keep the GPU's tensor cores busy. It produces the first output token.
- **Decode**: generate one token at a time. For step $t$ you compute only the new token's $Q,K,V$ (a single row), append its K,V to the cache, and attend the new $Q$ against the **full cached** K,V. Decode is **memory-bandwidth-bound**: per token you must stream all model weights plus the growing KV cache from HBM, while doing tiny matrix-vector work. The GPU is starved on memory, not compute.

<pre class="mermaid">
flowchart TD
    P[&quot;Prompt (p tokens)&quot;] --&gt; PF[&quot;Prefill: 1 parallel pass&lt;br/&gt;compute-bound&quot;]
    PF --&gt; WR[&quot;Write K,V for all layers&lt;br/&gt;into KV cache&quot;]
    WR --&gt; KV[&quot;KV Cache&lt;br/&gt;(per layer, per head)&quot;]
    PF --&gt; T0[&quot;Emit first token (TTFT)&quot;]
    T0 --&gt; DEC[&quot;Decode step: compute Q,K,V&lt;br/&gt;for one new token&quot;]
    DEC --&gt; AP[&quot;Append new K,V to cache&quot;]
    AP --&gt; KV
    KV --&gt; AT[&quot;Attend new Q over cached K,V&lt;br/&gt;memory-bandwidth-bound&quot;]
    AT --&gt; NT[&quot;Emit next token (TPOT)&quot;]
    NT --&gt;|&quot;loop until EOS or max_len&quot;| DEC
</pre>
**KV-cache memory math.** The cache size in bytes is approximately

$$\text{KV bytes} \approx 2 \times L \times h_{kv} \times d_{head} \times s \times b \times \text{bytes}$$

where the leading $2$ is for K **and** V, $L$ = number of layers, $h_{kv}$ = number of **key/value** heads, $d_{head}$ = per-head dimension, $s$ = sequence length, $b$ = batch size, and bytes = 2 for FP16/BF16.

Worked example — a Llama-2-13B-style model ($L=40$, $h_{kv}=40$ since it uses multi-head attention, $d_{head}=128$) at FP16, one sequence of 2{,}048 tokens:

$$2 \times 40 \times 40 \times 128 \times 2048 \times 1 \times 2 \approx 1.07\ \text{GB}$$

That is **per request**. Batch 32 such requests and the cache alone is ~34 GB — more than the weights on many GPUs. This is why the cache, not the model, caps how many concurrent users (or how long a context) you can serve. Two structural fixes follow directly from the formula: shrink $h_{kv}$ (**GQA/MQA**) and shrink bytes (**quantization**).

## Variants / Trade-offs

| Technique | What it changes | Win | Cost / caveat |
|---|---|---|---|
| **KV cache** (baseline) | Store past K,V; recompute nothing | $O(n^2)\to O(n)$ decode work | Linear memory growth in $s,b$ |
| **MQA / GQA** | Shrink $h_{kv}$: 1 (MQA) or $g$ groups (GQA) share K,V across query heads | Cache shrinks $h/h_{kv}\times$; faster decode | Slight quality drop (MQA); GQA is the modern default (Llama-2/3) — see [Attention Variants and Efficiency](/notes/ml-algorithms/language-models/attention-variants-and-efficiency/) |
| **PagedAttention** (vLLM) | Store KV in fixed-size non-contiguous blocks, OS-paging style | ~0 fragmentation; high batch concurrency; prefix sharing | Custom attention kernel; block-table bookkeeping |
| **Continuous batching** | Add/evict requests from the batch every step | GPU stays busy with variable-length requests | Scheduler complexity |
| **KV-cache quantization** | Store K,V as INT8/FP8/INT4 instead of FP16 | 2–4× smaller cache, larger batch/context | Accuracy loss if keys are quantized too aggressively |
| **Prefix / prompt caching** | Reuse cached KV of a shared prefix across requests | Skip redundant prefill | Needs identical leading tokens; cache eviction policy |

**PagedAttention.** Classic serving pre-allocated one contiguous KV buffer per request sized to `max_len`. With variable output lengths this wastes 60–80% of memory to internal/external fragmentation. PagedAttention borrows virtual memory paging: the cache is split into fixed **blocks** (e.g., 16 tokens), a per-request **block table** maps logical positions to physical blocks, and blocks are allocated on demand. Result: near-zero waste, far higher batch sizes, and **copy-on-write block sharing** so a shared prompt prefix (system prompt, few-shot examples) is stored once.

**Continuous (in-flight) batching.** Static batching waits for every sequence in a batch to finish before starting the next batch, so a 500-token request stalls behind a 5-token neighbor. Continuous batching schedules at the **token** level: finished sequences are evicted and queued requests are injected at the next step, keeping the batch full. PagedAttention + continuous batching together are why vLLM and TensorRT-LLM hit much higher throughput than naive loops.

## Practical considerations

- **TTFT vs TPOT.** **TTFT** (Time To First Token) is dominated by **prefill** and scales with prompt length — long prompts feel slow to start. **TPOT** (Time Per Output Token, a.k.a. inter-token latency) is set by the **decode** step and is memory-bandwidth limited; total latency $\approx \text{TTFT} + (\text{output tokens}) \times \text{TPOT}$. Chunked prefill interleaves prefill chunks with decode to balance the two so big prompts don't freeze ongoing generations.
- **Compute vs bandwidth.** Decode's low arithmetic intensity is exactly why bigger batches help throughput "for free" — you amortize the weight read across many sequences — until the KV cache exhausts HBM. The serving system constantly trades batch size against cache memory.
- **Defaults you'll see in production.** GQA is standard in modern open models; FP16/BF16 KV by default with optional FP8 KV on newer stacks; block size 16; vLLM/TensorRT-LLM/SGLang as the serving layer (see LLM Serving Platform). Prefix caching is on by default for chat system prompts.
- **Long context** (see [Long Context](/notes/ml-algorithms/language-models/long-context/)) makes the cache the hard limit: at 128K tokens the per-request cache is dozens of GB, so GQA + KV quantization + paging are mandatory, not optional.
- **Composability.** KV optimizations stack cleanly with [Quantization for LLMs](/notes/ml-algorithms/language-models/quantization-for-llms/) (weights and cache), FlashAttention kernels, and speculative decoding, which attacks the sequential-decode bottleneck from a different angle.

## Related

- [Attention](/notes/ml-algorithms/deep-learning/attention/) · [Attention Variants and Efficiency](/notes/ml-algorithms/language-models/attention-variants-and-efficiency/) (GQA/MQA, FlashAttention)
- [Long Context](/notes/ml-algorithms/language-models/long-context/) · [Quantization for LLMs](/notes/ml-algorithms/language-models/quantization-for-llms/) · LLM Serving Platform
- [Decoding Strategies](/notes/ml-algorithms/language-models/decoding-strategies/) · [Speculative Decoding and Distillation](/notes/ml-algorithms/language-models/speculative-decoding-and-distillation/) · [Transformers](/notes/ml-algorithms/deep-learning/transformers/)
