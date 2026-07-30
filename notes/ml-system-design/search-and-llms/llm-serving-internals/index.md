---
layout: note
title: "LLM Serving Internals (Databricks favorite)"
description: "Why LLM inference is its own systems discipline, in five concepts."
note: true
note_collection: "ML system design"
note_section: "Search and LLMs"
section_order: 3
note_order: 4
updated: 2026-06-09 12:41:40 -0700
keywords:
  - LLMs
  - Retrieval
  - Serving
  - Transformers
  - Evaluation
math: true
mermaid: false
---
Why LLM inference is its own systems discipline, in five concepts.

## 1. Prefill vs decode
Serving a request has two phases: **prefill** (process the whole prompt in parallel — compute-bound, big matrix multiplies) and **decode** (generate one token at a time — each step reads all model weights to produce one token → **memory-bandwidth-bound**). Most production pain lives in decode.

## 2. The KV cache
Attention at step $t$ needs Keys/Values of all previous tokens ([Transformers from Scratch](/notes/ml-system-design/foundations/transformers-from-scratch/)). Recomputing them every step would be O(T²) per token; instead cache them. Cost: per-token cache memory = 2 (K and V) × layers × kv-heads × head_dim × bytes. For a 70B-class model this is on the order of **hundreds of KB–MBs per token**, i.e., *gigabytes per long conversation* — KV cache, not weights, limits concurrent users. Mitigations: **Multi-Query / Grouped-Query Attention** (share K/V across heads — fewer KV heads), quantized KV cache, sliding-window attention.

## 3. PagedAttention (vLLM)
Naively each request pre-allocates a contiguous max-length KV buffer → massive internal fragmentation (most of it unused). PagedAttention manages the KV cache like an OS manages RAM: fixed-size **blocks/pages** + a per-request page table; allocate on demand, share pages across requests (e.g., a common system-prompt **prefix is cached once** — prefix caching). Result: 2–4× higher throughput from memory efficiency alone.

## 4. Continuous (in-flight) batching
Static batching waits to assemble a batch and holds it until the *longest* sequence finishes — GPU idles on stragglers. **Continuous batching** schedules at the *iteration* level: every decode step, finished sequences exit and queued requests join immediately. This is the single biggest throughput lever in LLM serving; "vLLM = continuous batching + PagedAttention" is a fine one-liner.

## 5. Quantization & speculative decoding
- **Quantization** (see Model Compression - Distillation and Quantization): weights to INT8/INT4 (GPTQ/AWQ) → less memory traffic → faster *decode* (it's bandwidth-bound!), bigger batches per GPU.
- **Speculative decoding:** a small **draft model** proposes k tokens; the big model verifies all k in *one parallel forward pass*; accepted prefix keeps, first rejection resamples. Output distribution provably unchanged; 2–3× latency win when draft acceptance is high. Variants: Medusa heads, self-speculation.

## Metrics & SLOs to name
**TTFT** (time to first token — prefill latency), **TPOT/ITL** (per-token latency), tokens/sec/GPU (throughput), goodput under SLO. Tradeoff knob: batch size ↑ → throughput ↑, TTFT/TPOT ↑. Also mention: separate prefill/decode pools ("disaggregated serving"), priority queues, streaming responses, autoscaling on queue depth.
