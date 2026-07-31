---
layout: note
title: "Quantization for LLMs"
description: "Quantization maps high-precision floating-point numbers (FP16/BF16) to low-bit integers (INT8/INT4) or compact floats (FP8/NF4) using a learned scale and zero-point, shrinking a…"
note: true
note_collection: "ML algorithms"
note_section: "Language Models"
section_order: 6
note_order: 14
updated: 2026-06-07 03:58:56 -0700
keywords:
  - LLMs
  - Transformers
  - Evaluation
  - Inference
  - Optimization
math: true
mermaid: true
---
> Storing weights (and sometimes activations) in fewer bits to cut memory and speed up memory-bound decoding, with minimal accuracy loss. Related: [KV Cache and Inference Optimization](/notes/ml-algorithms/language-models/kv-cache-and-inference-optimization/), [Parameter-Efficient Fine-Tuning](/notes/ml-algorithms/language-models/parameter-efficient-fine-tuning/), LLM Serving Platform

## TL;DR

Quantization maps high-precision floating-point numbers (FP16/BF16) to low-bit integers (INT8/INT4) or compact floats (FP8/NF4) using a learned **scale** and **zero-point**, shrinking a model 2–4x or more. Because LLM decoding is **memory-bandwidth bound** — every token requires reading all weights from memory — fewer bits means both less VRAM and faster generation. The dominant approach is **weight-only post-training quantization** (GPTQ, AWQ) at INT4: roughly lossless at INT8, a small drop at INT4, and degrading rapidly below that. The hard part is **activation outliers**, which methods like LLM.int8(), SmoothQuant, and AWQ are designed to handle.

## Why it matters

A 70B-parameter model in FP16 (2 bytes/param) needs ~140 GB just for weights — more than a single 80 GB H100. At INT4 (0.5 bytes/param) it drops to ~35 GB, fitting comfortably on one GPU and freeing room for the [KV cache](/notes/ml-algorithms/language-models/kv-cache-and-inference-optimization/). This is the difference between "needs a multi-GPU server" and "runs on one card" or even a laptop.

Quantization also directly attacks the **decode bottleneck**. Autoregressive generation produces one token at a time, and each step streams the entire weight matrix through the arithmetic units; the GPU is idle waiting on memory, not compute. Halving the bytes-per-weight roughly halves the memory traffic and can nearly double decode throughput, even when arithmetic still happens in FP16. This is why **weight-only** quantization is so popular: it targets exactly the resource (memory bandwidth) that limits LLM inference.

## How it works

**Affine (asymmetric) quantization** maps a floating-point value $x$ to a $b$-bit integer:

$$ x_q = \text{clamp}\!\left(\text{round}\!\left(\frac{x}{s}\right) + z,\; q_{min},\; q_{max}\right), \qquad \hat{x} = s\,(x_q - z) $$

where $s$ (the **scale**) sets the step size, $z$ (the **zero-point**) is the integer mapping to real zero, $q_{min}/q_{max}$ are the representable range (e.g. $0..15$ for INT4), and $\hat{x}$ is the dequantized approximation. **Symmetric** quantization fixes $z=0$ and uses a signed range (e.g. $-8..7$). For a tensor with range $[\alpha,\beta]$ over $b$ bits, $s=\frac{\beta-\alpha}{2^b-1}$. The quantization error per element is roughly $s/2$, so a tighter range (better $s$) means less error.

**Granularity** controls how many values share one scale: **per-tensor** (one $s$ for the whole matrix — cheapest, least accurate), **per-channel/per-row** (one $s$ per output channel), or **per-group** (one $s$ per block of, say, 64 or 128 weights — the standard for INT4, balancing overhead and fidelity).

**PTQ vs QAT.** *Post-Training Quantization (PTQ)* quantizes an already-trained model using a small **calibration set** (a few hundred sequences) to estimate ranges — fast, no gradients, the default for LLMs. *Quantization-Aware Training (QAT)* simulates quantization during training/fine-tuning so the model learns weights robust to rounding, using a **straight-through estimator (STE)** to pass gradients through the non-differentiable round. QAT gives better low-bit accuracy but costs a training run, so it is reserved for aggressive (sub-4-bit) targets.

**The outlier problem.** Beyond ~6.7B parameters, a few activation channels develop huge magnitudes (10–100x the rest). A single per-tensor scale must stretch to cover them, blowing up the step size and destroying precision for the common small values. This is why naive INT8 *activation* quantization breaks on large models. Approaches:

- **LLM.int8() (bitsandbytes):** decompose the matmul — keep the ~0.1% outlier dimensions in FP16, quantize the rest to INT8, then recombine. Robust and simple; some overhead from the mixed path.
- **SmoothQuant:** migrate the difficulty from activations into weights. Scale activation channel $j$ down by $d_j$ and the corresponding weight up by $d_j$ (mathematically invariant), making activations smooth and easy to quantize to INT8.
- **GPTQ:** a **weight-only**, layer-wise PTQ that quantizes columns one at a time and uses **second-order (Hessian) information** ($H = X X^\top$ from calibration activations) to update the remaining un-quantized weights, compensating for each rounding error. Excellent INT4 with cheap calibration.
- **AWQ (Activation-aware Weight Quantization):** observes that a small fraction of weights are **salient** (those multiplying large-magnitude activation channels). It searches a per-channel scaling that protects those weights before rounding — no backprop, fast, very strong at INT4, and friendly to hardware.
- **NF4 (4-bit NormalFloat, from QLoRA):** a non-uniform 4-bit type whose 16 levels are placed at quantiles of a normal distribution, matching the near-Gaussian shape of weights better than uniform INT4. Used to freeze a base model at 4 bits while training [LoRA](/notes/ml-algorithms/language-models/parameter-efficient-fine-tuning/) adapters on top.

**FP8 and KV-cache quantization.** Modern GPUs (Hopper H100, Ada, Blackwell) have native **FP8** (E4M3/E5M2) tensor cores; FP8's floating exponent tolerates outliers far better than INT8, enabling near-lossless **weight + activation** quantization with real compute speedups. Separately, the **KV cache** grows linearly with context and batch and often dominates memory at long context — quantizing cached keys/values to INT8 or FP8 (e.g. KVQuant) can halve cache size for little quality loss.

<pre class="mermaid">
flowchart TD
    FP[&quot;FP16/BF16 model&quot;] --&gt; CHOICE{&quot;PTQ or QAT?&quot;}
    CHOICE --&gt;|&quot;no retrain&quot;| PTQ[&quot;PTQ + calibration set&quot;]
    CHOICE --&gt;|&quot;train/fine-tune&quot;| QAT[&quot;QAT with STE&quot;]
    PTQ --&gt; WO[&quot;Weight-only&lt;br/&gt;INT8 / INT4&quot;]
    PTQ --&gt; WA[&quot;Weight + Activation&lt;br/&gt;INT8 / FP8&quot;]
    WO --&gt; GA[&quot;GPTQ / AWQ / NF4&quot;]
    WA --&gt; OUT[&quot;LLM.int8 / SmoothQuant&quot;]
    GA --&gt; RT[&quot;Runtime: GGUF llama.cpp&lt;br/&gt;or GPTQ/AWQ on GPU&quot;]
    OUT --&gt; RT
    QAT --&gt; RT
</pre>
## Variants / Trade-offs

| Method | Bits (typical) | Scope | Key idea | Retrain? | Best for |
|---|---|---|---|---|---|
| LLM.int8() | INT8 | Weight + act | Keep FP16 outlier dims, INT8 rest | No | Easy, robust drop-in |
| SmoothQuant | INT8 | Weight + act | Shift outliers from act to weight | No (calib) | INT8 serving with act quant |
| GPTQ | INT4 (3–4) | Weight-only | Hessian-based error compensation | No (calib) | Strong INT4 on GPU |
| AWQ | INT4 | Weight-only | Protect salient weights via scaling | No (calib) | Fast, accurate INT4 GPU |
| NF4 (QLoRA) | 4 (NormalFloat) | Weight-only | Quantile-spaced 4-bit float | No | 4-bit base + LoRA fine-tune |
| FP8 | FP8 (E4M3) | Weight + act | Native HW float, outlier-friendly | No | Near-lossless on H100+ |
| GGUF k-quants | 2–8 (mixed) | Weight-only | Per-block mixed-precision | No | CPU/edge via llama.cpp |
| QAT | 2–4 | W/A | Learn quant-robust weights | Yes | Sub-4-bit, max accuracy |

**Accuracy vs size rule of thumb:** INT8 is effectively lossless; INT4 (with GPTQ/AWQ/NF4) costs a small perplexity bump (often <1%); INT3 is noticeably worse; INT2 usually needs QAT to be usable. **When to use which:** GPU serving → AWQ or GPTQ INT4, or FP8 on Hopper/Blackwell; CPU/laptop → GGUF k-quants in llama.cpp; fine-tuning a big model on one GPU → NF4 + LoRA.

## Practical considerations

- **Memory math example.** Llama-2 70B, $7\times10^{10}$ params. FP16: $7\!\times\!10^{10}\times 2\text{ B}\approx 140$ GB. INT8: ~70 GB. INT4: ~35 GB (plus small per-group scale overhead, a few %). Add KV cache and CUDA context, so budget ~40–45 GB at INT4 — single-H100 territory. A 7B model at INT4 is ~3.5–4 GB, runnable on a laptop GPU.
- **Formats and runtimes.** **GGUF** (the llama.cpp format) packs mixed per-block "k-quants" (Q4_K_M, Q5_K_M, Q6_K) and targets CPU/Apple-Silicon/edge. On GPU, **GPTQ** and **AWQ** checkpoints are loaded by serving stacks like vLLM, TGI, and TensorRT-LLM with fused dequant kernels. Marlin/Machete kernels make INT4 matmul fast on Ampere/Hopper.
- **Speedup is from memory, not math.** Weight-only INT4 still typically dequantizes to FP16 for the matmul, so the win is reduced memory traffic during decode — large for small-batch generation, smaller for compute-bound prefill or big batches. FP8 and INT8 W/A schemes additionally use faster tensor cores.
- **Gotchas.** Quantize **embeddings, final LM head, and LayerNorm/RMSNorm** carefully (often kept higher precision). Calibration data should resemble deployment data. Stacking quantization with long context stresses the KV cache — consider quantizing it too. Always re-measure quality (perplexity + a downstream eval), not just file size, since damage can be task-specific.

## Related

- [Parameter-Efficient Fine-Tuning](/notes/ml-algorithms/language-models/parameter-efficient-fine-tuning/) — NF4 underpins QLoRA's 4-bit base + LoRA adapters.
- [KV Cache and Inference Optimization](/notes/ml-algorithms/language-models/kv-cache-and-inference-optimization/) — KV-cache quantization and the memory-bound decode story.
- LLM Serving Platform — where GPTQ/AWQ/FP8 checkpoints get loaded and served.
- [Speculative Decoding and Distillation](/notes/ml-algorithms/language-models/speculative-decoding-and-distillation/) — complementary ways to speed up decoding and shrink models.
- [Decoding Strategies](/notes/ml-algorithms/language-models/decoding-strategies/) · [Mixture of Experts](/notes/ml-algorithms/language-models/mixture-of-experts/) · [Attention Variants and Efficiency](/notes/ml-algorithms/language-models/attention-variants-and-efficiency/)
- Foundations: [Transformers](/notes/ml-algorithms/deep-learning/transformers/) · [Neural Networks](/notes/ml-algorithms/deep-learning/neural-networks/) · [Activation Functions and Optimizers](/notes/ml-algorithms/core-concepts/activation-functions-and-optimizers/)
