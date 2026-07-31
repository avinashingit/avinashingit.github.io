---
layout: note
title: "Parameter-Efficient Fine-Tuning"
description: "Parameter-Efficient Fine-Tuning (PEFT) freezes the pretrained model and trains only a small set of extra or selected parameters, getting most of the quality of full fine-tuning…"
note: true
note_collection: "ML algorithms"
note_section: "Language Models"
section_order: 6
note_order: 10
updated: 2026-06-07 03:58:33 -0700
keywords:
  - LLMs
  - Transformers
  - Training
  - Evaluation
  - Inference
math: true
mermaid: true
---
> Adapting a large pretrained model to a task by training a tiny fraction of (often new) parameters while keeping the base weights frozen. Related: [Supervised Fine-Tuning](/notes/ml-algorithms/language-models/supervised-fine-tuning/), [Quantization for LLMs](/notes/ml-algorithms/language-models/quantization-for-llms/), [RLHF and Preference Optimization](/notes/ml-algorithms/language-models/rlhf-and-preference-optimization/), LLM Serving Platform

## TL;DR

Parameter-Efficient Fine-Tuning (PEFT) freezes the pretrained model and trains only a small set of extra or selected parameters, getting most of the quality of full fine-tuning at a fraction of the memory and storage. The dominant method is **LoRA** (Low-Rank Adaptation): freeze weight $W$ and learn a low-rank update $\Delta W = BA$, training <1% of parameters. **QLoRA** quantizes the frozen base to 4-bit so a 65B model fine-tunes on a single 48GB GPU. Because LoRA deltas are tiny and mergeable, you can serve many task- or customer-specific adapters on one shared base model.

## Why it matters

Full fine-tuning of an LLM updates **all** weights. The pain is not the forward pass — it is the optimizer state. With Adam, each trainable parameter needs the weight plus two moment estimates (first and second moments), plus gradients. In mixed precision this is roughly **12–16 bytes per parameter**, so a 7B model needs ~100GB+ just for optimizer state and gradients, and a 65B model is far out of reach on a single GPU. You also produce a full-size checkpoint (tens to hundreds of GB) **per task**.

PEFT attacks both problems. By freezing the base, there is **no optimizer state for the frozen weights** — you only store moments for the tiny trainable set. The artifact you save per task is a few megabytes, not gigabytes. This makes fine-tuning feasible on commodity GPUs, makes storing hundreds of task variants cheap, and enables multi-tenant serving where one base model is shared across many adapters. PEFT sits right after pretraining and alongside [Supervised Fine-Tuning](/notes/ml-algorithms/language-models/supervised-fine-tuning/) and [RLHF and Preference Optimization](/notes/ml-algorithms/language-models/rlhf-and-preference-optimization/) in the adaptation stack — the same SFT/preference objectives, but with a parameter-efficient delta instead of a full-weight update.

## How it works

### LoRA in detail

Take a pretrained weight matrix $W \in \mathbb{R}^{d \times k}$ (e.g., an attention projection). Full fine-tuning would learn $W + \Delta W$ with a dense $\Delta W$. LoRA's insight is that the **update** needed to adapt a model has low intrinsic rank, so we constrain $\Delta W$ to be low-rank:

$$\Delta W = B A, \qquad A \in \mathbb{R}^{r \times k},\quad B \in \mathbb{R}^{d \times r},\quad r \ll \min(d,k)$$

The adapted forward pass for input $x$ is:

$$h = W x + \frac{\alpha}{r}\, B A x$$

where:
- $W$ is **frozen** (no gradients, no optimizer state),
- $A$ and $B$ are the **only trained** parameters,
- $r$ is the **rank** (typically 8–64), the bottleneck dimension,
- $\alpha$ is a scaling constant; the factor $\alpha/r$ keeps the update magnitude roughly stable as you change $r$.

Initialization matters: $A$ is set with a random Gaussian and $B$ is initialized to **zero**, so $\Delta W = 0$ at the start — training begins exactly at the pretrained model and learns the delta from there. The number of trained parameters is $r(d+k)$ versus $dk$ for the full matrix; with $d=k=4096$ and $r=16$ that is ~131k vs ~16.8M per matrix, often **<1%** of total params.

**Merging for zero inference latency.** Because $h = (W + \frac{\alpha}{r}BA)x$ is linear, after training you can compute $W' = W + \frac{\alpha}{r}BA$ **once** and fold it back into the weight. The served model then has the exact same shape and FLOPs as the original — **no extra layers, no added latency**. This is LoRA's big serving advantage over adapter layers (below), which add compute at every forward pass.

**Choosing $r$ and $\alpha$.** Higher $r$ = more capacity but more params; $r=8$–$16$ is a strong default, raise to 32–64 for harder/larger-shift tasks. A common heuristic is $\alpha = 2r$ (or $\alpha = r$). Too-large $\alpha/r$ can destabilize training; too-small under-fits.

**Which layers.** The original LoRA paper applied it to the attention **query and value** projections ($W_q$, $W_v$) and found that sufficient. Modern practice (e.g., QLoRA) often applies LoRA to **all linear layers** — $q,k,v,o$ plus the MLP up/down projections — which improves quality at modest extra cost. See [Attention Variants and Efficiency](/notes/ml-algorithms/language-models/attention-variants-and-efficiency/) for what these projections are.

<pre class="mermaid">
flowchart LR
    X[&quot;input x&quot;] --&gt; W[&quot;Frozen W&lt;br/&gt;(d x k)&quot;]
    X --&gt; A[&quot;A (r x k)&lt;br/&gt;trainable&quot;]
    A --&gt; B[&quot;B (d x r)&lt;br/&gt;trainable&quot;]
    B --&gt; SCALE[&quot;scale by alpha/r&quot;]
    W --&gt; ADD[&quot;sum&quot;]
    SCALE --&gt; ADD
    ADD --&gt; H[&quot;output h = Wx + (alpha/r) BAx&quot;]
</pre>
### QLoRA

QLoRA (Quantized LoRA) pushes memory further so a **65B model fine-tunes on one 48GB GPU**. Three ingredients:

1. **4-bit NF4 base.** The frozen weights are quantized to **NF4** (4-bit NormalFloat), a data type whose quantization levels are information-theoretically optimal for normally-distributed weights. The base is loaded in 4-bit and **dequantized on the fly** to compute the forward/backward pass; gradients flow only into the LoRA adapters, which are kept in higher precision (BF16). The frozen base is never trained, so its lossy quantization is acceptable.
2. **Double quantization.** The per-block quantization constants are themselves quantized, saving ~0.4 bits/param on average.
3. **Paged optimizers.** Optimizer state lives in CPU/unified memory and is paged to GPU as needed (like OS virtual memory), preventing OOM spikes on long sequences.

QLoRA showed near-full-fine-tuning quality at a tiny fraction of the memory. See [Quantization for LLMs](/notes/ml-algorithms/language-models/quantization-for-llms/) for NF4, blockwise quantization, and the broader quant landscape.

### Other PEFT methods

- **Adapters** — insert small **bottleneck** MLP layers (down-project to dimension $m$, nonlinearity, up-project, residual) inside each transformer block. Trains only the adapters, but **adds layers**, so inference has extra latency unless fused.
- **Prefix / prompt tuning** — prepend learnable "soft prompt" vectors to the input (or to keys/values at every layer for prefix tuning). The model weights are untouched; only the prompt embeddings are learned. Very few params, but typically lower ceiling than LoRA and consumes context length. Related to [Prompt Engineering](/notes/ml-algorithms/language-models/prompt-engineering/) but learned, not hand-written.
- **(IA)³** — learns per-feature **scaling vectors** that rescale keys, values, and FFN activations. Extremely few parameters; the scalings can be merged into the weights for no added latency.

## Variants / Types / Trade-offs

| Method | Trainable params | Mem savings | Added inference latency | Quality vs full FT | When to use |
|---|---|---|---|---|---|
| **Full fine-tuning** | 100% | none (huge optimizer state) | none | baseline (best) | Big distribution shift, ample GPUs, single dedicated model |
| **LoRA** | ~0.1–1% | large (no base optimizer state) | **none if merged** | within ~1–2% of full | Default PEFT; many tasks/adapters on one base |
| **QLoRA** | ~0.1–1% | largest (4-bit base + paged opt) | none if merged (back to FP16) | ~LoRA, slight quant gap | Fine-tune very large models on tiny hardware budgets |
| **Adapters** | ~1–5% | large | **yes** (extra layers) | strong | When mergeability isn't required; modular task stacking |
| **Prefix / prompt tuning** | <0.1% | largest | small (extra tokens) | lower ceiling | Tiny shifts, very many tasks, minimal storage |
| **(IA)³** | <0.1% | large | none if merged | good for its size | Ultra-low-param budget, mergeable |

## Practical considerations

- **Serving many adapters (multi-tenant).** A core production win: keep **one** base model resident in GPU memory and **hot-swap** small LoRA adapters per request — per-customer style, per-task tone, per-tenant policy. Frameworks (e.g., S-LoRA, vLLM multi-LoRA, Punica) batch requests using **different adapters together**, applying each request's $BA$ on top of the shared frozen $W$. Hundreds of MB-sized adapters fit where you could never hold hundreds of full models. Note the trade-off: if you **merge** an adapter you get zero latency but lose the ability to mix adapters in one batch, so multi-tenant servers keep adapters **unmerged** and apply them dynamically. This is how a LLM Serving Platform offers per-customer customization cheaply — e.g., generating on-brand Recruiter Outreach Generation messages with a customer-specific LoRA.
- **Targets and rank defaults.** Start with LoRA on all linear projections, $r=16$, $\alpha=32$, dropout ~0.05. Increase $r$ for code/math or large domain shifts.
- **Quality gap.** PEFT can lag full FT when the task demands large representational change (new language, very different domain). Raise rank, add target modules, or fall back to full FT.
- **It composes with the same objectives.** LoRA is orthogonal to *what* you train on — you can do LoRA-based SFT ([Supervised Fine-Tuning](/notes/ml-algorithms/language-models/supervised-fine-tuning/)) and LoRA-based DPO/PPO ([RLHF and Preference Optimization](/notes/ml-algorithms/language-models/rlhf-and-preference-optimization/)).
- **Saving/loading.** You ship only the adapter (a few MB) plus a pointer to the base; reproducibility requires pinning the exact base checkpoint and the $\alpha/r$ scaling.

## Related

- [Supervised Fine-Tuning](/notes/ml-algorithms/language-models/supervised-fine-tuning/) · [RLHF and Preference Optimization](/notes/ml-algorithms/language-models/rlhf-and-preference-optimization/) · [Quantization for LLMs](/notes/ml-algorithms/language-models/quantization-for-llms/) · [Pretraining and Language Modeling](/notes/ml-algorithms/language-models/pretraining-and-language-modeling/)
- [Attention Variants and Efficiency](/notes/ml-algorithms/language-models/attention-variants-and-efficiency/) · [LLM Architecture](/notes/ml-algorithms/language-models/llm-architecture/) · [Prompt Engineering](/notes/ml-algorithms/language-models/prompt-engineering/)
- LLM Serving Platform · Recruiter Outreach Generation · [KV Cache and Inference Optimization](/notes/ml-algorithms/language-models/kv-cache-and-inference-optimization/)
