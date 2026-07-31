---
layout: note
title: "Mixture of Experts"
description: "Mixture of Experts (MoE) swaps the dense feed-forward network (FFN) in a transformer block for $N$ parallel expert FFNs plus a lightweight router (gating network) that, per toke…"
note: true
note_collection: "ML algorithms"
note_section: "Language Models"
section_order: 6
note_order: 8
updated: 2026-06-07 03:55:30 -0700
keywords:
  - LLMs
  - Transformers
  - Inference
  - Training
  - Optimization
math: true
mermaid: true
---
> A transformer architecture that replaces each block's single dense FFN with many parallel "expert" FFNs and a router that activates only the top-k experts per token, decoupling total parameter count from per-token compute. Related: [LLM Architecture](/notes/ml-algorithms/language-models/llm-architecture/), [Scaling Laws](/notes/ml-algorithms/language-models/scaling-laws/), [Normalization and Activations in LLMs](/notes/ml-algorithms/language-models/normalization-and-activations-in-llms/), [Transformers](/notes/ml-algorithms/deep-learning/transformers/)

## TL;DR

Mixture of Experts (MoE) swaps the dense feed-forward network (FFN) in a transformer block for $N$ parallel expert FFNs plus a lightweight **router** (gating network) that, per token, picks the **top-k** experts (usually $k=1$ or $k=2$). Because each token only touches a few experts, the model can hold a huge number of total parameters while keeping per-token FLOPs (floating-point operations) roughly constant — this is **sparse activation**. The result: far more "knowledge capacity" per unit of compute, at the cost of high memory (all experts must be resident) and tricky load balancing.

## Why it matters

In a standard transformer, capacity and compute are welded together: to make the model "know" more, you widen or deepen it, which raises the FLOPs spent on every single token. [Scaling Laws](/notes/ml-algorithms/language-models/scaling-laws/) tell us that loss falls predictably as we add parameters and data, but dense scaling means a 10x bigger model costs ~10x more compute per token at inference.

MoE breaks that coupling. The intuition: not every token needs the full network. The word "the" is easy; a rare technical term or a code symbol might benefit from a specialized subnetwork. A router learns to send each token to the experts best suited to it, so we get a model with, say, 47B total parameters but only ~13B *active* per token (Mixtral 8x7B). You pay the compute of a 13B-class model but get accuracy closer to a much larger dense model. This is why MoE delivers more knowledge per FLOP and is now standard in frontier-scale systems.

Where it sits: MoE is an architectural choice *inside* the transformer block (it modifies the FFN sublayer, not attention). Everything else — [Attention](/notes/ml-algorithms/deep-learning/attention/), [Positional Encodings](/notes/ml-algorithms/language-models/positional-encodings/), normalization — stays the same.

## How it works

A transformer block has two sublayers: self-attention and a position-wise FFN. MoE leaves attention untouched and replaces the FFN with $N$ expert FFNs $E_1, \dots, E_N$ (each its own up/down projection) plus a router.

**Routing math.** For a token with hidden vector $x \in \mathbb{R}^d$, the router applies a learned weight matrix $W_g \in \mathbb{R}^{N \times d}$ and a softmax to score the experts (see [Softmax](/notes/ml-algorithms/core-concepts/softmax/)):

$$g(x) = \mathrm{softmax}(W_g\, x) \in \mathbb{R}^{N}$$

We then keep only the **top-k** entries of $g(x)$ (set the rest to 0) and, in many implementations, renormalize them so the kept gates sum to 1. Let $\mathcal{T}$ be the indices of the top-k experts. The MoE output is the gate-weighted combination of just those experts:

$$y = \sum_{i \in \mathcal{T}} g_i(x)\, E_i(x)$$

Only the $k$ chosen experts run; the other $N-k$ are skipped entirely for this token. With $N=8$ and $k=2$, each token does the work of 2 FFNs regardless of how many experts exist — that is the sparsity.

**Symbols:** $d$ = hidden size; $N$ = number of experts; $k$ = experts activated per token; $g_i$ = gate weight for expert $i$; $E_i(x)$ = expert $i$'s FFN output. Total parameters scale with $N$; active FLOPs scale with $k$.

<pre class="mermaid">
flowchart TD
    T[&quot;Token hidden state x&quot;] --&gt; R[&quot;Router (softmax over N experts)&quot;]
    R --&gt;|&quot;top-2 gating&quot;| G[&quot;Select experts 3 and 7&quot;]
    G --&gt;|&quot;g3 = 0.7&quot;| E3[&quot;Expert 3 FFN&quot;]
    G --&gt;|&quot;g7 = 0.3&quot;| E7[&quot;Expert 7 FFN&quot;]
    E1[&quot;Expert 1 (skipped)&quot;] -. &quot;not activated&quot; .-&gt; X[&quot;inactive&quot;]
    E3 --&gt; C[&quot;Weighted combine&quot;]
    E7 --&gt; C
    C --&gt;|&quot;y = 0.7*E3 + 0.3*E7&quot;| O[&quot;Block output&quot;]
</pre>
**The load-balancing problem.** Left alone, routing tends to **collapse**: the router finds a few "popular" experts, sends most tokens there, and those experts get more gradient, get better, and attract even more tokens — a rich-get-richer feedback loop. The other experts starve and waste their parameters.

Two mechanisms fix this:

1. **Auxiliary load-balancing loss.** Add a small penalty (typically weight $\alpha \approx 0.01$) that pushes the router toward uniform expert usage. The Switch Transformer formulation is $L_\text{aux} = \alpha\, N \sum_{i=1}^{N} f_i \cdot P_i$, where $f_i$ is the fraction of tokens dispatched to expert $i$ and $P_i$ is the mean router probability for expert $i$. This term is minimized when load is balanced, nudging the router away from collapse without dominating the language-modeling [Cross-Entropy Loss](/notes/ml-algorithms/core-concepts/cross-entropy-loss/).

2. **Expert capacity + token dropping.** Each expert gets a fixed buffer: `capacity = capacity_factor × (tokens_per_batch / N)`. A `capacity_factor` of 1.0–1.25 is common. If too many tokens route to one expert in a batch, the overflow tokens are **dropped** (they skip the FFN and pass through via the residual connection). Capacity keeps tensor shapes static for efficient batched GPU/TPU execution, at the cost of occasionally dropping tokens — higher capacity reduces drops but raises compute and memory.

DeepSeek-MoE adds two refinements now widely copied: **fine-grained experts** (many small experts instead of few large ones, for better specialization) and **shared experts** (one or more experts that *every* token always uses, capturing common knowledge so the routed experts can specialize). DeepSeek-V3 further uses an *auxiliary-loss-free* balancing trick (a learned per-expert bias added to router logits) to avoid the small accuracy hit that the auxiliary loss can cause.

## Variants and trade-offs

| System | Experts $N$ | Top-k | Active / Total params | Notes |
|---|---|---|---|---|
| GShard | up to 2048/layer | 2 | — | First large-scale MoE transformer; introduced capacity + aux loss |
| Switch Transformer | up to 1000s | **1** | — | Top-1 routing for simplicity/speed; showed top-1 works well |
| Mixtral 8x7B | 8 | 2 | ~13B / ~47B | Open-weight, strong quality at small active cost |
| DeepSeek-MoE / V3 | many fine-grained + shared | varies | DeepSeek-V3: 37B / 671B | Fine-grained + shared experts; aux-loss-free balancing |

**Top-1 vs top-2.** Top-1 (Switch) maximizes the sparsity ratio and minimizes compute and all-to-all communication; it can be slightly less expressive and more sensitive to routing mistakes. Top-2 (GShard, Mixtral) gives the router a "second opinion" and smoother gradients, at ~2x the FFN FLOPs of top-1. Top-2 is the common default for quality; top-1 when compute/communication is the binding constraint.

**Dense vs MoE FFN.** Dense FFN: simpler, uniform compute, easy to serve, but capacity is tied to compute. MoE FFN: more capacity per FLOP and better quality-per-active-param, but much higher memory footprint, balancing complexity, and distributed-systems overhead. Use dense when memory is tight or the model is small; use MoE when you want frontier quality under a compute budget and can afford the memory.

## Practical considerations

- **Memory dominates serving.** Even though only $k$ experts fire per token, *all* $N$ experts must sit in memory because any token in a batch might route to any expert. Mixtral 8x7B needs VRAM for the full ~47B parameters even though it computes like a 13B model. So MoE trades cheap compute for expensive memory — the opposite of most optimizations. See LLM Serving Platform and [Quantization for LLMs](/notes/ml-algorithms/language-models/quantization-for-llms/) (INT4/FP8 quantization is especially valuable here to fit experts in memory).
- **All-to-all communication.** In distributed training/inference, experts are sharded across devices (**expert parallelism**). Tokens must be shuffled to the device holding their chosen expert and the results shuffled back — two `all-to-all` collectives per MoE layer. This communication often becomes the bottleneck, not the FLOPs, and is sensitive to network topology.
- **Batch-size sensitivity.** Token dropping depends on batch composition; an unlucky batch can overload one expert. Inference with small/uneven batches can route inefficiently, so capacity factor is often raised at inference (or dropping disabled).
- **Training instability.** MoE routers can be unstable; common fixes include router z-loss (penalize large logits), bf16/fp32 router computation, and careful initialization.
- **Activation/normalization unchanged.** Experts are ordinary FFNs (typically SwiGLU); the gating uses softmax. See [Normalization and Activations in LLMs](/notes/ml-algorithms/language-models/normalization-and-activations-in-llms/).

## Related

- [LLM Architecture](/notes/ml-algorithms/language-models/llm-architecture/) — where the FFN/MoE sublayer sits in the transformer block
- [Scaling Laws](/notes/ml-algorithms/language-models/scaling-laws/) — the capacity-vs-compute relationship MoE exploits
- [Normalization and Activations in LLMs](/notes/ml-algorithms/language-models/normalization-and-activations-in-llms/) — the expert FFNs and gating internals
- LLM Serving Platform — memory and all-to-all serving costs
- [Quantization for LLMs](/notes/ml-algorithms/language-models/quantization-for-llms/) — fitting all experts in memory
- [Attention Variants and Efficiency](/notes/ml-algorithms/language-models/attention-variants-and-efficiency/) · [Transformers](/notes/ml-algorithms/deep-learning/transformers/) · [Softmax](/notes/ml-algorithms/core-concepts/softmax/) · [Cross-Entropy Loss](/notes/ml-algorithms/core-concepts/cross-entropy-loss/)
