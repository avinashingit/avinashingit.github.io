---
layout: note
title: "Scaling Laws"
description: "Scaling laws say that an LLM's test loss decreases predictably — as a power law — as you grow the number of parameters $N$, training tokens $D$, and compute $C$. Kaplan et al. (…"
note: true
note_collection: "ML algorithms"
note_section: "Language Models"
section_order: 6
note_order: 18
updated: 2026-06-07 03:58:26 -0700
keywords:
  - LLMs
  - Transformers
  - Probability
  - Training
  - Deep Learning
math: true
mermaid: true
---
> Neural scaling laws are the empirical finding that LLM test loss falls as a smooth **power law** in model size, data, and compute — letting you predict and budget capability before you train. Related: [Pretraining and Language Modeling](/notes/ml-algorithms/language-models/pretraining-and-language-modeling/), [LLM Architecture](/notes/ml-algorithms/language-models/llm-architecture/), [Mixture of Experts](/notes/ml-algorithms/language-models/mixture-of-experts/), [Cross-Entropy Loss](/notes/ml-algorithms/core-concepts/cross-entropy-loss/)

## TL;DR

Scaling laws say that an LLM's test loss decreases predictably — as a power law — as you grow the number of parameters $N$, training tokens $D$, and compute $C$. Kaplan et al. (2020) established the power-law form; Chinchilla (Hoffmann et al., 2022) corrected the **allocation**: for a fixed compute budget you should scale $N$ and $D$ roughly **equally**, about **20 training tokens per parameter**, meaning GPT-3-era models were badly undertrained. The practical payoff is that you can fit a curve on small, cheap runs and extrapolate the loss (and rough capability) of a much larger model before paying for it.

## Why it matters

Training a frontier LLM costs millions of dollars and weeks of GPU time; you get essentially one shot. Without a predictive theory, choosing how big to make the model and how much data to feed it would be guesswork. Scaling laws turn that into engineering: run a sweep of small models, fit a power law, and read off the configuration that minimizes loss for your budget.

They also reframed the whole field. Before Chinchilla, the instinct was "bigger model = better," so labs raced to add parameters (GPT-3 at 175B, Gopher at 280B) while keeping data roughly fixed. Chinchilla showed this was a misallocation: those models were **too big for the data they saw**. A smaller model trained on more tokens could beat them at the same compute — **Chinchilla 70B outperformed Gopher 280B** despite being 4x smaller. This is one of the most consequential empirical results in modern deep learning, and it directly shapes how every lab now sizes its pretraining runs (see [Pretraining and Language Modeling](/notes/ml-algorithms/language-models/pretraining-and-language-modeling/)).

## How it works

**The power law.** Empirically, when one resource is the bottleneck and the others are abundant, test loss $L$ follows a power law plus an irreducible floor:

$$L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N} + L_\infty$$

Here $N$ is parameter count, $N_c$ and $\alpha_N$ are fitted constants, and $L_\infty$ is the **irreducible loss** (the entropy of natural language you can never model away). Analogous laws hold in $D$ (tokens) and $C$ (compute). On a log-log plot these are straight lines — that linearity is what makes extrapolation trustworthy. The exponents are small (≈0.05–0.1 for Transformers), so loss falls slowly: each constant-factor drop costs an order of magnitude more resources.

**The compute approximation.** A standard rule of thumb estimates training compute in floating-point operations (FLOPs) as:

$$C \approx 6 N D$$

where $N$ = parameters and $D$ = tokens. The factor 6 comes from roughly 2 FLOPs per parameter for the forward pass plus ~4 for the backward pass, applied to every token. This simple identity is the lever for all compute-optimal analysis: it ties the three quantities together so that fixing $C$ defines a trade-off curve between $N$ and $D$.

**Compute-optimal allocation (the optimization).** Given a fixed budget $C$, you want to minimize loss subject to $C \approx 6ND$. Kaplan's 2020 analysis concluded you should spend most extra compute on **bigger models** and relatively little on more data. Chinchilla redid the experiment more carefully — sweeping hundreds of runs and fitting the joint loss surface $L(N, D)$ — and found a very different answer: $N$ and $D$ should grow **at roughly the same rate**, $N \propto C^{0.5}$ and $D \propto C^{0.5}$. The optimal ratio lands near **20 tokens per parameter**.

<pre class="mermaid">
flowchart TD
    C[&quot;Fixed compute budget C (FLOPs)&quot;] --&gt; CON[&quot;Constraint: C = 6 N D&quot;]
    CON --&gt; K[&quot;Kaplan 2020 allocation&quot;]
    CON --&gt; CH[&quot;Chinchilla 2022 allocation&quot;]
    K --&gt; KR[&quot;Most extra compute to N&lt;br/&gt;D grows slowly&quot;]
    CH --&gt; CHR[&quot;Scale N and D equally&lt;br/&gt;about 20 tokens per param&quot;]
    KR --&gt; OUT1[&quot;Big, undertrained models&lt;br/&gt;(GPT-3 175B, Gopher 280B)&quot;]
    CHR --&gt; OUT2[&quot;Smaller, well-fed models&lt;br/&gt;(Chinchilla 70B beats Gopher 280B)&quot;]
</pre>
**Why the disagreement?** Kaplan held the learning-rate schedule roughly fixed and tuned it for large models, which penalized the small-model data points and biased the slope. Chinchilla used a properly tuned schedule per run (cosine decay matched to each token count) and a more thorough sweep, which is now the accepted methodology.

## Kaplan vs Chinchilla allocation

| Aspect | Kaplan et al. (2020) | Chinchilla (Hoffmann et al., 2022) |
|---|---|---|
| Core claim | Loss is a power law in $N$, $D$, $C$ | Same form, but joint $L(N,D)$ fitted carefully |
| Optimal scaling with $C$ | $N$ grows fast, $D$ slowly | $N \propto C^{0.5}$, $D \propto C^{0.5}$ (equal) |
| Tokens per parameter | Low (data treated as cheap to skimp on) | **~20 : 1** |
| Implication for GPT-3 | Roughly sized | **Undertrained** — too big, too few tokens |
| Methodology gap | Fixed/large-model LR schedule | Per-run tuned cosine LR schedule |
| Legacy | Sparked the "scale up params" era | Reset best practice toward more data |

**The inference-cost twist (Llama philosophy).** Compute-optimal minimizes *training* loss for a fixed *training* budget — but it ignores serving. If you will run inference on a model billions of times, inference cost dominates total cost of ownership, and a smaller model is cheaper *forever*. So Llama-family models deliberately train **well past** the 20:1 point — Llama and later open models use hundreds or thousands of tokens per parameter (e.g. Llama 3 8B trained on ~15T tokens, far beyond compute-optimal). You spend more training FLOPs than Chinchilla recommends to get a smaller model at the same quality, then save on every inference call. This is the right objective when you serve at scale (see [KV Cache and Inference Optimization](/notes/ml-algorithms/language-models/kv-cache-and-inference-optimization/)).

## Practical considerations

- **The recipe in practice.** Run a sweep of small models at several $(N, D)$ points, fit the power law, extrapolate to your target budget, then commit one large run. Loss extrapolation is reliable; predicting a *specific* downstream score is much harder.
- **Loss vs. capability.** Scaling laws predict pretraining **loss** well. They do **not** directly predict whether the model can pass a benchmark — that mapping from loss to task accuracy is noisy and metric-dependent.
- **Data wall.** At 20:1 (and especially at Llama-style ratios), frontier models need tens of trillions of high-quality tokens — and the web is finite. Repeating data, synthetic data, and data quality/filtering now matter as much as raw scale.
- **MoE changes the accounting.** Mixture-of-Experts models have many parameters but activate only a few per token, so "parameters" and "compute per token" decouple; the $C \approx 6ND$ rule uses **active** parameters. See [Mixture of Experts](/notes/ml-algorithms/language-models/mixture-of-experts/).
- **Quality and architecture shift the constants.** Better data, better tokenizer, and architectural tweaks (RoPE, GQA, SwiGLU) move the loss curve down without changing its power-law shape — the slope is robust, the offset is not.
- **Defaults.** If someone asks "how much data for a model of size $N$?", the compute-optimal answer is **~20N tokens**; the production answer is "more than that if you'll serve it heavily."

## Related

- [Pretraining and Language Modeling](/notes/ml-algorithms/language-models/pretraining-and-language-modeling/) — where the loss being scaled comes from
- [LLM Architecture](/notes/ml-algorithms/language-models/llm-architecture/) — what the parameters $N$ actually are
- [Mixture of Experts](/notes/ml-algorithms/language-models/mixture-of-experts/) — decoupling parameters from per-token compute
- [KV Cache and Inference Optimization](/notes/ml-algorithms/language-models/kv-cache-and-inference-optimization/) — why inference cost justifies overtraining
- [Quantization for LLMs](/notes/ml-algorithms/language-models/quantization-for-llms/) — another lever on serving cost
- [Cross-Entropy Loss](/notes/ml-algorithms/core-concepts/cross-entropy-loss/) — the loss the power law describes
- [LLM Evaluation](/notes/ml-algorithms/language-models/llm-evaluation/) — mapping loss to capability and the emergence debate
