---
layout: note
title: "Decoding Strategies"
description: "An autoregressive LLM only gives you a probability distribution over the next token; a decoding strategy decides which token to actually emit. Greedy picks the argmax (determini…"
note: true
note_collection: "ML algorithms"
note_section: "Language Models"
section_order: 6
note_order: 2
updated: 2026-06-07 03:58:34 -0700
keywords:
  - LLMs
  - Transformers
  - Inference
  - Optimization
  - Probability
math: true
mermaid: true
---
> The rules for turning an LLM's next-token probability distribution into actual generated text, trading off coherence against diversity. Related: [Softmax](/notes/ml-algorithms/core-concepts/softmax/) · [Reasoning and Test-Time Compute](/notes/ml-algorithms/language-models/reasoning-and-test-time-compute/) · [KV Cache and Inference Optimization](/notes/ml-algorithms/language-models/kv-cache-and-inference-optimization/) · [LLM Architecture](/notes/ml-algorithms/language-models/llm-architecture/)

## TL;DR

An autoregressive LLM only gives you a probability distribution over the next token; a **decoding strategy** decides which token to actually emit. **Greedy** picks the argmax (deterministic, fast, repetitive). **Beam search** keeps the top-$b$ partial sequences (great for low-entropy tasks like translation, dull for open-ended text). **Sampling** methods — **temperature**, **top-k**, and **top-p / nucleus** — draw randomly from the distribution and are the default for creative generation. The knobs all manage one trade-off: **coherence vs. diversity**.

## Why it matters

A trained LLM is a function that, given a prefix, outputs a vector of logits $z$ over the vocabulary (tens to hundreds of thousands of tokens). [Softmax](/notes/ml-algorithms/core-concepts/softmax/) turns that into a probability distribution $p$. But generation needs a single token at each step, fed back in autoregressively. **Decoding is the bridge between the distribution and the text** — and it lives entirely at inference time, changing nothing about the weights.

This matters because the *same model* can feel robotic, repetitive, broken, or brilliant depending purely on decoding settings. Two failure modes bracket the space:

- **Too greedy** → degenerate repetition ("the the the"), bland boilerplate, and myopic choices (locally-optimal tokens that paint the model into a corner).
- **Too random** → incoherent, off-topic, or hallucinated text as low-probability tokens get sampled.

Decoding is also where you enforce *structure* (valid JSON, a grammar) and where techniques like [Speculative Decoding and Distillation](/notes/ml-algorithms/language-models/speculative-decoding-and-distillation/) and the [KV Cache and Inference Optimization](/notes/ml-algorithms/language-models/kv-cache-and-inference-optimization/) hook in to make all of this fast. For reasoning models, sampling diversity is the substrate for [Reasoning and Test-Time Compute](/notes/ml-algorithms/language-models/reasoning-and-test-time-compute/) tricks like self-consistency.

## How it works

### From logits to probabilities

At step $t$, the model produces logits $z_i$ for each vocabulary token $i$. The **temperature-scaled softmax** is:

$$p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

where $T > 0$ is the **temperature**. Intuition:

- $T = 1$: the model's native distribution.
- $T < 1$ (e.g., 0.2): logits are divided by a small number → *amplified* gaps → distribution **sharpens** toward the argmax (greedier, more deterministic). $T \to 0$ is the limit of pure greedy.
- $T > 1$ (e.g., 1.5): gaps shrink → distribution **flattens** toward uniform (more random, more diverse).

Temperature is a *global* reshaping; the truncation methods below decide *which tokens are even eligible* before (or with) sampling.

### The core strategies

- **Greedy decoding**: $x_t = \arg\max_i p_i$. One forward pass per token, deterministic. Fast and fine for short factual answers, but prone to repetition and **myopia** — it never reconsiders, so an early locally-good token can doom the sequence.
- **Beam search**: maintain $b$ candidate sequences ("beams"). At each step, expand every beam by every token, score by cumulative log-probability, and keep the top $b$. Approximates the **maximum-likelihood sequence**, not just per-token greedy. Excellent for *low-entropy, closed-ended* tasks (Machine Translation, summarization) where one right answer dominates.
- **Top-k sampling**: keep only the $k$ highest-probability tokens, renormalize, and sample. Caps the "tail" of garbage tokens. Weakness: $k$ is fixed, so it's too restrictive when the model is genuinely uncertain (many good options) and too permissive when it's confident (one obvious option).
- **Top-p / nucleus sampling**: keep the smallest set of tokens whose **cumulative probability $\geq p$**, renormalize, and sample. The set size is **adaptive** — small when the model is confident, large when it's uncertain — which is why $p = 0.9$ is the workhorse default.
- **Min-p sampling**: keep tokens whose probability $\geq p_{\min} \times p_{\max}$ (a fraction of the top token's probability). Robust at high temperatures: it scales the cutoff with the model's confidence rather than to an absolute mass.

### The likelihood trap (why beam search fails open-ended)

For creative/open-ended generation, the highest-likelihood sequence is often a *bad* sequence. Human text is **not** the most probable text — natural language has a steady level of surprise. Beam search and low temperature relentlessly chase high probability, collapsing into bland, repetitive loops (the "neural text degeneration" result behind nucleus sampling). Sampling deliberately injects the surprise that makes text feel human.

<pre class="mermaid">
flowchart TD
    Z[&quot;Logits z (one per vocab token)&quot;] --&gt; T[&quot;Temperature scale: z / T&quot;]
    T --&gt; SM[&quot;Softmax -&gt; distribution p&quot;]
    SM --&gt; G[&quot;Greedy: argmax p&quot;]
    SM --&gt; B[&quot;Beam search: keep top-b sequences by sum log-p&quot;]
    SM --&gt; K[&quot;Top-k: keep k highest, renormalize&quot;]
    SM --&gt; P[&quot;Top-p: keep cumulative &gt;= p, renormalize&quot;]
    K --&gt; S[&quot;Sample 1 token&quot;]
    P --&gt; S
    G --&gt; OUT[&quot;Next token&quot;]
    B --&gt; OUT
    S --&gt; OUT
    OUT -. &quot;append, repeat autoregressively&quot; .-&gt; Z
</pre>
### Anti-repetition controls

Layered on top of any strategy:

- **Repetition penalty**: divide (or for negative logits, multiply) the logit of any already-seen token by $r > 1$ before softmax, discouraging reuse.
- **Frequency / presence penalty**: subtract a term proportional to how often (frequency) or whether at all (presence) a token has appeared — the OpenAI-style additive form.
- **No-repeat n-gram**: hard-ban any n-gram (e.g., 3-gram) that has already occurred, eliminating verbatim loops.

### Constrained / structured decoding

To force valid output (JSON, a regex, a formal grammar), apply **logit masking**: at each step, set the logits of all tokens that would violate the constraint to $-\infty$, so they get zero probability after softmax. A grammar/state-machine (e.g., GBNF grammars, or a compiled JSON schema as in Outlines / XGrammar) drives the mask. This **guarantees** structural validity regardless of the underlying strategy — invaluable for tool-calling and API responses ([Tool Use and Agents](/notes/ml-algorithms/language-models/tool-use-and-agents/)).

## Variants / Trade-offs

| Strategy | Determinism | Diversity | Eligible-set size | Best for | Watch out for |
|---|---|---|---|---|---|
| **Greedy** | Deterministic | None | 1 (argmax) | Short factual Q&A, code | Repetition, myopia |
| **Beam search** ($b$=4–8) | Deterministic | Low | top-$b$ sequences | Translation, summarization | Dull/looping on open-ended text; slower |
| **Top-k** ($k$≈40) | Stochastic | Medium | fixed $k$ | General sampling | Fixed cutoff ignores model confidence |
| **Top-p / nucleus** ($p$≈0.9) | Stochastic | Medium–high | adaptive | Creative, chat, default | Tune jointly with $T$ |
| **Min-p** ($p_{\min}$≈0.05) | Stochastic | Tunable | adaptive (rel. to top) | High-temperature creative | Newer, less standardized |
| **Temperature** ($T$) | Modifier | — | — | Knob for any sampler | $T$ too high → incoherence |

**Typical defaults**: factual / code / extraction → greedy or low temp ($T \approx 0$–$0.3$). Creative / chat → $T \approx 0.7$ with **top-p $0.9$** (and often top-k ~40 as a backstop). Reasoning self-consistency → moderate temp (~0.6–0.8) to get *diverse* sampled chains, then majority-vote.

## Practical considerations

- **Order of operations matters**: most stacks apply repetition/frequency penalties → temperature → top-k → top-p → sample. Different libraries differ slightly; check before debugging "weird" output.
- **Reproducibility**: only greedy (and beam) are deterministic. For reproducible sampling you must fix the RNG **seed** — and even then, batching, GPU non-determinism, and FP rounding can perturb results.
- **Temperature 0 ≈ greedy**, but implementations short-circuit to argmax to avoid divide-by-zero. "Deterministic" still isn't bit-identical across hardware.
- **Cost**: greedy/sampling are one forward pass per token. Beam search runs $b$ hypotheses → ~$b\times$ the compute and KV-cache memory ([KV Cache and Inference Optimization](/notes/ml-algorithms/language-models/kv-cache-and-inference-optimization/)), a big reason chat serving rarely uses beams.
- **Structured decoding overhead**: computing logit masks per step can stall the GPU; modern engines (XGrammar, vLLM's guided decoding) precompile grammars to keep masking cheap.
- **Interaction with the model**: instruction-tuned models are calibrated for sampling around $T\approx0.7$–$1.0$; pushing $T$ very high breaks even a good model. Decoding can't fix a wrong model — it only shapes *how* the distribution is exposed.
- **Self-consistency / best-of-n**: sampling $n$ outputs and selecting (vote or reward model) is a cheap, decoding-level boost to quality — the entry point to [Reasoning and Test-Time Compute](/notes/ml-algorithms/language-models/reasoning-and-test-time-compute/).

## Related

- Siblings: [LLM Architecture](/notes/ml-algorithms/language-models/llm-architecture/) · [KV Cache and Inference Optimization](/notes/ml-algorithms/language-models/kv-cache-and-inference-optimization/) · [Speculative Decoding and Distillation](/notes/ml-algorithms/language-models/speculative-decoding-and-distillation/) · [Reasoning and Test-Time Compute](/notes/ml-algorithms/language-models/reasoning-and-test-time-compute/) · [Tool Use and Agents](/notes/ml-algorithms/language-models/tool-use-and-agents/) · [Long Context](/notes/ml-algorithms/language-models/long-context/) · [Quantization for LLMs](/notes/ml-algorithms/language-models/quantization-for-llms/)
- Foundations: [Softmax](/notes/ml-algorithms/core-concepts/softmax/) · [Cross-Entropy Loss](/notes/ml-algorithms/core-concepts/cross-entropy-loss/) · [Transformers](/notes/ml-algorithms/deep-learning/transformers/)
- Systems: LLM Serving Platform · Machine Translation
