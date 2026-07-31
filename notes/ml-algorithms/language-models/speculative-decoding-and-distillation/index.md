---
layout: note
title: "Speculative Decoding and Distillation"
description: "Speculative decoding runs a small, fast draft model to guess the next $k$ tokens, then has the big target model verify all $k$ in a single parallel forward pass; you accept the…"
note: true
note_collection: "ML algorithms"
note_section: "Language Models"
section_order: 6
note_order: 19
updated: 2026-06-07 03:59:05 -0700
keywords:
  - LLMs
  - Transformers
  - Inference
  - Optimization
  - Probability
math: true
mermaid: true
---
> Two complementary ways to make LLM inference faster and cheaper — speculative decoding speeds up *one* model with a cheap draft-then-verify loop, while distillation trains a smaller model to imitate a bigger one. Related: [Decoding Strategies](/notes/ml-algorithms/language-models/decoding-strategies/), [KV Cache and Inference Optimization](/notes/ml-algorithms/language-models/kv-cache-and-inference-optimization/), [Quantization for LLMs](/notes/ml-algorithms/language-models/quantization-for-llms/), LLM Serving Platform

## TL;DR

**Speculative decoding** runs a small, fast *draft* model to guess the next $k$ tokens, then has the big *target* model verify all $k$ in a **single parallel forward pass**; you accept the longest matching prefix and a rejection-sampling correction guarantees the output is distributed *exactly* as if you had sampled from the target alone — typically 2–3x faster with **no quality loss**. **Knowledge distillation** instead trains a small *student* model to match a large *teacher's* soft probability distribution (via KL divergence), producing a permanently smaller, cheaper serving model. The first changes *how* you decode; the second changes *which* model you deploy. They stack: distill a small model, then use it as your draft.

## Why it matters

Autoregressive decoding is the dominant cost of serving LLMs, and it is fundamentally serial: token $t+1$ depends on token $t$, so you cannot start it until the previous one is done. Crucially, generating one token is **memory-bandwidth-bound**, not compute-bound — each step must stream the entire weight matrix (and the [KV cache](/notes/ml-algorithms/language-models/kv-cache-and-inference-optimization/)) from HBM through the cores, but only does a tiny amount of arithmetic on a single token. The GPU's FLOPs sit mostly idle. This is the inefficiency both techniques attack from different angles:

- **Speculative decoding** exploits the idle compute. Verifying $k$ proposed tokens is *one* forward pass over $k$ positions — roughly the same memory traffic as a single decode step, but now the FLOPs are actually used. You amortize a fixed memory cost over many tokens.
- **Distillation** shrinks the model itself, so every step moves fewer bytes. A 7B student that approximates a 70B teacher is ~10x cheaper to serve per token.

Both are central to production serving (see LLM Serving Platform) and complement [quantization](/notes/ml-algorithms/language-models/quantization-for-llms/) (fewer bits/weight) and KV-cache tricks (less memory traffic per step).

## How it works

### Part 1 — Speculative decoding

**Setup.** A large *target* model $p$ (the one whose distribution you want) and a small *draft* model $q$ that is much cheaper to run. Per round:

1. **Draft.** Run $q$ autoregressively to propose $k$ candidate tokens $x_1,\dots,x_k$, recording each draft probability $q(x_i\mid\text{context})$.
2. **Verify.** Run $p$ **once** over the whole proposed sequence. Because the tokens already exist, $p$ scores all $k+1$ positions in a single parallel forward pass, yielding $p(x_i\mid\text{context})$ for each.
3. **Accept / reject.** Walk left to right. Accept token $x_i$ with probability $\min\!\left(1,\ \dfrac{p(x_i)}{q(x_i)}\right)$. If $p$ agrees ($p\ge q$) the token is always accepted; otherwise it may be rejected.
4. **Correct on first rejection.** At the first rejected position, resample a token from the **adjusted residual distribution** $p_{\text{adj}}(x)=\dfrac{\big(p(x)-q(x)\big)_+}{\sum_{x'}\big(p(x')-q(x')\big)_+}$ (where $(\cdot)_+=\max(0,\cdot)$). Discard the rest of the draft.
5. **Free bonus token.** If *all* $k$ drafts are accepted, the same verification pass already gives you $p$ at position $k+1$, so you sample one extra token for free.

**Why it's exact.** The accept rule plus the residual resample form a **rejection/importance-sampling** scheme whose stationary output distribution is provably identical to sampling from $p$ directly — for any $q$. The draft model never affects *correctness*, only *speed*. A bad draft just means more rejections (slower), never wrong tokens.

**Speedup math.** Let $\alpha$ be the per-token acceptance rate (how often the draft matches the target). The expected number of tokens produced per verification round is $\dfrac{1-\alpha^{k+1}}{1-\alpha}$. Higher $\alpha$ → more accepted tokens per expensive target pass. A draft that matches the target 80% of the time with $k=4$ yields roughly 3 tokens per target pass. The trade-off: a *bigger* draft model has higher $\alpha$ but costs more per draft step, so there's a sweet spot (drafts are usually 10–20x smaller than the target).

<pre class="mermaid">
flowchart TD
    N1[&quot;Context so far&quot;] --&gt; N2[&quot;Draft model q&lt;br/&gt;propose k tokens&quot;]
    N2 --&gt; N3[&quot;Target model p&lt;br/&gt;verify all k in ONE pass&quot;]
    N3 --&gt; N4{&quot;For each token&lt;br/&gt;accept with min(1, p/q)?&quot;}
    N4 --&gt;|&quot;all k accepted&quot;| N5[&quot;Emit k tokens&lt;br/&gt;+ 1 free bonus from p&quot;]
    N4 --&gt;|&quot;first rejection at i&quot;| N6[&quot;Emit accepted prefix&lt;br/&gt;resample token i from residual&quot;]
    N5 --&gt; N1
    N6 --&gt; N1
</pre>
**Variants** differ mainly in *where the draft comes from*:

- **Two-model (vanilla).** Separate small draft model — classic Leviathan/Chen 2023 setup.
- **Self-speculative.** One model drafts with some of its own layers skipped, then verifies with the full stack — no second model to host.
- **Medusa.** Bolt extra lightweight "heads" onto the target that each predict a *future* position ($t{+}1, t{+}2, \dots$) in parallel; a tree of candidates is verified at once. No separate draft model.
- **EAGLE / EAGLE-2/3.** Draft at the *feature* (hidden-state) level rather than the token level, giving much higher acceptance ($\alpha$); among the strongest methods as of 2024–2025.
- **Lookahead / n-gram / prompt-lookup.** Use Jacobi-style parallel guesses or copy candidate spans from the prompt — zero training, great for code/RAG where output echoes input.

### Part 2 — Knowledge distillation

Train a small **student** $p_S$ to imitate a large **teacher** $p_T$. Instead of (or alongside) hard one-hot labels, the student matches the teacher's full *soft* distribution, which carries "dark knowledge" — the relative probabilities of wrong answers encode similarity structure the hard label throws away.

**Loss (Hinton).** Soften both distributions with a temperature $T$ in the [softmax](/notes/ml-algorithms/core-concepts/softmax/): $\sigma(z)_i = \dfrac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}$. The distillation loss combines a KL term against the soft teacher with the ordinary [cross-entropy](/notes/ml-algorithms/core-concepts/cross-entropy-loss/) against true labels:

$$\mathcal{L} = \alpha\, T^2 \cdot \mathrm{KL}\!\big(\sigma_T(z_T)\,\Vert\,\sigma_T(z_S)\big) \;+\; (1-\alpha)\,\mathrm{CE}(y,\, p_S)$$

The $T^2$ factor rescales gradient magnitudes (soft targets have small gradients $\propto 1/T^2$). Higher $T$ exposes more of the teacher's low-probability structure.

**Word-level vs sequence-level.** The above is **word/token-level** (match the teacher's per-position distribution on a fixed corpus). **Sequence-level distillation** instead has the teacher *generate* outputs (e.g., greedy/sampled completions) and trains the student to reproduce those sequences — this aligns the student with the teacher's whole-sequence behavior and is the basis of modern "distill from a strong model's transcripts" recipes. **Reasoning distillation** (e.g., DeepSeek-R1's distilled Qwen/Llama variants) is exactly this: a strong reasoning teacher generates long chain-of-thought traces, and small students are fine-tuned on them — often recovering most of the teacher's reasoning skill at a fraction of the size (see [Reasoning and Test-Time Compute](/notes/ml-algorithms/language-models/reasoning-and-test-time-compute/)).

## Speculative decoding vs distillation — trade-offs

| Dimension | Speculative decoding | Knowledge distillation |
|---|---|---|
| What it changes | *How* you decode (runtime) | *Which* model you serve (offline) |
| Output quality | **Identical** to target (provably exact) | Approximate — student ≤ teacher |
| Cost paid | Extra draft compute + memory at serve time | One-time training cost |
| Latency win | 2–3x (more with EAGLE/Medusa) | Lower per-token cost from smaller model |
| Needs training? | Vanilla: no; Medusa/EAGLE: light training | Yes (full student training/fine-tune) |
| Failure mode | Low acceptance → little speedup | Capacity gap → quality drop |
| Best when | Latency-critical, must preserve exact dist. | Want a permanently cheap serving model |

They are **not** mutually exclusive: distill a small model, [quantize](/notes/ml-algorithms/language-models/quantization-for-llms/) it, and use it as the speculative draft for the big target — three multipliers on the same pipeline.

## Practical considerations

- **Acceptance rate is everything** for speculative decoding. It depends on draft/target *alignment*, not just draft size — a draft from the same family/training data accepts far more. Measure tokens-accepted-per-pass on your real traffic.
- **Batching tension.** Speculative decoding shines at *low batch size / low latency* (where decode is memory-bound). At high batch sizes the target is already compute-bound, so the speedup shrinks — production servers sometimes switch it on only under low load.
- **Greedy vs sampling.** With greedy decoding, "acceptance" is just exact-match; with temperature/top-p sampling you need the full $\min(1,p/q)$ rule to stay exact. Mind that draft and target must share the same tokenizer/vocab (see [Tokenization](/notes/ml-algorithms/language-models/tokenization/)).
- **Memory cost.** You host *two* models (or extra heads) plus two KV caches; on memory-constrained GPUs that can offset the win.
- **Distillation gotchas.** A too-large teacher→student capacity gap distills poorly ("capacity gap"); intermediate-size teacher assistants help. Sequence-level distillation can amplify teacher biases/hallucinations since the student trains on teacher text. Always evaluate the student independently (see [LLM Evaluation](/notes/ml-algorithms/language-models/llm-evaluation/)).
- **Defaults.** Typical setups: draft 10–20x smaller, $k=4$–$8$; distillation temperature $T\in[2,5]$, $\alpha\approx 0.5$–$0.9$. EAGLE/Medusa are common production choices; sequence-level distillation underlies most small "instruct" models shipped today.

## Related

- Siblings: [Decoding Strategies](/notes/ml-algorithms/language-models/decoding-strategies/) · [KV Cache and Inference Optimization](/notes/ml-algorithms/language-models/kv-cache-and-inference-optimization/) · [Quantization for LLMs](/notes/ml-algorithms/language-models/quantization-for-llms/) · [Reasoning and Test-Time Compute](/notes/ml-algorithms/language-models/reasoning-and-test-time-compute/) · [Supervised Fine-Tuning](/notes/ml-algorithms/language-models/supervised-fine-tuning/) · [Tokenization](/notes/ml-algorithms/language-models/tokenization/) · [LLM Evaluation](/notes/ml-algorithms/language-models/llm-evaluation/)
- Systems: LLM Serving Platform
- Foundations: [Softmax](/notes/ml-algorithms/core-concepts/softmax/) · [Cross-Entropy Loss](/notes/ml-algorithms/core-concepts/cross-entropy-loss/)
