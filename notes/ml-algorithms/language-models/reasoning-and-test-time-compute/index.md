---
layout: note
title: "Reasoning and Test-Time Compute"
description: "Test-time (inference-time) compute is the idea that you can trade extra computation at query time for higher accuracy, instead of only making the model bigger at training time.…"
note: true
note_collection: "ML algorithms"
note_section: "Language Models"
section_order: 6
note_order: 15
updated: 2026-06-07 04:01:49 -0700
keywords:
  - LLMs
  - Transformers
  - Inference
  - Optimization
  - Retrieval
math: true
mermaid: true
---
> Letting an LLM spend more tokens/compute "thinking" before it answers — via chain-of-thought, sampling-and-voting, search, or RL-trained long reasoning — reliably improves accuracy on hard problems, opening a new scaling axis beyond train-time scale. Related: [Prompt Engineering](/notes/ml-algorithms/language-models/prompt-engineering/), [Scaling Laws](/notes/ml-algorithms/language-models/scaling-laws/), [RLHF and Preference Optimization](/notes/ml-algorithms/language-models/rlhf-and-preference-optimization/), [Decoding Strategies](/notes/ml-algorithms/language-models/decoding-strategies/)

## TL;DR

Test-time (inference-time) compute is the idea that you can trade extra computation **at query time** for higher accuracy, instead of only making the model bigger at training time. The simplest form is **chain-of-thought (CoT)**: prompt the model to "reason step by step" so intermediate tokens carry the computation. You can amplify this by sampling many reasoning paths and voting (**self-consistency**), by scoring candidates with a **verifier** (best-of-N), or by searching over reasoning steps (**tree-of-thoughts**). Modern **reasoning models** (OpenAI o1/o3, DeepSeek-R1) are trained with large-scale RL to emit long internal "thinking" before answering, and their accuracy scales smoothly with the number of thinking tokens — a genuinely new axis next to [Scaling Laws](/notes/ml-algorithms/language-models/scaling-laws/).

## Why it matters

A standard LLM does a **fixed** amount of compute per output token: one forward pass, regardless of whether the question is "what is 2+2" or a competition math proof. That is a poor match for problem difficulty — hard problems need more steps of reasoning than a single pass can encode. Forcing the answer out immediately ("answer now") makes the model guess; giving it room to work makes it derive.

This reframes scaling. Pretraining scaling (Chinchilla, see [Scaling Laws](/notes/ml-algorithms/language-models/scaling-laws/)) says capability grows with parameters and tokens spent **before deployment**, which is enormously expensive and one-shot. Test-time compute adds an **orthogonal** lever: hold the weights fixed and spend more **at inference** to get better answers on the queries that warrant it. The headline empirical result from the o1 generation is that accuracy rises roughly **log-linearly with thinking tokens**, the same shape as train-time scaling curves. For agents, math, and coding — where a wrong answer is worthless — this is often a better dollar-for-accuracy trade than training a larger base model.

## How it works

**Chain-of-thought.** Instead of mapping question $q$ directly to answer $a$, the model generates a reasoning trace $r = (r_1, \dots, r_T)$ then the answer:

$$p(a, r \mid q) = \prod_{t} p\big(r_t \mid q, r_{<t}\big)\cdot p\big(a \mid q, r\big)$$

The trace tokens act as a scratchpad: each step conditions on the previous ones, so the model performs serial computation it could not fit into one pass. CoT is elicited zero-shot ("Let's think step by step"), few-shot (worked examples in the prompt — see [Prompt Engineering](/notes/ml-algorithms/language-models/prompt-engineering/)), or baked in by training.

**Self-consistency.** A single greedy trace can go wrong early. Self-consistency samples $N$ **independent** traces with temperature $>0$, extracts each final answer, and takes the **majority vote**:

$$\hat{a} = \arg\max_{a}\ \sum_{i=1}^{N}\mathbb{1}\big[a_i = a\big]$$

Different reasoning paths that converge on the same answer are mutually corroborating; errors are usually idiosyncratic and don't gain a majority. This monotonically improves accuracy with $N$ (diminishing returns), at $N\times$ the cost.

**Best-of-N and verifiers.** When answers aren't cleanly votable (e.g., free-form proofs), sample $N$ candidates and pick the best with a **scorer**: a trained **verifier**/reward model, or a **PRM** (process reward model) that scores each *step* rather than only the final answer. With a verifiable domain you can check directly — run the unit tests, evaluate the math expression — which is the strongest signal.

**Tree-of-thoughts / search.** Generalize the linear chain into a tree: branch at each reasoning step, evaluate partial states with a value/heuristic, and explore via BFS/DFS or beam search, backtracking from dead ends. This helps on problems needing exploration (puzzles, planning) but costs many more forward passes.

**Reasoning models (RL-trained).** The 2024–2025 leap: instead of prompting CoT, **train** the model to produce long internal reasoning. OpenAI o1/o3 and **DeepSeek-R1** use large-scale **RL with verifiable rewards (RLVR)** — reward = "is the final answer correct" on math/code — so the model learns *on its own* to think longer, self-check, and backtrack. DeepSeek used **GRPO** (Group Relative Policy Optimization), a critic-free variant of PPO that estimates advantage from the mean reward of a *group* of sampled answers, making RL cheap at scale (contrast with [RLHF and Preference Optimization](/notes/ml-algorithms/language-models/rlhf-and-preference-optimization/), which optimizes human preference, not correctness). The emergent behavior — long "wait, let me reconsider" traces — was not hand-engineered; it fell out of optimizing for correct answers.

<pre class="mermaid">
flowchart TD
    Q[&quot;Question q&quot;]
    Q --&gt; P1[&quot;CoT path 1&quot;]
    Q --&gt; P2[&quot;CoT path 2&quot;]
    Q --&gt; P3[&quot;CoT path N (temp &gt; 0)&quot;]
    P1 --&gt; A1[&quot;answer = 18&quot;]
    P2 --&gt; A2[&quot;answer = 18&quot;]
    P3 --&gt; A3[&quot;answer = 24&quot;]
    A1 --&gt; V[&quot;Majority vote / verifier score&quot;]
    A2 --&gt; V
    A3 --&gt; V
    V --&gt; F[&quot;Final answer: 18&quot;]
</pre>
**Inference-time scaling curve.** Empirically, $\text{accuracy} \approx \alpha + \beta\log(\text{thinking tokens})$ over a wide range — more deliberation buys more accuracy until it saturates. This holds both for **sequential** scaling (one longer chain) and **parallel** scaling (more samples + voting).

**Distillation.** Long reasoning is expensive to *serve*. You can **distill** it: generate high-quality traces from a strong reasoner (o3, R1), then supervised-fine-tune a small model on them. DeepSeek-R1's distilled 7B–70B models inherited much of the reasoning ability without the RL run — a cheap way to put reasoning into small models (see [Speculative Decoding and Distillation](/notes/ml-algorithms/language-models/speculative-decoding-and-distillation/), [Supervised Fine-Tuning](/notes/ml-algorithms/language-models/supervised-fine-tuning/)).

## Variants / Trade-offs

| Method | Extra compute | How it picks the answer | Best for | Cost |
|---|---|---|---|---|
| Direct answer | 1× | model emits answer | easy/lookup tasks | lowest |
| Zero/few-shot CoT | ~1× (longer output) | single trace | broad reasoning lift | low |
| Self-consistency | $N\times$ (parallel) | majority vote | clean, votable answers (math) | medium |
| Best-of-N + verifier | $N\times$ + scorer | highest verifier score | free-form / code | medium-high |
| Tree-of-thoughts / search | many× | value-guided search | exploration, planning | high |
| RL reasoning model (o1/R1) | adaptive, long chains | trained long CoT | hardest math/coding/agents | high per query |
| Distilled reasoner | 1× (short) | learned compact CoT | cheap deployment | low, some accuracy loss |

Rule of thumb: **sequential** thinking (longer chain) and **parallel** thinking (more samples) are complementary; for a fixed budget, hard problems favor more sequential depth, while parallel voting cheaply cleans up variance. A small model with lots of test-time compute can match a much larger model with little — but only up to a point.

## Practical considerations

- **Latency and cost explode.** A reasoning model can emit thousands of hidden thinking tokens, so a query may cost 5–50× a normal completion in tokens, money, and time-to-first-answer. Reserve it for queries that need it; route easy ones to a fast non-reasoning model.
- **Overthinking.** Reasoning models can ramble on trivial questions ("2+2") or talk themselves *out* of a correct answer. Newer APIs expose a **reasoning effort** / thinking-budget knob (low/medium/high) to cap deliberation.
- **Verifiable domains win.** RLVR and best-of-N shine where correctness is checkable (math, code, formal logic). In open-ended domains you need a learned verifier or PRM, which is itself fallible.
- **Hidden vs visible CoT.** Some providers (o1/o3) hide the raw thinking trace and bill for it as output tokens; you see only a summary. Don't rely on the visible trace being the literal computation, and never put secrets in prompts assuming the trace is private.
- **CoT is not a faithful explanation.** The stated reasoning can be post-hoc; a "right answer for wrong reasons" still happens. Treat traces as a performance tool, not a guaranteed audit trail (relevant to [Hallucination and Safety](/notes/ml-algorithms/language-models/hallucination-and-safety/)).
- **Decoding matters.** Self-consistency needs temperature $>0$ for diverse paths; a single best chain often wants near-greedy decoding (see [Decoding Strategies](/notes/ml-algorithms/language-models/decoding-strategies/)).

## Related

- [Prompt Engineering](/notes/ml-algorithms/language-models/prompt-engineering/) — how CoT / few-shot exemplars elicit reasoning.
- [RLHF and Preference Optimization](/notes/ml-algorithms/language-models/rlhf-and-preference-optimization/) — the RL machinery (PPO/GRPO) behind reasoning models.
- [Decoding Strategies](/notes/ml-algorithms/language-models/decoding-strategies/) — temperature/sampling that powers self-consistency.
- [Scaling Laws](/notes/ml-algorithms/language-models/scaling-laws/) — the train-time scaling axis this complements.
- [Speculative Decoding and Distillation](/notes/ml-algorithms/language-models/speculative-decoding-and-distillation/) — distilling and cheaply serving reasoning.
- [Supervised Fine-Tuning](/notes/ml-algorithms/language-models/supervised-fine-tuning/) — fine-tuning small models on distilled traces.
- [Tool Use and Agents](/notes/ml-algorithms/language-models/tool-use-and-agents/) — reasoning loops that call tools.
- [Hallucination and Safety](/notes/ml-algorithms/language-models/hallucination-and-safety/) — faithfulness limits of stated reasoning.
- [LLM Evaluation](/notes/ml-algorithms/language-models/llm-evaluation/) — measuring reasoning (math/code benchmarks).
