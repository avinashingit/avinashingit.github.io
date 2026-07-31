---
layout: note
title: "LLM Evaluation"
description: "Evaluating an LLM is hard because most useful outputs are open-ended — there is no single correct string to match. So we triangulate: intrinsic measures like perplexity (how wel…"
note: true
note_collection: "ML algorithms"
note_section: "Language Models"
section_order: 6
note_order: 6
updated: 2026-06-07 04:01:53 -0700
keywords:
  - LLMs
  - Transformers
  - Evaluation
  - Optimization
  - Retrieval
math: true
mermaid: true
---
> How we measure whether a large language model is good — across knowledge, reasoning, code, safety, and open-ended chat — using a mix of benchmarks, model judges, and human preference. Related: [Metrics](/notes/ml-algorithms/core-concepts/metrics/), [Pretraining and Language Modeling](/notes/ml-algorithms/language-models/pretraining-and-language-modeling/), [Hallucination and Safety](/notes/ml-algorithms/language-models/hallucination-and-safety/), [Reasoning and Test-Time Compute](/notes/ml-algorithms/language-models/reasoning-and-test-time-compute/)

## TL;DR
Evaluating an LLM is hard because most useful outputs are open-ended — there is no single correct string to match. So we triangulate: **intrinsic** measures like perplexity (how well the model predicts held-out text), **benchmarks** with known answers (MMLU, GSM8K, HumanEval, SWE-bench), **LLM-as-judge** for scalable grading of free-form responses (MT-Bench, AlpacaEval), and **human preference arenas** (Chatbot Arena Elo). No single number suffices; the discipline is using a balanced suite while fighting contamination, saturation, and overfitting to the test.

## Why it matters
A model's loss curve tells you it trained, not that it's *useful*. Capabilities are broad (recall, math, coding, instruction-following, safety) and the most valuable behaviors — writing a good email, debugging a repo, refusing a harmful request — produce free-form text where many answers are acceptable and many are subtly wrong. Evaluation is the feedback signal that drives the entire LLM stack: it decides which checkpoint ships, which fine-tuning recipe ([Supervised Fine-Tuning](/notes/ml-algorithms/language-models/supervised-fine-tuning/), [RLHF and Preference Optimization](/notes/ml-algorithms/language-models/rlhf-and-preference-optimization/)) helped, whether [Retrieval-Augmented Generation](/notes/ml-algorithms/language-models/retrieval-augmented-generation/) reduced hallucination, and where the model is unsafe. Get eval wrong and you optimize for the wrong thing — chasing a leaderboard number that doesn't transfer to real users.

## How it works

### Intrinsic: perplexity
The cheapest signal comes straight from language modeling. On a held-out corpus, compute the average negative log-likelihood (NLL) per token and exponentiate:

$$\text{PPL} = \exp\!\Big(-\tfrac{1}{N}\sum_{i=1}^{N}\log p_\theta(x_i \mid x_{<i})\Big)$$

where $p_\theta$ is the model, $x_i$ the $i$-th token, $x_{<i}$ its context, and $N$ the token count. Lower is better; perplexity is just $\exp$ of the [Cross-Entropy Loss](/notes/ml-algorithms/core-concepts/cross-entropy-loss/). It correlates with general capability and is great for tracking [Pretraining and Language Modeling](/notes/ml-algorithms/language-models/pretraining-and-language-modeling/) runs, but it measures *fluency / next-token fit*, not usefulness, factuality, or instruction-following — and it isn't comparable across different tokenizers ([Tokenization](/notes/ml-algorithms/language-models/tokenization/)).

### Benchmarks: tasks with known answers
For measurable skills we use curated test sets, usually scored by exact match or multiple-choice accuracy. Common families:

<pre class="mermaid">
flowchart TD
  M[&quot;Model under test&quot;] --&gt; H[&quot;Eval harness&quot;]
  H --&gt;|&quot;prompt + few-shot&quot;| K[&quot;Knowledge / Reasoning&lt;br/&gt;MMLU, GPQA, ARC&quot;]
  H --&gt;|&quot;chain-of-thought&quot;| MA[&quot;Math&lt;br/&gt;GSM8K, MATH&quot;]
  H --&gt;|&quot;unit tests&quot;| C[&quot;Code&lt;br/&gt;HumanEval, MBPP, SWE-bench&quot;]
  H --&gt;|&quot;judge model&quot;| J[&quot;Open-ended&lt;br/&gt;MT-Bench, AlpacaEval&quot;]
  H --&gt;|&quot;human votes&quot;| AR[&quot;Arena Elo&lt;br/&gt;Chatbot Arena&quot;]
  K --&gt; AGG[&quot;Aggregate scorecard&quot;]
  MA --&gt; AGG
  C --&gt; AGG
  J --&gt; AGG
  AR --&gt; AGG
  AGG --&gt; D[&quot;Ship / iterate decision&quot;]
</pre>
The harness fixes the prompt template, the number of few-shot examples, and the answer-extraction logic — all of which materially shift scores, so reported numbers are only comparable under identical settings.

### LLM-as-judge
For free-form outputs (a chat reply, a summary), there's no gold string. A strong model (e.g., a frontier model used as grader) scores a response on a rubric, or does a **pairwise** comparison: given prompt + answer A + answer B, pick the better one. This is how MT-Bench and AlpacaEval scale beyond hand grading. It's cheap and fast but biased (see below).

### Human eval & arenas
The gold standard for "which model do people prefer" is human pairwise voting. **Chatbot Arena** shows two anonymous models the same prompt and asks a user to pick the winner, then converts votes into **Elo** ratings (the chess system): each model has a rating $R$, and the expected win probability of A over B is

$$E_A = \frac{1}{1 + 10^{(R_B - R_A)/400}}.$$

Ratings update toward observed outcomes. Elo captures holistic preference but is slow, expensive, and reflects vibes/formatting as much as correctness.

## Variants / Types / Trade-offs

| Method | What it measures | Cost | Strength | Weakness |
|---|---|---|---|---|
| Perplexity (intrinsic) | Held-out NLL / next-token fit | Very low | Cheap, tracks pretraining | Not usefulness; tokenizer-dependent |
| MMLU / GPQA | Broad knowledge & reasoning (MC) | Low | Objective, broad | Saturated; contamination; MC ≠ generation |
| GSM8K / MATH | Multi-step math | Low | Tests reasoning chains | Memorizable; format-sensitive |
| HumanEval / MBPP | Function-level code (pass@k) | Low | Executable, objective | Small, narrow, leaked |
| SWE-bench | Repo-level bug fixing (real PRs) | High | Realistic agentic coding | Expensive; env setup |
| BIG-bench / HELM | Aggregated multi-task suites | Medium | Coverage + standardized | Heavy to run |
| LLM-as-judge (MT-Bench, AlpacaEval) | Open-ended quality (pairwise) | Medium | Scalable for free-form | Position / verbosity / self bias |
| Chatbot Arena Elo | Human preference | High | Holistic, hard to game | Slow; subjective; vibes |
| BLEU / ROUGE | n-gram overlap (MT, summ.) | Low | Automatic, reproducible | Surface-only; penalizes valid paraphrases |
| RAGAS | Faithfulness / groundedness (RAG) | Medium | Targets hallucination | Judge-dependent |

**When to use which:** perplexity for pretraining health; closed-answer benchmarks for capability gates; LLM-as-judge for fast iteration on chat quality; human arena for final preference; task-specific metrics (BLEU/ROUGE, pass@k, RAGAS) when the deployment is that task.

## Practical considerations

**Contamination & saturation.** The biggest threat: test data leaks into the giant web-scraped training corpus, so a model "knows" the answers. Detect with n-gram overlap scans, canary strings, or held-out private splits; mitigate with fresh/private test sets (GPQA, SWE-bench, live arenas). Meanwhile popular benchmarks **saturate** — MMLU is near ceiling for frontier models, so it stops discriminating, pushing the field to harder sets (GPQA, MMLU-Pro, FrontierMath).

**LLM-as-judge biases** and mitigations:
- *Position bias* — judges favor the first (or second) answer → swap A/B order and average both directions.
- *Verbosity bias* — longer answers score higher → length-control (AlpacaEval 2.0) or penalize length.
- *Self-preference* — a judge favors its own family's style → use a different/ensemble judge; calibrate against human votes.

**Task-specific metrics.** BLEU/ROUGE measure n-gram overlap with references for Machine Translation and summarization; they're cheap but reward surface matches and punish correct paraphrases, so pair them with human or judge eval. For code, run the code: **pass@k** estimates the chance at least one of $k$ samples passes all unit tests:

$$\text{pass@}k = \mathbb{E}\Big[\,1 - \tfrac{\binom{n-c}{k}}{\binom{n}{k}}\Big]$$

where $n$ samples are drawn and $c$ pass — an unbiased estimator that avoids high-variance small-$k$ sampling.

**RAG / agent eval.** Beyond answer accuracy, measure **faithfulness** (is every claim supported by retrieved context?), **groundedness/answer relevance**, and **context precision/recall** — RAGAS automates these with a judge. Ties directly to [Retrieval-Augmented Generation](/notes/ml-algorithms/language-models/retrieval-augmented-generation/) and [Hallucination and Safety](/notes/ml-algorithms/language-models/hallucination-and-safety/).

**Safety / red-team evals.** Run refusal and jailbreak suites, toxicity, and adversarial prompts to measure harmful-output rates; red-teaming actively searches for failures rather than scoring a fixed set.

**The cardinal rule:** use a *mix* and don't optimize directly on a public benchmark — that's overfitting to the test (Goodhart's law). Keep a private eval set, report variance across seeds/prompts, and weight real-user signals.

## Related
- Foundational: [Metrics](/notes/ml-algorithms/core-concepts/metrics/) · [Cross-Entropy Loss](/notes/ml-algorithms/core-concepts/cross-entropy-loss/) · [Softmax](/notes/ml-algorithms/core-concepts/softmax/)
- Siblings: [Pretraining and Language Modeling](/notes/ml-algorithms/language-models/pretraining-and-language-modeling/) · [Hallucination and Safety](/notes/ml-algorithms/language-models/hallucination-and-safety/) · [Retrieval-Augmented Generation](/notes/ml-algorithms/language-models/retrieval-augmented-generation/) · [Reasoning and Test-Time Compute](/notes/ml-algorithms/language-models/reasoning-and-test-time-compute/) · [Supervised Fine-Tuning](/notes/ml-algorithms/language-models/supervised-fine-tuning/) · [RLHF and Preference Optimization](/notes/ml-algorithms/language-models/rlhf-and-preference-optimization/) · [Tokenization](/notes/ml-algorithms/language-models/tokenization/)
- System design: Machine Translation
