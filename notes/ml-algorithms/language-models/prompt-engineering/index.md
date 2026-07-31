---
layout: note
title: "Prompt Engineering"
description: "Prompt engineering is the practice of designing the input text so a pretrained LLM performs a task well. Because large models exhibit in-context learning (ICL) — learning a task…"
note: true
note_collection: "ML algorithms"
note_section: "Language Models"
section_order: 6
note_order: 13
updated: 2026-06-07 04:01:35 -0700
keywords:
  - LLMs
  - Transformers
  - Inference
  - Retrieval
  - Supervised Learning
math: true
mermaid: true
---
> Steering a frozen LLM to a task purely through the text of the prompt — instructions, examples, and reasoning scaffolds — with no weight updates. Related: [Supervised Fine-Tuning](/notes/ml-algorithms/language-models/supervised-fine-tuning/) · [Retrieval-Augmented Generation](/notes/ml-algorithms/language-models/retrieval-augmented-generation/) · [Reasoning and Test-Time Compute](/notes/ml-algorithms/language-models/reasoning-and-test-time-compute/) · [Tool Use and Agents](/notes/ml-algorithms/language-models/tool-use-and-agents/)

## TL;DR
Prompt engineering is the practice of designing the input text so a pretrained LLM performs a task well. Because large models exhibit **in-context learning (ICL)** — learning a task from instructions and a handful of examples in the prompt at inference time, with zero gradient updates — you can often go from zero-shot to strong few-shot performance just by editing text. **Chain-of-thought (CoT)** prompting (asking the model to reason step by step) unlocks multi-step and math reasoning by letting the model spend intermediate tokens as scratch compute. The catch: prompts are brittle — output quality is sensitive to wording, example order, and formatting.

## Why it matters
The cheapest, fastest way to adapt an LLM to a new task is to change the prompt, not the weights. Fine-tuning needs labeled data, a training pipeline, GPUs, and a deployment cycle; a prompt edit ships in seconds. This is viable only because pretraining produces an *emergent* capability: a model trained purely to predict the next token (see [Pretraining and Language Modeling](/notes/ml-algorithms/language-models/pretraining-and-language-modeling/)) learns enough latent task structure that it can be conditioned on a few examples and "figure out" what you want.

In the LLM stack, prompting sits at the **inference/application** layer and is the first tool you reach for. It is also the substrate everything else is built on: [Retrieval-Augmented Generation](/notes/ml-algorithms/language-models/retrieval-augmented-generation/) injects retrieved context *into the prompt*; agents and [Tool Use and Agents](/notes/ml-algorithms/language-models/tool-use-and-agents/) are prompting patterns that interleave reasoning with tool calls; reasoning models in [Reasoning and Test-Time Compute](/notes/ml-algorithms/language-models/reasoning-and-test-time-compute/) internalize CoT that prompting first surfaced. Mastering prompts is the foundation for all of these.

## How it works

**In-context learning.** Given a prompt that contains a task description and/or $k$ input-output demonstrations, the model conditions its next-token distribution on that context. Formally, with demonstrations $(x_1,y_1),\dots,(x_k,y_k)$ and a query $x_q$, the model computes

$$ p_\theta\big(y \mid x_1,y_1,\dots,x_k,y_k,\,x_q\big) $$

with parameters $\theta$ **fixed**. No backprop, no [Gradient Descent](/notes/ml-algorithms/core-concepts/gradient-descent/) — the "learning" is the forward pass attending over the examples. $k=0$ is **zero-shot**; $k>0$ is **few-shot** (typically 1–32 examples). Few-shot helps because demonstrations (a) pin down the exact output format, (b) disambiguate the task, and (c) supply the label space and style the model should imitate.

**Chain-of-thought.** Instead of mapping $x_q \to y$ directly, CoT elicits an intermediate reasoning trace $r$ before the answer: $x_q \to r \to y$. The trigger can be as simple as appending *"Let's think step by step"* (zero-shot CoT) or showing worked examples that include reasoning (few-shot CoT). It works because a Transformer does a fixed amount of compute per generated token; producing more tokens lets the model externalize intermediate results into the context window and attend back to them, effectively turning a hard one-step problem into many easy steps. CoT mainly helps large models on multi-step tasks (arithmetic, logic, multi-hop QA) and can *hurt* trivial tasks by adding noise.

**Prompt anatomy.** A production prompt is usually assembled from components:

<pre class="mermaid">
flowchart LR
  SYS[&quot;System / role&lt;br/&gt;(persona, rules)&quot;]
  TASK[&quot;Task instruction&lt;br/&gt;(what to do)&quot;]
  CTX[&quot;Context&lt;br/&gt;(docs, retrieved data)&quot;]
  EX[&quot;Few-shot examples&lt;br/&gt;(input to output)&quot;]
  FMT[&quot;Output constraints&lt;br/&gt;(JSON, schema, length)&quot;]
  Q[&quot;User query&quot;]
  M[&quot;LLM&lt;br/&gt;(frozen weights)&quot;]
  OUT[&quot;Completion&quot;]
  SYS --&gt; M
  TASK --&gt; M
  CTX --&gt; M
  EX --&gt; M
  FMT --&gt; M
  Q --&gt; M
  M --&gt; OUT
</pre>
The model sees the concatenation; ordering and delimiters matter because they shape what attention keys are available. Clear section markers (XML-like tags, headers) and explicit output-format constraints are the highest-leverage levers in practice.

**Beyond a single pass.** The strongest techniques sample or structure the reasoning:

<pre class="mermaid">
flowchart TD
  P[&quot;Prompt + question&quot;]
  Z[&quot;Zero-shot:&lt;br/&gt;answer directly&quot;]
  C[&quot;CoT:&lt;br/&gt;reason then answer&quot;]
  S[&quot;Self-consistency:&lt;br/&gt;sample N CoT paths&quot;]
  V[&quot;Majority vote&quot;]
  P --&gt; Z
  P --&gt; C
  P --&gt; S
  S --&gt; V
</pre>
- **Self-consistency:** sample many CoT traces with temperature greater than 0, then take the **majority-vote** answer. Trades extra inference cost for accuracy by marginalizing over reasoning paths.
- **ReAct:** interleave *Reason* and *Act* — the model emits a thought, then a tool/API call, observes the result, and continues. Foundation of [Tool Use and Agents](/notes/ml-algorithms/language-models/tool-use-and-agents/).
- **Least-to-most / decomposition:** prompt the model to break a hard problem into ordered subproblems and solve them sequentially, feeding earlier answers forward.

## Techniques and trade-offs

| Technique | Idea | Cost | Best for |
|---|---|---|---|
| Zero-shot | Instruction only, no examples | 1 call, cheapest | Simple/common tasks; well-aligned instruct models |
| Few-shot ICL | $k$ demonstrations in prompt | 1 call, longer prompt | Format pinning, niche label spaces, style transfer |
| Chain-of-thought | "Think step by step" before answer | More output tokens | Math, logic, multi-hop reasoning |
| Self-consistency | $N$ CoT samples then majority vote | $N\times$ inference | Hard reasoning where accuracy beats latency/cost |
| ReAct | Reason and act with tools | Multi-turn loop | Tasks needing search, code, or live data |
| Least-to-most | Decompose then solve in order | Multi-step | Compositional problems with dependencies |

**Prompt vs retrieve vs fine-tune** — the decision every candidate should be able to make:

| Approach | What it changes | When to choose |
|---|---|---|
| Prompting | Instructions/examples in context | Fast iteration; task expressible in a few examples; no private data needed |
| [Retrieval-Augmented Generation](/notes/ml-algorithms/language-models/retrieval-augmented-generation/) | Injects fresh/external knowledge into prompt | Answer needs up-to-date or proprietary facts; reduce hallucination; cite sources |
| [Supervised Fine-Tuning](/notes/ml-algorithms/language-models/supervised-fine-tuning/) / [Parameter-Efficient Fine-Tuning](/notes/ml-algorithms/language-models/parameter-efficient-fine-tuning/) | Model weights | Stable, high-volume task; need shorter prompts, lower latency, or behavior prompts can't reliably hit |

Rule of thumb: **prompt first, add retrieval for knowledge, fine-tune for behavior** once the task is stable and worth the data/training cost.

## Practical considerations
- **Brittleness is real.** Accuracy can swing several points with the *order* of few-shot examples, the choice of examples, and surface formatting (newlines, separators, label words). Test prompts on a held-out set, not vibes.
- **Recency/primacy bias.** Models over-weight examples near the start and end of the context; bury nothing important in the middle (the "lost in the middle" effect). Relevant to [Long Context](/notes/ml-algorithms/language-models/long-context/).
- **Format with constraints, validate output.** For structured output, demand JSON/a schema and parse it; pair with constrained decoding (see [Decoding Strategies](/notes/ml-algorithms/language-models/decoding-strategies/)) when the API supports it.
- **Decoding settings interact with prompts.** CoT and self-consistency want temperature greater than 0 for diverse samples; deterministic extraction wants temperature 0.
- **Cost scales with tokens.** Few-shot and CoT inflate prompt/output length, raising latency and dollar cost. **Prompt caching** (reusing the prefix's [KV Cache and Inference Optimization](/notes/ml-algorithms/language-models/kv-cache-and-inference-optimization/) across calls) makes long static system prompts and demonstrations far cheaper.
- **Instruction-tuned models change the defaults.** Modern aligned models often do well zero-shot, so heavy few-shot is less necessary than in early GPT-3 days; reasoning models bake in CoT, so explicit "think step by step" can be redundant or even discouraged.

## Related
- [Supervised Fine-Tuning](/notes/ml-algorithms/language-models/supervised-fine-tuning/) · [Parameter-Efficient Fine-Tuning](/notes/ml-algorithms/language-models/parameter-efficient-fine-tuning/) · [Retrieval-Augmented Generation](/notes/ml-algorithms/language-models/retrieval-augmented-generation/) · [Tool Use and Agents](/notes/ml-algorithms/language-models/tool-use-and-agents/) · [Reasoning and Test-Time Compute](/notes/ml-algorithms/language-models/reasoning-and-test-time-compute/)
- [Decoding Strategies](/notes/ml-algorithms/language-models/decoding-strategies/) · [KV Cache and Inference Optimization](/notes/ml-algorithms/language-models/kv-cache-and-inference-optimization/) · [Long Context](/notes/ml-algorithms/language-models/long-context/) · [Pretraining and Language Modeling](/notes/ml-algorithms/language-models/pretraining-and-language-modeling/)
- Foundational: [Transformers](/notes/ml-algorithms/deep-learning/transformers/) · [Attention](/notes/ml-algorithms/deep-learning/attention/)
