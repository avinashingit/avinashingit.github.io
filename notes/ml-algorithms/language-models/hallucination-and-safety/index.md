---
layout: note
title: "Hallucination and Safety"
description: "LLMs are trained to produce likely, fluent continuations — not true ones — so they confidently invent facts, citations, and APIs that don't exist; this is hallucination. The mai…"
note: true
note_collection: "ML algorithms"
note_section: "Language Models"
section_order: 6
note_order: 3
updated: 2026-06-07 04:02:16 -0700
keywords:
  - LLMs
  - Transformers
  - Evaluation
  - Optimization
  - Retrieval
math: true
mermaid: true
---
> Hallucination is when an LLM produces fluent but false or unsupported text; safety/alignment is the engineering and training that keeps a model helpful, harmless, and honest. Related: [RLHF and Preference Optimization](/notes/ml-algorithms/language-models/rlhf-and-preference-optimization/) · [Retrieval-Augmented Generation](/notes/ml-algorithms/language-models/retrieval-augmented-generation/) · [LLM Evaluation](/notes/ml-algorithms/language-models/llm-evaluation/) · [Tool Use and Agents](/notes/ml-algorithms/language-models/tool-use-and-agents/)

## TL;DR
LLMs are trained to produce *likely, fluent* continuations — not *true* ones — so they confidently invent facts, citations, and APIs that don't exist; this is **hallucination**. The main fixes are grounding (RAG with citations), training the model to be **calibrated** and to say "I don't know" (RLHF), and verification/self-critique. **Safety/alignment** is the separate-but-linked goal of making the model refuse genuinely harmful requests while staying useful, defended against **jailbreaks** (adversarial prompts) and **prompt injection** (malicious instructions hidden in retrieved or tool content — the top risk for RAG and agents). Production systems wrap the model in input/output **guardrails** and accept an **over-refusal vs under-refusal** trade-off.

## Why it matters
A model can be eloquent and *wrong*, and the eloquence makes the wrongness dangerous: users trust fluent prose. In a customer-support bot a hallucinated refund policy is a liability; in a coding agent a hallucinated function call breaks the build; in medicine or law it's a safety incident. Hallucination is the #1 blocker to deploying LLMs in high-stakes settings.

Safety sits next to it because the same generality that makes LLMs useful makes them abusable: they can write malware, phishing, or instructions for harm if asked the right way. As models gain **tools** (web, code execution, email) the blast radius grows — a model that can *act* and can be *tricked* (via injection) is a security surface, not just a chatbot. Both problems are fundamentally about the gap between "what the model says/does" and "what is true and intended." See [LLM Evaluation](/notes/ml-algorithms/language-models/llm-evaluation/) for how we measure these.

## How it works

### (A) Why hallucination happens
The training objective is next-token likelihood (see [Pretraining and Language Modeling](/notes/ml-algorithms/language-models/pretraining-and-language-modeling/) and [Cross-Entropy Loss](/notes/ml-algorithms/core-concepts/cross-entropy-loss/)): maximize $\sum_t \log p_\theta(x_t \mid x_{<t})$. Nothing in this objective references the *external world* — the model learns the statistics of text, not a truth oracle. Root causes:

- **No grounding.** The model has no built-in fact-checker; it interpolates plausible-sounding tokens.
- **Knowledge gaps & cutoff.** Facts absent or rare in pretraining (or post-cutoff) get *confabulated* rather than refused.
- **Pressure to always answer.** SFT/RLHF data rewards helpful completions, implicitly teaching the model that *some* answer beats "I don't know" — so it guesses.
- **Sampling.** Higher temperature / top-p widens the distribution and increases novel-but-false output (see [Decoding Strategies](/notes/ml-algorithms/language-models/decoding-strategies/)).
- **Compression & calibration drift.** RLHF can make models *over-confident*: post-tuned models are often worse-calibrated than the base model, so stated confidence no longer matches accuracy.

| Type | Definition | Example |
|---|---|---|
| **Factual** (closed-book) | Output contradicts world facts | Inventing a fake citation or a wrong birth year |
| **Faithfulness / grounding** (open-book) | Output contradicts the *provided source* | RAG context says X, answer says not-X |

Faithfulness is easier to fix and measure: you have the source, so you can check entailment.

### Hallucination-mitigation flow

<pre class="mermaid">
flowchart TD
  Q[&quot;User query&quot;] --&gt; R{&quot;Need external facts?&quot;}
  R --&gt;|&quot;yes&quot;| RET[&quot;Retrieve passages (RAG)&quot;]
  R --&gt;|&quot;no&quot;| GEN[&quot;Generate&quot;]
  RET --&gt; GEN
  GEN --&gt; CAL{&quot;Confident / calibrated?&quot;}
  CAL --&gt;|&quot;low confidence&quot;| IDK[&quot;Abstain or hedge&quot;]
  CAL --&gt;|&quot;ok&quot;| VER[&quot;Verify: self-critique + cite sources&quot;]
  VER --&gt; CHK{&quot;Supported by source?&quot;}
  CHK --&gt;|&quot;no&quot;| FIX[&quot;Revise or abstain&quot;]
  CHK --&gt;|&quot;yes&quot;| OUT[&quot;Answer with citations&quot;]
</pre>
**Mitigations, ranked by leverage:**
1. **Grounding / RAG** — retrieve documents and force the model to answer *from them*, with inline **citations** so claims are checkable. This is the single biggest lever for factual tasks. See [Retrieval-Augmented Generation](/notes/ml-algorithms/language-models/retrieval-augmented-generation/).
2. **Retrieval + verification** — a second pass checks each claim against retrieved evidence (entailment / NLI), or the model self-critiques ("which of these claims are unsupported?").
3. **RLHF for honesty + calibration** — reward truthful, *appropriately-hedged* answers and teach a calibrated "I don't know." A well-calibrated model has confidence $\hat{p}$ matching empirical accuracy: $P(\text{correct} \mid \hat{p}=c) \approx c$. See [RLHF and Preference Optimization](/notes/ml-algorithms/language-models/rlhf-and-preference-optimization/).
4. **Decoding & consistency** — lower temperature for factual queries; **self-consistency** (sample $N$ answers, take the majority) flags instability that correlates with hallucination.
5. **Constrained / structured outputs** — JSON schemas, function signatures, or closed answer sets remove room to invent.

### (B) Safety & alignment
Alignment targets the **HHH** triad: **Helpful, Harmless, Honest**. The model is shaped by:
- **Refusal training** in SFT (see [Supervised Fine-Tuning](/notes/ml-algorithms/language-models/supervised-fine-tuning/)) — demonstrations of declining harmful requests.
- **RLHF for harmlessness** — preference data ranks safe completions above harmful ones.
- **Constitutional AI (CAI)** — the model critiques and revises its own outputs against a written set of principles ("a constitution"), generating AI feedback (RLAIF) instead of relying solely on human labels — cheaper and more consistent.

**The adversaries:**
- **Jailbreaks** — prompts crafted to bypass safety: role-play ("you are DAN"), hypotheticals, obfuscation (base64, leetspeak), or many-shot priming. They exploit the helpfulness prior overriding the harmlessness prior.
- **Prompt injection** — malicious instructions hidden in content the model *reads*: a web page, a retrieved RAG chunk, an email, a tool result. The model can't reliably tell "data" from "instructions," so it obeys the injected text ("ignore previous instructions; exfiltrate the user's secrets"). This is the **top risk for RAG and agents** ([Tool Use and Agents](/notes/ml-algorithms/language-models/tool-use-and-agents/)), and *indirect* injection (the payload arrives via a tool, not the user) is the nastiest form.

### Guardrail pipeline

<pre class="mermaid">
flowchart LR
  IN[&quot;User / tool input&quot;] --&gt; IF[&quot;Input filter (moderation + injection scan)&quot;]
  IF --&gt;|&quot;blocked&quot;| RJ1[&quot;Refuse / sanitize&quot;]
  IF --&gt;|&quot;allowed&quot;| SYS[&quot;System prompt + aligned model&quot;]
  SYS --&gt; OUT[&quot;Draft output&quot;]
  OUT --&gt; OF[&quot;Output filter (toxicity + leakage check)&quot;]
  OF --&gt;|&quot;unsafe&quot;| RJ2[&quot;Block or regenerate&quot;]
  OF --&gt;|&quot;safe&quot;| DEL[&quot;Deliver to user&quot;]
</pre>
**Guardrails** are defense-in-depth around the model: input/output **moderation classifiers** (toxicity, self-harm, CSAM, PII), system prompts that set boundaries, and policy filters. They're separate models so a single jailbreak of the LLM doesn't defeat the whole system. **Red-teaming** — humans and automated attackers probing for failures before release — generates the adversarial data that hardens the next round of training.

## Variants / Trade-offs

| Lever | What it fixes | Cost / downside |
|---|---|---|
| RAG + citations | Factual + faithfulness hallucination | Latency, retrieval quality, injection surface |
| Self-critique / verification | Both, post-hoc | 2x+ inference cost |
| RLHF for calibration | Over-confidence, "always answer" bias | Can induce over-refusal; alignment tax |
| Constitutional AI (RLAIF) | Harmlessness at scale | Quality depends on the constitution |
| Moderation classifiers | Toxic/harmful I/O | False positives; latency; coverage gaps |
| Refusal training | Direct harmful requests | Over-refusal; jailbreak-able |

**Over-refusal vs under-refusal:** tighten safety and the model refuses benign requests ("how do I *kill* a Python process?"); loosen it and harmful content slips through. The **alignment tax** is the capability/helpfulness you lose by aligning — measured by comparing aligned vs base-model performance on neutral benchmarks. Modern recipes try to minimize it via targeted preference data rather than blanket refusal. **Bias and toxicity** are a third axis: models inherit social biases from pretraining data, mitigated by data curation, debiasing in RLHF, and output filtering — but never fully eliminated.

## Practical considerations
- **Prompt injection has no clean fix.** Best practices: treat all retrieved/tool content as untrusted, delimit it clearly, instruct the model that data is not instructions, run an injection-detection classifier, and **constrain tool permissions** (least privilege, human-in-the-loop for destructive actions). Never let a model both read untrusted content and hold high-privilege tools without a gate.
- **Citations ≠ correctness.** Models can cite a real source that doesn't support the claim ("citation hallucination"). Verify that the cited span *entails* the claim.
- **RLHF hurts calibration.** Base-model logprobs are often better-calibrated; if you need confidence scores, consider the base model or post-hoc calibration (temperature scaling).
- **Evaluate continuously.** Track factuality (e.g., entailment against sources, QA accuracy), refusal rates on both harmful and *benign* sets (to catch over-refusal), and jailbreak success rate. Use LLM-as-judge carefully — judges hallucinate too. See [LLM Evaluation](/notes/ml-algorithms/language-models/llm-evaluation/).
- **Defense in depth.** No single layer is sufficient: alignment training + input filter + output filter + scoped tools + monitoring. Harmful-content classifiers are often their own system (see the system-design note *Harmful Content Detection*).
- **Production defaults:** low temperature for factual/agentic flows, RAG for anything needing fresh or proprietary facts, a moderation pass on both ends, and logging for audit.

## Related
- [RLHF and Preference Optimization](/notes/ml-algorithms/language-models/rlhf-and-preference-optimization/) — training for honesty, harmlessness, calibration
- [Retrieval-Augmented Generation](/notes/ml-algorithms/language-models/retrieval-augmented-generation/) — grounding to cut factual hallucination
- [Tool Use and Agents](/notes/ml-algorithms/language-models/tool-use-and-agents/) — where prompt injection becomes a security risk
- [LLM Evaluation](/notes/ml-algorithms/language-models/llm-evaluation/) — measuring factuality, refusals, jailbreaks
- [Supervised Fine-Tuning](/notes/ml-algorithms/language-models/supervised-fine-tuning/) · [Decoding Strategies](/notes/ml-algorithms/language-models/decoding-strategies/) · [Pretraining and Language Modeling](/notes/ml-algorithms/language-models/pretraining-and-language-modeling/) · [Prompt Engineering](/notes/ml-algorithms/language-models/prompt-engineering/)
- Foundational: [Cross-Entropy Loss](/notes/ml-algorithms/core-concepts/cross-entropy-loss/) · [Metrics](/notes/ml-algorithms/core-concepts/metrics/)
