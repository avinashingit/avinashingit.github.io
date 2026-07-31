---
layout: note
title: "Supervised Fine-Tuning"
description: "Supervised Fine-Tuning (SFT), also called instruction tuning, takes a pretrained base model — which only predicts likely next tokens — and trains it on human-written or curated…"
note: true
note_collection: "ML algorithms"
note_section: "Language Models"
section_order: 6
note_order: 20
updated: 2026-06-07 03:58:19 -0700
keywords:
  - LLMs
  - Transformers
  - Supervised Learning
  - Training
  - Optimization
math: true
mermaid: true
---
> Continuing training a pretrained base model on curated (instruction → ideal response) demonstrations so it follows instructions and behaves as a helpful assistant. Related: [Pretraining and Language Modeling](/notes/ml-algorithms/language-models/pretraining-and-language-modeling/), [RLHF and Preference Optimization](/notes/ml-algorithms/language-models/rlhf-and-preference-optimization/), [Cross-Entropy Loss](/notes/ml-algorithms/core-concepts/cross-entropy-loss/)

## TL;DR

Supervised Fine-Tuning (SFT), also called instruction tuning, takes a pretrained base model — which only predicts likely next tokens — and trains it on human-written or curated demonstration pairs of the form (prompt, ideal response). It uses the **same next-token cross-entropy loss** as pretraining, but typically **masks the loss on the prompt tokens** so the model only learns to *produce* good responses, not to reproduce prompts. The result is a model that follows instructions and answers in a chat format. SFT is the first alignment stage, sitting between pretraining and preference optimization (pretrain → SFT → RLHF/DPO).

## Why it matters

A pretrained base model is a *next-token predictor* trained on raw web text. If you give it "What is the capital of France?", a base model might continue with another plausible *question* ("What is the capital of Germany?") because that's a likely continuation in its training data — it has no notion of "the human wants an answer." It has enormous latent knowledge and skill, but no *interface*: it doesn't know it should be a helpful, honest assistant that responds to instructions.

SFT bridges that gap. By showing the model thousands of examples of "here is a request, here is how a good assistant responds," we teach it the *behavior* of instruction-following without teaching it much new *knowledge* (the knowledge largely came from [Pretraining and Language Modeling](/notes/ml-algorithms/language-models/pretraining-and-language-modeling/)). This is why a relatively small amount of high-quality SFT data can dramatically change a model's usefulness: we are eliciting and shaping existing capabilities, not building them from scratch. SFT is what turns GPT-style base models into the chat assistants people actually use, and it produces the starting policy that [RLHF and Preference Optimization](/notes/ml-algorithms/language-models/rlhf-and-preference-optimization/) then refines.

## How it works

**Same loss, different data.** SFT reuses the autoregressive cross-entropy objective from pretraining. For a sequence of tokens $x_1, \dots, x_T$, the model with parameters $\theta$ minimizes the negative log-likelihood of each token given the previous ones:

$$\mathcal{L}(\theta) = -\sum_{t} m_t \log p_\theta(x_t \mid x_{<t})$$

where $p_\theta(x_t \mid x_{<t})$ is the model's predicted probability of the correct next token and $m_t \in \{0, 1\}$ is a **loss mask**. The mask is the key difference from pretraining. See [Cross-Entropy Loss](/notes/ml-algorithms/core-concepts/cross-entropy-loss/) for the per-token term.

**Prompt masking.** Each training example is a (prompt, response) pair concatenated into one sequence. We set $m_t = 0$ for tokens in the prompt (instruction, system message, prior turns) and $m_t = 1$ only for the **assistant response tokens**. Intuition: we don't want the model to get better at *generating instructions* — those are given at inference time. We only want it to learn the conditional distribution $p(\text{response} \mid \text{prompt})$. The model still *attends to* the prompt tokens (they're in context), but they contribute no gradient. Some recipes train on the full sequence ("train on completions only" off), but completion-only masking is the standard and usually better.

**Chat templates and role tokens.** Real assistants are multi-turn, so SFT data is formatted with a fixed **chat template** that wraps each turn in special tokens marking roles: `system`, `user`, `assistant`. For example, a template might render a turn as `<|im_start|>user\n...<|im_end|>` then `<|im_start|>assistant\n...<|im_end|>`. Formatting matters enormously:

- The model learns to **stop** by emitting the end-of-turn token, so generation terminates cleanly.
- The role tokens let the model distinguish instructions (which it should obey) from its own prior output and from the system prompt (which sets persona/policy).
- **Train-time and inference-time templates must match exactly** — a mismatched template at serving time is a very common bug that silently degrades quality or causes runaway generation.

<pre class="mermaid">
flowchart LR
    BASE[&quot;Base model&lt;br/&gt;(next-token predictor)&quot;]
    DATA[&quot;Demonstrations&lt;br/&gt;(prompt -&gt; ideal response)&quot;]
    SFT[&quot;SFT&lt;br/&gt;(masked cross-entropy)&quot;]
    IF[&quot;Instruction-following&lt;br/&gt;chat model&quot;]
    RLHF[&quot;RLHF / DPO&lt;br/&gt;(preference optimization)&quot;]
    DATA --&gt;|&quot;curated pairs&quot;| SFT
    BASE --&gt;|&quot;continue training&quot;| SFT
    SFT --&gt;|&quot;produces policy&quot;| IF
    IF --&gt;|&quot;init for alignment&quot;| RLHF
</pre>
**Data.** SFT quality is dominated by the demonstration set. Notable sources and findings:

- **FLAN** — Google's large collection of NLP tasks reformatted as instructions; pioneered multi-task instruction tuning and strong zero-shot generalization.
- **Alpaca** — ~52k instructions generated by prompting a stronger model (self-instruct distillation); cheap and influential, but inherits the teacher's quirks.
- **ShareGPT** — real user/assistant conversations; good for multi-turn and natural style.
- **OpenAssistant (OASST)** — community-contributed, human-graded conversation trees.
- **LIMA ("Less Is More for Alignment")** — Meta showed that **~1,000 carefully curated, diverse, high-quality examples** can produce a strong instruction-follower, arguing that SFT mostly *surfaces* pretrained knowledge and that **data quality and diversity beat sheer quantity**.

The practical lesson: a small, clean, diverse dataset usually beats a large noisy one. Deduplication, format consistency, and response quality matter more than example count past a few thousand.

## Variants / Trade-offs

| Dimension | Option A | Option B | When to use which |
|---|---|---|---|
| What to update | **Full fine-tuning** (all weights) | **PEFT / LoRA** (small adapters) | Full FT for max quality and large data budgets; LoRA when compute/memory-limited or serving many task variants — see [Parameter-Efficient Fine-Tuning](/notes/ml-algorithms/language-models/parameter-efficient-fine-tuning/) |
| Loss masking | **Completion-only** (mask prompt) | **Full-sequence** (train on all) | Completion-only is the default and avoids learning to emit prompts; full-sequence sometimes helps tiny datasets |
| Data source | **Human-written** demos | **Distilled** from a stronger model | Human for highest fidelity/safety; distillation for cheap scale, but you inherit the teacher's errors and style |
| Data philosophy | **Quantity** (FLAN-scale) | **Quality** (LIMA-scale) | Broad task coverage benefits from scale; style/behavior alignment benefits most from small curated sets |
| Stage in pipeline | **SFT only** | **SFT → RLHF/DPO** | SFT-only is fine for narrow tasks; add preference optimization to align nuanced helpfulness/harmlessness |

## Practical considerations

- **Catastrophic forgetting.** Over-training on a narrow SFT set can erode pretrained capabilities (coding, reasoning, multilinguality). Mitigate with low learning rates, few epochs (often **1–3**), and mixing in diverse data. LoRA helps because base weights are frozen.
- **Overfitting to style, not substance.** Models readily mimic surface formatting (length, bullet lists, "Certainly!") without improving correctness. Diverse data and held-out eval prevents shipping a model that *sounds* aligned but isn't.
- **Hallucination from teaching answers beyond knowledge.** If demonstrations always answer confidently — including questions the base model can't actually know — SFT teaches the model to **fabricate confidently** instead of saying "I don't know." Include refusals and calibrated uncertainty in the data. This ties into [Hallucination and Safety](/notes/ml-algorithms/language-models/hallucination-and-safety/).
- **Hyperparameters.** Typical recipe: small LR (e.g., $1\!-\!2\times10^{-5}$ full FT, higher for LoRA), 1–3 epochs, cosine decay, sequence packing for efficiency, and **packing-aware masking** so attention/loss don't leak across packed examples.
- **Template discipline.** Ship the exact chat template alongside weights; serving systems must apply it verbatim. Most production regressions trace to template or special-token mismatches.
- **Evaluation.** Use held-out instruction sets and LLM-as-judge / pairwise win-rates, not just loss — low loss can coexist with poor helpfulness.

## Related

- [Pretraining and Language Modeling](/notes/ml-algorithms/language-models/pretraining-and-language-modeling/) — where the base model and its knowledge come from
- [RLHF and Preference Optimization](/notes/ml-algorithms/language-models/rlhf-and-preference-optimization/) — the next alignment stage after SFT
- [Parameter-Efficient Fine-Tuning](/notes/ml-algorithms/language-models/parameter-efficient-fine-tuning/) — LoRA and adapters for cheap SFT
- [Cross-Entropy Loss](/notes/ml-algorithms/core-concepts/cross-entropy-loss/) — the per-token objective SFT reuses
- [Hallucination and Safety](/notes/ml-algorithms/language-models/hallucination-and-safety/) — risks SFT data design must address
- Recruiter Outreach Generation — an application where an SFT'd assistant generates tailored text
