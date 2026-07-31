---
layout: note
title: "RLHF and Preference Optimization"
description: "RLHF (Reinforcement Learning from Human Feedback) is the alignment step that turns a capable-but-raw fine-tuned model into a helpful, harmless assistant. The classic InstructGPT…"
note: true
note_collection: "ML algorithms"
note_section: "Language Models"
section_order: 6
note_order: 17
updated: 2026-06-07 03:58:29 -0700
keywords:
  - LLMs
  - Transformers
  - Optimization
  - Probability
  - Supervised Learning
math: true
mermaid: true
---
> Aligning an LLM to human preferences by training on pairwise comparisons of its outputs, either via a reward model + RL or directly. Related: [Supervised Fine-Tuning](/notes/ml-algorithms/language-models/supervised-fine-tuning/), [Cross-Entropy Loss](/notes/ml-algorithms/core-concepts/cross-entropy-loss/), [Hallucination and Safety](/notes/ml-algorithms/language-models/hallucination-and-safety/), [Gradient Descent](/notes/ml-algorithms/core-concepts/gradient-descent/)

## TL;DR

RLHF (Reinforcement Learning from Human Feedback) is the alignment step that turns a capable-but-raw fine-tuned model into a helpful, harmless assistant. The classic InstructGPT recipe is three stages: supervised fine-tuning (SFT), then training a **reward model (RM)** on human **pairwise preference** comparisons, then optimizing the policy with **PPO** to maximize reward **minus a KL penalty** that anchors it to the SFT model. **DPO (Direct Preference Optimization)** collapses stages 2–3 into a single classification-style loss on preference pairs — no separate reward model, no RL loop — which is simpler and more stable, and is now the common default.

## Why it matters

[Supervised Fine-Tuning](/notes/ml-algorithms/language-models/supervised-fine-tuning/) teaches a model the *format* of good answers by imitation — given a prompt, copy this curated response. But imitation has a ceiling: it can only be as good as the demonstrations, it can't easily teach "answer A is *slightly* better than answer B," and writing gold demonstrations for nuanced qualities (tone, safety, refusal style, helpfulness) is hard and expensive. Humans are far better at *judging* ("which of these two is better?") than at *authoring* the ideal answer.

Preference optimization exploits exactly that. Instead of demonstrations, we collect cheap pairwise judgments and train the model to produce outputs humans **prefer**. This is where alignment lives in the stack: [Pretraining and Language Modeling](/notes/ml-algorithms/language-models/pretraining-and-language-modeling/) gives raw knowledge, SFT gives instruction-following, and RLHF/DPO shapes the *behavior and values* — the difference between a model that can answer and one you'd ship.

## How it works

### Stage 1 — SFT

Start from the [Supervised Fine-Tuning](/notes/ml-algorithms/language-models/supervised-fine-tuning/) model $\pi^{\text{SFT}}$. It is both the initialization for the policy and the reference point everything stays close to.

### Stage 2 — Reward model from pairwise preferences

Collect prompts $x$, sample two responses, and have humans label which is "chosen" ($y_w$) vs "rejected" ($y_l$). Train a scalar reward model $r_\phi(x,y)$ (usually the SFT model with the LM head swapped for a single scalar head) using the **Bradley–Terry** model of preferences: the probability a human prefers $y_w$ over $y_l$ is a logistic function of the reward gap. The loss is a **pairwise-logistic** (binary cross-entropy) objective:

$$\mathcal{L}_{RM}(\phi) = -\,\mathbb{E}_{(x,y_w,y_l)}\Big[\log \sigma\big(r_\phi(x,y_w) - r_\phi(x,y_l)\big)\Big]$$

where $\sigma$ is the sigmoid. Minimizing this pushes the reward of the chosen response above the rejected one. Reward is only meaningful *relative* to other responses for the same prompt — the absolute scale is arbitrary.

### Stage 3 — Policy optimization with PPO + KL

Now optimize the policy $\pi_\theta$ (init from SFT) with **PPO (Proximal Policy Optimization)** to maximize expected reward, but with a per-token **KL-divergence penalty** back to $\pi^{\text{SFT}}$:

$$\max_{\theta}\;\mathbb{E}_{x,\,y\sim\pi_\theta}\big[\,r_\phi(x,y)\big]\;-\;\beta\,\mathbb{D}_{\mathrm{KL}}\!\big(\pi_\theta(y\mid x)\,\|\,\pi^{\text{SFT}}(y\mid x)\big)$$

**Why the KL term?** The reward model is an imperfect proxy. Pure reward maximization leads to **reward hacking**: the policy finds degenerate outputs that score high under $r_\phi$ but are bad to humans — repeating flattering phrases the RM over-rewards, exploiting length bias, or collapsing to a few "safe" high-reward responses (**mode collapse**). The KL penalty, weighted by $\beta$, keeps $\pi_\theta$ in the neighborhood of the SFT distribution where the reward model was trained and is still trustworthy. Small $\beta$ → more reward gain but more drift and hacking; large $\beta$ → safer but barely moves. The effective per-token reward is $r_\phi(x,y) - \beta\big(\log\pi_\theta - \log\pi^{\text{SFT}}\big)$.

<pre class="mermaid">
flowchart LR
    SFT[&quot;SFT model (reference)&quot;]
    POL[&quot;Policy pi_theta&quot;]
    GEN[&quot;Sample responses y&quot;]
    RM[&quot;Reward model r_phi(x,y)&quot;]
    KL[&quot;KL penalty vs SFT&quot;]
    PPO[&quot;PPO update&quot;]

    SFT --&gt;|&quot;init&quot;| POL
    POL --&gt; GEN
    GEN --&gt;|&quot;score&quot;| RM
    GEN -. &quot;log-prob ratio&quot; .-&gt; KL
    SFT -. &quot;anchor&quot; .-&gt; KL
    RM --&gt;|&quot;reward&quot;| PPO
    KL --&gt;|&quot;minus penalty&quot;| PPO
    PPO --&gt;|&quot;gradient step&quot;| POL
</pre>
The PPO loop is heavy: it requires keeping up to **four models** in memory (policy, reference/SFT, reward model, and a value/critic head), online sampling, and careful tuning — which is what motivates DPO.

### DPO — skip the RM and the RL loop

**DPO (Direct Preference Optimization)** uses a key insight: the KL-constrained objective above has a *closed-form optimal policy*, $\pi^*(y\mid x) \propto \pi^{\text{SFT}}(y\mid x)\,\exp\!\big(r(x,y)/\beta\big)$. Inverting this lets you express the reward in terms of the policy itself:

$$r(x,y) = \beta\,\log\frac{\pi_\theta(y\mid x)}{\pi^{\text{SFT}}(y\mid x)} + \beta\log Z(x)$$

Plug this reward into the Bradley–Terry loss. The intractable partition function $Z(x)$ **cancels** (it's the same for $y_w$ and $y_l$), leaving a simple supervised loss directly on preference pairs:

$$\mathcal{L}_{DPO}(\theta) = -\,\mathbb{E}_{(x,y_w,y_l)}\Big[\log \sigma\Big(\beta\log\tfrac{\pi_\theta(y_w\mid x)}{\pi^{\text{SFT}}(y_w\mid x)} - \beta\log\tfrac{\pi_\theta(y_l\mid x)}{\pi^{\text{SFT}}(y_l\mid x)}\Big)\Big]$$

Intuitively: raise the log-prob of chosen responses and lower that of rejected ones, each measured *relative to the frozen reference* — so the $\beta\log(\pi_\theta/\pi^{\text{SFT}})$ term is an *implicit reward* and the KL constraint is baked in. No reward model, no sampling, no critic — just a [Cross-Entropy Loss](/notes/ml-algorithms/core-concepts/cross-entropy-loss/)-style gradient step you train with ordinary [Gradient Descent](/notes/ml-algorithms/core-concepts/gradient-descent/). This is why DPO is simpler, cheaper, and far more stable than PPO.

### RLAIF / Constitutional AI

Human labels are the bottleneck. **RLAIF (RL from AI Feedback)** replaces (or augments) human labelers with an LLM judge that picks the preferred response. Anthropic's **Constitutional AI** is a specific recipe: a written set of principles (a "constitution") guides the model to critique and revise its own outputs (the SFT-like phase) and to generate the AI preference labels (the RL phase). This scales preference data cheaply and makes the value signal auditable, at the cost of inheriting the judge model's biases.

## Variants and trade-offs

| Method | Needs reward model? | Needs online sampling/RL? | Signal | Notes / when to use |
|---|---|---|---|---|
| **PPO (RLHF)** | Yes | Yes | Scalar reward + KL | Most expressive, can exceed offline data; heavy, unstable, ~4 models in memory |
| **DPO** | No | No (offline) | Pairwise preference | Simple, stable, common default; tied to its fixed preference dataset |
| **RLAIF / Constitutional AI** | Yes (AI-labeled) | Yes (or feeds DPO) | AI preferences | Scales labeling; auditable via principles; inherits judge bias |
| **KTO** | No | No | *Per-example* good/bad (unpaired) | Use when you have thumbs-up/down, not pairs |
| **IPO** | No | No | Pairwise | Adds regularizer to fix DPO over-fitting on near-deterministic prefs |
| **GRPO** | Yes (or rule reward) | Yes | Group-relative advantage | Drops the critic; normalizes reward within a sampled group — popular for reasoning/RLVR (e.g. DeepSeek) |
| **RRHF / RAFT** | Ranker / filter | No | Ranking / best-of-N | Rank or filter samples and SFT on the winners; lightweight |

**Rule of thumb:** start with DPO (cheap, strong). Reach for PPO/GRPO when you need an online reward signal — especially **RLVR** (RL with *Verifiable* Rewards) for math/code where correctness is checkable — or when the static preference set caps quality. Use KTO when feedback is unpaired binary.

## Practical considerations

- **Data quality dominates.** Preference data is noisy and subjective; annotator agreement is often only ~70%. Garbage preferences cap the final model regardless of algorithm.
- **Length bias / verbosity.** Both RMs and LLM judges reward longer answers; DPO can blow up response length. Length-normalize or penalize.
- **$\beta$ and KL budget.** Too low → reward hacking and drift from the reference; too high → no learning. PPO practitioners watch the running KL like a hawk and sometimes adapt $\beta$.
- **DPO's reference model.** The frozen $\pi^{\text{SFT}}$ must stay loaded for the log-ratio; a common trick precomputes its log-probs. DPO can over-suppress chosen-response probability if the SFT model already underweights it — pair with a small SFT/NLL term.
- **Reward overoptimization is measurable.** True quality rises then *falls* as you push RM reward — a hallmark of hacking; hold out a fresh eval and stop early.
- **Where it shows up:** every shipped chat model goes through this. It is the difference between a model that follows instructions and one that is helpful, honest, and safe — directly tied to [Hallucination and Safety](/notes/ml-algorithms/language-models/hallucination-and-safety/) (refusals, honesty) and to product behavior like Recruiter Outreach Generation (tone, brand voice).

## Related

- Pipeline neighbors: [Supervised Fine-Tuning](/notes/ml-algorithms/language-models/supervised-fine-tuning/) · [Parameter-Efficient Fine-Tuning](/notes/ml-algorithms/language-models/parameter-efficient-fine-tuning/) · [Pretraining and Language Modeling](/notes/ml-algorithms/language-models/pretraining-and-language-modeling/)
- Behavior & evaluation: [Hallucination and Safety](/notes/ml-algorithms/language-models/hallucination-and-safety/) · [LLM Evaluation](/notes/ml-algorithms/language-models/llm-evaluation/) · [Reasoning and Test-Time Compute](/notes/ml-algorithms/language-models/reasoning-and-test-time-compute/)
- Application: Recruiter Outreach Generation
- Foundations: [Cross-Entropy Loss](/notes/ml-algorithms/core-concepts/cross-entropy-loss/) · [Gradient Descent](/notes/ml-algorithms/core-concepts/gradient-descent/) · [Softmax](/notes/ml-algorithms/core-concepts/softmax/)
