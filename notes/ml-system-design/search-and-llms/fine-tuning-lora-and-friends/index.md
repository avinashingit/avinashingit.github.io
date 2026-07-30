---
layout: note
title: "Fine-Tuning — LoRA and Friends"
description: "Updating all weights of a 70B model needs the weights + gradients + optimizer states ≈ 16 bytes/param with Adam (fp16 weight 2 + grad 2 + fp32 master 4 + two moments 8) ≈ over a…"
note: true
note_collection: "ML system design"
note_section: "Search and LLMs"
section_order: 3
note_order: 3
math: true
mermaid: false
---
## Why not full fine-tuning
Updating all weights of a 70B model needs the weights + gradients + optimizer states ≈ **16 bytes/param with Adam** (fp16 weight 2 + grad 2 + fp32 master 4 + two moments 8) ≈ over a TB of GPU state, plus you must store/serve a full copy per customer/task.

## LoRA — Low-Rank Adaptation
Freeze the pretrained weight $W \in \mathbb{R}^{d\times k}$; learn a low-rank update:
$$W' = W + \frac{\alpha}{r} BA, \quad B \in \mathbb{R}^{d\times r},\ A \in \mathbb{R}^{r\times k},\ r \in [4, 64]$$
Hypothesis: task adaptation lives in a low-dimensional subspace. Train only A,B (often just on attention projections) → **trainable params drop ~1000×**; optimizer state shrinks accordingly. At inference either merge ($W+BA$, zero overhead) or keep adapters separate and **hot-swap per request** — one base model serving many fine-tunes (multi-LoRA serving), which is the Databricks-relevant systems point.

**QLoRA:** base weights quantized to 4-bit (NF4) and frozen; LoRA adapters in bf16 on top → fine-tune a 70B on a single big GPU. Know the headline.

## The menu (one line each)
- **Full FT:** max quality, max cost, catastrophic-forgetting risk.
- **LoRA/QLoRA:** 95–100% of FT quality for most tasks; the default.
- **Prompt/prefix tuning:** learn soft tokens; cheapest, weakest; fine to name.
- **Instruction tuning (SFT):** supervised FT on (instruction, response) pairs — turns a base LM into an assistant.
- **Preference optimization:** RLHF (reward model + PPO) vs **DPO** (directly optimize on preference pairs with a closed-form objective; no reward model or RL loop; the pragmatic default). Know both at definition level.

## The decision interviewers want: prompt vs RAG vs fine-tune
- Need **knowledge** that changes / must be cited → [RAG](/notes/ml-system-design/search-and-llms/rag-system-design/) (fine-tuning is bad at injecting facts and can't update).
- Need **format/style/behavior/domain reasoning** → fine-tune (LoRA).
- Often: RAG for facts + small SFT for format. Always try prompting first; build the **eval set before** any tuning.

## Data & ops
Quality >> quantity (thousands of curated examples beat millions noisy); dedupe against eval; mix in general data to limit forgetting; track everything in a registry; eval-gate releases (AB Testing and Online Evaluation). Distributed mechanics → Distributed Training.
