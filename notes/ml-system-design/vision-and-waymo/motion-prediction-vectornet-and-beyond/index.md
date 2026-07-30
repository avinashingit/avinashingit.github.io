---
layout: note
title: "Motion Prediction — VectorNet and Beyond (Waymo core)"
description: "Task: given each agent's past track + the HD map + traffic context, predict each agent's next ~8 seconds. Sits between perception and planning; Waymo wrote the canonical papers,…"
note: true
note_collection: "ML system design"
note_section: "Vision"
section_order: 6
note_order: 5
updated: 2026-06-09 12:46:21 -0700
keywords:
  - Vision
  - Embeddings
  - Transformers
  - Graphs
  - Training
math: true
mermaid: false
---
**Task:** given each agent's past track + the HD map + traffic context, predict each agent's next ~8 seconds. Sits between perception and planning; Waymo wrote the canonical papers, so know them.

## Input representation: rasters vs vectors
- **Raster (pre-2020):** render map + tracks into a top-down image, run a CNN. Simple; but rasterization loses precision, wastes compute on empty pixels, and locks a fixed receptive field.
- **VectorNet (Waymo, 2020):** represent everything as **polylines** — a lane is a sequence of vectors, an agent's history is a sequence of displacement vectors. Encode each polyline with a small subgraph network (per-vector MLPs + max-pool, PointNet-flavored); then a **global interaction graph**: full attention among polyline embeddings (agents ↔ lanes ↔ agents). Structured, precise, efficient — the representation switch that defined the modern stack.

## The defining property: multimodality
At an intersection a driver may turn left **or** go straight — the future is a **distribution with distinct modes**. A model trained with plain L2 to a single trajectory predicts the *average* of modes — a physically impossible path through the middle. Standard solution: predict **K trajectories + probabilities**, train with **winner-takes-all** (only the best-matching mode receives the regression loss; classification loss on which mode was closest) — i.e., a mixture model fit with hard EM. 

**Anchor/goal-conditioned variants:** TNT/DenseTNT (Waymo): first predict candidate **goal points** (sampled from lanes / dense heatmap), then complete a trajectory per goal — anchoring modes in map structure improves coverage and interpretability.

**Scene-level transformers:** Wayformer / Scene Transformer / MTR — factorized or unified attention over [agents × time × map], multimodal decoding with learned queries (DETR-style). Distinguish **marginal** prediction (each agent independently — modes may be mutually inconsistent: two cars' "most likely" paths collide) from **joint/scene-consistent** prediction (predict combinations) — the planner ultimately needs joint consistency; saying so is a senior signal.

## Metrics (memorize)
- **minADE_K:** average displacement error of the best of K trajectories; **minFDE_K:** final-point error of the best mode; **miss rate:** fraction where even the best mode misses by >2m; **mAP** over behavior buckets (Waymo Open Motion Dataset's headline metric — precision of mode *probabilities*, not just geometry). "min over K" metrics reward coverage of modes; probability quality needs mAP/NLL — know that distinction.

## System notes
Latency ~10s of ms for dozens of agents → shared scene encoding, per-agent light decoders; consume tracker uncertainty (don't treat perception output as truth); interactive agents (yielding) motivate joint models; predictions carry uncertainty into the planner's cost ([Planning, Imitation Learning, and the Waymo Stack](/notes/ml-system-design/vision-and-waymo/planning-imitation-learning-and-the-waymo-stack/)).
