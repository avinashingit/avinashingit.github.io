---
layout: note
title: "Scaling GNNs — PinSage and Sampling"
description: "Full-graph GNN training needs the whole adjacency + all activations in memory — impossible at 10⁹ nodes / 10¹⁰ edges. Know four scaling strategies plus the PinSage case study."
note: true
note_collection: "ML system design"
note_section: "Graph Neural Networks"
section_order: 4
note_order: 5
math: true
mermaid: false
---
Full-graph GNN training needs the whole adjacency + all activations in memory — impossible at 10⁹ nodes / 10¹⁰ edges. Know four scaling strategies plus the PinSage case study.

## 1. Neighbor sampling (GraphSAGE)
Per layer, sample fixed fan-outs (e.g., 25 then 10). Cost per node is constant; the price is sampling variance and the **fan-out explosion with depth** (L layers ⇒ ∏ fanouts nodes) — another reason GNNs stay 2–3 layers.

## 2. Subgraph methods
- **Cluster-GCN:** partition (METIS) the graph into dense clusters; each batch = a cluster (+a few merged) → message passing stays inside the batch.
- **GraphSAINT:** sample subgraphs by random walks/edges with unbiasedness corrections.
Good when neighbor sampling's redundant computation hurts.

## 3. Decouple propagation from learning
**SGC / "precompute then MLP":** for some architectures, neighbor averaging is a *fixed linear* operation → precompute $A^k X$ offline in a batch job (Spark!), then train a plain MLP on the propagated features. Massive simplification — "can we precompute propagation offline?" is a smart cost question in interviews.

## 4. PinSage (Pinterest, 2018) — the canonical web-scale deployment
3-billion-node pin–board graph. Key engineering ideas:
- **Random-walk importance sampling:** define a node's "neighbors" as the top-T nodes by visit count of short random walks from it — picks the *most relevant* neighborhood (not just 1-hop) and gives **importance weights** for aggregation.
- **Producer–consumer pipeline:** CPU workers prepare sampled subgraphs while GPUs train — keeps GPUs saturated.
- **Curriculum hard negatives:** progressively harder negatives by walk-distance ([Negative Sampling and Hard Negatives](/notes/ml-system-design/foundations/negative-sampling-and-hard-negatives/)).
- **MapReduce inference:** computing embeddings for billions of nodes = a batch dataflow job with shared computation, then load into [ANN](/notes/ml-system-design/foundations/approximate-nearest-neighbor-search/).

## Production serving patterns
- **Batch embeddings (most common):** recompute node embeddings daily/hourly offline → KV store + ANN index. Simple; staleness is the cost.
- **Hybrid:** batch for the graph at large + on-demand inductive inference (run the sampled aggregation live) for new/changed nodes — exactly what a new account needs in AM-09 Enforcement, Serving, and Infra.
- Graph storage: adjacency lists in a KV/graph store with typed edges + timestamps; cap stored degree per (node, relation) with recency-based eviction.
