---
layout: note
title: "Link Prediction"
description: "Task: given a graph, predict missing/future edges. Two flagship applications in your interviews: PYMK (\"People You May Know\" — predict future friendships, LinkedIn Prep) and sam…"
note: true
note_collection: "ML system design"
note_section: "Graph Neural Networks"
section_order: 4
note_order: 4
math: true
mermaid: false
---
**Task:** given a graph, predict missing/future edges. Two flagship applications in your interviews: **PYMK** ("People You May Know" — predict future friendships, LinkedIn Prep) and **same-owner prediction** between accounts (AM-07 GNN over the Identity Graph).

## Classical (non-deep) baselines — name them first

For candidate pair $(u,v)$, score by neighborhood overlap:
- **Common neighbors:** $|N(u) \cap N(v)|$.
- **Jaccard:** $\frac{|N(u)\cap N(v)|}{|N(u)\cup N(v)|}$ — normalizes for degree.
- **Adamic–Adar:** $\sum_{w \in N(u)\cap N(v)} \frac{1}{\log |N(w)|}$ — shared *rare* neighbors count more (sound familiar? it's IDF again — a mutual friend with 50 friends is far stronger evidence than a mutual friend with 50,000). 
- **Preferential attachment:** $|N(u)|\cdot|N(v)|$.

These remain strong features inside any learned system and are cheap at scale. PYMK historically = friends-of-friends candidate generation + GBDT over exactly these features.

## GNN-based link prediction

1. Run a GNN ([GraphSAGE](/notes/ml-system-design/graph-neural-networks/gcn-graphsage-gat/)/[R-GCN](/notes/ml-system-design/graph-neural-networks/heterogeneous-graphs-and-r-gcn/)) to get node embeddings $h_u, h_v$.
2. Score the pair with a head: dot product $h_u^\top h_v$, or an MLP over $[h_u \| h_v \| h_u \odot h_v \| |h_u - h_v|]$ (use symmetric combinations if the relation is undirected — same-owner is symmetric!).
3. **Training:** positives = observed edges; negatives = sampled non-edges ([Negative Sampling and Hard Negatives](/notes/ml-system-design/foundations/negative-sampling-and-hard-negatives/) — uniform + hard ones like "high common-neighbor non-edges"); loss = BCE or BPR.

## The leakage trap (interviewers love this)

If the edge you're predicting is **in the message-passing graph during training**, the GNN can trivially "predict" it (the endpoints already exchanged messages through it). You must **remove target edges from the graph before computing embeddings** for those training pairs (edge masking / disjoint train-message edges). The same discipline applies temporally: predicting next month's friendships must use only this month's graph. This is the graph version of point-in-time correctness — connecting those two earns points.

## Evaluation
Rank-based: AUC on held-out edges vs sampled non-edges (easy to game with easy negatives — report against *hard* negative sets too), Hits@K / MRR for "rank the true edge among 1000 candidates." Always **time-split** for dynamic graphs.

## Serving pattern
Exactly the two-stage funnel: candidate generation (friends-of-friends / shared-entity blocking / embedding ANN) → pairwise scorer → threshold or rank. Compare [The Two-Stage RecSys Funnel](/notes/ml-system-design/recommendation-systems/the-two-stage-recsys-funnel/) and AM-04 Blocking and Candidate Generation — it's the same diagram with different nouns.
