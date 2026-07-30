---
layout: note
title: "GCN, GraphSAGE, GAT"
description: "The three names you must distinguish crisply. All are instances of message passing; they differ in how they aggregate and whether they generalize to unseen nodes."
note: true
note_collection: "ML system design"
note_section: "Graph Neural Networks"
section_order: 4
note_order: 1
updated: 2026-06-09 12:43:15 -0700
keywords:
  - Graphs
  - Training
  - Embeddings
  - Transformers
  - Fraud
math: true
mermaid: false
---
The three names you must distinguish crisply. All are instances of [message passing](/notes/ml-system-design/graph-neural-networks/graphs-and-message-passing-from-scratch/); they differ in **how they aggregate** and **whether they generalize to unseen nodes**.

## GCN — Graph Convolutional Network (2017)

$$H^{(l+1)} = \sigma\big(\tilde D^{-1/2}\tilde A \tilde D^{-1/2} H^{(l)} W^{(l)}\big)$$

In words: every node takes a **degree-normalized weighted average** of its neighbors (+ itself, via the self-loop in $\tilde A = A + I$), then a shared linear transform + nonlinearity. The $1/\sqrt{d_u d_v}$ normalization shrinks messages to/from high-degree hubs — a baked-in, non-learned hub discount.

Limitations: defined via the full adjacency matrix → full-graph training (doesn't scale) and **transductive** in its classic form; uniform-ish averaging can't emphasize informative neighbors.

## GraphSAGE (2017) — the industrial default

Two changes that make GNNs production-ready:

1. **Sample, don't sum-over-all:** at each layer, aggregate over a **fixed-size random sample** of neighbors (e.g., 25 at layer 1, 10 at layer 2). Per-node cost becomes constant (25×10=250 nodes for 2 hops) regardless of true degree → mini-batch training on billion-edge graphs, and hub explosion is capped by construction.
2. **Learn aggregator functions, not node vectors:** the parameters are the aggregation/transform weights. Embedding a node = *running the function* on its features+neighborhood → works on **nodes never seen in training** (inductive). New account signs up → embed it immediately.

$$h_v^{(l+1)} = \sigma\Big(W^{(l)} \cdot \text{concat}\big(h_v^{(l)},\ \text{AGG}\{h_u^{(l)}: u \in \text{Sample}(N(v))\}\big)\Big)$$

AGG ∈ {mean, max-pool(MLP), LSTM}. Train supervised (task loss) or unsupervised (nearby nodes similar, random nodes dissimilar — a graph [contrastive loss](/notes/ml-system-design/foundations/contrastive-learning-infonce-triplet-siamese/)).

## GAT — Graph Attention Network (2018)

Replace fixed averaging with **learned per-edge attention** ([attention](/notes/ml-system-design/foundations/transformers-from-scratch/), restricted to graph edges):

$$\alpha_{vu} = \text{softmax}_{u \in N(v)}\big(\text{LeakyReLU}(a^\top [W h_v \,\|\, W h_u])\big), \qquad h_v' = \sigma\Big(\sum_u \alpha_{vu} W h_u\Big)$$

The model learns *which neighbors matter*. Why this is gold for identity graphs: a device shared by 2 accounts is strong same-owner evidence; an IP shared by 10,000 accounts (a school NAT) is almost none. GAT can learn exactly this discount **from data** — attention as learned IDF. Multi-head attention stabilizes training. (Edge features — e.g., "how many times did this account use this device, how recently" — can be fed into the attention logit; crucial in AM-07 GNN over the Identity Graph.)

## Comparison table

| | GCN | GraphSAGE | GAT |
|---|---|---|---|
| Neighbor weighting | fixed (degree-norm) | uniform over a sample | **learned attention** |
| Inductive | not classically | **yes** | yes |
| Scales by | — | **sampling** | sampling (apply SAGE-style) |
| Use when | small/static graphs, baselines | **default at scale** | neighbor importance varies (fraud!) |

Production answer: **GraphSAGE-style sampling + attention aggregation + relation awareness** ([Heterogeneous Graphs and R-GCN](/notes/ml-system-design/graph-neural-networks/heterogeneous-graphs-and-r-gcn/)) — that composite is what "a GNN" means in a modern fraud stack. Scaling details: [Scaling GNNs - PinSage and Sampling](/notes/ml-system-design/graph-neural-networks/scaling-gnns-pinsage-and-sampling/).
