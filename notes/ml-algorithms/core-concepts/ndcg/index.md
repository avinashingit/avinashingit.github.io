---
layout: note
title: "NDCG"
description: "NDCG measures ranking quality by checking whether highly relevant items appear near the top of a ranked list. It is computed in three steps: DCG → IDCG → normalize."
note: true
note_collection: "ML algorithms"
note_section: "Core Concepts"
section_order: 1
note_order: 10
updated: 2026-06-29 22:35:03 -0700
keywords:
  - Optimization
  - Training
  - Evaluation
math: true
mermaid: false
---
NDCG measures ranking quality by checking whether highly relevant items appear near the top of a ranked list. It is computed in three steps: **DCG → IDCG → normalize**.

---

## 1. Discounted Cumulative Gain (DCG)

Each item's relevance is discounted by how far down the list it sits (a relevant item at position 5 is worth less than the same item at position 1).

$$DCG_p = \sum_{i=1}^{p} \frac{rel_i}{\log_2(i+1)}$$

A common alternative that more heavily rewards highly relevant items (standard in many IR / ML libraries):

$$DCG_p = \sum_{i=1}^{p} \frac{2^{rel_i} - 1}{\log_2(i+1)}$$

## 2. Ideal DCG (IDCG)

Sort the items by **true relevance**, descending, then compute DCG on that perfect ordering. This is the maximum DCG achievable.

## 3. Normalize

$$NDCG_p = \frac{DCG_p}{IDCG_p}$$

The result lands in **[0, 1]**, where 1 is a perfect ranking. Normalizing lets you compare across queries that have different numbers of relevant results.

---

## Worked Example (traditional formula)

System's ranking with true relevance scores `rel = [3, 2, 3, 0, 1, 2]`:

|Position $i$|$rel_i$|$\log_2(i+1)$|$rel_i / \log_2(i+1)$|
|---|---|---|---|
|1|3|1.000|3.000|
|2|2|1.585|1.262|
|3|3|2.000|1.500|
|4|0|2.322|0.000|
|5|1|2.585|0.387|
|6|2|2.807|0.712|

$$DCG = 6.861$$

**Ideal ordering** (relevance sorted descending): `[3, 3, 2, 2, 1, 0]`

|Position $i$|$rel_i$|$rel_i / \log_2(i+1)$|
|---|---|---|
|1|3|3.000|
|2|3|1.893|
|3|2|1.000|
|4|2|0.861|
|5|1|0.387|
|6|0|0.000|

$$IDCG = 7.141$$

$$NDCG = \frac{6.861}{7.141} \approx 0.961$$

---

## What Is the Relevance Score?

The **relevance score** ($rel_i$) is a number assigned to each item that says _how relevant_ it is to the query — the "ground truth" judgment that NDCG measures the ranking against. It is an **input you supply**, not something the formula computes.

### Where it comes from

- **Human raters / annotators** — judges grade how well a result matches a query.
- **Implicit feedback** — clicks, dwell time, purchases, or conversions used as a proxy.
- **Editorial labels** — predefined gold-standard judgments for a benchmark dataset.

### Common scales

A **graded relevance** scale, often 0–3 or 0–4:

|Score|Meaning|
|---|---|
|0|Not relevant / irrelevant|
|1|Marginally relevant|
|2|Relevant|
|3|Highly relevant / perfect match|

**Binary relevance** (just 0 or 1) is simpler but discards the distinction between "okay" and "perfect" results.

### Why the scale matters

- **Traditional formula** ($rel_i / \log_2(i+1)$): a score of 3 contributes exactly 3× what a score of 1 does — linear.
- **Exponential formula** ($2^{rel_i} - 1$): a 3 is worth 7 while a 1 is worth 1 — amplifying the gap between "highly relevant" and "marginally relevant."

This is why the two formulas can give noticeably different NDCG values on the same data.

> **Caution:** Relevance scores must be consistent and meaningful across the dataset. If annotators interpret "2" differently, NDCG faithfully measures ranking quality against noisy labels — garbage in, garbage out.

---

## Practical Notes

- Use **NDCG@k** (truncate to the top _k_ positions) when you only care about the first page of results.
- The `2^{rel} - 1` gain formula is the default in scikit-learn and most learning-to-rank toolkits — confirm which formula your benchmark expects before comparing numbers.
- If a query has no relevant documents, IDCG is 0 and NDCG is conventionally defined as **0**.
