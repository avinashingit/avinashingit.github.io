---
layout: note
title: "Multi-Label vs Multi-Task vs Multi-Task Multi-Label"
description: "Most confusion comes from treating these as three points on one line. They are not. There are two independent questions:"
note: true
note_collection: "ML system design"
note_section: "Recommendation Systems"
section_order: 2
note_order: 9
updated: 2026-06-17 10:15:07 -0700
keywords:
  - Ranking
  - Recommendation
  - Training
  - Vision
  - Serving
math: true
mermaid: true
---
## The single most important idea: these live on two _different_ axes

Most confusion comes from treating these as three points on one line. They are not. There are **two independent questions**:

|Axis|Question it answers|Options|
|---|---|---|
|**Output structure of a task**|For _one_ task, how many labels can a single input have?|_Multi-class_ (exactly 1) vs **Multi-label** (any subset, 0…N)|
|**Number of objectives**|How many distinct tasks/objectives does the model optimize at once?|_Single-task_ (1) vs **Multi-task** (≥2, usually with shared features)|

From this:

- **Multi-label model** = _one_ task, but the output is a **set** of co-occurring labels.
- **Multi-task model** = _several_ tasks (each may be classification, regression, segmentation…), usually sharing a backbone.
- **Multi-task multi-label model** = _several_ tasks where **at least one task is itself multi-label**.

A quick mental test:

- "An image can be both _dog_ AND _outdoor_ AND _grass_." → one task, **multi-label**.
- "Predict the person's _age_ AND their _gender_ AND their _emotion_ from one face." → three tasks, **multi-task**.
- "From a chest X-ray, flag _all present pathologies_ (a set), AND predict a _severity score_, AND _segment_ the lungs." → **multi-task multi-label** (the pathology task is multi-label; the others aren't).

---

## 1. Multi-Label Model

### What it is

A **single** prediction task whose label space contains classes that are **not mutually exclusive**. The target is a binary vector `y ∈ {0,1}^N`, and the model emits `N` **independent** probabilities. This is fundamentally different from multi-class, where exactly one class is correct.

### Architecture flowchart

<pre class="mermaid">
flowchart TD
    X[&quot;Input x&quot;] --&gt; E[&quot;Shared Encoder / Backbone&quot;]
    E --&gt; H[&quot;Single output head: N units (logits z)&quot;]
    H --&gt; S[&quot;Sigmoid applied PER unit (independent)&quot;]
    S --&gt; P1[&quot;P(label_1)&quot;]
    S --&gt; P2[&quot;P(label_2)&quot;]
    S --&gt; Pn[&quot;P(label_N)&quot;]
    P1 --&gt; T[&quot;Threshold each (e.g. &gt; 0.5)&quot;]
    P2 --&gt; T
    Pn --&gt; T
    T --&gt; O[&quot;Output = subset of labels, e.g. {label_2, label_5}&quot;]
</pre>
The defining detail: **sigmoid per output unit**, _not_ a softmax. Softmax forces the outputs to compete and sum to 1 (good for "pick one"); sigmoid lets each label be on/off independently.

### Loss computation

Binary cross-entropy (BCE) applied **independently to each of the N labels**, then averaged:

$$ \mathcal{L}_{\text{ML}} = -\frac{1}{N}\sum_{i=1}^{N}\Big[,y_i \log \hat{y}_i + (1-y_i)\log(1-\hat{y}_i),\Big], \qquad \hat{y}_i = \sigma(z_i) $$

Each label is treated as its own little binary classifier. Common refinements for label imbalance (most labels are usually 0):

- **Weighted BCE**: scale positive terms by `pos_weight`.
- **Focal loss**: down-weights easy negatives so rare positives drive learning.

### How gradient descent behaves

With sigmoid + BCE the gradient at each logit is famously clean:

$$ \frac{\partial \mathcal{L}_{\text{ML}}}{\partial z_i} = \frac{1}{N}\big(\hat{y}_i - y_i\big) $$

- Per-label gradients are **independent at the output layer** but **sum together** in the shared backbone (one input, one set of features).
- There is **no task-conflict** in the multi-task sense — it's a single objective.
- The practical pitfall is **label imbalance**: thousands of negatives can dominate the gradient and drown out rare positives. That's what `pos_weight`/focal loss fix.

### When and where to use

- A single instance genuinely carries **multiple co-occurring tags** from **one** label space.
- Examples: image auto-tagging, document/topic categorization (an article tagged _politics_ + _economy_), audio event detection, gene-function prediction, toxic-comment flagging (toxic + threat + insult).
- Choose this whenever labels are **not** mutually exclusive and there's only **one** semantic question being asked.

---

## 2. Multi-Task Model

### What it is

**One** model trained on **several distinct tasks** simultaneously. The standard design is **hard parameter sharing**: a shared backbone learns general features, and each task gets its own small **head**. Tasks can be heterogeneous — one head classifies, another regresses, another segments. The motivation is **inductive transfer**: related tasks regularize each other, improving generalization and cutting inference cost (one backbone serves all).

### Architecture flowchart

<pre class="mermaid">
flowchart TD
    X[&quot;Input x&quot;] --&gt; E[&quot;Shared Encoder / Backbone (shared θ_s)&quot;]
    E --&gt; HA[&quot;Head A (θ_A)&quot;]
    E --&gt; HB[&quot;Head B (θ_B)&quot;]
    E --&gt; HC[&quot;Head C (θ_C)&quot;]
    HA --&gt; OA[&quot;Softmax → class (multi-class)&quot;]
    HB --&gt; OB[&quot;Linear → value (regression)&quot;]
    HC --&gt; OC[&quot;Softmax → class&quot;]
    OA --&gt; LA[&quot;L_A = Cross-Entropy&quot;]
    OB --&gt; LB[&quot;L_B = MSE&quot;]
    OC --&gt; LC[&quot;L_C = Cross-Entropy&quot;]
    LA --&gt; SUM[&quot;Total Loss = Σ_t λ_t · L_t&quot;]
    LB --&gt; SUM
    LC --&gt; SUM
</pre>
### Loss computation

A **weighted sum** of each task's own loss:

$$ \mathcal{L}_{\text{MTL}} = \sum_{t=1}^{T} \lambda_t , \mathcal{L}_t $$

where each `L_t` uses whatever loss fits that task (CE for classification, MSE for regression, Dice for segmentation, etc.). The weights `λ_t` matter a lot. Strategies:

- **Fixed weights** — simplest, but needs tuning.
- **Uncertainty weighting** (Kendall et al.): learn a per-task noise term and let the model balance itself — roughly $\mathcal{L} = \sum_t \frac{1}{2\sigma_t^2}\mathcal{L}_t + \log\sigma_t$, so noisy/hard tasks get down-weighted automatically.
- **GradNorm / Dynamic Weight Averaging** — balance based on training dynamics.

### How gradient descent behaves (this is where it really differs)

The crux: gradients from every task **accumulate on the shared parameters** `θ_s`, but each head only sees its own task.

$$ \underbrace{\frac{\partial \mathcal{L}}{\partial \theta_s} = \sum_{t} \lambda_t \frac{\partial \mathcal{L}_t}{\partial \theta_s}}_{\text{shared backbone: a SUM of task gradients}} \qquad \underbrace{\frac{\partial \mathcal{L}}{\partial \theta_t} = \lambda_t \frac{\partial \mathcal{L}_t}{\partial \theta_t}}_{\text{each head: only its own task}} $$

Two failure modes arise _only_ in multi-task and not in multi-label:

1. **Gradient conflict / negative transfer** — two task gradients can point in opposing directions (negative cosine similarity), so improving one _hurts_ another. Fix: **PCGrad** (project a task's gradient onto the normal plane of any conflicting task's gradient — "gradient surgery").
2. **Gradient magnitude imbalance** — a task with large-scale loss dominates the shared update. Fix: **GradNorm**, loss normalization, or uncertainty weighting.

### When and where to use

- You have **related** tasks that can plausibly share a representation.
- You want better generalization from limited per-task data (auxiliary tasks act as regularizers), or one efficient backbone at inference.
- Examples: autonomous driving (detection + lane segmentation + depth), NLP backbones (POS + NER + parsing), face analysis (age + gender + emotion).
- **Caution**: if tasks are unrelated, you get **negative transfer** — they fight over shared capacity and all get worse. Relatedness is the precondition.

---

## 3. Multi-Task Multi-Label Model

### What it is

A multi-task setup in which **one or more of the tasks is itself multi-label**. It inherits _both_ sets of mechanics: per-label BCE inside the multi-label head(s), and cross-task gradient accumulation/conflict at the shared backbone.

### Architecture flowchart

<pre class="mermaid">
flowchart TD
    X[&quot;Input x&quot;] --&gt; E[&quot;Shared Encoder / Backbone (θ_s)&quot;]
    E --&gt; HA[&quot;Head A: MULTI-LABEL (M units)&quot;]
    E --&gt; HB[&quot;Head B: regression&quot;]
    E --&gt; HC[&quot;Head C: multi-class&quot;]
    HA --&gt; OA[&quot;Sigmoid × M (independent)&quot;]
    HB --&gt; OB[&quot;Linear&quot;]
    HC --&gt; OC[&quot;Softmax&quot;]
    OA --&gt; LA[&quot;L_A = mean BCE over M labels&quot;]
    OB --&gt; LB[&quot;L_B = MSE&quot;]
    OC --&gt; LC[&quot;L_C = Cross-Entropy&quot;]
    LA --&gt; SUM[&quot;L = λ_A·L_A + λ_B·L_B + λ_C·L_C&quot;]
    LB --&gt; SUM
    LC --&gt; SUM
</pre>
### Loss computation

Same weighted-sum structure as multi-task, but the multi-label task's term is the averaged BCE:

$$ \mathcal{L} = \sum_{t} \lambda_t , \mathcal{L}_t, \qquad \mathcal{L}_{t=\text{multilabel}} = -\frac{1}{M}\sum_{i=1}^{M}\big[y_i\log\hat{y}_i + (1-y_i)\log(1-\hat{y}_i)\big] $$

### How gradient descent behaves

You now juggle **two** balancing problems at once:

- **Within** the multi-label head: per-label gradients `(ŷ_i − y_i)`, vulnerable to **label imbalance** (fix with weighted BCE / focal loss).
- **Across** tasks: the multi-label task contributes one aggregated gradient to `θ_s`, which then competes with the other tasks' gradients — vulnerable to **task imbalance and conflict** (fix with λ-weighting, uncertainty weighting, PCGrad).

A subtle trap: a multi-label head with many labels can produce a large-magnitude aggregate gradient simply because it sums many BCE terms, accidentally dominating the shared backbone. Normalizing per-task gradient scale (or careful λ) matters more here than in either pure case.

### When and where to use

- Several objectives where at least one is "tag all that apply."
- Examples: medical imaging (detect _all_ present pathologies = multi-label, + severity regression + organ segmentation); e-commerce product understanding (attribute tags = multi-label, + category = multi-class, + price = regression); video understanding (multiple co-occurring action tags + scene classification).

---

## Side-by-side summary

||Multi-Label|Multi-Task|Multi-Task Multi-Label|
|---|---|---|---|
|**# of tasks**|1|≥2|≥2|
|**Output per task**|subset of N labels|one label/value per task|mixed; ≥1 task is multi-label|
|**Output activation**|sigmoid (per unit)|per-head (softmax/linear/…)|per-head; sigmoid on the multi-label head|
|**Loss**|mean/weighted **BCE**|$\sum_t \lambda_t \mathcal{L}_t$|$\sum_t \lambda_t \mathcal{L}_t$, BCE term inside|
|**Output-layer gradient**|$\frac{1}{N}(\hat{y}_i - y_i)$|per-head, standard|both, combined|
|**Shared-param gradient**|sum of per-label grads (one task)|**sum of per-task grads**|per-task grads, one of which aggregates labels|
|**Main pitfall**|label imbalance|task conflict / magnitude imbalance|**both** at once|
|**Key remedies**|weighted BCE, focal loss, per-label thresholds|λ-tuning, uncertainty weighting, GradNorm, PCGrad|all of the above together|
|**Use when**|labels not mutually exclusive, one question|related tasks share useful features|several objectives, ≥1 is "tag all that apply"|

## Decision guide

<pre class="mermaid">
flowchart TD
    Q1{&quot;How many distinct tasks / objectives?&quot;}
    Q1 -- &quot;One&quot; --&gt; Q2{&quot;Can an input have several labels at once?&quot;}
    Q1 -- &quot;Several (related)&quot; --&gt; Q3{&quot;Is any single task multi-label?&quot;}
    Q2 -- &quot;No, exactly one&quot; --&gt; MC[&quot;Multi-CLASS (softmax + CE)&quot;]
    Q2 -- &quot;Yes, any subset&quot; --&gt; ML[&quot;MULTI-LABEL (sigmoid + BCE)&quot;]
    Q3 -- &quot;No&quot; --&gt; MT[&quot;MULTI-TASK (shared backbone + Σ λ_t L_t)&quot;]
    Q3 -- &quot;Yes&quot; --&gt; MTML[&quot;MULTI-TASK MULTI-LABEL (BCE head inside Σ λ_t L_t)&quot;]
</pre>
## Practical cheat-sheet

- **Never use softmax for multi-label.** Softmax makes classes compete; you'll cap the model at picking one. Use sigmoid + BCE.
- **Multi-task ≠ multi-label.** Different heads/objectives is multi-task; a set-valued single output is multi-label.
- **Tune λ early in multi-task.** A mis-scaled loss silently lets one task hijack the backbone. Uncertainty weighting is a strong default.
- **Watch gradient conflict.** If multi-task accuracy is _worse_ than single-task baselines, suspect negative transfer — try PCGrad or split the conflicting task into its own branch.
- **In multi-task multi-label, debug in layers.** First confirm the multi-label head works alone (per-label PR curves), then confirm task balance — otherwise you can't tell whether a problem is label imbalance or task imbalance.


#### A few problems with Multi Task Learning

1. Individual task learning at different pace, loss reduction in each task differs. A few ways to handle this are 
	1. Uncertainty Weighting
		1. Noisy tasks get higher weight vs normal tasks. The parameter $\sigma$ is learnt per task and inverse variance weighted for each loss to get the weight for each task.
			1. $L_{overall} = \sum_{1}^{t} \frac{1}{2\sigma_t^2} L_t + log(\sigma_t)$ 
			2. So each task is scaled by the variance of its loss and we have the log term to make sure the model just doesn’t make $\sigma$ equal to infinity to push the loss to zero. The log term balances this out.
	2. Dynamic Gradient Weighting Average
		1. Lets say we have two tasks A and B the loss for A decreases for 20 to 16 in one step and for the other it decreases from 10 to 2. So for A, the the weight will be higher (16 / 20) and for B the weight will be lower (2/10). Then the softmax is taken over all these tasks’ weights to get the final weight.
	3. Grad Norm
		1. Learned mechanics of the gradient descent and dynamically weight tasks based on it
2. Some other times, the gradients from each task point in different directions. On task’s gradient could be positive and other task’s gradient could be negative and then they cancel out each other. To remediate this, there are techniques like PC Grad
	1. From each gradient we remove the component that is in that gradients direction by projecting that gradient onto it.

#### The practical workflow most people use

Run UW first — it's cheap insurance. Watch the per-task validation curves. If everything trains acceptably, stop; you're done. If you see a specific hard-but-learnable task getting starved (low weight, flat loss, but you have reason to believe it _could_ improve), that's your cue to switch to GradNorm with a moderate α (≈0.5–1.0) and bump α up if the laggard still isn't getting enough push.
