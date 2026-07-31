---
layout: note
title: "Calibration"
description: "A model is calibrated if, among all cases where it says \"70%\", roughly 70% are actually positive. Many classifiers violate this:"
note: true
note_collection: "ML algorithms"
note_section: "Core Concepts"
section_order: 1
note_order: 4
updated: 2026-06-15 21:39:30 -0700
keywords:
  - Optimization
  - Training
  - Evaluation
  - Graphs
  - Linear Models
math: true
mermaid: true
---
> **Abstract:** In one line A classifier's raw scores are often not honest probabilities. **Calibration** is a post-hoc map from scores to probabilities, learned on a held-out set. The two workhorse methods — **Platt scaling** (fit a sigmoid) and **isotonic regression** (fit any monotone step function) — are the _same regression of label on score_, differing only in the function class they allow.

> **Tip:** Rendering Graphs use Mermaid `xychart-beta` / `flowchart` (Obsidian ≥ 1.5, Mermaid ≥ 10.3). If a chart doesn't render, the table directly beneath it carries the identical numbers.

## Contents

- [1 The calibration problem](#1-the-calibration-problem)
- [2 The two methods at a glance](#2-the-two-methods-at-a-glance)
- [3 Platt scaling](#3-platt-scaling)
- [4 Isotonic regression and PAVA](#4-isotonic-regression-and-pava)
- [5 From fit to prediction](#5-from-fit-to-prediction)
- [6 Deciding whether and how to calibrate](#6-deciding-whether-and-how-to-calibrate)
- [7 Brier score](#7-brier-score)
- [8 Why train with cross-entropy not MSE](#8-why-train-with-cross-entropy-not-mse)
- [9 Formula reference](#9-formula-reference)
- [10 Related concepts](#10-related-concepts)

---

## 1 The calibration problem

A model is **calibrated** if, among all cases where it says _"70%"_, roughly 70% are actually positive. Many classifiers violate this:

- **SVMs** output a signed distance from the hyperplane — not a probability at all.
- **Boosted trees** (AdaBoost) push scores toward the middle, producing a sigmoidal reliability curve.
- **Naive Bayes / deep nets** tend to be overconfident, pushing scores toward 0 and 1.

> **Important:** Always use held-out data Calibration is assessed and fit on a **held-out set** (or via cross-validation) — never the data the base model trained on, or you bake in its optimism. Note also that **logistic regression fit by maximum likelihood is usually already well-calibrated**, because it directly optimizes log loss; the methods below matter most for SVMs, boosted trees, and naive Bayes.

---

## 2 The two methods at a glance

||Platt scaling|Isotonic regression|
|---|---|---|
|Type|Parametric (sigmoid)|Non-parametric (monotone step fn)|
|Assumption|Distortion is **sigmoidal**|Distortion is **monotonic**|
|Parameters|2 ($A$, $B$)|Many, data-dependent|
|Data appetite|Small sets fine|Needs more (~1000+)|
|Overfit risk|Low|Higher on small sets|
|Output|Smooth continuous curve|Piecewise-constant steps|
|Fitting objective|Maximum likelihood (log loss)|Least squares (via PAVA)|

> **Note:** The unifying view Both regress the binary label $y$ on a **single feature**, the raw score $f$. Platt restricts the regressor to the sigmoid family; isotonic restricts it only to be non-decreasing. That is the _entire_ difference.

### What "sigmoid-shaped distortion" means

The **distortion** is the curve mapping raw score → true probability. Platt assumes that curve is a member of the two-parameter logistic family $1/(1+e^{Af+B})$: monotone, a single inflection, symmetric saturation at both ends. When the true curve really is an S (SVMs, boosting) two knobs suffice. When it is monotone-but-not-sigmoidal, **no** $A,B$ can fit it and a systematic _bias_ remains that more data cannot remove — which is exactly the gap isotonic fills.

---

## 3 Platt scaling

$$ P(y=1 \mid f) = \frac{1}{1 + \exp(Af + B)} $$

Fit $A, B$ by maximizing likelihood (minimizing log loss) on the held-out pairs $(f_i, y_i)$ — i.e. **a 1-D logistic regression whose only feature is the base model's score**. Platt's practical tweak: use softened targets to avoid blow-up, with $N_+$ positives and $N_-$ negatives:

$$ t_+ = \frac{N_+ + 1}{N_+ + 2}, \qquad t_- = \frac{1}{N_- + 2} $$

Low variance, works on small sets — but biased if the distortion isn't sigmoidal.

> **Info:** Modern cousin **Temperature scaling** (Guo et al., 2017) is the one-parameter version for deep nets: divide the logits by a single learned scalar $T$ before the softmax.

---

## 4 Isotonic regression and PAVA

**Goal.** Given score-ordered points, find fitted values minimizing squared error subject to monotonicity:

$$ \min_{\hat m}\ \sum_i (y_i - \hat m_i)^2 \quad\text{s.t.}\quad \hat m_1 \le \hat m_2 \le \dots \le \hat m_n $$

**The one idea.** Where the data already climbs, keep it. Where it _dips_ (a violation), replace the offending run with its **mean** (the least-squares constant). That replacement is _pooling_.

**Pool Adjacent Violators Algorithm (PAVA).** Scan left→right; add each point as its own block; if a block's level is below the block to its left, merge them at their average; a merge can create a new violation to the left, so keep merging until non-decreasing. $O(n)$ after sorting.

### Worked example

Input $y = 1,\ 4,\ 3,\ 2,\ 6$:

|Step|Sees|Check / action|Block levels|
|---|---|---|---|
|1|1|start|`1`|
|2|4|$1 \le 4$ ✓|`1, 4`|
|3|3|$4 > 3$ ✗ → pool(4,3) = **3.5**|`1, 3.5`|
|4|2|$3.5 > 2$ ✗ → pool(3.5-block, 2) = **3**|`1, 3`|
|5|6|$3 \le 6$ ✓|`1, 3, 6`|

**Result:** $\hat y = 1,\ 3,\ 3,\ 3,\ 6$ — monotone, and each flat run is the mean of the points it absorbed. (Step 4 is the _cascade_: pooling created a fresh violation that pulled the whole block down.)

> **Warning:** Overfitting With few points, isotonic happily flattens real signal as if it were a violation. This is why it needs ~1000+ examples to reliably beat Platt.

---

## 5 From fit to prediction

PAVA gives a fitted value at each _training_ score — equivalently, a set of monotone **breakpoints** $(x_i, \hat y_i)$. To predict for a new score $x^\ast$:

1. Locate which two breakpoints it falls between.
2. **Interpolate linearly** between their fitted values (scikit-learn's default; a piecewise-constant "hold" is the alternative).
3. **Clip** anything past the ends to the nearest endpoint value (no data out there; stay monotone & bounded).

### Prediction example

Toy calibration set — scores $s$ with binary labels:

$$ s = 1,2,3,4,5,6 \qquad y = 0,0,1,0,1,1 $$

PAVA pools the `1,0` violation at $s=3,4$ into $0.5$, giving fitted levels and breakpoints through $(1,0),(2,0),(3,0.5),(4,0.5),(5,1),(6,1)$. Each level is the **fraction of positives** among the pooled points.

|New score $s^\ast$|Lands between|Predicted prob|
|---|---|---|
|0.4|below range|**0.00** (clipped)|
|2.5|$(2,0)$–$(3,0.5)$|**0.25**|
|4.5|$(4,0.5)$–$(5,1)$|**0.75**|
|5.5|flat top|**1.00**|

---

## 6 Deciding whether and how to calibrate

### Decision workflow

<pre class="mermaid">
flowchart TD
    A[&quot;Held-out set: predicted probs + true labels&quot;] --&gt; B[&quot;Bin predictions, plot reliability diagram&quot;]
    B --&gt; C{&quot;Points on the diagonal? (ECE ~ 0)&quot;}
    C -- &quot;Yes&quot; --&gt; D[&quot;No calibration needed&quot;]
    C -- &quot;No: systematic deviation&quot; --&gt; E{&quot;How much calibration data?&quot;}
    E -- &quot;Small (~100s), sigmoidal distortion&quot; --&gt; F[&quot;Platt scaling&quot;]
    E -- &quot;Large (~1000+), unknown shape&quot; --&gt; G[&quot;Isotonic regression&quot;]
    F --&gt; H[&quot;Cross-validate both; keep lower log loss / Brier&quot;]
    G --&gt; H
    H --&gt; I[&quot;Apply fitted map to new scores&quot;]
</pre>
### Worked case: 100 held-out examples

A deliberately **overconfident** model; base rate 47% positive. Bin into 5 (≈20 each — far cleaner than 10 bins at $n=100$):

|Predicted range|n|Mean predicted|Observed positive rate|
|---|---|---|---|
|0.0–0.2|20|0.107|0.200|
|0.2–0.4|16|0.323|0.312|
|0.4–0.6|20|0.471|0.400|
|0.6–0.8|25|0.683|0.640|
|0.8–1.0|19|0.894|0.737|

**How to read it:** a _systematic tilt_ (low bin above the diagonal, high bins below) signals miscalibration; dots scattered _on_ the diagonal within noise → skip calibration. Here the tilt is the overconfidence signature. **ECE = 0.075**, Brier = 0.222, log loss = 0.651 → calibrate.

#### Reliability diagram

<pre class="mermaid">
xychart-beta
    title &quot;Reliability — raw overconfident model&quot;
    x-axis &quot;bin (predicted prob)&quot; [0.1, 0.3, 0.5, 0.7, 0.9]
    y-axis &quot;probability&quot; 0 --&gt; 1
    line [0.107, 0.323, 0.471, 0.683, 0.894]
    bar [0.2, 0.312, 0.4, 0.64, 0.737]
</pre>
_Line = mean predicted (perfect-calibration reference); bars = observed rate. Bar above line at the low end and below it at the high end = overconfidence._

#### Choosing the method

Small set + sigmoidal distortion → **Platt** is the principled pick. Confirm by cross-validating both on log loss / Brier:

||Brier|Log loss|
|---|---|---|
|Raw (uncalibrated)|0.222|0.651|
|**Platt scaling**|0.213|0.615|
|Isotonic regression|0.194|0.555|

> **Warning:** Read the isotonic win with suspicion Isotonic scores lower here, but this is measured on the very 100 points it was fit to — flattering the more flexible method. Its giveaway: the fitted ends snap to **exactly 0 and 1** from a handful of points. Cross-validated at $n=100$, Platt's stability usually wins; the verdict flips around $n \approx 1000+$.

#### The two fitted maps

- **Platt:** $A = 3.05,\ B = -1.69$ → calibrated $= \sigma(3.05,p - 1.69)$ (smooth sigmoid; store 2 numbers).
- **Isotonic:** pooled positive-rate levels $[,0,\ 0.333,\ 0.4,\ 0.556,\ 0.667,\ 0.762,\ 1,]$ across increasing score (a lookup staircase; store breakpoints).

<pre class="mermaid">
xychart-beta
    title &quot;Calibration maps (raw score -&gt; calibrated prob)&quot;
    x-axis &quot;raw model score&quot; [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    y-axis &quot;calibrated probability&quot; 0 --&gt; 1
    line [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    line [0.155, 0.200, 0.253, 0.315, 0.384, 0.458, 0.534, 0.609, 0.679, 0.741, 0.795]
    line [0, 0, 0.333, 0.333, 0.4, 0.4, 0.4, 0.667, 0.762, 0.762, 1.0]
</pre>
_Three lines: straight diagonal = no change; the smooth one pulling extremes inward = Platt; the staircase = isotonic. Both sit below the diagonal at high scores and above it at low scores — squeezing overconfident predictions toward the middle._

End result on a few raw scores:

|Raw score|Platt|Isotonic|
|---|---|---|
|0.10|0.200|0.000|
|0.50|0.458|0.400|
|0.90|0.741|0.762|

The `0.10 → 0.000` from isotonic is the overfitting risk made visible.

---

## 7 Brier score

The **mean squared error of probability predictions**:

$$ \text{Brier} = \frac{1}{N}\sum_{i=1}^{N}(p_i - y_i)^2, \qquad y_i \in {0,1} $$

### Worked example

|Predicted $p_i$|Outcome $y_i$|$(p_i - y_i)^2$|
|---|---|---|
|0.9|1|0.01|
|0.8|0|0.64|
|0.3|0|0.09|
|0.6|1|0.16|

$$\text{Brier} = (0.01 + 0.64 + 0.09 + 0.16)/4 = 0.225$$

Properties worth knowing:

- Range **0 → 1**, lower better. A confident _wrong_ prediction (row 2: 0.8 on a negative) dominates the sum.
- It is a **proper scoring rule** — minimized in expectation only when you report your true beliefs, so it can't be gamed by shading toward 0/1. That's why it's trustworthy for comparing calibrators.
- Baseline: always predicting the base rate scores $\bar y(1-\bar y)$ = **0.25** for a balanced problem — any real model should beat that.
- **Multiclass** version sums over $K$ class probabilities, $\sum_k (p_{ik}-y_{ik})^2$ (range 0 → 2).
- **Murphy decomposition** splits Brier into _calibration_ + _refinement_ if you need to diagnose which is failing.

---

## 8 Why train with cross-entropy not MSE

The question: why not fit a sigmoid-output model (logistic regression / Platt) by minimizing squared error?

**Reason 1 — convexity.** Cross-entropy is convex in the weights for logistic regression (one global basin). Squared-error-through-a-sigmoid is **non-convex**: for $y=1$, $f(z)=(\sigma(z)-1)^2$ has $f''(z) = 2,s(1-s),g'(s)$ which is **negative when $s < 1/3$** (the confidently-wrong region). The global-optimum guarantee is lost.

**Reason 2 — the gradient (the real killer).** Per example, with $s=\sigma(z)$:

$$ \frac{\partial,\text{MSE}}{\partial z} = 2(s-y),\underbrace{s(1-s)}_{\sigma'(z)} \qquad \frac{\partial,\text{CE}}{\partial z} = s - y $$

The $\sigma'(z)=s(1-s)$ factor **cancels** in cross-entropy. So when the model is confidently wrong ($y=1$, $s\approx 0$): MSE gradient $\to 2(-1)(0) = 0$ (**vanishes exactly where error is largest**), while CE gradient $\to -1$ (full-strength correction).

<pre class="mermaid">
xychart-beta
    title &quot;Training loss vs logit z (true label = 1)&quot;
    x-axis &quot;model score z&quot; [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6]
    y-axis &quot;loss&quot; 0 --&gt; 6
    line [6.002, 5.007, 4.018, 3.049, 2.127, 1.313, 0.693, 0.313, 0.127, 0.049, 0.018, 0.007, 0.002]
    line [0.995, 0.987, 0.964, 0.908, 0.776, 0.535, 0.25, 0.072, 0.014, 0.002, 0.000, 0.000, 0.000]
</pre>
_The line climbing to 6 on the left is **cross-entropy** — steep slope when wrong. The line capped at 1 is **MSE** — it flattens into a plateau on the left, where the gradient has died._

|z (true label = 1)|Cross-entropy loss|MSE loss|
|---|---|---|
|-6|6.002|0.995|
|-4|4.018|0.964|
|-2|2.127|0.776|
|0|0.693|0.250|
|2|0.127|0.014|
|4|0.018|0.000|

> **Note:** Refining "multiple local minima" It _is_ non-convex, as the intuition says. But for a single sigmoid the practically lethal symptom isn't a maze of distinct minima — it's the **flat saturated plateaus** where learning stalls, plus losing the convexity guarantee. Cross-entropy wins on both: convex landscape _and_ a non-vanishing gradient.

> **Example:** The bow on the thread This is exactly why Platt fits its sigmoid by minimizing **log loss**, even though we _evaluate_ the result with **Brier**. Squared error is a fine proper scoring rule for _grading_ probabilities; it's a poor _objective_ for _training_ a sigmoid.

---

## 9 Formula reference

|Concept|Formula|
|---|---|
|Platt scaling|$P = 1/(1+e^{Af+B})$|
|Isotonic objective|$\min \sum (y_i-\hat m_i)^2$ s.t. $\hat m_1 \le \dots \le \hat m_n$|
|Sigmoid|$\sigma(z) = 1/(1+e^{-z})$, $\sigma'(z)=\sigma(1-\sigma)$|
|Cross-entropy|$-[y\ln s + (1-y)\ln(1-s)]$, grad $= s-y$|
|Squared error|$(s-y)^2$, grad $= 2(s-y),s(1-s)$|
|Brier (binary)|$\frac{1}{N}\sum (p_i-y_i)^2$|
|Expected Calibration Error|$\sum_b \frac{n_b}{N},|
|Base-rate Brier baseline|$\bar y(1-\bar y)$|

---

## 10 Related concepts

Platt scaling · Isotonic regression · Pool Adjacent Violators Algorithm · Reliability diagram · Expected Calibration Error · Brier score · Proper scoring rule · Cross-entropy loss · Temperature scaling · Logistic regression
