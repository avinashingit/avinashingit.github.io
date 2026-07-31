---
layout: note
title: "Recurrent Neural Networks"
description: "MLPs and CNNs assume fixed-size inputs with no inherent ordering. But many problems involve variable-length sequences where order matters:"
note: true
note_collection: "ML algorithms"
note_section: "Deep Learning"
section_order: 4
note_order: 4
updated: 2026-06-06 11:30:50 -0700
keywords:
  - Deep Learning
  - Training
  - Transformers
  - Graphs
  - Inference
math: true
mermaid: false
---
> RNNs are neural networks designed for **sequential data** — text, time series, audio. The defining idea: process inputs one step at a time while carrying a **hidden state** that summarizes everything seen so far.

---

## Table of Contents

- [1. Why RNNs?](#1-why-rnns)
- [2. The Vanilla RNN](#2-the-vanilla-rnn)
- [3. Unrolling Through Time](#3-unrolling-through-time)
- [4. Common Misconception — Words Don't Shift Through Cells](#4-common-misconception-words-don-t-shift-through-cells)
- [5. Sequence Modeling Patterns](#5-sequence-modeling-patterns)
- [6. Backpropagation Through Time (BPTT)](#6-backpropagation-through-time-bptt)
- [7. Vanishing and Exploding Gradients](#7-vanishing-and-exploding-gradients)
- [8. LSTM — Long Short-Term Memory](#8-lstm-long-short-term-memory)
- [9. GRU — Gated Recurrent Unit](#9-gru-gated-recurrent-unit)
- [10. Bidirectional RNNs](#10-bidirectional-rnns)
- [11. Stacked / Deep RNNs](#11-stacked-deep-rnns)
- [12. Encoder-Decoder (Seq2Seq)](#12-encoder-decoder-seq2seq)
- [13. Teacher Forcing](#13-teacher-forcing)
- [14. Handling Variable-Length Sequences](#14-handling-variable-length-sequences)
- [15. Hyperparameters](#15-hyperparameters)
- [16. Strengths and Weaknesses](#16-strengths-and-weaknesses)
- [17. When to Use vs Not Use](#17-when-to-use-vs-not-use)
- [18. RNN vs LSTM vs GRU vs Transformer](#18-rnn-vs-lstm-vs-gru-vs-transformer)
- [19. Common Pitfalls](#19-common-pitfalls)
- [20. Production Considerations](#20-production-considerations)

---

## 1. Why RNNs?

MLPs and CNNs assume fixed-size inputs with no inherent ordering. But many problems involve variable-length sequences where order matters:

- **Language:** "dog bites man" ≠ "man bites dog"
- **Time series:** today's stock price depends on history
- **Speech:** phonemes occur in order

You could flatten a sequence and feed it to an MLP, but you'd lose order, can't handle variable lengths, and would have an explosion of parameters.

RNNs solve this by:

- **Sharing parameters across time** (like CNNs share across space)
- **Maintaining a hidden state** that carries information forward

---

## 2. The Vanilla RNN

At each timestep $t$, given input $\mathbf{x}_t$ and previous hidden state $\mathbf{h}_{t-1}$:

$$\mathbf{h}_t = \tanh(\mathbf{W}_{xh} \mathbf{x}_t + \mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{b}_h)$$

$$\mathbf{y}_t = \mathbf{W}_{hy} \mathbf{h}_t + \mathbf{b}_y$$

**Shapes** (sequence length $T$, hidden dim $H$, input dim $D$, output dim $K$):

|Param|Shape|
|---|---|
|$\mathbf{W}_{xh}$|$H \times D$|
|$\mathbf{W}_{hh}$|$H \times H$|
|$\mathbf{W}_{hy}$|$K \times H$|
|$\mathbf{h}_t$|$H$|

**Key property:** $\mathbf{W}_{xh}, \mathbf{W}_{hh}, \mathbf{W}_{hy}$ are **the same at every timestep**. Total params don't grow with sequence length.

Initial hidden state $\mathbf{h}_0$ is typically zeros (or learned).

> **Important:** An RNN is equivalent to an infinitely deep feedforward network where each layer shares weights. Depth in time = depth in layers.

---

## 3. Unrolling Through Time

**Unrolling = drawing each iteration of the RNN's loop as a separate node in a computation graph.** It's a visualization trick, not what actually happens in memory.

### Rolled view (what executes)

```
       ┌──────┐
   x_t →│ RNN  │→ y_t
       │ Cell │
       └──┬───┘
          │ (h loops back)
          └─────────────┐
                        ▼
                    (next step)
```

One box, with a loop. The actual computation.

### Unrolled view (a drawing convention for backprop)

```
x_1     x_2     x_3
 │       │       │
 ▼       ▼       ▼
[RNN]→ [RNN]→ [RNN]
 │       │       │
 ▼       ▼       ▼
y_1     y_2     y_3
```

We **draw** the cell $T$ times to make the dataflow visible. But all boxes are the **same cell with the same weights**.

### Why we unroll

Backprop needs a directed acyclic graph (DAG). The rolled view has a **cycle** — backprop can't handle cycles directly. Unrolling breaks the cycle into a DAG where each timestep is a separate node.

This is called **Backpropagation Through Time (BPTT)** — but it's literally just regular backprop on the unrolled graph.

|Concept|Feedforward MLP|Unrolled RNN|
|---|---|---|
|"Layers"|Each layer has its **own** weights|Every "layer" (timestep) shares the **same** weights|
|Depth|Fixed at design time|Equal to sequence length $T$|
|Parameters|Grow with depth|Constant — independent of $T$|
|Backprop|Standard|Standard, but gradients **sum** across shared weights|

---

## 4. Common Misconception — Words Don't Shift Through Cells

> **Warning:** Common confusion "First word at cell 1, then at cell 2 next timestep…" — **No.** Each word enters the cell exactly once, at its own timestep.

What flows from step to step is the **hidden state**, not the words.

### Walkthrough: "cats eat fish"

```
TIMESTEP 1:
  Input: "cats" + h_0   →   cell   →   h_1
  After: "cats" is done. Never used again.

TIMESTEP 2:
  Input: "eat"  + h_1   →   cell   →   h_2
  After: "eat" is done. The influence of "cats" lives inside h_2.

TIMESTEP 3:
  Input: "fish" + h_2   →   cell   →   h_3
  After: "fish" is done. h_3 contains influence of all 3 words.
```

### Mental model

Think of the cell as a **person reading a sentence one word at a time**, with the hidden state as their **mental summary** so far. They don't re-read past words — they just update their understanding as new words arrive.

```python
h = h_0
for t in range(T):
    h = cell(x[t], h)   # SAME cell every iteration
```

- One `cell` object — same weights every iteration.
- `h` gets overwritten each iteration: $h_0 \to h_1 \to h_2 \to \ldots$
- `x[t]` is used in iteration $t$ only.

The $T$ boxes in an unrolled diagram are **$T$ time-snapshots of the same cell**, not $T$ different cells.

---

## 5. Sequence Modeling Patterns

|Pattern|Example|Structure|
|---|---|---|
|**One-to-one**|Image classification|One input → one output|
|**One-to-many**|Image captioning|One input → sequence output|
|**Many-to-one**|Sentiment classification|Sequence → one output|
|**Many-to-many (aligned)**|POS tagging, frame-level video|Sequence → sequence (same length)|
|**Many-to-many (encoder-decoder)**|Machine translation|Sequence → sequence (different lengths)|

---

## 6. Backpropagation Through Time (BPTT)

Unroll the RNN for $T$ steps and apply backprop to the unrolled graph.

For a many-to-one task with loss $\mathcal{L}$ at the final step:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_{hh}} = \sum_{t=1}^{T} \frac{\partial \mathcal{L}}{\partial \mathbf{h}_t} \cdot \frac{\partial \mathbf{h}_t}{\partial \mathbf{W}_{hh}}$$

Since $\mathbf{W}_{hh}$ is shared, gradient contributions from every timestep are **summed**.

The error at timestep $t$ propagates back through all earlier timesteps via:

$$\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}} = \mathbf{W}_{hh}^\top \cdot \text{diag}(\tanh'(\cdot))$$

Going back $k$ steps multiplies $k$ such matrices.

### Truncated BPTT

For long sequences, full BPTT is too expensive and unstable. **Truncated BPTT** backprops only through the last $k$ steps (typical $k = 50$–$200$).

> **Important:** Two different "lengths"
> 
> - **Forward length** = number of times the cell runs (= sequence length)
> - **Backprop length / truncation window** = how far gradients flow back
> 
> Truncated BPTT is the technique where forward length > backprop length. The cell still runs once per token; only the **gradient** flow is truncated.

### Detaching hidden state between chunks

```python
for chunk in chunks(sequence, k=20):
    h = h.detach()             # cut gradient at chunk boundary
    out, h = rnn(chunk, h)
    loss = compute_loss(out)
    loss.backward()
```

Hidden state flows forward (so the model still "remembers" across chunks), but gradients don't flow back beyond the current chunk.

---

## 7. Vanishing and Exploding Gradients

When propagating gradients through time:

$$\frac{\partial \mathbf{h}_T}{\partial \mathbf{h}_0} = \prod_{t=1}^{T} \frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}}$$

This product of $T$ Jacobians either:

- **Vanishes** if matrix norms < 1 → can't learn long-range dependencies
- **Explodes** if matrix norms > 1 → NaN, unstable training

**Vanilla RNNs effectively can't learn dependencies longer than ~10 steps.**

### Fixes

|Problem|Solution|
|---|---|
|Exploding|**Gradient clipping** — rescale gradient if $\|\nabla\| > c$|
|Vanishing|**LSTM/GRU** — gating preserves gradient flow|
|Vanishing|**Skip connections** in time|
|General|**LayerNorm** instead of BatchNorm|

---

## 8. LSTM — Long Short-Term Memory

LSTMs solve the vanishing gradient problem with **gating**. They introduce a **cell state** $\mathbf{c}_t$ that acts as a "memory highway" — information can flow through it nearly unchanged.

### The four gates

At each timestep, given $\mathbf{x}_t$ and $\mathbf{h}_{t-1}$, compute:

$$\mathbf{f}_t = \sigma(\mathbf{W}_f [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f) \quad \text{(forget gate)}$$

$$\mathbf{i}_t = \sigma(\mathbf{W}_i [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i) \quad \text{(input gate)}$$

$$\tilde{\mathbf{c}}_t = \tanh(\mathbf{W}_c [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c) \quad \text{(candidate cell)}$$

$$\mathbf{o}_t = \sigma(\mathbf{W}_o [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o) \quad \text{(output gate)}$$

**Update cell state:**

$$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$$

**Output hidden state:**

$$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)$$

where $\sigma$ is sigmoid, $\odot$ is element-wise product, $[\cdot, \cdot]$ is concatenation.

### Intuition

- **Forget gate** $\mathbf{f}_t$: how much of the previous cell state to keep
- **Input gate** $\mathbf{i}_t$: how much of the new candidate to write
- **Candidate** $\tilde{\mathbf{c}}_t$: new information being proposed
- **Output gate** $\mathbf{o}_t$: how much of the cell state to expose as hidden state

> **Important:** Why LSTMs solve vanishing gradients The cell state update is **additive** ($\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \ldots$). The gradient w.r.t. $\mathbf{c}_{t-1}$ involves $\mathbf{f}_t$ (not a saturating activation derivative). If the forget gate is near 1, gradients flow back nearly unchanged.

### Parameter count

For input dim $D$ and hidden dim $H$: $$\text{params} = 4 \times (H \cdot (D + H) + H) = 4(DH + H^2 + H)$$

4× a vanilla RNN — four gate-like transformations.

---

## 9. GRU — Gated Recurrent Unit

A simplified LSTM with **2 gates** and no separate cell state.

$$\mathbf{z}_t = \sigma(\mathbf{W}_z [\mathbf{h}_{t-1}, \mathbf{x}_t]) \quad \text{(update gate)}$$

$$\mathbf{r}_t = \sigma(\mathbf{W}_r [\mathbf{h}_{t-1}, \mathbf{x}_t]) \quad \text{(reset gate)}$$

$$\tilde{\mathbf{h}}_t = \tanh(\mathbf{W} [\mathbf{r}_t \odot \mathbf{h}_{t-1}, \mathbf{x}_t])$$

$$\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t$$

- **Update gate** $\mathbf{z}_t$: balances old state and new candidate (combines forget + input)
- **Reset gate** $\mathbf{r}_t$: controls how much of the previous state goes into the new candidate

### GRU vs LSTM

|Aspect|LSTM|GRU|
|---|---|---|
|Gates|4|2|
|Cell state|Separate from hidden state|No separate cell state|
|Parameters|More (~33% more)|Fewer|
|Training speed|Slower|Faster|
|Performance|Slightly better on long sequences|Often comparable|

> **Tip:** Practical advice Try GRU first (simpler, faster); fall back to LSTM if needed. Both are largely superseded by Transformers for most tasks.

---

## 10. Bidirectional RNNs

A unidirectional RNN only sees the past. For tasks like translation or NER, future context also helps. A **BiRNN** runs two RNNs:

- Forward: processes $\mathbf{x}_1 \to \mathbf{x}_T$
- Backward: processes $\mathbf{x}_T \to \mathbf{x}_1$

Concatenate hidden states:

$$\mathbf{h}_t = [\overrightarrow{\mathbf{h}}_t; \overleftarrow{\mathbf{h}}_t]$$

> **Warning:** Caveat Can't be used for **autoregressive** tasks (language modeling, generation) since you don't have access to the future at inference.

---

## 11. Stacked / Deep RNNs

Stack multiple RNN layers — output of layer $\ell$ at time $t$ becomes input to layer $\ell+1$ at time $t$.

$$\mathbf{h}_t^{(\ell)} = \text{RNN}^{(\ell)}(\mathbf{h}_t^{(\ell-1)}, \mathbf{h}_{t-1}^{(\ell)})$$

Typical 2–4 layers. Deeper = harder to train. Often combined with residual connections across layers.

---

## 12. Encoder-Decoder (Seq2Seq)

For tasks like translation (English → French):

- **Encoder RNN:** consumes input sequence, produces a fixed-size context vector (final hidden state)
- **Decoder RNN:** initialized from context, generates output sequence one token at a time

$$\mathbf{c} = \text{Encoder}(\mathbf{x}_1, \ldots, \mathbf{x}_T)$$ $$\mathbf{y}_t = \text{Decoder}(\mathbf{y}_{t-1}, \mathbf{h}_{t-1}, \mathbf{c})$$

**Bottleneck problem:** Compressing all source info into one fixed vector $\mathbf{c}$ loses information for long sequences.

**Solution: Attention** — let the decoder look at all encoder hidden states, weighted by relevance to current decoder state. This was the precursor to Transformers.

---

## 13. Teacher Forcing

During training, the decoder receives the **true** previous token as input rather than its own prediction:

- Training: feed ground-truth $\mathbf{y}_{t-1}$ as decoder input
- Inference: feed model's own $\hat{\mathbf{y}}_{t-1}$

**Problem: exposure bias** — model never sees its own mistakes during training, so errors compound at inference.

**Mitigations:** Scheduled sampling (randomly use prediction sometimes), beam search at inference.

---

## 14. Handling Variable-Length Sequences

### The problem

Sentences have different lengths, but mini-batches need fixed-shape tensors.

```
Sentence A: "I love cats"                          (3 words)
Sentence B: "the cat sat on the mat"               (6 words)
Sentence C: "machine learning"                     (2 words)
```

### Solution 1: Padding + Masking (standard)

Pad every sentence to the longest in the batch with `<PAD>` tokens:

```
A: I love cats <PAD> <PAD> <PAD>
B: the cat sat on the mat
C: machine learning <PAD> <PAD> <PAD> <PAD>
```

Create a **mask** (1 for real, 0 for pad):

```
A mask: [1, 1, 1, 0, 0, 0]
B mask: [1, 1, 1, 1, 1, 1]
C mask: [1, 1, 0, 0, 0, 0]
```

Use mask to:

- Zero out loss at pad positions
- Extract the "real" final hidden state (at the last non-pad position)

The RNN cell **still runs `max_len` times for every sentence** — the mask handles correctness.

### Solution 2: `pack_padded_sequence` (PyTorch — more efficient)

Skips computation on padded positions:

```
Timestep 1: process all 3 sentences
Timestep 2: process all 3 sentences
Timestep 3: process A, B (C is done)
Timestep 4: process B (A is done)
Timestep 5: process B
Timestep 6: process B
```

Requires sorting by length (decreasing).

### Solution 3: Bucketing (reduces padding waste)

Group sentences of similar length into batches:

```
Bucket 1: length 1–10    → batch padded to 10
Bucket 2: length 11–20   → batch padded to 20
Bucket 3: length 21–50   → batch padded to 50
```

Standard in seq2seq training.

### Solution 4: Truncation

For very long sequences, hard-truncate to a max length, or chunk and use truncated BPTT.

### How "number of timesteps" is decided

|Setting|Timesteps =|
|---|---|
|Per sentence, theoretically|Length of that sentence|
|Per batch (with padding)|Length of longest sentence in batch|
|Per batch (with bucketing)|Bucket's max length|
|With truncation|min(sentence length, max_length)|

**The cell always runs once per token** — the only question is how many tokens you decide to process and how you handle batches of varying lengths.

---

## 15. Hyperparameters

|Hyperparameter|Typical Range|Notes|
|---|---|---|
|Hidden size|128–2048|Bigger = more capacity, harder to train|
|Number of layers|1–4|Rarely more|
|Dropout|0.2–0.5|Apply between layers, not within recurrence|
|Sequence length / truncation|50–500|For BPTT|
|Gradient clipping threshold|1.0–5.0|Critical for stability|
|Learning rate|$10^{-4}$ to $10^{-3}$|Adam typically|

---

## 16. Strengths and Weaknesses

### Strengths

- Naturally handle variable-length sequences
- Share parameters across time (compact)
- LSTM/GRU can capture moderately long dependencies (~100s of steps)
- Streaming-friendly — process one step at a time, low latency per step

### Weaknesses

- **Sequential** — no parallelism across timesteps during training
- **Slow** compared to Transformers
- **Long-range dependencies** still difficult beyond ~hundreds of steps
- **Hard to train** — gradient pathologies, hyperparameter sensitive
- **Largely superseded** by Transformers for most tasks

---

## 17. When to Use vs Not Use

### Use RNNs when:

- **Streaming / online inference** — predict as data arrives
- **Low-resource deployment** — RNNs can be smaller than Transformers
- **Truly sequential, causal** processes where attention is overkill
- **Short to medium sequences** with strong temporal structure
- Small datasets where Transformers overfit

### Don't use RNNs when:

- **Long sequences with long-range dependencies** → Transformers
- **You have lots of data and compute** → Transformers usually win
- **Highly parallel training** is needed
- Most NLP tasks today

---

## 18. RNN vs LSTM vs GRU vs Transformer

|Aspect|RNN|LSTM|GRU|Transformer|
|---|---|---|---|---|
|Long-range deps|Bad|Good|Good|Excellent|
|Parallel training|No|No|No|Yes|
|Inference latency per step|Low|Low|Low|High (attention is $O(T)$)|
|Parameters|Lowest|Most|Mid|Highest|
|Streaming|Yes|Yes|Yes|Awkward (needs caching)|
|Default choice today|No|Niche|Niche|Yes|

---

## 19. Common Pitfalls

- **Forgetting gradient clipping** → NaN losses on long sequences
- **Using BatchNorm in RNNs** — broken; use LayerNorm
- **Not masking padded positions** — model learns from pad tokens
- **Same dropout mask across timesteps?** — Variational dropout uses same mask; standard dropout uses new mask each step
- **Initializing forget gate bias to 0** in LSTM — initialize to 1 or 2
- **Using BiRNN for autoregressive generation** — impossible (no future at inference)
- **Bad truncation length** for BPTT — too short loses long-range, too long is unstable
- **Mixing teacher forcing and free running incorrectly**

---

## 20. Production Considerations

- **Pack padded sequences** in PyTorch (`pack_padded_sequence`) for efficiency
- **Use cuDNN-optimized RNN ops** (e.g., `nn.LSTM`) rather than custom Python loops
- **Quantization** for deployment — possible but trickier than CNNs
- **Streaming inference** — RNNs natural fit; maintain hidden state across calls
- **For new projects, default to Transformers** unless streaming/latency strongly favors RNNs

---

## Related Notes

- [Neural Networks](/notes/ml-algorithms/deep-learning/neural-networks/)
- CNNs
- [Attention](/notes/ml-algorithms/deep-learning/attention/) (next)
- [Transformers](/notes/ml-algorithms/deep-learning/transformers/) (upcoming)
