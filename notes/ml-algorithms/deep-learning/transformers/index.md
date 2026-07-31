---
layout: note
title: "Transformers"
description: "RNNs were the standard for sequences but had fundamental problems:"
note: true
note_collection: "ML algorithms"
note_section: "Deep Learning"
section_order: 4
note_order: 5
updated: 2026-06-06 11:30:50 -0700
keywords:
  - Deep Learning
  - Training
  - Transformers
  - Inference
  - LLMs
math: true
mermaid: false
---
> Transformers are a neural architecture built entirely on **attention** (no recurrence, no convolution). Introduced in "Attention Is All You Need" (Vaswani et al., 2017), they have become the dominant architecture in NLP and increasingly in vision, audio, and multimodal AI.

---

## Table of Contents

- [1. Why Transformers?](#1-why-transformers)
- [2. The Original Architecture (Encoder-Decoder)](#2-the-original-architecture-encoder-decoder)
- [3. Components of a Transformer Layer](#3-components-of-a-transformer-layer)
- [4. The Feedforward Network (FFN)](#4-the-feedforward-network-ffn)
- [5. Positional Encoding](#5-positional-encoding)
- [6. Multi-Head Self-Attention — Refresher](#6-multi-head-self-attention-refresher)
- [7. Putting It Together — Full Transformer Block](#7-putting-it-together-full-transformer-block)
- [8. Parameter Count of a Transformer](#8-parameter-count-of-a-transformer)
- [9. The Three Flavors of Transformer](#9-the-three-flavors-of-transformer)
- [10. Training a Transformer](#10-training-a-transformer)
- [11. Inference — Autoregressive Generation](#11-inference-autoregressive-generation)
- [12. Variants and Optimizations](#12-variants-and-optimizations)
- [13. Tokenization](#13-tokenization)
- [14. Scaling Laws](#14-scaling-laws)
- [15. Strengths and Weaknesses](#15-strengths-and-weaknesses)
- [16. Transformer vs RNN vs CNN](#16-transformer-vs-rnn-vs-cnn)
- [17. Common Pitfalls](#17-common-pitfalls)
- [18. Production Considerations](#18-production-considerations)
- [19. WALKTHROUGH — Text Classification (Encoder-Only)](#19-walkthrough-text-classification-encoder-only)
- [20. WALKTHROUGH — Translation (Encoder-Decoder)](#20-walkthrough-translation-encoder-decoder)
- [21. The Three Attention Patterns](#21-the-three-attention-patterns)

---

## 1. Why Transformers?

RNNs were the standard for sequences but had fundamental problems:

|Problem|RNN|Transformer|
|---|---|---|
|**Sequential computation**|Must process tokens one-by-one|All tokens in parallel|
|**Long-range dependencies**|Difficult (vanishing gradients)|Direct attention to any position|
|**Training speed**|Slow|Fast (GPU-friendly matmuls)|
|**Scalability**|Limited|Excellent — scales to billions of params|

Transformers solve all four by replacing recurrence with self-attention.

---

## 2. The Original Architecture (Encoder-Decoder)

The 2017 Transformer was an **encoder-decoder** model for machine translation:

```
                 Input sequence
                       │
                       ▼
              ┌────────────────┐
              │   ENCODER      │  (N stacked layers)
              │   (self-attn)  │
              └───────┬────────┘
                      │
                      ▼ (encoder outputs)
              ┌────────────────┐
              │   DECODER      │  (N stacked layers)
              │ (self-attn +   │
              │  cross-attn)   │
              └───────┬────────┘
                      │
                      ▼
               Output sequence
```

Modern variants:

- **Encoder-only:** BERT, RoBERTa — for understanding tasks (classification, NER)
- **Decoder-only:** GPT, LLaMA — for generation (LLMs)
- **Encoder-decoder:** T5, BART — for seq2seq tasks (translation, summarization)

---

## 3. Components of a Transformer Layer

A transformer encoder layer has two sublayers:

1. **Multi-Head Self-Attention**
2. **Position-wise Feedforward Network (FFN)**

Each sublayer is wrapped with **residual connection** + **layer normalization**:

$$\mathbf{x}' = \text{LayerNorm}(\mathbf{x} + \text{Sublayer}(\mathbf{x}))$$

This is **Post-LN** (original). Modern Transformers usually use **Pre-LN**:

$$\mathbf{x}' = \mathbf{x} + \text{Sublayer}(\text{LayerNorm}(\mathbf{x}))$$

Pre-LN is more stable and doesn't require learning rate warmup.

### Decoder layer adds a third sublayer

1. Masked multi-head self-attention (causal)
2. **Cross-attention** to encoder outputs
3. Feedforward network

---

## 4. The Feedforward Network (FFN)

After attention mixes information across positions, the FFN processes each position independently:

$$\text{FFN}(\mathbf{x}) = \mathbf{W}_2 \cdot \text{ReLU}(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2$$

- Inner dim is typically **4× the model dim** ($d_{\text{ff}} = 4d$).
- Same FFN weights applied to every position (parameter sharing across positions, like CNNs across space).
- Modern variants use **GELU** or **SwiGLU** instead of ReLU.

**Why FFN?** Attention mixes information _across positions_ but is linear (just weighted sums). The FFN adds _nonlinear_ processing at each position — essential for expressive power. Roughly, attention = "communicate," FFN = "think."

> **Tip:** Where the parameters live In a typical Transformer, **about 2/3 of parameters are in FFN layers**, not attention. Attention is the famous part, but FFN is doing a lot of heavy lifting.

---

## 5. Positional Encoding

Self-attention is **permutation-equivariant** — it doesn't know token order. We add **positional encodings** to embeddings so the model knows position.

### Sinusoidal (original)

$$PE_{(\text{pos}, 2i)} = \sin\left(\text{pos} / 10000^{2i/d}\right)$$ $$PE_{(\text{pos}, 2i+1)} = \cos\left(\text{pos} / 10000^{2i/d}\right)$$

- Fixed (not learned)
- Allows extrapolation to longer sequences (in theory)
- Each dimension is a sinusoid of different frequency

### Learned positional embeddings (BERT, GPT)

A learnable embedding vector for each position. Simple but doesn't extrapolate beyond training length.

### Rotary Position Embedding (RoPE) — modern (LLaMA, GPT-NeoX)

Rotates query/key vectors based on position. Naturally encodes relative position. State of the art.

### ALiBi (Attention with Linear Biases)

Adds a position-dependent bias to attention scores. Simple, extrapolates well.

---

## 6. Multi-Head Self-Attention — Refresher

Recap from [Attention](/notes/ml-algorithms/deep-learning/attention/):

$$\text{head}_i = \text{softmax}\left(\frac{\mathbf{Q}_i \mathbf{K}_i^\top}{\sqrt{d_k}}\right)\mathbf{V}_i$$

$$\text{MultiHead}(\mathbf{X}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\mathbf{W}_O$$

In a Transformer:

- **Encoder self-attention:** unmasked
- **Decoder self-attention:** masked (causal)
- **Decoder cross-attention:** unmasked, queries from decoder, keys/values from encoder

---

## 7. Putting It Together — Full Transformer Block

### Encoder block (Pre-LN version):

```
x → LN → MHSA → + (residual) → x'
x' → LN → FFN → + (residual) → output
```

### Decoder block:

```
x → LN → Masked-MHSA → + → x'
x' → LN → Cross-Attn(K,V from encoder) → + → x''
x'' → LN → FFN → + → output
```

Stack $N$ of these (typically $N = 6$–$96$).

---

## 8. Parameter Count of a Transformer

For model dim $d$, FFN dim $4d$, $h$ heads:

|Component|Params per layer|
|---|---|
|MHSA ($\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V, \mathbf{W}_O$)|$4 d^2$|
|FFN ($\mathbf{W}_1, \mathbf{W}_2$)|$8 d^2$|
|LayerNorm|$4d$ (small)|
|**Total per layer**|$\approx 12 d^2$|

For $N$ layers + embeddings: $\approx 12 N d^2 + V \cdot d$ (where $V$ is vocab size).

**Example: GPT-2 small** ($d=768, N=12$): $\approx 12 \cdot 12 \cdot 768^2 \approx 85M$ params + embedding. Actual: 124M.

---

## 9. The Three Flavors of Transformer

### Encoder-only (BERT, RoBERTa)

- Bidirectional self-attention (no mask)
- Trained with masked language modeling (predict masked tokens)
- For **understanding** tasks: classification, NER, QA
- Outputs: contextual embeddings for each token

### Decoder-only (GPT, LLaMA, Claude, etc.)

- Causal (masked) self-attention only
- Trained with next-token prediction (autoregressive LM)
- For **generation** tasks
- Standard architecture for modern LLMs

### Encoder-Decoder (T5, BART)

- Encoder reads input bidirectionally
- Decoder generates output autoregressively, cross-attending to encoder
- For seq2seq: translation, summarization, structured generation

---

## 10. Training a Transformer

### Pre-training objectives

|Model type|Objective|Description|
|---|---|---|
|**Decoder-only (GPT)**|Causal LM|Predict next token given previous|
|**Encoder-only (BERT)**|Masked LM|Predict randomly masked tokens|
|**Encoder-decoder (T5)**|Span corruption|Predict masked spans|

### Loss

Standard cross-entropy over vocabulary:

$$\mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T} \log P(y_t \mid y_{<t}, \mathbf{x})$$

### Key training details

- **Warmup + cosine decay** for learning rate (critical for stability)
- **AdamW** optimizer (with $\beta_1 = 0.9, \beta_2 = 0.95$ for LLMs)
- **Gradient clipping** (typically max norm 1.0)
- **Mixed precision** (BF16) for speed
- **Large batches** (millions of tokens per batch for LLMs)
- **Weight tying** between input embedding and output projection (saves params)

---

## 11. Inference — Autoregressive Generation

For decoder-only models, generation is one token at a time:

```python
tokens = [start_token]
for t in range(max_length):
    logits = model(tokens)
    next_token = sample(logits[-1])      # or argmax / top-k / top-p
    tokens.append(next_token)
    if next_token == eos:
        break
```

### Decoding strategies

|Method|How it picks next token|Property|
|---|---|---|
|**Greedy**|argmax|Deterministic, often repetitive|
|**Beam search**|Track top-k partial sequences|Better quality, slower|
|**Sampling**|Random per probability|Diverse but can be incoherent|
|**Top-k sampling**|Sample from top $k$ probable|Balanced|
|**Top-p (nucleus)**|Sample from smallest set with cumulative prob ≥ $p$|Adaptive, popular|
|**Temperature**|Scale logits: $\text{logits}/T$ before softmax|$T<1$ sharpens, $T>1$ flattens|

### KV Caching — Critical Optimization

At each generation step, we recompute attention over all previous tokens — but the keys and values for past tokens don't change! Cache them:

- **Without caching:** $O(T^2)$ per step → $O(T^3)$ for full generation
- **With caching:** $O(T)$ per step → $O(T^2)$ total

KV caches consume significant memory in LLM serving — often the main inference bottleneck.

---

## 12. Variants and Optimizations

### Long-context Attention

- **Sparse attention** (Longformer, BigBird): only attend to nearby + global tokens
- **Linear attention** (Linformer, Performer): kernel approximations, $O(T)$
- **FlashAttention**: IO-aware exact attention, much faster
- **Sliding window** (Mistral): local attention with limited range

### Mixture of Experts (MoE)

Replace FFN with multiple "expert" FFNs, route each token to top-$k$ experts. More parameters with similar compute per token. Used in Mixtral, GPT-4 (rumored), DeepSeek.

### Grouped-Query Attention (GQA) / Multi-Query Attention (MQA)

Share keys/values across multiple query heads. Reduces KV cache size significantly. Used in LLaMA-2, LLaMA-3.

### Parameter-Efficient Fine-Tuning

- **LoRA**: low-rank updates to attention weights — fine-tune 0.1% of params
- **Adapters**: small inserted modules
- **Prompt tuning**: learn input prefixes

---

## 13. Tokenization

Transformers operate on **tokens**, not raw text. Methods:

|Tokenizer|Used by|How|
|---|---|---|
|**Word-level**|Old NLP|Each word → token. Huge vocab.|
|**BPE (Byte-Pair Encoding)**|GPT-2, RoBERTa|Merge frequent character pairs|
|**WordPiece**|BERT|Similar to BPE|
|**SentencePiece**|T5, LLaMA|Language-agnostic, subword|
|**Tiktoken (BPE variant)**|GPT-3/4|Optimized BPE|

Subword tokenization handles rare words (split into known subwords) and keeps vocab manageable (typically 32k–100k).

---

## 14. Scaling Laws

Empirical finding: model performance follows power laws in:

- **Parameters** ($N$)
- **Dataset size** ($D$)
- **Compute** ($C$)

$$L(N) \propto N^{-\alpha}$$

Larger models, more data, more compute → predictably better loss. Driving the LLM scaling race.

**Chinchilla scaling law:** For optimal compute budget, train smaller models on more data — many LLMs were "compute-inefficient" by being too large for their training data.

---

## 15. Strengths and Weaknesses

### Strengths

- **Parallel** training over the whole sequence
- **Long-range dependencies** handled natively
- **Scales beautifully** with data and compute
- **General-purpose** — text, vision, audio, multimodal
- **Transfer learning** — pretrain once, fine-tune for many tasks
- **Pretraining-friendly** — billions of tokens

### Weaknesses

- **$O(T^2)$ attention** — long context expensive (mitigated by FlashAttention, sparse variants)
- **Memory-hungry** at training and inference (KV cache)
- **Data-hungry** — needs huge corpora for good performance
- **No inherent inductive bias** for locality (must learn from data)
- **Hallucinations** in generation
- **Inference latency** for long generations

---

## 16. Transformer vs RNN vs CNN

|Aspect|RNN/LSTM|CNN|Transformer|
|---|---|---|---|
|Parallel training|No|Yes|Yes|
|Long-range deps|Hard|Limited|Easy|
|Inductive bias|Sequential|Locality|None|
|Compute per token|$O(d^2)$|$O(k d^2)$|$O(T d)$|
|Memory|$O(T d)$|$O(T d)$|$O(T^2)$|
|Streaming inference|Natural|Natural|Awkward (KV cache)|
|Default today|Niche|Vision|Most things|

---

## 17. Common Pitfalls

- **No positional encoding** → permutation-equivariant (treats inputs as a set)
- **Wrong mask in decoder** → model peeks at future tokens, gets impossibly good train loss, fails at inference
- **No warmup** → unstable training, loss diverges (especially with Post-LN)
- **Forgetting to scale by $\sqrt{d_k}$** → vanishing gradients
- **Wrong shape conventions** — `(batch, seq, dim)` vs `(seq, batch, dim)` (PyTorch nn.Transformer uses the latter by default!)
- **Tokenization mismatch** between train and inference
- **Not using KV cache** during generation → very slow
- **Massive KV cache** running out of memory for long contexts
- **Numerical instability** in attention softmax with FP16 → use BF16 or stable softmax

---

## 18. Production Considerations

- **Pretrained models are the starting point** — fine-tune or prompt; rarely train from scratch
- **Quantization** (INT8, INT4, GGUF) for inference, especially LLMs
- **Speculative decoding** — small draft model proposes, big model verifies (2–3× speedup)
- **FlashAttention** for fast attention without quality loss
- **Tensor parallelism / pipeline parallelism** for very large models
- **Distillation** — train smaller student from larger teacher
- **LoRA fine-tuning** instead of full fine-tuning to save memory and compute
- **Continuous batching** in serving (vLLM, TensorRT-LLM) for throughput

---

## 19. WALKTHROUGH — Text Classification (Encoder-Only)

**Task:** Sentiment classification — given a sentence, predict positive/negative.

**Example:** `"the movie was really good"` → label: positive (1)

**Model:** Tiny encoder-only Transformer with 2 layers, $d=8$, 2 heads, FFN dim 32, vocab ~30k.

### Step 1: Tokenization

```
"the movie was really good"
       ↓ (tokenizer)
[CLS] the movie was really good [SEP]
       ↓
[101, 1996, 3185, 2001, 2428, 2204, 102]
```

- **`[CLS]`** at start — its final embedding will represent the whole sentence.
- **`[SEP]`** at end — sentence separator.
- Sequence length: $T = 7$.

### Step 2: Embedding Lookup

Each token ID is converted to a dense vector via an embedding table of shape `(vocab_size, d) = (30000, 8)`.

```
Token IDs:  [101, 1996, 3185, 2001, 2428, 2204, 102]
                ↓ embedding table
Embeddings: shape (7, 8)
```

Input to Transformer: $\mathbf{X} \in \mathbb{R}^{7 \times 8}$.

### Step 3: Add Positional Encoding

```
Position embeddings (shape (7, 8)):
position 0: [0.1, -0.2, 0.0, ...]
position 1: [0.0, 0.3, 0.1, ...]
...

X = token_embedding + position_embedding
shape: (7, 8)
```

Now each row of $\mathbf{X}$ encodes both **what** the token is and **where** it is.

### Step 4: First Encoder Layer

**4a. LayerNorm:** Normalize each row independently across 8 features.

**4b. Multi-Head Self-Attention** (2 heads, $d_k = 4$ each):

For each head:

```
Q = X @ W_Q  → split into (7, 4) per head
K = X @ W_K  → split
V = X @ W_V  → split

Scores  = Q @ K^T           shape: (7, 7)
Scaled  = Scores / sqrt(4)
Weights = softmax(Scaled)   each row sums to 1
Output  = Weights @ V       shape: (7, 4)
```

**Interpretation of the (7,7) weight matrix:**

- The `[CLS]` token (row 0) likely attends broadly to all tokens — it needs to summarize.
- `"really"` (row 4) might attend strongly to `"good"` (row 5) — modifier ↔ adjective.

Concatenate heads and project:

```
head_1: (7, 4)
head_2: (7, 4)
Concat: (7, 8)
@ W_O:  (7, 8)
```

**4c. Residual:**

```
X' = X + attention_output    shape: (7, 8)
```

**4d. FFN** (applied per row independently):

```
hidden = ReLU(x @ W_1 + b_1)   shape (32,)
out    = hidden @ W_2 + b_2    shape (8,)
```

> **Note:** Same $\mathbf{W}_1, \mathbf{W}_2$ for every position. FFN does NOT mix info across tokens — only attention does that.

**4e. Residual:**

```
X'' = X' + FFN(X')           shape: (7, 8)
```

### Step 5: Stack Encoder Layer 2

Repeat with a second set of weights. After layer 2:

```
Final output: shape (7, 8)
```

Each row is now a **contextualized embedding** of that token — aggregated information from the whole sentence.

### Step 6: Extract Classification Representation

```
cls_vector = final_output[0]   # shape (8,)
```

The `[CLS]` token's final embedding represents the whole sentence (attention let it aggregate info from all positions).

### Step 7: Classification Head

```
logits = cls_vector @ W_cls + b_cls    # shape (2,)
prob   = softmax(logits)
```

For our example, hopefully `prob ≈ [0.05, 0.95]` → "positive."

### Step 8: Loss

```
true_label = 1 (positive)
loss = cross_entropy(prob, true_label)
```

Backprop through everything — token embeddings, position embeddings, attention weights, FFN weights, classifier.

### Classification Pipeline Summary

```
text
  → tokenize
  → embedding lookup           (T, d)
  → + positional encoding      (T, d)
  → encoder layer × N          (T, d)
  → take [CLS] embedding       (d,)
  → classification head        (num_classes,)
  → softmax → predict
```

---

## 20. WALKTHROUGH — Translation (Encoder-Decoder)

**Task:** English → French. Translate `"the cat sat"` → `"le chat assis"`.

**Model:** 2 encoder layers, 2 decoder layers, $d=8$, 2 heads, FFN dim 32.

### Step 1: Tokenize Both Sequences

**Source (English):**

```
"the cat sat" → [CLS] the cat sat [SEP]
              → [101, 1996, 4937, 2938, 102]
              T_src = 5
```

**Target (French) — Teacher Forcing:**

```
Gold: "le chat assis"

Decoder input:  [BOS] le chat assis          (shifted right)
Decoder target: le chat assis [EOS]          (what to predict)
T_tgt = 4
```

> **Important:** Why "shifted right"? Position $t$ of the decoder's output should predict target token $t$. So we feed the gold prefix as input and predict the next token at each position.

### Step 2: Encoder — Process Source

Identical to classification's encoder:

```
[CLS] the cat sat [SEP]
   ↓ embedding + positional
   ↓ encoder layer 1
   ↓ encoder layer 2
encoder_output: shape (5, 8)
```

Each row is a contextual embedding of the corresponding English token. The decoder will use these to know what to translate.

### Step 3: Decoder Inputs

```
Decoder input: [BOS] le chat assis  →  [2, 105, 678, 423]
Embedding + positional → shape (4, 8)
```

### Step 4: Decoder Layer (3 Sublayers)

**4a. Masked Self-Attention**

Compute Q, K, V from decoder input. Apply causal mask:

```
[1  0  0  0]
[1  1  0  0]
[1  1  1  0]
[1  1  1  1]
```

Set masked positions to $-\infty$ before softmax. This prevents position 2 (`"chat"`) from peeking at position 3 (`"assis"`).

> **Important:** Why masking matters During training, the decoder sees `[BOS] le chat assis` all at once (parallel for efficiency). But we want to _simulate_ generation, where position 2 doesn't know future tokens. The mask enforces this.

Output: `(4, 8)`.

**4b. Cross-Attention — Where Encoder Meets Decoder**

This is the magic step.

```
Q = decoder_state @ W_Q       shape: (4, 8)
K = encoder_output @ W_K      shape: (5, 8)
V = encoder_output @ W_V      shape: (5, 8)

Weights: Q @ K^T              shape: (4, 5)
Softmax over English tokens.
Output:  Weights @ V          shape: (4, 8)
```

**Interpretation:**

- Row 0 (`[BOS]`) attends to source — likely focuses on `"the"`.
- Row 1 (`"le"`) attends to `"cat"` to figure out next French word (`"chat"`).
- Row 2 (`"chat"`) attends to `"sat"` to predict `"assis"`.
- Row 3 (`"assis"`) attends to `[SEP]` — translation winding down.

This is the **soft alignment** between source and target — solving the "context vector bottleneck" of old seq2seq.

**4c. FFN**

Position-wise, like in the encoder. Output: `(4, 8)`.

(All three sublayers have residual + LayerNorm.)

### Step 5: Stack Decoder Layer 2

Repeat. Final decoder output: shape `(4, 8)`.

### Step 6: Output Projection

```
logits = decoder_output @ W_vocab    # shape (4, 30000)
probs  = softmax(logits, dim=-1)
```

For each of the 4 positions, a probability distribution over the French vocab.

### Step 7: Loss (Training Time)

```
Position 0 should predict "le"     (token 105)
Position 1 should predict "chat"   (token 678)
Position 2 should predict "assis"  (token 423)
Position 3 should predict [EOS]    (token 3)

loss = average cross-entropy over 4 positions
```

Backprop through encoder, decoder, all attention layers, embeddings, output projection.

### Step 8: Inference (Generation)

At inference, we **don't have the gold target**. We generate one token at a time:

```
Step 1:
  Decoder input: [BOS]
  → forward pass → predict "le"

Step 2:
  Decoder input: [BOS] le
  → forward pass → predict "chat"

Step 3:
  Decoder input: [BOS] le chat
  → forward pass → predict "assis"

Step 4:
  Decoder input: [BOS] le chat assis
  → forward pass → predict [EOS]

Stop. Final output: "le chat assis"
```

> **Important:** Training vs Inference
> 
> - **Training:** all positions processed in parallel with teacher forcing.
> - **Inference:** sequential, one token at a time. Each step does a full forward pass.
> 
> **KV caching** speeds this up: encoder runs once at start; decoder's keys/values for past tokens are cached.

### Translation Pipeline Summary

```
source text                             target text (training only)
  → tokenize                              → tokenize
  → embedding + pos                       → shift right + embedding + pos
  → encoder × N                                   ↓
  → encoder_output  ──────► cross-attn ◄──  decoder × N (masked self-attn + cross-attn + FFN)
                                                  ↓
                                          → output projection
                                          → softmax → token probs
```

---

## 21. The Three Attention Patterns

In a full encoder-decoder Transformer, there are three distinct attention layers, each with its own role:

|Where|Q from|K, V from|Masked?|
|---|---|---|---|
|**Encoder self-attention**|Source|Source|No|
|**Decoder self-attention**|Target (so far)|Target (so far)|Yes (causal)|
|**Decoder cross-attention**|Target|**Encoder output**|No|

### What each one does

- **Encoder self-attn:** Source tokens talk to each other (resolves pronouns, captures syntax).
- **Decoder self-attn:** Target tokens talk to past target tokens (ensures fluent output language).
- **Decoder cross-attn:** Target tokens read from the source (the actual "translation" linking).
- **FFN:** Adds nonlinear processing at each position.
- **Output projection:** Converts representations to vocabulary probabilities.

> **Tip:** What if you removed one?
> 
> - **No cross-attn:** Decoder becomes a language model that ignores the source.
> - **No decoder self-attn:** Model generates each word without considering what it already wrote.
> - **No encoder self-attn:** Source representations don't contextualize each other.
> 
> Each piece plays a role.

### Mental models

**Classification:** "Read the input, summarize it in the `[CLS]` token, classify."

**Translation:** "Encoder builds a rich representation of the source. Decoder generates the target one token at a time, looking back at what it's already written (self-attn, masked) and looking at the source (cross-attn)."

---

## Related Notes

- [Neural Networks](/notes/ml-algorithms/deep-learning/neural-networks/)
- CNNs
- RNNs
- [Attention](/notes/ml-algorithms/deep-learning/attention/)
- Vision Transformers (next)
- LLMs (upcoming)
