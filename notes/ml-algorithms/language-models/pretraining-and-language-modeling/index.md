---
layout: note
title: "Pretraining and Language Modeling"
description: "Pretraining trains a model on a single, brutally simple objective: given the tokens seen so far, predict the next one. The loss is per-token cross-entropy (negative log-likeliho…"
note: true
note_collection: "ML algorithms"
note_section: "Language Models"
section_order: 6
note_order: 12
updated: 2026-06-07 03:58:33 -0700
keywords:
  - LLMs
  - Transformers
  - Training
  - Probability
  - Supervised Learning
math: true
mermaid: true
---
> Pretraining is the self-supervised stage where a model learns to predict the next token over trillions of tokens of text, minimizing token-level cross-entropy. Related: [Cross-Entropy Loss](/notes/ml-algorithms/core-concepts/cross-entropy-loss/), [Tokenization](/notes/ml-algorithms/language-models/tokenization/), [Scaling Laws](/notes/ml-algorithms/language-models/scaling-laws/), [LLM Architecture](/notes/ml-algorithms/language-models/llm-architecture/)

## TL;DR

Pretraining trains a model on a single, brutally simple objective: given the tokens seen so far, predict the next one. The loss is per-token cross-entropy (negative log-likelihood) over the vocabulary, summed across the sequence. We measure it intrinsically with **perplexity** = $\exp(\text{average NLL})$. This objective needs no human labels — the text *is* its own supervision — yet to lower its loss the model is forced to absorb grammar, facts, and reasoning, which is why it yields broadly capable base models that [Supervised Fine-Tuning](/notes/ml-algorithms/language-models/supervised-fine-tuning/) and RLHF later steer.

## Why it matters

Pretraining is the foundation of the entire LLM stack. Everything downstream — instruction following, chat, tool use, reasoning — is a *refinement* of a base model whose knowledge and skills were almost entirely acquired here. The key insight is **self-supervision**: labeled data is scarce and expensive, but raw text is essentially free at web scale, and next-token prediction turns every document into millions of "predict the masked-off future" training examples for free.

Why does such a trivial-looking objective produce world knowledge and reasoning? Because accurately predicting the next token in arbitrary text is **AI-complete in the limit**. To predict the token after "The capital of France is", the model must store a fact. To finish a proof, it must follow logic. To continue code, it must model syntax and semantics. Compression is understanding: lowering loss on diverse text *requires* internalizing the structure of the world that generated that text. This is the empirical engine behind [Scaling Laws](/notes/ml-algorithms/language-models/scaling-laws/) — more data, parameters, and compute monotonically lower loss and unlock capabilities.

## How it works

**The objective (causal / autoregressive LM).** Text is first turned into a sequence of integer token IDs by the [Tokenization](/notes/ml-algorithms/language-models/tokenization/) step. The model factorizes the joint probability of a sequence $x = (x_1, \dots, x_T)$ using the chain rule, predicting each token conditioned only on *earlier* tokens:

$$p_\theta(x) = \prod_{t=1}^{T} p_\theta(x_t \mid x_{<t})$$

where $x_{<t} = (x_1, \dots, x_{t-1})$ is the prefix and $\theta$ are the model weights. At each position the network ([LLM Architecture](/notes/ml-algorithms/language-models/llm-architecture/), a decoder-only Transformer) outputs a logit vector, and [Softmax](/notes/ml-algorithms/core-concepts/softmax/) converts it to a distribution over the $V$-sized vocabulary. **Causal masking** in self-attention guarantees position $t$ cannot peek at $x_{\geq t}$ — that is what makes the factorization valid.

**The loss.** We minimize the negative log-likelihood, which is exactly token-level [Cross-Entropy Loss](/notes/ml-algorithms/core-concepts/cross-entropy-loss/) against the one-hot true next token:

$$\mathcal{L}(\theta) = -\sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t})$$

Here $p_\theta(x_t \mid x_{<t})$ is the softmax probability the model assigned to the *actual* next token. The gradient of this loss flows through every layer via backprop. Note the supervision is automatic: the "label" for position $t$ is simply the token at position $t{+}1$, obtained by shifting the input sequence by one.

**Teacher forcing.** During training we always feed the model the *ground-truth* prefix, not its own predictions, computing all $T$ losses in one parallel forward pass. This is **teacher forcing** — it makes training fully parallel and stable. (At inference there is no ground truth, so the model is fed its own previous outputs autoregressively; the train/inference mismatch is called *exposure bias*.)

**Perplexity.** The intrinsic quality metric is perplexity, the exponentiated average per-token NLL:

$$\text{PPL} = \exp\!\left(\frac{1}{T}\sum_{t=1}^{T} -\log p_\theta(x_t \mid x_{<t})\right)$$

Intuitively, PPL is the **effective branching factor**: a perplexity of 10 means the model is, on average, as uncertain as if choosing uniformly among 10 equally likely next tokens. Lower is better; PPL $= 1$ is perfect prediction. It is tokenizer- and dataset-dependent, so PPL is only comparable across models that share a vocabulary and eval set (see [Metrics](/notes/ml-algorithms/core-concepts/metrics/)).

<pre class="mermaid">
flowchart LR
  C[&quot;Web-scale corpus&lt;br/&gt;(Common Crawl, code, books)&quot;] --&gt; F[&quot;Dedup + quality filter&lt;br/&gt;+ data mixture&quot;]
  F --&gt; TOK[&quot;Tokenize&lt;br/&gt;(BPE to token IDs)&quot;]
  TOK --&gt; SHIFT[&quot;Shift by one&lt;br/&gt;(input vs target)&quot;]
  SHIFT --&gt; M[&quot;Decoder-only Transformer&lt;br/&gt;(causal mask)&quot;]
  M --&gt; P[&quot;Softmax over vocab&lt;br/&gt;p(x_t | x_&lt;t)&quot;]
  P --&gt; L[&quot;Cross-entropy / NLL loss&quot;]
  L --&gt;|&quot;backprop + AdamW&quot;| M
</pre>
**The data side.** Model quality is bounded by data quality, and curation is where much of the real work happens:

- **Sources.** Web text ([Common Crawl]), curated code (e.g., GitHub), books, Wikipedia, academic papers, and increasingly high-quality synthetic data. A deliberate **data mixture** weights these (e.g., upsampling code and books, downweighting low-quality web spam).
- **Deduplication.** Exact and near-duplicate (MinHash / fuzzy) removal. Dedup improves generalization, reduces memorization/regurgitation, and prevents wasting compute re-learning repeated text.
- **Quality filtering.** Heuristic filters (language ID, perplexity filtering, toxicity/PII removal) plus learned classifiers that score "is this document high quality." Benchmark decontamination removes test-set leakage.
- **One epoch over trillions of tokens.** Frontier models train on the order of $10^{13}$ (multiple trillions of) tokens for roughly a *single* pass. Because the corpus is so large, repeating data risks memorization with little benefit, so pretraining is usually $\approx$ 1 epoch (some high-value subsets are mildly upsampled).
- **Compute scale.** This costs $10^{24}$–$10^{26}$ FLOPs, thousands of GPUs/TPUs running for weeks to months, and is the single most expensive step in the lifecycle — which is why the [Scaling Laws](/notes/ml-algorithms/language-models/scaling-laws/) question of "how to spend a fixed compute budget" is so consequential.

## Variants / Types / Trade-offs

The "predict the masked-off text" idea has three major flavors:

| Objective | Examples | Attention | Predicts | Best for | Limitation |
|---|---|---|---|---|---|
| **Causal LM** (autoregressive) | GPT, Llama, most modern LLMs | Unidirectional (causal mask) | Next token $x_t$ from $x_{<t}$ | Open-ended **generation**, chat, reasoning | Each position sees only the left context |
| **Masked LM** (MLM) | BERT, RoBERTa | Bidirectional (full attention) | Randomly masked tokens (~15%) from both sides | **Encoding / understanding** (classification, retrieval embeddings) | Not generative; train/test mask mismatch; less sample-efficient (only ~15% of tokens give signal) |
| **Span corruption** (denoising seq2seq) | T5, UL2 | Encoder bidirectional, decoder causal | Contiguous **spans** replaced by sentinel tokens, regenerated by the decoder | Flexible text-to-text, summarization, translation | Heavier encoder-decoder; more complex than pure decoder |

**When to use which.** For a general-purpose generative LLM, **causal LM** has won — it is the most data-efficient (every token is a prediction target), scales cleanly, and naturally produces text. MLM remains the default for **embedding/understanding** models where you want a single rich bidirectional representation and don't need to generate. Span corruption (T5) is a middle ground that frames every task as text-to-text. Note BERT can't generate fluently and GPT can't see the future — the asymmetry is fundamental to the objective, not the size.

## Practical considerations

- **Loss is the north star.** During a multi-week run, engineers watch the train/val loss curve obsessively — a smooth, predictable decline (per [Scaling Laws](/notes/ml-algorithms/language-models/scaling-laws/)) signals health; a spike usually means a bad batch, a data bug, or an instability requiring a restart from the last checkpoint.
- **Loss vs perplexity vs benchmarks.** Cross-entropy loss and perplexity are *intrinsic* and great for monitoring, but they don't perfectly predict downstream task scores. Final model selection uses *extrinsic* benchmarks ([LLM Evaluation](/notes/ml-algorithms/language-models/llm-evaluation/)).
- **Document packing.** To avoid wasting compute on padding, multiple short documents are concatenated up to the context length; an attention mask (or document separators) prevents cross-document attention bleed.
- **Optimizer defaults.** AdamW, cosine learning-rate schedule with linear warmup, gradient clipping, mixed precision (BF16), and a large batch (often millions of tokens) for stable gradients.
- **Data quality dominates.** Going from raw Common Crawl to a well-filtered, deduplicated mixture (the lesson of datasets like FineWeb) often buys more than scaling parameters at fixed compute. Garbage in, garbage out is literal here.
- **Base vs instruct.** Pretraining yields a **base model** that *continues* text but doesn't follow instructions or chat. Usable assistants require post-training: [Supervised Fine-Tuning](/notes/ml-algorithms/language-models/supervised-fine-tuning/) then [RLHF and Preference Optimization](/notes/ml-algorithms/language-models/rlhf-and-preference-optimization/). Pretraining provides the raw capability; alignment makes it useful and safe.
- **Memorization and contamination.** Insufficient dedup leads to verbatim regurgitation (a privacy and copyright risk) and inflated benchmark scores from test-set leakage — both are caught by aggressive filtering and decontamination.

## Related

- Foundations: [Cross-Entropy Loss](/notes/ml-algorithms/core-concepts/cross-entropy-loss/) · [Softmax](/notes/ml-algorithms/core-concepts/softmax/) · [Tokenization](/notes/ml-algorithms/language-models/tokenization/) · [Metrics](/notes/ml-algorithms/core-concepts/metrics/)
- Architecture: [LLM Architecture](/notes/ml-algorithms/language-models/llm-architecture/) · [Attention Variants and Efficiency](/notes/ml-algorithms/language-models/attention-variants-and-efficiency/)
- Scaling the objective: [Scaling Laws](/notes/ml-algorithms/language-models/scaling-laws/)
- Post-training (what builds on this): [Supervised Fine-Tuning](/notes/ml-algorithms/language-models/supervised-fine-tuning/) · [RLHF and Preference Optimization](/notes/ml-algorithms/language-models/rlhf-and-preference-optimization/) · [Parameter-Efficient Fine-Tuning](/notes/ml-algorithms/language-models/parameter-efficient-fine-tuning/)
- Evaluation: [LLM Evaluation](/notes/ml-algorithms/language-models/llm-evaluation/)
