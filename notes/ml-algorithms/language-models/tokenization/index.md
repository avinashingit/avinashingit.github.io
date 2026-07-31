---
layout: note
title: "Tokenization"
description: "A transformer never sees characters or words — it sees a sequence of integer token IDs, each of which indexes a row in the embedding table. Tokenization is the algorithm that sp…"
note: true
note_collection: "ML algorithms"
note_section: "Language Models"
section_order: 6
note_order: 21
updated: 2026-06-07 03:55:26 -0700
keywords:
  - LLMs
  - Transformers
  - Deep Learning
  - Embeddings
  - Probability
math: true
mermaid: true
---
> Tokenization is the reversible mapping from raw text to a sequence of integer IDs (and back) that a language model actually consumes and emits. Related: [LLM Architecture](/notes/ml-algorithms/language-models/llm-architecture/), [Pretraining and Language Modeling](/notes/ml-algorithms/language-models/pretraining-and-language-modeling/), [Transformers](/notes/ml-algorithms/deep-learning/transformers/)

## TL;DR

A transformer never sees characters or words — it sees a sequence of integer **token IDs**, each of which indexes a row in the embedding table. Tokenization is the algorithm that splits text into these tokens. Modern LLMs use **subword** tokenizers (BPE, WordPiece, Unigram/SentencePiece) that sit between characters and words: common words become one token, rare words split into pieces, and nothing is ever unrepresentable. The choice of tokenizer and vocabulary size silently shapes sequence length, cost, arithmetic ability, code quality, and how fairly the model treats non-English languages.

## Why it matters

Neural networks operate on vectors of floats, not strings. So the very first thing any LLM does is convert text into a sequence of discrete symbols drawn from a fixed **vocabulary** $V$, then look up an embedding vector per symbol. Everything downstream — context length, FLOPs per forward pass, the cross-entropy loss target, the cost you pay per API call — is denominated in *tokens*, not words or characters.

The naive options are bad at the extremes:

- **Character-level:** vocabulary is tiny (~256 bytes), nothing is ever out-of-vocabulary, but sequences become very long. A 1,000-word page becomes ~5,000+ symbols, and attention is $O(n^2)$ in sequence length $n$, so this is expensive and forces the model to relearn spelling for every word.
- **Word-level:** sequences are short, but the vocabulary explodes (millions of words across morphology, typos, and languages) and you inevitably hit unknown words that map to a single `[UNK]` token — destroying information.

**Subword tokenization** is the compromise that won: a vocabulary of typically 30k–256k pieces where frequent words are single tokens and rare/novel strings decompose into smaller known pieces. This keeps sequences short *and* keeps the vocabulary bounded *and* (with byte-level fallback) never emits `[UNK]`.

## How it works

The dominant family is **Byte Pair Encoding (BPE)**, originally a compression algorithm. Training is greedy and frequency-driven:

1. Start with a base vocabulary of individual characters (or bytes).
2. Count all adjacent symbol pairs in the training corpus.
3. **Merge the single most frequent adjacent pair** into a new symbol; add it to the vocabulary as a new **merge rule**.
4. Repeat until you reach the target vocabulary size (number of merges is the main hyperparameter).

At inference, you apply the learned merge rules **in the order they were learned**, greedily, until no rule applies.

**Worked example.** Corpus (word : count): `low:5, lower:2, newest:6, widest:3`. Start from characters, with a word-boundary marker `</w>`:

| Step | Most frequent pair | New token | Why |
|------|--------------------|-----------|-----|
| 1 | `e` + `s` (9: newest×6 + widest×3) | `es` | most common adjacency |
| 2 | `es` + `t` (9) | `est` | merges build on prior merges |
| 3 | `est` + `</w>` (9) | `est</w>` | a frequent suffix emerges |
| 4 | `l` + `o` (7: low×5 + lower×2) | `lo` | next most frequent |

After enough merges, `newest` is one token, while an unseen word like `lowest` still decomposes cleanly into `lo` + `w` + `est</w>`. The merges have effectively discovered morphology (the suffix `-est`) without any linguistic supervision.

<pre class="mermaid">
flowchart LR
  T[&quot;raw text&lt;br/&gt;&#39;unhappiness&#39;&quot;] --&gt; P[&quot;pre-tokenize&lt;br/&gt;(split on whitespace&lt;br/&gt;and punctuation)&quot;]
  P --&gt; M[&quot;apply merge rules&lt;br/&gt;greedily, in order&quot;]
  M --&gt; S[&quot;subword tokens&lt;br/&gt;un | happ | iness&quot;]
  S --&gt; I[&quot;token IDs&lt;br/&gt;[894, 2051, 7720]&quot;]
  I --&gt; E[&quot;embedding lookup&lt;br/&gt;rows of W_emb (V x d)&quot;]
  E --&gt; X[&quot;to transformer layers&quot;]
</pre>
**Byte-level BPE (GPT-2, GPT-3, GPT-4, Llama family in spirit).** Instead of starting from Unicode characters, start from the **256 raw bytes**. Any string in any language — emoji, CJK text, control characters — is just a byte sequence, so the base alphabet covers *everything* and there is **no `[UNK]` token, ever**. Merges then operate over bytes. GPT-2's vocabulary is 50,257; GPT-4's `o200k_base` is ~200k, which shortens sequences (fewer tokens per word) at the cost of a larger embedding table.

**WordPiece (BERT).** Same character-merging spirit, but instead of merging the *most frequent* pair, it merges the pair that most increases the **likelihood** of the training corpus under a unigram language model. Concretely it picks the pair maximizing $\frac{\text{count}(ab)}{\text{count}(a)\,\text{count}(b)}$ — i.e., pairs that co-occur more than chance, not just often. WordPiece marks word-internal continuation with `##` (e.g., `playing` → `play`, `##ing`).

**Unigram LM / SentencePiece.** SentencePiece is a *library* (it implements both BPE and Unigram); **Unigram** is a distinct *algorithm*. Unigram starts from a large candidate vocabulary and *prunes* it: it keeps a probabilistic model where each token has a probability $p(x_i)$, scores a sentence's best segmentation via $\arg\max \prod_i p(x_i)$, and iteratively removes tokens whose loss in corpus likelihood is smallest. Crucially, SentencePiece treats input as a **raw stream of bytes/characters with no pre-tokenization**, encoding whitespace explicitly as the meta-symbol `▁` (U+2581). This makes tokenization fully reversible and language-agnostic — vital for languages like Chinese, Japanese, and Thai that don't delimit words with spaces. Llama and T5 use SentencePiece.

## Variants / Types / Trade-offs

| Tokenizer | Merge criterion | Base units | `[UNK]`? | Used by | When to use |
|-----------|-----------------|------------|----------|---------|-------------|
| **BPE (char)** | most frequent adjacent pair | characters | possible (rare chars) | early GPT, RoBERTa | simple, strong default |
| **Byte-level BPE** | most frequent adjacent pair | 256 bytes | **never** | GPT-2/3/4, many open models | multilingual, code, any Unicode |
| **WordPiece** | max corpus likelihood (pointwise mutual info) | characters | yes (`[UNK]`) | BERT, DistilBERT | encoder/classification models |
| **Unigram (SentencePiece)** | prune to max likelihood | chars/bytes, `▁` for space | rare | Llama, T5, ALBERT, mBART | space-free languages, reversibility |

**Vocabulary size is a direct trade-off.** A larger $|V|$ means:

- **Shorter sequences** → less compute (attention is $O(n^2 d)$) and more text fits in a fixed context window. This is the win.
- **Bigger embedding + output (unembedding) matrices**, each of size $|V| \times d$. At $d = 4096$ and $|V| = 256{,}000$, that's ~1B parameters *per* matrix, plus a wider final softmax that costs FLOPs at every decoding step.
- **Sparser statistics** per token (rare tokens get few gradient updates → undertrained embeddings — the seed of "glitch tokens", below).

Typical sweet spots: 32k (Llama 2), 100k–256k (GPT-4, Llama 3, Gemma). English averages **~1.3 tokens per word** (~4 characters per token) with a 50k+ BPE vocab.

## Practical considerations

- **Numbers and arithmetic.** If `1234` tokenizes as `12`+`34` (or worse, inconsistently across contexts), the model can't see digits positionally and struggles with math. Newer tokenizers split numbers into individual digits or fixed 3-digit chunks specifically to fix this. This is *the* tokenization reason LLMs are flaky at arithmetic.
- **Code.** Indentation, brackets, and `camelCase`/`snake_case` matter. GPT-4's tokenizer added explicit multi-space tokens (e.g., a single token for 4 or 8 spaces) so Python indentation doesn't blow up token counts; good code tokenizers materially improve code modeling.
- **Multilingual fairness ("fertility").** Fertility = tokens per word in a language. English might be 1.3; the *same sentence* in Hindi, Burmese, or Thai can take 3–6× more tokens under an English-centric tokenizer. That means non-English users pay more per API call, fit less in context, and the model wastes capacity on encoding — a real equity issue. Llama 3 and Gemma widened vocabularies partly to reduce this.
- **Special tokens.** Reserved IDs outside normal text: **BOS** (beginning of sequence), **EOS** (end of sequence — the model emits this to *stop* generating), **PAD** (fills batches to equal length; masked out of attention/loss), and chat/role markers like `<|system|>`, `<|user|>`, `<|assistant|>` that delimit turns. Getting these wrong (e.g., a missing EOS, or a chat template mismatch between training and serving) causes runaway generation or degraded quality. See [Decoding Strategies](/notes/ml-algorithms/language-models/decoding-strategies/).
- **Tokenizer = part of the model contract.** You cannot swap tokenizers without retraining; the embedding table is indexed by exact token IDs. Counting tokens (for cost/context budgeting) requires the *model's own* tokenizer (e.g., `tiktoken` for GPT models).
- **Spelling and "count the r's in strawberry".** Because the model sees `straw`+`berry`, not individual letters, character-level tasks (reversing strings, counting letters, rhyming) are unnaturally hard — the information is *averaged into* one embedding.
- **Glitch tokens / "SolidGoldMagikarp".** Some tokens (e.g., scraped Reddit usernames, forum artifacts) entered the vocabulary from the tokenizer's corpus but barely appeared in the *model's* training data. Their embeddings stayed near-random/undertrained, so prompting them produces bizarre, evasive, or unhinged output. It's a vivid reminder that the tokenizer and the model are trained on different data, and undertrained rare tokens are a live failure mode.

## Related

- Siblings: [LLM Architecture](/notes/ml-algorithms/language-models/llm-architecture/) · [Pretraining and Language Modeling](/notes/ml-algorithms/language-models/pretraining-and-language-modeling/) · [Positional Encodings](/notes/ml-algorithms/language-models/positional-encodings/) · [Decoding Strategies](/notes/ml-algorithms/language-models/decoding-strategies/) · [Long Context](/notes/ml-algorithms/language-models/long-context/)
- Foundations: [Transformers](/notes/ml-algorithms/deep-learning/transformers/) · [Cross-Entropy Loss](/notes/ml-algorithms/core-concepts/cross-entropy-loss/) · [Softmax](/notes/ml-algorithms/core-concepts/softmax/) · [Neural Networks](/notes/ml-algorithms/deep-learning/neural-networks/)
