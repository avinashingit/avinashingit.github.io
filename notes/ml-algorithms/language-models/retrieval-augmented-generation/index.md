---
layout: note
title: "Retrieval-Augmented Generation"
description: "RAG = retrieval + generation. Offline, you chunk a document corpus, embed each chunk with a bi-encoder text-embedding model, and store the vectors in a vector database. Online,…"
note: true
note_collection: "ML algorithms"
note_section: "Language Models"
section_order: 6
note_order: 16
updated: 2026-06-07 04:02:52 -0700
keywords:
  - LLMs
  - Transformers
  - Retrieval
  - Embeddings
  - Evaluation
math: true
mermaid: true
---
> Instead of relying only on what the model memorized in its weights, **retrieve** relevant text from an external corpus at query time and put it in the prompt so the LLM generates a **grounded**, citable answer. Related: Retrieval-Augmented Generation System, [Long Context](/notes/ml-algorithms/language-models/long-context/), [Prompt Engineering](/notes/ml-algorithms/language-models/prompt-engineering/), [Hallucination and Safety](/notes/ml-algorithms/language-models/hallucination-and-safety/)

## TL;DR

**RAG** = retrieval + generation. Offline, you **chunk** a document corpus, **embed** each chunk with a bi-encoder text-embedding model, and store the vectors in a **vector database**. Online, you embed the user's query, run **approximate nearest-neighbor (ANN)** search to pull the top-$k$ most similar chunks, optionally **rerank** them, paste them into the prompt, and let the LLM answer *from that evidence*. It cuts hallucination, injects fresh/private knowledge without retraining, and enables citations — usually far cheaper than fine-tuning for knowledge. This note is the **technique**; for the full production architecture (caching, scaling, monitoring, freshness) see Retrieval-Augmented Generation System.

## Why it matters

An LLM's knowledge is **parametric** — frozen in its weights at pretraining time. That creates three problems: it goes **stale** (doesn't know last week's events), it doesn't know your **private/proprietary** data (internal wikis, a customer's tickets), and when unsure it tends to **hallucinate** a fluent-but-wrong answer (see [Hallucination and Safety](/notes/ml-algorithms/language-models/hallucination-and-safety/)). The naive fix — fine-tune on new knowledge — is expensive, slow to update, hard to attribute, and prone to forgetting.

RAG sidesteps this by treating knowledge as **non-parametric**: keep it in an external index and fetch on demand. The LLM becomes a *reasoner over supplied evidence* rather than a *memorizer of facts*. Update the index and the system's knowledge changes instantly — no retraining. Because the answer is built from retrieved passages, you can show **citations**, essential for trust in enterprise, legal, and medical settings. RAG sits in the **application layer**, wrapping a frozen model with a retrieval pipeline.

## How it works

RAG has two phases: an **offline indexing** pipeline (run once per corpus, refreshed on changes) and an **online query** pipeline (run per request).

<pre class="mermaid">
flowchart TD
    subgraph OFF[&quot;Offline: ingest and index&quot;]
        D1[&quot;Raw documents (PDF, HTML, DB)&quot;] --&gt; C1[&quot;Chunk (size + overlap)&quot;]
        C1 --&gt; E1[&quot;Embed chunks (bi-encoder)&quot;]
        E1 --&gt; V1[&quot;Vector DB (HNSW index)&quot;]
    end
    subgraph ON[&quot;Online: retrieve and generate&quot;]
        Q1[&quot;User query&quot;] --&gt; QR[&quot;Query rewrite (optional)&quot;]
        QR --&gt; E2[&quot;Embed query (same model)&quot;]
        E2 --&gt; AN[&quot;ANN search: top-k chunks&quot;]
        V1 -. &quot;index lookup&quot; .-&gt; AN
        AN --&gt; RR[&quot;Rerank (cross-encoder, optional)&quot;]
        RR --&gt; PR[&quot;Assemble prompt: query + chunks&quot;]
        PR --&gt; GEN[&quot;LLM generates grounded answer&quot;]
        GEN --&gt; OUT[&quot;Answer + citations&quot;]
    end
</pre>
### Embeddings and the bi-encoder

An **embedding** maps text to a dense vector $\mathbf{x} \in \mathbb{R}^d$ (commonly $d = 384$ to $3072$) such that semantically similar texts land close together. RAG uses a **bi-encoder**: query and documents are encoded **independently**, so all document vectors are precomputed offline. Similarity is usually **cosine similarity**:

$$\text{sim}(\mathbf{q}, \mathbf{x}) = \frac{\mathbf{q} \cdot \mathbf{x}}{\lVert \mathbf{q} \rVert \, \lVert \mathbf{x} \rVert}$$

where $\mathbf{q}$ is the query vector and $\mathbf{x}$ a chunk vector. If vectors are L2-normalized, cosine is equivalent to dot product and to ranking by Euclidean distance — so retrieval reduces to **nearest-neighbor search** in embedding space (this is exactly [kNN](/notes/ml-algorithms/supervised-learning/knn/), just at scale). Critically, query and documents **must use the same embedding model**, or the spaces don't align.

### Chunking

Documents are split into **chunks** because (1) embeddings represent a bounded span well but smear meaning over a whole book, and (2) you want to retrieve *just* the relevant passage, not a 50-page file. Typical chunks are **256–512 tokens** with **10–20% overlap** so a fact straddling a boundary survives in at least one chunk. Smarter strategies split on **semantic/structural boundaries** (paragraphs, headings, code functions) rather than fixed lengths. Chunk size is a core tuning knob: too small loses context, too large dilutes the embedding and wastes prompt budget.

### ANN search and the vector database

A corpus has millions–billions of chunks; comparing the query to every vector (**exact / brute-force kNN**) is $O(N d)$ and too slow. A **vector database** builds an **ANN** index that trades a tiny bit of recall for sub-linear latency. The dominant index is **HNSW** (Hierarchical Navigable Small World) — a multi-layer proximity graph you greedily walk toward the nearest neighbors in roughly $O(\log N)$ hops. Other families: **IVF** (cluster the space, search only nearby cells) and **PQ** (Product Quantization, compress vectors to cut memory). This is **dense / semantic retrieval**: it matches *meaning*, so "how do I reset my password" can retrieve a chunk about "account recovery" with no shared keywords.

### Reranking with a cross-encoder

ANN over a bi-encoder is fast but coarse — it never lets the query and document *interact*. A **cross-encoder** concatenates `[query, chunk]` and runs them **jointly** through a transformer to output one relevance score. It's far more accurate but $O(k)$ model calls, so you can't run it over the whole corpus. The standard pattern is **retrieve-then-rerank**: bi-encoder fetches top-$k$ (e.g. 50–100) cheaply, cross-encoder reranks them down to the best top-$n$ (e.g. 5) that go into the prompt.

### Hybrid retrieval

Dense retrieval can miss exact matches — rare names, product codes, IDs, acronyms — because they're poorly represented in embedding space. **Sparse** retrieval (**BM25**, a TF-IDF-style keyword score) nails those but misses paraphrases. **Hybrid retrieval** runs both and **fuses** the rankings, commonly with **Reciprocal Rank Fusion**:

$$\text{RRF}(d) = \sum_{r \in \text{retrievers}} \frac{1}{k_0 + \text{rank}_r(d)}$$

where $\text{rank}_r(d)$ is document $d$'s rank under retriever $r$ and $k_0 \approx 60$ a smoothing constant. Hybrid reliably improves **recall** — the chance the right chunk is in the candidate set — which is the single biggest determinant of RAG quality.

## Variants / Trade-offs: RAG vs Long Context vs Fine-tuning

These three are the canonical ways to give an LLM knowledge it didn't have. They're complementary, not mutually exclusive.

| Dimension | RAG (retrieve chunks) | [Long Context](/notes/ml-algorithms/language-models/long-context/) (stuff it all in) | [Supervised Fine-Tuning](/notes/ml-algorithms/language-models/supervised-fine-tuning/) (bake into weights) |
|---|---|---|---|
| **Best for** | Large, changing knowledge bases | One self-contained doc that fits | Style, format, skills (not facts) |
| **Freshness** | Instant — update the index | Limited to what you paste in | Stale until you retrain |
| **Cost per query** | Low — only top-$k$ chunks in prompt | High — pay for *all* tokens each call | Low at inference; training is costly |
| **Citations** | Native — you know the source chunks | Possible but weaker | None — knowledge is opaque |
| **Hallucination** | Lower (grounded in evidence) | Lower if it fits, "lost in the middle" risk | Can still confidently fabricate |
| **Engineering** | High — chunking, embeddings, vector DB, rerank | Low — just a long prompt | Medium — data curation + training |
| **Main failure** | Bad retrieval = missing evidence | Quadratic cost; mid-context dilution | Catastrophic forgetting; drift |

**Rule of thumb:** use **fine-tuning to teach *behavior*** (tone, JSON schemas, a domain's reasoning style) and **RAG to supply *knowledge*** (facts that change or are too large to memorize). For a small, static document that fits the window, plain long context is simplest. Hybrids are common: retrieve the best chunks to *fill* a long window (see [Long Context](/notes/ml-algorithms/language-models/long-context/) for the cost/latency trade-off).

## Practical considerations

- **Retrieval is the bottleneck, not generation.** If the right chunk isn't retrieved, no LLM cleverness recovers it. Optimize recall first: tune chunking, add hybrid + reranking, fix embedding-model mismatch.
- **Chunking defaults.** Start ~512 tokens with ~15% overlap; prefer structural splits. Store metadata (title, URL, timestamp) per chunk for citations and **metadata filtering** (e.g. one tenant or date range).
- **Top-$k$ and the prompt budget.** More chunks raises recall but adds noise/cost and worsens **lost-in-the-middle** — place the strongest chunks first/last. Typical $k$ after rerank is 3–8.
- **Query rewriting.** Conversational queries ("and its pricing?") are bad retrieval keys. Rewrite to a standalone query, or use **multi-query** / **HyDE** (embed a hypothetical answer) to widen recall.
- **Index maintenance.** Knowledge updates require **re-ingesting and re-embedding**; build delete/upsert paths. Switching embedding model means **re-embedding the whole corpus**.
- **Grounding prompt.** Instruct the model to answer *only* from the context and say "I don't know" otherwise — this is what actually converts retrieved evidence into reduced hallucination.

## Related

- Retrieval-Augmented Generation System — the full end-to-end production architecture (this note is the concept)
- [Long Context](/notes/ml-algorithms/language-models/long-context/) — the main alternative/complement; stuff vs retrieve
- [Supervised Fine-Tuning](/notes/ml-algorithms/language-models/supervised-fine-tuning/) — bake-knowledge-into-weights alternative; behavior vs facts
- [Prompt Engineering](/notes/ml-algorithms/language-models/prompt-engineering/) — how retrieved context is assembled and grounded in the prompt
- [Hallucination and Safety](/notes/ml-algorithms/language-models/hallucination-and-safety/) — the failure mode RAG is designed to reduce
- [kNN](/notes/ml-algorithms/supervised-learning/knn/) — nearest-neighbor search, the math under vector retrieval
- [Transformers](/notes/ml-algorithms/deep-learning/transformers/) · [LLM Evaluation](/notes/ml-algorithms/language-models/llm-evaluation/) · [Tool Use and Agents](/notes/ml-algorithms/language-models/tool-use-and-agents/)
