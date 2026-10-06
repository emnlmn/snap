# snap grep

`snap grep "<query>" [path]` finds the code that answers a question about a
repository: where database connections are pooled, what rebuilds the cache
after a failed decode, which function decides the prompt layout. It
returns ranked chunks, each a `path:start-end` range with the probability
that it answers the query and the source copied verbatim. Like the rest of
snap it generates no text: the model reads each candidate once and answers
one yes/no letter.

This page explains the design and the numbers behind it.

## Why retrieve-and-rerank, not a guided walk

[jevgrep](https://github.com/dzhng/jevgrep) walks the repository with a
hosted model: it judges folders, then files, then declarations, the way
Agentless-style localizers do. Each level waits for the one above it, so a
query costs several rounds of model calls in sequence.

Two results point to a cheaper shape. SweRank (ICLR 2026) shows that a
retriever followed by an LLM reranker beats agent-based localization on
SWE-bench Lite and LocBench at a fraction of the cost. Late-interaction
code retrievers point the same way: LightOn reports that ColGrep, its
local LateOn-Code index, let a coding agent find the right code with 56%
fewer search operations than plain grep.

snap grep is retrieve-and-rerank. It also moves the expensive half of the
rerank out of the query, as the next sections explain.

## The pipeline

1. **Corpus** (`corpus.rs`). ripgrep's own walker (the `ignore` crate)
   lists the files, honoring `.gitignore`, `.ignore`, hidden files and `-g`
   globs. A language-agnostic splitter cuts each file into
   declaration-sized chunks of at most 60 lines and 1,600 characters.
2. **Recall** (`lexical.rs`). BM25F scores chunks over code-aware terms:
   identifiers split at `_` and camelCase humps, paths and declared names
   weighted above bodies. A file-level ranking spreads its files' chunks
   into the candidate list, and the two rankings are fused by reciprocal
   rank. `-e` narrows the corpus with a ripgrep regex first. This stage
   needs no model and takes milliseconds.
3. **Rerank.** Every candidate gets one `boolean` probe with the
   `state_first` layout: the chunk is the state, and the query sits in the
   question. The probability of "yes" is the score.
4. **Snapshots** (`kvstore.rs`, `kv.rs`). The prompt is
   `[template head + chunk]` followed by `[question tail]`. Attention is
   causal, so the memory the model builds while reading the first part does
   not depend on the query. snap saves that memory once per chunk with
   `llama_state_seq_get_data` and stores it on disk. At query time it
   restores the memory and decodes only the tail. The logits are exactly
   those of decoding the whole prompt, with no approximation. Snapshots are
   saved and restored whole, so `kv.rs`'s one rule (sequences are copied and
   removed whole) still holds, and hybrid or recurrent memories take the
   same path.
5. **Cascade.** Candidates are scored in waves of up to 64 (one decode per
   wave), in recall order. The search stops early once enough hits are in
   and a wave adds none.

Prior work on the same idea: PreTTR (SIGIR 2020) precomputed
document-side term representations for BERT rerankers. HyperRAG (2025)
reuses document-side KV caches of decoder rerankers for a 2–3× throughput
gain. miniReranker (2026) combines cache reuse with early exit. Because
snap's prompt puts the document first and the query last, reuse is exact
here. Chunk-concatenation schemes such as CacheBlend need approximate
fixes; this one does not.

### What a snapshot is bound to

The store's key hashes the exact prefix tokens. A changed chunk, path,
chat template or prompt format is simply a different key. Line numbers
stay out of the prompt, so an edit above a chunk does not invalidate it.

The directory is bound to everything else that shapes the memory: the
snap version, the model file (identity, size and mtime), `PROMPT_VERSION`,
the grep prompt format and the KV cache type. A mismatch opens a
different store. It never feeds the model foreign memory.

## Cost model

The KV memory per token is `2 × n_layer × n_kv_head × head_dim × bytes`.
For snap1-2b (MiniCPM5-2B: 42 layers, 2 KV heads, head dim 128):

| KV type | per token | 460-token snapshot (400-token chunk + head) |
|---|---:|---:|
| f16 | 43.0 KB | 19.8 MB |
| q8_0 | 22.8 KB | 10.5 MB |

A query that reranks 64 candidates:

| | from scratch | from snapshots |
|---|---:|---:|
| tokens decoded | 64 × ~460 ≈ 29k | 64 × ~60 ≈ 3.8k |
| read from disk | 0 | 1.3 GB (f16) / 0.7 GB (q8_0) |

An NVMe disk reads 3–7 GB/s, and the prefetcher overlaps the reads of one
wave with the decode of the previous one. The compute drops about 7–8×,
and the disk time hides under it.

Disk use scales with the code that queries touch. A repository of 500k
code tokens is about 1,250 chunks, roughly 25 GB in f16 and 13 GB in
q8_0. By default the store fills lazily: a query snapshots the candidates
it decodes, so later queries in the same area start warm. `snap index`
fills it for the whole tree ahead of time.

## Where the next factor comes from

These items are ranked by expected gain per unit of work. Each one has to
earn its place on `eval/grep.jsonl` before it ships.

1. **A dense recall channel.** BM25 misses paraphrases that share no word
   with the code. A small code embedder (CodeRankEmbed, jina-code,
   Qwen3-Embedding-0.6B) or a late-interaction one (LateOn-Code) fused by
   RRF would recall better. With better recall, fewer candidates are
   needed, and the rerank cost drops linearly.
2. **A shorter tail.** The tail carries the question boilerplate (about 60
   tokens). A grep-specific format could move the fixed text into the
   prefix and leave only the query and the answer slot, about 25 tokens,
   with snap1 fine-tuned on that format through `training/`.
3. **Smaller snapshots.** A q8_0 or q4_0 KV cache halves or quarters the
   disk. Query-agnostic KV eviction (KVzip, NeurIPS 2025: 3–4× smaller with
   negligible loss on code comprehension) cuts further.
4. **Early exit.** Pointwise relevance saturates in intermediate layers:
   about 95% of the final quality at under 60% of the depth
   (miniReranker, 2026; E2Rank narrows candidates layer by layer). A
   truncated, fine-tuned snap1 grep model would save compute and snapshot
   size in the same proportion.
5. **Listwise final order.** A `choice` over the top ≤26 hits reads a full
   ranking from one logits row (FIRST, 2024), if pointwise probabilities
   prove too flat at the top.

## Measuring

`eval/grep.jsonl` holds queries over snap's own tree. Each one names the
lines that answer it as `{path, contains}` anchors, which stay valid as
line numbers move. `snap grep --eval eval/grep.jsonl --recall-only`
measures the first stage without a model. The full run reports hit@1,
recall@k, MRR and the latency of each stage.
