# snap grep

`snap grep "<query>" [path]` finds the code that answers a question about a
repository: where database connections are pooled, what rebuilds the cache
after a failed decode, which function decides the prompt layout. It
returns ranked chunks, each a `path:start-end` range with the probability
that it answers the query and the source copied verbatim. Like the rest of
snap it generates no text: the model answers one yes/no letter, on a
preview while it descends the tree and on each chunk it reranks.

This page explains the design and the numbers behind it. Status: the
hybrid described here (BM25 recall and a model-judged descent over scrubbed
previews, reranked together) has been run on the 40 cases of
`eval/grep.jsonl`; the table is under [Measuring](#measuring), next to the
BM25 pipeline of 0.6.0 it is compared with.

## Why both

0.6.0 recalled candidates with BM25 and had the model rerank them, the
shape SweRank (ICLR 2026) found cheaper and better than agent-based
localization. Its ceiling was recall: 0.62 of the 16 paraphrases of
`eval/grep.jsonl` were inside the 64 candidates, because BM25 cannot reach
code that shares no word with the question, and the model reorders what it
is given.

[jevgrep](https://github.com/dzhng/jevgrep) walks the repository with a
hosted model instead: it judges folders, then files, then declarations,
the way Agentless-style localizers do. snap takes that walk and changes
two things. The judgment is the same yes/no probe on every level, and a
whole level goes to the model as one batch rather than one call per
folder. And what the model reads first, the previews, does not depend on
the query, so its memory is snapshotted on disk like a chunk's.

The walk alone is not enough either. Run by itself, it reaches what BM25
never does and loses what BM25 finds, and the two fail on different cases.
The paraphrases BM25 cannot recall
need the descent. Lexical questions, and questions answered by docs, where
BM25 had 0.86 at rank 1, need the keywords: a preview is a summary, and
the model can prefer a manifest or a build script to the document that
answers. So both run on every query and the model reranks the union.

There is no cascade that runs the descent only when no BM25 candidate is
confident. It was considered and rejected, because the model is
overconfident. On the 16 paraphrases of `eval/grep.jsonl`, 0.6.0 declared a
confident hit in 62% of them and was right at rank 1 in 25%. A confidence
trigger would skip the descent exactly where it is needed.

## The pipeline

1. **Corpus** (`grep/corpus.rs`). ripgrep's own walker (the `ignore` crate)
   lists the files, honoring `.gitignore`, `.ignore` and `-g` globs.
   Dotfiles are read: nothing leaves the machine, and `.github/` is code
   too. What the ignore files hide is build output and dependencies, about
   100 times the chunks of snap's own code, so they stay honored; an
   ignored directory named as the path is still searched. A
   language-agnostic splitter cuts each file into
   declaration-sized chunks of at most 60 lines and 1,600 characters.
2. **Recall** (`grep/lexical.rs`). BM25F scores chunks over code-aware terms:
   identifiers split at `_` and camelCase humps, paths and declared names
   weighted above bodies. A file-level ranking spreads its files' chunks
   into the candidate list, and the two rankings are fused by reciprocal
   rank. Docs, specs, mockups and data files describe a feature in the
   very words a question uses and outrank the code behind it (over a
   whole repository with its design docs they took 60 of 64 candidates),
   so code and the rest take turns in the list while both last. The list
   holds `--candidates` chunks (64 by default). `-e` narrows the corpus
   with a ripgrep regex first. This stage needs no model and takes
   milliseconds (2.8 s over 2 million lines); `--recall-only` stops here.
3. **Descent** (`grep/tree.rs`). Folders, files and groups of chunks become
   previews of at most 1,200 characters. A preview is not a symbol list: it
   is the code scrubbed to plain text, deterministically (`scrub` extracts,
   it never generates). It keeps comment text without its markers (`//`,
   `///`, `//!`, `/* */`, `#` in script and config files, Python
   docstrings), string contents and identifiers split into lowercase words
   at `_`, `::`, `.` and camelCase humps, with no stemming; keywords,
   punctuation, operators and Rust attributes are dropped. A file preview
   leads with the file's leading comment block (module docs, at most 6
   lines), then one outline line per chunk, `<symbol words>: <its first
   comment line>`, then the rest of the scrubbed text while it fits. A
   folder preview lists each child file as `<name words>: <first line of
   its module docs>`. A group preview is the same outline followed by the
   scrubbed bodies. For `src/server.rs` the file preview starts:

   ```
   axum surface the single post v1 systemone api jev wire snap extras
   app state: the engine survives a panicking request the lock is taken back
   switch model: post v1 models model spec load and hot swap the resident
   shutdown signal: sigterm snap stop or sigint drain in flight requests then exit
   ```

   The model answers the same yes/no probe on every preview of a level in
   one batched call. Per level the descent opens the `--beam` best folders
   (3 by default), the best files that are not terminal and the best groups
   that are not terminal; the unopened branches are dropped. A file is
   never reached whole: its chunks are split into 4 contiguous groups,
   recursively, until a group holds at most 4 chunks, and such a group is
   reached with its probability. It always runs, whatever recall found.
   The chunks it reaches that recall did not pick are the tree's picks, at
   most 24. Only the descent reads scrubbed text; the rerank of step 4
   still reads the real chunk, since the code is the evidence there (a
   literal such as `"0.0.0.0"` says what no comment does).
4. **Union and rerank.** The candidates are BM25's list plus the tree's
   picks, interleaved (BM25, tree, BM25, tree, and so on) so the first
   wave holds both: the early stop of step 6 must not leave the tree's
   picks unread behind 64 BM25 candidates. Every candidate gets one
   `boolean` probe with the `state_first` layout: the chunk is the state,
   and the query sits in the question. The probability of "yes" is the
   score.
5. **Snapshots** (`grep/kvstore.rs`, `kv.rs`). The prompt of a chunk is
   `[template head + chunk]` followed by `[question tail]`. Attention is
   causal, so the memory the model builds while reading the first part does
   not depend on the query. snap saves that memory once per chunk with
   `llama_state_seq_get_data` and stores it on disk. At query time it
   restores the memory and decodes only the tail. Nothing is approximated:
   the logits are those of decoding the whole prompt, up to the rounding
   noise any change of batch composition brings (at most about 4e-3 on
   logits of scale 2, measured under llama.cpp with tiny random-weight
   llama, gemma2, mamba and hybrid models). Snapshots are saved and restored
   whole, so `kv.rs`'s one rule (sequences are copied and removed whole)
   still holds, and hybrid or recurrent memories take the same path. The
   previews of the descent are query-independent too, so their snapshots
   are stored and restored the same way.
6. **Waves.** Candidates are scored in the order of the union, in waves
   of as many as one restore can hold: a restored snapshot takes a sequence
   and private cells for its whole prompt, so a wave is up to 64
   candidates whose prompts fit the context together (17 snap1-2b probes
   at the default `--ctx 8192`, 64 at `--ctx 32768`). At most about two
   waves of snapshots sit in memory, the one being scored and the one read
   ahead: 0.7 GB in f16 at the default context, about half in the default q8_0. The search stops early once
   enough hits are in and a wave adds none; `--eval` always scores every
   candidate.

Prior work on the same idea: PreTTR (SIGIR 2020) precomputed
document-side term representations for BERT rerankers. HyperRAG (2025)
reuses document-side KV caches of decoder rerankers for a 2–3× throughput
gain. miniReranker (2026) combines cache reuse with early exit. Because
snap's prompt puts the document first and the query last, the reused
memory is the one a full decode would build. Chunk-concatenation schemes
such as CacheBlend reuse memory computed without its neighbors and need
approximate fixes; this one does not.

### What a snapshot is bound to

The store's key hashes the exact prefix tokens. A changed chunk, path,
chat template or prompt format is simply a different key. Line numbers
stay out of the prompt, so an edit above a chunk does not invalidate it.

The directory is bound to everything else that shapes the memory: the
snap version, the model file (identity, size and mtime), `PROMPT_VERSION`,
the grep prompt format and the KV cache type. A mismatch opens a
different store. It never feeds the model foreign memory.

## Cost model

The figures below were measured on the 0.6.0 pipeline, with 64 candidates
and snap1-2b; they describe the rerank and the snapshot store, which the
hybrid keeps. The only timing of the hybrid is a median of 21.6 s and a p90
of 28.6 s per query on an M1 Max with a partly warm store; how it splits
between preview calls and the rerank is not measured.

`snap grep` keeps its snapshots in q8_0 by default, which halves the disk
for hits and probabilities that moved by at most 4e-3 on a test query;
`--kv f16` is full precision. The table below is measured per type. The KV
memory per token is `2 × n_layer × n_kv_head × head_dim × bytes`.
For snap1-2b (MiniCPM5-2B: 42 layers, 2 KV heads, head dim 128) that is
43,008 bytes in f16. llama.cpp's snapshots match it: with a random-weight
model of the same KV geometry, a snapshot measures 43,020 bytes per token
(the extra 12 are cell metadata) plus 1,032 bytes per blob.

| KV type | per token | 460-token snapshot (400-token chunk + head) | 64 candidates |
|---|---:|---:|---:|
| f16 | 43.0 KB | 19.8 MB | 1.27 GB |
| q8_0 | 22.9 KB | 10.5 MB | 0.67 GB |
| q4_0 | 12.1 KB | 5.6 MB | 0.36 GB |

A query that reranks 64 candidates decodes 64 × ~60 ≈ 3.8k tail tokens
instead of 64 × ~460 ≈ 29k from scratch, 7.7× fewer.

What the same model measured on CPU (4 vCPU, a shared virtual disk), 16
sequences of 460 tokens:

- **Restore + tail against a full decode: 4.4–5.1× faster.** The tails
  attend over the whole 400-token prefix, and the toy model's tiny weights
  make attention half of its per-token cost. On a 2B model, where
  attention over 460 tokens is a small share of the work, the estimate is
  about 7×.
- **The copies are real work.** `seq_load` of a 460-token f16 snapshot
  takes 6.4 ms (3 GB/s), 0.4 s for 64 candidates on the decode thread;
  `seq_save` takes 10.8 ms, 41–46% of it spent allocating the buffer.
  q8_0 halves both, but on CPU it decodes 1.56× slower than f16.
- **The disk hides behind the decode** when the prefetch reads a whole
  wave ahead: only the first wave's read shows, about 6% of the query.
  The store read a cold pack at 1.1–1.5 GB/s (the raw disk does
  1.5–2 GB/s). The overlap holds as long as decoding a wave's tails takes
  longer than reading its snapshots: at 1.2 GB/s, up to about 3.5k tail
  tokens per second in f16 (CPU, small GPUs). A fast GPU needs a faster
  disk (3 GB/s for about 9k tokens/s, 7 GB/s for about 21k), or smaller
  snapshots.

Disk use scales with the code that queries touch. A repository of 500k
code tokens is about 1,250 chunks, roughly 13 GB in q8_0 (the default) and
25 GB in f16. The store fills lazily and only that way: a query snapshots the
candidates it decodes, so later queries in the same area start warm. A full
pre-fill is not offered, since a 2M-line tree is about 52k chunks, hundreds of
GB. `snap grep --cache [path]` reports what the store holds of a tree, and
`--gc [path]` drops the snapshots no chunk uses any more together with the
stores of other engines (another snap version, model, prompt format or KV
type), which nothing would open again.

## Where the next factor comes from

These items are ranked by expected gain per unit of work. Each one has to
earn its place on `eval/grep.jsonl` before it ships.

1. **A preview-specific question.** The probe asks whether a chunk answers
   the question; on a folder or a file the better wording is whether the
   answer is likely to be inside. A wording written for previews changes
   only the tail, so no snapshot is invalidated. It has not been tried.
2. **Neighbours of the top hits.** On the traced failures the right file
   was chosen and the wrong chunk of it won (prompts-02, calibrate-02).
   Reading the chunks next to the top hits is an idea to measure.
3. **A chunk header in the rerank.** The first line of the file's module
   docs, in front of each chunk, would tell the rerank what the file is for
   as the previews already do for the descent. It changes how a chunk
   renders into the STATE block, so `GREP_FORMAT` bumps and every snapshot
   is made again.
4. **The flat scores and the beam.** A wider beam where the scores are
   flat, a threshold relative to the best score instead of a fixed count,
   or keeping the best pruned branch as a fallback when the reached chunks
   all score low. A pruned branch never comes back today.
5. **A shorter tail.** The tail carries the question boilerplate (about 60
   tokens). A grep-specific format could move the fixed text into the
   prefix and leave only the query and the answer slot, about 25 tokens,
   with snap1 fine-tuned on that format through `training/`. It would also
   shorten every call of the descent.
6. **Smaller snapshots, fewer copies.** q8_0 already halves the disk and is
   the default; q4_0 would quarter it (28% of f16). Query-agnostic KV eviction
   (KVzip, NeurIPS 2025: 3–4× smaller with negligible loss on code
   comprehension) cuts further. Reused buffers take `seq_save` from 10.8
   to 6.4 ms, and memory-mapping the pack would let llama.cpp copy
   snapshots straight from the page cache.
7. **Early exit.** Pointwise relevance saturates in intermediate layers:
   about 95% of the final quality at under 60% of the depth
   (miniReranker, 2026; E2Rank narrows candidates layer by layer). A
   truncated, fine-tuned snap1 grep model would save compute and snapshot
   size in the same proportion.
8. **Listwise final order.** A `choice` among siblings, or over the top ≤26
   hits, reads a full ranking from one logits row (FIRST, 2024), if
   pointwise probabilities prove too flat at the top.
9. **A larger eval set on other trees.** Forty cases on snap's own tree
   cannot separate changes of a few points, and snap's comments may be
   unusually informative for scrubbed previews. The next result worth
   trusting needs more cases on code written by others.

## Measuring

`eval/grep.jsonl` holds 40 queries over snap's own tree: 14 that share
words with the code that answers them, 16 paraphrases that share none, and
10 whose answer spans several places. Each one names the lines that answer
it as `{path, contains}` anchors, which stay valid as line numbers move.
`snap grep --eval eval/grep.jsonl` reports hit@1, recall@5 and @10 and MRR
over the model's order, and the latency of each stage. `--recall-only`
measures BM25 alone, with no model.

Seven hard cases were run on a clean copy of the tree, with the eval file
excluded from the corpus: the five paraphrases BM25 never recalls
(engine-02, prompts-02, server-02, calibrate-02, instances-02), plus
readme-01 and calibration-01. hit@1 and MRR, for two models and two
architectures:

| | snap1-2b | qwen3.8-4b |
|---|---:|---:|
| 0.6.0 (BM25 only) | 2/7, MRR 0.31 | 1/7, MRR 0.17 |
| hybrid + scrubbed previews | 3/7, MRR 0.50 | 2/7, MRR 0.37 |

With snap1-2b the hybrid put server-02 first, which the descent reached for
the first time, and kept readme-01 and calibration-01. instances-02 got
into the candidates (8th) for the first time. It lost engine-02, which
qwen3.8-4b won by choosing `engine.rs`, then the half, then the group with
`fn warmup`. prompts-02 and calibrate-02 were lost with the right file
chosen and the wrong chunk.

The gain comes from the previews, not from a bigger model. Descent only,
and the hybrid with symbol-list previews, did not beat 0.6.0: file previews
scored nearly flat around 0.5 and the beam pruned the right file (on
server-02, `server.rs` scored 0.31 among files at about 0.33). A preview
carries meaning in its comments: `warmup`'s doc says "before the first real
request", and `fold_of`'s says "stable fold assignment independent of case
order", words close to the questions that find them. qwen3.8-4b reranks
worse than snap1-2b (it lost readme-01 to `Cargo.toml`), since snap1-2b is
trained on snap's format, which is why `snap grep` keeps snap's default
model.

The whole suite, 40 cases, snap1-2b, rerank over the candidates. 0.6.0
against the hybrid with scrubbed previews:

| kind | hit@1 | @5 | @10 | MRR |
|---|---:|---:|---:|---:|
| lexical (14) | 0.79 → 0.79 | 0.93 → 0.93 | 0.93 → 0.93 | 0.86 → 0.86 |
| concept (10) | 0.80 → 0.80 | 0.73 → 0.67 | 0.83 → 0.83 | 0.86 → 0.88 |
| paraphrase (16) | 0.25 → 0.31 | 0.50 → 0.69 | 0.56 → 0.69 | 0.35 → 0.47 |
| overall (40) | 0.57 → 0.60 | 0.71 → 0.77 | 0.76 → 0.81 | 0.66 → 0.71 |

The share of queries whose best hit clears the threshold (`answered`)
went from 0.85 to 0.88. The gain is
in the paraphrases, which is where the descent was meant to help; lexical
questions did not move. A query of the hybrid took a median of 21.6 s and a
p90 of 28.6 s on an M1 Max with a partly warm store; the timing of the
0.6.0 run is not comparable, since it shared the GPU with another eval.

These are 40 cases on one tree, and one case is 2.5 points: the dip of
concept recall@5 is within that. Known risk: a branch the descent pruned
never comes back, and the union only helps if the tree's picks, at most 24,
survive the rerank.

## History

0.6.0 shipped BM25 recall and the model rerank on top of it, with no
descent. The recall (`grep/lexical.rs`) scored chunks with BM25F over
code-aware terms: identifiers split at `_` and camelCase humps, paths and declared
names weighted above bodies. A file-level ranking spread its files' chunks
into the candidate list, and the two rankings were fused by reciprocal
rank. Docs, specs, mockups and data files describe a feature in the very
words a question uses and outranked the code behind them (over a whole
repository with its design docs they took 60 of 64 candidates), so code
and the rest took turns in the list while both lasted. It needed no model
and took 2.8 s over 2 million lines.

Its anchor recall over the 64 candidates, on `eval/grep.jsonl`:

| kind | @1 | @5 | @10 | @64 |
|---|---:|---:|---:|---:|
| lexical (14) | 0.86 | 1.00 | 1.00 | 1.00 |
| concept (10) | 0.15 | 0.45 | 0.48 | 0.97 |
| paraphrase (16) | 0.00 | 0.00 | 0.00 | 0.62 |
| overall (40) | 0.34 | 0.46 | 0.47 | 0.84 |

Recall@64 was the ceiling the model could reach: it reordered the 64
candidates, so the paraphrases recall missed (38% of them) stayed missed.
That is what the descent is meant to fix, and it is why the descent now
runs beside recall. A per-file cap on the candidates
was tried and measured worse (0.73 at 4 a file): answers live in long
files, and a cap pushes out the chunk that answers.

The model's ranking on top of it, measured on snap1-2b with f16 snapshots
before code and docs took turns: hit@1 0.53 (lexical alone 0.31),
recall@5 0.72, @10 0.77, MRR 0.64. It lifted concept queries most (hit@1
0.15 to 0.70); paraphrases stayed bound by what recall handed it.
