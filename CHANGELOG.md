# Changelog

## Unreleased

- grep: a model-judged tree descent now runs beside the BM25 recall of
  0.6.0, on every query. Folders, files and groups of chunks become
  previews (at most 1,200 characters), the model answers the same yes/no
  probe on each, one batched call per level, and per level the `--beam`
  best are opened (default 3). A file is split into 4 contiguous groups,
  recursively, until groups of at most 4 chunks; unopened branches are
  dropped. The model reranks the union: BM25's 64 candidates plus at most
  24 chunks the descent reached that BM25 did not pick, interleaved so the
  first wave holds both. Why: BM25 cannot reach a paraphrase that shares no
  word with the code (recall@64 0.62 on the 16 paraphrases of
  eval/grep.jsonl)
- grep: the descent reads scrubbed previews, not symbol lists. `scrub` in
  `grep/tree.rs` flattens code to plain text without generating any: comment
  text without markers, string contents and identifiers split into
  lowercase words; keywords, punctuation and attributes are dropped. A file
  preview leads with its module docs, then one line per chunk (symbol words
  and the chunk's first comment line); a folder lists each child file with
  the first line of its docs. Why: with symbol lists the model scored file
  previews nearly flat and the beam pruned the right file. The final rerank
  still reads the real chunk
- grep: no cascade. Running the descent only when no BM25 candidate is
  confident was rejected: on those 16 paraphrases 0.6.0 declared a
  confident hit in 62% and was right at rank 1 in 25%
- grep: previews do not depend on the question, so their snapshots are
  cached like the chunks'
- grep: `--beam N` added. `--recall-only` stays (BM25 only, no model); there
  is no `--tree`, the descent is always on. The default model is unchanged
  (snap1-2b): qwen3.8-4b reranks worse, as snap1-2b is trained on snap's
  format
- grep: measured on the 40 cases of eval/grep.jsonl with snap1-2b, 0.6.0
  against the hybrid: overall hit@1 0.57 to 0.60, recall@5 0.71 to 0.77,
  @10 0.76 to 0.81, MRR 0.66 to 0.71; paraphrases (16) hit@1 0.25 to 0.31,
  @5 0.50 to 0.69, MRR 0.35 to 0.47; lexical unchanged; concept recall@5
  0.73 to 0.67. On 7 hard cases hit@1 went from 2/7 to 3/7. 40 cases on one
  tree: one case is 2.5 points. A hybrid query took a median of 21.6 s
  (p90 28.6 s) on an M1 Max with a partly warm store
- docs: GREP.md describes both stages, the scrubbed previews and the
  union; the 0.6.0 numbers stay there as history

## 0.6.0 - 2026-10-07

- grep: `snap grep "<question>" [path]` finds the code that answers a
  question about a tree, still without generating text. A lexical stage
  with no model picks 64 candidates; the model reads each once and answers
  one yes/no letter, whose probability is the hit's score. On snap's own
  tree (eval/grep.jsonl, snap1-2b) the model takes hit@1 from 0.31 to 0.53,
  MRR 0.64, concept queries from 0.15 to 0.70 (GREP.md)
- grep: the reading costs once per chunk, not once per query — the memory
  of `[prompt head + chunk]` does not depend on the question, so it is
  snapshotted to disk and a later query restores it and decodes only its
  question: 64 candidates in 4.3 s warm against 23.5 s cold on an M1 Max,
  with the same probabilities
- grep: recall is ripgrep's walker (ignore files honored, dotfiles read:
  nothing leaves the machine) and BM25F over code-aware terms —
  identifiers split at `_` and camelCase, Porter stemming, file and chunk
  rankings fused; code and docs take turns in the candidate list, since
  over a whole repository specs and mockups otherwise crowd out the code.
  2.8 s over two million lines
- grep: output for people — one line per hit (a tinted bullet, path ·
  symbol · percent), the source on a rail where the query's words fall,
  `⋮` for what is left out, progress while reading, a plain footer; under
  three hits past `--threshold`, the best others down to 20% fill in with a
  hollow bullet; exit status 1 when nothing is shown, as grep. `--json` and
  `-l` for tools, `--color`/`NO_COLOR`
- grep: the cache lives in `$SNAP_GREP_DIR`, else `$XDG_CACHE_HOME/snap/grep`,
  else `~/.cache/snap/grep`, one directory per tree and engine, bound to
  the snap version, the GGUF, the prompt format and the KV type — another
  binding opens another store, never foreign memory. q8_0 KV by default,
  ~9 MB a chunk (`--kv f16` doubles it); `--cache` reports what it holds of
  a tree, `--gc` drops what no chunk uses and the stores of other engines
- grep: known limits — the cache has no size cap yet (a cold query in a new
  area writes ~0.6 GB); queries match the code's own words, so a question
  in another language than the code, or a feature's old name after a
  rename, can miss the code and find only the docs that mention it
- kv: whole-seq snapshots — `Backend::seq_save`/`seq_load`, `Kv::snapshot`
  and `Kv::run_restored`, checked on the sim against from-scratch rows and
  on tiny llama, gemma2, mamba and hybrid GGUFs under llama.cpp; seqs are
  still copied and removed whole
- llamac: `KvType` (`--kv f16|q8_0` on grep), threaded through
  `Engine::load_kv`; every other command stays on f16, and `snap evaluate`
  answers are identical to 0.5.0
- eval: `eval/grep.jsonl`, 40 cases over snap's own tree (lexical,
  paraphrase, concept), anchored by code text instead of line numbers;
  `snap grep --eval` scores recall and the model's ranking
- tests: a flaky instances test no longer spawns a child (its fork held
  another test's file lock); clippy 1.98 clean

## 0.5.0 - 2026-10-02

- prompts: noul `{true, false}` criteria render as `Yes:`/`No:` outcome
  lines in the question block and `null` descriptions fall back to the
  option key or level index — PROMPT_VERSION 6; calibrations bound to v5
  must be refit, ordinary requests are byte-identical
- api: the TypeSafe JS SDK runs unchanged against `snap serve` — question
  `instructions` take the full `EntryType` (text, JSON, or null), `null`
  criterion descriptions render the option key or level index instead of the
  word "null", noul `{true, false}` criteria reach the prompt as outcome
  descriptions (the slots stay yes/no, so `noul` keeps meaning P(yes)),
  score answers carry the Jev shape (`score` is the expected level index in
  0..n−1, `legend` maps indexes to rubric text, `probabilities` are keyed by
  index), `usage.output_tokens` is 0, `GET /v1/models` answers the SDK's
  `{models: [...]}` shape, and responses carry `x-typesafe-request-id`
- eval: typed-decisions on v6 — 0.655 accuracy against 0.643 on 0.4.0, all
  from noul (+4 points, the outcome lines); choice and score unchanged,
  ECE 0.041 raw, M1 Max
- instances: windows build fixed — `cmdline_is_serve` is unix+test only,
  tasklist has no argv
- models: leftover minicpm5-2b defaults move to snap1-2b
- site/readme: snap1-2b model section + Performance nav, terminal demo
  gif, score example shows the Jev legend shape

## 0.4.0 - 2026-10-01

- prompts: object/array states always render TOON (spec v4.1, encode-only) — `[N]` lengths and `{fields}`/`[N:]` tabular headers; PROMPT_VERSION 5, calibrations bound to v4 must be refit
- api: `compact_state` is gone — the yaml-lite renderer it selected is deleted; the tolerant envelope still accepts the flag and ignores it, so Jev clients are unaffected
- engine: `SNAP_STATE_FORMAT=json` and `snap evaluate --state-format json|toon` keep a debug escape for format bisection; eval reports label the forced renderer
- eval: three-renderer benchmark, 108 structured-state cases × 4 models — yaml-lite and TOON answered identically on every case, JSON cost ~1% more tokens; the standard format wins
- cli: `snap models` is a docker-style group — the list shows size on disk + real cache path, `pull`/`rm` manage the shared HF cache; resolve+load+calibrate fold into one `ModelArgs::engine`

## 0.3.0 - 2026-09-28

- models: snap1-2b, MiniCPM5-2B fine-tuned for the letter readout, joins the tested set as the default — typed-decisions 0.624 zero-shot (base 0.502), eval suite 85.0% (base 69.1%), same speed; q4_k_m, q8_0 and bf16 from `logitlab/snap1-2b-GGUF`
- brand: the snapping-jaw mark replaces the crocodile — README, console header and favicon, site; svg + png in `assets/`
- engine: model ids get the `snap-` prefix once — a GGUF named `snap1 2B` is `snap1-2b`, not `snap-snap1-2b`; every other model keeps its id and its calibrations
- training: the fine-tuning pipeline lands in `training/` — data preparation, teacher review flags, prompt export, LoRA, GGUF merge and paired evaluation, documented in TRAINING.md; code only, data and runs stay local
- license: MIT `LICENSE` file at the root — Cargo.toml already declared MIT, the text was missing
- cli: `export-prompts` renders through `decide`'s own compile step and honors `--layout` plus the case `layout`/`expand`/`compact_state` pins — header preamble and catalog included, byte-exact with what evaluate decodes; choices past 26 letters are skipped instead of panicking; ids must be unique; `--output` is claimed before the model loads and removed on a failed run
- eval: malformed `layout`/`expand`/`compact_state` case pins are errors, not silent defaults
- api: `noul` answers expose the full `probabilities` map — the abstain mass the scalar folded away
- engine: byte-exact training export surface; `state_tokens` lands behind `/playground/tokenize`
- cli: loading folds into the ready line — one dense line, `snap:` prefix on all stderr
- eval: `core`+`edge` folded into `eval/cases.jsonl` (301 cases, tty progress line); `vs_ollama`/`vs_openai` comparisons — json under a schema, unseen states, one letter as the floor; `typed_decisions.py` — the LocalLLaMA/typed-decisions card against a running server
- playground: dark/light theme switch (system-aware, self-hosted faces); live token count in the state head; run settings (mode + temperature) moved into the request head; histogram bars read value + probability on hover; missile reads its last frame through the console's dial
- serve: `/` redirects to `/playground`; theme + fonts served from the binary
- deploy: `snap serve` demo on Modal — `deploy/Dockerfile` image, models cached on a volume
- site: hero console fans one state out to every answer; scroll-driven race against a chosen rival; use-case state as fields; accuracy table on the merged `cases.jsonl`
- readme: rewrite — what it's for, how it works, the honest numbers; demo re-recorded on minicpm5-2b
- docs: agent writing-style rules

## 0.2.1 — 2026-09-25

- release: guard the tag against the crate version; bash shell for the version guard — windows defaults to pwsh

## 0.2.0 — 2026-09-25

- engine: cross-request state-prefix cache + in-wave sub-prefix clustering; `question_first` layout caches qheads across requests; fused decode call, qhead-safe scratch seqs, `--cpu-threads`; fork-and-discard batching on hybrid archs; request text tokenized as plain text — no control-token injection
- prompts: catalog layout, paged choice expansion, compact state
- api: per-answer `coverage` — how much raw next-token mass landed on the allowed letters
- serve: `snap ps`/`snap stop` over a pidfile registry
- playground: response pane as an instrument — receipt stats, ruled bars, run-over-run deltas
- eval: typesafe public eval
- site: github pages landing at `docs/`; live recorded demo + one-line install; readme badges + binary install notes
- refactor: engine and kv consolidation

## 0.1.0 — 2026-09-23

- first release: `snap serve` with `POST /v1/systemone`, drop-in Jev compatible
- decisions from one forward pass — letter-alphabet prompts, probabilities over `choice`, `noul`, `score`, `numeric`, abstain and out-of-range anchors; up to 256 options via per-option probes
- `snap bench`, `snap evaluate`, `snap calibrate`; reports are create-only
- model hot-swap on `POST /v1/models`; playground console + missile command demo, light theme
- engine: prefix-row reuse; ci matrix synced — macOS, Linux, Windows release targets
