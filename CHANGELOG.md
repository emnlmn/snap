# Changelog

## Unreleased

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
