# AGENTS.md — snap

Technical directives for working on this repository.

What the project does, the API surface, and usage: see [README.md](README.md).
Each `src/*.rs` file carries a `//!` module header explaining its role.

## Build & verify

Everything goes through the Makefile (cargo resolves via rustup shim, not PATH):

```bash
make setup        # rustup + cmake/C++ compiler check (llama.cpp is vendored via llama-cpp-sys-2)
make build        # cargo build --release -> ./target/release/snap
make test         # unit tests + functional smoke (downloads ~1.5 GB GGUF)
make test-unit    # cargo test --release — pure logic, never loads a model
make lint         # cargo fmt --check + cargo clippy --release -- -D warnings
make check        # unit tests + cargo check --release
make serve        # build + run the HTTP server (MODEL=minicpm5-2b default)
```

- Before calling a change done: `make lint` and `cargo test` must pass.
- `make smoke` needs a real model download; run it when engine/prompt/schema
  behavior changes, not for pure refactors.
- CI mirrors these exactly: `cargo fmt --check`,
  `cargo clippy --locked --release -- -D warnings`, `cargo test --locked`
  (debug profile — tests must not require a model).
- GPU backends are opt-in features, additive and combinable: `cuda`, `vulkan`,
  `metal`, `rocm`, `opencl`, `openmp`, `mkl`, `dynamic-backends`,
  `static-stdcxx`. Default build: Metal on Apple Silicon, CPU elsewhere.

## Invariants — do not break

- **No generated text, ever.** The output surface is the letter alphabet
  (`A`–`Z`) + special slots. If a change needs prose output, the design is
  wrong.
- **`PROMPT_VERSION` (prompts.rs) bumps on any prompt-format change.**
  Calibration files bind to `(model_id, PROMPT_VERSION)` and refuse to load
  on mismatch — that's deliberate.
- **Budgets** (schema.rs / llamac.rs): `MAX_SLOTS=26` letters, `MAX_OPTIONS=256`
  (two-stage per-option probes past 26), `MAX_QUESTIONS=64`,
  `MAX_SEQS=65` (seq 0 owns the shared prefix).
- **Jev compatibility.** `POST /v1/systemone` is the single API surface —
  no parallel endpoints for the same job. Snap extensions are optional fields
  and defaults must preserve Jev semantics.
- **Tested-models only.** `MODELS` in models.rs is a curated set; a new GGUF
  gets a row only after `snap evaluate` on `eval/*.jsonl` proves it.
- **`llama-cpp-sys-2` is pinned (`=0.1.157`)** — deliberate, don't float it.
- **`hf-hub` stays `ureq`-only.** native-tls/openssl can't build on musl;
  the static release target depends on this.
- **Portability matters.** Release matrix: macOS arm64, Linux gnu
  (dynamic-backends + x86-64-v3), aarch64, musl (fully static), vulkan,
  windows. Avoid platform-specific assumptions (e.g. `char` signedness,
  `tar` flags).
- **Reports are create-only** — eval/bench/calibrate outputs refuse to
  overwrite existing files. Keep it that way.

## Code style

- anyhow for errors, clap derive for CLI, `serde_json` with `preserve_order`,
  `tracing` to stderr (llama.cpp logs routed through it — `--debug` enables).
- Module docs (`//!`) explain the *why*, matching existing density — sparse,
  pointed, no comment noise.
- Compact code, no defensive nesting; error handling at real boundaries.
- Engine changes that touch seq management / KV rewind are the riskiest
  area — read `llamac.rs` memory ops before modifying.

## Git

- Commit messages: lowercase, short area prefix (`ci:`, `deps:`, `serve:`),
  concise, "why" over "what" — see `git log`.
- **Never add co-author trailers or agent attribution.** No `Co-Authored-By`,
  no "Generated with …" lines — commits are authored by the repo owner only.
- Don't push unless asked.

## Housekeeping

- `src/web/` assets are served via `include_str!` — they're compiled into the
  binary; no runtime file serving.
- `snap serve` registers a pidfile (`$SNAP_RUN_DIR` > `$XDG_RUNTIME_DIR/snap`
  > `~/.snap/run`) — `snap ps`/`snap stop` read it. Kill paths live in
  `instances.rs`; never signal pid 0 (unix: whole process group).
- `.serena/` and `.impeccable/` are local tooling dirs (gitignored).
- `openapi.yaml` documents the HTTP surface — keep it in sync with server.rs.
