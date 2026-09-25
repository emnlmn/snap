<p align="center">
  <img src="assets/logo.png" width="180" alt="SNAP logo"/>
</p>

<h1 align="center">SNAP</h1>

<p align="center">
  <b>Single-pass Neural Answer Probabilities</b><br/>
  Typed decisions from unstructured state — one forward pass, zero generated text.<br/>
  <sub>local · deterministic · drop-in Jev compatible</sub>
</p>

<p align="center">
  <a href="https://github.com/emnlmn/snap/actions/workflows/ci.yml"><img src="https://github.com/emnlmn/snap/actions/workflows/ci.yml/badge.svg" alt="CI"/></a>
  <a href="https://github.com/emnlmn/snap/releases/latest"><img src="https://img.shields.io/github/v/release/emnlmn/snap" alt="release"/></a>
  <img src="https://img.shields.io/badge/platform-macOS%20%C2%B7%20Linux-lightgrey" alt="platforms: macOS · Linux"/>
  <a href="https://opensource.org/license/mit"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="license: MIT"/></a>
</p>

---

Your LLM pipeline doesn't need to *write* anything. It needs to *decide*.

Yet somewhere in your stack there's a prompt asking for JSON, a parser that
mostly works, a retry loop for when it doesn't — and a token bill for
thousands of generated words you immediately throw away. All that machinery
exists for one reason: the only way you can interrogate a chat model is to
let it talk.

SNAP never lets it talk.

Every question compiles to a prompt whose first answer token must be a
letter — `A` through `Z`, one per option. SNAP reads the logits at that
single position, pools the letter variants, and softmaxes them into a
distribution. The jaw snaps shut on the answer. One `llama_decode` per
question. No prose, no parsing, no retries, no hallucinated JSON keys.

```jsonc
// in
{
  "state": "ticket, CRM record, sensor dump — anything",
  "questions": {
    "route":  {"type": "choice",  "criteria": {"billing": "…", "tech": "…"}},
    "urgent": {"type": "boolean", "instructions": "SLA breach likely?"},
    "impact": {"type": "score",   "criteria": ["low", "mid", "high"]}
  }
}

// out — one distribution per question
{
  "answers": {
    "route":  {"choice": "billing", "probabilities": {"billing": 0.87, "tech": 0.13}},
    "urgent": {"boolean": true,  "confidence": 0.94},
    "impact": {"level": 2, "score": 0.81}
  }
}
```

## Why it bites

- **Decisions, not documents.** The output surface is a fixed alphabet of
  letters. Structured by construction — there is nothing downstream to
  parse, validate, or repair.
- **The prompt is the API.** `noul`, `choice`, `score`, `numeric`,
  abstention (`__abstain__`), out-of-range anchors — typed contracts
  with explicit uncertainty instead of a confident hallucination.
- **Probabilities you can threshold.** `snap calibrate` fits a
  temperature per question type on your labeled cases — bound to the
  model and prompt version, with ECE printed before and after.
- **Choices past the alphabet.** More than 26 options? Each candidate
  gets its own yes/no probe, batched over the same shared prefix, and
  the scores merge into one distribution. Up to 256.
- **State amortized.** Every prompt of a request decodes as one
  shared-prefix trie in **a single batched call**: the state — and any
  text several questions share — is evaluated once, on every architecture,
  hybrids included. The constant prompt head stays resident between
  requests.
- **Questions amortized too.** `layout: question_first` flips the prompt —
  the question head is identical across requests and stays cached on its
  own KV sequence (LRU-bounded). Repeat
  workloads — the same triage questions on a stream of tickets — decode
  only the state.
- **Honest internals.** `x_snap` reports exactly what happened:
  `prompt_tokens` against the tokens actually decoded
  (`usage.input_tokens`), `cached_head_tokens`, `shared_prefix_tokens`,
  `cache_hits/misses`, `waves`, decode/total ms. Every answer also carries
  `coverage` — how much of the model's raw next-token mass landed on the
  allowed letters at all.

## The API — drop-in Jev, extended

`POST /v1/systemone` speaks the TypeSafe/Jev wire format. Point your
existing client at a local `snap serve` and it works — `noul`, `choice`,
`score`, same response shape. One surface, no parallel API.

The same request accepts optional snap extensions; Jev clients can ignore
them and defaults preserve Jev semantics:

| field | extension |
|---|---|
| question `"type": "numeric"` | `{min, max, granularity}` — distribution over a numeric range |
| question `allow_abstain` | default `false`; `true` adds an `__abstain__` slot (status `abstained`) |
| request `mode` | `shared` (default: shared text decoded once, spans cached across requests) or `direct` (every question decoded alone — the reference path) |
| request `layout` | `auto` (default), `state_first`, `question_first`, `header`, `catalog` |
| request `expand` | `probes` (default) or `pages` — how >26-option choices expand |
| request `compact_state` | default `false`; `true` renders object states as compact lines/csv rows |

`layout` controls where the question sits relative to the state.
`question_first` makes the question head cacheable across requests and
reads more accurately on short states; `state_first` prefills a long
document once for all questions; `header` lists every question before the
state so the document is encoded with all of them in view. `auto` picks
`question_first` only when the question heads outweigh the state — the
case where warm caches win (qf re-decodes the state per question, sf
decodes it once). It falls back to `state_first` for states > 2000 chars
(even a single question reads a long document better after it), whenever any question has `allow_abstain` (an
abstain slot read before the evidence primes abstention), or when the
state is bigger than the heads. `catalog` lists every question (numbered,
instructions only) before the state, then each item is just a
`QUESTION i — name` pointer plus its OPTIONS — the `[head+catalog]` span
caches across requests on the same question set, so a stream of different
states decodes only the state once plus a ~20-token tail per question.
The resolved layout is in `x_snap`.

Answers carry extras on top of the Jev shape — `status`, `confidence`,
full `probabilities`, and typed fields (`boolean`, `level`, `value`) —
which Jev clients simply ignore.

`choice` is not bounded by the alphabet: past 26 options (256 max) the
question expands. `expand: probes` (default) scores each option with an
independent probe — *is this candidate the correct answer?* — batched in
waves over the shared prefix, then yes-masses normalize into the returned
distribution; with `allow_abstain`, no candidate reaching 0.5 abstains.
`expand: pages` instead splits candidates into equal-size pages of ≤26
real choices — far fewer decode items, but page-conditional probabilities
and no abstention.

## Local AI — security by construction

- **Data never leaves the machine.** Weights are a GGUF on disk,
  downloaded once from HuggingFace. No API key, no telemetry, no third
  party reading your customers' tickets.
- **Deterministic.** Same input, same distribution. Log it, replay it,
  threshold it — every answer ships with its full probability map.
- **Nothing to jailbreak.** The model cannot emit a payload because it
  cannot emit text. Prompt injection can nudge a distribution; it cannot
  speak.
- **One static binary.** `cargo build --release` → `snap`. A container, a
  VM, an air-gapped box — no Python, no runtime deps.

## Performance

Apple M1 Max, Metal, in-process `snap bench --requests 20`, p50. Every
request carries a fresh state — a stream of new documents — so only what
production can reuse gets reused: the template head and the question
heads.

| scenario | minicpm5-2b Q4_K_M | spark-4b Q8_0 | qwen3.8-4b Q4_K_M |
|---|---:|---:|---:|
| single question | 50 ms | 89 ms | 82 ms |
| 4 questions | 126 ms (32 ms/q) | 232 ms (58 ms/q) | 227 ms (57 ms/q) |
| 8 questions | 252 ms (32 ms/q) | 406 ms (51 ms/q) | 472 ms (59 ms/q) |
| 8 questions, `mode: direct` (nothing shared) | 1170 ms (146 ms/q) | 2158 ms (270 ms/q) | 2381 ms (298 ms/q) |
| 8 questions on a 1.8 KB document | 1887 ms (236 ms/q) | 3278 ms (410 ms/q) | 3327 ms (416 ms/q) |
| 4 KB state, 1 question | 1364 ms | 2476 ms | 2164 ms |

The `direct` row decodes every question on its own: the gap to the row
above it is what prefix sharing and the cached question heads buy. The
hybrid qwen3.8 runs the same batched path, with 17 parallel sequences
instead of 65.

### …and against the same weights through Ollama

Ollama 0.34.3 serving `openbmb/minicpm5-2b` (its own packaging of the
same MiniCPM5-2B Q4_K_M), same machine, both over HTTP, p50 of 15, each
engine's requests back to back, and every request a state that engine
has never seen. Ollama runs `think:false` at temperature 0, with the
fixed question text first so its prefix cache reuses it the way SNAP's
does. `python3 eval/vs_ollama.py` reproduces the table.

| workload | Ollama, JSON + probabilities | Ollama, JSON answers | Ollama, one letter each | SNAP | vs fastest Ollama |
|---|---:|---:|---:|---:|---:|
| 1 question | 898 ms (69 tok) | 163 ms (9 tok) | 61 ms | 54 ms | −11% |
| 4 questions | 3704 ms (282 tok) | 342 ms (26 tok) | 257 ms | 135 ms | −48% |
| 8 questions | 7874 ms (562 tok) | 732 ms (50 tok) | 504 ms | 263 ms | −48% |
| 5 KB state, 1 question | 3872 ms (73 tok) | 2738 ms (8 tok) | 2650 ms | 2226 ms | −16% |

- **One letter per question** is Ollama's floor: a letter of text to
  parse, no probabilities, one request per question (asked for every
  letter in one reply, `"A,C,B"`, it stopped early at 4 and 8
  questions). On a single question it comes close to SNAP (61 vs 54 ms):
  same prompt, one position read. SNAP pulls ahead as questions are
  added, because they share one pass instead of one request each.
- **JSON** is what a pipeline actually wires, under a schema. Asked for
  what a SNAP answer carries (key, confidence, a probability per option)
  Ollama writes every token of it, and those probabilities are text the
  model made up; SNAP's come from the logits.
- **A long state with one question** is bound by prefill on both sides;
  the gap there is prefill speed, not the method.

## Accuracy

`snap evaluate eval/*.jsonl` — accuracy and **balanced accuracy** (mean
per-class recall) against ground truth, plus the quality of the
distributions themselves: **Brier score** and **ECE** (expected
calibration error on the top probability). Cases can ship `variants` —
paraphrased `state`/`question` fields — and the report adds a
**consistency** block: answer agreement and mean probability drift,
broken down per perturbation kind. SNAP also auto-generates stability
probes per case — option-order reversal (positional bias), a
meaning-preserving criterion rewording, and an unrelated-context
injection — `--no-perturb` skips them. Line format:
`{"id", "state", "question", "expect", "variants"?}`.

Measured with the API's own defaults — `layout: auto`, no abstain slot
unless a case asks for one — i.e. exactly what `/v1/systemone` serves:

| model | core (52) | edge (19) | ms/case |
|---|---:|---:|---:|
| qwen3.8-4b Q4_K_M | **94.2%** | **79.0%** | ~210 |
| spark-4b Q8_0 | 92.3% | **79.0%** | ~180 |
| minicpm5-2b Q4_K_M | 80.8% | 57.9% | ~95 |

These sets are small: the 95% interval is about ±7–11 points on the 52
core cases and ±17–20 on the 19 edge cases, so a few points between the
4B models is noise.

Distribution quality (lower is better; same runs):

| model | brier core | brier edge | ECE core | ECE edge |
|---|---:|---:|---:|---:|
| qwen3.8-4b Q4_K_M | 0.126 | 0.201 | 0.159 | 0.151 |
| spark-4b Q8_0 | 0.157 | 0.370 | 0.101 | 0.190 |
| minicpm5-2b Q4_K_M | 0.300 | 0.627 | 0.079 | 0.281 |

The ECE column is why calibration exists: qwen is *under*confident (73%
mean confidence at 94% accuracy on core), spark is overconfident where
it's weaker (87% confidence at 79% on edge) — both fixable by
`snap calibrate`, both invisible to accuracy alone.

Stability on core — how often the answer survives a perturbation that
shouldn't change it:

| model | options reversed | instruction reworded | unrelated context added |
|---|---:|---:|---:|
| qwen3.8-4b | 91% | 90% | 90% |
| spark-4b | 91% | 81% | 92% |
| minicpm5-2b | 68% | 81% | 88% |

Reversing the option order flips a third of MiniCPM's choices: letter
and position bias is the main weakness of small models answering by
letter. MiniCPM-2B is the speed/footprint option; the 4B models are the
production pick.

### TypeSafe's public cases

`eval/typesafe_public.py` runs the 20 public cases from
[evals.typesafe.ai](https://evals.typesafe.ai) — four workflows, 373
decisions, reference = frontier consensus — against a running `snap serve`,
then scores agreement the same way
[jev-on-a-laptop](https://github.com/rorshopping/jev-on-a-laptop) does, so
numbers are comparable across Jev reproductions (extraction logic adapted
from it, MIT). TypeSafe's raw case data is fetched at runtime, never
committed.

```bash
python3 eval/typesafe_public.py fetch && python3 eval/typesafe_public.py extract
snap serve --model qwen3.8-4b --ctx 32768 &   # invoices need room
python3 eval/typesafe_public.py answer --out results/typesafe-public/snap-qwen38.json
python3 eval/typesafe_public.py score results/typesafe-public/snap-qwen38.json
```

qwen3.8-4b on an M1 Max: **73.2%** agreement over the 373 decisions
(Security 35/48, AgentTrace 32/49, Invoice 130/184, CustomerSvc 76/92) —
87% on yes/no questions, 56% on scores, 48% on choices, ~1.1 s per
decision. The misses are not random: half are on the invoices, where
whole question families cross-check a ~7k-token document (is this line
really delivered, is this price really approved) and a 4B model
answering in one token disagrees systematically with reasoning frontier
models — the kind of question to route elsewhere when a calibrated
answer comes back `contested`.

## Calibration

Raw letter logits are honest but uncalibrated: `0.9` does not mean
"right 90% of the time" until you measure it. `snap calibrate` runs
your eval cases, collects every emitted distribution with its
ground-truth target, and fits one temperature per question type:

```bash
snap calibrate --model qwen3.8-4b eval/core.jsonl eval/edge.jsonl -o calibration.json
# fitted on 69 cases (2 skipped)
#   boolean  T=0.477   choice  T=0.474   numeric  T=1.000   score  T=0.774
# ece  0.127 raw -> 0.089 in-sample | 0.106 out-of-fold  CI95 [0.055, 0.185]
# caveat: OOF interval overlaps raw ECE — gain not proven at this n
snap serve --model qwen3.8-4b --calibration calibration.json
```

(T &lt; 1 sharpens: qwen3.8 is *under*confident on this set. The caveat
line is printed whenever the evidence doesn't separate — that's the
point of reporting it.)

Three ECE numbers are printed, and they mean different things. **Raw**
is where you start. **In-sample** is scored on the same cases the fit
saw — optimistic by construction, kept for reference. **Out-of-fold**
is the honest one: every case is scored with a temperature fit on the
*other* folds (5-fold, group-disjoint — a case's variants never leak
across folds). The 95% bootstrap interval resamples whole cases, and
the report flags whether the gain actually separates from raw at that
confidence — at these sample sizes, read the interval, not the point.

The file binds to the exact model id and prompt version — a calibration
fitted on another build refuses to load. `--calibration` is accepted by
`serve`, `-p`, `evaluate`, `bench`. This is post-hoc scaling, not
retraining: a single scalar corrects over/under-confidence, not the
shape of the distribution, and it holds only as far as your eval data
resembles production traffic.

## Quickstart

Prebuilt binaries on
[GitHub Releases](https://github.com/emnlmn/snap/releases):

macOS (Apple Silicon) via Homebrew:

```bash
brew install emnlmn/snap/snap
```

or the tarball:

```bash
mkdir -p ~/snap && curl -L https://github.com/emnlmn/snap/releases/latest/download/snap-macos-arm64.tar.gz | tar xz -C ~/snap
```

Linux x86_64:

```bash
mkdir -p ~/snap && curl -L https://github.com/emnlmn/snap/releases/latest/download/snap-linux-x86_64.tar.gz | tar xz -C ~/snap
```

`snap` and the `lib*.so*` files must stay in the same directory
(`$ORIGIN` rpath) — that's why it installs to a folder, not a bin dir.
`~/snap/snap` runs as-is; `ln -sf ~/snap/snap ~/.local/bin/snap` puts
it on PATH. Other assets: `linux-x86_64-v3` (single-file AVX2),
`linux-x86_64-musl` (fully static), `linux-aarch64`,
`linux-x86_64-vulkan`. macOS isn't notarized — a browser-quarantined
tarball may need `xattr -d com.apple.quarantine ~/snap/snap`. First
inference pulls the model GGUF (~1.5 GB), then offline.

Once a day `snap` checks GitHub for a newer release and prints a line on
stderr — never on stdout, never in `-p`. `SNAP_NO_UPDATE_CHECK=1` turns
it off.

Or build from source — llama.cpp is vendored and compiled at first
build (~2 min), Metal on by default on Apple Silicon:

```bash
make setup    # rustup + cmake/clang check
make build    # → ./target/release/snap
make test     # unit tests + functional smoke (pulls minicpm, ~1.5 GB)
make lint     # cargo fmt --check + clippy -D warnings
```

### Build variants

The default build targets the host: Metal on Apple Silicon, baseline CPU
elsewhere. GPU and portable-CPU variants are cargo features:

```bash
make build FEATURES="cuda"              # NVIDIA — needs CUDA toolkit
make build FEATURES="vulkan"            # AMD/Intel/generic GPU driver
make build FEATURES="dynamic-backends"  # every CPU variant, dispatched at load
```

`dynamic-backends` ships `snap` plus `libggml*`/`libllama*` runtime libs
and `libggml-cpu-*.so` modules — the portable pick for servers. Single-file instead: `RUSTFLAGS="-C target-cpu=x86-64-v3"
cargo build --release` (AVX2, ~2013+ CPUs). CI builds all of them per
platform — see `.github/workflows/`.

```bash
snap models                                 # known shortcuts
snap -p '{"state":"…","questions":{"urgent":{"type":"noul"}}}'   # one-shot, claude-style
snap -p request.json                        # same thing, from a file (stdin works too)
snap serve --model qwen3.8-4b --port 8018   # HTTP server
snap ps                                   # running servers (pid, model, uptime, state)
snap stop                                 # stop the one server; --all / --port / --pid for more
snap evaluate eval/core.jsonl               # accuracy + brier/ece + consistency
snap calibrate eval/*.jsonl -o cal.json     # fit temperatures -> calibration file
snap bench --requests 20                    # latency/throughput
```

`--model` takes a name from `snap models` — the tested set only. A GGUF
that loads is not a GGUF that answers correctly; new candidates get a
row in `src/models.rs` after they pass the eval suite.

`--ctx` (default 8192 tokens ≈ ~6k words) is the KV pool: every prompt —
state + question + options — must fit in it, and the questions in flight
share it with the cache (more questions than fit just run in more
waves). It is reserved in full at startup: ~42 KB/token for minicpm5-2b,
~144 KB/token for spark-4b (its sliding-window layers keep full-size KV so
prefixes can be forked), ~32 KB/token for qwen3.8-4b plus ~0.85 GB of
recurrent state. `8192` ≈ 0.34 / 1.1 / 0.25 GB, `32768` ≈ 1.3 / 4.5 /
1 GB on top of the weights. Long inputs are rejected 422 (never
truncated), so raise it only when your states need it:
`snap serve --ctx 32768`. Deliberately not "model max": spark advertises
1M tokens, which would be ~140 GB of reservation.

`--threads` (default 0 = all available cores) sets llama.cpp's decode
threads — relevant on CPU-only builds; on GPU backends it barely matters.
llama.cpp's own default is 4, which starves prefill on a bigger box.

## HTTP

| endpoint | purpose |
|---|---|
| `POST /v1/systemone` | the API — Jev wire (`noul` / `choice` / `score`) + `numeric`, abstain, `mode` extensions |
| `GET /v1/models` | served model id |
| `GET /healthz` | readiness — answers only after engine warm-up |
| `GET /playground` | built-in console, embedded in the binary |

`snap serve` registers a pidfile so other terminals can see it: `snap ps`
lists every running server (including `starting` while the model loads
and strays answering on :8018), `snap stop` shuts one down — bare `stop`
when there's exactly one, `--port`/`--pid`/`--all` otherwise. SIGTERM
drains in-flight requests; `--force` is SIGKILL.

## Playground

`snap serve` already carries it — no assets, no build step:

```bash
snap serve --model spark-4b   # then open http://localhost:8018/playground
```

- **`/playground`** — an API console. Build a request channel per
  question (all five types), flip between builder, raw JSON and cURL,
  and read every answer as a probability map on a shared 0–100% scale
  plus the `x_snap` internals (decode/total ms, tokens decoded vs prompt
  tokens, cache hits, waves). `⌘↵` runs.

## Production

```bash
./snap serve --model qwen3.8-4b --host 0.0.0.0 --port 8018
```

- **One process = one resident model.** Requests serialize on the engine;
  parallelism lives *inside* a request (one batched decode per wave). Scale
  with N processes behind a load balancer.
- **Memory** ≈ weights + KV (`--ctx` × the per-token cost above) +
  recurrent state on hybrids. Peak RSS at 8192: minicpm5-2b 1.9 GB,
  qwen3.8-4b 3.8 GB, spark-4b 5.4 GB.
- **Boot** ~4 s on an M1 Max once the GGUF is downloaded (mmap, resident
  head, warm-up requests). `/healthz` is your readiness probe.
- **Logs** quiet by default (warnings only); `--debug` or
  `RUST_LOG=llamac=info` for llama.cpp internals on stderr.
- Over-context input → 422. Never silently truncated.

```ini
[Service]
ExecStart=/usr/local/bin/snap serve --model qwen3.8-4b --port 8018
Restart=on-failure
Environment=RUST_LOG=info
```

## Limits

- `choice` scales to 256 options via per-option probes; `score` and
  `numeric` stay inside the 26-letter alphabet.
- Hybrid/recurrent models run the same batched path with fewer parallel
  sequences (17 instead of 65 — each pins a recurrent-state row), so large
  question sets take more waves there.
- Probabilities are calibrated only after `snap calibrate` — and only as
  far as your eval data resembles production.

---

<p align="center"><i>One pass. One distribution. The jaw snaps shut.</i></p>
