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
- **State amortized.** All questions in a request share the same state
  prefix, evaluated once. On full-attention architectures every question
  suffix decodes in **a single batched call** across parallel KV
  sequences. The constant prompt head stays resident between requests.
- **Questions amortized too.** `layout: question_first` flips the prompt —
  the question head is identical across requests and stays resident on its
  own KV sequence (state snapshots on hybrid archs, LRU-bounded). Repeat
  workloads — the same triage questions on a stream of tickets — decode
  only the state.
- **Honest internals.** `x_snap` reports exactly what happened:
  `cached_head_tokens`, `shared_prefix_tokens`, `qhead_hits/misses`,
  `rewind: kv|snapshot`, `suffix_decode: batched|sequential`,
  prefill/total ms. Every answer also carries `coverage` — how much of the
  model's raw next-token mass landed on the allowed letters at all.

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
| request `mode` | `shared` (default, prefix amortized) or `direct` |
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
with several questions, whenever any question has `allow_abstain` (an
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

Apple Silicon, Metal, in-process, p50:

| scenario | minicpm5-2b Q4_K_M | spark-4b Q8_0 | qwen3.8-4b Q4_K_M |
|---|---:|---:|---:|
| single question | 136 ms | 272 ms | 284 ms |
| 4 questions, shared prefix | 454 ms (114 ms/q) | 816 ms (204 ms/q) | 992 ms (249 ms/q) |
| 8 questions, shared prefix | 803 ms (100 ms/q) | 1760 ms (217 ms/q) | 1914 ms (248 ms/q) |
| 8 questions, direct mode | 1008 ms (126 ms/q) | 2135 ms (271 ms/q) | 2450 ms (305 ms/q) |
| 8 KB state, 1 question | 1245 ms | 2818 ms | 2427 ms |

MiniCPM and Spark run the batched multi-sequence path; Qwen's hybrid KV
falls back to sequential suffixes (still prefix-cached — a startup probe
decides, you don't configure it).

### …and against the same weights through Ollama

Same MiniCPM-2B GGUF, same machine, `think:false`, minimal
`num_predict`, temperature 0, cold prompts:

| workload | Ollama | SNAP | |
|---|---:|---:|---|
| 1 question | 193 ms | 136 ms | −30% |
| 4 questions, one request each | 788 ms | 454 ms | −42% |
| 8 questions, one request each | 1884 ms | 803 ms | **−57%** |
| 4 questions, combined prompt | 559 ms | 454 ms | −19% |
| 8 questions, combined prompt | 962 ms | 803 ms | −17% |
| ~8k-token state | rejected (ctx) | 1245 ms | — |

The combined-prompt column is Ollama's best case for N questions — and it
hands you `"A,C,B"` as *text* to parse, with no probabilities. SNAP's
whole answer is the distribution.

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

| model | core (52) | edge (19) | ms/case |
|---|---:|---:|---:|
| qwen3.8-4b Q4_K_M | **90.4%** | **89.5%** | ~240 |
| spark-4b Q8_0 | 88.5% | 73.7% | ~230 |
| minicpm5-2b Q4_K_M | 67.3% | 57.9% | ~135 |

Distribution quality (lower is better; same runs):

| model | brier core | brier edge | ECE core | ECE edge |
|---|---:|---:|---:|---:|
| qwen3.8-4b Q4_K_M | 0.156 | 0.205 | 0.120 | 0.081 |
| spark-4b Q8_0 | 0.228 | 0.342 | 0.130 | 0.182 |
| minicpm5-2b Q4_K_M | 0.472 | 0.690 | 0.175 | 0.347 |

The ECE column is why calibration exists: spark is accurate but
overconfident (94% mean confidence vs 74–88% accuracy), qwen is actually
*under*confident (72% vs 90%) — both fixable by `snap calibrate`, both
invisible to accuracy alone. On `variants` all three agreed 100% of the
time with ≤0.03 mean drift — a tiny sample (n=2), but the harness works.

MiniCPM-2B is the speed/footprint option; the 4B models are the
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

## Calibration

Raw letter logits are honest but uncalibrated: `0.9` does not mean
"right 90% of the time" until you measure it. `snap calibrate` runs
your eval cases, collects every emitted distribution with its
ground-truth target, and fits one temperature per question type:

```bash
snap calibrate eval/core.jsonl eval/edge.jsonl -o calibration.json
# fitted on 69 cases (2 skipped)
#   boolean  T=0.536   choice  T=0.125   score  T=1.006
# ece  0.110 raw -> 0.068 in-sample | 0.074 out-of-fold  CI95 [0.04, 0.15]
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

macOS (Apple Silicon):

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

`--ctx` (default 8192 tokens ≈ ~6k words) is the per-question context
limit — state + question + options — and the KV arena is reserved in
full at startup, ~140 KB/token on llama.cpp: `8192` ≈ 1 GB, `32768` ≈
4.5 GB, `131072` ≈ 18 GB on top of the weights. Long inputs are
rejected 422 (never truncated), so raise it only when your states need
it: `snap serve --ctx 32768`. Deliberately not "model max": spark
advertises 1M tokens, which would be ~140 GB of reservation.

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
  plus the `x_snap` internals (prefill/total ms, cached head, batched
  vs sequential suffix decode). `⌘↵` runs.

## Production

```bash
./snap serve --model qwen3.8-4b --host 0.0.0.0 --port 8018
```

- **One process = one resident model.** Requests serialize on the engine;
  parallelism lives *inside* a request (the batched suffix decode). Scale
  with N processes behind a load balancer.
- **Memory** ≈ weights + KV (`--ctx` × ~140 KB/token, less on
  hybrid-attention models) + the multi-seq context. qwen3.8-4b Q4_K_M @
  8192: ~4 GB total.
- **Boot** ~10 s (mmap, probes, head warm). `/healthz` is your readiness
  probe.
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
- Batched decode needs unified-KV full attention; hybrids fall back to
  sequential automatically.
- Probabilities are calibrated only after `snap calibrate` — and only as
  far as your eval data resembles production.

---

<p align="center"><i>One pass. One distribution. The jaw snaps shut.</i></p>
