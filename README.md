<p align="center">
  <img src="assets/logo.png" width="180" alt="snap logo"/>
</p>

<h1 align="center">snap</h1>

<p align="center">
  <b>Single-pass Neural Answer Probabilities</b><br/>
  Ask everything. Generate nothing.<br/>
  <sub>local · deterministic · Jev wire-compatible · <a href="https://emnlmn.github.io/snap/">emnlmn.github.io/snap</a></sub>
</p>

<p align="center">
  <a href="https://github.com/emnlmn/snap/actions/workflows/ci.yml"><img src="https://github.com/emnlmn/snap/actions/workflows/ci.yml/badge.svg" alt="CI"/></a>
  <a href="https://github.com/emnlmn/snap/releases/latest"><img src="https://img.shields.io/github/v/release/emnlmn/snap" alt="release"/></a>
  <img src="https://img.shields.io/badge/platform-macOS%20%C2%B7%20Linux%20%C2%B7%20Windows-lightgrey" alt="platforms: macOS · Linux · Windows"/>
  <a href="https://opensource.org/license/mit"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="license: MIT"/></a>
</p>

---

snap is a decision engine for LLM pipelines. The input is your data plus
every question you have about it; the output is a set of typed answers,
each with its full probability distribution, from a single pass over the
model. There is no generated text, so there is nothing to parse, retry or
clean up.

A pipeline that asks a chat model for a label usually carries a prompt
that asks for JSON, a parser that works most of the time, a retry loop for
the times it doesn't, and a bill for tokens that are thrown away right
after parsing. All of that exists because a chat model answers by
writing. snap reads the answer out of the model instead.

Every question becomes a prompt whose answer is a single letter, `A` to
`Z`, one per option. snap runs the model once, reads the logits at that
one position, merges the tokens that spell the same letter and turns them
into a distribution with a softmax. The questions of a request share one
batched `llama_decode` call.

```jsonc
// in
{
  "state": "a ticket, a CRM record, a sensor dump: anything",
  "questions": {
    "route":  {"type": "choice", "criteria": {"billing": "…", "tech": "…"}},
    "urgent": {"type": "noul",   "instructions": "SLA breach likely?"},
    "impact": {"type": "score",  "criteria": ["low", "mid", "high"]}
  }
}

// out: one distribution per question (status, coverage and x_snap left out)
{
  "answers": {
    "route":  {"choice": "billing", "probabilities": {"billing": 0.87, "tech": 0.13}},
    "urgent": {"noul": 0.97, "boolean": true, "confidence": 0.94},
    "impact": {"level": 2, "score": 0.81}
  }
}
```

## What it's for

snap is the shorter path for any label, flag or score you would otherwise
prompt a model for and then parse: routing and triage of support tickets,
a check on an agent's tool call before it runs, the severity of an
incident page, a moderation verdict on a listing, the document that
answers a search query, a judgment on whether an answer is faithful to its
context, the qualification of an inbound lead. The
[site](https://emnlmn.github.io/snap/#use-cases) shows each of these with
a real request and snap's real response.

## How it works

1. **Letter prompts.** Each question becomes a prompt that stops exactly
   where the first token of the answer goes. Each option gets a letter,
   so the whole answer is a single token: a yes/no question (`noul`) gets
   A and B, a `choice` one letter per option, a `score` one per level, a
   `numeric` question one per anchor value across your range.
2. **One shared prefill.** The state goes through the model once, and
   what the model computes from it, the KV cache, is shared by every
   question. The prompts of a request form a shared-prefix tree, so any
   text several questions have in common is computed once, and the
   constant prompt head stays resident between requests.
3. **One batched decode.** All the questions decode together in a single
   `llama_decode` call, with one KV sequence each: four questions cost one
   pass, not four conversations. Hybrid and recurrent architectures take
   the same path with fewer parallel sequences (17 instead of 65). snap
   detects the architecture at startup, with nothing to configure.
4. **One logit row.** For each question snap reads a single row of
   logits, keeps only the option letters, merges the variants of the same
   letter (like `A` and ` A`) and applies a softmax. `coverage` reports
   how much of the model's raw next-token probability landed on those
   letters.

These 26 letters are the entire output surface. With more than 26
options, each option gets its own yes/no probe over the same shared
prefix, up to 256.

The question can also come before the state. With `layout:
question_first` the question head is identical across requests and stays
cached on its own KV sequence (LRU-bounded), so a repeat workload, such as
the same triage questions on a stream of tickets, decodes only the state.

Every response carries `x_snap`, an account of what the engine did: the
prompt tokens against the tokens actually decoded
(`usage.input_tokens`), `cached_head_tokens`, `shared_prefix_tokens`,
`cache_hits`/`cache_misses`, `waves`, and decode and total milliseconds.

## The API — drop-in Jev, extended

`POST /v1/systemone` speaks the TypeSafe/Jev wire format. A client
written for Jev works unchanged against a local `snap serve`: same `noul`,
`choice` and `score` questions, same response shape. There is one
endpoint and no parallel API.

The same request accepts optional snap extensions. Jev clients can ignore
them, and the defaults preserve Jev semantics:

| field | extension |
|---|---|
| question `"type": "numeric"` | `{min, max, granularity}` or `{min, max, step}`: a distribution over a numeric range |
| question `allow_abstain` | default `false`; `true` adds an `__abstain__` slot (status `abstained`) |
| request `mode` | `shared` (default: shared text decoded once, spans cached across requests) or `direct` (every question decoded alone, the reference path) |
| request `layout` | `auto` (default), `state_first`, `question_first`, `header`, `catalog` |
| request `expand` | `probes` (default) or `pages`: how a choice with more than 26 options expands |
| request `compact_state` | default `false`; `true` renders object states as compact lines or CSV rows |

`layout` decides where the question sits relative to the state:

- `state_first` puts the state first, so a long document is prefilled
  once for all the questions.
- `question_first` puts the question head first. The head caches across
  requests and reads more accurately on short states, but the state is
  decoded again for each question.
- `header` lists every question before the state, so the document is
  encoded with all of them in view.
- `catalog` lists every question before the state, numbered and with
  instructions only, then each item is a `QUESTION i — name` pointer plus
  its options. The head and catalog span caches across requests on the
  same question set, so a stream of different states costs one state
  decode plus a tail of about 20 tokens per question.
- `auto`, the default, picks `question_first` only when the question heads
  outweigh the state, which is where warm caches win. It falls back to
  `state_first` for states over 2000 characters (even a single question
  reads a long document better after it), whenever a question has
  `allow_abstain` (an abstain slot read before the evidence primes
  abstention), and whenever the state is bigger than the heads.

The resolved layout is reported in `x_snap`.

Answers carry extras on top of the Jev shape, which Jev clients simply
ignore: `status`, `confidence`, `coverage`, the full `probabilities`, and
typed fields (`boolean`, `level`, `value`).

A `choice` is not bounded by the alphabet: past 26 options, up to 256,
the question expands. With `expand: probes`, the default, each option gets
an independent probe (*is this candidate the correct answer?*), batched in
waves over the shared prefix, and the yes-probabilities are normalized
into the returned distribution; with `allow_abstain`, the question
abstains when no candidate reaches 0.5. With `expand: pages` the
candidates are split into equal pages of at most 26 real options: far
fewer decode items, at the cost of page-conditional probabilities and no
abstention.

The full surface is in [`openapi.yaml`](openapi.yaml).

## Security

**No text channel for prompt injection.** Injection works by getting a
model to produce text you didn't intend: a leaked system prompt, a
smuggled tool call, a link your UI renders. snap never produces text.
What leaves the engine is a probability over options you wrote, in a JSON
shape the server builds, not the model.

| attack | LLM that generates text | snap |
|---|---|---|
| data exfiltration through the response | possible, the response is free text | no text channel: answers are numbers keyed by your option names |
| a smuggled tool call or command | possible, the model writes the call | nothing to execute: your code maps a decided option to an action you wrote |
| a leaked system prompt or context | possible, it can be asked to repeat itself | nothing is generated for it to leak into |
| markup or links injected downstream | possible, output gets rendered in a UI or an email | the response schema is fixed by the server |
| a broken output format | handled with validators, repair and retries | impossible, the model doesn't write the response |
| persuasion toward a different answer | possible, and invisible in the output | still possible, but limited and measurable |

The last row is the residual risk. Text in the state can only shift
probability between your options, and that shift shows up as a split
`confidence` or a low `coverage`; `allow_abstain` gives the model a way to
say it can't tell. Set a threshold and send those cases to a person.

- **On your hardware.** A single binary with llama.cpp built in, the same
  on a laptop, a VM, a Kubernetes pod or an air-gapped rack. There is also
  a fully static Linux build, with no Python and no runtime to keep
  patched.
- **No data egress.** No API key, no telemetry, no third party handling
  your records. The weights are a GGUF file on disk, downloaded once from
  Hugging Face; after that snap runs offline.
- **Auditable.** The same input always gives the same distribution. The
  full probabilities can be logged with every decision and replayed
  later, and `x_snap` shows what the engine did.
- **Open source.** MIT-licensed Rust, readable end to end. Models come
  from a tested list, and a calibration file refuses to load with a
  different model or prompt version than the one it was fitted on.

## Performance

Apple M1 Max with Metal, in-process `snap bench --requests 20`, medians.
Every request carries a fresh state, as in a stream of new documents, so
only what production can reuse gets reused: the template head and the
question heads.

| scenario | minicpm5-2b Q4_K_M | spark-4b Q8_0 | qwen3.8-4b Q4_K_M |
|---|---:|---:|---:|
| single question | 50 ms | 89 ms | 82 ms |
| 4 questions | 126 ms (32 ms/q) | 232 ms (58 ms/q) | 227 ms (57 ms/q) |
| 8 questions | 252 ms (32 ms/q) | 406 ms (51 ms/q) | 472 ms (59 ms/q) |
| 8 questions, `mode: direct` (nothing shared) | 1170 ms (146 ms/q) | 2158 ms (270 ms/q) | 2381 ms (298 ms/q) |
| 8 questions on a 1.8 KB document | 1887 ms (236 ms/q) | 3278 ms (410 ms/q) | 3327 ms (416 ms/q) |
| 4 KB state, 1 question | 1364 ms | 2476 ms | 2164 ms |

The `direct` row decodes every question on its own: its distance from the
row above is the gain from prefix sharing and cached question heads. The
hybrid qwen3.8 runs the same batched path with 17 parallel sequences
instead of 65.

### …and against the same weights through Ollama

Ollama 0.34.3 serving `openbmb/minicpm5-2b`, its own packaging of the same
MiniCPM5-2B Q4_K_M, on the same machine. Both engines run over HTTP,
medians of 15, each engine's requests back to back, and every request
carries a state that engine has never seen. Ollama runs with
`think:false` at temperature 0, with the fixed question text first so its
prefix cache reuses it the way snap's does. `python3 eval/vs_ollama.py`
reproduces the table.

| workload | Ollama, JSON + probabilities | Ollama, JSON answers | Ollama, one letter each | snap | vs fastest Ollama |
|---|---:|---:|---:|---:|---:|
| 1 question | 898 ms (69 tok) | 163 ms (9 tok) | 61 ms | 54 ms | −11% |
| 4 questions | 3704 ms (282 tok) | 342 ms (26 tok) | 257 ms | 135 ms | −48% |
| 8 questions | 7874 ms (562 tok) | 732 ms (50 tok) | 504 ms | 263 ms | −48% |
| 5 KB state, 1 question | 3872 ms (73 tok) | 2738 ms (8 tok) | 2650 ms | 2226 ms | −16% |

- **One letter per question** is the quickest Ollama can go, and it still
  means a letter of text to parse, no probabilities and one request per
  question. Asked for every letter in a single reply (`"A,C,B"`), it
  stopped early at 4 and 8 questions. On a single question it comes close
  to snap (61 against 54 ms), since the prompt is the same and one
  position is read; snap pulls ahead as questions are added, because they
  share one pass instead of one request each.
- **JSON under a schema** is what a pipeline actually wires. Asked for
  what a snap answer carries (key, confidence, a probability per option),
  Ollama writes every token of it, and its probabilities are text the
  model made up; snap's come from the logits.
- **A long state with one question** is bound by prefill on both sides,
  so the gap there comes from prefill speed, not from the method.

### …and against a hosted API

`eval/vs_openai.py` sends the same prompts and JSON schemas to
gpt-5.6-luna with reasoning turned off, medians of 5 requests (each call
is billed). The time includes network and queue, because that is the
wait a pipeline sees when the model it would call instead is an API away.

| workload | OpenAI, JSON + probabilities | OpenAI, JSON answers | snap (M1 Max) |
|---|---:|---:|---:|
| 1 question | 1677 ms (60 tok) | 1170 ms (13 tok) | 54 ms |
| 4 questions | 2829 ms (219 tok) | 1235 ms (31 tok) | 135 ms |
| 8 questions | 3707 ms (431 tok) | 2013 ms (55 tok) | 263 ms |

## Accuracy

`snap evaluate eval/cases.jsonl` produces one report from one file:
accuracy and **balanced accuracy** (mean per-class recall) against ground
truth, plus the quality of the distributions themselves, as **Brier
score** and **ECE** (expected calibration error on the top probability).
Cases can ship `variants`, paraphrased `state` or `question` fields, and
the report then adds a **consistency** block with answer agreement and
mean probability drift, broken down per perturbation kind. snap also
generates three stability probes per case: the option order reversed
(positional bias), the criterion reworded without changing its meaning,
and an unrelated context injected; `--no-perturb` skips them.

Line format: `{"id", "state", "question", "expect", "variants"?,
"requires_abstain"?, "layout"?, "expand"?, "compact_state"?}`. The pins are
the request's own knobs on that case — a malformed pin is an error, not a
quiet default, and `--layout` overrides all of them. Base cases
use `<domain>-NN` ids and adversarial cases `edge-<stress>-NN`.

`snap export-prompts` reads the same case files and writes, one JSONL
record per case, the exact prompt `evaluate` decodes for it — chat template
applied, token ids, resolved layout, letter-to-key slots. It is the
supervision surface the fine-tuning pipeline trains on (see
[TRAINING.md](TRAINING.md)). A choice with more options than the
26 letters has no single prompt, so it is skipped (with a note on stderr)
instead of blocking the run; ids must be unique and `--output` is
create-only.

The numbers below come from the 303 cases of `eval/cases.jsonl`, measured
with the API's own defaults (`layout: auto`, no abstain slot unless a
case asks for one), which is exactly what `/v1/systemone` serves. The two
`edge-contested` cases have no single right answer and carry no
expectation, so accuracy is scored on 301.

| model | accuracy | balanced | ms/case |
|---|---:|---:|---:|
| qwen3.8-4b Q4_K_M | **86.4%** | **66.7%** | ~300 |
| spark-4b Q8_0 | 85.7% | 57.5% | ~245 |
| minicpm5-2b Q4_K_M | 69.1% | 48.1% | ~135 |

The 95% interval is about ±4 points at 301 cases, so the gap between the
two 4B models is noise. Balanced accuracy averages recall over answer
positions (A, B, C…), so rare late positions, such as numeric anchors or
the tail of a long choice list, weigh as much as A and B: that is why it
sits well below plain accuracy.

Accuracy per question type:

| model | choice | noul | boolean | score | numeric |
|---|---:|---:|---:|---:|---:|
| qwen3.8-4b | 91% | 91% | 84% | 86% | 58% |
| spark-4b | 83% | 95% | 95% | 79% | 76% |
| minicpm5-2b | 68% | 74% | 90% | 67% | 52% |

`numeric` is the weakest type on every model: a value read off anchor
letters misses more often than a label does.

Distribution quality, lower is better for Brier and ECE, from the same
runs:

| model | brier | ECE | mean confidence |
|---|---:|---:|---:|
| qwen3.8-4b Q4_K_M | 0.200 | 0.121 | 67.0% |
| spark-4b Q8_0 | 0.242 | 0.059 | 86.0% |
| minicpm5-2b Q4_K_M | 0.457 | 0.124 | 66.8% |

The ECE column is the reason calibration exists, and accuracy alone
doesn't show it. spark comes out of the box close to calibrated, with
86.0% mean confidence at 85.7% accuracy. qwen is *under*confident, with
67% confidence at 86% accuracy, and `snap calibrate` brings its ECE from
0.121 to 0.047 out of fold (see [Calibration](#calibration)).

Stability, as the share of answers that survive a perturbation that
shouldn't change them (option reversal applies only to the 127 `choice`
cases):

| model | options reversed | instruction reworded | unrelated context added |
|---|---:|---:|---:|
| qwen3.8-4b | 87% | 85% | 84% |
| spark-4b | 83% | 77% | 83% |
| minicpm5-2b | 69% | 72% | 79% |

Reversing the option order flips almost a third of MiniCPM's choices:
letter and position bias is the main weakness of small models answering
by letter. MiniCPM-2B is the option for speed and footprint; the 4B
models are the production pick.

### TypeSafe's public cases

`eval/typesafe_public.py` runs the 20 public cases from
[evals.typesafe.ai](https://evals.typesafe.ai) (four workflows, 373
decisions, with frontier consensus as the reference) against a running
`snap serve`. It scores agreement the same way
[jev-on-a-laptop](https://github.com/rorshopping/jev-on-a-laptop) does,
so the numbers are comparable across Jev reproductions; the extraction
logic is adapted from it (MIT). TypeSafe's raw case data is fetched at
runtime and never committed.

```bash
python3 eval/typesafe_public.py fetch && python3 eval/typesafe_public.py extract
snap serve --model qwen3.8-4b --ctx 32768 &   # invoices need room
python3 eval/typesafe_public.py answer --out results/typesafe-public/snap-qwen38.json
python3 eval/typesafe_public.py score results/typesafe-public/snap-qwen38.json
```

qwen3.8-4b on an M1 Max reaches **73.2%** agreement over the 373
decisions (Security 35/48, AgentTrace 32/49, Invoice 130/184, CustomerSvc
76/92): 87% on yes/no questions, 56% on scores and 48% on choices, at
about 1.1 s per decision. The misses are not random. Half of them are on
the invoices, where whole families of questions cross-check a document of
about 7k tokens (is this line really delivered, is this price really
approved), and a 4B model answering in one token disagrees systematically
with frontier models that reason first. That is the kind of question to
route elsewhere when a calibrated answer comes back `contested`.

### LocalLLaMA/typed-decisions

`eval/typed_decisions.py` is the external benchmark in the reports: 400
cases / 2,000 decisions, five questions over one shared state, gold being
the mean of three samples from a ~4B teacher (agreement, not correctness;
the card reads ~0.75 as saturation). It replays the dataset's own
`state`+`questions` protocol, so its
numbers line up with the card's (Jev 1.13.0 zero-shot: 0.727). What it
catches that `eval/cases.jsonl` cannot: the same question repeats over
100 different states, so **argmax constancy** measures whether the model
reads the state at all. `score --detail` prints it per question — a run
collapsing ≥95% of answers on a question is predicting the prior, not
reading (minicpm5-2b zero-shot: 6/20 questions constant, acc 0.502;
qwen3.8-4b: 8/20, acc 0.561, but its wins concentrate on lucky constants).

```bash
python3 eval/typed_decisions.py fetch       # test + train, pinned revision
snap serve --model minicpm5-2b &
python3 eval/typed_decisions.py answer --kind zero-shot \
    --out results/typed-decisions/minicpm5-2b.json
python3 eval/typed_decisions.py score results/typed-decisions/*.json --detail
```

`--kind` is required and means what the card means: `zero-shot` if the
model never trained on these workflows, `in-domain` if it did — a run
fine-tuned on the benchmark's train split is in-domain, and its number is
not a generalization claim. Runs of different kinds are reported but
flagged as not comparable. The corpus `tasksource-jev-typed-decisions`
train rows can feed fine-tuning; the test split never does.

### Runs and history

Every report is a create-only JSON — a file name is a run, never
repeated, never overwritten. The standard battery (evaluate + bench +
typed-decisions) is one command, in `training/`:

```bash
cd training
BASE_GGUF=none eval/compare.sh qwen3.8-4b qwen-4b        # solo run
eval/compare.sh runs/gguf/snap-2b-q4_k_m.gguf run1       # paired vs base
python3 ../eval/history.py                               # the whole timeline
```

`history.py` scans `results/` and `training/eval/results/` and
prints one line per report — accuracy, ECE, KL, latency — so whether a
change helped or regressed is read off a table, not reconstructed from
memory. Old runs stay on disk as history; a paired view of two specific
reports is `eval/typed_decisions.py score A.json B.json --against A.json`.

## Calibration

Raw letter probabilities are honest but uncalibrated: `0.9` does not mean
"right 90% of the time" until you measure it. `snap calibrate` runs your
eval cases, collects every distribution with its ground-truth target and
fits one temperature per question type:

```bash
snap calibrate --model qwen3.8-4b eval/cases.jsonl -o calibration.json
# fitted on 295 cases (8 skipped)
#   boolean  T=0.558
#   choice   T=0.395
#   numeric  T=0.867
#   score    T=0.573
# ece  0.121 raw -> 0.041 in-sample | 0.047 out-of-fold  CI95 [0.037, 0.087]
# gain is CI-separated from raw at 95% — real, not fitting noise
snap serve --model qwen3.8-4b --calibration calibration.json
```

A temperature below 1 sharpens the distribution, and here all four are
below 1: qwen3.8 is *under*confident on every question type. The last
line is the verdict. In this run the whole out-of-fold interval sits
below the raw ECE, so the gain is real; when the interval overlaps the
raw value, snap prints `caveat: … gain not proven at this n` instead.

The report prints three ECE numbers with different meanings. **Raw** is
the starting point. **In-sample** is scored on the same cases the fit
saw, optimistic by construction and kept for reference. **Out-of-fold**
is the honest one: every case is scored with a temperature fitted on the
*other* folds (5-fold, group-disjoint, so a case's variants never leak
across folds). The 95% bootstrap interval resamples whole cases, and the
report flags whether the gain separates from raw at that confidence. At
these sample sizes, read the interval, not the point.

The calibration file binds to the exact model id and prompt version, and
a file fitted on another build refuses to load. `--calibration` is
accepted by `serve`, `-p`, `evaluate` and `bench`. This is post-hoc
scaling, not retraining: a single scalar corrects over- or
under-confidence, not the shape of the distribution, and it holds only as
far as your eval data resembles production traffic.

## Quickstart

Prebuilt binaries are on
[GitHub Releases](https://github.com/emnlmn/snap/releases).

macOS on Apple Silicon, with Homebrew:

```bash
brew install emnlmn/snap/snap
```

or with the tarball:

```bash
mkdir -p ~/snap && curl -L https://github.com/emnlmn/snap/releases/latest/download/snap-macos-arm64.tar.gz | tar xz -C ~/snap
```

Linux x86_64:

```bash
mkdir -p ~/snap && curl -L https://github.com/emnlmn/snap/releases/latest/download/snap-linux-x86_64.tar.gz | tar xz -C ~/snap
ln -sf ~/snap/snap ~/.local/bin/snap   # optional: put it on PATH
```

Windows x86_64: download
[`snap-windows-x86_64.tar.gz`](https://github.com/emnlmn/snap/releases/latest/download/snap-windows-x86_64.tar.gz),
unpack it with `tar xzf snap-windows-x86_64.tar.gz` and run `.\snap serve`
from that folder.

The Linux and Windows builds ship `snap` next to its runtime libraries,
and the two must stay in the same directory (on Linux through the
`$ORIGIN` rpath): that is why the install goes to a folder rather than a
bin directory. Other Linux assets: `linux-x86_64-v3` (single file, AVX2),
`linux-x86_64-musl` (fully static), `linux-aarch64`,
`linux-x86_64-vulkan`. The macOS build isn't notarized, so a tarball
quarantined by the browser may need
`xattr -d com.apple.quarantine ~/snap/snap`.

Then:

```bash
snap serve   # minicpm5-2b on port 8018, then open http://localhost:8018/playground
```

The first run downloads the model GGUF once (1.5 to 4 GB depending on the
model); after that everything runs offline. Once a day `snap` checks
GitHub for a newer release and prints a line on stderr, never on stdout
and never in `-p`; `SNAP_NO_UPDATE_CHECK=1` turns the check off.

### Commands

```bash
snap models                                 # the tested models
snap -p '{"state":"…","questions":{"urgent":{"type":"noul"}}}'   # one-shot request
snap -p request.json                        # the same, from a file (stdin works too)
snap serve --model qwen3.8-4b --port 8018   # HTTP server
snap ps                                     # running servers (pid, model, uptime, state)
snap stop                                   # stop the only server; --all / --port / --pid for more
snap evaluate eval/cases.jsonl              # accuracy + brier/ece + consistency
snap calibrate eval/cases.jsonl -o cal.json # fit temperatures into a calibration file
snap export-prompts eval/cases.jsonl > p.jsonl  # the prompts evaluate decodes, for training
snap bench --requests 20                    # latency and throughput
```

`--model` takes a name from `snap models`, and only from that tested set.
A GGUF that loads is not a GGUF that answers correctly, so a new
candidate gets a row in `src/models.rs` only after it passes the eval
cases.

`--ctx` (default 8192 tokens, about 6k words) is the KV pool. Every
prompt, meaning state, question and options, must fit in it, and the
questions in flight share it with the cache; more questions than fit
simply run in more waves. The pool is reserved in full at startup: about
42 KB per token for minicpm5-2b, about 144 KB per token for spark-4b (its
sliding-window layers keep full-size KV so that prefixes can be forked),
and about 32 KB per token for qwen3.8-4b plus about 0.85 GB of recurrent
state. That is roughly 0.34 / 1.1 / 0.25 GB at `8192` and 1.3 / 4.5 / 1
GB at `32768`, on top of the weights. Inputs that don't fit are rejected
with a 422, never truncated, so raise the pool only when your states need
it: `snap serve --ctx 32768`. The default is deliberately not the model's
maximum: spark advertises 1M tokens, which would mean a reservation of
about 140 GB.

`--threads` (default 0, meaning all available cores) sets llama.cpp's
decode threads. It matters on CPU-only builds and barely at all on GPU
backends; llama.cpp's own default of 4 starves prefill on a bigger box.

### From source

llama.cpp is vendored and compiled on the first build (about 2 minutes),
with Metal on by default on Apple Silicon:

```bash
make setup    # rustup + cmake/clang check
make build    # → ./target/release/snap
make test     # unit tests + functional smoke (pulls minicpm, ~1.5 GB)
make lint     # cargo fmt --check + clippy -D warnings
```

The default build targets the host: Metal on Apple Silicon, baseline CPU
elsewhere. GPU backends and portable CPU builds are cargo features,
additive and combinable:

```bash
make build FEATURES="cuda"              # NVIDIA, needs the CUDA toolkit
make build FEATURES="rocm"              # AMD, needs ROCm
make build FEATURES="vulkan"            # AMD, Intel or any Vulkan driver
make build FEATURES="dynamic-backends"  # every CPU variant, dispatched at load
```

`metal`, `opencl`, `openmp`, `mkl` and `static-stdcxx` are available the
same way. `dynamic-backends` ships `snap` together with the
`libggml*`/`libllama*` runtime libraries and the `libggml-cpu-*.so`
modules, and it is the portable pick for servers. For a single file
instead, `RUSTFLAGS="-C target-cpu=x86-64-v3" cargo build --release`
builds for AVX2 (x86 CPUs from about 2013 on). CI builds every release
variant per platform; see `.github/workflows/`.

## HTTP

| endpoint | purpose |
|---|---|
| `POST /v1/systemone` | the API: Jev wire (`noul` / `choice` / `score`) plus the `numeric`, abstain, `mode` and `layout` extensions |
| `GET /v1/models` | the model this process serves |
| `POST /v1/models` | `{"model": "<name>"}` loads another model from the tested set and swaps it in; the current one keeps serving until the new one is ready |
| `GET /healthz` | readiness: answers only after the engine warm-up |
| `GET /playground` | the built-in console, embedded in the binary |

`snap serve` registers a pidfile so that other terminals can see it.
`snap ps` lists every running server, including those still `starting`
while the model loads and strays answering on :8018. `snap stop` shuts
one down: bare `stop` when there is exactly one, `--port`, `--pid` or
`--all` otherwise. SIGTERM drains the requests in flight; `--force` sends
SIGKILL.

## Playground

`snap serve` already carries it, with no assets and no build step:

```bash
snap serve --model spark-4b   # then open http://localhost:8018/playground
```

`/playground` is an API console. It builds a request one question at a
time, across all five question types, switches between the form, raw JSON
and cURL, and shows every answer as a probability map on a shared 0–100%
scale, together with the `x_snap` internals (decode and total ms, tokens
decoded against prompt tokens, cache hits, waves). `⌘↵` runs the request.

## Production

```bash
./snap serve --model qwen3.8-4b --host 0.0.0.0 --port 8018
```

- **One process, one resident model.** Requests serialize on the engine,
  and the parallelism lives inside a request, as one batched decode per
  wave. To scale, run N processes behind a load balancer.
- **Memory** is roughly the weights plus the KV pool (`--ctx` times the
  per-token cost above) plus the recurrent state on hybrids. Peak RSS at
  8192: minicpm5-2b 1.9 GB, qwen3.8-4b 3.8 GB, spark-4b 5.4 GB.
- **Boot** takes about 4 s on an M1 Max once the GGUF is downloaded (mmap,
  resident head, warm-up requests). `/healthz` is the readiness probe.
- **Logs** are quiet by default (warnings only); `--debug` or
  `RUST_LOG=llamac=info` shows llama.cpp internals on stderr.
- **Over-context input** gets a 422 and is never silently truncated.

```ini
[Service]
ExecStart=/usr/local/bin/snap serve --model qwen3.8-4b --port 8018
Restart=on-failure
Environment=RUST_LOG=info
```

## Limits

- A request holds up to 64 questions.
- `choice` scales to 256 options through per-option probes; `score` and
  `numeric` stay inside the 26-letter alphabet, special slots included,
  and a `numeric` question uses 2 to 24 anchors.
- Hybrid and recurrent models run the same batched path with fewer
  parallel sequences (17 instead of 65, since each pins a recurrent-state
  row), so large question sets take more waves there.
- Probabilities are calibrated only after `snap calibrate`, and only as
  far as your eval data resembles production.

## Next to Jev and SemIf

The two closest projects are Jev, the hosted original, and
[SemIf](https://github.com/TheoLeeCJ/SemIf), an open research scorer.

| | snap | Jev | SemIf |
|---|---|---|---|
| deployment | local, one Rust binary | hosted service | local Python, WebGPU demo |
| open weights | GGUF | no | Qwen |
| Jev-compatible HTTP API | yes | the original | no, CLI only |
| numeric answers | yes | no | no |
| calibrated probabilities | per question type | claimed | per workload |
| TypeSafe public eval | 73% on a 4B model | ~88%, self-reported | not reported |

The Jev column comes from its public description and the SemIf column
from its repository. Jev is a hosted product made by its own authors;
snap is an independent implementation of the same API.

---

<p align="center"><i>One pass. One distribution. The jaw snaps shut.</i></p>
