# Calibration

Raw letter probabilities are honest but uncalibrated: `0.9` does not mean
"right 90% of the time" until you measure it. Calibration measures it on
your own labeled cases and fits one temperature per question type, so that
a stated confidence matches the observed accuracy. It is optional, and
worth doing when you act on the probabilities (thresholds, routing on
`contested`, escalation to a human) rather than only on the top answer.

## 1. Prepare the data

Calibration reads the same JSONL as `snap evaluate`: one case per line, each
with a `state`, a `question` and the answer you know is right in `expect`.

```jsonl
{"id":"route-01","state":{"subject":"Charged twice","body":"Invoice #4411 appears twice, please refund."},"question":{"type":"choice","instructions":"Which team?","criteria":{"billing":"invoices, refunds","technical":"bugs, outages","other":"everything else"}},"expect":{"choice":"billing"}}
{"id":"spam-01","state":"You won a free cruise, click here","question":{"type":"noul","instructions":"Is this spam?"},"expect":{"boolean":true}}
```

- One question per case and a unique `id` per case. `expect` takes the key
  of its type: `choice`, `boolean`, `level` or `score`, `value` (numeric).
  The full line format is in [Accuracy](README.md#accuracy), and
  `eval/cases.jsonl` is a working example.
- `variants` are ignored: only the base case is decoded. They matter for
  consistency checks, not here.
- Cases that cannot supervise the fit are counted as *skipped* and left out:
  an `expect` without a usable key, or `status` values other than `abstained`
  and `out_of_bounds`.
- Use real traffic, labeled by someone who knows the answer, in the
  proportions you see in production. The temperature fits whatever mix you
  give it, easy cases included. Don't reuse cases the model was fine-tuned
  on or that you tuned the prompt against: it would look more accurate
  than it is on new data.

## 2. How many

Each question type is fitted on its own (`noul` and `boolean` share one), and
a type with fewer than 4 cases keeps `T=1`. Four is the floor, not a
target. One temperature settles early, but the report also has to show
that the gain is real, and that takes volume: aim for 100 or more cases
per type you use, and several hundred overall if you can. The shipped
suite fits 295. With fewer, the out-of-fold interval gets wide and snap
says so instead of claiming a gain. A type absent from the file stays
uncalibrated.

## 3. Fit

```bash
snap calibrate --model qwen3.8-4b my-cases.jsonl -o calibration.json
# fitted on 295 cases (8 skipped)
#   boolean  T=0.558
#   choice   T=0.395
#   numeric  T=0.867
#   score    T=0.573
# ece  0.121 raw -> 0.041 in-sample | 0.047 out-of-fold  CI95 [0.037, 0.087]
# gain is CI-separated from raw at 95% — real, not fitting noise
```

You can pass several files at once. The output is create-only, so a second
run needs a new `-o` path. Each case is decoded once at temperature 1, and
snap looks for the temperature that minimizes the log-loss on the true
answers (`softmax(z/T)`, the same as `p^(1/T)` renormalized). Below 1 the
distribution sharpens, above 1 it flattens. All four here are below 1:
qwen3.8 is *under*confident on every type.

The report prints three ECE numbers. Raw is the starting point. In-sample
is scored on the same cases the fit saw, so it is optimistic, and it is
there for reference. Out-of-fold is the one to trust: every case is scored
with a temperature fitted on the other folds (5 folds, split by case, so
nothing leaks across them). The 95% bootstrap interval resamples whole
cases. If the whole interval sits below the raw ECE, the gain is real. If
it overlaps, snap prints `caveat: … gain not proven at this n` and still
writes the file, and it's up to you to use it as is or collect more cases.
At these sample sizes, read the interval, not the point.

## 4. Use it

```bash
snap serve --model qwen3.8-4b --calibration calibration.json
snap evaluate --model qwen3.8-4b --calibration calibration.json holdout.jsonl
```

`--calibration` is accepted by `serve`, `-p`, `evaluate` and `bench`. The
fitted temperature multiplies the request's own `temperature` for that
question type, so a request at the default 1.0 gets exactly the fitted value.
Nothing else in the API changes: the same fields and answers come back,
only the probabilities move.

To confirm it on data the fit never saw, run `snap evaluate` on a held-out
file with and without the flag and compare the ECE and Brier lines. Refit
whenever the model or the prompt format changes: the file binds to the exact
model id and prompt version, and a file fitted on another build refuses to
load. Keep in mind that this is post-hoc scaling, not retraining. One number per
type fixes over- or underconfidence but not the shape of the distribution,
and it only holds as far as your cases resemble production traffic.
