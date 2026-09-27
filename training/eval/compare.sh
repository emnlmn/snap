#!/usr/bin/env bash
# The eval battery — one command per run, create-only reports under
# eval/results/<tag>/, paired analysis via eval/paired.py.
#
#   eval/compare.sh <ft.gguf> <tag>            paired vs BASE_GGUF
#   BASE_GGUF=none eval/compare.sh <m> <tag>   solo battery, *-solo.json
#   eval/compare.sh <m> <tag> <cases...>       extra args override case files
#
# Each model gets: `snap evaluate` on every case file (default: the engine's
# eval/cases.jsonl + this pipeline's dev/holdout cases), `snap bench`, and
# the typed-decisions external bench. Paired reports are <case>-{base,ft}.json
# for paired.py; solo runs use -solo. BASE_GGUF defaults to the registry
# minicpm5-2b — pass the base converted+quantized with the SAME pipeline as
# the ft when imatrix is in play, or the delta conflates quantization
# quality with the training delta. TD kinds default to base=zero-shot,
# ft/solo=in-domain (layerA carries the benchmark's train split; its test
# states never reach training) — TD_KIND overrides the solo/ft kind,
# TD_KIND_BASE the base's. TD=0 skips the external bench.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)  # the engine repo this ft/ lives in

FT=${1:?model gguf path}
TAG=${2:?run tag}
shift 2
SNAP=${SNAP_BIN:-$ROOT/target/release/snap}
BASE=${BASE_GGUF:-minicpm5-2b}
OUTDIR=eval/results/$TAG

if [ $# -gt 0 ]; then
  CASES="$*"
else
  CASES="$ROOT/eval/cases.jsonl"
  for f in data/train/final-dev-cases.jsonl data/train/final-holdout-cases.jsonl; do
    [ -f "$f" ] && CASES="$CASES $f"
  done
fi

# "name:path:kind" per model — the kind is the typed-decisions run kind.
if [ "$BASE" = none ]; then
  MODELS="solo:$FT:${TD_KIND:-in-domain}"
else
  MODELS="base:$BASE:${TD_KIND_BASE:-zero-shot} ft:$FT:${TD_KIND:-in-domain}"
fi

mkdir -p "$OUTDIR"
for spec in $MODELS; do
  m="${spec%%:*}"; rest="${spec#*:}"; path="${rest%%:*}"
  for f in $CASES; do
    name=$(basename "$f" .jsonl)
    out="$OUTDIR/${name}-${m}.json"
    if [ -f "$out" ]; then
      echo "== $out exists, skipping (create-only)"
      continue
    fi
    echo "== $f on $m -> $out"
    "$SNAP" evaluate --model "$path" "$f" --output "$out"
  done
  out="$OUTDIR/bench-${m}.json"
  if [ -f "$out" ]; then
    echo "== $out exists, skipping (create-only)"
  else
    echo "== bench on $m -> $out"
    "$SNAP" bench --model "$path" --requests "${BENCH_REQUESTS:-20}" --output "$out"
  fi
done
echo "reports in $OUTDIR — paired analysis: python3 eval/paired.py $OUTDIR"

# --- external bench: LocalLLaMA/typed-decisions test split ------------------
# Server mode (the benchmark replays one /v1/systemone request per case), one
# snap serve per model, answer files are create-only like the eval reports.
TD=${TD:-1}
TD_PY=${TD_PY:-$ROOT/eval/typed_decisions.py}
TD_PORT=${TD_PORT:-8377}
if [ "$TD" = 1 ]; then
  for spec in $MODELS; do
    m="${spec%%:*}"; rest="${spec#*:}"; path="${rest%%:*}"; kind="${rest##*:}"
    out="$OUTDIR/td-${m}.json"
    if [ -f "$out" ]; then
      echo "== $out exists, skipping (create-only)"
      continue
    fi
    echo "== typed-decisions on $m ($kind) -> $out"
    "$SNAP" serve --model "$path" --port "$TD_PORT" >/dev/null 2>&1 &
    srv=$!
    ok=""; served=""
    for _ in $(seq 1 240); do
      served=$(curl -sf "http://127.0.0.1:$TD_PORT/healthz" 2>/dev/null) \
        && { ok=1; break; }
      sleep 1
    done
    if [ -z "$ok" ]; then
      echo "!! server on :$TD_PORT did not come up for $m — skipping TD"
      kill "$srv" 2>/dev/null
      continue
    fi
    # a stale server already on the port makes healthz pass while our serve
    # dies on bind — the TD run would then score a different model silently
    if ! kill -0 "$srv" 2>/dev/null; then
      echo "!! :$TD_PORT already in use ($served) — skipping TD for $m"
      continue
    fi
    python3 "$TD_PY" answer --url "http://127.0.0.1:$TD_PORT" --kind "$kind" \
      --note "compare:$TAG $path" --out "$out" || rm -f "$out"
    kill "$srv" 2>/dev/null
    wait "$srv" 2>/dev/null || true
  done
  if [ "$BASE" != none ]; then
    echo "td paired score: python3 $TD_PY score $OUTDIR/td-base.json $OUTDIR/td-ft.json" \
      "--against $OUTDIR/td-base.json --detail"
  fi
fi
