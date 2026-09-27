#!/usr/bin/env bash
# Standard eval battery for ONE model: internal eval + latency bench +
# typed-decisions. Create-only reports land in results/ — same file name
# for a model means the run is reused, never repeated.
#
#   eval/run_all.sh <model> <name>
#
# <model>: registry name (minicpm5-2b) or a .gguf path. <name> becomes the
# report filename stem. TD_KIND=in-domain for fine-tuned runs.
set -euo pipefail

MODEL=${1:?model — registry name or gguf path}
NAME=${2:?run name}
HERE=$(cd "$(dirname "$0")" && pwd)
RES="$HERE/../results"
SNAP="$HERE/../target/release/snap"
PORT=${SNAP_PORT:-8420}

mkdir -p "$RES/typed-decisions"

"$SNAP" evaluate --model "$MODEL" "$HERE/cases.jsonl" --output "$RES/eval-$NAME.json"
"$SNAP" bench --model "$MODEL" --requests 20 --output "$RES/bench-$NAME.json"

if [ ! -f "$RES/typed-decisions/$NAME.json" ]; then
  "$SNAP" serve --model "$MODEL" --port "$PORT" >/dev/null 2>&1 &
  srv=$!
  trap 'kill "$srv" 2>/dev/null' EXIT
  ok=""
  for _ in $(seq 1 240); do
    curl -sf "http://127.0.0.1:$PORT/healthz" >/dev/null && { ok=1; break; }
    sleep 1
  done
  [ -n "$ok" ] || { echo "server on :$PORT never came up"; exit 1; }
  python3 "$HERE/typed_decisions.py" answer --url "http://127.0.0.1:$PORT" \
    --kind "${TD_KIND:-zero-shot}" --note "run_all:$NAME" \
    --out "$RES/typed-decisions/$NAME.json"
  kill "$srv" 2>/dev/null
fi

echo "reports: $RES/{eval,bench}-$NAME.json + typed-decisions/$NAME.json"
echo "timeline: python3 eval/history.py"
