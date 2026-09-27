#!/usr/bin/env bash
# Corpus to training rows: validate the synthetic shards, collect teacher
# review flags, export snap's own prompts, split. Everything up to training.
#
# Resumable: a step whose output already exists is skipped. The review step
# needs teacher models served by snap on TEACHER_PORTS (default 8018 8022
# 8023); LAYA_URL adds an optional second voice.
set -euo pipefail
cd "$(dirname "$0")"

SNAP=${SNAP_BIN:-$PWD/../target/release/snap}
PORTS=${TEACHER_PORTS:-8018 8022 8023}
SERVERS=""
for p in $PORTS; do SERVERS="$SERVERS --server http://127.0.0.1:$p"; done

if [ -f data/splits/layerB.jsonl ]; then
  echo "== layerB.jsonl exists, skipping validate"
else
  echo "== validate synthetic shards -> layerB"
  python3 prepare/validate.py data/raw/s*.jsonl --write data/splits/layerB.jsonl
fi

# preflight: a teacher must answer before hours of labeling start
first=${PORTS%% *}
if ! curl -sf --max-time 5 "http://127.0.0.1:$first/healthz" > /dev/null; then
  echo "!! no teacher on :$first — start the snap servers first:"
  echo "   for p in $PORTS; do $SNAP serve --model <teacher> --port \$p & done"
  exit 1
fi

echo "== teacher review flags on layerB (resume-safe)"
python3 teacher/label.py data/splits/layerB.jsonl \
    --out data/labeled/layerB.jsonl $SERVERS \
    ${LAYA_URL:+--laya "$LAYA_URL"} --workers 9 --resume

echo "== export prompts (byte-exact, N shuffled-criteria copies per choice row)"
for layer in A B; do
  out="data/train/export-layer${layer}.jsonl"
  if [ -f "$out" ]; then
    echo "  $out exists, skipping"
    continue
  fi
  extra=""
  [ "$layer" = "A" ] && extra="--max-per-source ${MAX_PER_SOURCE:-0}"
  [ -f data/drop-ids.txt ] && extra="$extra --drop-ids data/drop-ids.txt"
  python3 lora/export_prompts.py "data/labeled/layer${layer}.jsonl" \
      --out "$out" --model minicpm5-2b --permute 2 $extra
done

echo "== merge + split (global state/chain groups)"
if [ -f data/train/final-train.jsonl ]; then
  echo "  splits exist, skipping"
else
  python3 lora/make_splits.py \
      data/train/export-layerA.jsonl data/train/export-layerB.jsonl \
      --manifest data/raw/manifest.jsonl \
      --cases data/labeled/layerA.jsonl data/labeled/layerB.jsonl \
      --out-prefix data/train/final --dev 0.05 --holdout 0.05
fi

echo "== ready. Train on a CUDA machine (see TRAINING.md):"
echo "  python3 lora/train_lora.py --train data/train/final-train.jsonl \\"
echo "      --dev data/train/final-dev.jsonl --out runs/adapter-r16 \\"
echo "      --rank 16 --alpha 32 --dropout 0.05 --epochs 1 --bsz 8 --accum 8 \\"
echo "      --grad-ckpt --eval-rows 512"
