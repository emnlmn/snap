#!/usr/bin/env bash
# Merge a trained LoRA adapter into MiniCPM5-2B and export GGUF quants.
#
#   lora/merge_gguf.sh <adapter_dir> <out_dir> [name]
#
# Requires: transformers+peft env (training venv), a llama.cpp checkout with
# convert_hf_to_gguf.py and llama-quantize (LLAMA_CPP_DIR env or ~/llama.cpp).
set -euo pipefail

ADAPTER=${1:?adapter dir}
OUTDIR=${2:?output dir}
NAME=${3:-snap-2b}
LLAMA=${LLAMA_CPP_DIR:-$HOME/llama.cpp}
BASE=openbmb/MiniCPM5-2B

test -d "$ADAPTER" || { echo "no adapter at $ADAPTER"; exit 1; }
test -f "$LLAMA/convert_hf_to_gguf.py" || { echo "llama.cpp not at $LLAMA (set LLAMA_CPP_DIR)"; exit 1; }
mkdir -p "$OUTDIR"

MERGED="$OUTDIR/${NAME}-merged-hf"
python3 - "$ADAPTER" "$MERGED" "$BASE" <<'PY'
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

adapter, out, base = sys.argv[1:4]
tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
m = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16,
                                         trust_remote_code=True)
m = PeftModel.from_pretrained(m, adapter).merge_and_unload()
m.save_pretrained(out, safe_serialization=True)
tok.save_pretrained(out)
print("merged ->", out)
PY

F16="$OUTDIR/${NAME}-bf16.gguf"
python3 "$LLAMA/convert_hf_to_gguf.py" "$MERGED" \
    --outfile "$F16" --outtype bf16

# optional importance matrix from exported training prompts — on a 2B the
# quant loss can rival the fine-tune gain, so Q4_K_M deserves the extra help.
# IMATRIX_DATA may be an export JSONL (prompts auto-extracted) or a plain
# text file. NOTE: -c sets the context; llama.cpp's "--chunk" is an alias
# for --from-chunk (resume offset) and would skip the data entirely.
if [ -n "${IMATRIX_DATA:-}" ]; then
  cmake --build "$LLAMA/build" --target llama-imatrix -j"$(nproc)"
  IDATA="$IMATRIX_DATA"
  if [[ "$IDATA" == *.jsonl ]]; then
    IDATA="$OUTDIR/imatrix-prompts.txt"
    python3 -c 'import sys, json
[print(json.loads(l)["prompt"]) for l in open(sys.argv[1])]' \
        "$IMATRIX_DATA" > "$IDATA"
    echo "imatrix corpus: $(wc -l < "$IDATA") prompts -> $IDATA"
  fi
  "$LLAMA/build/bin/llama-imatrix" -m "$F16" -f "$IDATA" \
      -o "$OUTDIR/imatrix.dat" -c 512
  IMATRIX_ARGS=(--imatrix "$OUTDIR/imatrix.dat")
else
  IMATRIX_ARGS=()
fi

for q in q8_0 q4_k_m; do
  "$LLAMA/build/bin/llama-quantize" "${IMATRIX_ARGS[@]}" \
      "$F16" "$OUTDIR/${NAME}-${q}.gguf" "${q^^}"
done

sha256sum "$OUTDIR"/*.gguf | tee "$OUTDIR/SHA256SUMS"
echo "done -> $OUTDIR"
