#!/usr/bin/env bash
# One-shot setup for a fresh RunPod CUDA pod (RTX 4090 or better).
# Run inside the pod, from training/:  bash lora/pod_setup.sh
set -euo pipefail

# pinned llama.cpp — conversion + quantization must be reproducible; the
# F16 parity check (eval/smoke_parity.py) is what proves the converter's
# output matches what snap's pinned llama-cpp-sys-2 reads. b11000 is the
# near-matching tag that knows the minicpm5 pre-tokenizer (earlier tags
# fail HF->GGUF conversion of the MiniCPM5 vocab).
LLAMA_TAG="${LLAMA_TAG:-b11000}"

apt-get update -qq && apt-get install -y -qq git cmake build-essential > /dev/null
pip install -q -r requirements.txt
# torch comes from the pod image (CUDA build) — never let pip swap it for
# the CPU wheel; fail loudly if the image doesn't provide CUDA
python3 -c "import torch; assert torch.cuda.is_available(), \
    'torch has no CUDA — image without a cuda torch build?'"

# base weights (~4.7GB) + llama.cpp for convert/quantize
python3 -c "from huggingface_hub import snapshot_download; \
    snapshot_download('openbmb/MiniCPM5-2B'); print('weights ok')"
if [ ! -d "$HOME/llama.cpp" ]; then
  git clone --depth 1 --branch "$LLAMA_TAG" \
      https://github.com/ggml-org/llama.cpp "$HOME/llama.cpp"
  cmake -S "$HOME/llama.cpp" -B "$HOME/llama.cpp/build" -DCMAKE_BUILD_TYPE=Release
  cmake --build "$HOME/llama.cpp/build" --target llama-quantize -j"$(nproc)"
  pip install -q -r "$HOME/llama.cpp/requirements.txt"
fi

# F16 base GGUF for the strict parity check — compares HF bf16 against the
# same weights snap reads, without Q4 noise masking implementation drift
if [ ! -f "$HOME/models/minicpm5-2b-f16.gguf" ]; then
  mkdir -p "$HOME/models"
  python3 "$HOME/llama.cpp/convert_hf_to_gguf.py" \
      ~/.cache/huggingface/hub/models--openbmb--MiniCPM5-2B/snapshots/*/ \
      --outfile "$HOME/models/minicpm5-2b-f16.gguf" --outtype f16
  echo "f16 gguf ready: $HOME/models/minicpm5-2b-f16.gguf"
fi
echo "pod ready. export LLAMA_CPP_DIR=$HOME/llama.cpp"
