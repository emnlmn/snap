# Training

snap reads a decision from one forward pass: a softmax over the option
letters, with no generated text. `training/` fine-tunes a small model for
exactly that readout, with a LoRA trained on the letter logits of snap's own
prompts. It ships the result as a GGUF that `snap` serves like any other
model.

## Layout

| Path | Role |
|---|---|
| `training/prepare/` | public datasets converted to snap's case format (`convert_hf.py`); schema, dedup and contamination checks (`validate.py`) |
| `training/teacher/` | review flags from teacher models served by snap (`label.py`) |
| `training/lora/` | prompt export, grouped splits, LoRA training, GGUF merge and quantization, GPU setup |
| `training/eval/` | paired base-vs-fine-tune evaluation (`compare.sh`, `paired.py`) and the HF/GGUF parity check (`smoke_parity.py`) |
| `training/pipeline.sh` | validate → review → export → split, resumable |
| `training/data/`, `runs/`, `models/`, `eval/results/` | local only (gitignored) |

Commands run from `training/`. The scripts call the engine at
`../target/release/snap` (build it with `make build` at the root; the
`SNAP_BIN` variable overrides the path) and read the engine's
`../eval/cases.jsonl` for contamination checks and evaluation.

Requirements: Python 3.11+ and `pip install -r requirements.txt`. Training
needs a CUDA GPU with the image's own CUDA build of torch
(`lora/pod_setup.sh` checks it). Apple-silicon (MPS) training is not
supported. Conversion and quantization use a pinned llama.cpp checkout
(`LLAMA_CPP_DIR`, see `lora/pod_setup.sh`).

## Data

The corpus has two parts, in the same case format as `eval/cases.jsonl`:

- **Public datasets**, converted by `prepare/convert_hf.py`. Every row keeps
  its source and license. Rows whose license does not allow commercial use
  are dropped by default.
- **Synthetic cases**, produced by a separate generator that is not part of
  this repository. The pipeline reads its output as JSONL shards in
  `data/raw/`.

Every shard passes `prepare/validate.py`: schema, near-duplicate removal,
and contamination against `eval/cases.jsonl`. Rows that slip through are
listed in `data/drop-ids.txt` and removed at export. The typed-decisions
*test* split never enters training. Its *train* split may, and that makes
typed-decisions results in-domain (see Evaluation).

## Invariants

- **The supervision surface is byte-exact.** Prompts come from
  `snap export-prompts`, which renders through the same compile step as
  `decide`. Prompts are never rebuilt in Python. Exported rows carry the
  serve-time `token_ids`, and the trainer uses them directly.
  `eval/smoke_parity.py` verifies the HF retokenization fallback.
- **The loss is the readout.** It is a softmax over the case's legal letter
  tokens only. Each letter's logit is the max over its single-token variants
  (`"A"`, `" A"`, `"\nA"`), pooled the way `engine.rs` reads a row.
- **Targets are simple.** A target is the source's own soft distribution
  where one exists, and a smoothed one-hot on the verified label otherwise.
  Teacher models never shape a target: their disagreement only sets a
  `review` flag, and `--drop-contested` can drop the row. Calibration is a
  post-training step (`snap calibrate` on dev, never on holdout).
- **Abstention must not become an attractor.** The gate reports accuracy on
  abstain-enabled rows whose gold is a real option.
- **Splits are leak-free.** Rows are grouped globally, never per stratum:
  rows with the same normalized state or the same chain id land in the same
  fold. Holdout is touched once, and all reports are create-only.

## Pipeline

```bash
cd training

# 1-4. validate the synthetic shards, collect teacher review flags, export
#      byte-exact prompts with per-letter targets, split into train/dev/
#      holdout plus eval-format dev/holdout case files. Resumable.
#      Teachers: `snap serve --model <teacher> --port <p>` on TEACHER_PORTS.
./pipeline.sh

# 5. parity gate before paying for a GPU: HF bf16 vs snap on the same
#    prompts. Token ids must match and the KL must be near zero. Strict
#    parity needs a BF16/F16 GGUF of the base model; a Q4 comparison mixes
#    quantization loss into the gap.
snap serve --model models/minicpm5-2b-base-bf16.gguf --port 8099 &
python3 eval/smoke_parity.py data/train/export-layerA.jsonl \
    --cases data/labeled/layerA.jsonl --n 80 --server http://127.0.0.1:8099

# 6. train (CUDA)
python3 lora/train_lora.py --train data/train/final-train.jsonl \
    --dev data/train/final-dev.jsonl --out runs/adapter-r16 \
    --rank 16 --alpha 32 --dropout 0.05 --epochs 1 --bsz 8 --accum 8 \
    --grad-ckpt --eval-rows 512

# 7. merge and quantize (bf16, q8_0, q4_k_m). IMATRIX_DATA takes the export
#    JSONL or a plain text file.
IMATRIX_DATA=data/train/final-train.jsonl \
    lora/merge_gguf.sh runs/adapter-r16 runs/gguf snap-2b

# 8. paired evaluation: base and fine-tune on the same cases
BASE_GGUF=models/minicpm5-2b-base-q4_k_m.gguf \
    eval/compare.sh runs/gguf/snap-2b-q4_k_m.gguf run1
python3 eval/paired.py eval/results/run1
```

## Evaluation

`eval/compare.sh` runs the whole battery for each model:

- `snap evaluate` on `../eval/cases.jsonl` and the dev/holdout case files;
- `snap bench`;
- the typed-decisions test split, in server mode.

Every report is create-only. `eval/paired.py` pairs base and fine-tune per
case and reports the accuracy delta with a bootstrap CI, McNemar's test,
abstention deltas and latency. `../eval/history.py` prints the timeline of
every run.

What gates a release:

- accuracy, balanced accuracy, ECE, Brier;
- abstention precision and recall, and committed-gold accuracy with the
  abstain slot on;
- stability under the automatic perturbations;
- latency per decision.

All of these are compared paired against the base model. On typed-decisions,
the gate also looks at per-question argmax constancy (`score --detail`): a
model that repeats one answer across 100 different states is predicting the
prior, not reading.

The base runs as `zero-shot` there, and a fine-tune that saw the train split
runs as `in-domain`. The two kinds are never compared as if they were the
same claim. A GGUF gets a row in `src/models.rs::MODELS` only after this
gate and `snap calibrate`.
