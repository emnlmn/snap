# Releases

Published weights are tags over training runs. A run lives in
`runs/<name>/` (adapter, logs, meta.json, gguf/); only runs listed here
with a `snap*` name ship on Hugging Face. The name lives inside the GGUF
(`general.name`), so `snap serve` reports e.g. `snap1-2b-15` — rerunning
a training never produces the public name until it is tagged here.

| name | run | repo | quant files | key results |
|---|---|---|---|---|
| snap1-2b | runs/run2-clean | logitlab/snap1-2b-GGUF | q4_k_m, q8_0, bf16 | TD 0.624 zero-shot; exam 85.0%; holdout 71.3% |

## snap1-2b (run2-clean, 2026-09-28)

- sha256 of all three published files: `runs/run2-clean/gguf/SHA256SUMS`
  (hashes cover the renamed files — `general.name` = "snap1 2B")
- trained on the v2 corpus minus all `tdc-*` rows; typed-decisions
  score is workflow-unseen (verified zero state overlap, train+test)
- eval reports: `training/eval/results/v2/` (`*-solo.*` files)
- calibration: `runs/run2-clean/calibration-dev.json` — **not shipped**:
  fitted on the internal dev set, it sharpens probabilities and worsened
  TD ECE 0.043→0.087 (temperatures <1 on an already-calibrated domain).
  Ship raw; users calibrate on their own data with `snap calibrate`

Run history: `run1` = first fine-tune (never published);
`run2-full` = same corpus including typed-decisions train rows
(in-domain TD 0.704 — reference only, not publishable as zero-shot);
`run2-clean` = the publishable one.
