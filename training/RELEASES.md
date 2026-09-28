# Releases

Published weights are tags over training runs. A run lives in
`runs/<name>/` (adapter, logs, meta.json, gguf/); only runs listed here
with a `snap*` name ship on Hugging Face. The name lives inside the GGUF
(`general.name`), so `snap serve` reports e.g. `snap1-2b-15` — rerunning
a training never produces the public name until it is tagged here.

| name | run | repo | quant files | key results |
|---|---|---|---|---|
| snap1-2b | runs/run2-clean | logitlab/snap1-2b-GGUF | q4_k_m, q8_0, bf16 | TD 0.624 zero-shot; exam 85.1%; holdout 71.3% |

## snap1-2b (run2-clean, 2025-09-28)

- `snap1-2b-q4_k_m.gguf` sha256 `fbf24e74a4a79ee9063fb000186eedc712039b6fd1533de4cb24c89491de611e` (pre-rename hash; renamed copy differs only in header metadata)
- `snap1-2b-q8_0.gguf` — see `runs/run2-clean/SHA256SUMS` (same caveat)
- trained on the v2 corpus minus all `tdc-*` rows; typed-decisions
  score is workflow-unseen (verified zero state overlap, train+test)
- eval reports: `training/eval/results/v2/` (`*-solo.*` files)
- calibration: `training/eval/results/v2/calibrations/ft-B.json`
  (bound to the pre-rename id `snap-ft-2b-v2clean-merged-hf-15`;
  recalibrate under the public name before shipping)

Run history: `run1` = first fine-tune (never published);
`run2-full` = same corpus including typed-decisions train rows
(in-domain TD 0.704 — reference only, not publishable as zero-shot);
`run2-clean` = the publishable one.
