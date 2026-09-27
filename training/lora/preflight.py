#!/usr/bin/env python3
"""Pre-train gate — aborts the pod run if any check fails.

  python3 lora/preflight.py --train data/train/v2/final-train.jsonl \
      --labeled data/labeled/contrastive.jsonl data/labeled/oob-numeric.jsonl

Checks, in order:
  1. counts: contrastive families and oob rows meet minimums
  2. zero near-dup overlap of new-case states vs eval cases AND the
     typed-decisions test split (same banded-shingle rule as validate.py)
  3. exported row sanity: every train row has a prompt, token ids, a
     normalized target on the letter alphabet, and a gold letter
  4. group atomicity: no chain/state_key group straddles folds
Exits non-zero on the first failure.
"""
import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "prepare"))
import validate


def load_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def state_str(s):
    return validate.state_text(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev")
    ap.add_argument("--holdout")
    ap.add_argument("--labeled", nargs="+", required=True)
    ap.add_argument("--eval", default="../eval/cases.jsonl")
    ap.add_argument("--td-test", default="../eval/typed-decisions/test.jsonl")
    ap.add_argument("--min-families", type=int, default=800)
    ap.add_argument("--min-oob", type=int, default=900)
    args = ap.parse_args()

    # 1. counts
    fams, oob = set(), 0
    new_cases = []
    for f in args.labeled:
        for c in load_jsonl(f):
            new_cases.append(c)
            if c.get("chain", "").startswith("ctr-"):
                fams.add(c["chain"])
            elif c["id"].startswith("oob-"):
                oob += 1
    if len(fams) < args.min_families or oob < args.min_oob:
        sys.exit(f"FAIL counts: families={len(fams)} oob={oob}")

    # 2. overlap: banded shingles of new states vs eval + TD test states
    eval_states = validate.load_eval_states(args.eval)
    if Path(args.td_test).exists():
        for c in load_jsonl(args.td_test):
            s = c["state"] if isinstance(c.get("state"), str) \
                else json.dumps(c.get("state"), ensure_ascii=False)
            eval_states.append((validate.shingles(s), validate.norm_text(s)))
    bad = []
    for c in new_cases:
        txt = state_str(c["state"])
        sh = validate.shingles(txt)
        if not sh:
            continue
        for esh, enorm in eval_states:
            j = len(sh & esh) / max(1, len(sh | esh))
            if j >= 0.6 or validate.norm_text(txt) == enorm:
                bad.append(c["id"])
                break
    if bad:
        sys.exit(f"FAIL overlap: {len(bad)} new cases touch eval/TD "
                 f"(first: {bad[:5]})")

    # 3. export sanity on train rows
    n = 0
    for r in load_jsonl(args.train):
        t = r.get("target") or {}
        if not r.get("prompt") or not r.get("token_ids"):
            sys.exit(f"FAIL row {r.get('id')}: empty prompt/token_ids")
        if abs(sum(t.values()) - 1.0) > 0.01:
            sys.exit(f"FAIL row {r.get('id')}: target sums "
                     f"{sum(t.values()):.3f}")
        if r.get("gold_letter") not in (r.get("letters") or []):
            sys.exit(f"FAIL row {r.get('id')}: gold letter not in letters")
        if len(r.get("letters") or []) > 26:
            sys.exit(f"FAIL row {r.get('id')}: >26 letters")
        n += 1

    # 4. group atomicity across folds (recomputed on the split files)
    if args.dev and args.holdout:
        folds = {}
        for fold, p in (("train", args.train), ("dev", args.dev),
                        ("holdout", args.holdout)):
            for r in load_jsonl(p):
                g = f"chain:{r['chain']}" if r.get("chain") \
                    else f"state:{r['state_key']}"
                if g in folds and folds[g] != fold:
                    sys.exit(f"FAIL group {g} in {folds[g]} AND {fold}")
                folds[g] = fold

    print(f"PASS: families={len(fams)} oob={oob} new_cases={len(new_cases)} "
          f"train_rows={n} groups={len(set(list(folds))) if args.dev else '-'}")


if __name__ == "__main__":
    main()
