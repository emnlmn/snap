#!/usr/bin/env python3
"""Paired base-vs-ft analysis over `snap evaluate` report files.

compare.sh writes <case>-base.json and <case>-ft.json into a results dir;
this pairs rows by case id and reports what raw accuracy can't say:

  - accuracy delta with a paired bootstrap CI (resample case ids)
  - McNemar's exact test on discordant pairs
  - abstention deltas (rate + committed-gold accuracy)
  - latency ratio

Usage:
  python3 eval/paired.py eval/results/run1 [--seed 17]
"""
import argparse
import glob
import json
import os
import random
import sys


def load(path):
    d = json.load(open(path))
    return d, {r["id"]: r for r in d.get("rows", []) if r.get("ok") is not None}


def abstained(r):
    a = r.get("answer") or {}
    return a.get("status") == "abstained" or a.get("choice") == "__abstain__"


def paired_report(base_rows, ft_rows, rng, boot=10000):
    ids = sorted(set(base_rows) & set(ft_rows))
    if not ids:
        return None
    b_ok = [bool(base_rows[i]["ok"]) for i in ids]
    f_ok = [bool(ft_rows[i]["ok"]) for i in ids]
    delta = sum(f_ok) / len(ids) - sum(b_ok) / len(ids)

    # paired bootstrap CI on the accuracy delta
    n = len(ids)
    diffs = [int(f) - int(b) for b, f in zip(b_ok, f_ok)]
    draws = sorted(
        sum(diffs[i] for i in (rng.randrange(n) for _ in range(n))) / n
        for _ in range(boot))
    lo, hi = draws[int(0.025 * boot)], draws[int(0.975 * boot)]

    # McNemar exact (two-sided binomial on discordant pairs)
    b01 = sum(1 for b, f in zip(b_ok, f_ok) if b and not f)
    b10 = sum(1 for b, f in zip(b_ok, f_ok) if f and not b)
    k, n_disc = min(b01, b10), b01 + b10
    p = min(1.0, 2 * sum(_binom(n_disc, i) for i in range(k + 1))) \
        if n_disc else 1.0

    b_ab = sum(abstained(base_rows[i]) for i in ids) / n
    f_ab = sum(abstained(ft_rows[i]) for i in ids) / n
    b_ms = [base_rows[i].get("ms") for i in ids if base_rows[i].get("ms")]
    f_ms = [ft_rows[i].get("ms") for i in ids if ft_rows[i].get("ms")]
    return {
        "n_paired": n,
        "acc_base": round(sum(b_ok) / n, 4),
        "acc_ft": round(sum(f_ok) / n, 4),
        "delta": round(delta, 4),
        "delta_ci95": [round(lo, 4), round(hi, 4)],
        "mcnemar": {"ft_only": b10, "base_only": b01, "p": round(p, 4)},
        "abstain_rate": {"base": round(b_ab, 4), "ft": round(f_ab, 4)},
        "ms_median": {"base": _med(b_ms), "ft": _med(f_ms)},
    }


def _binom(n, k):
    from math import comb
    return comb(n, k) * 0.5 ** n


def _med(xs):
    xs = sorted(xs)
    return round(xs[len(xs) // 2], 1) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    out = {}
    for bf in sorted(glob.glob(os.path.join(args.results_dir, "*-base.json"))):
        name = os.path.basename(bf)[:-len("-base.json")]
        if name in ("td", "bench"):  # td has its own scorer, bench is a row list
            continue
        ff = os.path.join(args.results_dir, f"{name}-ft.json")
        if not os.path.exists(ff):
            print(f"no ft pair for {name}, skipped", file=sys.stderr)
            continue
        _, b = load(bf)
        _, f = load(ff)
        rep = paired_report(b, f, rng)
        if rep:
            out[name] = rep
    if not out:
        sys.exit("no base/ft report pairs found")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
