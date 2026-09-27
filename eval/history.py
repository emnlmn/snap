#!/usr/bin/env python3
"""Regression timeline over every eval artifact on disk — history.py scans
the results dirs and prints one line per run, newest last, so 'are we
getting better' is a table, not a hunch. Read-only: never writes.

Suites recognized:
  eval    snap evaluate reports   (model, accuracy, ece, brier, ms/decision)
  td      typed-decisions answers (model, kind, acc, KL — gold-scored)
  bench   snap bench reports      (p50/p95/req_s on the default scenario)
  other   anything else           (listed by name so nothing hides)

Scans snap-rs/results/ and snap-ft/eval/results/ when they exist; extra
dirs can be appended as argv.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from typed_decisions import decisions, load as td_load  # noqa: E402

ROOTS = [HERE.parent / "results", HERE.parent.parent / "snap-ft" / "eval" / "results"]
ROWS = []


def add(path: Path, suite: str, model: str, metrics: str):
    ROWS.append((datetime.fromtimestamp(path.stat().st_mtime), suite,
                 str(path.relative_to(HERE.parent)), model, metrics))


def scan(path: Path):
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return add(path, "?", "", "unreadable")
    if isinstance(d, list) and d and "scenario" in d[0]:  # bench
        s = next((r for r in d if r.get("scenario") == "single"), d[0])
        return add(path, "bench", "",
                   f"p50 {s.get('p50', 0):.0f}ms p95 {s.get('p95', 0):.0f}ms "
                   f"{s.get('req_s', 0):.1f} req/s")
    if not isinstance(d, dict):
        return add(path, "?", "", "unrecognized")
    if "benchmark" in d and "cases" in d:  # typed-decisions answer file
        rows = td_load("test")
        preds = {c["id"]: c["answers"] for c in d["cases"] if "answers" in c}
        es = decisions(rows, preds)
        s = summary_of(es)
        return add(path, "td", d.get("model", ""),
                   f"{d.get('kind','?'):10} acc {s['accuracy']:.3f} "
                   f"KL {s['kl']:.3f} ECE {s['ece']:.3f}")
    if "accuracy" in d and "rows" in d:  # snap evaluate
        return add(path, "eval", d.get("model", ""),
                   f"acc {d['accuracy']:.3f} ece {d.get('ece', 0):.3f} "
                   f"brier {d.get('brier', 0):.3f} {d.get('ms_mean', 0):.0f}ms")
    if "temperatures" in d:
        e = d.get("ece", {})
        if isinstance(e, dict):
            e = e.get("out_of_fold", e.get("raw", 0))
        return add(path, "cal", d.get("model", ""), f"ece_oof {e:.3f}")
    return add(path, "?", "", "unrecognized")


def summary_of(es):
    import statistics
    mean = statistics.fmean
    bins: dict[int, list] = {}
    for e in es:
        bins.setdefault(min(int(e["peak"] * 10), 9), []).append(e)
    return {
        "accuracy": mean(e["correct"] for e in es),
        "kl": mean(e["kl"] for e in es),
        "ece": sum(len(b) / len(es) * abs(mean(e["peak"] for e in b) - mean(e["correct"] for e in b))
                   for b in bins.values()),
    }


def main():
    roots = [Path(p) for p in sys.argv[1:]] or [r for r in ROOTS if r.exists()]
    for r in roots:
        for f in sorted(r.rglob("*.json")):
            scan(f)
    if not ROWS:
        sys.exit("no reports found")
    ROWS.sort(key=lambda r: r[0])
    print(f"{'date':16} {'suite':5} {'run':52} {'model':26} metrics")
    print("-" * 128)
    for dt, suite, path, model, m in ROWS:
        print(f"{dt:%Y-%m-%d %H:%M} {suite:5} {path:52} {model:26} {m}")


if __name__ == "__main__":
    main()
