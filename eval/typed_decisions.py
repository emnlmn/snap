#!/usr/bin/env python3
"""LocalLLaMA/typed-decisions (test split) against a running snap server.

    python3 eval/typed_decisions.py fetch       # test + train at the pinned revision
    python3 eval/typed_decisions.py baselines   # Uniform + Prior, checked against the card
    python3 eval/typed_decisions.py answer --url http://127.0.0.1:8018 --kind zero-shot \\
        --out results/typed-decisions/minicpm5-2b.json
    python3 eval/typed_decisions.py score results/typed-decisions/*.json \\
        [--against results/typed-decisions/minicpm5-2b.json] [--detail]

400 cases, 2,000 decisions: five questions over one shared state. `answer`
replays every row unchanged as ONE /v1/systemone request (`state` +
`questions`), the protocol of the dataset card. Gold is the
mean of three samples from a ~4B teacher: a score measures agreement with that
teacher, not correctness, and the card reads ~0.75 as saturation.

The card names its metrics but not their formulas. KL (1e-6 floor), TV and
Brier (summed over classes) reproduce the card's Uniform row exactly, and
`baselines` exits non-zero if they drift. Accuracy is the argmax against the
gold label, so it compares with the card by construction (ties are
negligible on a real model). The card's ECE, soft accuracy and macro-F1
formulas can't be recovered: its Uniform row looks tie-broken at random
(expected accuracy 0.3175, card 0.308) and its Prior row doesn't reproduce
under label counts, add-one or mean distributions. Read those three against
other runs of this script, not against the card; `score` leaves Jev's ECE blank.

`--kind` is required because the card's two modes are not comparable:
zero-shot means the model never trained on these four workflows, in-domain
means it trained on this dataset's train split. Serve without --calibration
for raw numbers; latency is client-side, end to end, like the card's.

`score --against BASE` pairs every other run with BASE decision by decision:
accuracy/KL/Brier deltas with a 95% bootstrap interval over cases (the five
questions of a case are not independent) and an exact McNemar test.

Stdlib only, except `fetch`, which needs pyarrow to read the parquet.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
import ssl
import statistics
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "typed-decisions"
REPO = "LocalLLaMA/typed-decisions"
# the card revision that carries the measured Jev 1.13.0 row (2026-09-18)
REVISION = "f7a2487edd7a043a5441a5e9ccc7fe5ddbd9ebe8"
EPS = 1e-6  # probability floor inside KL; the card does not say which one it uses
TYPES = ("noul", "choice", "score")

# the card's test-split rows. Uniform's distribution metrics are the formula
# check; Jev's ECE is omitted because the card's ECE formula is unknown.
CARD = {
    "uniform": {"accuracy": 0.308, "soft_accuracy": 0.311, "macro_f1": 0.152,
                "kl": 0.444, "tv": 0.381, "brier": 0.238, "ece": 0.169},
    "prior": {"accuracy": 0.470, "soft_accuracy": 0.430, "macro_f1": 0.207,
              "kl": 0.347, "tv": 0.317, "brier": 0.189, "ece": 0.088},
    "jev-1.13.0": {"accuracy": 0.727, "soft_accuracy": 0.580, "macro_f1": 0.613,
                   "kl": 1.442, "tv": 0.251, "brier": 0.148,
                   "score_mae": 0.391, "within_1_level": 0.952},
}
VERIFIED = ("kl", "tv", "brier")  # reproduced on Uniform, independent of tie-breaking

try:  # python.org macOS builds don't see the system keychain — certifi does
    import certifi

    CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    CTX = ssl.create_default_context()


# ---------------------------------------------------------------- data

def fetch(args) -> None:
    import pyarrow.parquet as pq

    hub = Path(os.environ.get("HF_HUB_CACHE")
               or Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub")
    snap = hub / f"datasets--{REPO.replace('/', '--')}" / "snapshots" / REVISION / "all"
    DATA.mkdir(parents=True, exist_ok=True)
    for split in ("test", "train"):
        name = f"{split}-00000-of-00001.parquet"
        if (snap / name).is_file():  # already in the local HF cache at this revision
            raw = (snap / name).read_bytes()
        else:
            url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/all/{name}"
            req = urllib.request.Request(url, headers={"User-Agent": "snap-eval/1.0"})
            raw = urllib.request.urlopen(req, timeout=120, context=CTX).read()
        rows = pq.read_table(io.BytesIO(raw),
                             columns=["id", "workflow", "state", "questions", "gold"]).to_pylist()
        with open(DATA / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  {split}: {len(rows)} cases -> {DATA / f'{split}.jsonl'}")


def load(split: str) -> list[dict]:
    path = DATA / f"{split}.jsonl"
    if not path.is_file():
        raise SystemExit(f"{path} missing — run `fetch` first")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def labels(q: dict) -> list[str]:
    """Gold label keys in a fixed order; argmax ties go to the first."""
    if q["type"] == "noul":
        return ["false", "true"]
    if q["type"] == "choice":
        return list(q["criteria"])
    return [str(i) for i in range(len(q["criteria"]))]


# ---------------------------------------------------------------- metrics

def decisions(rows: list[dict], preds: dict) -> list[dict]:
    """One entry per answered decision. `preds[case][question]` is a
    distribution over that question's gold label keys."""
    out = []
    for row in rows:
        gold = json.loads(row["gold"])
        for name, q in json.loads(row["questions"]).items():
            p = preds.get(row["id"], {}).get(name)
            if p is None:
                continue
            keys = labels(q)
            g = {k: float(gold[name]["probabilities"].get(k, 0.0)) for k in keys}
            p = {k: float(p.get(k, 0.0)) for k in keys}
            guess = max(keys, key=p.get)
            clip = {k: max(p[k], EPS) for k in keys}
            z = sum(clip.values())
            e = {
                "case": row["id"],
                "question": f"{row['workflow']}/{name}",
                "type": q["type"],
                "gold": gold[name]["label"],
                "guess": guess,
                "correct": guess == gold[name]["label"],
                "soft": g[guess],
                "peak": p[guess],
                "kl": sum(g[k] * math.log(g[k] / (clip[k] / z)) for k in keys if g[k] > 0),
                "tv": 0.5 * sum(abs(g[k] - p[k]) for k in keys),
                "brier": sum((g[k] - p[k]) ** 2 for k in keys),  # summed over classes
            }
            if q["type"] == "score":
                e["mae"] = abs(sum(int(k) * p[k] for k in keys) - gold[name]["score"])
            out.append(e)
    return out


def summary(es: list[dict]) -> dict:
    bins = defaultdict(list)
    for e in es:
        bins[min(int(e["peak"] * 10), 9)].append(e)
    mean = statistics.fmean
    s = {
        "decisions": len(es),
        "accuracy": mean(e["correct"] for e in es),
        "soft_accuracy": mean(e["soft"] for e in es),
        "kl": mean(e["kl"] for e in es),
        "tv": mean(e["tv"] for e in es),
        "brier": mean(e["brier"] for e in es),
        "ece": sum(len(b) / len(es) * abs(mean(e["peak"] for e in b) - mean(e["correct"] for e in b))
                   for b in bins.values()),
    }
    scored = [e for e in es if "mae" in e]
    if scored:
        s["score_mae"] = mean(e["mae"] for e in scored)
        s["within_1_level"] = mean(e["mae"] <= 1 for e in scored)
    return s


def macro_f1(es: list[dict]) -> float:
    """F1 averaged over each question's labels, then over questions."""
    by_q = defaultdict(list)
    for e in es:
        by_q[e["question"]].append(e)
    f1s = []
    for q_es in by_q.values():
        per = []
        for k in sorted({e["gold"] for e in q_es} | {e["guess"] for e in q_es}):
            tp = sum(e["gold"] == k and e["guess"] == k for e in q_es)
            fp = sum(e["gold"] != k and e["guess"] == k for e in q_es)
            fn = sum(e["gold"] == k and e["guess"] != k for e in q_es)
            if tp + fp + fn:
                per.append(2 * tp / (2 * tp + fp + fn))
        f1s.append(statistics.fmean(per))
    return statistics.fmean(f1s)


def report(es: list[dict]) -> dict:
    return {
        "overall": summary(es) | {"macro_f1": macro_f1(es)},
        "by_type": {t: summary([e for e in es if e["type"] == t]) for t in TYPES
                    if any(e["type"] == t for e in es)},
        "by_workflow": {w: summary([e for e in es if e["question"].startswith(w + "/")])
                        for w in sorted({e["question"].split("/")[0] for e in es})},
        "by_question": {q: summary([e for e in es if e["question"] == q])
                        for q in sorted({e["question"] for e in es})},
    }


def baselines(args) -> None:
    test, train = load("test"), load("train")
    freq = defaultdict(Counter)
    for row in train:
        for name, a in json.loads(row["gold"]).items():
            freq[(row["workflow"], name)][a["label"]] += 1
    uniform, prior = defaultdict(dict), defaultdict(dict)
    for row in test:
        for name, q in json.loads(row["questions"]).items():
            keys = labels(q)
            c = freq[(row["workflow"], name)]
            uniform[row["id"]][name] = dict.fromkeys(keys, 1 / len(keys))
            prior[row["id"]][name] = {k: c[k] / sum(c.values()) for k in keys}
    drift = []
    for name, preds in (("uniform", uniform), ("prior", prior)):
        got = report(decisions(test, preds))["overall"]
        print(f"{name:8s} " + "  ".join(f"{m} {got[m]:.3f} (card {want:.3f})"
                                        for m, want in CARD[name].items()))
        if name == "uniform":
            drift += [f"uniform.{m}: ours {got[m]:.4f}, card {CARD[name][m]:.3f}"
                      for m in VERIFIED if abs(got[m] - CARD[name][m]) > 0.0005 + 1e-9]
    if drift:  # the card rounds to 3 decimals
        raise SystemExit("KL/TV/Brier drift from the card:\n  " + "\n  ".join(drift))
    print("uniform KL, TV and Brier match the card: the distribution metrics are its "
          "formulas.\naccuracy/soft/F1/ECE of a flat guess are tie-breaking, and the card's "
          "Prior row is not reproducible — both are shown for reference only.")


# ---------------------------------------------------------------- answer

def post(url: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        f"{url.rstrip('/')}/v1/systemone",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def distribution(q: dict, ans: dict) -> dict:
    """snap's answer -> distribution over the gold label keys. A shape that
    doesn't map is a broken contract, not a wrong answer: fail loudly."""
    if q["type"] == "noul":
        p = float(ans["noul"])  # P(yes), the Jev field
        return {"false": 1.0 - p, "true": p}
    probs = ans.get("probabilities") or {}
    if q["type"] == "choice":
        keys = list(q["criteria"])
    else:  # score: snap keys each level by its description text
        keys = [c if isinstance(c, str) else json.dumps(c) for c in q["criteria"]]
        if len(set(keys)) != len(keys):
            raise SystemExit(f"score with duplicate level texts: {keys}")
    missing = [k for k in keys if k not in probs]
    if missing:
        raise SystemExit(f"answer has no probability for {missing}: {ans}")
    return {lab: float(probs[k]) for lab, k in zip(labels(q), keys)}


def answer(args) -> None:
    rows = load("test")
    if args.limit:  # evenly spaced: the file is grouped by workflow
        rows = rows[:: max(1, len(rows) // args.limit)][: args.limit]
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists (reports are create-only)")
    out.parent.mkdir(parents=True, exist_ok=True)

    def body(row):
        b = {"state": json.loads(row["state"]), "questions": json.loads(row["questions"])}
        return b | ({"layout": args.layout} if args.layout != "auto" else {})

    post(args.url, body(rows[0]), args.timeout)  # warm-up, not timed
    started, t0 = datetime.now(timezone.utc).isoformat(timespec="seconds"), time.time()
    cases, model, ms = [], None, []
    for i, row in enumerate(rows, 1):
        t = time.perf_counter()
        try:
            res = post(args.url, body(row), args.timeout)
        except urllib.error.HTTPError as e:  # the engine refused this case: record it
            cases.append({"id": row["id"], "error": f"{e.code} {e.read()[:300].decode(errors='replace')}"})
            continue
        ms.append((time.perf_counter() - t) * 1000)
        model = model or res.get("model")
        answers = {}
        for name, q in json.loads(row["questions"]).items():
            if name not in res["answers"]:
                raise SystemExit(f"{row['id']}: no answer for {name!r}")
            answers[name] = distribution(q, res["answers"][name])
        cases.append({
            "id": row["id"],
            "ms": round(ms[-1], 1),
            "input_tokens": res.get("usage", {}).get("input_tokens"),
            "layout": res.get("x_snap", {}).get("layout"),
            "answers": answers,
        })
        if i % 50 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)} cases, p50 {statistics.median(ms):.0f} ms", flush=True)

    run = {
        "benchmark": REPO, "revision": REVISION, "split": "test",
        "kind": args.kind, "note": args.note, "url": args.url, "model": model,
        "layout": args.layout, "started": started, "seconds": round(time.time() - t0, 1),
        "cases": cases,
    }
    with open(out, "x", encoding="utf-8") as f:
        json.dump(run, f, indent=1, ensure_ascii=False)
    errors = sum("error" in c for c in cases)
    print(f"wrote {out} — {len(cases) - errors} cases answered, {errors} refused")


# ---------------------------------------------------------------- score

def preds_of(run: dict) -> dict:
    return {c["id"]: c["answers"] for c in run["cases"] if "answers" in c}


def paired(rows: list[dict], base: dict, run: dict, boot: int) -> str:
    """Deltas run − base on the decisions both answered, 95% bootstrap over
    cases, exact two-sided McNemar on accuracy."""
    a = {(e["case"], e["question"]): e for e in decisions(rows, preds_of(base))}
    b = {(e["case"], e["question"]): e for e in decisions(rows, preds_of(run))}
    common = a.keys() & b.keys()
    per_case = defaultdict(lambda: defaultdict(float))  # case -> metric -> summed delta
    count = Counter()
    for k in common:
        count[k[0]] += 1
        for m in ("correct", "kl", "brier"):
            per_case[k[0]][m] += float(b[k][m]) - float(a[k][m])
    cases = sorted(count)
    rng = random.Random(0)
    cells = []
    for m, label in (("correct", "accuracy"), ("kl", "KL"), ("brier", "Brier")):
        def delta(sample):
            return sum(per_case[c][m] for c in sample) / sum(count[c] for c in sample)
        boots = sorted(delta([rng.choice(cases) for _ in cases]) for _ in range(boot))
        cells.append(f"Δ{label} {delta(cases):+.3f} [{boots[int(0.025 * boot)]:+.3f}, "
                     f"{boots[int(0.975 * boot) - 1]:+.3f}]")
    worse = sum(a[k]["correct"] and not b[k]["correct"] for k in common)
    better = sum(b[k]["correct"] and not a[k]["correct"] for k in common)
    n = worse + better
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(min(worse, better) + 1)) / 2 ** n) if n else 1.0
    return (" · ".join(cells)
            + f" · McNemar p={p:.3g} ({better} fixed, {worse} broken) on {len(common)} decisions")


def score(args) -> None:
    rows = load("test")
    total = sum(len(json.loads(r["questions"])) for r in rows)
    runs = {Path(f).stem: json.loads(Path(f).read_text(encoding="utf-8")) for f in args.files}
    hdr = (f"{'run':<28} {'kind':<10} {'answered':>10} {'acc':>6} {'soft':>6} {'F1':>6} "
           f"{'KL':>6} {'TV':>6} {'Brier':>6} {'ECE':>6} {'MAE':>6} {'±1':>6} {'p50 ms':>7}")
    print(hdr + "\n" + "-" * len(hdr))

    def line(name, kind, answered, o, p50):
        cells = [o.get(m) for m in ("accuracy", "soft_accuracy", "macro_f1", "kl", "tv",
                                    "brier", "ece", "score_mae", "within_1_level")]
        print(f"{name:<28} {kind:<10} {answered:>10} "
              + " ".join(f"{c:6.3f}" if c is not None else f"{'–':>6}" for c in cells)
              + f" {p50:>7}")

    reports = {}
    for name, run in runs.items():
        es = decisions(rows, preds_of(run))
        reports[name] = r = report(es)
        ms = [c["ms"] for c in run["cases"] if "ms" in c]
        line(name, run["kind"], f"{len(es)}/{total}", r["overall"],
             f"{statistics.median(ms):.0f}" if ms else "–")
    line("jev-1.13.0 (card)", "zero-shot", f"{total}/{total}", CARD["jev-1.13.0"], "710")

    print("\naccuracy by type and workflow")
    for name, r in reports.items():
        parts = [f"{k} {v['accuracy']:.3f}" for k, v in (r["by_type"] | r["by_workflow"]).items()]
        print(f"  {name:<26} " + "  ".join(parts))
    if args.detail:
        for name, r in reports.items():
            print(f"\n{name}: accuracy per question (the card's ceilings range 0.56–0.94)")
            for q, s in r["by_question"].items():
                print(f"  {q:<44} {s['accuracy']:.3f}  (n={s['decisions']})")
    if args.against:
        base_name = Path(args.against).stem
        base = json.loads(Path(args.against).read_text(encoding="utf-8"))
        print(f"\npaired against {base_name}")
        for name, run in runs.items():
            if name != base_name:
                print(f"  {name}: {paired(rows, base, run, args.boot)}")
    kinds = sorted({run["kind"] for run in runs.values()})
    if len(kinds) > 1:
        print(f"\nnote: runs mix kinds {kinds} — the card reads them as not comparable")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch")
    sub.add_parser("baselines")
    a = sub.add_parser("answer")
    a.add_argument("--url", default="http://127.0.0.1:8018")
    a.add_argument("--out", required=True, help="create-only JSON with every case's distributions")
    a.add_argument("--kind", required=True, choices=["zero-shot", "in-domain"],
                   help="in-domain = the model trained on this dataset's train split")
    a.add_argument("--note", default="", help="free text kept in the report (GGUF, quant, run id)")
    a.add_argument("--layout", default="auto",
                   choices=["auto", "state_first", "question_first", "header", "catalog"])
    a.add_argument("--limit", type=int, default=0,
                   help="N evenly spaced cases, every workflow covered (smoke runs)")
    a.add_argument("--timeout", type=float, default=300)
    s = sub.add_parser("score")
    s.add_argument("files", nargs="+", help="answer files from `answer`")
    s.add_argument("--against", help="a run to pair the others with (e.g. the base model)")
    s.add_argument("--detail", action="store_true", help="accuracy per question")
    s.add_argument("--boot", type=int, default=2000, help="bootstrap resamples")
    args = ap.parse_args()
    {"fetch": fetch, "baselines": baselines, "answer": answer, "score": score}[args.cmd](args)


if __name__ == "__main__":
    main()
