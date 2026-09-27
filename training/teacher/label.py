"""Teacher soft-labels for generated cases.

Runs each case through snap server(s) hosting a teacher model
via POST /v1/systemone, captures the letter-surface
probabilities, and merges them with the case's `expect` into a training
target:

  label==argmax(teacher)  -> "agree": soft target = anchor mass on the
                             verified label + teacher shape on the rest
  disagree                -> "review": flatter target — the disagreement
                             IS the ambiguity signal (calibration)

Response-shape facts (verified live against snap):
  choice  -> probabilities keyed by criteria keys + "__abstain__"
  score   -> probabilities keyed by level DESCRIPTION text
  numeric -> probabilities keyed by anchor values ("4.2857…") +
             "__below__"/"__above__"; expect.value maps to nearest anchor
  boolean/noul -> NO probabilities map: scalar `noul` = p(yes),
             `status` carries the abstain outcome
  probabilities are keyed SEMANTICALLY (criteria keys), not by letter —
  keys "A".."D" appear only when criteria keys literally are letters.

Multiple snap serve instances can teach in parallel: pass several
--server URLs (each instance decodes serially under its own lock).

Usage:
  snap serve --model <teacher> --port 8018 &
  python3 teacher/label.py data/splits/layerA-clean.jsonl \
      --out data/labeled/layerA.jsonl \
      --server http://127.0.0.1:8018 --workers 6
"""

import argparse
import itertools
import json
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SERVER = "http://127.0.0.1:8018"
LAYA = "http://localhost:8710"


def decide(server: str, state, qname: str, q: dict, timeout: int = 180) -> dict:
    body = json.dumps({
        "state": state,
        "questions": {qname: q},
        "temperature": 1.0,
        "mode": "shared",
    }).encode()
    req = urllib.request.Request(
        f"{server}/v1/systemone", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["answers"][qname]


def numeric_anchors(q: dict) -> list[float]:
    """Anchor values exactly as schema.rs `anchors()` computes them:
    step wins over granularity when present, clamped to 2..24."""
    mn, mx = float(q["min"]), float(q["max"])
    st = q.get("step")
    if st is not None and float(st) > 0:
        n = int(round((mx - mn) / float(st))) + 1
    else:
        n = int(q.get("granularity", 8))
    n = max(2, min(24, n))
    return [mn + (mx - mn) * i / (n - 1) for i in range(n)]


def fmt_rust(v: float) -> str:
    """Rust `format!("{v}")` for f64: integer-valued floats print without
    the fraction ("40"), others use shortest round-trip like repr()."""
    return str(int(v)) if v == int(v) else repr(v)


def expected_key(case: dict) -> tuple[str | None, float]:
    """Map expect to the probability key snap emits, plus target anchor mass."""
    q, exp = case["question"], case["expect"]
    if exp.get("status") == "abstained":
        return "__abstain__", 0.85
    t = q["type"]
    if t == "choice":
        return exp.get("choice"), 0.80
    if t in ("boolean", "noul"):
        return ("yes" if exp.get("boolean") else "no"), 0.80
    if t == "score":
        lv = exp.get("level")
        crit = q.get("criteria") or []
        if isinstance(lv, int) and 0 <= lv < len(crit):
            return str(crit[lv]), 0.75
    if t == "numeric":
        v = exp.get("value")
        if isinstance(v, (int, float)):
            a = numeric_anchors(q)
            nearest = min(a, key=lambda x: abs(x - v))
            return fmt_rust(nearest), 0.75
    return None, 0.0


def teacher_probs(case: dict, ans: dict) -> dict:
    """Uniform prob-map view of an answer across all types.
    Prefer the `probabilities` map — it carries `__abstain__` mass when the
    slot exists; the bare `noul` scalar cannot express it."""
    probs = dict(ans.get("probabilities") or {})
    t = case["question"]["type"]
    if t in ("boolean", "noul"):
        if probs:
            return probs
        p = ans.get("noul")
        return {} if p is None else {"yes": float(p), "no": 1.0 - float(p)}
    return probs


def teacher_argmax(case: dict, ans: dict, probs: dict) -> str | None:
    if ans.get("status") == "abstained":
        return "__abstain__"
    t = case["question"]["type"]
    if t in ("boolean", "noul"):
        if probs:
            return max(probs, key=probs.get)
        p = ans.get("noul")
        return None if p is None else ("yes" if p >= 0.5 else "no")
    if t == "choice":
        return ans.get("choice")
    if t == "score":
        return None if ans.get("level") is None else (
            str((case["question"].get("criteria") or [])[ans["level"]]))
    return max(probs, key=probs.get) if probs else None


def merge_target(probs: dict, key: str | None, anchor: float) -> dict:
    """Soft target: `anchor` mass on key, teacher shape on the rest."""
    if not probs:
        return {}
    if key is None or key not in probs:
        return dict(probs)
    rest = {k: v for k, v in probs.items() if k != key}
    s = sum(rest.values())
    if s <= 0:
        return {key: 1.0}
    return {key: anchor, **{k: v / s * (1 - anchor) for k, v in rest.items()}}


def laya_predict(url: str, state, q: dict, timeout: int = 60) -> dict | None:
    """laya /predict — same question shape minus abstain/numeric (unsupported);
    boolean folds into noul."""
    qq = {k: v for k, v in q.items() if k != "allow_abstain"}
    if qq["type"] == "boolean":
        qq["type"] = "noul"
    if qq["type"] == "numeric" or q.get("allow_abstain"):
        return None
    body = json.dumps({"state": state, "questions": {"q": qq}}).encode()
    req = urllib.request.Request(f"{url}/predict", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["answers"]["q"]


def laya_probs(case: dict, ans: dict) -> dict:
    """laya answer -> prob map on the shared semantic key space
    (score indices become level text, noul becomes yes/no)."""
    t = case["question"]["type"]
    if t in ("boolean", "noul"):
        p = ans.get("noul")
        return {} if p is None else {"yes": float(p), "no": 1.0 - float(p)}
    probs = dict(ans.get("probabilities") or {})
    if t == "score":
        crit = case["question"].get("criteria") or []
        out = {}
        for k, v in probs.items():
            if k.isdigit() and int(k) < len(crit):
                out[str(crit[int(k)])] = v
            else:
                out[k] = v
        return out
    return probs


def laya_argmax(case: dict, ans: dict, probs: dict) -> str | None:
    t = case["question"]["type"]
    if t in ("boolean", "noul"):
        p = ans.get("noul")
        return None if p is None else ("yes" if p >= 0.5 else "no")
    if t == "score":
        lv = ans.get("level")
        crit = case["question"].get("criteria") or []
        if isinstance(lv, int) and 0 <= lv < len(crit):
            return str(crit[lv])
    c = ans.get("choice")
    if c:
        return c
    return max(probs, key=probs.get) if probs else None


def label_one(case: dict, servers_cycle, laya_url: str | None) -> dict:
    server = next(servers_cycle)
    q = case["question"]
    key, anchor = expected_key(case)
    rec = dict(case)
    voices = {}

    ans = decide(server, case["state"], "q", q)
    qp = teacher_probs(case, ans)
    qt = teacher_argmax(case, ans, qp)
    voices["snap"] = {"probs": qp, "argmax": qt, "agree": qt == key,
                      "confidence": ans.get("confidence"),
                      "status": ans.get("status")}

    if laya_url:
        try:
            la = laya_predict(laya_url, case["state"], q)
            if la is not None:
                lp = laya_probs(case, la)
                lt = laya_argmax(case, la, lp)
                voices["laya"] = {"probs": lp, "argmax": lt,
                                  "agree": lt == key}
        except Exception:
            pass  # laya is a bonus voice — never block on it

    rec["teacher"] = dict(voices["snap"])
    if "laya" in voices:
        rec["teacher"]["laya"] = voices["laya"]
    agreeing = [v for v in voices.values() if v["agree"]]
    rec["teacher"]["voices_agree"] = len(agreeing)
    rec["teacher"]["voices_total"] = len(voices)
    rec["teacher"]["agree"] = len(agreeing) == len(voices)

    # shape: average the distributions of voices that endorse the gold key;
    # contested rows keep a weak prior (export drops their shape anyway)
    shape = {}
    if agreeing:
        for v in agreeing:
            for k, p in v["probs"].items():
                shape[k] = shape.get(k, 0.0) + p / len(agreeing)
    else:
        shape = voices["snap"]["probs"]
    rec["target"] = merge_target(
        shape, key, anchor if rec["teacher"]["agree"] else min(anchor, 0.6))
    return rec


def label_one_safe(case, cycle, laya_url=None):
    try:
        return label_one(case, cycle, laya_url)
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--server", action="append", default=None,
                    help="snap server URL; repeat for parallel instances")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--laya", default=None,
                    help=f"laya server URL for the second voice (e.g. {LAYA})")
    ap.add_argument("--resume", action="store_true",
                    help="append to --out, skipping ids already labeled")
    args = ap.parse_args()
    servers = args.server or [SERVER]

    done_ids = set()
    if args.resume and os.path.exists(args.out):
        # a crash mid-write leaves a torn last line — compact the file to
        # valid rows only, or every downstream json.loads reader breaks
        good = []
        torn = 0
        for l in open(args.out):
            if not l.strip():
                continue
            try:
                done_ids.add(json.loads(l)["id"])
                good.append(l if l.endswith("\n") else l + "\n")
            except json.JSONDecodeError:
                torn += 1
        if torn:
            with open(args.out, "w") as fh:
                fh.writelines(good)
            print(f"resume: dropped {torn} torn line(s), file compacted")
        print(f"resume: {len(done_ids)} already labeled")

    stats = {"agree": 0, "review": 0, "error": 0}
    mode = "a" if args.resume else "x"
    with open(args.out, mode) as out:
        for f in args.files:
            all_cases = [json.loads(l) for l in open(f)
                         if l.strip() and not l.startswith("#")]
            cases = [c for c in all_cases if c["id"] not in done_ids]
            if len(all_cases) != len(cases):
                print(f"  {Path(f).name}: skip {len(all_cases)-len(cases)} done",
                      flush=True)
            done_ids.update(c["id"] for c in cases)
            cycle = itertools.cycle(servers)
            done = 0
            work = lambda c: label_one_safe(c, cycle, args.laya)
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                for rec in ex.map(work, cases):
                    t = rec.get("teacher") if rec else None
                    if t:
                        stats["agree" if t["agree"] else "review"] += 1
                        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    else:
                        stats["error"] += 1
                    done += 1
                    if done % 500 == 0:
                        out.flush()
                        print(f"  {Path(f).name}: {done}/{len(cases)} {stats}",
                              flush=True)
            print(f"{Path(f).name}: done {stats}", flush=True)
    print("done:", stats)


if __name__ == "__main__":
    main()
