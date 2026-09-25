#!/usr/bin/env python3
"""TypeSafe's public eval cases (evals.typesafe.ai) against a running snap server.

    python eval/typesafe_public.py fetch                 # download the 4 *-cases.js
    python eval/typesafe_public.py extract               # -> eval/typesafe/full_eval.json
    python eval/typesafe_public.py answer --url http://127.0.0.1:8018 --out OUT.json
    python eval/typesafe_public.py score OUT.json [more.json ...]

fetch+extract reproduce jev-on-a-laptop's evals/full_eval.json locally — the
extraction logic is adapted from its evals/extract_full.py (MIT,
github.com/rorshopping/jev-on-a-laptop). TypeSafe's raw case data is fetched at
runtime, never committed.

`answer` sends each case as ONE /v1/systemone request (all its questions share
the document — the shared-prefix use case) and writes jol-compatible output:
answers[workflow][case_id][qid] = {"raw": value, "kind": type}. noul reports
P(yes) as a float; score reports the argmax level index, matching how the
reference is an argmax over levels.

`score` measures agreement with TypeSafe's reference (argmax of the consensus
distribution) — same canonicalization as jol's evals/score_full.py, so numbers
are comparable across reproductions. Extra answer files (including
published_answers.json rows via --published) join the same table.

Large-document cases need room: serve with `--ctx 32768` or invoice cases will
fail validation. Stdlib only; no dependencies.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import ssl
import urllib.request
import urllib.error
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "typesafe"
FULL = DATA / "full_eval.json"

WORKFLOWS = [
    "security_incidents",
    "agent_trace_observability",
    "invoice_processing",
    "customer_service",
]
LABEL = {
    "security_incidents": "Security",
    "agent_trace_observability": "AgentTrace",
    "invoice_processing": "Invoice",
    "customer_service": "CustomerSvc",
}
BASE = "https://evals.typesafe.ai"

try:  # python.org macOS builds don't see the system keychain — certifi does
    import certifi

    CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    CTX = ssl.create_default_context()


# ---------------------------------------------------------------- fetch/extract

def fetch() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    for wf in WORKFLOWS:
        dst = DATA / f"{wf}-cases.js"
        url = f"{BASE}/{wf}-cases.js"
        print(f"  {url} -> {dst}")
        req = urllib.request.Request(url, headers={"User-Agent": "snap-eval/1.0"})
        dst.write_bytes(urllib.request.urlopen(req, timeout=60, context=CTX).read())


def load_wf(wf: str) -> dict:
    raw = (DATA / f"{wf}-cases.js").read_text(encoding="utf-8", errors="replace")
    m = re.search(r"__VIEWER_DATA__\((.*)\)\s*;?\s*$", raw, re.S) or re.search(
        r"__VIEWER_DATA__\((.*)\)", raw, re.S
    )
    if not m:
        raise SystemExit(f"{wf}: not a __VIEWER_DATA__ payload — run `fetch` first?")
    return json.loads(m.group(1))["eval"]


def canonical_ref_value(value, qtype: str):
    if value is None:
        return None
    if qtype == "noul":
        if isinstance(value, bool):
            return str(value).lower()
        if isinstance(value, (int, float)):
            return str(float(value) >= 0.5).lower()
        return str(value).lower()
    if qtype == "score":
        try:
            return str(max(0, min(3, int(round(float(value))))))
        except (TypeError, ValueError):
            return None
    return str(value)


def consensus_ref(entry: dict):
    """(canonical value, averaged probs): distribution form averages + argmaxes,
    value form majority-votes. Mirrors extract_full.py."""
    sets = entry.get("sets", [])
    if not sets:
        return None, {}
    acc, values = {}, []
    for s in sets:
        for k, v in (s.get("probabilities") or {}).items():
            acc[k] = acc.get(k, 0.0) + float(v)
        if s.get("value") is not None:
            values.append(s["value"])
    if acc:
        probs = {k: v / len(sets) for k, v in acc.items()}
        return max(probs, key=probs.get), probs
    canon = [canonical_ref_value(v, entry.get("type")) for v in values]
    canon = [c for c in canon if c is not None]
    if canon:
        return max(set(canon), key=canon.count), {}
    return None, {}


def render_doc(doc) -> str:
    return doc if isinstance(doc, str) else json.dumps(doc, indent=1, ensure_ascii=False)


def extract() -> None:
    out = {"workflows": {}}
    total_cases = total_pairs = 0
    for wf in WORKFLOWS:
        ev = load_wf(wf)
        docs = ev["documents"]
        catalog = ev["questions"]

        # workflow-wide qid -> catalog index (union over every case's nodes)
        wf_qmap: dict[str, int] = {}
        for ex in ev["examples"]:
            case = ev["cases"][ex["case_id"]]
            for mv in case["models"].values():
                for n in mv.get("nodes", []):
                    for qid, idx in (n.get("questions") or {}).items():
                        wf_qmap.setdefault(qid, int(idx))

        cases = []
        for ex in ev["examples"]:
            cid = ex["case_id"]
            case = ev["cases"][cid]

            doc_idx: set[int] = set()
            for mv in case["models"].values():
                for n in mv.get("nodes", []):
                    if n.get("doc") is not None:
                        doc_idx.add(int(n["doc"]))

            reference = {}
            for node_answers in case.get("reference_answers", {}).values():
                for qid, entry in node_answers.items():
                    val, probs = consensus_ref(entry)
                    reference[qid] = {"value": val, "probs": probs, "type": entry["type"]}

            questions = []
            for qid in reference:
                idx = wf_qmap.get(qid)
                if idx is None or not (0 <= idx < len(catalog)):
                    print(f"  WARN {wf}/{cid}: no catalog entry for {qid}")
                    continue
                q = catalog[idx]
                questions.append({
                    "qid": qid,
                    "type": q["type"],
                    "instructions": q["instructions"],
                    "criteria": q["criteria"],
                })

            input_text = "\n\n".join(
                f"## Document {i}\n{render_doc(docs[i])}"
                for i in sorted(doc_idx)
                if 0 <= i < len(docs)
            )
            cases.append({
                "case_id": cid,
                "name": ex.get("name", cid),
                "doc_indices": sorted(doc_idx),
                "input_text": input_text,
                "input_chars": len(input_text),
                "questions": questions,
                "reference": reference,
            })
            total_cases += 1
            total_pairs += len(reference)

        out["workflows"][wf] = {
            "title": wf,
            "catalog_size": len(catalog),
            "cases": cases,
        }

    FULL.write_text(json.dumps(out))
    for wf, data in out["workflows"].items():
        pairs = sum(len(c["reference"]) for c in data["cases"])
        chars = [c["input_chars"] for c in data["cases"]]
        print(f"  {wf:30} cases={len(data['cases']):2} pairs={pairs:3} "
              f"inputs={min(chars)}-{max(chars)} chars")
    print(f"wrote {FULL} — {total_cases} cases, {total_pairs} reference pairs")


# ---------------------------------------------------------------- answer

def raw_answer(ans: dict):
    t = ans.get("type")
    if t == "noul":
        return ans.get("noul")          # float P(yes); scorer thresholds at 0.5
    if t == "choice":
        return ans.get("choice")
    if t == "score":
        return ans.get("level")         # argmax level index, like the reference
    return ans.get("value")


def post(client_url: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        f"{client_url.rstrip('/')}/v1/systemone",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.loads(urllib.request.urlopen(req, timeout=timeout, context=CTX).read())


def answer(args) -> None:
    full = json.loads(FULL.read_text())
    if Path(args.out).exists():
        raise SystemExit(f"{args.out} exists (reports are create-only)")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    answers, timings = {}, {}
    for wf, wdata in full["workflows"].items():
        if args.workflows and wf not in args.workflows:
            continue
        answers[wf] = {}
        for case in wdata["cases"]:
            cid = case["case_id"]
            qs = {
                q["qid"]: {
                    "type": q["type"],
                    "instructions": q["instructions"],
                    "criteria": q["criteria"],
                }
                for q in case["questions"]
            }
            payload = {"state": case["input_text"], "questions": qs}
            if args.layout != "auto":
                payload["layout"] = args.layout
            t = time.perf_counter()
            try:
                res = post(args.url, payload, timeout=args.timeout)
            except urllib.error.HTTPError as e:
                # one bad question sinks a whole case — retry per question
                print(f"{wf}/{cid}: case failed ({e.code} {e.read()[:120]}), "
                      "falling back to per-question", file=sys.stderr)
                res = {"answers": {}}
                for qid, q in qs.items():
                    try:
                        r = post(args.url, {**payload, "questions": {qid: q}},
                                 timeout=args.timeout)
                        res["answers"].update(r["answers"])
                    except Exception as e2:
                        print(f"  {qid}: {e2}", file=sys.stderr)
            ms = (time.perf_counter() - t) * 1000
            answers[wf][cid] = {
                qid: {"raw": raw_answer(a), "kind": a.get("type")}
                for qid, a in res["answers"].items()
            }
            timings[f"{wf}/{cid}"] = {
                "total_ms": round(ms, 1),
                "questions": len(qs),
                "input_tokens": res.get("usage", {}).get("input_tokens"),
            }
            print(f"{wf:28s} {cid[:40]:40s} {len(qs):3d}q "
                  f"{res.get('usage', {}).get('input_tokens') or 0:7d} tok "
                  f"{ms / 1000:6.1f}s", flush=True)
    Path(args.out).write_text(json.dumps(
        {"engine": "snap", "timings": timings, "answers": answers}, indent=1))
    print(f"wrote {args.out}")


# ---------------------------------------------------------------- score

def canonical(value, kind: str, qtype: str):
    if value is None:
        return None
    if qtype == "noul":
        try:
            return str(float(value) >= 0.5).lower()
        except (TypeError, ValueError):
            return None
    if qtype == "score":
        try:
            return str(max(0, min(3, int(round(float(value))))))
        except (TypeError, ValueError):
            return None
    return str(value)


def consensus(ref: dict):
    probs = ref.get("probs") or {}
    if probs:
        return str(max(probs, key=probs.get))
    v = ref.get("value")
    return str(v) if v is not None else None


def load_answers(path: str, qtype_by_wf) -> dict:
    """Normalize an answers file: jol shape answers[wf][cid][qid]={raw,kind},
    or the published_answers shape [wf][cid][model][qid]={raw,kind} (pass the
    model key as 'published:opus')."""
    data = json.loads(Path(path).read_text())
    out = {}
    for wf, cases in data.get("answers", data).items():
        if wf in ("model", "timings", "engine"):
            continue
        out[wf] = {}
        for cid, answers in cases.items():
            out[wf][cid] = {}
            for qid, a in answers.items():
                t = qtype_by_wf.get(wf, {}).get(qid)
                if t is None:
                    continue
                c = canonical(a.get("raw"), a.get("kind", ""), t)
                if c is not None:
                    out[wf][cid][qid] = c
    return out


def score(args) -> None:
    full = json.loads(FULL.read_text())
    qtype_by_wf = {
        wf: {q["qid"]: q["type"] for c in wdata["cases"] for q in c["questions"]}
        for wf, wdata in full["workflows"].items()
    }

    models = {}
    for path in args.files:
        name = Path(path).stem
        models[name] = load_answers(path, qtype_by_wf)
    if args.published and Path(args.published).is_file():
        pub = json.loads(Path(args.published).read_text())
        for mkey in ("opus", "sol", "typesafe"):
            m = {}
            for wf, cases in pub.items():
                m[wf] = {}
                for cid, per_model in cases.items():
                    m[wf][cid] = {}
                    for qid, a in per_model.get(mkey, {}).items():
                        t = qtype_by_wf.get(wf, {}).get(qid)
                        if t is None:
                            continue
                        c = canonical(a.get("raw"), a.get("kind", ""), t)
                        if c is not None:
                            m[wf][cid][qid] = c
            models[f"published:{mkey}"] = m

    hdr = f"{'model':<28} {'overall':>13} " + " ".join(f"{LABEL[w]:>15}" for w in WORKFLOWS)
    print(f"\n=== Agreement with TypeSafe's reference (frontier consensus) ===\n{hdr}")
    print("-" * len(hdr))
    rows = []
    for name, m in models.items():
        cells, tot_ok, tot_n = [], 0, 0
        for wf in WORKFLOWS:
            ok = n = 0
            for case in full["workflows"][wf]["cases"]:
                cid = case["case_id"]
                for qid, ref in case["reference"].items():
                    want, got = consensus(ref), m.get(wf, {}).get(cid, {}).get(qid)
                    if want is None or got is None:
                        continue
                    ok += got == want
                    n += 1
            tot_ok += ok
            tot_n += n
            cells.append(f"{ok}/{n}" if n else "-")
        rows.append((name, tot_ok, tot_n, cells))
    for name, ok, n, cells in sorted(rows, key=lambda r: -(r[1] / r[2] if r[2] else 0)):
        acc = f"{ok / n * 100:.1f}%" if n else "n/a"
        print(f"{name:<28} {acc:>6} {ok:>3}/{n:<3} " + " ".join(f"{c:>15}" for c in cells))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch")
    sub.add_parser("extract")
    a = sub.add_parser("answer")
    a.add_argument("--url", default="http://127.0.0.1:8018")
    a.add_argument("--out", required=True)
    a.add_argument("--layout", default="auto",
                   choices=["auto", "state_first", "question_first", "header"])
    a.add_argument("--workflows", nargs="*")
    a.add_argument("--timeout", type=float, default=900)
    s = sub.add_parser("score")
    s.add_argument("files", nargs="+", help="answer JSON files to compare")
    s.add_argument("--published", default=str(DATA / "published_answers.json"),
                   help="jol's published_answers.json, if available")
    args = ap.parse_args()
    {"fetch": fetch, "extract": extract, "answer": lambda: answer(args),
     "score": lambda: score(args)}[args.cmd]()


if __name__ == "__main__":
    main()
