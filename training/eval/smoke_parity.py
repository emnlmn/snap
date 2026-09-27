#!/usr/bin/env python3
"""Train/serve parity check: HF bf16 forward vs snap GGUF on the same
exported rows. Catches the failure class where the trainer and llama.cpp
see different tokens/logits for the same prompt:

  1. tokenization — exported `token_ids` (snap's serve-time ids) vs a plain
     HF retokenize of the prompt string;
  2. letter pooling — per-letter max over single-token variants
     ("A", " A", "\nA"), same on both sides;
  3. distribution — KL between snap's probabilities map and the HF
     restricted softmax over the row's letters (shared keys only).

Usage:
  snap serve --model minicpm5-2b --port 8099 &
  python3 eval/smoke_parity.py data/train/export-layerA.jsonl \
      --cases data/labeled/layerA.jsonl --n 60 --server http://127.0.0.1:8099
"""
import argparse
import json
import sys
import urllib.request

import torch
import torch.nn.functional as F

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def decide(server, state, q):
    body = json.dumps({
        "state": state, "questions": {"q": q},
        "temperature": 1.0, "mode": "shared",
    }).encode()
    req = urllib.request.Request(
        f"{server}/v1/systemone", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())["answers"]["q"]


def snap_letter_probs(ans, keys, texts):
    """probabilities map (semantic keys, slot order) -> per-position probs.
    Older snap builds strip `probabilities` from noul answers — fall back
    to the P(yes) scalar; the abstain slot (if any) stays unknown there."""
    probs = ans.get("probabilities")
    if probs is None and isinstance(ans.get("noul"), (int, float)):
        p = float(ans["noul"])
        probs = {"yes": p, "no": 1.0 - p}
    probs = probs or {}
    out = [None] * len(keys)
    for k, v in probs.items():
        for i, (kk, tt) in enumerate(zip(keys, texts)):
            if k == kk or k == tt:
                out[i] = float(v)
                break
    return out


def stratified(ids, rows, n):
    """Round-robin across (source, qtype) strata so the sample isn't just
    whatever source happens to sort first in the export file."""
    by_stratum = {}
    for i in ids:
        r = rows[i]
        key = ((r.get("meta") or {}).get("source") or "?", r.get("qtype"))
        by_stratum.setdefault(key, []).append(i)
    buckets = [by_stratum[k] for k in sorted(by_stratum)]
    out = []
    while len(out) < n:
        moved = False
        for b in buckets:
            if b and len(out) < n:
                out.append(b.pop(0))
                moved = True
        if not moved:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exported", help="exported training JSONL (prompt+token_ids)")
    ap.add_argument("--cases", required=True, help="original case JSONL")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--server", default="http://127.0.0.1:8099")
    ap.add_argument("--hf", default="openbmb/MiniCPM5-2B")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = {r["id"]: r for r in load_jsonl(args.exported)}
    cases = {c["id"]: c for c in load_jsonl(args.cases)}
    pool = [i for i in rows if i in cases and "~p" not in i]
    ids = stratified(pool, rows, args.n)
    if not ids:
        sys.exit("no overlapping ids between export and cases")

    # CPU default: MPS can drift enough to fake a parity failure
    dev = args.device or "cpu"
    tok = AutoTokenizer.from_pretrained(args.hf, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.hf, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(dev).eval()

    # letter variants — same pooling as the trainer / snap's read_row
    variants = []
    for ch in LETTERS:
        v = set()
        for s in (ch, f" {ch}", f"\n{ch}"):
            e = tok(s, add_special_tokens=False)["input_ids"]
            if len(e) == 1:
                v.add(e[0])
        variants.append(sorted(v))

    stats = {"n": 0, "tok_same": 0, "argmax_agree": 0}
    kls, gaps = [], []
    mismatches = []
    with torch.no_grad():
        for rid in ids:
            row, case = rows[rid], cases[rid]
            exported_ids = row.get("token_ids")
            hf_ids = tok(row["prompt"], add_special_tokens=False)["input_ids"]
            if exported_ids is not None and exported_ids != hf_ids:
                stats["tok_same"] += 0
                mismatches.append((rid, len(exported_ids), len(hf_ids)))
                use_ids = hf_ids
            else:
                stats["tok_same"] += int(exported_ids is not None)
                use_ids = exported_ids or hf_ids

            out = model(input_ids=torch.tensor([use_ids], device=dev))
            bl = out.logits[0, -1]
            lidx = [LETTERS.index(l) for l in row["letters"]]
            pooled = [max(float(bl[t]) for t in variants[l]) for l in lidx]
            hf_p = F.softmax(torch.tensor(pooled), dim=-1).tolist()

            ans = decide(args.server, case["state"], case["question"])
            snap_p = snap_letter_probs(ans, row["letter_keys"],
                                       row.get("letter_texts") or row["letter_keys"])
            # compare over positions present on both sides
            pairs = [(h, s) for h, s in zip(hf_p, snap_p) if s is not None]
            hs = torch.tensor([p[0] for p in pairs])
            ss = torch.tensor([p[1] for p in pairs])
            hs, ss = hs / hs.sum(), ss / ss.sum()
            kl = float((ss * (ss / hs.clamp_min(1e-9)).log()).sum())
            kls.append(kl)
            gaps.append(float(hs.max() - ss.max()))
            stats["argmax_agree"] += int(hs.argmax() == ss.argmax())
            stats["n"] += 1

    kls.sort()
    print(json.dumps({
        "rows": stats["n"],
        "tok_ids_identical": f"{stats['tok_same']}/{stats['n']}",
        "argmax_agree": f"{stats['argmax_agree']}/{stats['n']}",
        "kl_mean": round(sum(kls) / len(kls), 5),
        "kl_p95": round(kls[int(len(kls) * 0.95)], 5),
        "kl_max": round(kls[-1], 5),
        "max_gap_mean": round(sum(gaps) / len(gaps), 4),
        "tok_mismatch_examples": mismatches[:5],
    }, indent=2))
    if stats["argmax_agree"] / stats["n"] < 0.9 or kls[len(kls) // 2] > 0.1:
        print("PARITY SUSPECT — investigate token ids / pooling before training")
        sys.exit(1)


if __name__ == "__main__":
    main()
