#!/usr/bin/env python3
"""Merge case sources and split train/dev/holdout — leak-free.

Rows sharing a supervision group are never split across folds, and grouping
is GLOBAL (not per-stratum): the same underlying state may carry questions
of different types/langs, so per-stratum grouping would let identical
states straddle train and holdout.

Group key:
  - `chain:<chain id>` when the row carries a chain field (evidence chains:
    same question, growing state — different state_key, same group)
  - `state:<state_key>` otherwise (same normalized state text)

Assignment is greedy per stratum, rarest strata first, smallest candidate
group first — groups spanning several strata count toward every stratum
they touch, so quotas stay approximately proportional without splitting.

`--cases` takes the original labeled case files so the script can also
emit snap-evaluate-format dev/holdout files (id/state/question/expect —
augmented permutations collapse back to the base case id).

Output files are create-only.
"""
import argparse
import collections
import json
import os
import random
import re
import sys


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def shard_meta(manifest_path):
    """manifest.jsonl -> {shard_id: generation spec}."""
    meta = {}
    if not manifest_path or not os.path.exists(manifest_path):
        return meta
    for r in load_jsonl(manifest_path):
        spec = r.get("shard") or {}
        sid = spec.get("shard_id")
        if sid:
            meta[sid] = spec
    return meta


def attach_meta(rec, meta):
    sid = rec["id"].split("-")[0]
    spec = meta.get(sid)
    if spec:
        for k in ("lang", "domain", "difficulty", "evidence_shape"):
            rec["meta"].setdefault(k, spec.get(k))
    return rec


def stratum(rec):
    m = rec["meta"]
    src = m.get("source") or "gen"
    fam = src.split("/")[0] if "/" in src else ("gen" if src == "gen" else src.split("-")[0])
    return (fam, rec["qtype"], m.get("lang") or "en", rec["abstain"])


def group_key(rec, chains):
    cid = rec.get("chain") or chains.get(base_id(rec["id"]))
    if cid:
        return "chain:" + str(cid)
    return "state:" + rec["state_key"]


def base_id(rid):
    """'s0001-003~p2' -> 's0001-003' (permutation augment shares the group)."""
    return re.sub(r"~p\d+$", "", rid)


def eval_case(case):
    """Strip a labeled row down to the snap-evaluate case shape."""
    out = {"id": case["id"], "state": case["state"],
           "question": case["question"], "expect": case["expect"]}
    for k in ("requires_abstain", "variants", "layout", "expand"):
        if k in case:
            out[k] = case[k]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="exported training JSONL files")
    ap.add_argument("--manifest", help="orchestrator manifest.jsonl for shard meta")
    ap.add_argument("--cases", nargs="*", default=[],
                    help="original labeled case JSONLs for eval-format outputs")
    ap.add_argument("--out-prefix", required=True, help="writes <p>-train/-dev/-holdout.jsonl")
    ap.add_argument("--dev", type=float, default=0.05)
    ap.add_argument("--holdout", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--freeze",
                    help="previous <p>-train/-dev/-holdout.jsonl prefix: "
                         "groups already assigned there keep their fold, so "
                         "dev/holdout stay byte-identical for old cases")
    args = ap.parse_args()

    outs = {k: f"{args.out_prefix}-{k}.jsonl" for k in ("train", "dev", "holdout")}
    case_outs = {k: f"{args.out_prefix}-{k}-cases.jsonl" for k in ("dev", "holdout")}
    for p in list(outs.values()) + ([f"{args.out_prefix}-report.json"] +
                                    (list(case_outs.values()) if args.cases else [])):
        if os.path.exists(p):
            sys.exit(f"{p} exists (create-only)")
    os.makedirs(os.path.dirname(os.path.abspath(args.out_prefix)) or ".", exist_ok=True)

    meta = shard_meta(args.manifest)

    # chain membership lives on the *original* cases, so index it there;
    # the exported rows only carry id/state_key/sig
    chains, case_by_id = {}, {}
    for f in args.cases:
        for c in load_jsonl(f):
            case_by_id[c["id"]] = c
            if c.get("chain"):
                chains[c["id"]] = c["chain"]

    rows, seen, dup = [], set(), 0
    for f in args.inputs:
        for rec in load_jsonl(f):
            if rec["sig"] in seen:
                dup += 1
                continue
            seen.add(rec["sig"])
            rows.append(attach_meta(rec, meta))

    # global grouping: chain members and same-state rows are atomic
    groups = collections.defaultdict(list)
    for r in rows:
        groups[group_key(r, chains)].append(r)

    strata_size = collections.Counter(stratum(r) for r in rows)

    # per-fold row quota per stratum; strata too small to sample stay in
    # train entirely
    min_stratum = 10
    need = {"dev": collections.Counter(), "holdout": collections.Counter()}
    for s, n in strata_size.items():
        if n >= min_stratum:
            need["dev"][s] = max(1, round(n * args.dev))
            need["holdout"][s] = max(1, round(n * args.holdout))

    rng = random.Random(args.seed)
    group_ids = list(groups)
    rng.shuffle(group_ids)
    gstrata = {g: {stratum(r) for r in groups[g]} for g in group_ids}

    assigned = {}
    n_frozen = 0
    if args.freeze:
        frozen = {}
        for fold in ("train", "dev", "holdout"):
            p = f"{args.freeze}-{fold}.jsonl"
            if not os.path.exists(p):
                continue
            for r in load_jsonl(p):
                frozen[group_key(r, chains)] = fold
        for g in group_ids:
            if g in frozen:
                fold = frozen[g]
                assigned[g] = fold
                n_frozen += 1
                for r in groups[g]:
                    if fold in need:
                        need[fold][stratum(r)] -= 1

    for s in sorted(strata_size, key=strata_size.get):
        for fold in ("dev", "holdout"):
            while need[fold][s] > 0:
                cand = [g for g in group_ids
                        if g not in assigned and s in gstrata[g]]
                if not cand:
                    break
                # random pick, not smallest-first — min-by-size would never
                # let multi-row groups (chains) reach dev/holdout
                g = rng.choice(cand)
                assigned[g] = fold
                for r in groups[g]:
                    need[fold][stratum(r)] -= 1

    handles = {k: open(p, "x") for k, p in outs.items()}
    counts = {"train": 0, "dev": 0, "holdout": 0}
    fold_rows = {"train": [], "dev": [], "holdout": []}
    try:
        for g in group_ids:
            fold = assigned.get(g, "train")
            for r in groups[g]:
                handles[fold].write(json.dumps(r, ensure_ascii=False) + "\n")
                counts[fold] += 1
                fold_rows[fold].append(r)
    finally:
        for h in handles.values():
            h.close()

    if args.cases:
        for fold in ("dev", "holdout"):
            emitted = set()
            with open(case_outs[fold], "x") as f:
                for r in fold_rows[fold]:
                    bid = base_id(r["id"])
                    if bid in emitted:
                        continue
                    emitted.add(bid)
                    c = case_by_id.get(bid)
                    if c is not None:
                        f.write(json.dumps(eval_case(c), ensure_ascii=False) + "\n")

    # strata coverage sanity: count per stratum per fold
    per_stratum = {}
    for fold, frs in fold_rows.items():
        c = collections.Counter(stratum(r) for r in frs)
        for s, n in c.items():
            per_stratum.setdefault(str(s), {})[fold] = n

    n_multi = sum(1 for g in group_ids if len(gstrata[g]) > 1)
    report = {
        "total": len(rows), "dropped_dup": dup, "counts": counts,
        "groups": len(group_ids), "multi_stratum_groups": n_multi,
        "frozen_groups": n_frozen,
        "strata_sizes": {str(k): v for k, v in sorted(strata_size.items())},
        "per_stratum_folds": per_stratum,
        "files": outs, "case_files": case_outs if args.cases else None,
    }
    with open(f"{args.out_prefix}-report.json", "x") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps({"total": len(rows), "dropped_dup": dup,
                      "counts": counts, "groups": len(group_ids),
                      "multi_stratum_groups": n_multi,
                      "strata": len(strata_size)}, indent=2))


if __name__ == "__main__":
    main()
