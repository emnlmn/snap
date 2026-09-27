"""Convert HuggingFace decision datasets into snap eval-schema cases.

Each adapter emits one JSON object per line:
  {"id", "state", "question", "expect",
   "source", "license",            # provenance, kept on every row
   "soft": {...},                 # optional soft label distribution
   "requires_abstain": true}      # on abstain-positive rows

Slot semantics (from snap prompts.rs):
  choice object criteria -> answer.choice = criteria key
  choice array criteria  -> answer.choice = option TEXT
  score                  -> criteria array low->high, expect.level = index
  noul                   -> never allow_abstain; expect.boolean
  boolean                -> expect.boolean; allow_abstain ok

License policy: tasksource/jev-bench carry per-row licenses; rows whose
license is non-commercial are dropped by default (--licenses to relax).
"""

import argparse
import json
import random
import sys
from collections import Counter

try:
    from datasets import load_dataset
except ImportError:
    sys.exit("pip install datasets")

PERMISSIVE = {
    "cc0-1.0", "cc-by-4.0", "cc-by-3.0", "cc-by-2.5", "cc-by-sa-4.0",
    "cc-by-sa-3.0", "apache-2.0", "mit", "odc-by", "pddl", "bsd",
    "gpl-3.0", "lgpl-3.0", "unlicense", "unspecified", "other", "unknown",
}
DROP_USE = {"non-commercial"}
# substrings that mark a row research-only / non-commercial even when the
# leading token looks permissive (e.g. "other (Yelp Dataset License,
# research use)")
LIC_DENY_SUBSTR = ("non-commercial", "noncommercial", "non commercial",
                   "research use", "research only", "research purposes",
                   "academic")


def license_ok(lic: str, use: str = "") -> bool:
    """Normalize 'cc-by-4.0 (mirror: mit)' -> 'cc-by-4.0', then policy check."""
    full = lic.lower()
    if use.lower() in DROP_USE or any(b in full for b in LIC_DENY_SUBSTR):
        return False
    return full.split("(")[0].strip() in PERMISSIVE


def emit(src, lic, state, question, expect, soft=None, **extra):
    o = {"id": extra.pop("id"), "state": state, "question": question,
         "expect": expect, "source": src, "license": lic}
    if soft:
        o["soft"] = soft
    o.update(extra)
    return o


def norm_probs(p):
    p = [float(x) for x in p]
    s = sum(p)
    return [x / s for x in p] if s > 0 else None


def argmax(p):
    return max(range(len(p)), key=lambda i: p[i]) if p else None


# ---------------------------------------------------------------------------
# Open-Jev release-v2-redistributable (CC0): state_json, question, options as
# "key: description" strings, target = prob list, kind = choice|score.
# ---------------------------------------------------------------------------

def conv_openjev(n, seed, split="train"):
    ds = load_dataset("ZefanCai/Open-Jev", "release-v2-redistributable",
                      split=split)
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    out = []
    for i in idx:
        if len(out) >= n:
            break
        r = ds[i]
        try:
            state = json.loads(r["state_json"])
        except Exception:
            state = r["state_json"]
        opts = r.get("options") or []
        tgt = norm_probs(r.get("target") or [])
        if not tgt:
            continue
        win = argmax(tgt)
        kind = r.get("kind")
        if kind == "score":
            if not (2 <= len(opts) <= 25):
                continue
            q = {"type": "score", "instructions": r["question"],
                 "criteria": [str(o) for o in opts]}
            expect = {"level": win}
        elif kind == "choice":
            crit = {}
            ok = True
            for o in opts:
                k, _, v = str(o).partition(":")
                k = k.strip()
                if not k or k in crit:
                    ok = False
                    break
                crit[k] = v.strip() or k
            if not ok or not (2 <= len(crit) <= 25):
                continue
            keys = list(crit)
            q = {"type": "choice", "instructions": r["question"],
                 "criteria": crit}
            expect = {"choice": keys[win]}
        else:
            continue
        soft = None
        if kind == "score":
            soft = {"levels": tgt}
        else:
            soft = {"options": tgt}
        out.append(emit("openjev", "cc0-1.0", state, q, expect, soft,
                        id=f"ojv-{split}-{len(out):05d}"))
    return out


# ---------------------------------------------------------------------------
# tasksource jev-typed-decisions: 2.5M procedural rows, streamed.
# Per-row license; license_use=="non-commercial" dropped.
# noul target order [no, yes] — verified on samples.
# ---------------------------------------------------------------------------

def conv_tasksource(n, seed, permissive_only=True, shards=2):
    """Read N parquet shards directly (streaming the full 2.5M is too slow)."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    rng = random.Random(seed)
    pool, seen, lic_dropped = [], 0, Counter()
    for s in range(shards):
        path = hf_hub_download(
            "tasksource/tasksource-jev-typed-decisions",
            f"data/train-{s:05d}-of-00012.parquet", repo_type="dataset")
        rows = pq.read_table(
            path, columns=["kind", "question", "target", "options", "state",
                           "license", "license_use", "source"]).to_pylist()
        rng.shuffle(rows)
        for r in rows:
            lic = (r.get("license") or "unspecified").lower()
            use = (r.get("license_use") or "").lower()
            if permissive_only and (use in DROP_USE):
                lic_dropped[lic] += 1
                continue
            kind = r.get("kind")
            raw_tgt = r.get("target") or []
            state = r["state"]
            if not state or len(state) > 20000:
                continue
            if kind == "noul":
                # scalar p(yes); [0.0] is a valid target — normalize AFTER
                # (norm_probs([0.0]) -> None would silently drop every "no")
                if not raw_tgt:
                    continue
                p_yes = float(raw_tgt[0]) if len(raw_tgt) == 1 else \
                    float(raw_tgt[-1])
                q = {"type": "noul", "instructions": r["question"]}
                expect = {"boolean": p_yes >= 0.5}
                soft = {"p_yes": p_yes}
                case = emit(f"tasksource/{r.get('source', '?')}", lic, state,
                            q, expect, soft, id="")
                seen += 1
                if len(pool) < n:
                    case["id"] = f"tsk-{len(pool):05d}"
                    pool.append(case)
                elif rng.random() < n / seen:
                    j = rng.randrange(n)
                    case["id"] = f"tsk-{j:05d}"
                    pool[j] = case
                continue
            tgt = norm_probs(raw_tgt)
            if not tgt:
                continue
            win = argmax(tgt)
            if kind == "choice":
                opts = r.get("options") or []
                if not (2 <= len(opts) <= 26):
                    continue
                q = {"type": "choice", "instructions": r["question"],
                     "criteria": [str(o) for o in opts]}
                expect = {"choice": str(opts[win])}
                soft = {"options": tgt}
            elif kind == "score":
                opts = r.get("options") or []
                if not (2 <= len(opts) <= 25):
                    continue
                q = {"type": "score", "instructions": r["question"],
                     "criteria": [str(o) for o in opts]}
                expect = {"level": win}
                soft = {"levels": tgt}
            else:
                continue
            case = emit(f"tasksource/{r.get('source', '?')}", lic, state, q,
                        expect, soft, id="")
            seen += 1
            if len(pool) < n:
                case["id"] = f"tsk-{len(pool):05d}"
                pool.append(case)
            elif rng.random() < n / seen:
                j = rng.randrange(n)
                case["id"] = f"tsk-{j:05d}"
                pool[j] = case
    if lic_dropped:
        print(f"  license-dropped: {dict(lic_dropped.most_common(5))}",
              file=sys.stderr)
    return pool


# ---------------------------------------------------------------------------
# jev-bench: 22 configs, meta.license per row, soft_label sometimes present.
# question JSON is already near-snap shape {type, instructions, criteria}.
# ---------------------------------------------------------------------------

JEVBENCH_CONFIGS = [
    "arc_challenge", "banking77", "boolq", "chaosnli", "civil_comments",
    "clinc150", "fever_evidence", "go_emotions", "helpsteer2_helpfulness",
    "helpsteer2_verbosity", "ledgar", "massive", "measuring_hate_speech",
    "mmlu", "mnli", "paws", "sms_spam", "sst5", "strategyqa_closed",
    "strategyqa_grounded", "stsb", "yelp5",
]


def conv_jevbench(per_config, seed, permissive_only=True):
    rng = random.Random(seed)
    out = []
    for cfg in JEVBENCH_CONFIGS:
        try:
            ds = load_dataset("Praveenrajus/jev-bench", cfg, split="train")
        except Exception as e:
            print(f"  {cfg}: load failed ({e})", file=sys.stderr)
            continue
        idx = list(range(len(ds)))
        rng.shuffle(idx)
        kept = 0
        for i in idx:
            if kept >= per_config:
                break
            r = ds[i]
            try:
                meta = json.loads(r["meta"]) if isinstance(r["meta"], str) else (r["meta"] or {})
            except Exception:
                meta = {}
            lic = str(meta.get("license", "unspecified")).lower()
            if permissive_only and not license_ok(lic):
                continue
            ncrit = len((json.loads(r["question"]) if isinstance(r["question"], str) else {}).get("criteria") or {})
            if ncrit > 25:
                continue   # beyond the 26-slot letter budget
            try:
                state = json.loads(r["state"])
                qq = json.loads(r["question"])
            except Exception:
                continue
            prim = r.get("primitive") or qq.get("type")
            label = r.get("label")
            soft = None
            if r.get("soft_label"):
                try:
                    sl = json.loads(r["soft_label"]) if isinstance(r["soft_label"], str) else r["soft_label"]
                    soft = {"distribution": sl}
                except Exception:
                    pass
            if prim == "choice":
                crit = qq.get("criteria")
                if not isinstance(crit, dict) or label not in crit:
                    continue
                if not (2 <= len(crit) <= 25):
                    continue
                q = {"type": "choice", "instructions": qq.get("instructions", ""),
                     "criteria": crit}
                expect = {"choice": label}
            elif prim == "noul":
                q = {"type": "noul", "instructions": qq.get("instructions", "")}
                expect = {"boolean": str(label).lower() in ("true", "yes", "1", "a")}
            elif prim == "score":
                crit = qq.get("criteria")
                if isinstance(crit, dict):
                    keys = sorted(crit)          # letter-keyed levels
                    levels = [crit[k] for k in keys]
                    lv = keys.index(label) if label in keys else (
                        int(label) if str(label).isdigit() else None)
                elif isinstance(crit, list):
                    levels = crit
                    lv = (int(label) if str(label).isdigit() else
                          crit.index(label) if label in crit else
                          ord(str(label)) - 65 if len(str(label)) == 1
                          else None)
                else:
                    continue
                if lv is None or not (2 <= len(levels) <= 25):
                    continue
                q = {"type": "score", "instructions": qq.get("instructions", ""),
                     "criteria": levels}
                expect = {"level": lv}
            else:
                continue
            out.append(emit(f"jevbench/{cfg}", lic, state, q, expect, soft,
                            id=f"jvb-{cfg[:12]}-{kept:04d}"))
            kept += 1
        print(f"  {cfg}: {kept}", file=sys.stderr)
    return out


# ---------------------------------------------------------------------------
# LocalLLaMA typed-decisions (Apache-2.0): gold has real probability
# distributions + agreement stats. TRAIN SPLIT ONLY — its test split is the
# published benchmark; keep it clean for eval.
# ---------------------------------------------------------------------------

def conv_typeddec():
    ds = load_dataset("LocalLLaMA/typed-decisions", "all", split="train")
    out = []
    for r in ds:
        if r.get("split") != "train":
            continue
        try:
            state = json.loads(r["state"])
            questions = json.loads(r["questions"])
            gold = json.loads(r["gold"])
        except Exception:
            continue
        for qi, (qname, spec) in enumerate(sorted(questions.items())):
            g = gold.get(qname) or {}
            gtype = g.get("type") or spec.get("type")
            label = g.get("label")
            probs = g.get("probabilities")
            soft = {"distribution": probs} if probs else None
            if gtype == "choice":
                crit = spec.get("criteria")
                if not isinstance(crit, dict) or label not in crit:
                    continue
                q = {"type": "choice",
                     "instructions": spec.get("instructions", ""),
                     "criteria": crit}
                expect = {"choice": label}
            elif gtype == "noul":
                q = {"type": "noul",
                     "instructions": spec.get("instructions", "")}
                expect = {"boolean": str(label).lower() == "true"}
            elif gtype == "score":
                crit = spec.get("criteria")
                if isinstance(crit, dict):
                    keys = sorted(crit)
                    levels = [crit[k] for k in keys]
                    lv = int(label) if str(label).isdigit() else (
                        keys.index(label) if label in keys else None)
                elif isinstance(crit, list):
                    levels = crit
                    lv = int(label) if str(label).isdigit() else None
                else:
                    continue
                if lv is None or lv >= len(levels):
                    continue
                q = {"type": "score",
                     "instructions": spec.get("instructions", ""),
                     "criteria": levels}
                expect = {"level": lv}
            else:
                continue
            out.append(emit(f"typeddec/{r['workflow']}", "apache-2.0", state,
                            q, expect, soft,
                            id=f"tdc-{r['id'][-8:]}-{qi}"))
    return out


# ---------------------------------------------------------------------------
# trilemma-of-truth (CC-BY-4.0): statement + {true,false,neither}.
# "neither" = fabricated/unverifiable -> natural abstain-positive rows.
# ---------------------------------------------------------------------------

TRILEMMA_SUBSETS = ["city_locations", "med_indications", "word_definitions"]

TRILEMMA_INSTRUCTIONS = [
    "Is the statement in `state` factually correct? Answer yes only if it is verifiably correct, no if it is verifiably wrong; abstain if it cannot be verified.",
    "Decide whether the claim is correct. Abstain when it refers to entities or facts that cannot be verified.",
]


def conv_trilemma(n, seed):
    rng = random.Random(seed)
    per_subset = max(1, n // len(TRILEMMA_SUBSETS))
    out = []
    for sub in TRILEMMA_SUBSETS:
        try:
            ds = load_dataset("carlomarxx/trilemma-of-truth", sub, split="train")
        except Exception as e:
            print(f"  {sub}: load failed ({e})", file=sys.stderr)
            continue
        buckets = {"true": [], "false": [], "neither": []}
        for r in ds:
            lab = r.get("multiclass_label")
            if isinstance(lab, int):        # ClassLabel: 0=false 1=true 2=neither
                key = {0: "false", 1: "true", 2: "neither"}.get(lab)
            else:                            # string form "0false"/"1true"/...
                lab = str(lab)
                key = ("true" if lab.endswith("true") else
                       "false" if lab.endswith("false") else
                       "neither" if lab.endswith("neither") else None)
            if key:
                buckets[key].append(r["statement"])
        per_class = max(1, per_subset // 3)
        for cls, stmts in buckets.items():
            rng.shuffle(stmts)
            for st in stmts[:per_class]:
                q = {"type": "boolean",
                     "instructions": rng.choice(TRILEMMA_INSTRUCTIONS),
                     "allow_abstain": True}
                if cls == "neither":
                    expect = {"status": "abstained"}
                    extra = {"requires_abstain": True}
                else:
                    expect = {"boolean": cls == "true"}
                    extra = {}
                out.append(emit(f"trilemma/{sub}", "cc-by-4.0",
                                {"claim": st}, q, expect, None,
                                id=f"trl-{sub[:10]}-{len(out):05d}", **extra))
    return out


CONVERTERS = {
    "openjev": conv_openjev,
    "tasksource": conv_tasksource,
    "jevbench": conv_jevbench,
    "typeddec": lambda n, seed: conv_typeddec(),
    "trilemma": conv_trilemma,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("source", choices=[*CONVERTERS, "all"])
    p.add_argument("--n", type=int, default=5000,
                   help="cases per source (or per config for jevbench)")
    p.add_argument("--out-dir", default="data/raw")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--include-noncommercial", action="store_true",
                   help="keep rows whose license_use is non-commercial")
    p.add_argument("--force", action="store_true", help="overwrite outputs")
    args = p.parse_args()

    sources = list(CONVERTERS) if args.source == "all" else [args.source]
    for src in sources:
        fn = CONVERTERS[src]
        print(f"[{src}] converting...", file=sys.stderr)
        if src in ("tasksource", "jevbench"):
            rows = fn(args.n, args.seed, not args.include_noncommercial)
        else:
            rows = fn(args.n, args.seed)
        out_path = f"{args.out_dir}/hf-{src}.jsonl"
        mode = "w" if args.force else "x"
        with open(out_path, mode) as fh:
            for o in rows:
                fh.write(json.dumps(o, ensure_ascii=False) + "\n")
        lic = Counter(r["license"] for r in rows)
        print(f"[{src}] wrote {len(rows)} -> {out_path}  licenses: "
              f"{dict(lic.most_common(6))}", file=sys.stderr)


if __name__ == "__main__":
    main()
