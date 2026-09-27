#!/usr/bin/env python3
"""Join labeled cases with snap-rendered prompts -> training JSONL.

`snap export-prompts` emits, per case, the byte-exact prompt (chat template
applied by llama.cpp) plus the ordered letter->key slot map. This script
fuses that with the case supervision into a per-letter target distribution.

Target policy (deliberately simple — calibration lives in the data mix and
post-hoc `snap calibrate`, not in per-row mass bookkeeping):

- real human-vote `soft` distributions are used as-is where they exist;
  supported shapes: {"options":[p..]}, {"levels":[p..]}, {"p_yes":x},
  {"distribution": {key:p} | [p..] | p_yes scalar}, or a flat {key:p} map;
- otherwise a smoothed one-hot on the verified `expect` label: uniform
  over the remaining real slots for choice/boolean, ±1 neighbors for the
  ordinal types (score/numeric). Special slots (abstain/below/above) get
  mass only when they ARE the verified label;
- teacher voices never shape the target. They set `review` (recomputed
  here from stored argmax vs the gold slot, so stale labeling bugs can't
  leak through) and optionally drop rows via --drop-contested.

`--permute N` replaces every choice row with N shuffled-criteria copies —
the generator's original order carries its own position bias, so it is
never emitted. Semantic keys survive the shuffle because matching is by
key/text, not position.

Usage:
  python3 lora/export_prompts.py data/labeled/layerA.jsonl \
      --out data/train/export-layerA.jsonl [--model minicpm5-2b]
"""
import argparse
import hashlib
import itertools
import json
import os
import random
import subprocess
import sys
import tempfile

SNAP = os.environ.get("SNAP_BIN", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "target", "release", "snap"))


def state_text(state):
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, sort_keys=True)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def export_prompts(snap, model, case_file):
    """Run `snap export-prompts` on one JSONL file -> {id: record}."""
    p = subprocess.run(
        [snap, "export-prompts", "--model", model, case_file],
        capture_output=True, text=True,
    )
    if p.returncode != 0:
        sys.exit(f"snap export-prompts failed on {case_file}:\n{p.stderr[-3000:]}")
    out = {}
    for line in p.stdout.splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        out[r["id"]] = r
    return out


def expected_key(case):
    """Verified-label semantic key — the center of every target."""
    ex, qt = case["expect"], case["question"]["type"]
    if ex.get("status") == "abstained" or ex.get("requires_abstain"):
        return "__abstain__"
    if qt in ("boolean", "noul"):
        v = ex.get("boolean", ex.get("value"))
        return "yes" if v else "no"
    if qt == "choice":
        return ex["choice"]
    if qt == "score":
        return ex["level"]
    if qt == "numeric":
        # may be absent on a bare out_of_bounds expect — numeric_slot
        # resolves the boundary side from the value, None drops the row
        return ex.get("value")
    return None


def _float(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


BOOL_ALIAS = {"true": "yes", "false": "no"}


def slot_index(letters, key):
    """Map a semantic key to a letter position: key, then text, then numeric
    equality on either surface (Rust `Display` vs Python `str` disagree on
    integer-valued floats, so strings alone are not enough)."""
    if key is None:
        return None
    key = str(key)
    for i, s in enumerate(letters):
        if s["key"] == key or s["text"] == key:
            return i
    kf = _float(key)
    if kf is not None:
        for i, s in enumerate(letters):
            if _float(s["key"]) == kf or _float(s["text"]) == kf:
                return i
    return None


def numeric_slot(letters, question, value):
    """Numeric expectation -> nearest interior anchor, or the below/above
    marker when the value sits outside [min, max]."""
    if value < question["min"]:
        return slot_index(letters, "__below__")
    if value > question["max"]:
        return slot_index(letters, "__above__")
    best, bd = None, None
    for i, s in enumerate(letters):
        a = _float(s["key"]) if not s["special"] else None
        if a is not None and (bd is None or abs(a - value) < bd):
            best, bd = i, abs(a - value)
    return best


def gold_index(letters, case):
    """Letter position of the verified label."""
    key = expected_key(case)
    qt = case["question"]["type"]
    if qt == "score" and isinstance(key, int) and not isinstance(key, bool):
        return key if 0 <= key < len(letters) else None
    if qt == "numeric" and isinstance(key, (int, float)) and not isinstance(key, bool):
        return numeric_slot(letters, case["question"], float(key))
    return slot_index(letters, key)


def normalize_soft(case):
    """Human-vote distributions arrive in several shapes; return a semantic
    {key: p} map plus a shape tag, or (None, tag, reason) when malformed.

      {"options": [p..]}      choice/boolean, positional over real slots
      {"levels": [p..]}       score, positional over levels
      {"distribution": {...}} semantic keys (typed-decisions)
      {"distribution": [...]} positional (rare)
      {"distribution": p}     boolean vote fraction -> p_yes
      {"p_yes": p}            boolean vote fraction
      {key: p, ...}           already a semantic map
    """
    soft = case.get("soft")
    if not isinstance(soft, dict) or not soft:
        return None, "none", None
    qt = case["question"]["type"]
    q = case["question"]
    crit = q.get("criteria")

    def positional(v):
        if not isinstance(v, list) or not v or not all(
            isinstance(x, (int, float)) for x in v
        ):
            return None, "not a numeric list"
        keys = []
        if isinstance(crit, dict):
            keys = list(crit.keys())
        elif isinstance(crit, list):
            # array criteria: choice keys are the option texts, score keys
            # are level indices
            keys = [str(v_) if qt == "choice" else str(i)
                    for i, v_ in enumerate(crit)]
        elif qt in ("boolean", "noul"):
            keys = ["yes", "no"]
        if len(v) != len(keys):
            return None, f"len {len(v)} != {len(keys)} slots"
        return {keys[i]: float(x) for i, x in enumerate(v)}, None

    if "options" in soft:
        d, err = positional(soft["options"])
        return d, "options", err
    if "levels" in soft:
        d, err = positional(soft["levels"])
        return d, "levels", err
    if "p_yes" in soft:
        try:
            p = float(soft["p_yes"])
        except (TypeError, ValueError):
            return None, "p_yes", "not numeric"
        return {"yes": p, "no": 1.0 - p}, "p_yes", None
    if "distribution" in soft:
        d = soft["distribution"]
        if isinstance(d, dict):
            if all(isinstance(v, (int, float)) for v in d.values()):
                return {str(k): float(v) for k, v in d.items()}, "distribution", None
            return None, "distribution", "non-numeric dict values"
        if isinstance(d, list):
            out, err = positional(d)
            return out, "distribution", err
        if isinstance(d, (int, float)):
            if qt in ("boolean", "noul"):
                return {"yes": float(d), "no": 1.0 - float(d)}, "distribution", None
            return None, "distribution", "scalar on non-boolean"
        return None, "distribution", "unsupported type"
    if all(isinstance(v, (int, float)) for v in soft.values()):
        return {str(k): float(v) for k, v in soft.items()}, "flat", None
    return None, "unknown", "unrecognized keys"


def project(letters, dist):
    """{semantic key: p} -> per-position weights; returns (vec, unmatched)."""
    vec = [0.0] * len(letters)
    unmatched = 0.0
    for k, p in dist.items():
        i = slot_index(letters, k)
        if i is None:
            unmatched += float(p)
        else:
            vec[i] += float(p)
    return vec, unmatched


def vote_slots(letters, case):
    """Each teacher voice's argmax mapped to a letter position, recomputed
    here so stale labeling artifacts (e.g. the repr(40.0) bug) can't leak
    into the contested flag. Returns list of (slot|None)."""
    t = case.get("teacher") or {}
    if not t:
        return []
    votes = []
    for voice in (t, t.get("laya") or {}):
        if not voice:
            continue
        a = voice.get("argmax")
        if a is None and voice.get("status") == "abstained":
            a = "__abstain__"
        if a is not None:
            votes.append(slot_index(letters, a))
    return votes


def smoothed_gold(letters, gi, qtype, smoothing):
    """Gold-centered target. Ordinal types spread the eps mass on the ±1
    neighbors (close anchors are genuinely confusable); others spread it
    uniformly over the non-special slots. Special slots only carry mass
    when they are the verified answer."""
    vec = [0.0] * len(letters)
    gs = letters[gi]
    vec[gi] = 1.0 - smoothing
    if gs["special"]:
        real = [i for i, s in enumerate(letters) if not s["special"]]
        for i in real:
            vec[i] = smoothing / len(real)
    elif qtype in ("score", "numeric"):
        nb = [j for j in (gi - 1, gi + 1)
              if 0 <= j < len(letters) and not letters[j]["special"]]
        if nb:
            # a missing side means the gold sits at an edge — giving the
            # single inward neighbor the full eps taught boundary answers
            # to pull inside; each side keeps a fixed share and the
            # missing side's share stays on the gold
            share = smoothing / 2
            for j in nb:
                vec[j] = share
            vec[gi] += smoothing - share * len(nb)
        else:
            vec[gi] = 1.0
    else:
        real = [i for i, s in enumerate(letters)
                if not s["special"] and i != gi]
        for i in real:
            vec[i] = smoothing / len(real)
    return vec


def build_target(letters, case, smoothing):
    """Per-letter target distribution. Human `soft` votes first (the honest
    uncertainty signal); otherwise smoothed gold. Teacher disagreement is
    a review flag, never a target shape."""
    gi = gold_index(letters, case)
    dist, kind, err = normalize_soft(case)
    # typed-decisions noul votes key on true/false; snap slots are yes/no
    if dist is not None and case["question"]["type"] in ("boolean", "noul"):
        dist = {BOOL_ALIAS.get(str(k).lower(), k): v for k, v in dist.items()}
    # degenerate one-hot "soft" labels (max>=0.99, mostly Open-Jev/tasksource
    # hard labels stored as lists) are confidence 1.0 — hotter than the 0.9
    # gold rows get. Treat them as gold so they share the smoothing.
    if dist is not None and max(dist.values()) < 0.99:
        vec, unmatched = project(letters, dist)
        tot = sum(vec)
        if tot > 0:
            vec = [v / tot for v in vec]
        if unmatched > 0.05 and gi is not None:
            vec[gi] += unmatched
            tot = sum(vec)
            vec = [v / tot for v in vec]
        return vec, gi, "soft:" + kind, None, unmatched
    if err:
        return None, gi, None, err, 0.0
    if gi is None:
        return None, None, None, "nogold", 0.0
    ord_flag = case["question"]["type"] in ("score", "numeric")
    vec = smoothed_gold(letters, gi, case["question"]["type"], smoothing)
    degen = dist is not None
    return vec, gi, "soft-degen" if degen else (
        "ordinal" if ord_flag else "gold"), None, 0.0


def permute_choice(case, rng, seen_orders):
    """One copy of `case` with choice criteria shuffled to an ordering not
    already used for this case (tracked in `seen_orders`). expect.choice is
    a semantic key (dict) or option text (array), so it travels unchanged;
    only positional soft maps and int expects need remapping."""
    q = dict(case["question"])
    crit = q.get("criteria")
    if not isinstance(crit, (dict, list)) or len(crit) < 2:
        return None
    c = dict(case)
    c["question"] = q

    def fresh_order(n):
        # any ordering works — the generator's own order is just as biased
        # as any other, so copies sample the full permutation space and the
        # caller only enforces distinctness across this row's copies
        for _ in range(50):
            o = list(range(n))
            rng.shuffle(o)
            t = tuple(o)
            if t not in seen_orders:
                seen_orders.add(t)
                return o
        return None

    if isinstance(crit, dict):
        items = list(crit.items())
        order = fresh_order(len(items))
        if order is None:
            return None
        q["criteria"] = {items[j][0]: items[j][1] for j in order}
    elif isinstance(crit, list):
        order = fresh_order(len(crit))
        if order is None:
            return None
        q["criteria"] = [crit[j] for j in order]
        ex = dict(case["expect"])
        if isinstance(ex.get("choice"), int) and 0 <= ex["choice"] < len(order):
            ex["choice"] = order.index(ex["choice"])
        c["expect"] = ex
    else:
        return None
    # positional soft arrays must follow the same permutation
    soft = case.get("soft")
    if isinstance(soft, dict):
        soft = dict(soft)
        for k in ("options", "levels", "distribution"):
            if isinstance(soft.get(k), list) and len(soft[k]) == len(order):
                soft[k] = [soft[k][j] for j in order]
        c["soft"] = soft
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="labeled/validated case JSONL files")
    ap.add_argument("--out", required=True, help="output JSONL (create-only)")
    ap.add_argument("--model", default="minicpm5-2b")
    ap.add_argument("--snap", default=SNAP)
    ap.add_argument("--smoothing", type=float, default=0.10,
                    help="non-gold mass: ±1 neighbors for ordinal types, "
                         "uniform over real slots otherwise")
    ap.add_argument("--permute", type=int, default=0, metavar="N",
                    help="emit N shuffled-criteria copies per choice row "
                         "(the generator's original order is replaced — "
                         "it carries position bias)")
    ap.add_argument("--permute-seed", type=int, default=17)
    ap.add_argument("--drop-ids", metavar="FILE",
                    help="file of case ids to drop (eval twins etc.)")
    ap.add_argument("--drop-source", action="append", default=[],
                    help="skip rows whose source starts with this prefix")
    ap.add_argument("--max-per-source", type=int, default=0,
                    help="cap rows per source (0 = no cap)")
    ap.add_argument("--drop-contested", action="store_true",
                    help="drop rows where a teacher voice disagrees with gold "
                         "(rows with usable human soft labels are kept)")
    args = ap.parse_args()

    if os.path.exists(args.out):
        sys.exit(f"{args.out} exists (create-only)")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    stats = {"rows": 0, "noprompt": 0, "nogold": 0, "malformed_soft": 0,
             "dropped_source": 0, "dropped_ids": 0, "capped": 0,
             "contested": 0, "contested_dropped": 0, "abstain_gold": 0,
             "permuted": 0, "soft_unmatched_rows": 0,
             "soft_unmatched_mass": 0.0}
    rng = random.Random(args.permute_seed)
    drop_ids = set()
    if args.drop_ids:
        with open(args.drop_ids) as f:
            drop_ids = {l.strip() for l in f
                        if l.strip() and not l.startswith("#")}

    with open(args.out, "x") as out:
        for cf in args.inputs:
            cases = load_jsonl(cf)
            # source pruning happens before snap renders anything
            kept, per_source = [], {}
            for c in cases:
                if c.get("id") in drop_ids:
                    stats["dropped_ids"] += 1
                    continue
                src = c.get("source") or "gen"
                if any(src.startswith(p) for p in args.drop_source):
                    stats["dropped_source"] += 1
                    continue
                if args.max_per_source:
                    n = per_source.get(src, 0)
                    if n >= args.max_per_source:
                        stats["capped"] += 1
                        continue
                    per_source[src] = n + 1
                kept.append(c)

            expanded = []
            for c in kept:
                if args.permute and c["question"]["type"] == "choice":
                    seen_orders = set()
                    n_perm = 0
                    for k in range(args.permute):
                        p = permute_choice(c, rng, seen_orders)
                        if p is None:
                            continue
                        p["id"] = f"{c['id']}~p{k+1}"
                        expanded.append(p)
                        stats["permuted"] += 1
                        n_perm += 1
                    if n_perm == 0:
                        expanded.append(c)  # unpermutable — keep original
                else:
                    expanded.append(c)

            with tempfile.NamedTemporaryFile(
                "w", suffix=".jsonl", delete=False
            ) as tf:
                for c in expanded:
                    tf.write(json.dumps(c, ensure_ascii=False) + "\n")
                tmp = tf.name
            try:
                prompts = export_prompts(args.snap, args.model, tmp)
            finally:
                os.unlink(tmp)

            for c in expanded:
                p = prompts.get(c["id"])
                if p is None:
                    stats["noprompt"] += 1
                    continue
                letters = p["letters"]
                vec, gi, kind, err, unmatched = build_target(
                    letters, c, args.smoothing)
                if err == "nogold":
                    stats["nogold"] += 1
                    continue
                if err:
                    stats["malformed_soft"] += 1
                    continue
                if unmatched > 0.01:
                    stats["soft_unmatched_rows"] += 1
                    stats["soft_unmatched_mass"] += unmatched
                votes = vote_slots(letters, c)
                contested = bool(votes) and any(v != gi for v in votes)
                if contested:
                    stats["contested"] += 1
                    # human soft labels already encode the ambiguity — only
                    # contested *gold* rows are candidates for dropping
                    if args.drop_contested and not kind.startswith("soft"):
                        stats["contested_dropped"] += 1
                        continue
                if expected_key(c) == "__abstain__":
                    stats["abstain_gold"] += 1
                stats.setdefault(kind, 0)
                stats[kind] += 1
                rec = {
                    "id": c["id"],
                    "prompt": p["prompt"],
                    "token_ids": p.get("token_ids"),
                    "layout": p["layout"],
                    "state_key": hashlib.sha1(
                        state_text(c["state"]).encode()).hexdigest(),
                    # sig must see criteria ORDER (permuted copies differ)
                    # — sort top-level question keys only; nested dicts keep
                    # their insertion order when dumped
                    "sig": hashlib.sha1(
                        (state_text(c["state"]) + json.dumps(
                            {k: c["question"][k] for k in sorted(c["question"])},
                            ensure_ascii=False)).encode()).hexdigest(),
                    "review": contested,
                    "letters": [s["letter"] for s in letters],
                    "letter_keys": [s["key"] for s in letters],
                    "letter_texts": [s["text"] for s in letters],
                    "target": {s["letter"]: round(vec[i], 6)
                               for i, s in enumerate(letters) if vec[i] > 1e-9},
                    "gold_letter": letters[gi]["letter"],
                    "expect": c["expect"],
                    "meta": {k: c[k] for k in
                             ("source", "license", "lang", "domain", "difficulty",
                              "evidence_shape") if k in c},
                    "abstain": bool(c["question"].get("allow_abstain")),
                    "qtype": c["question"]["type"],
                    "chain": c.get("chain"),
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                stats["rows"] += 1
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
