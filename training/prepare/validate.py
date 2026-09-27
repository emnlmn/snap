"""Validate generated cases against snap's question schema + dedup/contamination.

Rules mirror src/schema.rs Question::validate and eval judge() keys:
  choice  -> criteria object {key: desc} or array; expect.choice in keys/indexes
  score   -> criteria array (levels low->high); expect.level = 0-based index
  numeric -> min<max, granularity 2..=24, anchors+2+abstain<=26;
             expect.value + optional tol
  boolean -> expect.boolean bool
  noul    -> expect.boolean bool; allow_abstain always false (Jev semantics)
  abstain -> expect.status "abstained" requires question.allow_abstain=true

Dedup: normalized-state sha256 for exact dupes; 5-word-shingle minhash-lite
(banding) for near-dupes inside the dataset and against the snap eval set —
the contamination check (state text overlap >= 0.5 Jaccard vs any eval state).
"""

import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

MAX_SLOTS = 26

WORD_RE = re.compile(r"[a-z0-9àèéìòù]+", re.IGNORECASE)


def norm_text(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).lower()
    return " ".join(WORD_RE.findall(s))


def shingles(text: str, n: int = 5) -> set:
    toks = norm_text(text).split()
    if len(toks) < n:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def state_text(state) -> str:
    """All text the model would read from the state, flattened."""
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, sort_keys=True)


def anchors(q: dict) -> int:
    mn, mx, step = q.get("min"), q.get("max"), q.get("step")
    if mn is not None and mx is not None and step and step > 0:
        return max(2, min(24, round((mx - mn) / step) + 1))
    return max(2, min(24, int(q.get("granularity", 8))))


def check_case(o: dict) -> list[str]:
    """Return list of validation errors (empty = ok)."""
    errs = []
    for k in ("id", "state", "question", "expect"):
        if k not in o:
            errs.append(f"missing key {k}")
    if errs:
        return errs

    q, exp = o["question"], o["expect"]
    if not isinstance(q, dict) or not isinstance(exp, dict):
        return ["question/expect not objects"]

    qt = q.get("type")
    if qt not in ("boolean", "noul", "choice", "score", "numeric"):
        errs.append(f"bad type {qt!r}")

    # snap Question has deny_unknown_fields
    allowed = {"type", "instructions", "criteria", "min", "max", "step",
               "granularity", "allow_abstain"}
    extra = set(q) - allowed
    if extra:
        errs.append(f"question extra keys {sorted(extra)}")

    abstain = bool(q.get("allow_abstain", False))
    if qt == "noul" and abstain:
        errs.append("noul cannot allow_abstain")

    if qt in ("choice", "score"):
        crit = q.get("criteria")
        n = (len(crit) if isinstance(crit, (dict, list)) else 0)
        if n < 2:
            errs.append(f"{qt}: <2 criteria")
        if qt == "score" and not isinstance(crit, list):
            errs.append("score: criteria must be an array")
        if qt == "score" and n + abstain > MAX_SLOTS:
            errs.append(f"score: {n} levels + abstain > {MAX_SLOTS} slots")
        if qt == "choice" and n + abstain > MAX_SLOTS:
            errs.append(f"choice: {n} options + abstain > {MAX_SLOTS} slots")
    elif qt == "numeric":
        mn, mx = q.get("min"), q.get("max")
        if not (isinstance(mn, (int, float)) and isinstance(mx, (int, float)) and mn < mx):
            errs.append("numeric: needs min<max")
        g = q.get("granularity", 8)
        if not (2 <= g <= 24):
            errs.append("numeric: granularity out of 2..=24")
        if isinstance(mn, (int, float)) and isinstance(mx, (int, float)):
            if anchors(q) + 2 + abstain > MAX_SLOTS:
                errs.append("numeric: slots exceed 26")

    # expect checks — the eval judge() keys
    if "status" in exp:
        if exp["status"] not in ("abstained", "decided", "contested", "out_of_bounds"):
            errs.append(f"bad status {exp['status']!r}")
        if exp["status"] == "abstained" and not abstain:
            errs.append("expect.status=abstained without allow_abstain")
    elif qt == "choice":
        c = exp.get("choice")
        crit = q.get("criteria")
        if isinstance(crit, dict) and c not in crit:
            errs.append("expect.choice not in criteria keys")
        # array criteria: slot key IS the option text (prompts.rs), so
        # expect.choice must equal a criteria element, not an index
        elif isinstance(crit, list) and str(c) not in [str(x) for x in crit]:
            errs.append("expect.choice not in criteria array")
    elif qt in ("boolean", "noul"):
        if not isinstance(exp.get("boolean"), bool):
            errs.append("expect.boolean missing/not bool")
    elif qt == "score":
        lv = exp.get("level")
        crit = q.get("criteria") or []
        if not (isinstance(lv, int) and 0 <= lv < len(crit)):
            errs.append("expect.level out of range")
    elif qt == "numeric":
        v = exp.get("value")
        mn, mx = q.get("min"), q.get("max")
        if not isinstance(v, (int, float)):
            errs.append("expect.value missing")
        elif isinstance(mn, (int, float)) and isinstance(mx, (int, float)) and not (mn <= v <= mx):
            errs.append("expect.value outside [min,max]")

    st_norm = norm_text(state_text(o["state"]))
    if not st_norm:
        errs.append("empty state")
    elif len(st_norm) < 15:
        errs.append("state too short (<15 normalized chars)")
    return errs


class Deduper:
    """Exact-hash dedup + banded-shingle near-dup detection."""

    def __init__(self, n_bands: int = 16, band_size: int = 4, jaccard: float = 0.6):
        self.exact: set[str] = set()
        self.bands: dict[tuple[int, int], list[int]] = {}
        self.sigs: list[set] = []
        self.n_bands, self.band_size, self.jaccard = n_bands, band_size, jaccard

    def _minhash(self, sh: set) -> list[int]:
        return [min(hash(f"{i}|{s}") for s in sh) if sh else 0 for i in range(self.n_bands * self.band_size)]

    def is_dup(self, text: str) -> bool:
        n = norm_text(text)
        h = hashlib.sha256(n.encode()).hexdigest()
        if h in self.exact:
            return True
        sh = shingles(text)
        if not sh:
            return False
        sig = self._minhash(sh)
        cands = set()
        for b in range(self.n_bands):
            key = (b, hash(tuple(sig[b * self.band_size : (b + 1) * self.band_size])))
            cands.update(self.bands.get(key, []))
        for ci in cands:
            other = self.sigs[ci]
            j = len(sh & other) / max(1, len(sh | other))
            if j >= self.jaccard:
                return True
        # not a dup → index
        self.exact.add(h)
        idx = len(self.sigs)
        self.sigs.append(sh)
        for b in range(self.n_bands):
            key = (b, hash(tuple(sig[b * self.band_size : (b + 1) * self.band_size])))
            self.bands.setdefault(key, []).append(idx)
        return False


def load_eval_states(path: str) -> list[tuple[set, str]]:
    """[(shingles, normalized_text)] per eval state."""
    out = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            o = json.loads(line)
            txt = state_text(o.get("state", ""))
            out.append((shingles(txt), norm_text(txt)))
        except Exception:
            pass
    return out


def validate_file(path: str, dedup: Deduper, eval_pairs: list) -> dict:
    ok, bad, dups, contam = 0, [], 0, 0
    kept = []
    for i, line in enumerate(open(path)):
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except Exception as e:
            bad.append((i, f"json: {e}"))
            continue
        errs = check_case(o)
        if errs:
            bad.append((i, errs))
            continue
        st = state_text(o["state"])
        # dedup key = state + question: multi-question rows (converted HF data)
        # legitimately share a state
        qq = o["question"]
        dedup_text = st + " || " + str(qq.get("instructions", "")) + str(qq.get("criteria", ""))
        if dedup.is_dup(dedup_text):
            dups += 1
            continue
        sh = shingles(st)
        norm = norm_text(st)
        # contamination: long states get Jaccard on 5-word shingles (the
        # >=60-char floor — shorter texts shingle too coarsely and
        # false-positive) PLUS exact containment both ways — an eval text
        # verbatim inside a longer candidate state is still a twin, and a
        # >=30-char normalized substring can't false-positive; short states
        # get normalized exact containment in either direction — the eval
        # median state is ~56 chars, so a shingles-only check would leave
        # most of the eval uncovered
        if eval_pairs and len(norm) >= 15:
            hit = False
            for ev_sh, ev_txt in eval_pairs:
                if len(norm) >= 60:
                    j = len(sh & ev_sh) / max(1, len(sh | ev_sh))
                    hit = (j >= 0.5
                           or (len(ev_txt) >= 30 and ev_txt in norm)
                           or norm in ev_txt)
                else:
                    hit = norm in ev_txt or (len(ev_txt) >= 15
                                             and ev_txt in norm)
                if hit:
                    break
            if hit:
                contam += 1
                continue
        kept.append(o)
    return {"file": path, "ok": len(kept), "schema_bad": len(bad),
            "dupes": dups, "contaminated": contam, "errors": bad[:10],
            "kept": kept}


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    p.add_argument("--eval", default=str(Path(__file__).resolve().parents[2] / "eval" / "cases.jsonl"))
    p.add_argument("--jaccard", type=float, default=0.6,
                   help="near-dup Jaccard threshold; ~0.9 for curated HF data")
    p.add_argument("--write", help="write cleaned merged output to this file")
    args = p.parse_args()

    eval_pairs = load_eval_states(args.eval)
    dedup = Deduper(jaccard=args.jaccard)
    all_kept, stats = [], []
    for f in args.files:
        r = validate_file(f, dedup, eval_pairs)
        stats.append(r)
        all_kept.extend(r.pop("kept"))
        print(f"{Path(f).name}: ok={r['ok']} bad={r['schema_bad']} dup={r['dupes']} contam={r['contaminated']}")
        for i, e in r["errors"]:
            print(f"   line {i}: {e}")
    print(f"\nTOTAL kept: {len(all_kept)}")
    if args.write and all_kept:
        with open(args.write, "x") as fh:
            for o in all_kept:
                fh.write(json.dumps(o, ensure_ascii=False) + "\n")
        print(f"wrote {args.write}")


if __name__ == "__main__":
    main()
