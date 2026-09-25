#!/usr/bin/env python3
"""snap vs Ollama on the same weights, the same machine, the same requests.

Workloads follow `snap bench` (src/bench.rs): a fresh state per request and N
choice questions with 5 options each; plus a ~5 KB state with one question,
where prefill dominates and snap's edge is smallest. Every request goes to
each engine in turn, every one of them a state that engine has never seen:

  snap            POST /v1/systemone — the Jev wire, production defaults
  letter-each     Ollama, one request per question, one generated token:
                  a letter to parse, no probabilities — its fastest mode
  letter-joined   Ollama, one request, one letter per question as "A,C,B"
  json-answers    Ollama writes {"q0": "<key>", ...} under a JSON schema —
                  the least a pipeline asks a chat model for
  json-snap       Ollama writes what a snap answer carries: per question the
                  key, a confidence and a probability per option

Timed at the client over HTTP; Ollama's prompt_eval / eval durations and
token counts are kept, so its prefill/generation split is measured. One
recorded response per workload is kept as `sample` (the site replays it).

--order blocked (default) runs each engine's requests back to back, the
steady state of a deployment; --order interleaved alternates engines per
request, which leaves both sharing one GPU's caches and clocks.

  python3 eval/vs_ollama.py --snap http://localhost:8018 --out results/vs-ollama.json
"""
import argparse
import json
import os
import statistics
import time
import urllib.request

OPTIONS = {
    "c0": "Latte intero 1L — 1.19€",
    "c1": "Latte parzialmente scremato 1L — 1.09€",
    "c2": "Latte UHT 6x1L — 6.90€",
    "c3": "Croccantini gatto 400g — ESAURITO",
    "c4": "Latte scremato 500ml — 0.69€",
}
KEYS = list(OPTIONS)
LETTERS = "ABCDE"


# same builders as src/bench.rs
def small_state(i):
    return {"item": "latte", "quantity": 1 + i % 3, "order": i}


# bench.rs's 4 KB state changes only its first field between requests, and a
# prefix cache that can resume past a mismatch reuses the rest; here every
# line differs, so the whole state is new to both engines
def big_state(i):
    items = ["latte", "pane", "pasta", "olio", "vino", "riso", "uova", "caffè"]
    history = [{"order": 100 * i + k, "items": items[(i + k) % 8:] + items[:(i + k) % 8][:5 - (i + k) % 3],
                "total": 20 + (i * 7 + k * 13) % 90} for k in range(40)]
    return {"request": i, "customer": f"Cliente {i:05d}", "history": history,
            "notes": " ".join(f"Nota {i}-{k}: preferisce bio, consegna giorno {(i + k) % 28 + 1}." for k in range(16))}


def instructions(k):
    return f"Scegli il prodotto giusto per la voce della lista spesa. Mai ESAURITO. (variante {k}: ragiona sul caso {k})"


def post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"content-type": "application/json"})
    t0 = time.perf_counter()
    out = json.load(urllib.request.urlopen(req, timeout=600))
    return out, (time.perf_counter() - t0) * 1000


def snap(a, state, n):
    qs = {f"q{k}": {"type": "choice", "instructions": instructions(k), "criteria": OPTIONS} for k in range(n)}
    out, ms = post(a.snap + "/v1/systemone", {"state": state, "questions": qs})
    return {"ms": ms, "answers": {q: v["choice"] for q, v in out["answers"].items()}, "raw": out}


def chat(a, user, **extra):
    body = {"model": a.model, "stream": False, "think": False, "keep_alive": "30m",
            "options": {"temperature": 0, "seed": 0, **extra.pop("options", {})},
            "messages": [{"role": "user", "content": user}], **extra}
    out, ms = post(a.ollama + "/api/chat", body)
    return out, {"ms": ms, "text": out["message"]["content"], "prompt_tokens": out.get("prompt_eval_count", 0), "gen_tokens": out.get("eval_count", 0),
                 "prefill_ms": out.get("prompt_eval_duration", 0) / 1e6, "gen_ms": out.get("eval_duration", 0) / 1e6}


def state_text(state):
    return json.dumps(state, ensure_ascii=False)


# fixed text first, state last: Ollama's prefix cache reuses the questions
# across requests, as snap's question_first layout does
def letter_prompt(k, state):
    opts = "\n".join(f"{LETTERS[j]}) {d}" for j, d in enumerate(OPTIONS.values()))
    return f"QUESTION\n{instructions(k)}\n\nOPTIONS\n{opts}\n\nSTATE\n{state_text(state)}\n\nReply with one letter only."


def questions_block(n):
    lines = ["QUESTIONS"]
    for k in range(n):
        lines.append(f"q{k}: {instructions(k)}")
        lines += [f"  {key}: {desc}" for key, desc in OPTIONS.items()]
    return "\n".join(lines)


def letters_ok(text, n):
    got = [c for c in text.upper() if c in LETTERS]
    return len(got) == n


def letter_each(a, state, n):
    runs = [chat(a, letter_prompt(k, state), options={"num_predict": 1})[1] for k in range(n)]
    m = {key: sum(r[key] for r in runs) for key in runs[0] if key != "text"}
    m["text"] = ",".join(r["text"].strip() for r in runs)
    m["valid"] = letters_ok(m["text"], n)
    return m


def letter_joined(a, state, n):
    opts = "\n".join(f"{LETTERS[j]}) {d}" for j, d in enumerate(OPTIONS.values()))
    qs = "\n".join(f"{k + 1}. {instructions(k)}" for k in range(n))
    user = (f"QUESTIONS\n{qs}\n\nOPTIONS (the same for every question)\n{opts}\n\nSTATE\n{state_text(state)}\n\n"
            f"Reply with one letter per question, in order, comma-separated (e.g. A,C,B).")
    m = chat(a, user, options={"num_predict": 2 * n})[1]
    m["valid"] = letters_ok(m["text"], n)  # a joined reply that stops early answered nothing
    return m


def json_spec(n, shape):
    if shape == "answers":
        ask, item = "Answer every question with the key of the best option. Reply with JSON only.", \
            {"type": "string", "enum": KEYS}
    else:
        ask = ("For every question give the key of the best option, your confidence from 0 to 1, and a "
               "probability for every option key (they sum to 1). Reply with JSON only.")
        item = {"type": "object", "required": ["choice", "confidence", "probabilities"], "properties": {
            "choice": {"type": "string", "enum": KEYS}, "confidence": {"type": "number"},
            "probabilities": {"type": "object", "required": KEYS, "properties": {k: {"type": "number"} for k in KEYS}}}}
    return ask, {"type": "object", "properties": {f"q{k}": item for k in range(n)}, "required": [f"q{k}" for k in range(n)]}


def json_prompt(n, state, ask):
    return f"{questions_block(n)}\n\nSTATE\n{state_text(state)}\n\n{ask}"


def json_valid(text, n, shape):
    try:
        return len({q: (v if shape == "answers" else v["choice"]) for q, v in json.loads(text).items()}) == n
    except (ValueError, KeyError, TypeError):
        return False


def json_call(a, state, n, shape):
    ask, schema = json_spec(n, shape)
    _, m = chat(a, json_prompt(n, state, ask), format=schema)
    m["valid"] = json_valid(m["text"], n, shape)
    return m


MODES = {
    "letter-each": letter_each,
    "letter-joined": letter_joined,
    "json-answers": lambda a, s, n: json_call(a, s, n, "answers"),
    "json-snap": lambda a, s, n: json_call(a, s, n, "snap"),
}
WORKLOADS = [("1q", small_state, 1), ("4q", small_state, 4), ("8q", small_state, 8), ("5kb-1q", big_state, 1)]


def med(rows, key):
    return round(statistics.median(r[key] for r in rows), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", default="http://localhost:8018")
    ap.add_argument("--ollama", default="http://localhost:11434")
    ap.add_argument("--model", default="openbmb/minicpm5-2b")
    ap.add_argument("--requests", type=int, default=15)
    ap.add_argument("--order", choices=["blocked", "interleaved"], default="blocked")
    ap.add_argument("--out", help="report JSON (create-only)")
    a = ap.parse_args()
    if a.out and os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists — reports are create-only")

    # warm-up: both models resident, pipelines compiled
    for mode in MODES.values():
        mode(a, small_state(10**7), 8)
    snap(a, small_state(10**7), 8)

    report = {"model": a.model, "requests": a.requests, "order": a.order, "rows": []}
    calls = {"snap": snap, **MODES}
    for name, mk, n in WORKLOADS:
        runs = {m: [] for m in calls}
        # every engine and mode gets states nobody has seen: both engines keep
        # prefix caches, and a state another mode already sent would be free
        states = {m: [mk(10000 * n + 100 * j + i) for i in range(a.requests)] for j, m in enumerate(calls)}
        order = [(m, st) for m in calls for st in states[m]] if a.order == "blocked" \
            else [(m, states[m][i]) for i in range(a.requests) for m in calls]
        for m, st in order:
            runs[m].append(calls[m](a, st, n))
        row = {"workload": name, "questions": n, "snap_ms": med(runs["snap"], "ms")}
        for m in MODES:
            rs = runs[m]
            row[m] = {k: med(rs, k) for k in ("ms", "prefill_ms", "gen_ms", "gen_tokens", "prompt_tokens")}
            row[m]["valid"] = sum(r["valid"] for r in rs)  # replies that answered every question
            row[m]["sample"] = rs[0]["text"]
        row["sample"] = {"state": states["snap"][0], "snap": runs["snap"][0]["raw"]}
        report["rows"].append(row)
        print(f"{name:7s} snap {row['snap_ms']:7.1f} | " + " | ".join(
            f"{m} {row[m]['ms']:7.1f} ({row[m]['gen_tokens']:.0f} tok, valid {row[m]['valid']}/{a.requests})" for m in MODES), flush=True)
    if a.out:
        with open(a.out, "x") as f:
            json.dump(report, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
