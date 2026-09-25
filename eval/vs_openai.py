#!/usr/bin/env python3
"""The race's JSON requests sent to a hosted OpenAI model, for the site.

Same prompts, schemas and never-seen states as eval/vs_ollama.py's json
modes, a few requests each — every call is billed. What it measures is the
wait a pipeline sees when the model it would call instead is an API away:
streamed, so the time to the first token (network, queue, prefill) is split
from the writing; one keep-alive connection, as an SDK client holds.

  OPENAI_API_KEY=... python3 eval/vs_openai.py --model <id> --out results/vs-openai.json
"""
import argparse
import datetime
import http.client
import json
import os
import statistics
import time

from vs_ollama import json_prompt, json_spec, json_valid, small_state


def strict(s):
    # structured outputs in strict mode want every object closed
    if s.get("type") == "object":
        s["additionalProperties"] = False
        for v in s["properties"].values():
            strict(v)
    return s


def call(conn, a, n, shape, state):
    ask, schema = json_spec(n, shape)
    body = {"model": a.model, "stream": True, "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": json_prompt(n, state, ask)}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "answers", "strict": True, "schema": strict(schema)}}}
    if a.effort:
        body["reasoning_effort"] = a.effort
    t0 = time.perf_counter()
    conn.request("POST", "/v1/chat/completions", json.dumps(body),
                 {"authorization": f"Bearer {os.environ['OPENAI_API_KEY']}", "content-type": "application/json"})
    r = conn.getresponse()
    if r.status != 200:
        raise SystemExit(f"{r.status} {r.read().decode()}")
    text, first, usage = "", None, {}
    for line in r:
        if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
            continue
        ev = json.loads(line[6:])
        for c in ev.get("choices", []):
            if c["delta"].get("content"):
                first = first or time.perf_counter()
                text += c["delta"]["content"]
        usage = ev.get("usage") or usage
    end = time.perf_counter()
    first = first or end
    return {"ms": (end - t0) * 1000, "ttft_ms": (first - t0) * 1000, "gen_ms": (end - first) * 1000,
            "gen_tokens": usage.get("completion_tokens", 0),
            "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0),
            "text": text, "valid": json_valid(text, n, shape)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--effort", help="reasoning_effort, for models that think before answering")
    ap.add_argument("--requests", type=int, default=5)
    ap.add_argument("--out", help="report JSON (create-only)")
    a = ap.parse_args()
    if a.out and os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists — reports are create-only")

    conn = http.client.HTTPSConnection("api.openai.com", timeout=600)
    call(conn, a, 1, "answers", small_state(10**7))  # TLS handshake out of the timings
    report = {"model": a.model, "effort": a.effort, "date": datetime.date.today().isoformat(),
              "requests": a.requests, "rows": []}
    for n in (1, 4, 8):
        row = {"questions": n}
        # states vs_ollama.py never sends (it uses 100*j for j < 5)
        for j, shape in enumerate(("answers", "snap"), 5):
            rs = [call(conn, a, n, shape, small_state(10000 * n + 100 * j + i)) for i in range(a.requests)]
            row[f"json-{shape}"] = {**{k: round(statistics.median(r[k] for r in rs), 1)
                                       for k in ("ms", "ttft_ms", "gen_ms", "gen_tokens", "reasoning_tokens")},
                                    "valid": sum(r["valid"] for r in rs), "sample": rs[0]["text"]}
        report["rows"].append(row)
        print(f"{n}q " + " | ".join(f"{m} {v['ms']:7.1f} (first token {v['ttft_ms']:.0f}, {v['gen_tokens']:.0f} tok, "
                                    f"valid {v['valid']}/{a.requests})" for m, v in row.items() if m != "questions"), flush=True)
    if a.out:
        with open(a.out, "x") as f:
            json.dump(report, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
