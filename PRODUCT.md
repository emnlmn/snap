# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

- **Primary: developers evaluating snap locally.** They have just run
  `snap serve` on their own machine (Apple Silicon or a Linux/GPU box) and
  want to fire real requests, read the typed answers and probability maps,
  and check the latency and cache internals before wiring snap into a
  pipeline.
- **Secondary: people seeing snap for the first time** on a shared screen
  (demo, booth, README GIF). They need to grasp "typed decisions from one
  forward pass, fast" without touching anything.

## Product Purpose

snap (Single-pass Neural Answer Probabilities) turns unstructured state
plus typed questions into typed decisions with a full probability map. It
reads letter logits after a single forward pass and generates zero text.
The playground is the local web surface served by `snap serve`: an API
console to build and run requests, plus a self-running game that shows
decision speed.

## Positioning

A local, Jev-wire-compatible decision engine. Every answer is a probability
distribution over a fixed alphabet (no parsing, no hallucinated JSON), with
a shared state prefix evaluated once and question suffixes batched in one
decode. Deterministic, no API keys, and data never leaves the machine.

## Operating Context

- Served by the same single Rust binary that exposes `POST /v1/systemone`,
  `GET /v1/models`, `GET /healthz`. One resident model per process;
  requests serialize on the engine.
- Tested models: minicpm5-2b, spark-4b, qwen3.8-4b (see `snap models`).
- Internals already in every response under `x_snap`: `total_ms`,
  `prefill_ms`, `cached_head_tokens`, `shared_prefix_tokens`,
  `rewind` (kv|snapshot), `suffix_decode` (batched|sequential),
  `decoded_items`; `usage.output_tokens` is always 0.

## Capabilities and Constraints

- Question types: `noul`/`boolean`, `choice` (≤26 letter slots; up to 256
  options via per-option probes), `score`, `numeric`; `allow_abstain`;
  request `mode` shared|direct; `temperature`; optional calibration.
- Playground must work offline and air-gapped: no CDN fonts or scripts, no
  build step; static assets are embedded in the binary.
- Internals shown are the ones `x_snap` already returns; exposing compiled
  prompts or raw logits is out of scope for now.
- Console scope on day one: assisted form builder, raw JSON mode, copyable
  cURL.
- UI language: English.

## Brand Commitments

- Name: SNAP / snap. Mascot: the crocodile head in `assets/logo.png`
  (purple→cyan gradient on near-black). Tagline voice from the README:
  "One pass. One distribution. The jaw snaps shut."
- Voice: terse, technical, confident, a little dry.

## Evidence on Hand

- Measured latencies and accuracy tables in README.md (Apple Silicon,
  Metal).
- No customers, testimonials, or pricing exist; do not fabricate them.

## Product Principles

1. The answer is the distribution: always show the probability map, never a
   bare label.
2. Show, don't claim: latency, cache and batching numbers come from the
   live response, never from marketing copy.
3. Dev tool first: fast to operate, keyboard-friendly, nothing in the way
   of firing a request.
4. Honest limits: uncalibrated probabilities, abstentions, contested
   answers and model errors stay visible.
