//! Benchmark runner: JSONL cases with ground-truth expectations.
//!
//! Line format:
//!   {"id": "route-01", "state": {...}, "question": {<Question fields>},
//!    "expect": {"choice"|"boolean"|"level"|"score"|"value"|"status": ..., "tol": float}}
//!
//! One question per case, keyed "q"; optional `layout`/`expand`/`compact_state`
//! pin the request knobs. `export_prompts` renders the very same requests
//! for training.

use std::io::{IsTerminal, Write};
use std::path::Path;
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use serde::de::DeserializeOwned;
use serde_json::{json, Map, Value};

use std::collections::BTreeSet;

use crate::calibrate;
use crate::engine::Engine;
use crate::schema::{DecideRequest, Layout, Mode, MAX_SLOTS};

pub fn load_cases(path: &str) -> Result<Vec<Value>> {
    let text = std::fs::read_to_string(path).with_context(|| format!("read {path}"))?;
    Ok(text
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.starts_with('#'))
        .map(serde_json::from_str)
        .collect::<Result<_, _>>()?)
}

/// None = case has no checkable expectation.
pub fn judge(ans: &Value, expect: &Value) -> Option<bool> {
    if let Some(v) = expect.get("status") {
        return Some(ans.get("status") == Some(v));
    }
    if let Some(v) = expect.get("choice") {
        return Some(ans.get("choice") == Some(v));
    }
    if let Some(v) = expect.get("boolean") {
        return Some(ans.get("boolean") == Some(v));
    }
    if let Some(v) = expect.get("level") {
        return Some(ans.get("level") == Some(v));
    }
    if let Some(v) = expect.get("score") {
        let got = ans.get("score").and_then(|x| x.as_f64()).unwrap_or(-9.0);
        let tol = expect.get("tol").and_then(|x| x.as_f64()).unwrap_or(0.34);
        return Some((got - v.as_f64()?).abs() <= tol);
    }
    if let Some(v) = expect.get("value") {
        let got = ans.get("value").and_then(|x| x.as_f64());
        let tol = expect.get("tol").and_then(|x| x.as_f64()).unwrap_or(0.0);
        return Some(got.is_some_and(|g| (g - v.as_f64().unwrap_or(f64::MAX)).abs() <= tol));
    }
    None
}

pub(crate) fn build_req(
    case: &Value,
    no_abstain: bool,
    layout: Option<Layout>,
) -> Result<DecideRequest> {
    let mut q = case["question"].clone();
    if no_abstain {
        q.as_object_mut()
            .map(|m| m.insert("allow_abstain".into(), json!(false)));
    }
    let mut questions = Map::new();
    questions.insert("q".into(), q);
    // parsed even under --layout: a malformed case is an error either way
    let pinned: Option<Layout> = pin(case, "layout")?;
    Ok(DecideRequest {
        model: None,
        state: case["state"].clone(),
        questions,
        temperature: 1.0,
        mode: Mode::Shared,
        // --layout wins; else cases may pin one for A/B runs; else auto
        layout: layout.or(pinned).unwrap_or(Layout::Auto),
        expand: pin(case, "expand")?.unwrap_or_default(),
        compact_state: pin(case, "compact_state")?.unwrap_or(false),
    })
}

/// A case's optional request knob: absent or null reads as None, anything
/// that doesn't parse is an error — a typo'd pin must not quietly run the
/// default.
fn pin<T: DeserializeOwned>(case: &Value, key: &str) -> Result<Option<T>> {
    serde_json::from_value(case[key].clone())
        .with_context(|| format!("case {}: bad {key:?}", case["id"]))
}

/// Training export: each case rendered to the exact prompt `evaluate`
/// decodes for it — same `build_req`, same compile step as `decide` — as one
/// JSONL record joined back to its case by `id`. A choice past the letter
/// budget has no single prompt: skipped, with a note on stderr. Returns
/// (written, skipped).
pub fn export_prompts(
    engine: &Engine,
    cases: &[Value],
    layout: Option<Layout>,
    out: &mut dyn Write,
) -> Result<(usize, usize)> {
    let mut ids = BTreeSet::new();
    let (mut written, mut skipped) = (0, 0);
    for (i, case) in cases.iter().enumerate() {
        let id = &case["id"];
        if id.is_null() {
            bail!(
                "case {} has no id: export records join back to their case on it",
                i + 1
            );
        }
        if !ids.insert(id.to_string()) {
            bail!("duplicate case id {id}");
        }
        let req = build_req(case, false, layout)?;
        match engine
            .render_prompt(&req)
            .with_context(|| format!("case {id}"))?
        {
            Some(mut rec) => {
                rec["id"] = id.clone();
                serde_json::to_writer(&mut *out, &rec)?;
                out.write_all(b"\n")?;
                written += 1;
            }
            None => {
                eprintln!(
                    "snap: skip {id}: a choice past {MAX_SLOTS} letters has no single prompt"
                );
                skipped += 1;
            }
        }
    }
    Ok((written, skipped))
}

fn answer_brief(ans: &Value) -> Value {
    let mut m = Map::new();
    for k in ["choice", "boolean", "level", "score", "value", "status"] {
        if let Some(v) = ans.get(k) {
            m.insert(k.into(), v.clone());
        }
    }
    Value::Object(m)
}

/// Collect (probability vector, target index) for Brier/ECE, when the case's
/// expectation maps to a key in the answer's probabilities.
fn collect_dist(case: &Value, ans: &Value, out: &mut Vec<(Vec<f64>, usize)>) {
    let Some(probs) = ans["probabilities"].as_object() else {
        return;
    };
    let Some(tk) = calibrate::target_key(&case["question"], &case["expect"], probs) else {
        return;
    };
    let Some(idx) = probs.keys().position(|k| *k == tk) else {
        return;
    };
    out.push((
        probs.values().map(|v| v.as_f64().unwrap_or(0.0)).collect(),
        idx,
    ));
}

/// The answer's pick as one comparable value; continuous answers rounded.
fn pick(ans: &Value) -> Value {
    for k in ["choice", "boolean", "level"] {
        if let Some(v) = ans.get(k) {
            if !v.is_null() {
                return v.clone();
            }
        }
    }
    for k in ["score", "value"] {
        if let Some(v) = ans.get(k).and_then(|v| v.as_f64()) {
            return json!((v * 100.0).round() / 100.0);
        }
    }
    Value::Null
}

/// Mean |Δp| over the union of both answers' probability keys.
fn drift(a: &Value, b: &Value) -> f64 {
    let (Some(pa), Some(pb)) = (
        a["probabilities"].as_object(),
        b["probabilities"].as_object(),
    ) else {
        return 0.0;
    };
    let keys: BTreeSet<&String> = pa.keys().chain(pb.keys()).collect();
    if keys.is_empty() {
        return 0.0;
    }
    keys.iter()
        .map(|k| {
            (pa.get(*k).and_then(|v| v.as_f64()).unwrap_or(0.0)
                - pb.get(*k).and_then(|v| v.as_f64()).unwrap_or(0.0))
            .abs()
        })
        .sum::<f64>()
        / keys.len() as f64
}

/// A variant applies its `state`/`question` fields over the base case.
fn merged_case(case: &Value, var: &Value) -> Value {
    let mut vc = case.clone();
    if let Some(s) = var.get("state") {
        vc["state"] = s.clone();
    }
    if let Some(vq) = var.get("question") {
        let mut qj = case["question"].clone();
        if let (Some(dst), Some(src)) = (qj.as_object_mut(), vq.as_object()) {
            for (k, v) in src {
                dst.insert(k.clone(), v.clone());
            }
        }
        vc["question"] = qj;
    }
    vc
}

const IRRELEVANT_NOTE: &str =
    "Unrelated note: a paperclip factory in another city changed its logo last Tuesday. \
     It has no bearing on the matter at hand.";

/// Stability probes generated from the case itself — no hand-writing needed.
/// SemIf-style: positional bias, meaning-preserving rewording, noise tolerance.
fn auto_variants(case: &Value) -> Vec<(&'static str, Value)> {
    let mut out = Vec::new();
    let q = &case["question"];
    if q["type"] == "choice" {
        let rev = match &q["criteria"] {
            Value::Object(m) => Some(Value::Object(
                m.iter()
                    .rev()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect(),
            )),
            Value::Array(a) => Some(Value::Array(a.iter().rev().cloned().collect())),
            _ => None,
        };
        if let Some(criteria) = rev {
            out.push((
                "option_reversal",
                json!({"question": {"criteria": criteria}}),
            ));
        }
    }
    if let Some(instr) = q["instructions"].as_str() {
        if !instr.is_empty() {
            out.push((
                "criterion_wrapper",
                json!({"question": {
                    "instructions": format!("Using only the supplied evidence, decide: {instr}")
                }}),
            ));
        }
    }
    match &case["state"] {
        Value::String(s) => out.push((
            "irrelevant_context",
            json!({"state": format!("{s}\n\n{IRRELEVANT_NOTE}")}),
        )),
        Value::Object(m) => {
            let mut m2 = m.clone();
            m2.insert("unrelated_note".into(), json!(IRRELEVANT_NOTE));
            out.push(("irrelevant_context", json!({"state": Value::Object(m2)})));
        }
        _ => {}
    }
    out
}

fn report(
    label: &str,
    rows: Vec<Value>,
    dist: Vec<(Vec<f64>, usize)>,
    cons: Vec<(&'static str, bool, f64)>,
) -> Value {
    // distribution quality: brier vs one-hot truth, ece on top-prob,
    // balanced accuracy (mean per-class recall — robust to class skew)
    let brier = if dist.is_empty() {
        Value::Null
    } else {
        json!(
            (dist
                .iter()
                .map(|(p, y)| calibrate::brier(p, *y))
                .sum::<f64>()
                / dist.len() as f64
                * 1e4)
                .round()
                / 1e4
        )
    };
    let ece_samples: Vec<(f64, bool)> = dist
        .iter()
        .map(|(p, y)| {
            let top = p
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
            (p[top], top == *y)
        })
        .collect();
    let ece = if ece_samples.is_empty() {
        Value::Null
    } else {
        json!((calibrate::ece(&ece_samples) * 1e4).round() / 1e4)
    };
    let balanced_accuracy = if dist.is_empty() {
        Value::Null
    } else {
        let mut per_class: std::collections::BTreeMap<usize, (usize, usize)> =
            std::collections::BTreeMap::new();
        for (p, y) in &dist {
            let pred = p
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
            let e = per_class.entry(*y).or_default();
            e.1 += 1;
            e.0 += (pred == *y) as usize;
        }
        let bacc = per_class
            .values()
            .map(|(ok, n)| *ok as f64 / *n as f64)
            .sum::<f64>()
            / per_class.len() as f64;
        json!((bacc * 1e4).round() / 1e4)
    };
    let cons_stats = |items: &[(&'static str, bool, f64)]| {
        json!({
            "n": items.len(),
            "agreement": (items.iter().filter(|(_, a, _)| *a).count() as f64 / items.len() as f64 * 1e4).round() / 1e4,
            "mean_drift": (items.iter().map(|(_, _, d)| d).sum::<f64>() / items.len() as f64 * 1e4).round() / 1e4,
        })
    };
    let consistency = if cons.is_empty() {
        Value::Null
    } else {
        let mut by_kind: std::collections::BTreeMap<&'static str, Vec<(&'static str, bool, f64)>> =
            std::collections::BTreeMap::new();
        for c in &cons {
            by_kind.entry(c.0).or_default().push(*c);
        }
        let kinds: Map<String, Value> = by_kind
            .iter()
            .map(|(k, v)| (k.to_string(), cons_stats(v)))
            .collect();
        let mut c = cons_stats(&cons);
        c["by_kind"] = Value::Object(kinds);
        c
    };
    let scored: Vec<&Value> = rows.iter().filter(|r| !r["ok"].is_null()).collect();
    let mut by_type: Map<String, Value> = Map::new();
    for r in &scored {
        let t = r["type"].as_str().unwrap_or("?");
        let e = by_type
            .entry(t.to_string())
            .or_insert(json!({"n": 0, "ok": 0}));
        e["n"] = json!(e["n"].as_i64().unwrap() + 1);
        e["ok"] = json!(e["ok"].as_i64().unwrap() + r["ok"].as_bool().unwrap_or(false) as i64);
    }
    let acc = if scored.is_empty() {
        0.0
    } else {
        scored
            .iter()
            .filter(|r| r["ok"].as_bool().unwrap_or(false))
            .count() as f64
            / scored.len() as f64
    };
    let conf = if scored.is_empty() {
        0.0
    } else {
        scored
            .iter()
            .map(|r| r["confidence"].as_f64().unwrap_or(0.0))
            .sum::<f64>()
            / scored.len() as f64
    };
    let timed: Vec<f64> = rows
        .iter()
        .filter_map(|r| r["ms"].as_f64())
        .filter(|m| *m > 0.0)
        .collect();
    let per_type: Map<String, Value> = by_type
        .iter()
        .map(|(k, v)| {
            (
                k.clone(),
                json!({"n": v["n"], "acc": ((v["ok"].as_f64().unwrap() / v["n"].as_f64().unwrap()) * 1000.0).round() / 1000.0}),
            )
        })
        .collect();
    let failures: Vec<Value> = scored
        .iter()
        .filter(|r| !r["ok"].as_bool().unwrap_or(false))
        .map(|r| json!({"id": r["id"], "type": r["type"], "answer": r["answer"], "confidence": r["confidence"]}))
        .collect();
    json!({
        "model": label,
        "prompt_version": crate::prompts::PROMPT_VERSION,
        "cases": rows.len(),
        "scored": scored.len(),
        "accuracy": (acc * 10000.0).round() / 10000.0,
        "balanced_accuracy": balanced_accuracy,
        "brier": brier,
        "ece": ece,
        "consistency": consistency,
        "mean_confidence": (conf * 10000.0).round() / 10000.0,
        "per_type": per_type,
        "ms_mean": if timed.is_empty() { 0.0 } else { (timed.iter().sum::<f64>() / timed.len() as f64 * 10.0).round() / 10.0 },
        "ms_total": (timed.iter().sum::<f64>() * 10.0).round() / 10.0,
        "failures": failures,
        "rows": rows,
    })
}

/// Decodes a case stands for: itself + declared variants + auto probes.
fn case_units(case: &Value, perturb: bool) -> usize {
    1 + case["variants"].as_array().map(|v| v.len()).unwrap_or(0)
        + if perturb {
            auto_variants(case).len()
        } else {
            0
        }
}

/// Live progress line on stderr, TTY only — pipes and CI stay quiet.
struct Progress {
    total: usize,
    done: usize,
    t0: Instant,
    last: Instant,
    tty: bool,
}

impl Progress {
    fn new(cases: &[Value], perturb: bool) -> Self {
        Self {
            total: cases.iter().map(|c| case_units(c, perturb)).sum(),
            done: 0,
            t0: Instant::now(),
            last: Instant::now(),
            tty: std::io::stderr().is_terminal(),
        }
    }

    /// `id` = case currently in flight; shown until the next tick.
    fn tick(&mut self, id: &str) {
        self.done += 1;
        if !self.tty || (self.done < self.total && self.last.elapsed().as_millis() < 60) {
            return;
        }
        self.last = Instant::now();
        let el = self.t0.elapsed().as_secs_f64();
        let eta = if self.done > 0 {
            el * (self.total - self.done) as f64 / self.done as f64
        } else {
            0.0
        };
        const W: usize = 26;
        let fill = self.done * W / self.total.max(1);
        let bar: String = "█".repeat(fill) + &"░".repeat(W - fill);
        eprint!(
            "\r\x1b[Keval {bar} {:>3}%  {}/{}  {:>4.0}s · ~{:.0}s left  {id}",
            self.done * 100 / self.total.max(1),
            self.done,
            self.total,
            el,
            eta,
        );
    }

    fn finish(self) {
        if self.tty {
            eprintln!();
        }
    }
}

pub fn evaluate(
    engine: &mut Engine,
    cases: &[Value],
    limit: Option<usize>,
    no_abstain: bool,
    perturb: bool,
    layout: Option<crate::schema::Layout>,
) -> Result<Value> {
    let n = limit.unwrap_or(cases.len()).min(cases.len());
    let mut prog = Progress::new(&cases[..n], perturb);
    let mut rows = Vec::new();
    let mut dist = Vec::new();
    let mut cons: Vec<(&'static str, bool, f64)> = Vec::new();
    for case in &cases[..n] {
        let cid = case["id"].as_str().unwrap_or("?");
        if no_abstain
            && case
                .get("requires_abstain")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
        {
            rows.push(
                json!({"id": case["id"], "type": case["question"]["type"], "ok": Value::Null,
                "confidence": Value::Null, "ms": 0.0, "answer": {"skipped": "requires_abstain"}}),
            );
            prog.tick(cid);
            continue;
        }
        let req = build_req(case, no_abstain, layout)?;
        let out = engine.decide(&req)?;
        prog.tick(cid);
        let ans = out["answers"]["q"].clone();
        let expect = case.get("expect").cloned().unwrap_or(json!({}));
        let ok = judge(&ans, &expect);
        collect_dist(case, &ans, &mut dist);
        if let Some(vars) = case["variants"].as_array() {
            for var in vars {
                let vreq = build_req(&merged_case(case, var), no_abstain, layout)?;
                let vans = engine.decide(&vreq)?["answers"]["q"].clone();
                prog.tick(cid);
                cons.push(("declared", pick(&ans) == pick(&vans), drift(&ans, &vans)));
            }
        }
        if perturb {
            for (kind, var) in auto_variants(case) {
                let vreq = build_req(&merged_case(case, &var), no_abstain, layout)?;
                let vans = engine.decide(&vreq)?["answers"]["q"].clone();
                prog.tick(cid);
                cons.push((kind, pick(&ans) == pick(&vans), drift(&ans, &vans)));
            }
        }
        rows.push(json!({
            "id": case["id"],
            "type": case["question"]["type"],
            "ok": ok,
            "confidence": ans.get("confidence"),
            "ms": out["x_snap"]["total_ms"],
            "answer": answer_brief(&ans),
        }));
    }
    prog.finish();
    Ok(report(&engine.model_id, rows, dist, cons))
}

pub fn evaluate_url(
    url: &str,
    cases: &[Value],
    limit: Option<usize>,
    no_abstain: bool,
    perturb: bool,
    layout: Option<crate::schema::Layout>,
) -> Result<Value> {
    let url = url.trim_end_matches('/');
    let n = limit.unwrap_or(cases.len()).min(cases.len());
    let mut prog = Progress::new(&cases[..n], perturb);
    let mut rows = Vec::new();
    let mut dist = Vec::new();
    let mut cons: Vec<(&'static str, bool, f64)> = Vec::new();
    let post = |req: &DecideRequest| -> Result<(Value, f64)> {
        let layout_str = match req.layout {
            crate::schema::Layout::Auto => "auto",
            crate::schema::Layout::StateFirst => "state_first",
            crate::schema::Layout::QuestionFirst => "question_first",
            crate::schema::Layout::Header => "header",
            crate::schema::Layout::Catalog => "catalog",
        };
        let expand_str = match req.expand {
            crate::schema::Expand::Probes => "probes",
            crate::schema::Expand::Pages => "pages",
        };
        let payload = json!({
            "state": req.state, "questions": req.questions,
            "temperature": req.temperature, "mode": "shared", "layout": layout_str,
            "expand": expand_str, "compact_state": req.compact_state,
        });
        let t0 = Instant::now();
        let out: Value = ureq::post(&format!("{url}/v1/systemone"))
            .header("content-type", "application/json")
            .send_json(&payload)?
            .body_mut()
            .read_json()?;
        Ok((out, t0.elapsed().as_secs_f64() * 1000.0))
    };
    for case in &cases[..n] {
        let cid = case["id"].as_str().unwrap_or("?");
        if no_abstain
            && case
                .get("requires_abstain")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
        {
            rows.push(
                json!({"id": case["id"], "type": case["question"]["type"], "ok": Value::Null,
                "confidence": Value::Null, "ms": 0.0, "answer": {"skipped": "requires_abstain"}}),
            );
            prog.tick(cid);
            continue;
        }
        let req = build_req(case, no_abstain, layout)?;
        let (out, ms) = post(&req)?;
        prog.tick(cid);
        let ans = out["answers"]["q"].clone();
        let expect = case.get("expect").cloned().unwrap_or(json!({}));
        let ok = judge(&ans, &expect);
        collect_dist(case, &ans, &mut dist);
        if let Some(vars) = case["variants"].as_array() {
            for var in vars {
                let vreq = build_req(&merged_case(case, var), no_abstain, layout)?;
                let (vout, _) = post(&vreq)?;
                let vans = &vout["answers"]["q"];
                prog.tick(cid);
                cons.push(("declared", pick(&ans) == pick(vans), drift(&ans, vans)));
            }
        }
        if perturb {
            for (kind, var) in auto_variants(case) {
                let vreq = build_req(&merged_case(case, &var), no_abstain, layout)?;
                let (vout, _) = post(&vreq)?;
                let vans = &vout["answers"]["q"];
                prog.tick(cid);
                cons.push((kind, pick(&ans) == pick(vans), drift(&ans, vans)));
            }
        }
        rows.push(json!({
            "id": case["id"], "type": case["question"]["type"], "ok": ok,
            "confidence": ans.get("confidence"), "ms": ms, "answer": answer_brief(&ans),
        }));
    }
    prog.finish();
    Ok(report(url, rows, dist, cons))
}

pub fn print_report(rep: &Value) {
    println!("model={} cases={}", rep["model"], rep["cases"]);
    println!(
        "accuracy {:.1}%  (balanced {:.1}%, mean confidence {:.1}%)",
        rep["accuracy"].as_f64().unwrap_or(0.0) * 100.0,
        rep["balanced_accuracy"].as_f64().unwrap_or(0.0) * 100.0,
        rep["mean_confidence"].as_f64().unwrap_or(0.0) * 100.0
    );
    if let (Some(b), Some(e)) = (rep["brier"].as_f64(), rep["ece"].as_f64()) {
        println!("brier {b:.4}   ece {e:.4}");
    }
    if let Some(c) = rep["consistency"].as_object() {
        println!(
            "consistency  {:.1}% agreement, mean drift {:.3}  (n={})",
            c["agreement"].as_f64().unwrap_or(0.0) * 100.0,
            c["mean_drift"].as_f64().unwrap_or(0.0),
            c["n"]
        );
        if let Some(kinds) = c["by_kind"].as_object() {
            for (k, s) in kinds {
                println!(
                    "  {k:18} {:.0}% agree, drift {:.3}  (n={})",
                    s["agreement"].as_f64().unwrap_or(0.0) * 100.0,
                    s["mean_drift"].as_f64().unwrap_or(0.0),
                    s["n"]
                );
            }
        }
    }
    if let Some(pt) = rep["per_type"].as_object() {
        for (t, s) in pt {
            println!(
                "  {t:8} {:.0}%  (n={})",
                s["acc"].as_f64().unwrap_or(0.0) * 100.0,
                s["n"]
            );
        }
    }
    println!(
        "latency  mean {:.0} ms/case, total {:.1} s",
        rep["ms_mean"].as_f64().unwrap_or(0.0),
        rep["ms_total"].as_f64().unwrap_or(0.0) / 1000.0
    );
    if let Some(f) = rep["failures"].as_array() {
        if !f.is_empty() {
            println!("failures:");
            let w = f
                .iter()
                .filter_map(|x| x["id"].as_str().map(str::len))
                .max()
                .unwrap_or(4);
            for x in f {
                println!(
                    "  {:w$} {:8} -> {} (conf {})",
                    x["id"].as_str().unwrap_or("?"),
                    x["type"].as_str().unwrap_or("?"),
                    x["answer"],
                    x["confidence"],
                    w = w
                );
            }
        }
    }
}

/// Create `path` and hand it to `fill`: never over an existing file (reports
/// are create-only), and a failed fill takes the file with it, so a torn
/// report can't pass for a finished run.
pub fn create_only<T>(path: &str, fill: impl FnOnce(&mut dyn Write) -> Result<T>) -> Result<T> {
    let p = Path::new(path);
    if let Some(d) = p.parent() {
        std::fs::create_dir_all(d)?;
    }
    let f = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(p)
        .map_err(|e| match e.kind() {
            std::io::ErrorKind::AlreadyExists => {
                anyhow!("{path} exists (reports are create-only)")
            }
            _ => anyhow::Error::new(e).context(format!("create {path}")),
        })?;
    let mut w = std::io::BufWriter::new(f);
    let r = fill(&mut w).and_then(|v| {
        w.flush()?;
        Ok(v)
    });
    drop(w);
    if r.is_err() {
        let _ = std::fs::remove_file(p);
    }
    r
}

pub fn write_report(rep: &Value, output: &str) -> Result<()> {
    // create-only: never overwrite a report
    create_only(output, |w| Ok(serde_json::to_writer_pretty(w, rep)?))?;
    eprintln!("report written: {output}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn judge_choice_boolean_level() {
        let a = json!({"choice": "billing", "status": "decided"});
        assert_eq!(judge(&a, &json!({"choice": "billing"})), Some(true));
        assert_eq!(judge(&a, &json!({"choice": "sales"})), Some(false));
        let b = json!({"boolean": true});
        assert_eq!(judge(&b, &json!({"boolean": true})), Some(true));
        let l = json!({"level": 3});
        assert_eq!(judge(&l, &json!({"level": 3})), Some(true));
        assert_eq!(judge(&l, &json!({"level": 2})), Some(false));
    }

    #[test]
    fn judge_status() {
        let a = json!({"status": "abstained"});
        assert_eq!(judge(&a, &json!({"status": "abstained"})), Some(true));
        assert_eq!(judge(&a, &json!({"status": "decided"})), Some(false));
    }

    #[test]
    fn judge_score_tolerance() {
        let a = json!({"score": 0.5});
        assert_eq!(judge(&a, &json!({"score": 0.5})), Some(true));
        assert_eq!(judge(&a, &json!({"score": 0.5, "tol": 0.1})), Some(true));
        assert_eq!(judge(&a, &json!({"score": 0.9, "tol": 0.1})), Some(false));
        // default tol 0.34
        assert_eq!(judge(&a, &json!({"score": 0.8})), Some(true));
        assert_eq!(judge(&a, &json!({"score": 0.9})), Some(false));
    }

    #[test]
    fn judge_value() {
        let a = json!({"value": 42.0});
        assert_eq!(judge(&a, &json!({"value": 42.0})), Some(true));
        assert_eq!(judge(&a, &json!({"value": 42.0, "tol": 0.5})), Some(true));
        assert_eq!(judge(&a, &json!({"value": 50.0, "tol": 0.5})), Some(false));
    }

    #[test]
    fn judge_no_expectation() {
        assert_eq!(judge(&json!({"choice": "x"}), &json!({})), None);
    }

    #[test]
    fn auto_variants_choice_reversal() {
        let case = json!({
            "id": "c", "state": {"x": 1},
            "question": {"type": "choice", "instructions": "Pick one",
                         "criteria": {"a": "A", "b": "B", "c": "C"}}
        });
        let vars = auto_variants(&case);
        let rev = vars.iter().find(|(k, _)| *k == "option_reversal").unwrap();
        let keys: Vec<&String> = rev.1["question"]["criteria"]
            .as_object()
            .unwrap()
            .keys()
            .collect();
        assert_eq!(keys, ["c", "b", "a"]); // reversed letter assignment
                                           // irrelevant context lands inside the object state
        let noise = vars
            .iter()
            .find(|(k, _)| *k == "irrelevant_context")
            .unwrap();
        assert!(noise.1["state"]["unrelated_note"].is_string());
        // wrapper rewords the instruction
        let wrap = vars
            .iter()
            .find(|(k, _)| *k == "criterion_wrapper")
            .unwrap();
        assert!(wrap.1["question"]["instructions"]
            .as_str()
            .unwrap()
            .contains("Pick one"));
    }

    #[test]
    fn auto_variants_string_state_and_noul() {
        let case = json!({
            "id": "n", "state": "a ticket",
            "question": {"type": "noul", "instructions": "Urgent?"}
        });
        let vars = auto_variants(&case);
        assert!(vars.iter().all(|(k, _)| *k != "option_reversal")); // not a choice
        let noise = vars
            .iter()
            .find(|(k, _)| *k == "irrelevant_context")
            .unwrap();
        assert!(noise.1["state"].as_str().unwrap().starts_with("a ticket"));
    }

    #[test]
    fn build_req_forces_no_abstain() {
        let case = json!({
            "state": "s",
            "question": {"type": "boolean"}
        });
        let r = build_req(&case, true, None).unwrap();
        assert_eq!(r.questions["q"]["allow_abstain"], false);
        let r = build_req(&case, false, None).unwrap();
        assert!(r.questions["q"].get("allow_abstain").is_none());
    }

    #[test]
    fn build_req_reads_case_pins() {
        use crate::schema::Expand;
        let case = json!({"id": "p", "state": "s", "question": {"type": "boolean"},
                          "layout": "header", "expand": "pages", "compact_state": true});
        let r = build_req(&case, false, None).unwrap();
        assert_eq!(
            (r.layout, r.expand, r.compact_state),
            (Layout::Header, Expand::Pages, true)
        );
        // --layout wins over the pin
        let r = build_req(&case, false, Some(Layout::Catalog)).unwrap();
        assert_eq!(r.layout, Layout::Catalog);
        // no pins, or null ones: the API's own defaults
        for bare in [
            json!({"id": "b", "state": "s", "question": {"type": "boolean"}}),
            json!({"id": "n", "state": "s", "question": {"type": "boolean"},
                   "layout": null, "expand": null, "compact_state": null}),
        ] {
            let r = build_req(&bare, false, None).unwrap();
            assert_eq!(
                (r.layout, r.expand, r.compact_state),
                (Layout::Auto, Expand::Probes, false)
            );
        }
    }

    #[test]
    fn build_req_rejects_bad_pins() {
        for (k, v) in [
            ("layout", json!("sideways")),
            ("expand", json!(3)),
            ("compact_state", json!("yes")),
        ] {
            let mut case = json!({"id": "x", "state": "s", "question": {"type": "boolean"}});
            case[k] = v;
            let e = build_req(&case, false, None).unwrap_err().to_string();
            assert_eq!(e, format!("case \"x\": bad \"{k}\""));
            // a forced layout doesn't hide a malformed case
            assert!(build_req(&case, false, Some(Layout::StateFirst)).is_err());
        }
    }

    /// Export `cases` on the simulated backend: (records, skipped).
    fn export(cases: &[Value], layout: Option<Layout>) -> Result<(Vec<Value>, usize)> {
        let sim = crate::kv::sim::Sim::new(1 << 16, 65, 512);
        let eng = Engine::new(Box::new(sim), "snap-sim".into())?;
        let mut buf = Vec::new();
        let (written, skipped) = export_prompts(&eng, cases, layout, &mut buf)?;
        let recs: Vec<Value> = String::from_utf8(buf)?
            .lines()
            .map(serde_json::from_str)
            .collect::<Result<_, _>>()?;
        assert_eq!(recs.len(), written);
        Ok((recs, skipped))
    }

    #[test]
    fn export_follows_case_pins_like_evaluate() {
        let noul = json!({"type": "noul", "instructions": "Spam?"});
        let big: Map<String, Value> = (0..30).map(|i| (format!("o{i}"), json!("x"))).collect();
        let cases = [
            json!({"id": "a", "state": "buy now", "question": noul}),
            json!({"id": "b", "state": "buy now", "question": noul, "layout": "header"}),
            json!({"id": "c", "state": {"k": "v"}, "question": noul, "compact_state": true}),
            json!({"id": "d", "state": "s", "question": {"type": "choice", "criteria": big}}),
        ];
        let (recs, skipped) = export(&cases, None).unwrap();
        // the over-budget choice has no single prompt: skipped, not fatal
        assert_eq!(skipped, 1);
        let ids: Vec<&str> = recs.iter().map(|r| r["id"].as_str().unwrap()).collect();
        assert_eq!(ids, ["a", "b", "c"]);
        // short state, no abstain: auto reads the question first
        assert_eq!(recs[0]["layout"], "question_first");
        assert_eq!(recs[1]["layout"], "header");
        assert!(recs[2]["prompt"]
            .as_str()
            .unwrap()
            .contains("\nSTATE\nk: v\n"));
        // --layout overrides every pin
        let (recs, _) = export(&cases, Some(Layout::Catalog)).unwrap();
        assert!(recs.iter().all(|r| r["layout"] == "catalog"));
    }

    #[test]
    fn export_refuses_cases_it_cannot_join_or_parse() {
        let case = |id: Value| json!({"id": id, "state": "s", "question": {"type": "noul"}});
        let err = |cases: &[Value]| format!("{:#}", export(cases, None).unwrap_err());
        assert!(err(&[case(json!("a")), case(json!("a"))]).contains("duplicate case id \"a\""));
        assert!(err(&[case(json!("a")), case(Value::Null)]).contains("case 2 has no id"));
        // a malformed question or pin names its case
        let bad_q = json!({"id": "q1", "state": "s", "question": {"type": "magic"}});
        assert!(err(&[bad_q]).starts_with("case \"q1\""));
        let mut bad_pin = case(json!("p1"));
        bad_pin["layout"] = json!("sideways");
        assert!(err(&[bad_pin]).starts_with("case \"p1\": bad \"layout\""));
    }

    #[test]
    fn create_only_never_overwrites_nor_leaves_a_torn_file() {
        let dir = std::env::temp_dir().join(format!("snap-test-create-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let p = dir.join("sub").join("r.json");
        let path = p.to_str().unwrap();
        create_only(path, |w| Ok(w.write_all(b"one")?)).unwrap();
        let e = create_only(path, |w| Ok(w.write_all(b"two")?)).unwrap_err();
        assert!(
            e.to_string().ends_with("exists (reports are create-only)"),
            "{e}"
        );
        assert_eq!(std::fs::read_to_string(&p).unwrap(), "one");
        // a fill that fails midway takes its file with it
        let torn = dir.join("torn.json");
        let r: Result<()> = create_only(torn.to_str().unwrap(), |w| {
            w.write_all(b"half")?;
            bail!("boom")
        });
        assert_eq!(r.unwrap_err().to_string(), "boom");
        assert!(!torn.exists());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
