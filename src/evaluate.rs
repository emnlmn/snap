//! Benchmark runner: JSONL cases with ground-truth expectations.
//!
//! Line format:
//!   {"id": "route-01", "state": {...}, "question": {<Question fields>},
//!    "expect": {"choice"|"boolean"|"level"|"score"|"value"|"status": ..., "tol": float}}
//!
//! One question per case, keyed "q".

use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result};
use serde_json::{json, Map, Value};

use std::collections::BTreeSet;

use crate::calibrate;
use crate::engine::Engine;
use crate::schema::{DecideRequest, Mode};

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
    layout: Option<crate::schema::Layout>,
) -> Result<DecideRequest> {
    let mut q = case["question"].clone();
    if no_abstain {
        q.as_object_mut()
            .map(|m| m.insert("allow_abstain".into(), json!(false)));
    }
    let mut questions = Map::new();
    questions.insert("q".into(), q);
    Ok(DecideRequest {
        model: None,
        state: case["state"].clone(),
        questions,
        temperature: 1.0,
        mode: Mode::Shared,
        // --layout wins; else cases may pin one for A/B runs; else auto
        layout: layout
            .or_else(|| serde_json::from_value(case["layout"].clone()).ok())
            .unwrap_or(crate::schema::Layout::Auto),
        expand: serde_json::from_value(case["expand"].clone()).unwrap_or_default(),
        compact_state: case["compact_state"].as_bool().unwrap_or(false),
    })
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

pub fn evaluate(
    engine: &mut Engine,
    cases: &[Value],
    limit: Option<usize>,
    no_abstain: bool,
    perturb: bool,
    layout: Option<crate::schema::Layout>,
) -> Result<Value> {
    let n = limit.unwrap_or(cases.len()).min(cases.len());
    let mut rows = Vec::new();
    let mut dist = Vec::new();
    let mut cons: Vec<(&'static str, bool, f64)> = Vec::new();
    for case in &cases[..n] {
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
            continue;
        }
        let req = build_req(case, no_abstain, layout)?;
        let out = engine.decide(&req)?;
        let ans = out["answers"]["q"].clone();
        let expect = case.get("expect").cloned().unwrap_or(json!({}));
        let ok = judge(&ans, &expect);
        collect_dist(case, &ans, &mut dist);
        if let Some(vars) = case["variants"].as_array() {
            for var in vars {
                let vreq = build_req(&merged_case(case, var), no_abstain, layout)?;
                let vans = engine.decide(&vreq)?["answers"]["q"].clone();
                cons.push(("declared", pick(&ans) == pick(&vans), drift(&ans, &vans)));
            }
        }
        if perturb {
            for (kind, var) in auto_variants(case) {
                let vreq = build_req(&merged_case(case, &var), no_abstain, layout)?;
                let vans = engine.decide(&vreq)?["answers"]["q"].clone();
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
            continue;
        }
        let req = build_req(case, no_abstain, layout)?;
        let (out, ms) = post(&req)?;
        let ans = out["answers"]["q"].clone();
        let expect = case.get("expect").cloned().unwrap_or(json!({}));
        let ok = judge(&ans, &expect);
        collect_dist(case, &ans, &mut dist);
        if let Some(vars) = case["variants"].as_array() {
            for var in vars {
                let vreq = build_req(&merged_case(case, var), no_abstain, layout)?;
                let (vout, _) = post(&vreq)?;
                let vans = &vout["answers"]["q"];
                cons.push(("declared", pick(&ans) == pick(vans), drift(&ans, vans)));
            }
        }
        if perturb {
            for (kind, var) in auto_variants(case) {
                let vreq = build_req(&merged_case(case, &var), no_abstain, layout)?;
                let (vout, _) = post(&vreq)?;
                let vans = &vout["answers"]["q"];
                cons.push((kind, pick(&ans) == pick(vans), drift(&ans, vans)));
            }
        }
        rows.push(json!({
            "id": case["id"], "type": case["question"]["type"], "ok": ok,
            "confidence": ans.get("confidence"), "ms": ms, "answer": answer_brief(&ans),
        }));
    }
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
            for x in f {
                println!(
                    "  {:14} {:8} -> {} (conf {})",
                    x["id"].as_str().unwrap_or("?"),
                    x["type"].as_str().unwrap_or("?"),
                    x["answer"],
                    x["confidence"]
                );
            }
        }
    }
}

pub fn write_report(rep: &Value, output: &str) -> Result<()> {
    let p = Path::new(output);
    if let Some(d) = p.parent() {
        std::fs::create_dir_all(d)?;
    }
    // create-only: never overwrite a report
    let mut f = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(p)
        .with_context(|| format!("{output} exists (reports are create-only)"))?;
    use std::io::Write;
    f.write_all(serde_json::to_string_pretty(rep)?.as_bytes())?;
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
}
