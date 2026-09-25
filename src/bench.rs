//! Latency / throughput benchmark against a running server or an in-process
//! engine — both send the same Jev wire request, so every default matches
//! production. Each request carries a fresh state: a stream of new
//! documents, the realistic case, where nothing is reusable across requests
//! but the template and the question heads.
//!
//! Scenarios:
//!   single-1q      small state, 1 question         -> per-decision latency
//!   shared-{4,8}q  small state, 4|8 questions      -> batched decode
//!   direct-{4,8}q  same, mode direct               -> nothing shared (reference)
//!   doc-8q         ~1.8 KB document, 8 questions   -> document decoded once
//!   state-4k       ~4 KB state, 1 question         -> prefill scaling
//!   load-cN        R requests over C threads (HTTP only) -> throughput + tails

use std::sync::mpsc;
use std::thread;
use std::time::Instant;

use anyhow::Result;
use serde_json::{json, Map, Value};

use crate::api::{from_native, SystemoneRequest};
use crate::engine::Engine;

fn small_state(i: usize) -> Value {
    json!({"item": "latte", "quantity": 1 + i % 3, "order": i})
}

fn doc_state(i: usize) -> Value {
    let lines: Vec<String> = (0..40)
        .map(|k| {
            format!(
                "Riga {k}: {} x prodotto P{:04}, consegna zona {}.",
                1 + k % 4,
                k * 7 + i,
                k % 9
            )
        })
        .collect();
    json!(format!("Ordine {i}. {}", lines.join(" ")))
}

fn big_state(i: usize) -> Value {
    let history: Vec<Value> = (0..40)
        .map(|k| json!({"order": k, "items": ["latte", "pane", "pasta", "olio", "vino"], "total": 20 + k}))
        .collect();
    json!({
        "request": i,
        "customer": "Mario Rossi",
        "history": history,
        "notes": "Cliente premium, preferisce bio, consegna giovedì. ".repeat(20),
    })
}

fn question_choice(i: usize) -> Value {
    json!({
        "type": "choice",
        "instructions": format!("Scegli il prodotto giusto per la voce della lista spesa. Mai ESAURITO. (variante {i}: ragiona sul caso {i})"),
        "criteria": {
            "c0": "Latte intero 1L — 1.19€",
            "c1": "Latte parzialmente scremato 1L — 1.09€",
            "c2": "Latte UHT 6x1L — 6.90€",
            "c3": "Croccantini gatto 400g — ESAURITO",
            "c4": "Latte scremato 500ml — 0.69€",
        },
    })
}

fn payload(state: Value, n_questions: usize, mode: &str) -> Value {
    let qs: Map<String, Value> = (0..n_questions)
        .map(|i| (format!("q{i}"), question_choice(i)))
        .collect();
    json!({"state": state, "questions": qs, "mode": mode})
}

fn post(url: &str, body: &Value) -> Result<(Value, f64)> {
    let t0 = Instant::now();
    let out: Value = ureq::post(&format!("{url}/v1/systemone"))
        .header("content-type", "application/json")
        .send_json(body)?
        .body_mut()
        .read_json()?;
    Ok((out, t0.elapsed().as_secs_f64() * 1000.0))
}

fn stats(mut lat: Vec<f64>, wall_s: f64) -> Map<String, Value> {
    lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = lat.len();
    let r1 = |v: f64| (v * 10.0).round() / 10.0;
    let mut m = Map::new();
    m.insert("n".into(), json!(n));
    m.insert("p50".into(), json!(r1(lat[n / 2])));
    m.insert(
        "p95".into(),
        json!(r1(lat[(n - 1).min((n as f64 * 0.95) as usize)])),
    );
    m.insert("mean".into(), json!(r1(lat.iter().sum::<f64>() / n as f64)));
    m.insert(
        "req_s".into(),
        json!(((n as f64 / wall_s) * 100.0).round() / 100.0),
    );
    m
}

/// name, questions per request, request builder (index -> wire body)
type Scenario = (&'static str, usize, fn(usize) -> Value);

/// Every sequential scenario through `call` (one wire request -> the answer
/// and its latency), plus the shared-vs-direct agreement check.
fn scenarios(
    call: &mut dyn FnMut(&Value) -> Result<(Value, f64)>,
    requests: usize,
) -> Result<Vec<Value>> {
    call(&payload(small_state(usize::MAX), 8, "shared"))?; // warm-up + question heads
    let mut rows = Vec::new();
    let runs: [Scenario; 7] = [
        ("single-1q", 1, |i| payload(small_state(i), 1, "shared")),
        ("shared-4q", 4, |i| payload(small_state(i), 4, "shared")),
        ("direct-4q", 4, |i| payload(small_state(i), 4, "direct")),
        ("shared-8q", 8, |i| payload(small_state(i), 8, "shared")),
        ("direct-8q", 8, |i| payload(small_state(i), 8, "direct")),
        ("doc-8q", 8, |i| payload(doc_state(i), 8, "shared")),
        ("state-4k", 1, |i| payload(big_state(i), 1, "shared")),
    ];
    for (name, k, mk) in &runs {
        let t0 = Instant::now();
        let lat = (0..requests)
            .map(|i| call(&mk(i)).map(|r| r.1))
            .collect::<Result<Vec<f64>>>()?;
        let mut row = stats(lat, t0.elapsed().as_secs_f64());
        if *k > 1 {
            let per_q = row["mean"].as_f64().unwrap_or(0.0) / *k as f64;
            row.insert(
                "ms_per_question".into(),
                json!((per_q * 10.0).round() / 10.0),
            );
        }
        row.insert("scenario".into(), json!(name));
        rows.push(Value::Object(row));
    }
    // direct decodes every question alone: the same answers or a bug
    let shared = call(&payload(small_state(7), 4, "shared"))?.0;
    let direct = call(&payload(small_state(7), 4, "direct"))?.0;
    let pick = |a: &Value, k: &str| a["answers"][k]["choice"].clone();
    let n = shared["answers"].as_object().map_or(0, |m| m.len());
    let agree = (0..4).all(|i| pick(&shared, &format!("q{i}")) == pick(&direct, &format!("q{i}")));
    rows.push(json!({"scenario": "mode-check", "n": n, "match": agree}));
    Ok(rows)
}

pub fn run_http(url: &str, requests: usize, concurrency: usize) -> Result<Vec<Value>> {
    let url = url.trim_end_matches('/');
    let mut rows = scenarios(&mut |body| post(url, body), requests)?;
    // concurrent load: requests serialize on the engine, so this measures
    // queueing + tail latency, not parallel speedup
    let t0 = Instant::now();
    let (tx, rx) = mpsc::channel();
    for c in 0..concurrency {
        let (tx, u) = (tx.clone(), url.to_string());
        thread::spawn(move || {
            for i in 0..requests {
                let body = payload(small_state(c * requests + i), 1, "shared");
                let _ = tx.send(post(&u, &body).map_or(-1.0, |(_, ms)| ms));
            }
        });
    }
    drop(tx);
    let lat: Vec<f64> = rx.iter().filter(|m| *m > 0.0).collect();
    let mut row = stats(lat, t0.elapsed().as_secs_f64());
    row.insert("scenario".into(), json!(format!("load-c{concurrency}")));
    rows.push(Value::Object(row));
    Ok(rows)
}

pub fn run_local(engine: &mut Engine, requests: usize) -> Result<Vec<Value>> {
    scenarios(
        &mut |body| {
            let req: SystemoneRequest = serde_json::from_value(body.clone())?;
            let out = from_native(&engine.decide(&req.to_native())?);
            let ms = out["x_snap"]["total_ms"].as_f64().unwrap_or(0.0);
            Ok((out, ms))
        },
        requests,
    )
}

pub fn print_bench(rows: &[Value]) {
    println!(
        "{:14} {:>4} {:>8} {:>8} {:>8} {:>7} {:>7}",
        "scenario", "n", "p50 ms", "p95 ms", "mean ms", "req/s", "ms/q"
    );
    println!("{}", "─".repeat(62));
    let mut checks = Vec::new();
    for r in rows {
        if r.get("p50").is_none() {
            checks.push(r);
            continue;
        }
        let f = |k: &str| r[k].as_f64().unwrap_or(0.0);
        let msq = r["ms_per_question"]
            .as_f64()
            .map(|v| format!("{v:.1}"))
            .unwrap_or("-".into());
        println!(
            "{:14} {:>4} {:>8.1} {:>8.1} {:>8.1} {:>7.2} {:>7}",
            r["scenario"].as_str().unwrap_or("?"),
            r["n"],
            f("p50"),
            f("p95"),
            f("mean"),
            f("req_s"),
            msq,
        );
    }
    if checks.is_empty() {
        return;
    }
    println!();
    for r in checks {
        let ok = r["match"].as_bool().unwrap_or(false);
        println!(
            "{:14} shared/direct answers {} (n={})",
            r["scenario"].as_str().unwrap_or("?"),
            if ok { "match" } else { "MISMATCH" },
            r["n"],
        );
    }
}
