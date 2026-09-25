//! Latency / throughput benchmark against a running server or in-process engine.
//!
//! Scenarios:
//!   single      small state, 1 question            -> baseline per-decision latency
//!   prefix      same state, 1|4|8 questions        -> shared-prefix amortization
//!   statesize   1 question, ~8 KB state            -> prefill scaling
//!   load        R requests over C concurrent threads -> real throughput + tail latency

use std::sync::mpsc;
use std::thread;
use std::time::Instant;

use anyhow::Result;
use serde_json::{json, Map, Value};

use crate::engine::Engine;
use crate::schema::{DecideRequest, Mode};

fn small_state() -> Value {
    json!({"item": "latte", "quantity": 1})
}

fn big_state() -> Value {
    let history: Vec<Value> = (0..40)
        .map(|i| json!({"order": i, "items": ["latte", "pane", "pasta", "olio", "vino"], "total": 20 + i}))
        .collect();
    json!({
        "customer": "Mario Rossi",
        "history": history,
        "notes": "Cliente premium, preferisce bio, consegna giovedì. ".repeat(20),
    })
}

fn question_choice(variant: Option<usize>) -> Value {
    let instr = match variant {
        Some(i) => format!("Scegli il prodotto giusto per la voce della lista spesa. Mai ESAURITO. (variante {i}: ragiona sul caso {i})"),
        None => "Scegli il prodotto giusto per la voce della lista spesa. Mai ESAURITO.".into(),
    };
    json!({
        "type": "choice",
        "instructions": instr,
        "criteria": {
            "c0": "Latte intero 1L — 1.19€",
            "c1": "Latte parzialmente scremato 1L — 1.09€",
            "c2": "Latte UHT 6x1L — 6.90€",
            "c3": "Croccantini gatto 400g — ESAURITO",
            "c4": "Latte scremato 500ml — 0.69€",
        },
        "allow_abstain": false,
    })
}

fn payload(state: Value, n_questions: usize, mode: &str) -> Value {
    let mut qs = Map::new();
    for i in 0..n_questions {
        qs.insert(format!("q{i}"), question_choice(Some(i)));
    }
    json!({"state": state, "questions": qs, "mode": mode})
}

fn post(url: &str, body: &Value) -> Result<(Value, f64)> {
    let t0 = Instant::now();
    let out: Value = ureq::post(&format!("{}/v1/systemone", url.trim_end_matches('/')))
        .header("content-type", "application/json")
        .send_json(body)?
        .body_mut()
        .read_json()?;
    Ok((out, t0.elapsed().as_secs_f64() * 1000.0))
}

fn stats(mut lat: Vec<f64>, wall_s: f64) -> Map<String, Value> {
    lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = lat.len();
    let p50 = lat[n / 2];
    let p95 = lat[(n - 1).min((n as f64 * 0.95) as usize)];
    let mean = lat.iter().sum::<f64>() / n as f64;
    let mut m = Map::new();
    m.insert("n".into(), json!(n));
    m.insert("p50".into(), json!((p50 * 10.0).round() / 10.0));
    m.insert("p95".into(), json!((p95 * 10.0).round() / 10.0));
    m.insert("mean".into(), json!((mean * 10.0).round() / 10.0));
    m.insert(
        "req_s".into(),
        json!(((n as f64 / wall_s) * 100.0).round() / 100.0),
    );
    m
}

pub fn run_http(url: &str, requests: usize, concurrency: usize) -> Result<Vec<Value>> {
    let url = url.trim_end_matches('/');
    let mut results = Vec::new();

    post(url, &payload(small_state(), 1, "shared"))?; // warm-up

    let t0 = Instant::now();
    let mut lat = Vec::new();
    for _ in 0..requests {
        lat.push(post(url, &payload(small_state(), 1, "shared"))?.1);
    }
    let mut row = stats(lat, t0.elapsed().as_secs_f64());
    row.insert("scenario".into(), json!("single-1q"));
    results.push(Value::Object(row));

    for k in [4usize, 8] {
        for mode in ["shared", "direct"] {
            let t0 = Instant::now();
            let mut lat = Vec::new();
            for _ in 0..requests {
                lat.push(post(url, &payload(small_state(), k, mode))?.1);
            }
            let mut row = stats(lat, t0.elapsed().as_secs_f64());
            let mean = row["mean"].as_f64().unwrap_or(0.0);
            row.insert(
                "ms_per_question".into(),
                json!((mean / k as f64 * 10.0).round() / 10.0),
            );
            row.insert(
                "scenario".into(),
                json!(if mode == "shared" {
                    format!("prefix-{k}q")
                } else {
                    format!("direct-{k}q")
                }),
            );
            results.push(Value::Object(row));
        }
    }

    let t0 = Instant::now();
    let mut lat = Vec::new();
    for _ in 0..requests {
        lat.push(post(url, &payload(big_state(), 1, "shared"))?.1);
    }
    let mut row = stats(lat, t0.elapsed().as_secs_f64());
    row.insert("scenario".into(), json!("state-big-8k"));
    results.push(Value::Object(row));

    // concurrent load
    let t0 = Instant::now();
    let (tx, rx) = mpsc::channel();
    let total = requests * concurrency;
    let url_owned = url.to_string();
    let handles: Vec<_> = (0..concurrency)
        .map(|_| {
            let tx = tx.clone();
            let u = url_owned.clone();
            thread::spawn(move || {
                for _ in 0..requests {
                    let ms = post(&u, &payload(small_state(), 1, "shared"))
                        .map(|(_, ms)| ms)
                        .unwrap_or(-1.0);
                    let _ = tx.send(ms);
                }
            })
        })
        .collect();
    drop(tx);
    let lat: Vec<f64> = rx.iter().take(total).filter(|m| *m > 0.0).collect();
    for h in handles {
        let _ = h.join();
    }
    let mut row = stats(lat, t0.elapsed().as_secs_f64());
    row.insert("scenario".into(), json!(format!("load-c{concurrency}")));
    results.push(Value::Object(row));
    Ok(results)
}

fn local_decide(engine: &mut Engine, state: Value, n: usize, mode: Mode) -> Result<Value> {
    let mut qs = Map::new();
    for i in 0..n {
        qs.insert(format!("q{i}"), question_choice(Some(i)));
    }
    let req = DecideRequest {
        model: None,
        state,
        questions: qs,
        temperature: 1.0,
        mode,
        layout: crate::schema::Layout::Auto,
    };
    engine.decide(&req)
}

pub fn run_local(engine: &mut Engine, requests: usize) -> Result<Vec<Value>> {
    let mut results = Vec::new();
    local_decide(engine, small_state(), 1, Mode::Shared)?; // warm-up

    let t0 = Instant::now();
    let mut lat = Vec::new();
    for _ in 0..requests {
        let r = local_decide(engine, small_state(), 1, Mode::Shared)?;
        lat.push(r["x_snap"]["total_ms"].as_f64().unwrap_or(0.0));
    }
    let mut row = stats(lat, t0.elapsed().as_secs_f64());
    row.insert("scenario".into(), json!("single-1q"));
    results.push(Value::Object(row));

    for k in [4usize, 8] {
        for mode in [Mode::Shared, Mode::Direct] {
            let t0 = Instant::now();
            let mut lat = Vec::new();
            for _ in 0..requests {
                let r = local_decide(engine, small_state(), k, mode)?;
                lat.push(r["x_snap"]["total_ms"].as_f64().unwrap_or(0.0));
            }
            let mut row = stats(lat, t0.elapsed().as_secs_f64());
            let mean = row["mean"].as_f64().unwrap_or(0.0);
            row.insert(
                "ms_per_question".into(),
                json!((mean / k as f64 * 10.0).round() / 10.0),
            );
            row.insert(
                "scenario".into(),
                json!(if mode == Mode::Shared {
                    format!("prefix-{k}q")
                } else {
                    format!("direct-{k}q")
                }),
            );
            results.push(Value::Object(row));
        }
    }

    // shared vs direct must return the same answers
    let shared_ans = local_decide(engine, small_state(), 4, Mode::Shared)?["answers"].clone();
    let direct_ans = local_decide(engine, small_state(), 4, Mode::Direct)?["answers"].clone();
    let mut all_match = true;
    let mut n = 0;
    if let (Some(sa), Some(da)) = (shared_ans.as_object(), direct_ans.as_object()) {
        n = sa.len();
        for (k, v) in sa {
            let pick = |a: &Value| a.get("choice").or_else(|| a.get("boolean")).cloned();
            if pick(v) != pick(&da[k]) {
                all_match = false;
            }
        }
    }
    results.push(json!({"scenario": "mode-check", "n": n, "match": all_match}));

    let t0 = Instant::now();
    let mut lat = Vec::new();
    for _ in 0..requests {
        let r = local_decide(engine, big_state(), 1, Mode::Shared)?;
        lat.push(r["x_snap"]["total_ms"].as_f64().unwrap_or(0.0));
    }
    let mut row = stats(lat, t0.elapsed().as_secs_f64());
    row.insert("scenario".into(), json!("state-big-8k"));
    results.push(Value::Object(row));
    Ok(results)
}

pub fn print_bench(rows: &[Value]) {
    println!(
        "{:14} {:>4} {:>8} {:>8} {:>8} {:>7} {:>7}",
        "scenario", "n", "p50", "p95", "mean", "req/s", "ms/q"
    );
    for r in rows {
        if r.get("p50").is_none() {
            println!(
                "{:14} {:>4}  match={}",
                r["scenario"].as_str().unwrap_or("?"),
                r["n"],
                r["match"]
            );
            continue;
        }
        println!(
            "{:14} {:>4} {:>7} {:>7} {:>7} {:>7} {:>7}",
            r["scenario"].as_str().unwrap_or("?"),
            r["n"],
            r["p50"],
            r["p95"],
            r["mean"],
            r["req_s"],
            r.get("ms_per_question")
                .map(|v| v.to_string())
                .unwrap_or("-".into()),
        );
    }
}
