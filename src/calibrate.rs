//! Post-hoc temperature scaling: one temperature per question type, fitted on
//! labeled eval cases. softmax(z/T) equals p^(1/T) renormalized, so the fit
//! works on the probability maps decode() already emits — no engine changes.
//!
//! A calibration file binds to (model_id, PROMPT_VERSION): the temperatures are
//! only meaningful for the exact model and prompt format they were fit on.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{bail, Context, Result};
use serde_json::{json, Map, Value};

use crate::engine::Engine;
use crate::prompts::{value_text, ABSTAIN, PROMPT_VERSION};

const MIN_SAMPLES: usize = 4;
const FOLDS: usize = 5;
const BOOT: usize = 1000;

pub struct Calibration {
    pub model: String,
    pub prompt_version: u32,
    /// question type -> fitted temperature (missing = uncalibrated, T = 1)
    pub temperatures: BTreeMap<String, f64>,
    pub fitted: usize,
    pub skipped: usize,
    /// ECE of the raw (T=1) probabilities.
    pub ece_before: f64,
    /// ECE with shipped temperatures, scored on the same data they were
    /// fit on — optimistic by construction, kept for reference.
    pub ece_after: f64,
    /// Honest generalization estimate: every row scored with a temperature
    /// fit on the OTHER folds (group-disjoint 5-fold CV).
    pub ece_oof: f64,
    /// 95% bootstrap intervals over case groups.
    pub ci_before: (f64, f64),
    pub ci_oof: (f64, f64),
    /// True when the OOF interval sits entirely below the raw one — the
    /// gain is statistically supported, not fitting noise.
    pub ci_separated: bool,
}

impl Calibration {
    /// Multiplier on the request temperature for this question type.
    pub fn temp(&self, bucket: &str) -> f64 {
        self.temperatures.get(bucket).copied().unwrap_or(1.0)
    }

    pub fn load(path: &str) -> Result<Self> {
        let v: Value = serde_json::from_str(
            &std::fs::read_to_string(path).with_context(|| format!("read {path}"))?,
        )?;
        if v["kind"] != "snap-calibration" {
            bail!("{path}: not a snap calibration file");
        }
        let mut temperatures = BTreeMap::new();
        for (k, t) in v["temperatures"].as_object().cloned().unwrap_or_default() {
            temperatures.insert(k, t.as_f64().unwrap_or(1.0));
        }
        Ok(Calibration {
            model: v["model"].as_str().unwrap_or_default().to_string(),
            prompt_version: v["prompt_version"].as_u64().unwrap_or(0) as u32,
            temperatures,
            fitted: v["fitted_cases"].as_u64().unwrap_or(0) as usize,
            skipped: 0,
            ece_before: v["ece"]["raw"]
                .as_f64()
                .or_else(|| v["ece"]["before"].as_f64())
                .unwrap_or(0.0),
            ece_after: v["ece"]["in_sample"]
                .as_f64()
                .or_else(|| v["ece"]["after"].as_f64())
                .unwrap_or(0.0),
            ece_oof: v["ece"]["out_of_fold"].as_f64().unwrap_or(0.0),
            ci_before: (0.0, 0.0),
            ci_oof: (0.0, 0.0),
            ci_separated: false,
        })
    }

    /// Refuse mismatched calibrations: a temperature fit on another model or
    /// prompt format is not transferable.
    pub fn check(&self, model_id: &str) -> Result<()> {
        if self.model != model_id {
            bail!(
                "calibration was fit on {:?}, this engine is {model_id:?}",
                self.model
            );
        }
        if self.prompt_version != PROMPT_VERSION {
            bail!(
                "calibration used prompt format v{}, this build uses v{PROMPT_VERSION}",
                self.prompt_version
            );
        }
        Ok(())
    }
}

/// noul and boolean share mechanics; expanded choice probes are booleans.
pub(crate) fn bucket(qtype: &str) -> &str {
    match qtype {
        "noul" => "boolean",
        t => t,
    }
}

/// Map a case's expectation onto a key in the answer's probability map.
/// None = the case can't supervise calibration (e.g. out_of_bounds, where the
/// correct anchor is ambiguous).
pub(crate) fn target_key(
    question: &Value,
    expect: &Value,
    probs: &Map<String, Value>,
) -> Option<String> {
    if let Some(s) = expect.get("status").and_then(|v| v.as_str()) {
        return match s {
            "abstained" if probs.contains_key(ABSTAIN) => Some(ABSTAIN.to_string()),
            _ => None,
        };
    }
    match question["type"].as_str()? {
        "choice" => expect.get("choice")?.as_str().map(String::from),
        "boolean" | "noul" => expect
            .get("boolean")?
            .as_bool()
            .map(|b| if b { "yes" } else { "no" }.to_string()),
        "score" => {
            // score probabilities are keyed by level text, not index
            let levels = question["criteria"].as_array()?;
            let idx = if let Some(l) = expect.get("level").and_then(|v| v.as_i64()) {
                l as usize
            } else {
                let s = expect.get("score")?.as_f64()?;
                (s * (levels.len() - 1) as f64).round() as usize
            };
            levels.get(idx).map(value_text)
        }
        "numeric" => {
            let target = expect.get("value")?.as_f64()?;
            probs
                .keys()
                .filter_map(|k| k.parse::<f64>().ok().map(|v| (k.clone(), v)))
                .min_by(|a, b| {
                    (a.1 - target)
                        .abs()
                        .partial_cmp(&(b.1 - target).abs())
                        .unwrap()
                })
                .map(|(k, _)| k)
        }
        _ => None,
    }
}

fn powered(probs: &[f64], t: f64) -> Vec<f64> {
    let p: Vec<f64> = probs.iter().map(|p| p.powf(1.0 / t)).collect();
    let s: f64 = p.iter().sum();
    if s > 0.0 {
        p.iter().map(|x| x / s).collect()
    } else {
        p
    }
}

fn nll(samples: &[(Vec<f64>, usize)], t: f64) -> f64 {
    samples
        .iter()
        .map(|(p, y)| -powered(p, t)[*y].max(1e-15).ln())
        .sum()
}

/// Golden-section search over ln T in [ln 0.05, ln 20].
fn fit_temperature(samples: &[(Vec<f64>, usize)]) -> f64 {
    let (mut a, mut b) = (0.05f64.ln(), 20f64.ln());
    let gr = (5f64.sqrt() - 1.0) / 2.0;
    let (mut x1, mut x2) = (b - gr * (b - a), a + gr * (b - a));
    let (mut f1, mut f2) = (nll(samples, x1.exp()), nll(samples, x2.exp()));
    for _ in 0..80 {
        if f1 > f2 {
            a = x1;
            x1 = x2;
            f1 = f2;
            x2 = a + gr * (b - a);
            f2 = nll(samples, x2.exp());
        } else {
            b = x2;
            x2 = x1;
            f2 = f1;
            x1 = b - gr * (b - a);
            f1 = nll(samples, x1.exp());
        }
    }
    ((a + b) / 2.0).exp()
}

/// FNV-1a: stable fold assignment independent of case order — rows of the
/// same case (its variants included) share the id and never leak across folds.
fn fold_of(id: &str) -> usize {
    let mut h = 0xcbf29ce484222325u64;
    for b in format!("snap-fold/{id}").bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    (h % FOLDS as u64) as usize
}

/// xorshift64*: tiny deterministic PRNG for the bootstrap (no rand dep).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545f4914f6cdd1d)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }
}

/// 95% CI for ECE by resampling whole case groups with replacement —
/// SemIf-style: groups, not rows, so correlated variants can't fake tight
/// intervals.
fn bootstrap_ci(rows: &[(usize, f64, bool)], seed: u64) -> (f64, f64) {
    if rows.is_empty() {
        return (0.0, 0.0);
    }
    let mut by_group: BTreeMap<usize, Vec<(f64, bool)>> = BTreeMap::new();
    for (g, c, ok) in rows {
        by_group.entry(*g).or_default().push((*c, *ok));
    }
    let groups: Vec<&Vec<(f64, bool)>> = by_group.values().collect();
    let mut rng = Rng(seed | 1);
    let mut vals = Vec::with_capacity(BOOT);
    for _ in 0..BOOT {
        let mut draw = Vec::new();
        for _ in 0..groups.len() {
            draw.extend(groups[rng.below(groups.len())].iter().copied());
        }
        vals.push(ece(&draw));
    }
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (vals[BOOT / 40], vals[(BOOT * 39 / 40).min(BOOT - 1)])
}

/// Expected calibration error: 10 equal-width bins over the top probability.
pub fn ece(samples: &[(f64, bool)]) -> f64 {
    const B: usize = 10;
    let (mut cnt, mut conf, mut acc) = ([0usize; B], [0f64; B], [0usize; B]);
    for (c, ok) in samples {
        let i = ((c * B as f64) as usize).min(B - 1);
        cnt[i] += 1;
        conf[i] += c;
        acc[i] += *ok as usize;
    }
    let n = samples.len() as f64;
    (0..B)
        .filter(|&i| cnt[i] > 0)
        .map(|i| {
            (cnt[i] as f64 / n) * (acc[i] as f64 / cnt[i] as f64 - conf[i] / cnt[i] as f64).abs()
        })
        .sum()
}

/// Brier score against a one-hot target: rewards calibrated distributions.
pub fn brier(probs: &[f64], target: usize) -> f64 {
    probs
        .iter()
        .enumerate()
        .map(|(i, p)| (p - if i == target { 1.0 } else { 0.0 }).powi(2))
        .sum()
}

fn argmax(probs: &[f64]) -> usize {
    probs
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Run every case through the engine (T = 1), collect (probs, target) per
/// question-type bucket, fit one temperature each. ECE is reported three
/// ways: raw, in-sample fitted (optimistic), and out-of-fold (each row
/// scored with a temperature fit on the other folds) — the OOF value is
/// the honest estimate of what calibration buys on new data.
pub fn fit(engine: &mut Engine, cases: &[Value]) -> Result<Calibration> {
    // bucket -> (case index, fold, prob vector, target index)
    type BucketSample = (usize, usize, Vec<f64>, usize);
    let mut by_bucket: BTreeMap<String, Vec<BucketSample>> = BTreeMap::new();
    let mut skipped = 0usize;
    for (ci, case) in cases.iter().enumerate() {
        let req = crate::evaluate::build_req(case, false)?;
        let out = engine.decide(&req)?;
        let ans = &out["answers"]["q"];
        let Some(probs) = ans["probabilities"].as_object() else {
            skipped += 1;
            continue;
        };
        let Some(tk) = target_key(&case["question"], &case["expect"], probs) else {
            skipped += 1;
            continue;
        };
        let Some(idx) = probs.keys().position(|k| *k == tk) else {
            skipped += 1;
            continue;
        };
        let p: Vec<f64> = probs.values().map(|v| v.as_f64().unwrap_or(0.0)).collect();
        let b = bucket(case["question"]["type"].as_str().unwrap_or("?"));
        let id = case["id"].as_str().unwrap_or_default();
        let fold = if id.is_empty() {
            ci % FOLDS
        } else {
            fold_of(id)
        };
        by_bucket
            .entry(b.to_string())
            .or_default()
            .push((ci, fold, p, idx));
    }

    let mut temperatures = BTreeMap::new();
    let (mut before, mut after, mut oof): (Vec<_>, Vec<_>, Vec<_>) = (vec![], vec![], vec![]);
    let mut fitted = 0;
    for (b, samples) in &by_bucket {
        let rows: Vec<(Vec<f64>, usize)> =
            samples.iter().map(|(_, _, p, y)| (p.clone(), *y)).collect();
        let t = if rows.len() >= MIN_SAMPLES {
            fit_temperature(&rows)
        } else {
            1.0
        };
        temperatures.insert(b.clone(), t);
        fitted += rows.len();
        for (ci, _, p, y) in samples {
            let top = argmax(p);
            before.push((*ci, p[top], top == *y));
            let pc = powered(p, t);
            let top = argmax(&pc);
            after.push((*ci, pc[top], top == *y));
        }
        // out-of-fold: score each fold's rows with a T fit on the rest
        for f in 0..FOLDS {
            let train: Vec<(Vec<f64>, usize)> = samples
                .iter()
                .filter(|(_, fo, _, _)| *fo != f)
                .map(|(_, _, p, y)| (p.clone(), *y))
                .collect();
            if train.len() < 2 {
                continue;
            }
            let tf = fit_temperature(&train);
            for (ci, fo, p, y) in samples {
                if *fo != f {
                    continue;
                }
                let pc = powered(p, tf);
                let top = argmax(&pc);
                oof.push((*ci, pc[top], top == *y));
            }
        }
    }
    let strip = |v: &[(usize, f64, bool)]| v.iter().map(|(_, c, ok)| (*c, *ok)).collect::<Vec<_>>();
    let ece_oof = if oof.is_empty() {
        ece(&strip(&after))
    } else {
        ece(&strip(&oof))
    };
    let ci_before = bootstrap_ci(&before, 0x5eed);
    let ci_oof = bootstrap_ci(if oof.is_empty() { &after } else { &oof }, 0x5eed);
    Ok(Calibration {
        model: engine.model_id.clone(),
        prompt_version: PROMPT_VERSION,
        temperatures,
        fitted,
        skipped,
        ece_before: ece(&strip(&before)),
        ece_after: ece(&strip(&after)),
        ece_oof,
        ci_before,
        ci_oof,
        ci_separated: ci_oof.1 < ci_before.0,
    })
}

/// Create-only write, same convention as eval reports.
pub fn write(cal: &Calibration, path: &str) -> Result<()> {
    let p = Path::new(path);
    if let Some(d) = p.parent() {
        std::fs::create_dir_all(d)?;
    }
    let per_type: Map<String, Value> = cal
        .temperatures
        .iter()
        .map(|(k, t)| (k.clone(), json!((t * 1e4).round() / 1e4)))
        .collect();
    let doc = json!({
        "kind": "snap-calibration",
        "model": cal.model,
        "prompt_version": cal.prompt_version,
        "fitted_cases": cal.fitted,
        "skipped_cases": cal.skipped,
        "ece": {
            "raw": cal.ece_before,
            "in_sample": cal.ece_after,
            "out_of_fold": cal.ece_oof,
            "ci95_raw": [cal.ci_before.0, cal.ci_before.1],
            "ci95_oof": [cal.ci_oof.0, cal.ci_oof.1],
            "improvement_ci_separated": cal.ci_separated,
        },
        "cv": {"folds": FOLDS, "bootstrap_samples": BOOT},
        "temperatures": per_type,
    });
    let mut f = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(p)
        .with_context(|| format!("{path} exists (reports are create-only)"))?;
    use std::io::Write;
    f.write_all(serde_json::to_string_pretty(&doc)?.as_bytes())?;
    eprintln!("calibration written: {path}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn powered_is_logit_temperature() {
        // p^(1/T) renormalized == softmax over scaled logits
        let p = powered(&[0.7, 0.2, 0.1], 2.0);
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        assert!(p[0] < 0.7); // T>1 flattens
        assert!(powered(&[0.7, 0.2, 0.1], 0.5)[0] > 0.7); // T<1 sharpens
    }

    #[test]
    fn fit_inflates_overconfident_probs() {
        // model claims 0.9 but is right only 70% of the time -> T must inflate
        // until confidence matches accuracy
        let samples: Vec<(Vec<f64>, usize)> = (0..70)
            .map(|i| (vec![0.9, 0.1], if i < 49 { 0 } else { 1 }))
            .collect();
        let t = fit_temperature(&samples);
        assert!(t > 1.3 && t < 6.0, "t={t}");
        // with perfect accuracy the same data prefers T -> sharp
        let perfect: Vec<(Vec<f64>, usize)> = (0..70).map(|_| (vec![0.9, 0.1], 0)).collect();
        assert!(fit_temperature(&perfect) < 1.0);
    }

    #[test]
    fn ece_perfect_calibration() {
        // 10 samples at conf 0.7 with 70% accuracy, all in one bin
        let s: Vec<(f64, bool)> = (0..10).map(|i| (0.7, i < 7)).collect();
        assert!(ece(&s).abs() < 1e-9);
    }

    #[test]
    fn brier_scores() {
        assert!((brier(&[1.0, 0.0], 0)).abs() < 1e-9);
        assert!((brier(&[0.5, 0.5], 0) - 0.5).abs() < 1e-9);
        assert!((brier(&[0.0, 1.0], 0) - 2.0).abs() < 1e-9);
    }

    #[test]
    fn fold_assignment_is_stable_and_bounded() {
        for id in ["route-01", "edge-03", "x"] {
            let f = fold_of(id);
            assert!(f < FOLDS);
            assert_eq!(f, fold_of(id)); // deterministic
        }
        // different ids spread across folds
        let seen: std::collections::BTreeSet<usize> =
            (0..40).map(|i| fold_of(&format!("case-{i}"))).collect();
        assert!(seen.len() > 1);
    }

    #[test]
    fn bootstrap_ci_brackets_the_point_estimate() {
        // 20 groups of 4 rows, conf 0.8, 75% accurate -> ece ~0.05
        let rows: Vec<(usize, f64, bool)> = (0..20)
            .flat_map(|g| (0..4).map(move |i| (g, 0.8, i < 3)))
            .collect();
        let (lo, hi) = bootstrap_ci(&rows, 42);
        let point = ece(&rows.iter().map(|(_, c, ok)| (*c, *ok)).collect::<Vec<_>>());
        assert!(
            lo <= point && point <= hi,
            "point {point} outside [{lo},{hi}]"
        );
        assert!(hi - lo < 0.3); // sane width at n=80
    }
}
