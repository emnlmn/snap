//! Pure decoding: option logits -> typed answers. No I/O, fully testable.

use serde_json::{json, Map, Value};

use crate::prompts::{Slot, ABOVE, ABSTAIN, BELOW};
use crate::schema::{QType, Question};

const UNCERTAIN_GAP: f64 = 0.15;

pub fn softmax(logits: &[f64], temperature: f64) -> Vec<f64> {
    let m = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits
        .iter()
        .map(|x| ((x - m) / temperature).exp())
        .collect();
    let s: f64 = exps.iter().sum();
    exps.iter().map(|e| e / s).collect()
}

/// Normalised top-probability: 1 = certain, 0 = uniform. Same formula as the Jev demo.
pub fn confidence(probs: &[f64]) -> f64 {
    let n = probs.len();
    if n < 2 {
        return 1.0;
    }
    let top = probs.iter().cloned().fold(0.0, f64::max);
    ((n as f64 * top - 1.0) / (n as f64 - 1.0)).max(0.0)
}

fn r6(v: f64) -> f64 {
    (v * 1e6).round() / 1e6
}

/// `logits[i]` is the (max-pooled) logit of `slots[i]`'s letter.
pub fn decode(q: &Question, slots: &[Slot], logits: &[f64], temperature: f64) -> Value {
    let probs = softmax(logits, temperature);
    let best = probs
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);
    let mut ordered = probs.clone();
    ordered.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let gap = if ordered.len() > 1 {
        ordered[0] - ordered[1]
    } else {
        1.0
    };

    let mut prob_map = Map::new();
    for (i, s) in slots.iter().enumerate() {
        if !s.special || s.key == ABSTAIN {
            prob_map.insert(s.key.clone(), json!(r6(probs[i])));
        }
    }
    let status = if slots[best].key == ABSTAIN {
        "abstained"
    } else if slots[best].key == BELOW || slots[best].key == ABOVE {
        "out_of_bounds"
    } else if gap < UNCERTAIN_GAP {
        "contested"
    } else {
        "decided"
    };

    let mut out = Map::new();
    out.insert("status".into(), json!(status));
    out.insert("probabilities".into(), Value::Object(prob_map));
    out.insert(
        "confidence".into(),
        json!((confidence(&probs) * 1e4).round() / 1e4),
    );

    match q.qtype {
        QType::Boolean | QType::Noul => {
            let mut by = (0.0, 0.0);
            for (i, s) in slots.iter().enumerate() {
                if s.key == "yes" {
                    by.0 = probs[i];
                } else if s.key == "no" {
                    by.1 = probs[i];
                }
            }
            let boolean = by.0 >= by.1;
            out.insert("boolean".into(), json!(boolean));
            out.insert("choice".into(), json!(if boolean { "yes" } else { "no" }));
        }
        QType::Choice => {
            out.insert("choice".into(), json!(slots[best].key));
        }
        QType::Score => {
            let levels: Vec<(usize, &Slot)> = slots
                .iter()
                .enumerate()
                .filter(|(_, s)| !s.special)
                .collect();
            let n = levels.len();
            let w: f64 = if n > 1 {
                levels
                    .iter()
                    .enumerate()
                    .map(|(i, (si, _))| probs[*si] * i as f64 / (n - 1) as f64)
                    .sum()
            } else {
                0.0
            };
            out.insert("score".into(), json!((w * 1e4).round() / 1e4));
            out.insert(
                "level".into(),
                if slots[best].special {
                    Value::Null
                } else {
                    slots[best]
                        .key
                        .parse::<i64>()
                        .map(Value::from)
                        .unwrap_or(Value::Null)
                },
            );
            let mut lm = Map::new();
            for (si, s) in &levels {
                lm.insert(s.text.clone(), json!(r6(probs[*si])));
            }
            out.insert("probabilities".into(), Value::Object(lm));
        }
        QType::Numeric => {
            let interior: Vec<(usize, &Slot)> = slots
                .iter()
                .enumerate()
                .filter(|(_, s)| s.key != BELOW && s.key != ABOVE && s.key != ABSTAIN)
                .collect();
            let mass: f64 = interior.iter().map(|(i, _)| probs[*i]).sum();
            let value = if mass > 0.0 {
                let v: f64 = interior
                    .iter()
                    .map(|(i, s)| probs[*i] * s.key.parse::<f64>().unwrap_or(0.0))
                    .sum::<f64>()
                    / mass;
                json!(r6(v))
            } else {
                Value::Null
            };
            out.insert("value".into(), value);
            for (i, s) in slots.iter().enumerate() {
                if s.key == BELOW || s.key == ABOVE {
                    if let Value::Object(ref mut pm) = out["probabilities"] {
                        pm.insert(s.key.clone(), json!(r6(probs[i])));
                    }
                }
            }
        }
    }
    Value::Object(out)
}

/// Merge independent per-option yes/no probabilities into one choice answer.
/// Used when a choice has more options than letter slots: each option was
/// scored on its own, P(yes) is normalized into a distribution. With
/// allow_abstain, no option reaching 0.5 means the question abstains.
pub fn merge_scored(q: &Question, keys: &[String], p_yes: &[f64]) -> Value {
    let sum: f64 = p_yes.iter().sum();
    let norm: Vec<f64> = if sum > 0.0 {
        p_yes.iter().map(|p| p / sum).collect()
    } else {
        vec![1.0 / keys.len().max(1) as f64; keys.len()]
    };
    let best = p_yes
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);
    // decisiveness is judged on the raw scores: normalised shares dilute with
    // option count, so a clear winner would always look contested
    let mut ordered = p_yes.to_vec();
    ordered.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let gap = if ordered.len() > 1 {
        ordered[0] - ordered[1]
    } else {
        1.0
    };
    let status = if q.allow_abstain && p_yes.get(best).copied().unwrap_or(0.0) < 0.5 {
        "abstained"
    } else if gap < UNCERTAIN_GAP {
        "contested"
    } else {
        "decided"
    };
    let mut prob_map = Map::new();
    for (i, k) in keys.iter().enumerate() {
        prob_map.insert(k.clone(), json!(r6(norm[i])));
    }
    Value::Object(Map::from_iter([
        ("status".into(), json!(status)),
        ("probabilities".into(), Value::Object(prob_map)),
        (
            "confidence".into(),
            json!((confidence(&norm) * 1e4).round() / 1e4),
        ),
        (
            "choice".into(),
            json!(keys.get(best).cloned().unwrap_or_default()),
        ),
    ]))
}

/// Merge paged choice answers (Expand::Pages): each page voted over ≤26
/// candidates, so an option's page-conditional probability is its share
/// within its own page. Shares are renormalized across all options — a
/// strong runner-up in a contested page can still lose to an easy winner
/// elsewhere, which is the documented trade-off of the paged mode.
pub fn merge_paged(q: &Question, keys: &[String], pages: &[Value]) -> Value {
    let mut raw: Vec<f64> = Vec::with_capacity(keys.len());
    for k in keys {
        raw.push(
            pages
                .iter()
                .map(|p| p["probabilities"][k.as_str()].as_f64().unwrap_or(0.0))
                .sum(),
        );
    }
    let sum: f64 = raw.iter().sum();
    let norm: Vec<f64> = if sum > 0.0 {
        raw.iter().map(|p| p / sum).collect()
    } else {
        vec![1.0 / keys.len().max(1) as f64; keys.len()]
    };
    let best = norm
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);
    let mut ordered = norm.clone();
    ordered.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let gap = if ordered.len() > 1 {
        ordered[0] - ordered[1]
    } else {
        1.0
    };
    // pages never abstain (no abstain slot); all-empty pages mean the
    // answers were missing — surface that as abstained, else gap-judged
    let status = if sum <= 0.0 && q.allow_abstain {
        "abstained"
    } else if gap < UNCERTAIN_GAP {
        "contested"
    } else {
        "decided"
    };
    let mut prob_map = Map::new();
    for (i, k) in keys.iter().enumerate() {
        prob_map.insert(k.clone(), json!(r6(norm[i])));
    }
    Value::Object(Map::from_iter([
        ("status".into(), json!(status)),
        ("probabilities".into(), Value::Object(prob_map)),
        (
            "confidence".into(),
            json!((confidence(&norm) * 1e4).round() / 1e4),
        ),
        (
            "choice".into(),
            json!(keys.get(best).cloned().unwrap_or_default()),
        ),
    ]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prompts::slots_for;

    fn question(v: serde_json::Value) -> Question {
        serde_json::from_value(v).unwrap()
    }

    #[test]
    fn softmax_normalises() {
        let p = softmax(&[1.0, 2.0, 3.0], 1.0);
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        assert!(p[2] > p[1] && p[1] > p[0]);
    }

    #[test]
    fn softmax_temperature() {
        let hot = softmax(&[0.0, 1.0], 10.0);
        let cold = softmax(&[0.0, 1.0], 0.1);
        assert!(hot[1] < cold[1]); // hotter -> flatter
        assert!(cold[1] > 0.99);
    }

    #[test]
    fn confidence_extremes() {
        assert_eq!(confidence(&[1.0]), 1.0);
        assert!((confidence(&[1.0, 0.0, 0.0]) - 1.0).abs() < 1e-9);
        assert_eq!(confidence(&[0.5, 0.5]), 0.0); // uniform -> 0
    }

    #[test]
    fn decode_boolean() {
        let q = question(serde_json::json!({"type": "boolean", "allow_abstain": true}));
        let slots = slots_for(&q);
        // yes, no, abstain
        let ans = decode(&q, &slots, &[5.0, 0.0, -3.0], 1.0);
        assert_eq!(ans["boolean"], true);
        assert_eq!(ans["choice"], "yes");
        assert_eq!(ans["status"], "decided");
        let ans = decode(&q, &slots, &[0.0, 5.0, -3.0], 1.0);
        assert_eq!(ans["boolean"], false);
    }

    #[test]
    fn decode_abstain_status() {
        let q = question(serde_json::json!({
            "type": "choice", "criteria": {"a": "A", "b": "B"}, "allow_abstain": true
        }));
        let slots = slots_for(&q);
        let ans = decode(&q, &slots, &[0.0, 0.0, 9.0], 1.0);
        assert_eq!(ans["status"], "abstained");
        // abstain mass is reported so callers can see how close the call was
        assert!(ans["probabilities"].get(ABSTAIN).is_some());
        // numeric reports the range markers too
        let qn = question(serde_json::json!({
            "type": "numeric", "min": 0, "max": 10, "granularity": 3, "allow_abstain": true
        }));
        let sn = slots_for(&qn);
        let an = decode(&qn, &sn, &[-9.0, 0.0, 6.0, 0.0, -9.0, 0.0], 1.0);
        assert!(an["probabilities"].get(BELOW).is_some());
        assert!(an["probabilities"].get(ABOVE).is_some());
    }

    #[test]
    fn decode_contested() {
        let q = question(serde_json::json!({"type": "boolean", "allow_abstain": false}));
        let slots = slots_for(&q);
        let ans = decode(&q, &slots, &[1.0, 1.0], 1.0); // 50/50 -> gap 0
        assert_eq!(ans["status"], "contested");
    }

    #[test]
    fn decode_choice() {
        let q = question(serde_json::json!({
            "type": "choice", "allow_abstain": false,
            "criteria": {"red": "R", "blue": "B", "green": "G"}
        }));
        let slots = slots_for(&q);
        let ans = decode(&q, &slots, &[0.0, 4.0, 1.0], 1.0);
        assert_eq!(ans["choice"], "blue");
    }

    #[test]
    fn decode_score() {
        let q = question(serde_json::json!({
            "type": "score", "allow_abstain": false,
            "criteria": ["bad", "ok", "great"]
        }));
        let slots = slots_for(&q);
        let ans = decode(&q, &slots, &[0.0, 0.0, 5.0], 1.0);
        assert_eq!(ans["level"], 2);
        let s = ans["score"].as_f64().unwrap();
        assert!(s > 0.95); // near top of [0,1]
                           // probabilities keyed by level text
        assert!(ans["probabilities"].get("great").is_some());
    }

    #[test]
    fn decode_numeric() {
        let q = question(serde_json::json!({
            "type": "numeric", "min": 0, "max": 10, "granularity": 3,
            "allow_abstain": false
        }));
        let slots = slots_for(&q);
        // slots: below, 0, 5, 10, above -> concentrate mass on "5"
        let ans = decode(&q, &slots, &[-9.0, 0.0, 6.0, 0.0, -9.0], 1.0);
        assert_eq!(ans["value"], 5.0);
    }

    #[test]
    fn decode_numeric_out_of_bounds() {
        let q = question(serde_json::json!({
            "type": "numeric", "min": 0, "max": 10, "granularity": 3,
            "allow_abstain": false
        }));
        let slots = slots_for(&q);
        let ans = decode(&q, &slots, &[9.0, 0.0, 0.0, 0.0, -9.0], 1.0);
        assert_eq!(ans["status"], "out_of_bounds");
    }

    #[test]
    fn merge_scored_picks_argmax() {
        let q = question(serde_json::json!({
            "type": "choice", "allow_abstain": false,
            "criteria": {"a": "A", "b": "B", "c": "C"}
        }));
        let keys = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let ans = merge_scored(&q, &keys, &[0.9, 0.1, 0.2]);
        assert_eq!(ans["choice"], "a");
        assert_eq!(ans["status"], "decided");
        let p = &ans["probabilities"];
        assert!((p["a"].as_f64().unwrap() - 0.75).abs() < 1e-6); // 0.9/1.2
        assert!((p["b"].as_f64().unwrap() - 0.1 / 1.2).abs() < 1e-6);
    }

    #[test]
    fn merge_scored_abstains_when_nothing_convinces() {
        let q = question(serde_json::json!({
            "type": "choice", "allow_abstain": true,
            "criteria": {"a": "A", "b": "B", "c": "C", "d": "D"}
        }));
        let keys: Vec<String> = ["a", "b", "c", "d"].iter().map(|s| s.to_string()).collect();
        let ans = merge_scored(&q, &keys, &[0.3, 0.4, 0.2, 0.1]);
        assert_eq!(ans["status"], "abstained");
        // abstain still reports the argmax so callers see the closest call
        assert_eq!(ans["choice"], "b");
    }
}
