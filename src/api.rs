//! The single wire API: POST /v1/systemone speaks the TypeSafe/Jev format,
//! extended with optional snap fields (numeric questions, allow_abstain,
//! mode). Jev clients send the subset and get Jev semantics by default.

use serde::Deserialize;
use serde_json::{json, Map, Value};

use crate::schema::{default_layout, default_mode, DecideRequest, Expand, Layout, Mode};

/// Jev wire format, tolerant of extra SDK keys. Optional snap extensions:
/// per-question `allow_abstain` (default false, same as every other entry
/// point), request `mode`, `layout`, `expand`.
#[derive(Debug, Deserialize)]
pub struct SystemoneRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub state: Value,
    pub questions: Map<String, Value>,
    #[serde(default = "one")]
    pub temperature: f64,
    #[serde(default = "default_mode")]
    pub mode: Mode,
    #[serde(default = "default_layout")]
    pub layout: Layout,
    #[serde(default)]
    pub expand: Expand,
}

fn one() -> f64 {
    1.0
}

/// Whether `k` is a level description (a legend value) — those get re-keyed
/// by index, while special keys (`__abstain__`) pass through untouched.
fn leg_contains(legend: &Map<String, Value>, k: &str) -> bool {
    legend.values().any(|d| d.as_str() == Some(k))
}

impl SystemoneRequest {
    pub fn to_native(&self) -> DecideRequest {
        DecideRequest {
            model: self.model.clone(),
            state: self.state.clone(),
            questions: self.questions.clone(),
            temperature: self.temperature,
            mode: self.mode,
            layout: self.layout,
            expand: self.expand,
        }
    }
}

/// Jev-shaped response — noul is P(yes) as a float, each answer carries its
/// type — plus extras Jev ignores: confidence, status, probabilities and the
/// typed fields (boolean, choice, level, value) for tooling.
pub fn from_native(native: &Value) -> Value {
    let mut answers = Map::new();
    if let Some(obj) = native["answers"].as_object() {
        for (name, ans) in obj {
            let mut a = Map::new();
            a.insert("confidence".into(), ans["confidence"].clone());
            a.insert("status".into(), ans["status"].clone());
            if let Some(c) = ans.get("coverage") {
                a.insert("coverage".into(), c.clone());
            }
            if let Some(b) = ans.get("boolean") {
                a.insert("type".into(), json!("noul"));
                let p_yes = ans["probabilities"].get("yes").cloned().unwrap_or_else(|| {
                    json!(if b.as_bool().unwrap_or(false) {
                        1.0
                    } else {
                        0.0
                    })
                });
                a.insert("noul".into(), p_yes);
                a.insert("boolean".into(), b.clone());
                a.insert("choice".into(), ans["choice"].clone());
                // full slot map too — Jev ignores it; consumers need the
                // abstain mass that the scalar folds away
                a.insert("probabilities".into(), ans["probabilities"].clone());
            } else if ans.get("choice").is_some() {
                a.insert("type".into(), json!("choice"));
                a.insert("choice".into(), ans["choice"].clone());
                a.insert("probabilities".into(), ans["probabilities"].clone());
            } else if ans.get("score").is_some() {
                a.insert("type".into(), json!("score"));
                a.insert("level".into(), ans["level"].clone());
                match ans["legend"].as_object() {
                    Some(legend) if !legend.is_empty() => {
                        // Jev scale: the expected level index in [0, n-1],
                        // probabilities keyed "0".."n-1" and a legend that
                        // maps each index to its rubric text
                        let s = ans["score"].as_f64().unwrap_or(0.0) * (legend.len() - 1) as f64;
                        a.insert("score".into(), json!((s * 1e4).round() / 1e4));
                        a.insert("legend".into(), Value::Object(legend.clone()));
                        let mut probs = Map::new();
                        if let Some(src) = ans["probabilities"].as_object() {
                            for (idx, desc) in legend {
                                if let Some(v) = desc.as_str().and_then(|d| src.get(d)) {
                                    probs.insert(idx.clone(), v.clone());
                                }
                            }
                            for (k, v) in src {
                                let described = leg_contains(legend, k);
                                if !probs.contains_key(k) && !described {
                                    probs.insert(k.clone(), v.clone());
                                }
                            }
                        }
                        a.insert("probabilities".into(), Value::Object(probs));
                    }
                    _ => {
                        a.insert("score".into(), ans["score"].clone());
                        a.insert("probabilities".into(), ans["probabilities"].clone());
                    }
                }
            } else {
                a.insert("type".into(), json!("numeric"));
                a.insert(
                    "value".into(),
                    ans.get("value").cloned().unwrap_or(Value::Null),
                );
                a.insert("probabilities".into(), ans["probabilities"].clone());
            }
            answers.insert(name.clone(), Value::Object(a));
        }
    }
    json!({
        "answers": answers,
        "model": native["model"],
        "usage": native["usage"],
        "x_snap": native["x_snap"],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn to_native_abstain_defaults_false_but_opt_in() {
        let r: SystemoneRequest = serde_json::from_value(json!({
            "state": "s",
            "questions": {
                "q1": {"type": "noul", "instructions": "?"},
                "q2": {"type": "choice", "criteria": {"a": "x", "b": "y"}, "allow_abstain": true}
            }
        }))
        .unwrap();
        let qs = r.to_native().questions().unwrap();
        assert!(!qs[0].1.allow_abstain); // Jev default
        assert!(qs[1].1.allow_abstain); // explicit opt-in survives
    }

    #[test]
    fn to_native_mode_defaults_shared() {
        let r: SystemoneRequest = serde_json::from_value(json!({
            "state": "s",
            "questions": {"q": {"type": "noul"}}
        }))
        .unwrap();
        assert_eq!(r.to_native().mode, Mode::Shared);
        let r: SystemoneRequest = serde_json::from_value(json!({
            "state": "s", "mode": "direct",
            "questions": {"q": {"type": "noul"}}
        }))
        .unwrap();
        assert_eq!(r.to_native().mode, Mode::Direct);
    }

    #[test]
    fn from_native_noul_is_probability() {
        let native = json!({
            "answers": {
                "q": {
                    "status": "decided", "confidence": 0.9, "boolean": true,
                    "choice": "yes",
                    "probabilities": {"yes": 0.97, "no": 0.03}
                }
            },
            "model": "m", "usage": {}, "x_snap": {}
        });
        let out = from_native(&native);
        let a = &out["answers"]["q"];
        assert_eq!(a["type"], "noul");
        assert_eq!(a["noul"], 0.97); // P(yes) as float
        assert_eq!(a["boolean"], true); // typed extras ride along
        assert_eq!(a["choice"], "yes");
        assert_eq!(a["probabilities"]["yes"], 0.97);
    }

    #[test]
    fn from_native_choice_and_score() {
        let native = json!({
            "answers": {
                "c": {"status": "decided", "confidence": 0.5, "choice": "opt",
                      "probabilities": {"opt": 0.9}},
                "s": {"status": "decided", "confidence": 0.5, "score": 0.7, "level": 1,
                      "probabilities": {"a": 0.1, "b": 0.9, "__abstain__": 0.0},
                      "legend": {"0": "a", "1": "b"}},
                "old": {"status": "decided", "confidence": 0.5, "score": 0.7, "level": 1,
                        "probabilities": {"a": 0.1, "b": 0.9}}
            },
            "model": "m", "usage": {}, "x_snap": {}
        });
        let out = from_native(&native);
        assert_eq!(out["answers"]["c"]["type"], "choice");
        assert_eq!(out["answers"]["c"]["choice"], "opt");
        let s = &out["answers"]["s"];
        assert_eq!(s["type"], "score");
        // Jev shape: score is the expected level index, probabilities are
        // keyed "0".."n-1", legend carries the descriptions
        assert_eq!(s["score"], 0.7); // 0.7 * (2-1)
        assert_eq!(s["level"], 1);
        assert_eq!(s["legend"], json!({"0": "a", "1": "b"}));
        assert_eq!(
            s["probabilities"],
            json!({"0": 0.1, "1": 0.9, "__abstain__": 0.0})
        );
        // a native answer without legend keeps the snap shape
        let old = &out["answers"]["old"];
        assert_eq!(old["score"], 0.7);
        assert_eq!(old["probabilities"]["a"], 0.1);
    }

    #[test]
    fn score_rescales_by_level_count() {
        let native = json!({
            "answers": {"s": {"status": "decided", "confidence": 0.5, "score": 0.5,
                             "level": 2, "probabilities": {"x": 0.2, "y": 0.3, "z": 0.5},
                             "legend": {"0": "x", "1": "y", "2": "z"}}},
            "model": "m", "usage": {}, "x_snap": {}
        });
        let s = &from_native(&native)["answers"]["s"];
        assert_eq!(s["score"], 1.0); // 0.5 * (3-1)
        assert_eq!(s["probabilities"], json!({"0": 0.2, "1": 0.3, "2": 0.5}));
    }
}
