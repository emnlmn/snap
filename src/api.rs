//! The single wire API: POST /v1/systemone speaks the TypeSafe/Jev format,
//! extended with optional snap fields (numeric questions, allow_abstain,
//! mode). Jev clients send the subset and get Jev semantics by default.

use serde::Deserialize;
use serde_json::{json, Map, Value};

use crate::schema::{default_layout, default_mode, DecideRequest, Layout, Mode};

/// Jev wire format, tolerant of extra SDK keys. Optional snap extensions:
/// per-question `allow_abstain` (default false), request `mode` and `layout`.
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
}

fn one() -> f64 {
    1.0
}

impl SystemoneRequest {
    /// Jev default: no abstention slot — unless a question opts in explicitly.
    pub fn to_native(&self) -> DecideRequest {
        let questions = self
            .questions
            .iter()
            .map(|(k, v)| {
                let mut v = v.clone();
                if let Value::Object(ref mut m) = v {
                    m.entry("allow_abstain").or_insert(json!(false));
                }
                (k.clone(), v)
            })
            .collect();
        DecideRequest {
            model: self.model.clone(),
            state: self.state.clone(),
            questions,
            temperature: self.temperature,
            mode: self.mode,
            layout: self.layout,
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
            } else if ans.get("choice").is_some() {
                a.insert("type".into(), json!("choice"));
                a.insert("choice".into(), ans["choice"].clone());
                a.insert("probabilities".into(), ans["probabilities"].clone());
            } else if ans.get("score").is_some() {
                a.insert("type".into(), json!("score"));
                a.insert("score".into(), ans["score"].clone());
                a.insert("level".into(), ans["level"].clone());
                a.insert("probabilities".into(), ans["probabilities"].clone());
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
        let n = r.to_native();
        assert_eq!(n.questions["q1"]["allow_abstain"], false); // Jev default
        assert_eq!(n.questions["q2"]["allow_abstain"], true); // explicit opt-in survives
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
    }

    #[test]
    fn from_native_choice_and_score() {
        let native = json!({
            "answers": {
                "c": {"status": "decided", "confidence": 0.5, "choice": "opt",
                      "probabilities": {"opt": 0.9}},
                "s": {"status": "decided", "confidence": 0.5, "score": 0.7, "level": 2,
                      "probabilities": {"a": 0.1, "b": 0.9}}
            },
            "model": "m", "usage": {}, "x_snap": {}
        });
        let out = from_native(&native);
        assert_eq!(out["answers"]["c"]["type"], "choice");
        assert_eq!(out["answers"]["c"]["choice"], "opt");
        assert_eq!(out["answers"]["s"]["type"], "score");
        assert_eq!(out["answers"]["s"]["score"], 0.7);
        assert_eq!(out["answers"]["s"]["level"], 2);
    }
}
