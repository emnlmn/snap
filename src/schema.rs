//! The decision request the engine consumes. The single HTTP surface
//! (/v1/systemone, Jev wire + snap extensions) deserializes into this.

use anyhow::{bail, Result};
use serde::Deserialize;
use serde_json::Value;

pub const MAX_SLOTS: usize = 26; // A-Z letters; special slots share the budget
/// Choice options past the letter budget are scored independently per option
/// and merged (two-stage choice); this caps that expansion.
pub const MAX_OPTIONS: usize = 256;
pub const MAX_QUESTIONS: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QType {
    Boolean,
    Noul,
    Choice,
    Score,
    Numeric,
}

impl QType {
    pub fn as_str(self) -> &'static str {
        match self {
            QType::Boolean => "boolean",
            QType::Noul => "noul",
            QType::Choice => "choice",
            QType::Score => "score",
            QType::Numeric => "numeric",
        }
    }
}

pub(crate) fn default_granularity() -> i64 {
    8
}
fn default_true() -> bool {
    true
}
fn default_temp() -> f64 {
    1.0
}
pub(crate) fn default_mode() -> Mode {
    Mode::Shared
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Question {
    #[serde(rename = "type")]
    pub qtype: QType,
    #[serde(default)]
    pub instructions: String,
    /// choice: object {key: description} or list of option strings;
    /// score: list of level descriptions ordered low -> high.
    #[serde(default)]
    pub criteria: Option<Value>,
    #[serde(default)]
    pub min: Option<f64>,
    #[serde(default)]
    pub max: Option<f64>,
    #[serde(default)]
    pub step: Option<f64>,
    #[serde(default = "default_granularity")]
    pub granularity: i64,
    #[serde(default = "default_true")]
    pub allow_abstain: bool,
}

impl Question {
    pub fn validate(&self) -> Result<()> {
        match self.qtype {
            QType::Choice | QType::Score => {
                let n = match &self.criteria {
                    Some(Value::Object(m)) => m.len(),
                    Some(Value::Array(a)) => a.len(),
                    _ => bail!("{}: criteria required", self.qtype.as_str()),
                };
                if self.qtype == QType::Score && !matches!(self.criteria, Some(Value::Array(_))) {
                    bail!("score: criteria must be a list of level descriptions");
                }
                if n < 2 {
                    bail!("{}: at least 2 options", self.qtype.as_str());
                }
                if self.qtype == QType::Score && n + self.allow_abstain as usize > MAX_SLOTS {
                    bail!("score: {n} levels exceed {MAX_SLOTS} letter slots");
                }
                if self.qtype == QType::Choice && n > MAX_OPTIONS {
                    bail!("choice: {n} options exceed {MAX_OPTIONS}");
                }
            }
            QType::Numeric => {
                let (min, max) = match (self.min, self.max) {
                    (Some(a), Some(b)) if a < b => (a, b),
                    _ => bail!("numeric: requires min < max"),
                };
                let _ = (min, max);
                if !(2..=24).contains(&self.granularity) {
                    bail!("numeric: granularity must be in 2..=24");
                }
                if self.granularity as usize + 2 + self.allow_abstain as usize > MAX_SLOTS {
                    bail!("numeric: granularity too high for letter slots");
                }
            }
            _ => {}
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    /// Evaluate the common state prefix once.
    Shared,
    /// Full prompt per question (verification/reference mode).
    Direct,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecideRequest {
    /// Accepted for client compatibility; the server always uses its loaded model.
    #[serde(default)]
    #[allow(dead_code)]
    pub model: Option<String>,
    pub state: Value,
    pub questions: serde_json::Map<String, Value>,
    #[serde(default = "default_temp")]
    pub temperature: f64,
    #[serde(default = "default_mode")]
    pub mode: Mode,
}

impl DecideRequest {
    pub fn questions(&self) -> Result<Vec<(String, Question)>> {
        if self.questions.is_empty() || self.questions.len() > MAX_QUESTIONS {
            bail!("questions: need 1..={MAX_QUESTIONS}");
        }
        let mut out = Vec::with_capacity(self.questions.len());
        for (k, v) in &self.questions {
            if k.contains('\u{1f}') {
                bail!("question name {k:?}: \\u001f is reserved");
            }
            let q: Question = serde_json::from_value(v.clone())?;
            q.validate()?;
            out.push((k.clone(), q));
        }
        Ok(out)
    }

    pub fn validate(&self) -> Result<()> {
        if !(self.temperature > 0.0 && self.temperature <= 10.0) {
            bail!("temperature must be in (0, 10]");
        }
        self.questions()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn q(v: Value) -> Result<Question> {
        let q: Question = serde_json::from_value(v)?;
        q.validate()?;
        Ok(q)
    }

    #[test]
    fn boolean_ok() {
        assert!(q(json!({"type": "boolean"})).is_ok());
        assert!(q(json!({"type": "noul"})).is_ok());
    }

    #[test]
    fn unknown_type_rejected() {
        assert!(serde_json::from_value::<Question>(json!({"type": "magic"})).is_err());
    }

    #[test]
    fn unknown_field_rejected() {
        assert!(
            serde_json::from_value::<Question>(json!({"type": "boolean", "question": "?"}))
                .is_err()
        );
    }

    #[test]
    fn choice_requires_criteria() {
        assert!(q(json!({"type": "choice"})).is_err());
        assert!(q(json!({"type": "choice", "criteria": {"a": "A"}})).is_err()); // <2 options
        assert!(q(json!({"type": "choice", "criteria": {"a": "A", "b": "B"}})).is_ok());
        // array criteria also allowed
        assert!(q(json!({"type": "choice", "criteria": ["A", "B"]})).is_ok());
    }

    #[test]
    fn choice_option_limit() {
        // past the letter budget a choice expands to per-option scoring
        let big: serde_json::Map<String, Value> =
            (0..256).map(|i| (format!("o{i}"), json!("x"))).collect();
        assert!(q(json!({"type": "choice", "criteria": big})).is_ok());
        let huge: serde_json::Map<String, Value> =
            (0..257).map(|i| (format!("o{i}"), json!("x"))).collect();
        assert!(q(json!({"type": "choice", "criteria": huge})).is_err());
    }

    #[test]
    fn score_slot_limit() {
        // scores stay within the letter budget: no expansion for ordinal levels
        let levels: Vec<Value> = (0..26).map(|i| json!(format!("l{i}"))).collect();
        assert!(q(json!({"type": "score", "criteria": levels})).is_err());
        let levels: Vec<Value> = (0..25).map(|i| json!(format!("l{i}"))).collect();
        assert!(q(json!({"type": "score", "criteria": levels})).is_ok());
    }

    #[test]
    fn score_requires_array_criteria() {
        assert!(q(json!({"type": "score", "criteria": {"a": "x", "b": "y"}})).is_err());
        assert!(q(json!({"type": "score", "criteria": ["low", "high"]})).is_ok());
    }

    #[test]
    fn numeric_bounds() {
        assert!(q(json!({"type": "numeric"})).is_err()); // no min/max
        assert!(q(json!({"type": "numeric", "min": 5, "max": 5})).is_err());
        assert!(q(json!({"type": "numeric", "min": 0, "max": 10})).is_ok());
        // granularity out of range
        assert!(q(json!({"type": "numeric", "min": 0, "max": 10, "granularity": 1})).is_err());
        assert!(q(json!({"type": "numeric", "min": 0, "max": 10, "granularity": 25})).is_err());
        // granularity 24 + below/above + abstain = 27 > 26
        assert!(q(json!({"type": "numeric", "min": 0, "max": 10, "granularity": 24})).is_err());
        assert!(q(json!({"type": "numeric", "min": 0, "max": 10, "granularity": 23})).is_ok());
    }

    #[test]
    fn request_validation() {
        let r: DecideRequest = serde_json::from_value(json!({
            "state": "x",
            "questions": {"q": {"type": "boolean"}}
        }))
        .unwrap();
        assert!(r.validate().is_ok());
        assert_eq!(r.temperature, 1.0);
        assert_eq!(r.mode, Mode::Shared);
    }

    #[test]
    fn bad_temperature() {
        let r: DecideRequest = serde_json::from_value(json!({
            "state": "x", "temperature": 0.0,
            "questions": {"q": {"type": "boolean"}}
        }))
        .unwrap();
        assert!(r.validate().is_err());
    }

    #[test]
    fn empty_questions_rejected() {
        let r: DecideRequest = serde_json::from_value(json!({
            "state": "x", "questions": {}
        }))
        .unwrap();
        assert!(r.validate().is_err());
    }
}
