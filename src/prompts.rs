//! Prompt compilation: state + typed question -> one user message ending on a letter slot.

use serde_json::Value;

use crate::schema::{QType, Question};

pub const ABSTAIN: &str = "__abstain__";
pub const BELOW: &str = "__below__";
pub const ABOVE: &str = "__above__";

/// Bump when the prompt format changes: calibration files bind to it.
pub const PROMPT_VERSION: u32 = 1;

pub const SYSTEM: &str = "You are a decision engine. Given a state and a question, you evaluate the options and reply with only the letter of the best option. Never explain.";

const ABSTAIN_TEXT: &str = "None of the above / insufficient information in the state";

pub const LETTERS: &[u8; 26] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ";

#[derive(Debug, Clone)]
pub struct Slot {
    pub key: String,   // external identifier echoed back in the answer
    pub text: String,  // description shown next to the letter
    pub special: bool, // abstain / out-of-range marker, not a real option
}

impl Slot {
    fn new(key: impl Into<String>, text: impl Into<String>) -> Self {
        Slot {
            key: key.into(),
            text: text.into(),
            special: false,
        }
    }
    fn special(key: impl Into<String>, text: impl Into<String>) -> Self {
        Slot {
            key: key.into(),
            text: text.into(),
            special: true,
        }
    }
}

pub fn render_state(state: &Value) -> String {
    match state {
        Value::String(s) => s.clone(),
        v => serde_json::to_string(v).unwrap_or_default(),
    }
}

fn fmt_g(v: f64) -> String {
    // Python's {a:g}-ish: shortest round-trip; trim a trailing ".0"
    let s = format!("{v}");
    s
}

/// Map a typed question to ordered letter slots.
pub fn slots_for(q: &Question) -> Vec<Slot> {
    let mut opts = match q.qtype {
        QType::Boolean | QType::Noul => {
            vec![Slot::new("yes", "Yes"), Slot::new("no", "No")]
        }
        QType::Choice => match q.criteria.as_ref().unwrap() {
            Value::Object(m) => m
                .iter()
                .map(|(k, v)| Slot::new(k.clone(), value_text(v)))
                .collect(),
            Value::Array(a) => a
                .iter()
                .map(|v| Slot::new(value_text(v), value_text(v)))
                .collect(),
            _ => vec![],
        },
        QType::Score => match q.criteria.as_ref().unwrap() {
            Value::Array(a) => a
                .iter()
                .enumerate()
                .map(|(i, v)| Slot::new(i.to_string(), value_text(v)))
                .collect(),
            _ => vec![],
        },
        QType::Numeric => {
            let (min, max) = (q.min.unwrap(), q.max.unwrap());
            let mut n = q.granularity.clamp(2, 24) as usize;
            if let Some(step) = q.step {
                if step > 0.0 {
                    n = (((max - min) / step).round() as usize + 1).clamp(2, 24);
                }
            }
            let mut v = Vec::with_capacity(n + 2);
            v.push(Slot::special(BELOW, format!("Below {}", fmt_g(min))));
            for i in 0..n {
                let a = min + (max - min) * i as f64 / (n - 1) as f64;
                v.push(Slot::new(fmt_g(a), fmt_g(a)));
            }
            v.push(Slot::special(ABOVE, format!("Above {}", fmt_g(max))));
            v
        }
    };
    if q.allow_abstain {
        opts.push(Slot::special(ABSTAIN, ABSTAIN_TEXT));
    }
    opts
}

pub(crate) fn value_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

pub fn user_message(state: &Value, q: &Question, slots: &[Slot]) -> String {
    let mut lines = vec![
        "STATE".to_string(),
        render_state(state),
        String::new(),
        "QUESTION".to_string(),
    ];
    if !q.instructions.is_empty() {
        lines.push(q.instructions.clone());
    }
    match q.qtype {
        QType::Boolean | QType::Noul => lines.push("Answer yes or no.".into()),
        QType::Numeric => lines.push(format!(
            "Pick the closest value in range [{}, {}], or below/above the range.",
            fmt_g(q.min.unwrap()),
            fmt_g(q.max.unwrap())
        )),
        _ => {}
    }
    lines.push(String::new());
    lines.push("OPTIONS".to_string());
    for (i, slot) in slots.iter().enumerate() {
        lines.push(format!("{}) {}", LETTERS[i] as char, slot.text));
    }
    lines.push(String::new());
    lines.push("Reply with one letter only.".to_string());
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn q(v: Value) -> Question {
        serde_json::from_value(v).unwrap()
    }

    #[test]
    fn boolean_slots() {
        let s = slots_for(&q(json!({"type": "boolean"})));
        assert_eq!(s.len(), 3); // yes, no, abstain
        assert_eq!(s[0].key, "yes");
        assert_eq!(s[1].key, "no");
        assert!(s[2].special && s[2].key == ABSTAIN);
        // no abstain when disabled
        let s = slots_for(&q(json!({"type": "boolean", "allow_abstain": false})));
        assert_eq!(s.len(), 2);
    }

    #[test]
    fn choice_slots_keys() {
        let s = slots_for(&q(json!({
            "type": "choice", "allow_abstain": false,
            "criteria": {"a": "desc a", "b": "desc b"}
        })));
        assert_eq!(s.len(), 2);
        assert_eq!(s[0].key, "a");
        assert_eq!(s[0].text, "desc a");
    }

    #[test]
    fn score_slots_numeric_keys() {
        let s = slots_for(&q(json!({
            "type": "score", "allow_abstain": false,
            "criteria": ["low", "mid", "high"]
        })));
        assert_eq!(s.len(), 3);
        assert_eq!(s[0].key, "0");
        assert_eq!(s[2].key, "2");
        assert_eq!(s[2].text, "high");
    }

    #[test]
    fn numeric_slots_anchors() {
        let s = slots_for(&q(json!({
            "type": "numeric", "min": 0, "max": 100, "granularity": 5,
            "allow_abstain": false
        })));
        assert_eq!(s.len(), 7); // below + 5 + above
        assert!(s[0].special && s[0].key == BELOW);
        assert_eq!(s[1].key, "0");
        assert_eq!(s[5].key, "100");
        assert!(s[6].special && s[6].key == ABOVE);
    }

    #[test]
    fn numeric_step_overrides_granularity() {
        let s = slots_for(&q(json!({
            "type": "numeric", "min": 0, "max": 10, "step": 5,
            "granularity": 8, "allow_abstain": false
        })));
        // 0,5,10 -> 3 interior slots
        assert_eq!(s.len(), 5);
    }

    #[test]
    fn user_message_shape() {
        let qu = q(json!({"type": "boolean", "instructions": "Is it spam?"}));
        let s = slots_for(&qu);
        let m = user_message(&json!("hello world"), &qu, &s);
        assert!(m.starts_with("STATE\nhello world"));
        assert!(m.contains("QUESTION\nIs it spam?"));
        assert!(m.contains("A) Yes"));
        assert!(m.contains("B) No"));
        assert!(m.contains("C) None of the above"));
        assert!(m.ends_with("Reply with one letter only."));
    }
}
