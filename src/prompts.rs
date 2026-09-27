//! Prompt compilation: state + typed question -> one user message ending on a letter slot.

use serde_json::Value;

use crate::schema::{Layout, QType, Question};

pub const ABSTAIN: &str = "__abstain__";
pub const BELOW: &str = "__below__";
pub const ABOVE: &str = "__above__";

/// Bump when the prompt format changes: calibration files bind to it.
/// v3: expanded probes put the candidate last (shared-stem clustering),
/// Layout::Catalog added, compact_state rendering added.
/// v4: empty choice descriptions show the key, request text tokenized
/// without special tokens.
pub const PROMPT_VERSION: u32 = 4;

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

/// yaml-lite state rendering: same data without the JSON punctuation noise —
/// "key: value" lines, `- item` lists, scalars quoted only when ambiguous.
/// Structured states cost ~10-20% fewer tokens than serde_json's output.
pub fn render_state_compact(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => {
            let mut s = String::new();
            compact_lines(other, 0, &mut s);
            s.trim_end().to_string()
        }
    }
}

/// Bare scalars: safe only when the text can't be misread as another type or
/// eat the "key: " / ", " structure; anything else falls back to JSON quoting.
fn bare_scalar(s: &str) -> bool {
    !s.is_empty()
        && !s.contains('\n')
        && !s.contains(": ")
        && !s.contains(", ")
        && !s.ends_with(':')
        && !s.starts_with(|c: char| "-[]{}\"'#,!?&*|>@`%".contains(c) || c.is_whitespace())
        && !s.ends_with(char::is_whitespace)
        && s.parse::<f64>().is_err()
        && !matches!(
            s,
            "true" | "false" | "null" | "~" | "yes" | "no" | "on" | "off"
        )
}

fn compact_scalar(v: &Value) -> String {
    match v {
        Value::String(s) if bare_scalar(s) => s.clone(),
        other => serde_json::to_string(other).unwrap_or_default(),
    }
}

/// Object-array table detection: all elements are objects sharing the same
/// scalar-valued keys, so a csv row form works — JSON repeats every key name
/// per row, which is where most of its waste lives. Returns the columns.
fn table_keys(a: &[Value]) -> Option<Vec<String>> {
    let first = a.first()?.as_object()?;
    if first.is_empty() {
        return None;
    }
    let keys: Vec<String> = first.keys().cloned().collect();
    let uniform = a.iter().all(|x| {
        x.as_object().is_some_and(|m| {
            m.len() == keys.len()
                && keys
                    .iter()
                    .all(|k| m.get(k).is_some_and(|v| !v.is_object() && !v.is_array()))
        })
    });
    uniform.then_some(keys)
}

fn compact_lines(v: &Value, ind: usize, out: &mut String) {
    let pad = "  ".repeat(ind);
    match v {
        Value::Object(m) => {
            for (k, val) in m {
                out.push_str(&pad);
                out.push_str(k);
                match val {
                    Value::Object(_) | Value::Array(_) => {
                        out.push_str(":\n");
                        compact_lines(val, ind + 1, out);
                    }
                    _ => {
                        out.push_str(": ");
                        out.push_str(&compact_scalar(val));
                        out.push('\n');
                    }
                }
            }
        }
        Value::Array(a) => {
            if let Some(cols) = table_keys(a) {
                // uniform objects: column header once, then bare csv rows
                out.push_str(&format!("{pad}({})\n", cols.join(", ")));
                for x in a {
                    let m = x.as_object().unwrap();
                    let row: Vec<String> = cols.iter().map(|k| compact_scalar(&m[k])).collect();
                    out.push_str(&format!("{pad}{}\n", row.join(", ")));
                }
            } else if a.iter().all(|x| !x.is_object() && !x.is_array()) {
                // scalar lists inline — "key: a, b, c" beats a dash per item
                let row: Vec<String> = a.iter().map(compact_scalar).collect();
                out.push_str(&format!("{pad}{}\n", row.join(", ")));
            } else {
                for x in a {
                    out.push_str(&format!("{pad}- {}\n", compact_scalar(x)));
                }
            }
        }
        _ => {
            out.push_str(&pad);
            out.push_str(&compact_scalar(v));
            out.push('\n');
        }
    }
}

/// Shortest round-trip, deliberately not %g: anchors like
/// `3.3333333333333335` look noisy, but rounding them to 6 or 3 significant
/// digits lost 1-3 numeric cases per model on eval/{core,edge} (none gained).
fn fmt_g(v: f64) -> String {
    format!("{v}")
}

/// Choice option label: the description alone — keys are often opaque
/// (`p0`), and `key — desc` measured worse on eval/cases (route-04, pick-05/07
/// flip on 2B and 4B). An empty description falls back to the key.
fn option_label(key: &str, desc: &str) -> String {
    match desc {
        "" => key.to_string(),
        d => d.to_string(),
    }
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
                .map(|(k, v)| Slot::new(k.clone(), option_label(k, &value_text(v))))
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
            let n = q.anchors();
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

/// The QUESTION + OPTIONS block shared by every layout.
pub fn question_block(q: &Question, slots: &[Slot]) -> Vec<String> {
    let mut lines = vec!["QUESTION".to_string()];
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
    lines
}

/// One entry of the QUESTIONS catalog used by Layout::Catalog: the numbered
/// question (name + instructions). Options live in the item tail instead —
/// letters read most reliably right next to the answer position.
pub fn catalog_entry(i: usize, name: &str, q: &Question) -> Vec<String> {
    let mut head = format!("{}) {}", i + 1, name);
    if !q.instructions.is_empty() {
        head.push_str(" — ");
        head.push_str(&q.instructions);
    }
    let mut lines = vec![head];
    if q.qtype == QType::Numeric {
        lines.push(format!(
            "   Pick the closest value in range [{}, {}], or below/above the range.",
            fmt_g(q.min.unwrap()),
            fmt_g(q.max.unwrap())
        ));
    }
    lines
}

/// Catalog layout: the question set (numbered, instructions only) sits before
/// the state so the model reads it with all questions in view; the per-item
/// tail carries the name pointer + the OPTIONS block right at the answer
/// position. The question-independent `[head+catalog]` span caches across
/// requests on the same question set.
pub fn catalog_message(
    cat: &[String],
    state: &str,
    idx: usize,
    name: &str,
    slots: &[Slot],
) -> String {
    let mut lines: Vec<String> = vec![
        "QUESTIONS".to_string(),
        "You will be asked each of these questions about the state below, one at a time, by number. \
         Read the state with all of them in mind."
            .to_string(),
    ];
    lines.extend(cat.iter().cloned());
    lines.push(String::new());
    lines.push("STATE".to_string());
    lines.push(state.to_string());
    lines.push(String::new());
    lines.push(format!("QUESTION {} — {}", idx + 1, name));
    lines.push(String::new());
    lines.push("OPTIONS".to_string());
    for (i, slot) in slots.iter().enumerate() {
        lines.push(format!("{}) {}", LETTERS[i] as char, slot.text));
    }
    lines.push(String::new());
    lines.push("Reply with one letter only.".to_string());
    lines.join("\n")
}

/// `preamble` lists every original question of the request ("1. name — instr")
/// and is only used by Layout::Header: the state is then encoded with all of
/// them in view while staying a single shared prefix.
pub fn user_message(
    state: &str,
    q: &Question,
    slots: &[Slot],
    layout: Layout,
    preamble: &[String],
) -> String {
    let qblock = question_block(q, slots);
    let mut lines: Vec<String> = Vec::new();
    match layout {
        Layout::QuestionFirst => {
            lines.extend(qblock);
            lines.push(String::new());
            lines.push("STATE".to_string());
            lines.push(state.to_string());
        }
        Layout::Header => {
            lines.push("QUESTIONS".to_string());
            lines.push(
                "You will be asked each of these questions about the state below, one at a time. \
                 Read the state with all of them in mind."
                    .to_string(),
            );
            lines.extend(preamble.iter().cloned());
            lines.push(String::new());
            lines.push("STATE".to_string());
            lines.push(state.to_string());
            lines.push(String::new());
            lines.extend(qblock);
        }
        // StateFirst (and a resolved Auto) — snap's original order.
        _ => {
            lines.push("STATE".to_string());
            lines.push(state.to_string());
            lines.push(String::new());
            lines.extend(qblock);
        }
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
        let s = slots_for(&q(json!({"type": "boolean", "allow_abstain": true})));
        assert_eq!(s.len(), 3); // yes, no, abstain
        assert_eq!(s[0].key, "yes");
        assert_eq!(s[1].key, "no");
        assert!(s[2].special && s[2].key == ABSTAIN);
        // no abstain by default
        let s = slots_for(&q(json!({"type": "boolean"})));
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
        // an empty description still shows the key
        let s = slots_for(&q(json!({"type": "choice", "criteria": {"refund": ""}})));
        assert_eq!(s[0].text, "refund");
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
        let qu =
            q(json!({"type": "boolean", "instructions": "Is it spam?", "allow_abstain": true}));
        let s = slots_for(&qu);
        let m = user_message("hello world", &qu, &s, Layout::StateFirst, &[]);
        assert!(m.starts_with("STATE\nhello world"));
        assert!(m.contains("QUESTION\nIs it spam?"));
        assert!(m.contains("A) Yes"));
        assert!(m.contains("B) No"));
        assert!(m.contains("C) None of the above"));
        assert!(m.ends_with("Reply with one letter only."));
    }

    #[test]
    fn user_message_question_first() {
        let qu = q(json!({"type": "boolean", "instructions": "Is it spam?"}));
        let s = slots_for(&qu);
        let m = user_message("hello world", &qu, &s, Layout::QuestionFirst, &[]);
        assert!(m.starts_with("QUESTION\nIs it spam?"));
        assert!(m.contains("\n\nSTATE\nhello world\n\nReply with one letter only."));
        assert!(!m.contains("QUESTIONS"));
    }

    #[test]
    fn user_message_header() {
        let qu = q(json!({"type": "boolean", "instructions": "Is it spam?"}));
        let s = slots_for(&qu);
        let preamble = vec![
            "1. spam — Is it spam?".to_string(),
            "2. mood — Mood?".to_string(),
        ];
        let m = user_message("hello world", &qu, &s, Layout::Header, &preamble);
        assert!(m.starts_with("QUESTIONS\n"));
        assert!(m.contains("1. spam — Is it spam?"));
        // state still comes before the concrete question
        assert!(m.find("\nSTATE\n").unwrap() < m.find("\nQUESTION\n").unwrap());
    }

    #[test]
    fn catalog_message_shape() {
        let qu = q(json!({"type": "boolean", "instructions": "Is it spam?"}));
        let s = slots_for(&qu);
        let cat = catalog_entry(0, "spam", &qu);
        let m = catalog_message(&cat, "hello world", 0, "spam", &s);
        assert!(m.starts_with("QUESTIONS\n"));
        assert!(m.contains("1) spam — Is it spam?"));
        // options live in the tail, right before the reply cue
        assert!(m.contains("\nSTATE\nhello world\n\nQUESTION 1 — spam\n\nOPTIONS\nA) Yes"));
        assert!(m.ends_with("Reply with one letter only."));
    }

    #[test]
    fn render_state_compact_shape() {
        let v = json!({"user": {"name": "ada", "n": 3}, "tags": ["x", "y"], "note": "see: this"});
        let s = render_state_compact(&v);
        assert!(s.contains("user:\n  name: ada\n  n: 3"));
        assert!(s.contains("tags:\n  x, y"));
        // ": " inside a string must stay quoted
        assert!(s.contains("note: \"see: this\""));
        // strings that look like scalars stay quoted too
        assert!(render_state_compact(&json!({"v": "true"})).contains("\"true\""));
        // plain strings pass through untouched
        assert_eq!(render_state_compact(&json!("plain text")), "plain text");
        // uniform object arrays become a csv table with one header line
        let v = json!({"items": [{"sku": "a", "qty": 1}, {"sku": "b", "qty": 2}]});
        let s = render_state_compact(&v);
        assert!(s.contains("items:\n  (sku, qty)\n  a, 1\n  b, 2"), "{s}");
    }
}
