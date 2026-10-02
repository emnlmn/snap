//! Prompt compilation: state + typed question -> one user message ending on a letter slot.

use serde_json::{Map, Value};

use crate::schema::{Layout, QType, Question};

pub const ABSTAIN: &str = "__abstain__";
pub const BELOW: &str = "__below__";
pub const ABOVE: &str = "__above__";

/// Bump when the prompt format changes: calibration files bind to it.
/// v3: expanded probes put the candidate last (shared-stem clustering),
/// Layout::Catalog added, compact_state rendering added.
/// v4: empty choice descriptions show the key, request text tokenized
/// without special tokens.
/// v5: states always render TOON — the compact_state flag is gone.
/// v6: noul/boolean {true, false} criteria render as Yes:/No: outcome
/// lines; null criterion descriptions fall back to key or level index.
pub const PROMPT_VERSION: u32 = 6;

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

/// Shortest round-trip, deliberately not %g: anchors like
/// `3.3333333333333335` look noisy, but rounding them to 6 or 3 significant
/// digits lost 1-3 numeric cases per model on eval/{core,edge} (none gained).
fn fmt_g(v: f64) -> String {
    format!("{v}")
}

/// TOON (spec v4.1, pinned — it's still a working draft) state rendering:
/// the same JSON data model with declared array lengths `[N]` and per-table
/// field lists `{f1,f2}` instead of repeated keys. Encode-only — answers are
/// letters, snap never parses TOON back.
pub fn render_state_toon(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => {
            let mut s = String::new();
            toon_root(other, &mut s);
            s.trim_end().to_string()
        }
    }
}

/// A header field-list entry: a bare name, or a name carrying a nested group
/// (`temp{min,max}`) for a column of nested-uniform objects.
enum Field {
    Leaf(String),
    Group(String, Vec<Field>),
}

/// §9.3/§9.5 column walk: all objects non-empty with the same key set, every
/// column uniform-primitive (all scalars) or nested-uniform (all non-empty
/// objects, recursively uniform). Returns the field list in the first
/// object's key order; None = not tabular-eligible.
fn fields_of(objs: &[&Map<String, Value>]) -> Option<Vec<Field>> {
    let first = objs.first()?;
    if first.is_empty() {
        return None;
    }
    let keys: Vec<&String> = first.keys().collect();
    let uniform = objs
        .iter()
        .all(|m| m.len() == keys.len() && keys.iter().all(|k| m.contains_key(*k)));
    if !uniform {
        return None;
    }
    keys.iter()
        .map(|k| {
            let col: Vec<&Value> = objs.iter().map(|m| &m[k.as_str()]).collect();
            if col
                .iter()
                .all(|v| v.as_object().is_some_and(|o| !o.is_empty()))
            {
                let subs: Vec<&Map<String, Value>> =
                    col.iter().map(|v| v.as_object().unwrap()).collect();
                Some(Field::Group(k.to_string(), fields_of(&subs)?))
            } else if col.iter().all(|v| !v.is_object() && !v.is_array()) {
                Some(Field::Leaf(k.to_string()))
            } else {
                None
            }
        })
        .collect()
}

/// §9.3 tabular detection on an array of objects.
fn tabular_fields(a: &[Value]) -> Option<Vec<Field>> {
    if a.is_empty() {
        return None;
    }
    let objs: Vec<&Map<String, Value>> = a.iter().map(|v| v.as_object()).collect::<Option<_>>()?;
    fields_of(&objs)
}

/// §9.5 keyed tabular detection on an object: ≥2 entries, every value a
/// non-empty object, uniform columns across them.
fn keyed_fields(m: &Map<String, Value>) -> Option<Vec<Field>> {
    if m.len() < 2 {
        return None;
    }
    let objs: Vec<&Map<String, Value>> =
        m.values().map(|v| v.as_object()).collect::<Option<_>>()?;
    fields_of(&objs)
}

fn field_list(fields: &[Field]) -> String {
    fields
        .iter()
        .map(|f| match f {
            Field::Leaf(k) => toon_key(k),
            Field::Group(k, subs) => format!("{}{{{}}}", toon_key(k), field_list(subs)),
        })
        .collect::<Vec<_>>()
        .join(",")
}

/// Row cells in depth-first leaf order over the field list.
fn cells_of<'a>(m: &'a Map<String, Value>, fields: &[Field], out: &mut Vec<&'a Value>) {
    for f in fields {
        match f {
            Field::Leaf(k) => out.push(&m[k.as_str()]),
            Field::Group(k, subs) => cells_of(m[k.as_str()].as_object().unwrap(), subs, out),
        }
    }
}

fn row_cells(m: &Map<String, Value>, fields: &[Field]) -> String {
    let mut cells = Vec::new();
    cells_of(m, fields, &mut cells);
    cells
        .iter()
        .map(|v| toon_scalar(v))
        .collect::<Vec<_>>()
        .join(",")
}

/// §7.3: unquoted only for `^[A-Za-z_][A-Za-z0-9_.]*$`.
fn toon_key(k: &str) -> String {
    let bare = k.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_')
        && k.chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.');
    if bare {
        k.to_string()
    } else {
        toon_quote(k)
    }
}

/// §7.1: the five escapes plus \uXXXX for other controls — serde_json's `\b`
/// and `\f` are not TOON escapes.
fn toon_quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// §4 number grammar lookalike: `^[+-]?[0-9]+(\.[0-9]+)?(e[+-]?[0-9]+)?$` —
/// such strings must be quoted so they don't read back as numbers.
fn numeric_like(s: &str) -> bool {
    let b = s.as_bytes();
    let mut i = (b.first() == Some(&b'+') || b.first() == Some(&b'-')) as usize;
    let d0 = i;
    while i < b.len() && b[i].is_ascii_digit() {
        i += 1;
    }
    if i == d0 {
        return false;
    }
    if b.get(i) == Some(&b'.') {
        i += 1;
        let d1 = i;
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
        }
        if i == d1 {
            return false;
        }
    }
    if matches!(b.get(i), Some(&b'e') | Some(&b'E')) {
        i += 1;
        if matches!(b.get(i), Some(&b'+') | Some(&b'-')) {
            i += 1;
        }
        let d2 = i;
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
        }
        if i == d2 {
            return false;
        }
    }
    i == b.len()
}

/// §7.2: bare only when quoting isn't required. Comma is both the document
/// and the active delimiter everywhere we emit, so it always forces quotes.
fn toon_bare(s: &str) -> bool {
    !s.is_empty()
        && !s.starts_with([' ', '\t', '-', '#'])
        && !s.ends_with([' ', '\t'])
        && !s.contains(|c: char| {
            matches!(c, ':' | '"' | '\\' | '[' | ']' | '{' | '}' | ',') || (c as u32) < 0x20
        })
        && !matches!(s, "true" | "false" | "null")
        && !numeric_like(s)
}

fn toon_scalar(v: &Value) -> String {
    match v {
        Value::String(s) if toon_bare(s) => s.clone(),
        Value::String(s) => toon_quote(s),
        Value::Null => "null".into(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => match (n.as_i64(), n.as_u64()) {
            (Some(i), _) => i.to_string(),
            (None, Some(u)) => u.to_string(),
            (None, None) => {
                let v = n.as_f64().unwrap_or_default();
                if v == 0.0 {
                    "0".into() // §2: -0 normalizes to 0
                } else {
                    fmt_g(v)
                }
            }
        },
        other => toon_quote(&value_text(other)),
    }
}

/// Emits one object field. `pre` is the line prefix — the indent pad, or
/// pad + "- " for a list-item object's first field (§10); `ind` is the
/// field's own depth, so its scope's content lands at ind+1.
fn toon_field(pre: &str, k: &str, val: &Value, ind: usize, out: &mut String) {
    let key = toon_key(k);
    match val {
        Value::Object(m) if m.is_empty() => out.push_str(&format!("{pre}{key}:\n")),
        Value::Object(m) => match keyed_fields(m) {
            Some(fields) => {
                out.push_str(&format!(
                    "{pre}{key}[{}:]{{{}}}:\n",
                    m.len(),
                    field_list(&fields)
                ));
                for (ek, ev) in m {
                    let row = row_cells(ev.as_object().unwrap(), &fields);
                    out.push_str(&format!(
                        "{}{}: {}\n",
                        "  ".repeat(ind + 1),
                        toon_key(ek),
                        row
                    ));
                }
            }
            None => {
                out.push_str(&format!("{pre}{key}:\n"));
                for (k2, v2) in m {
                    toon_field(&"  ".repeat(ind + 1), k2, v2, ind + 1, out);
                }
            }
        },
        Value::Array(a) => toon_array(pre, &key, a, ind, out),
        _ => out.push_str(&format!("{pre}{key}: {}\n", toon_scalar(val))),
    }
}

/// `key` is empty only at the root: a keyless header `[N]:`/`[N]{f}:`.
fn toon_array(pre: &str, key: &str, a: &[Value], ind: usize, out: &mut String) {
    if a.is_empty() {
        // §9.1: key: [] in field position, [] at the root
        if key.is_empty() {
            out.push_str("[]\n");
        } else {
            out.push_str(&format!("{pre}{key}: []\n"));
        }
        return;
    }
    if a.iter().all(|v| !v.is_object() && !v.is_array()) {
        let vals: Vec<String> = a.iter().map(toon_scalar).collect();
        out.push_str(&format!("{pre}{key}[{}]: {}\n", a.len(), vals.join(",")));
        return;
    }
    if let Some(fields) = tabular_fields(a) {
        out.push_str(&format!(
            "{pre}{key}[{}]{{{}}}:\n",
            a.len(),
            field_list(&fields)
        ));
        for x in a {
            out.push_str(&format!(
                "{}{}\n",
                "  ".repeat(ind + 1),
                row_cells(x.as_object().unwrap(), &fields)
            ));
        }
        return;
    }
    out.push_str(&format!("{pre}{key}[{}]:\n", a.len()));
    for x in a {
        toon_item(x, ind + 1, out);
    }
}

/// §9.4 list items at depth `ind`; §10 carries an object item's first field
/// on the hyphen line, its scope content at ind+2.
fn toon_item(v: &Value, ind: usize, out: &mut String) {
    let pad = "  ".repeat(ind);
    match v {
        Value::Object(m) if m.is_empty() => out.push_str(&format!("{pad}-\n")),
        Value::Object(m) => {
            let mut it = m.iter();
            let (k, val) = it.next().unwrap();
            toon_field(&format!("{pad}- "), k, val, ind + 1, out);
            for (k, val) in it {
                toon_field(&format!("{pad}  "), k, val, ind + 1, out);
            }
        }
        Value::Array(inner) if inner.is_empty() => out.push_str(&format!("{pad}- [0]:\n")),
        Value::Array(inner) if inner.iter().all(|x| !x.is_object() && !x.is_array()) => {
            let vals: Vec<String> = inner.iter().map(toon_scalar).collect();
            out.push_str(&format!("{pad}- [{}]: {}\n", inner.len(), vals.join(",")));
        }
        Value::Array(inner) => {
            // keyless fields-bearing headers aren't valid at item position
            // (§6), so nested uniform arrays use list form, never tabular
            out.push_str(&format!("{pad}- [{}]:\n", inner.len()));
            for x in inner {
                toon_item(x, ind + 1, out);
            }
        }
        _ => out.push_str(&format!("{pad}- {}\n", toon_scalar(v))),
    }
}

fn toon_root(v: &Value, out: &mut String) {
    match v {
        Value::Object(m) if m.is_empty() => {} // empty object: empty document
        Value::Object(m) => match keyed_fields(m) {
            Some(fields) => {
                out.push_str(&format!("[{}:]{{{}}}:\n", m.len(), field_list(&fields)));
                for (ek, ev) in m {
                    out.push_str(&format!(
                        "  {}: {}\n",
                        toon_key(ek),
                        row_cells(ev.as_object().unwrap(), &fields)
                    ));
                }
            }
            None => {
                for (k, v2) in m {
                    toon_field("", k, v2, 0, out);
                }
            }
        },
        Value::Array(a) => toon_array("", "", a, 0, out),
        _ => out.push_str(&format!("{}\n", toon_scalar(v))),
    }
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

/// A criterion description rendered for the prompt. `null` is the SDK's
/// "undescribed" marker — the caller's fallback (option key, level index)
/// decides what the model reads instead.
fn desc_text(v: &Value) -> String {
    match v {
        Value::Null => String::new(),
        other => value_text(other),
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
                .map(|(k, v)| Slot::new(k.clone(), option_label(k, &desc_text(v))))
                .collect(),
            Value::Array(a) => a
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let t = desc_text(v);
                    let t = if t.is_empty() {
                        format!("option {}", i + 1)
                    } else {
                        t
                    };
                    Slot::new(t.clone(), t)
                })
                .collect(),
            _ => vec![],
        },
        QType::Score => match q.criteria.as_ref().unwrap() {
            Value::Array(a) => a
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let t = desc_text(v);
                    Slot::new(i.to_string(), if t.is_empty() { i.to_string() } else { t })
                })
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
        QType::Boolean | QType::Noul => {
            lines.push("Answer yes or no.".into());
            // SDK criteria {"true":…, "false":…} describe the two outcomes —
            // they ride as question text; the yes/no slots never rename, so
            // P(yes) keeps its meaning
            if let Some(Value::Object(m)) = &q.criteria {
                for (label, key) in [("Yes", "true"), ("No", "false")] {
                    let t = m.get(key).map(desc_text).unwrap_or_default();
                    if !t.is_empty() {
                        lines.push(format!("{label}: {t}"));
                    }
                }
            }
        }
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
        // null (SDK "undescribed") falls back to the key, never "null"
        let s = slots_for(&q(json!({
            "type": "choice", "criteria": {"a": null, "b": "bee"}
        })));
        assert_eq!(s[0].key, "a");
        assert_eq!(s[0].text, "a");
        assert_eq!(s[1].text, "bee");
        // array form: a null entry gets a stable generated key
        let s = slots_for(&q(json!({"type": "choice", "criteria": ["x", null]})));
        assert_eq!(s[1].key, "option 2");
        assert_eq!(s[1].text, "option 2");
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
        // an undescribed level (null) shows its index, not the word "null"
        let s = slots_for(&q(
            json!({"type": "score", "criteria": ["low", null, "high"]}),
        ));
        assert_eq!(s[1].key, "1");
        assert_eq!(s[1].text, "1");
    }

    #[test]
    fn noul_criteria_describe_the_outcomes() {
        // SDK noul criteria ride as question text; slots stay yes/no
        let qu = q(json!({
            "type": "noul", "instructions": "Refund?",
            "criteria": {"true": "explicit money-back request", "false": "anything else"}
        }));
        let s = slots_for(&qu);
        assert_eq!(s[0].key, "yes");
        assert_eq!(s[1].key, "no");
        let block = question_block(&qu, &s).join("\n");
        assert!(block
            .contains("Answer yes or no.\nYes: explicit money-back request\nNo: anything else"));
        assert!(block.contains("A) Yes"));
        // no criteria, or a criteria shape that isn't the {true,false} map — unchanged
        let qu = q(json!({"type": "noul", "instructions": "Refund?"}));
        let block = question_block(&qu, &slots_for(&qu)).join("\n");
        assert!(!block.contains("\nYes:"));
        let qu = q(json!({"type": "noul", "criteria": ["x", "y"]}));
        let block = question_block(&qu, &slots_for(&qu)).join("\n");
        assert!(!block.contains("\nYes:"));
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
    fn toon_forms() {
        // object + inline array + quoting
        let v = json!({"user": {"name": "ada", "n": 3}, "tags": ["x", "y"], "note": "see: this"});
        let s = render_state_toon(&v);
        assert!(s.contains("user:\n  name: ada\n  n: 3"), "{s}");
        assert!(s.contains("tags[2]: x,y"), "{s}");
        assert!(s.contains("note: \"see: this\""), "{s}");
        // uniform object array: tabular form, header on the key line
        let v = json!({"items": [{"sku": "a", "qty": 1}, {"sku": "b", "qty": 2}]});
        assert_eq!(render_state_toon(&v), "items[2]{sku,qty}:\n  a,1\n  b,2");
        // nested-uniform column folds into a field group
        let v = json!({"fc": [{"d": "Mon", "t": {"min": -2, "max": 4}}]});
        assert_eq!(render_state_toon(&v), "fc[1]{d,t{min,max}}:\n  Mon,-2,4");
        // object of uniform objects: keyed tabular
        let v = json!({"env": {"prod": {"r": 1, "d": false}, "stg": {"r": 2, "d": true}}});
        assert_eq!(
            render_state_toon(&v),
            "env[2:]{r,d}:\n  prod: 1,false\n  stg: 2,true"
        );
        // non-uniform array: list form; object items carry their first field
        let v = json!({"xs": [1, {"a": 2, "b": 3}]});
        assert_eq!(render_state_toon(&v), "xs[2]:\n  - 1\n  - a: 2\n    b: 3");
        // object item whose first field is tabular: rows sit at +2 (§10)
        let v = json!({"xs": [{"rows": [{"a": 1}], "k": 2}]});
        assert_eq!(
            render_state_toon(&v),
            "xs[1]:\n  - rows[1]{a}:\n      1\n    k: 2"
        );
        // empties
        let v = json!({"o": {}, "a": [], "n": null});
        assert_eq!(render_state_toon(&v), "o:\na: []\nn: null");
        // root array, keyless header
        assert_eq!(render_state_toon(&json!([1, 2])), "[2]: 1,2");
        assert_eq!(
            render_state_toon(&json!([{"a": 1}, {"a": 2}])),
            "[2]{a}:\n  1\n  2"
        );
    }

    #[test]
    fn toon_quoting() {
        let cases = [
            ("", "\"\""),
            (" x", "\" x\""),
            ("x ", "\"x \""),
            ("true", "\"true\""),
            ("42", "\"42\""),
            ("+5", "\"+5\""),
            ("05", "\"05\""),
            ("1e-6", "\"1e-6\""),
            ("a,b", "\"a,b\""),
            ("a:b", "\"a:b\""),
            ("a[b", "\"a[b\""),
            ("-x", "\"-x\""),
            ("#c", "\"#c\""),
            ("a\tb", "\"a\\tb\""),
            ("a\nb", "\"a\\nb\""),
            ("hi there", "hi there"),
        ];
        for (s, want) in cases {
            let got = render_state_toon(&json!({"k": s}));
            assert_eq!(got, format!("k: {want}"), "input {s:?}");
        }
        // keys follow §7.3: bare only for [A-Za-z_][A-Za-z0-9_.]*
        assert_eq!(render_state_toon(&json!({"my key": 1})), "\"my key\": 1");
        assert_eq!(render_state_toon(&json!({"a.b_c": 1})), "a.b_c: 1");
        // -0 normalizes to 0
        assert_eq!(render_state_toon(&json!({"v": -0.0})), "v: 0");
        // strings still pass through untouched
        assert_eq!(render_state_toon(&json!("plain")), "plain");
    }
}
