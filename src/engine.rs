//! Engine: typed questions in, typed answers out, one logits row each.
//!
//! Instead of generating text, each question is compiled to a prompt whose
//! first answer token must be one of A-Z; the answer is the softmax over the
//! letter logits at that single position. `decide` compiles every question —
//! and every per-option probe a >26-option choice expands into — to a token
//! prompt plus the prefix of it worth caching across requests, hands the lot
//! to `kv::Kv` (one shared-prefix trie per batched decode), and folds the
//! rows back into answers.
//!
//! The layout decides what that cached prefix is: `state_first` and `header`
//! keep `[head+state]` (the same document, any questions), `question_first`
//! keeps `[head+question]` (the same question over a stream of states —
//! snapjudge's finding, faster and more accurate on short states),
//! `catalog` keeps `[head+QUESTIONS list]` (the same question set, the state
//! decoded once after it).

use std::time::Instant;

use anyhow::{bail, Context, Result};
use serde_json::{json, Map, Value};

use crate::calibrate::{bucket, Calibration};
use crate::decisions;
use crate::kv::{common, Backend, Job, Kv};
use crate::llamac::Llama;
use crate::prompts::{self, Slot};
use crate::schema::{DecideRequest, Expand, Layout, Mode, QType, Question, MAX_SLOTS};

/// Stands in for the user message when measuring the template head.
const HEAD_SENTINEL: &str = "\u{1}snap-head\u{1}";
/// Stands in for the state when measuring where a question_first head ends.
const STATE_SENTINEL: &str = "\u{1}snap-state\u{1}";
/// `layout: auto` flips to state_first past this rendered-state length:
/// several questions decode the document once instead of once per question
/// (snapjudge's threshold), and even a single question reads a long document
/// better before it (TypeSafe: question_first −11 pts vs state_first).
const LONG_STATE_CHARS: usize = 2000;

/// One decode unit: a question as written, or a probe/page expanded from an
/// over-budget choice. `keep` is the prompt prefix worth caching across
/// requests; `calib` the fitted temperature of its bucket.
struct Item {
    name: String,
    q: Question,
    slots: Vec<Slot>,
    toks: Vec<i32>,
    keep: usize,
    calib: f64,
}

pub struct Engine {
    llm: Box<dyn Backend>,
    kv: Kv,
    letter_ids: Vec<Vec<i32>>,
    /// The template text around the user message: (before, after).
    frame: (String, String),
    head: Vec<i32>,
    pub model_id: String,
    /// Fitted per-type temperatures (`snap calibrate`); None = T 1.0 everywhere.
    pub calibration: Option<Calibration>,
}

impl Engine {
    /// Load a GGUF on the production llama.cpp backend.
    pub fn load(model_path: &str, n_ctx: i32, n_batch: i32, n_threads: i32) -> Result<Self> {
        let t0 = Instant::now();
        let path = std::path::Path::new(model_path);
        let stem = path
            .file_stem()
            .map_or_else(|| "model".into(), |s| s.to_string_lossy().into_owned());
        eprint!(
            "snap: loading {} … ",
            path.file_name()
                .map_or(model_path.into(), |s| s.to_string_lossy())
        );
        let eng = Llama::load(model_path, n_ctx, n_batch, n_threads).and_then(|llama| {
            let name = llama.meta("general.name").unwrap_or(stem);
            let quant = llama.meta("general.file_type");
            let model_id = format!(
                "snap-{}{}",
                name.to_lowercase().replace(' ', "-"),
                quant.map(|q| format!("-{q}")).unwrap_or_default()
            );
            Engine::new(Box::new(llama), model_id)
        });
        match &eng {
            Ok(_) => eprintln!("ready in {:.1}s", t0.elapsed().as_secs_f32()),
            Err(_) => eprintln!("failed"),
        }
        eng
    }

    pub fn new(mut llm: Box<dyn Backend>, model_id: String) -> Result<Self> {
        let letter_ids = letter_ids(&*llm)?;
        let prompt = llm.render(prompts::SYSTEM, HEAD_SENTINEL)?;
        let (pre, post) = prompt
            .split_once(HEAD_SENTINEL)
            .context("chat template does not render the user message verbatim")?;
        let frame = (pre.to_string(), post.to_string());
        let head = llm.tokenize(&frame.0, true)?;
        let kv = Kv::new(&mut *llm, head.clone())?;
        Ok(Engine {
            llm,
            kv,
            letter_ids,
            frame,
            head,
            model_id,
            calibration: None,
        })
    }

    /// Compile the backend's hot pipelines before the first real request: a
    /// multi-question batched decode, then a single-question one.
    pub fn warmup(&mut self) -> Result<()> {
        let mut questions = Map::new();
        questions.insert(
            "urgent".into(),
            json!({"type": "boolean", "instructions": "Is the order urgent?"}),
        );
        questions.insert(
            "item".into(),
            json!({"type": "choice", "instructions": "Pick the first item.", "criteria": {"c0": "latte", "c1": "pane"}}),
        );
        let req = DecideRequest {
            model: None,
            state: json!({"order": "warmup", "items": ["latte", "pane", "pasta"], "note": "Synthetic ticket used to warm the inference pipelines at startup."}),
            questions,
            temperature: 1.0,
            mode: Mode::Shared,
            layout: Layout::StateFirst,
            expand: Expand::default(),
            compact_state: false,
        };
        self.decide(&req)?;
        let mut one = Map::new();
        one.insert(
            "vip".into(),
            json!({"type": "boolean", "instructions": "Is the customer premium?"}),
        );
        self.decide(&DecideRequest {
            questions: one,
            ..req
        })?;
        Ok(())
    }

    /// Load a calibration file produced by `snap calibrate`; refuses files
    /// bound to a different model or prompt format.
    pub fn load_calibration(&mut self, path: &str) -> Result<()> {
        let cal = Calibration::load(path)?;
        cal.check(&self.model_id)?;
        eprintln!(
            "snap: calibration {path} — {} cases fit, ece {:.3} raw / {:.3} oof",
            cal.fitted, cal.ece_before, cal.ece_oof
        );
        self.calibration = Some(cal);
        Ok(())
    }

    /// Only the template's own text may produce control tokens: request
    /// content is tokenized as plain text, so a `<|im_end|>` inside a state
    /// stays characters instead of closing the turn.
    fn prompt_tokens(&self, user: &str) -> Result<Vec<i32>> {
        let full = self.llm.render(prompts::SYSTEM, user)?;
        let (pre, post) = &self.frame;
        let body = full
            .strip_prefix(pre.as_str())
            .and_then(|b| b.strip_suffix(post.as_str()))
            .context("chat template does not render the user message verbatim")?;
        let mut toks = self.llm.tokenize(pre, true)?;
        toks.extend(self.llm.tokenize(body, false)?);
        toks.extend(self.llm.tokenize(post, true)?);
        Ok(toks)
    }

    /// Token index where a question_first head ends (STATE begins): render
    /// the same message with a sentinel state and take the shared prefix —
    /// BPE may merge the separator into the first state token, so this stays
    /// a conservative boundary either way.
    fn head_split(&self, q: &Question, slots: &[Slot], toks: &[i32]) -> Result<usize> {
        let probe = prompts::user_message(STATE_SENTINEL, q, slots, Layout::QuestionFirst, &[]);
        Ok(common(toks, &self.prompt_tokens(&probe)?))
    }

    /// Token index where the state-independent (catalog) or question-
    /// independent (state_first/header) prefix ends, measured by tokenizing
    /// the real message truncated at the marker.
    fn cache_prefix_len(&self, it: &Item, umsg: &str, layout: Layout) -> usize {
        let pos = match layout {
            Layout::Catalog => umsg.find("\nSTATE\n"),
            _ => umsg.rfind(&prompts::question_block(&it.q, &it.slots).join("\n")),
        };
        pos.and_then(|p| self.prompt_tokens(&umsg[..p]).ok())
            .map_or(0, |probe| common(&it.toks, &probe))
    }

    pub fn decide(&mut self, req: &DecideRequest) -> Result<Value> {
        req.validate()?;
        let t0 = Instant::now();
        let calib = |b: &str| self.calibration.as_ref().map_or(1.0, |c| c.temp(b));

        // expand over-budget choices first: the layout heuristic wants the
        // real item count (a 30-option choice is 30 probes sharing the state)
        let mut pending: Vec<(String, Question, f64)> = Vec::new();
        let mut expanded: Vec<(String, Question, Vec<String>, usize)> = Vec::new();
        let mut originals: Vec<(String, String)> = Vec::new();
        let mut any_abstain = false;
        for (name, q) in req.questions()? {
            originals.push((name.clone(), q.instructions.clone()));
            any_abstain |= q.allow_abstain;
            match expand_choice(&q, req.expand) {
                Some((subs, keys)) => {
                    // probes are booleans mechanically: boolean-bucket
                    // temperature; pages are real choices
                    let b = match req.expand {
                        Expand::Pages => "choice",
                        Expand::Probes => "boolean",
                    };
                    let n = subs.len();
                    for (i, sq) in subs.into_iter().enumerate() {
                        pending.push((format!("{name}\u{1f}{i}"), sq, calib(b)));
                    }
                    expanded.push((name, q, keys, n));
                }
                None => {
                    let c = calib(bucket(q.qtype.as_str()));
                    pending.push((name, q, c));
                }
            }
        }

        let state_txt = if req.compact_state {
            prompts::render_state_compact(&req.state)
        } else {
            prompts::render_state(&req.state)
        };
        let qs: Vec<&Question> = pending.iter().map(|p| &p.1).collect();
        let layout = resolve_layout(req.layout, &qs, any_abstain, state_txt.len());
        let preamble: Vec<String> = if layout == Layout::Header {
            originals
                .iter()
                .enumerate()
                .map(|(i, (n, instr))| match instr.is_empty() {
                    true => format!("{}. {n}", i + 1),
                    false => format!("{}. {n} — {instr}", i + 1),
                })
                .collect()
        } else {
            Vec::new()
        };
        // expanded subs carry the `name\x1fi` tag; prompts show the name
        let display = |name: &str| name.split('\u{1f}').next().unwrap_or(name).to_string();
        let catalog: Vec<String> = if layout == Layout::Catalog {
            let mut v = Vec::new();
            for (i, (name, q, _)) in pending.iter().enumerate() {
                if i > 0 {
                    v.push(String::new());
                }
                v.extend(prompts::catalog_entry(i, &display(name), q));
            }
            v
        } else {
            Vec::new()
        };

        let shared = req.mode == Mode::Shared;
        let n_ctx = self.llm.n_ctx();
        let mut items = Vec::with_capacity(pending.len());
        let mut first_msg = String::new();
        for (idx, (name, q, calib)) in pending.into_iter().enumerate() {
            let slots = prompts::slots_for(&q);
            let msg = match layout {
                Layout::Catalog => {
                    prompts::catalog_message(&catalog, &state_txt, idx, &display(&name), &slots)
                }
                _ => prompts::user_message(&state_txt, &q, &slots, layout, &preamble),
            };
            let toks = self.prompt_tokens(&msg)?;
            if toks.len() > n_ctx {
                bail!(
                    "question {name:?}: prompt is {} tokens, ctx is {n_ctx}",
                    toks.len()
                );
            }
            let keep = match layout {
                Layout::QuestionFirst if shared => self.head_split(&q, &slots, &toks)?,
                _ => 0,
            };
            if idx == 0 {
                first_msg = msg;
            }
            items.push(Item {
                name,
                q,
                slots,
                toks,
                keep,
                calib,
            });
        }
        let plen = match items.split_first() {
            Some((a, rest)) if shared && !rest.is_empty() => rest
                .iter()
                .map(|b| common(&a.toks, &b.toks))
                .min()
                .unwrap_or(0),
            _ => 0,
        };
        // every non-question_first item shares one cacheable prefix
        if shared && layout != Layout::QuestionFirst {
            let s = self.cache_prefix_len(&items[0], &first_msg, layout);
            let s = if plen > 0 { s.min(plen) } else { s };
            items.iter_mut().for_each(|it| it.keep = s);
        }

        let jobs: Vec<Job> = items
            .iter()
            .map(|it| Job {
                toks: &it.toks,
                keep: it.keep,
            })
            .collect();
        let mut rows: Vec<Option<(Vec<f64>, f64)>> = vec![None; items.len()];
        let letters = &self.letter_ids;
        let mut sink = |j: usize, row: &[f32]| {
            rows[j] = Some(read_row(row, &letters[..items[j].slots.len()]));
        };
        let stats = match self.kv.run(&mut *self.llm, &jobs, shared, &mut sink) {
            Ok(s) => s,
            Err(e) => {
                // memory may be half-written: rebuild it and retry once
                eprintln!("snap: decode failed ({e}); resetting the KV cache");
                self.kv.reset(&mut *self.llm)?;
                self.kv.run(&mut *self.llm, &jobs, shared, &mut sink)?
            }
        };

        let mut answers = Map::new();
        for (it, row) in items.iter().zip(rows) {
            let (logits, cov) = row.context("item produced no logits")?;
            let mut ans = decisions::decode(&it.q, &it.slots, &logits, req.temperature * it.calib);
            ans["coverage"] = json!(r4(cov));
            answers.insert(it.name.clone(), ans);
        }
        // fold per-option probes / pages back into one answer per choice
        for (name, q, keys, nsubs) in &expanded {
            let subs: Vec<Value> = (0..*nsubs)
                .map(|i| {
                    answers
                        .remove(&format!("{name}\u{1f}{i}"))
                        .unwrap_or_default()
                })
                .collect();
            let cov = subs
                .iter()
                .map(|a| a["coverage"].as_f64().unwrap_or(0.0))
                .sum::<f64>();
            let mut merged = match req.expand {
                Expand::Pages => decisions::merge_paged(q, keys, &subs),
                Expand::Probes => {
                    let p_yes: Vec<f64> = subs
                        .iter()
                        .map(|a| a["probabilities"]["yes"].as_f64().unwrap_or(0.0))
                        .collect();
                    decisions::merge_scored(q, keys, &p_yes)
                }
            };
            merged["coverage"] = json!(r4(cov / (*nsubs).max(1) as f64));
            answers.insert(name.clone(), merged);
        }

        let head = self.head.len();
        let head_used = items
            .iter()
            .all(|it| it.toks.len() > head && it.toks.starts_with(&self.head));
        let ms = |v: f64| (v * 10.0).round() / 10.0;
        Ok(json!({
            "answers": answers,
            "model": self.model_id,
            "usage": {"input_tokens": stats.decoded},
            "x_snap": {
                "layout": layout_str(layout),
                "decoded_items": items.len(),
                "prompt_tokens": items.iter().map(|it| it.toks.len()).sum::<usize>(),
                "cached_head_tokens": if head_used { head } else { 0 },
                "shared_prefix_tokens": plen,
                "cache_hits": stats.hits,
                "cache_misses": stats.misses,
                "waves": stats.waves,
                "decode_ms": ms(stats.decode_ms),
                "total_ms": ms(t0.elapsed().as_secs_f64() * 1000.0),
            },
        }))
    }
}

/// `layout: auto` resolved. Steady-state cost decides: warm question_first
/// re-decodes the state per question (n×state), state_first pays it once
/// plus every question head (state + Σheads) — qf wins only when the heads
/// dominate, big rubric questions over a short state (missile game: 13
/// questions on a 500-char state were 3× slower under qf even with every
/// head cached). Long documents and anything with
/// an abstain slot go state_first: the `__abstain__` option read before the
/// evidence primes abstention (measured on eval/cases across the model set).
fn resolve_layout(layout: Layout, qs: &[&Question], any_abstain: bool, state_len: usize) -> Layout {
    if layout != Layout::Auto {
        return layout;
    }
    let n = qs.len();
    let head_sum: usize = qs
        .iter()
        .map(|q| q.instructions.len() + q.criteria.as_ref().map_or(0, |c| c.to_string().len()))
        .sum();
    let heads_dominate = n <= 1 || head_sum > (n - 1) * state_len;
    if any_abstain || state_len > LONG_STATE_CHARS || !heads_dominate {
        Layout::StateFirst
    } else {
        Layout::QuestionFirst
    }
}

/// Letter logits (max over each letter's token variants) and coverage: the
/// share of the raw next-token mass landing on the allowed letters at all —
/// low coverage means the model wanted to say something else entirely.
fn read_row(row: &[f32], letters: &[Vec<i32>]) -> (Vec<f64>, f64) {
    let logits = letters
        .iter()
        .map(|ids| {
            ids.iter()
                .map(|&t| row[t as usize] as f64)
                .fold(f64::NEG_INFINITY, f64::max)
        })
        .collect();
    let max = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)) as f64;
    let all: f64 = row.iter().map(|&l| (l as f64 - max).exp()).sum();
    let hit: f64 = letters
        .iter()
        .flatten()
        .map(|&t| (row[t as usize] as f64 - max).exp())
        .sum();
    (logits, if all > 0.0 { hit / all } else { 0.0 })
}

/// Token ids per letter, pooling bare and space/newline-prefixed variants.
fn letter_ids(llm: &dyn Backend) -> Result<Vec<Vec<i32>>> {
    prompts::LETTERS
        .iter()
        .map(|&b| {
            let ch = (b as char).to_string();
            let mut ids = Vec::new();
            for s in [ch.clone(), format!(" {ch}"), format!("\n{ch}")] {
                if let Ok(t) = llm.tokenize(&s, false) {
                    if t.len() == 1 && !ids.contains(&t[0]) {
                        ids.push(t[0]);
                    }
                }
            }
            if ids.is_empty() {
                bail!("letter {ch:?} is not a single token in this tokenizer");
            }
            Ok(ids)
        })
        .collect()
}

/// A choice with more options than letter slots expands into multiple items.
/// `probes`: one boolean probe per option ("is this candidate the correct
/// answer?"), merged by decisions::merge_scored — absolute probabilities.
/// `pages`: chunks of ≤MAX_SLOTS candidates as real choice questions, merged
/// by decisions::merge_paged — far fewer items, page-conditional
/// probabilities, and no abstain (each page must pick). Returns the sub
/// questions and the option keys in slot order. Score/numeric stay within
/// the letter budget.
fn expand_choice(q: &Question, expand: Expand) -> Option<(Vec<Question>, Vec<String>)> {
    if q.qtype != QType::Choice {
        return None;
    }
    let opts: Vec<(String, String)> = match q.criteria.as_ref()? {
        Value::Object(m) => m
            .iter()
            .map(|(k, v)| (k.clone(), prompts::value_text(v)))
            .collect(),
        Value::Array(a) => a
            .iter()
            .map(|v| {
                let t = prompts::value_text(v);
                (t.clone(), t)
            })
            .collect(),
        _ => vec![],
    };
    if opts.len() + q.allow_abstain as usize <= MAX_SLOTS {
        return None;
    }
    let base = if q.instructions.is_empty() {
        "Choose the best option.".to_string()
    } else {
        q.instructions.clone()
    };
    let sub = |qtype, instructions, criteria| Question {
        qtype,
        instructions,
        criteria,
        min: None,
        max: None,
        step: None,
        granularity: crate::schema::default_granularity(),
        allow_abstain: false,
    };
    let subs = match expand {
        // the shared stem ends at "Candidate:" so the trie decodes only the
        // option text + yes/no tail per probe
        Expand::Probes => opts
            .iter()
            .map(|(key, desc)| {
                let cand = if key == desc {
                    key.clone()
                } else {
                    format!("{key} — {desc}")
                };
                sub(
                    QType::Boolean,
                    format!("{base}\nIs this candidate the correct answer?\nCandidate: {cand}"),
                    None,
                )
            })
            .collect(),
        // equal-size pages: within-page probabilities compare across pages,
        // so a 4-option tail page must not outscore a 26-option page
        Expand::Pages => {
            let npages = opts.len().div_ceil(MAX_SLOTS);
            let size = opts.len().div_ceil(npages);
            opts.chunks(size)
                .map(|page| {
                    let criteria: Map<String, Value> = page
                        .iter()
                        .map(|(k, d)| (k.clone(), Value::String(d.clone())))
                        .collect();
                    sub(
                        QType::Choice,
                        format!(
                            "{base}\nThese candidates are a subset — pick the best among them."
                        ),
                        Some(Value::Object(criteria)),
                    )
                })
                .collect()
        }
    };
    Some((subs, opts.into_iter().map(|(k, _)| k).collect()))
}

fn r4(v: f64) -> f64 {
    (v * 1e4).round() / 1e4
}

fn layout_str(l: Layout) -> &'static str {
    match l {
        Layout::Auto => "auto",
        Layout::StateFirst => "state_first",
        Layout::QuestionFirst => "question_first",
        Layout::Header => "header",
        Layout::Catalog => "catalog",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv::sim::{row, Sim};

    fn engine(n_ctx: usize, n_seq: usize, n_batch: usize) -> Engine {
        let sim = Sim::new(n_ctx, n_seq, n_batch);
        Engine::new(Box::new(sim), "snap-sim".into()).unwrap()
    }

    fn req(v: Value) -> DecideRequest {
        serde_json::from_value(v).unwrap()
    }

    fn big_choice(n: usize) -> Value {
        let c: Map<String, Value> = (0..n)
            .map(|i| (format!("o{i}"), json!(format!("option number {i}"))))
            .collect();
        Value::Object(c)
    }

    /// A question set touching every type, abstain, and both expansions.
    fn questions() -> Value {
        json!({
            "urgent": {"type": "noul", "instructions": "Is it urgent?"},
            "route": {"type": "choice", "instructions": "Which team?",
                      "criteria": {"billing": "payments", "tech": "bugs", "sales": "pricing"}},
            "mood": {"type": "score", "instructions": "How angry?", "criteria": ["calm", "annoyed", "furious"]},
            "days": {"type": "numeric", "instructions": "Refund in days?", "min": 0, "max": 10, "granularity": 5},
            "maybe": {"type": "boolean", "instructions": "Enough info?", "allow_abstain": true},
            "sku": {"type": "choice", "instructions": "Which product?", "criteria": big_choice(30)},
        })
    }

    const LAYOUTS: [&str; 5] = ["auto", "state_first", "question_first", "header", "catalog"];

    #[test]
    fn every_path_answers_like_direct_decoding() {
        // direct mode decodes every item alone off the head: the reference
        for (n_ctx, n_seq, n_batch) in [(1 << 16, 65, 512), (1 << 16, 17, 97), (6000, 5, 64)] {
            let mut eng = engine(n_ctx, n_seq, n_batch);
            for layout in LAYOUTS {
                for expand in ["probes", "pages"] {
                    let r = |mode: &str| {
                        req(json!({"state": {"ticket": "refund please, charged twice"},
                                   "questions": questions(), "layout": layout,
                                   "expand": expand, "mode": mode}))
                    };
                    let reference = eng.decide(&r("direct")).unwrap()["answers"].clone();
                    for pass in 0..2 {
                        let out = eng.decide(&r("shared")).unwrap();
                        assert_eq!(out["answers"], reference, "{layout}/{expand} pass {pass}");
                    }
                }
            }
        }
    }

    #[test]
    fn answer_is_the_softmax_of_the_prompt_letters() {
        let mut eng = engine(1 << 16, 65, 512);
        let qj = json!({"type": "boolean", "instructions": "Spam?"});
        let q: Question = serde_json::from_value(qj.clone()).unwrap();
        let out = eng
            .decide(&req(
                json!({"state": "buy now", "questions": {"q": qj}, "layout": "state_first"}),
            ))
            .unwrap();
        // rebuild the prompt by hand and read the same row the sim hands out
        let slots = prompts::slots_for(&q);
        let msg = prompts::user_message("buy now", &q, &slots, Layout::StateFirst, &[]);
        let toks = eng.prompt_tokens(&msg).unwrap();
        let (logits, _) = read_row(&row(&toks), &eng.letter_ids[..slots.len()]);
        let want = decisions::decode(&q, &slots, &logits, 1.0);
        assert_eq!(out["answers"]["q"]["probabilities"], want["probabilities"]);
    }

    #[test]
    fn repeats_hit_the_cache_and_decode_less() {
        let mut eng = engine(1 << 16, 65, 512);
        for layout in ["state_first", "question_first", "catalog"] {
            let r = req(
                json!({"state": "a long enough ticket body", "questions": questions(),
                               "layout": layout}),
            );
            let cold = eng.decide(&r).unwrap();
            let warm = eng.decide(&r).unwrap();
            assert_eq!(cold["answers"], warm["answers"]);
            let tok = |v: &Value| v["usage"]["input_tokens"].as_u64().unwrap();
            assert!(tok(&warm) < tok(&cold), "{layout}: no reuse");
            assert!(
                warm["x_snap"]["cache_hits"].as_u64().unwrap() > 0,
                "{layout}"
            );
            assert!(tok(&cold) < warm["x_snap"]["prompt_tokens"].as_u64().unwrap());
        }
    }

    #[test]
    fn shared_prefix_is_decoded_once() {
        let mut eng = engine(1 << 16, 65, 512);
        let qs: Map<String, Value> = (0..8)
            .map(|i| {
                (
                    format!("q{i}"),
                    json!({"type": "noul", "instructions": format!("Item {i}?")}),
                )
            })
            .collect();
        let r = |mode| req(json!({"state": "x".repeat(3000), "questions": qs, "mode": mode}));
        let shared = eng.decide(&r("shared")).unwrap();
        let direct = eng.decide(&r("direct")).unwrap();
        let tok = |v: &Value| v["usage"]["input_tokens"].as_u64().unwrap();
        assert_eq!(shared["x_snap"]["layout"], "state_first");
        assert!(
            tok(&shared) * 5 < tok(&direct),
            "{} vs {}",
            tok(&shared),
            tok(&direct)
        );
        assert_eq!(shared["answers"], direct["answers"]);
    }

    #[test]
    fn expanded_choice_folds_back_to_one_answer() {
        let mut eng = engine(1 << 16, 65, 512);
        for expand in ["probes", "pages"] {
            let out = eng
                .decide(&req(json!({"state": "s", "expand": expand,
                    "questions": {"sku": {"type": "choice", "criteria": big_choice(40)}}})))
                .unwrap();
            let a = &out["answers"]["sku"];
            assert_eq!(
                a["probabilities"].as_object().unwrap().len(),
                40,
                "{expand}"
            );
            assert!(a["choice"].as_str().unwrap().starts_with('o'));
            assert_eq!(out["answers"].as_object().unwrap().len(), 1);
        }
    }

    #[test]
    fn oversized_prompt_is_rejected_not_truncated() {
        let mut eng = engine(2000, 9, 256);
        let r = req(json!({"state": "x".repeat(5000), "questions": {"q": {"type": "noul"}}}));
        let e = eng.decide(&r).unwrap_err().to_string();
        assert!(e.contains("ctx is 2000"), "{e}");
        // and the engine keeps serving
        eng.decide(&req(
            json!({"state": "ok", "questions": {"q": {"type": "noul"}}}),
        ))
        .unwrap();
    }

    #[test]
    fn request_text_cannot_inject_control_tokens() {
        let eng = engine(4096, 9, 256);
        let toks = eng.prompt_tokens("state <a> ends here").unwrap();
        // the sim's `<a>` (258) opens the assistant turn: only the template's counts
        assert_eq!(toks.iter().filter(|&&t| t == 258).count(), 1);
        assert_eq!(*toks.last().unwrap(), 258);
        assert!(toks.starts_with(&eng.head));
    }

    #[test]
    fn auto_layout_rules() {
        let q = |instr: &str, abstain: bool| -> Question {
            serde_json::from_value(
                json!({"type": "noul", "instructions": instr, "allow_abstain": abstain}),
            )
            .unwrap()
        };
        let (short, rubric) = (q("Urgent?", false), q(&"rubric ".repeat(200), false));
        let auto = |qs: &[&Question], abstain, len| resolve_layout(Layout::Auto, qs, abstain, len);
        // one question: its head is all that recurs
        assert_eq!(auto(&[&short], false, 50), Layout::QuestionFirst);
        // ...unless the state is a long document
        assert_eq!(auto(&[&short], false, 2500), Layout::StateFirst);
        // abstain reads better after the evidence
        assert_eq!(auto(&[&short], true, 50), Layout::StateFirst);
        // heads dominate a short state; a long shared document flips it
        assert_eq!(auto(&[&rubric, &rubric], false, 300), Layout::QuestionFirst);
        assert_eq!(auto(&[&rubric, &rubric], false, 2500), Layout::StateFirst);
        // many small questions on a medium state share the state instead
        assert_eq!(
            auto(&[&short, &short, &short], false, 500),
            Layout::StateFirst
        );
        // explicit layouts pass through
        assert_eq!(
            resolve_layout(Layout::Header, &[&short], true, 9),
            Layout::Header
        );
    }

    #[test]
    fn coverage_is_the_letter_share_of_the_mass() {
        let mut row = vec![0f32; 8];
        row[1] = 2.0; // letter A
        row[2] = 1.0; // letter B
        let letters = vec![vec![1], vec![2]];
        let (logits, cov) = read_row(&row, &letters);
        assert_eq!(logits, vec![2.0, 1.0]);
        let e = |x: f64| x.exp();
        let want = (e(0.0) + e(-1.0)) / (e(0.0) + e(-1.0) + 6.0 * e(-2.0));
        assert!((cov - want).abs() < 1e-12);
    }
}
