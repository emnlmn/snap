//! Engine: typed questions in, typed answers out, one logits row each.
//!
//! Instead of generating text, each question is compiled to a prompt whose
//! first answer token must be one of A-Z; the answer is the softmax over the
//! letter logits at that single position. `decide` compiles every question —
//! and every per-option probe a >26-option choice expands into — to a token
//! prompt plus the prefix of it worth caching across requests, hands the lot
//! to `kv::Kv` (one shared-prefix trie per batched decode), and folds the
//! rows back into answers. The training export (`render_prompt`) reads its
//! prompt from the same compile step, so what it writes is what is decoded.
//!
//! The layout decides what that cached prefix is: `state_first` and `header`
//! keep `[head+state]` (the same document, any questions), `question_first`
//! keeps `[head+question]` (the same question over a stream of states —
//! snapjudge's finding, faster and more accurate on short states),
//! `catalog` keeps `[head+QUESTIONS list]` (the same question set, the state
//! decoded once after it).
//!
//! `snap grep` rides the same machinery: a chunk is the state of a one-question
//! `state_first` request, so its `[head+state]` prefix does not depend on the
//! query. `grep_snapshot` decodes that prefix once and hands out whole-seq
//! snapshots, `grep_score` restores one and decodes only the question tail —
//! the same row, hence the same P(yes), as `decide` reads from scratch.

use std::path::Path;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use serde_json::{json, Map, Value};

use crate::calibrate::{bucket, Calibration};
use crate::corpus::MAX_CHARS;
use crate::decisions;
use crate::kv::{self, common, Backend, Job, Kv, Restore, Room};
use crate::llamac::{KvType, Llama};
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

/// A decode unit before tokenization: name (expanded subs carry the
/// `name\x1fi` tag), question, calibration bucket.
type Unit = (String, Question, &'static str);

/// What every unit's user message shares once a request is compiled — the
/// one place layout, state rendering and choice expansion turn into prompt
/// text, read by both `decide` and the training export.
struct Compiled {
    layout: Layout,
    state: String,
    /// Layout::Header: every original question, numbered.
    preamble: Vec<String>,
    /// Layout::Catalog: the numbered question set.
    catalog: Vec<String>,
    /// Over-budget choices: (name, question, option keys, sub count).
    expanded: Vec<(String, Question, Vec<String>, usize)>,
}

impl Compiled {
    /// Letter slots and user message of unit `idx`, built on demand: a
    /// request never holds every message at once.
    fn message(&self, idx: usize, name: &str, q: &Question) -> (Vec<Slot>, String) {
        let slots = prompts::slots_for(q);
        let msg = match self.layout {
            Layout::Catalog => {
                prompts::catalog_message(&self.catalog, &self.state, idx, display(name), &slots)
            }
            _ => prompts::user_message(&self.state, q, &slots, self.layout, &self.preamble),
        };
        (slots, msg)
    }
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
    /// Debug escape hatch (`SNAP_STATE_FORMAT`, `snap evaluate
    /// --state-format`); None renders TOON, the only production rendering.
    pub state_format: Option<StateFormat>,
}

/// State renderings: TOON is the default; `json` survives only to bisect
/// "is it the format?" without a rebuild.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateFormat {
    Json,
    Toon,
}

impl StateFormat {
    pub fn as_str(self) -> &'static str {
        match self {
            StateFormat::Json => "json",
            StateFormat::Toon => "toon",
        }
    }

    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "json" => Ok(StateFormat::Json),
            "toon" => Ok(StateFormat::Toon),
            _ => bail!("expected json|toon, got {s:?}"),
        }
    }
}

/// `snap-<general.name>[-<file_type>]`, lowercased. Our own fine-tunes are
/// already named `snap<version>` (`snap1 2B` -> `snap1-2b`): no second prefix.
fn model_id(name: &str, quant: Option<&str>) -> String {
    let slug = name.to_lowercase().replace(' ', "-");
    let ours = slug.strip_prefix("snap").is_some_and(|rest| {
        rest.is_empty()
            || rest.starts_with(['-', '.'])
            || rest.starts_with(|c: char| c.is_ascii_digit())
    });
    let base = if ours { slug } else { format!("snap-{slug}") };
    match quant {
        Some(q) => format!("{base}-{q}"),
        None => base,
    }
}

fn state_text(state: &Value, fmt: Option<StateFormat>) -> String {
    match fmt.unwrap_or(StateFormat::Toon) {
        StateFormat::Json => prompts::render_state(state),
        StateFormat::Toon => prompts::render_state_toon(state),
    }
}

impl Engine {
    /// Load a GGUF on the production llama.cpp backend, KV cache in f16.
    pub fn load(model_path: &Path, n_ctx: i32, n_batch: i32, n_threads: i32) -> Result<Self> {
        Self::load_kv(model_path, n_ctx, n_batch, n_threads, KvType::F16)
    }

    /// `load` with the KV cache held as `kv`; stored snapshots bind to it.
    pub fn load_kv(
        model_path: &Path,
        n_ctx: i32,
        n_batch: i32,
        n_threads: i32,
        kv: KvType,
    ) -> Result<Self> {
        let t0 = Instant::now();
        let stem = model_path
            .file_stem()
            .map_or_else(|| "model".into(), |s| s.to_string_lossy().into_owned());
        eprint!(
            "snap: loading {} … ",
            model_path.file_name().map_or_else(
                || model_path.as_os_str().to_string_lossy(),
                |s| s.to_string_lossy()
            )
        );
        let eng = Llama::load(&model_path.to_string_lossy(), n_ctx, n_batch, n_threads, kv)
            .and_then(|llama| {
                let name = llama.meta("general.name").unwrap_or(stem);
                let quant = llama.meta("general.file_type");
                Engine::new(Box::new(llama), model_id(&name, quant.as_deref()))
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
        let state_format = std::env::var("SNAP_STATE_FORMAT")
            .ok()
            .map(|s| StateFormat::parse(&s))
            .transpose()?;
        if let Some(f) = state_format {
            eprintln!("snap: state_format override: {}", f.as_str());
        }
        Ok(Engine {
            llm,
            kv,
            letter_ids,
            frame,
            head,
            model_id,
            calibration: None,
            state_format,
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

    /// Tokens the model reads for `state`, rendered exactly as `decide`
    /// renders it — the shared prefix every question pays for once.
    /// Caller lands with the server-side usage probe (uncommitted work).
    #[allow(dead_code)]
    pub fn state_tokens(&self, state: &Value) -> Result<usize> {
        Ok(self
            .llm
            .tokenize(&state_text(state, self.state_format), false)?
            .len())
    }

    /// Render a one-question request exactly as `decide` sends it through
    /// the model's chat template — the supervision surface for training.
    /// Returns the full prompt string, its serve-time token ids, the resolved
    /// layout, and the letter->key slot map so callers never re-derive
    /// positions; None when the question has no single prompt (a choice past
    /// the letter budget expands into several decode items).
    pub fn render_prompt(&self, req: &DecideRequest) -> Result<Option<Value>> {
        req.validate()?;
        let (compiled, units) = compile(req, self.state_format)?;
        if !compiled.expanded.is_empty() {
            return Ok(None);
        }
        let [(name, q, _)] = units.as_slice() else {
            bail!("a prompt renders one question, got {}", units.len());
        };
        let (slots, msg) = compiled.message(0, name, q);
        let full = self.llm.render(prompts::SYSTEM, &msg)?;
        // serve-time tokenization: the frame's own text may produce control
        // tokens, the body never does — a plain re-tokenize of `full` would
        // let request content (or the seam) pick up different ids
        let token_ids = self.prompt_tokens(&msg)?;
        Ok(Some(json!({
            "prompt": full,
            "token_ids": token_ids,
            "layout": layout_str(compiled.layout),
            "letters": slots
                .iter()
                .enumerate()
                .map(|(i, s)| json!({
                    "letter": (prompts::LETTERS[i] as char).to_string(),
                    "key": s.key,
                    "text": s.text,
                    "special": s.special,
                }))
                .collect::<Vec<_>>(),
        })))
    }

    pub fn decide(&mut self, req: &DecideRequest) -> Result<Value> {
        req.validate()?;
        let t0 = Instant::now();
        let calib = |b: &str| self.calibration.as_ref().map_or(1.0, |c| c.temp(b));
        let (compiled, units) = compile(req, self.state_format)?;
        let layout = compiled.layout;

        let shared = req.mode == Mode::Shared;
        let n_ctx = self.llm.n_ctx();
        let mut items = Vec::with_capacity(units.len());
        let mut first_msg = String::new();
        for (idx, (name, q, bucket)) in units.into_iter().enumerate() {
            let (slots, msg) = compiled.message(idx, &name, &q);
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
                calib: calib(bucket),
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
        for (name, q, keys, nsubs) in &compiled.expanded {
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
            "usage": {"input_tokens": stats.decoded, "output_tokens": 0},
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

/// Bumps whenever the chunk rendering (the STATE block) changes: stored
/// snapshots bind to it. The probe wording sits in the tail, after the
/// snapshot, so rewording needs no bump.
pub const GREP_FORMAT: u32 = 1;

/// What the model reads for a chunk: path, then the code. No line numbers,
/// so an edit above a chunk does not invalidate its snapshot. Capped at
/// 2 × corpus::MAX_CHARS characters (char boundary) for single-line giants.
pub fn grep_state(path: &str, text: &str) -> String {
    let mut s = format!("{path}\n{text}");
    if let Some((end, _)) = s.char_indices().nth(2 * MAX_CHARS) {
        s.truncate(end);
    }
    s
}

/// One chunk's relevance probe: the full prompt and the length of its
/// query-independent prefix `[head + STATE block]`.
pub struct Probe {
    pub toks: Vec<i32>,
    pub at: usize,
}

/// The one-question request a probe stands for. The query is flattened to one
/// line: every token of this sentence is decoded once per candidate.
fn grep_request(state: &str, query: &str, prose: bool) -> DecideRequest {
    let q = query.split_whitespace().collect::<Vec<_>>().join(" ");
    let ask = if prose {
        format!("Does this passage answer or directly address the search \"{q}\"?")
    } else {
        format!("Is this code what someone searching for \"{q}\" is looking for — does it implement or directly handle it?")
    };
    let mut questions = Map::new();
    questions.insert("q".into(), json!({"type": "boolean", "instructions": ask}));
    DecideRequest {
        model: None,
        state: json!(state),
        questions,
        temperature: 1.0,
        mode: Mode::Shared,
        layout: Layout::StateFirst,
        expand: Expand::default(),
    }
}

/// A probe's logits row read as `decide` reads a boolean question: the same
/// letters, the same decode, the boolean bucket's calibration at request
/// temperature 1.0, down to the rounding of `probabilities.yes`.
struct Reader<'a> {
    letters: &'a [Vec<i32>],
    q: Question,
    slots: Vec<Slot>,
    temp: f64,
}

impl<'a> Reader<'a> {
    fn new(letters: &'a [Vec<i32>], calibration: Option<&Calibration>) -> Result<Self> {
        let q: Question = serde_json::from_value(json!({"type": "boolean"}))?;
        Ok(Reader {
            letters,
            slots: prompts::slots_for(&q),
            q,
            temp: calibration.map_or(1.0, |c| c.temp(bucket("boolean"))),
        })
    }

    fn p_yes(&self, row: &[f32]) -> f64 {
        let (logits, _) = read_row(row, &self.letters[..self.slots.len()]);
        let ans = decisions::decode(&self.q, &self.slots, &logits, self.temp);
        ans["probabilities"]["yes"].as_f64().unwrap_or(0.0)
    }
}

/// `decide`'s recovery for the grep entry points: a failed decode may leave
/// the memory half-written, so rebuild it and run `f` once more.
fn retry<T>(
    kv: &mut Kv,
    llm: &mut dyn Backend,
    mut f: impl FnMut(&mut Kv, &mut dyn Backend) -> Result<T>,
) -> Result<T> {
    f(kv, llm).or_else(|e| {
        eprintln!("snap: decode failed ({e}); resetting the KV cache");
        kv.reset(llm)?;
        f(kv, llm)
    })
}

impl Engine {
    /// A chunk's relevance probe, compiled exactly as `decide` compiles a
    /// one-question `state_first` request in shared mode: `state` is the
    /// STATE block (`grep_state`), the question a boolean asking whether it
    /// is what `query` looks for. `prose` words it for documentation and text
    /// files. The wording sits after the STATE block, so both wordings, and
    /// every query, share one prefix — and one stored snapshot.
    pub fn grep_probe(&self, state: &str, query: &str, prose: bool) -> Result<Probe> {
        let (compiled, units) = compile(&grep_request(state, query, prose), self.state_format)?;
        let [(name, q, _)] = units.as_slice() else {
            bail!("a probe is one question, got {}", units.len());
        };
        let (slots, msg) = compiled.message(0, name, q);
        let toks = self.prompt_tokens(&msg)?;
        let n_ctx = self.llm.n_ctx();
        if toks.len() > n_ctx {
            bail!("probe is {} tokens, ctx is {n_ctx}", toks.len());
        }
        let it = Item {
            name: name.clone(),
            q: q.clone(),
            slots,
            toks,
            keep: 0,
            calib: 1.0,
        };
        let at = self.cache_prefix_len(&it, &msg, compiled.layout);
        if at == 0 || at >= it.toks.len() {
            bail!("probe of {} tokens has no cacheable prefix", it.toks.len());
        }
        Ok(Probe { toks: it.toks, at })
    }

    /// Decode `prefixes` (each a probe's `toks[..at]`) and hand `save(i, blob)`
    /// each whole-seq snapshot. A failed decode rebuilds the memory and runs
    /// the call again, so `save` may see an index twice.
    pub fn grep_snapshot(
        &mut self,
        prefixes: &[&[i32]],
        save: &mut dyn FnMut(usize, Vec<u8>),
    ) -> Result<kv::Stats> {
        let jobs: Vec<Job> = prefixes.iter().map(|&toks| Job { toks, keep: 0 }).collect();
        retry(&mut self.kv, &mut *self.llm, |kv, llm| {
            kv.snapshot(llm, &jobs, true, &mut *save)
        })
    }

    /// P(yes) per probe restored from its snapshot (only the tail decodes);
    /// None where the snapshot was refused.
    pub fn grep_score(
        &mut self,
        probes: &[(&Probe, &[u8])],
    ) -> Result<(Vec<Option<f64>>, kv::Stats)> {
        let jobs: Vec<Restore> = probes
            .iter()
            .map(|&(p, blob)| Restore {
                toks: &p.toks,
                at: p.at,
                blob,
            })
            .collect();
        let rd = Reader::new(&self.letter_ids, self.calibration.as_ref())?;
        let mut out = vec![None; probes.len()];
        // the retry hands rows out again: a slot is written, never pushed
        let (stats, _) = retry(&mut self.kv, &mut *self.llm, |kv, llm| {
            out.fill(None);
            kv.run_restored(llm, &jobs, &mut |j, row| out[j] = Some(rd.p_yes(row)))
        })?;
        Ok((out, stats))
    }

    /// P(yes) per probe decoded from scratch: the reference path and the
    /// fallback for refused snapshots.
    pub fn grep_score_cold(&mut self, probes: &[&Probe]) -> Result<(Vec<f64>, kv::Stats)> {
        let jobs: Vec<Job> = probes
            .iter()
            .map(|p| Job {
                toks: &p.toks,
                keep: 0,
            })
            .collect();
        let rd = Reader::new(&self.letter_ids, self.calibration.as_ref())?;
        let mut out = vec![None; probes.len()];
        let stats = retry(&mut self.kv, &mut *self.llm, |kv, llm| {
            kv.run(llm, &jobs, true, &mut |j, row| out[j] = Some(rd.p_yes(row)))
        })?;
        let ps = out
            .into_iter()
            .map(|p| p.context("probe produced no logits"))
            .collect::<Result<_>>()?;
        Ok((ps, stats))
    }

    /// What one wave of probes can hold: free seqs, and the cells left next
    /// to the resident head, kv's slack kept back. A probe restored costs a
    /// seq and its whole prompt in private cells, so probes that `Room::fit`
    /// are one wave of `grep_snapshot` and one of `grep_score`, as kv counts
    /// them — and a wave of snapshots is no more than one KV memory's worth
    /// of bytes.
    pub fn grep_room(&self) -> Room {
        self.kv.restore_room(&*self.llm)
    }
}

/// Compile a request into its decode units and what their messages share:
/// over-budget choices expanded, state rendered, layout resolved, header
/// preamble or catalog laid out.
fn compile(req: &DecideRequest, fmt: Option<StateFormat>) -> Result<(Compiled, Vec<Unit>)> {
    // expand over-budget choices first: the layout heuristic wants the
    // real item count (a 30-option choice is 30 probes sharing the state)
    let mut units: Vec<Unit> = Vec::new();
    let mut expanded = Vec::new();
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
                    units.push((format!("{name}\u{1f}{i}"), sq, b));
                }
                expanded.push((name, q, keys, n));
            }
            None => {
                let b = bucket(q.qtype.as_str());
                units.push((name, q, b));
            }
        }
    }

    let state = state_text(&req.state, fmt);
    let qs: Vec<&Question> = units.iter().map(|u| &u.1).collect();
    let layout = resolve_layout(req.layout, &qs, any_abstain, state.len());
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
    let catalog: Vec<String> = if layout == Layout::Catalog {
        let mut v = Vec::new();
        for (i, (name, q, _)) in units.iter().enumerate() {
            if i > 0 {
                v.push(String::new());
            }
            v.extend(prompts::catalog_entry(i, display(name), q));
        }
        v
    } else {
        Vec::new()
    };
    let compiled = Compiled {
        layout,
        state,
        preamble,
        catalog,
        expanded,
    };
    Ok((compiled, units))
}

/// Expanded subs carry the `name\x1fi` tag; prompts show the name.
fn display(name: &str) -> &str {
    name.split('\u{1f}').next().unwrap_or(name)
}

/// `layout: auto` resolved. Steady-state cost decides: warm question_first
/// re-decodes the state per question (n×state), state_first pays it once
/// plus every question head (state + Σheads) — qf wins only when the heads
/// dominate, big rubric questions over a short state (missile game: 13
/// questions on a 500-char state were 3× slower under qf even with every
/// head cached). Long documents and anything with
/// an abstain slot go state_first: the `__abstain__` option read before the
/// evidence primes abstention (measured on eval/cases across the model set).
pub(crate) fn resolve_layout(
    layout: Layout,
    qs: &[&Question],
    any_abstain: bool,
    state_len: usize,
) -> Layout {
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
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use super::*;
    use crate::kv::sim::{blob, row, Sim};
    use crate::kv::KV_SLACK;

    fn engine(n_ctx: usize, n_seq: usize, n_batch: usize) -> Engine {
        let sim = Sim::new(n_ctx, n_seq, n_batch);
        Engine::new(Box::new(sim), "snap-sim".into()).unwrap()
    }

    fn req(v: Value) -> DecideRequest {
        serde_json::from_value(v).unwrap()
    }

    #[test]
    fn model_id_prefixes_once() {
        assert_eq!(model_id("MiniCPM5-2B", Some("15")), "snap-minicpm5-2b-15");
        assert_eq!(model_id("snap1 2B", Some("15")), "snap1-2b-15");
        assert_eq!(model_id("snap1.1 2B", None), "snap1.1-2b");
        assert_eq!(model_id("Snap 2B", Some("7")), "snap-2b-7");
        assert_eq!(model_id("Snappy 7B", Some("7")), "snap-snappy-7b-7");
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
    fn export_renders_exactly_what_decide_decodes() {
        // the sim's row hashes the whole history: an exported prompt that
        // drifts from the decoded one by a single token changes the answer
        let mut eng = engine(1 << 16, 65, 512);
        let shapes = [
            json!({"type": "noul", "instructions": "Is it urgent?"}),
            json!({"type": "boolean", "instructions": "Enough info?", "allow_abstain": true}),
            json!({"type": "choice", "criteria": {"billing": "payments", "tech": "", "sales": "pricing"}}),
            json!({"type": "score", "instructions": "How angry?", "criteria": ["calm", "annoyed", "furious"]}),
            json!({"type": "numeric", "instructions": "Refund in days?", "min": 0, "max": 10, "granularity": 5}),
            // right at the letter budget: still one prompt
            json!({"type": "choice", "instructions": "Which product?", "criteria": big_choice(26)}),
        ];
        let state = json!({"ticket": "refund please", "items": ["latte", "pane"]});
        for layout in LAYOUTS {
            for fmt in [StateFormat::Toon, StateFormat::Json] {
                eng.state_format = Some(fmt);
                for qj in &shapes {
                    let r = req(json!({"state": state, "questions": {"q": qj},
                                       "layout": layout}));
                    let rec = eng.render_prompt(&r).unwrap().unwrap();
                    let out = eng.decide(&r).unwrap();
                    let at = format!("{layout}/{}: {qj}", fmt.as_str());
                    assert_eq!(rec["layout"], out["x_snap"]["layout"], "{at}");
                    let toks: Vec<i32> = serde_json::from_value(rec["token_ids"].clone()).unwrap();
                    let q: Question = serde_json::from_value(qj.clone()).unwrap();
                    let slots = prompts::slots_for(&q);
                    let (logits, _) = read_row(&row(&toks), &eng.letter_ids[..slots.len()]);
                    let want = decisions::decode(&q, &slots, &logits, 1.0);
                    assert_eq!(
                        out["answers"]["q"]["probabilities"], want["probabilities"],
                        "{at}"
                    );
                    // the prompt string is the template around those same tokens
                    let prompt = rec["prompt"].as_str().unwrap();
                    assert_eq!(eng.llm.tokenize(prompt, true).unwrap(), toks, "{at}");
                    let keys: Vec<&str> = rec["letters"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|l| l["key"].as_str().unwrap())
                        .collect();
                    assert_eq!(
                        keys,
                        slots.iter().map(|s| s.key.as_str()).collect::<Vec<_>>()
                    );
                    assert_eq!(rec["letters"][0]["letter"], "A");
                    match layout {
                        "header" => assert!(
                            prompt.contains("QUESTIONS\n") && prompt.contains("\n1. q"),
                            "{at}"
                        ),
                        "catalog" => assert!(prompt.contains("\nQUESTION 1 — q\n"), "{at}"),
                        _ => {}
                    }
                }
            }
        }
    }

    #[test]
    fn export_has_no_single_prompt_past_the_letter_budget() {
        let eng = engine(1 << 16, 65, 512);
        let r =
            |qs: Value, expand: &str| req(json!({"state": "s", "questions": qs, "expand": expand}));
        for expand in ["probes", "pages"] {
            for qj in [
                json!({"type": "choice", "criteria": big_choice(30)}),
                // 26 options fit the letters until abstain takes one
                json!({"type": "choice", "criteria": big_choice(26), "allow_abstain": true}),
            ] {
                let one = r(json!({"q": qj}), expand);
                assert!(eng.render_prompt(&one).unwrap().is_none(), "{expand}");
            }
        }
        let two = r(
            json!({"a": {"type": "noul"}, "b": {"type": "noul"}}),
            "probes",
        );
        let e = eng.render_prompt(&two).unwrap_err().to_string();
        assert!(e.contains("one question"), "{e}");
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

    const QUERIES: [&str; 3] = [
        "what keeps the pool of connections alive?",
        "install the server",
        "größe \n of   a thing",
    ];

    /// STATE blocks over code, prose and non-ASCII text, one of them twice
    /// (two probes, one prefix), each with a query and its wording.
    fn probes(eng: &Engine) -> Vec<(Probe, DecideRequest)> {
        let states = [
            grep_state(
                "src/kv.rs",
                "pub fn reset(&mut self) {\n    // rebuild the head\n}",
            ),
            grep_state(
                "README.md",
                "# Install\n\nRun `make build`, then `snap serve`.",
            ),
            grep_state("src/ü.py", &"def größe():\n    return 1\n".repeat(20)),
            grep_state("src/a.rs", "fn a() {}"),
            grep_state("src/a.rs", "fn a() {}"),
        ];
        states
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let (q, prose) = (QUERIES[i % 3], i % 2 == 1);
                (
                    eng.grep_probe(s, q, prose).unwrap(),
                    grep_request(s, q, prose),
                )
            })
            .collect()
    }

    /// What `decide` reports as P(yes) for a probe's request.
    fn decided(eng: &mut Engine, r: &DecideRequest) -> f64 {
        let out = eng.decide(r).unwrap();
        out["answers"]["q"]["probabilities"]["yes"]
            .as_f64()
            .unwrap()
    }

    #[test]
    fn grep_state_is_path_then_code_capped_at_a_char_boundary() {
        assert_eq!(
            grep_state("src/a.rs", "fn a() {}\nfn b() {}"),
            "src/a.rs\nfn a() {}\nfn b() {}"
        );
        let cap = 2 * MAX_CHARS;
        // "p\n" and the code fill the cap exactly: nothing is cut
        let fits = "é".repeat(cap - 2);
        assert_eq!(grep_state("p", &fits).chars().count(), cap);
        assert!(grep_state("p", &fits).ends_with('é'));
        // one char more and the last one goes, whole
        let big = grep_state("p", &"é".repeat(cap - 1));
        assert_eq!(big.chars().count(), cap);
        assert_eq!(big.len(), 2 + 2 * (cap - 2));
        assert!(big.starts_with("p\nééé"));
    }

    #[test]
    fn grep_prefix_is_query_independent() {
        let eng = engine(1 << 16, 65, 512);
        let state = grep_state("src/kv.rs", "pub fn reset(&mut self) {\n    // rebuild\n}");
        let a = eng.grep_probe(&state, QUERIES[0], false).unwrap();
        let b = eng.grep_probe(&state, QUERIES[1], false).unwrap();
        let prose = eng.grep_probe(&state, QUERIES[0], true).unwrap();
        for p in [&a, &b, &prose] {
            assert!(0 < p.at && p.at < p.toks.len());
            assert!(p.toks.starts_with(&eng.head) && p.at > eng.head.len());
        }
        // other queries, and the other wording, share the prefix; only the tail moves
        assert_eq!(a.toks[..a.at], b.toks[..b.at]);
        assert_eq!(a.toks[..a.at], prose.toks[..prose.at]);
        assert_ne!(a.toks[a.at..], b.toks[b.at..]);
        assert_ne!(a.toks[a.at..], prose.toks[prose.at..]);
        // the sim tokenizes bytes: the prefix past the head is the STATE block
        let block: Vec<u8> = a.toks[eng.head.len()..a.at]
            .iter()
            .map(|&t| t as u8)
            .collect();
        assert_eq!(block, format!("STATE\n{state}\n\n").as_bytes());
        // another state, another prefix
        let other = eng.grep_probe("src/b.rs\nfn b() {}", QUERIES[0], false);
        assert_ne!(other.unwrap().toks[..a.at], a.toks[..a.at]);
    }

    #[test]
    fn a_probe_is_the_prompt_decide_and_the_export_render() {
        let eng = engine(1 << 16, 65, 512);
        for (p, r) in probes(&eng) {
            let rec = eng.render_prompt(&r).unwrap().unwrap();
            let toks: Vec<i32> = serde_json::from_value(rec["token_ids"].clone()).unwrap();
            assert_eq!(p.toks, toks);
            assert_eq!(rec["layout"], "state_first");
        }
        // the query is one line whatever it was typed as
        let q = eng.grep_probe("s", QUERIES[2], false).unwrap();
        let text: Vec<u8> = q.toks.iter().map(|&t| t as u8).collect();
        assert!(String::from_utf8_lossy(&text).contains("searching for \"größe of a thing\""));
    }

    #[test]
    fn a_probe_past_the_context_is_rejected() {
        let eng = engine(300, 9, 64);
        let state = grep_state("a.rs", &"x".repeat(500));
        let e = eng.grep_probe(&state, "q", false).err().unwrap();
        assert!(e.to_string().contains("ctx is 300"), "{e}");
    }

    #[test]
    fn restored_scores_equal_cold_scores_equal_decide() {
        // roomy, hybrid-sized, cramped and barely-alive contexts
        for (n_ctx, n_seq, n_batch) in [
            (1 << 16, 65, 512),
            (1 << 16, 17, 97),
            (6000, 5, 64),
            (1700, 3, 64),
        ] {
            let mut eng = engine(n_ctx, n_seq, n_batch);
            let ps = probes(&eng);
            let refs: Vec<&Probe> = ps.iter().map(|(p, _)| p).collect();
            let prefixes: Vec<&[i32]> = refs.iter().map(|p| &p.toks[..p.at]).collect();
            // the fitted temperature of the boolean bucket applies like in decide
            for temp in [None, Some(1.7)] {
                eng.calibration = temp.map(|t| Calibration {
                    model: "snap-sim".into(),
                    prompt_version: prompts::PROMPT_VERSION,
                    temperatures: BTreeMap::from([("boolean".to_string(), t)]),
                    fitted: 0,
                    skipped: 0,
                    ece_before: 0.0,
                    ece_after: 0.0,
                    ece_oof: 0.0,
                    ci_before: (0.0, 0.0),
                    ci_oof: (0.0, 0.0),
                    ci_separated: false,
                });
                let want: Vec<f64> = ps.iter().map(|(_, r)| decided(&mut eng, r)).collect();
                assert_eq!(eng.grep_score_cold(&refs).unwrap().0, want, "{n_ctx} cold");

                let mut saved = vec![None; refs.len()];
                eng.grep_snapshot(&prefixes, &mut |i, b| saved[i] = Some(b))
                    .unwrap();
                let blobs: Vec<Vec<u8>> = saved.into_iter().map(|b| b.unwrap()).collect();
                for (b, p) in blobs.iter().zip(&refs) {
                    assert_eq!(*b, blob(&p.toks[..p.at]), "a snapshot is its whole prefix");
                }
                let pairs: Vec<(&Probe, &[u8])> = refs
                    .iter()
                    .copied()
                    .zip(blobs.iter().map(|b| &b[..]))
                    .collect();
                let (got, st) = eng.grep_score(&pairs).unwrap();
                let want: Vec<Option<f64>> = want.into_iter().map(Some).collect();
                assert_eq!(got, want, "{n_ctx} restored");
                // only the question tails decode
                let tails: usize = refs.iter().map(|p| p.toks.len() - p.at).sum();
                assert_eq!((st.decoded, st.hits), (tails, refs.len()), "{n_ctx}");
            }
        }
    }

    #[test]
    fn a_corrupt_snapshot_scores_none_and_the_others_stay_right() {
        let mut eng = engine(1 << 16, 65, 512);
        let ps = probes(&eng);
        let refs: Vec<&Probe> = ps.iter().map(|(p, _)| p).collect();
        let prefixes: Vec<&[i32]> = refs.iter().map(|p| &p.toks[..p.at]).collect();
        let mut saved = vec![None; refs.len()];
        eng.grep_snapshot(&prefixes, &mut |i, b| saved[i] = Some(b))
            .unwrap();
        let mut blobs: Vec<Vec<u8>> = saved.into_iter().map(|b| b.unwrap()).collect();
        blobs[1] = b"not a snapshot".to_vec();
        blobs[2].pop();
        blobs[3].clear();
        let pairs: Vec<(&Probe, &[u8])> = refs
            .iter()
            .copied()
            .zip(blobs.iter().map(|b| &b[..]))
            .collect();
        let (got, st) = eng.grep_score(&pairs).unwrap();
        let (cold, _) = eng.grep_score_cold(&refs).unwrap();
        assert_eq!(got[1..4], [None, None, None]);
        assert_eq!((got[0], got[4]), (Some(cold[0]), Some(cold[4])));
        assert_eq!(st.hits, 2);
        // the refusals left the memory clean: the same call again answers the same
        assert_eq!(eng.grep_score(&pairs).unwrap().0, got);
    }

    /// The sim, with one hard decode failure on demand.
    struct Flaky {
        sim: Sim,
        calls: Arc<AtomicUsize>,
        /// the decode call (1-based) that fails; 0 = none
        fail_at: Arc<AtomicUsize>,
        failed: Arc<AtomicUsize>,
    }

    impl Backend for Flaky {
        fn n_ctx(&self) -> usize {
            self.sim.n_ctx()
        }
        fn n_seq(&self) -> usize {
            self.sim.n_seq()
        }
        fn render(&self, system: &str, user: &str) -> Result<String> {
            self.sim.render(system, user)
        }
        fn tokenize(&self, text: &str, special: bool) -> Result<Vec<i32>> {
            self.sim.tokenize(text, special)
        }
        fn decode(
            &mut self,
            groups: &[crate::kv::Dec],
            row: &mut dyn FnMut(usize, &[f32]),
        ) -> Result<(), crate::kv::DecodeError> {
            let n = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
            if n == self.fail_at.load(Ordering::SeqCst) {
                self.failed.fetch_add(1, Ordering::SeqCst);
                return Err(crate::kv::DecodeError::Failed("injected".into()));
            }
            self.sim.decode(groups, row)
        }
        fn seq_rm(&mut self, seq: i32) {
            self.sim.seq_rm(seq)
        }
        fn seq_cp(&mut self, src: i32, dst: i32, len: usize) {
            self.sim.seq_cp(src, dst, len)
        }
        fn seq_save(&mut self, seq: i32) -> Result<Vec<u8>> {
            self.sim.seq_save(seq)
        }
        fn seq_load(&mut self, seq: i32, blob: &[u8]) -> Result<(), crate::kv::DecodeError> {
            self.sim.seq_load(seq, blob)
        }
        fn clear(&mut self) {
            self.sim.clear()
        }
    }

    #[test]
    fn a_failed_decode_resets_the_memory_and_the_call_runs_again() {
        let (calls, fail_at, failed) = (
            Arc::new(AtomicUsize::new(0)),
            Arc::new(AtomicUsize::new(0)),
            Arc::new(AtomicUsize::new(0)),
        );
        // three seqs: a call takes several waves, so the failure lands in the
        // first, in a later one, or between them
        let flaky = Flaky {
            sim: Sim::new(1 << 16, 3, 64),
            calls: calls.clone(),
            fail_at: fail_at.clone(),
            failed: failed.clone(),
        };
        let mut eng = Engine::new(Box::new(flaky), "snap-sim".into()).unwrap();
        let ps = probes(&eng);
        let refs: Vec<&Probe> = ps.iter().map(|(p, _)| p).collect();
        let prefixes: Vec<&[i32]> = refs.iter().map(|p| &p.toks[..p.at]).collect();
        let (want, _) = eng.grep_score_cold(&refs).unwrap();
        let want_some: Vec<Option<f64>> = want.iter().map(|&p| Some(p)).collect();
        // fail the `after`-th decode (one per wave) from now
        let arm = |after: usize| {
            let at = calls.load(Ordering::SeqCst) + after;
            fail_at.store(at, Ordering::SeqCst);
        };
        let injected = || failed.load(Ordering::SeqCst);
        for after in [1, 2, 3] {
            let before = injected();
            // the blobs of jobs handed out before the failure come again with
            // the rest, into their slots
            arm(after);
            let mut saved = vec![None; refs.len()];
            eng.grep_snapshot(&prefixes, &mut |i, b| saved[i] = Some(b))
                .unwrap();
            assert_eq!(injected(), before + 1, "snapshot, call {after}");
            let blobs: Vec<Vec<u8>> = saved.into_iter().map(|b| b.unwrap()).collect();
            for (b, p) in blobs.iter().zip(&refs) {
                assert_eq!(*b, blob(&p.toks[..p.at]), "snapshot, call {after}");
            }

            arm(after);
            let pairs: Vec<(&Probe, &[u8])> = refs
                .iter()
                .copied()
                .zip(blobs.iter().map(|b| &b[..]))
                .collect();
            assert_eq!(
                eng.grep_score(&pairs).unwrap().0,
                want_some,
                "restore, call {after}"
            );
            assert_eq!(injected(), before + 2, "restore, call {after}");

            arm(after);
            assert_eq!(
                eng.grep_score_cold(&refs).unwrap().0,
                want,
                "cold, call {after}"
            );
            assert_eq!(injected(), before + 3, "cold, call {after}");
        }
    }

    #[test]
    fn a_wave_room_is_the_context_less_the_head_and_the_slack() {
        for (n_ctx, n_seq) in [(1 << 16, 65), (8192, 65), (4096, 17), (600, 3)] {
            let eng = engine(n_ctx, n_seq, 64);
            let room = Room {
                seqs: n_seq - 1,
                cells: n_ctx - eng.head.len() - KV_SLACK,
            };
            assert_eq!(eng.grep_room(), room, "{n_ctx} cells, {n_seq} seqs");
        }
        // snap1-2b-sized probes of 460 tokens: what the default context holds,
        // then what the 64 seqs hold once the context is no limit
        let wave = |n_ctx, n_seq| {
            let eng = engine(n_ctx, n_seq, 512);
            eng.grep_room().fit(std::iter::repeat(460))
        };
        assert_eq!(wave(8192, 65), 17);
        assert_eq!(wave(32768, 65), 64);
        // a hybrid memory has 16 seqs to give whatever the context
        assert_eq!(wave(32768, 17), 16);
        // a context that holds no probe at all: the decode will say so
        assert_eq!(wave(500, 65), 0);
    }

    #[test]
    fn a_wave_that_fits_the_room_is_one_wave_for_the_kv_and_one_more_probe_is_two() {
        for (n_ctx, n_seq) in [(2000, 65), (1700, 3), (6000, 5), (1 << 16, 65)] {
            let mut eng = engine(n_ctx, n_seq, 64);
            let ps = probes(&eng);
            let lens = ps.iter().map(|(p, _)| p.toks.len());
            let n = eng.grep_room().fit(lens);
            assert!(0 < n, "{n_ctx} cells: no probe fits");
            let wave: Vec<&Probe> = ps[..n].iter().map(|(p, _)| p).collect();
            let prefixes: Vec<&[i32]> = wave.iter().map(|p| &p.toks[..p.at]).collect();
            let mut saved = vec![None; n];
            let st = eng
                .grep_snapshot(&prefixes, &mut |i, b| saved[i] = Some(b))
                .unwrap();
            assert_eq!(st.waves, 1, "{n_ctx}/{n_seq}: the snapshots");
            let blobs: Vec<Vec<u8>> = saved.into_iter().map(|b| b.unwrap()).collect();
            let pairs: Vec<(&Probe, &[u8])> = wave
                .iter()
                .copied()
                .zip(blobs.iter().map(|b| &b[..]))
                .collect();
            let (got, st) = eng.grep_score(&pairs).unwrap();
            assert!(got.iter().all(Option::is_some));
            assert_eq!(st.waves, 1, "{n_ctx}/{n_seq}: the restores");
            assert_eq!(
                eng.grep_score_cold(&wave).unwrap().1.waves,
                1,
                "{n_ctx}/{n_seq}: from scratch"
            );
            // one probe more is what the room refuses: the kv takes two waves
            if let Some((extra, _)) = ps.get(n) {
                let mut more = pairs.clone();
                let blob_of = blob(&extra.toks[..extra.at]);
                more.push((extra, &blob_of));
                assert_eq!(
                    eng.grep_score(&more).unwrap().1.waves,
                    2,
                    "{n_ctx}/{n_seq}: one more"
                );
            }
        }
    }
}
