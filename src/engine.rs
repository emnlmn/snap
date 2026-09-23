//! Engine: shared-prefix evaluation + single-token letter logits.
//!
//! Instead of generating text, each question is compiled to a prompt whose first
//! answer token must be one of A-Z. All questions in a request share the same
//! state, so their tokenized prompts share a long prefix: it is evaluated once,
//! then each question rewinds the context to the prefix boundary. The constant
//! prompt head (template header + system) is kept resident so requests skip its
//! prefill entirely.
//!
//! Full-attention models use a second context with `n_seq_max` sequences on a
//! unified KV cache: the shared prefix is copied to seqs 1..k via
//! `memory_seq_cp` and every question suffix is decoded in ONE batched
//! `llama_decode`. Hybrid/recurrent models cannot rewind/copy arbitrary KV
//! positions, so the whole context state is snapshotted after the prefix and
//! restored per question. Zero sampling, zero parsing, one logit row each.

use std::ffi::{c_char, c_void, CStr, CString};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use llama_cpp_sys_2 as sys;
use serde_json::{json, Map, Value};

use crate::decisions;
use crate::llamac;
use crate::prompts::{self, Slot};
use crate::schema::{DecideRequest, Mode, QType, Question, MAX_SLOTS};

const HEAD_SENTINEL: &str = "\u{1}snap-head\u{1}";

/// One decodable unit: a question as written, or a per-option probe expanded
/// from an over-budget choice. `calib` is the fitted temperature for its bucket.
struct Item {
    name: String,
    q: Question,
    slots: Vec<Slot>,
    toks: Vec<i32>,
    calib: f64,
}

extern "C" {
    fn snap_chat_templates_init(
        model: *mut sys::llama_model,
        tmpl_override: *const c_char,
        bos_override: *const c_char,
        eos_override: *const c_char,
    ) -> *mut c_void;
    fn snap_chat_apply(
        tmpls: *mut c_void,
        system: *const c_char,
        user: *const c_char,
    ) -> *mut c_char;
    fn snap_chat_templates_free(tmpls: *mut c_void);
    fn snap_str_free(p: *mut c_char);
}

pub struct Engine {
    model: llamac::Model,
    ctx: *mut sys::llama_context,
    mctx: Option<*mut sys::llama_context>, // multi-seq context: batched suffix decode
    tmpls: *mut c_void,                    // llama.cpp jinja chat templates
    letter_ids: Vec<Vec<i32>>,
    can_rewind: bool,
    head_tokens: Vec<i32>,
    head_state: Option<Vec<u8>>,
    n_vocab: usize,
    n_ctx: i32,
    n_batch: i32,
    pub model_id: String,
    /// Fitted per-type temperatures (`snap calibrate`); None = T 1.0 everywhere.
    pub calibration: Option<crate::calibrate::Calibration>,
}

// The engine is only ever used behind a Mutex; llama.cpp objects aren't
// internally thread-safe but exclusive access makes sharing sound.
unsafe impl Send for Engine {}

fn common_prefix_len(seqs: &[&[i32]]) -> usize {
    if seqs.is_empty() {
        return 0;
    }
    let mut n = 0;
    'outer: loop {
        let mut v = None;
        for s in seqs {
            match (s.get(n), v) {
                (Some(&t), None) => v = Some(t),
                (Some(&t), Some(u)) if t == u => {}
                _ => break 'outer,
            }
        }
        n += 1;
    }
    n
}

impl Engine {
    pub fn new(model_path: &str, n_ctx: i32, n_batch: i32) -> Result<Self> {
        let t0 = std::time::Instant::now();
        let file = std::path::Path::new(model_path)
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| model_path.to_string());
        eprintln!("loading {file} …");
        llamac::backend_init();
        let model = llamac::load_model(model_path)?;

        let name = llamac::meta_val(&model, "general.name").unwrap_or_else(|| {
            std::path::Path::new(model_path)
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "model".into())
        });
        let quant = llamac::meta_val(&model, "general.file_type");
        let model_id = format!(
            "snap-{}{}",
            name.to_lowercase().replace(' ', "-"),
            quant.map(|q| format!("-{q}")).unwrap_or_default()
        );
        if llamac::meta_val(&model, "tokenizer.chat_template").is_none() {
            bail!("model has no tokenizer.chat_template in GGUF metadata");
        }
        let tmpls = unsafe {
            snap_chat_templates_init(
                model.0,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
            )
        };
        if tmpls.is_null() {
            bail!("common_chat_templates_init failed (unsupported chat template?)");
        }

        let ctx =
            llamac::new_ctx(&model, n_ctx, n_batch, n_batch, 1, false).context("main context")?;
        let n_vocab = llamac::n_vocab(&model) as usize;

        let mut eng = Engine {
            letter_ids: build_letter_ids(&model)?,
            can_rewind: false,
            head_tokens: vec![],
            head_state: None,
            mctx: None,
            tmpls,
            model,
            ctx: ctx.0,
            n_vocab,
            n_ctx,
            n_batch,
            model_id,
            calibration: None,
        };
        let _ = ctx; // Ctx is a plain newtype (no Drop); raw pointer lives in eng

        eng.can_rewind = eng.probe_rewind();
        eng.head_tokens = eng.head_prompt_tokens()?;
        if !eng.head_tokens.is_empty() {
            unsafe { llamac::mem_clear(eng.ctx) };
            unsafe {
                llamac::batch_decode(
                    eng.ctx,
                    &[(0, &eng.head_tokens, 0)],
                    n_batch as usize,
                    n_vocab,
                )?
            };
            if eng.can_rewind {
                eng.init_mctx(n_ctx, n_batch);
            } else {
                eng.head_state = Some(unsafe { llamac::state_save(eng.ctx) });
                unsafe { llamac::mem_clear(eng.ctx) };
            }
        }
        eprintln!("ready in {:.1}s", t0.elapsed().as_secs_f32());
        Ok(eng)
    }

    /// Second context with n_seq_max sequences over a unified KV cache:
    /// seq 0 keeps the resident prompt head, question suffixes are decoded in
    /// one batched call on seqs 1..k. Any failure -> sequential fallback.
    fn init_mctx(&mut self, n_ctx: i32, n_batch: i32) {
        let ctx =
            match llamac::new_ctx(&self.model, n_ctx, n_batch, n_batch, llamac::MAX_SEQS, true) {
                Ok(c) => c.0,
                Err(e) => {
                    eprintln!("snap: multi-seq ctx init failed: {e}");
                    return;
                }
            };
        let ok = unsafe {
            // head resident on seq 0 + probe a shared-seq decode on seq 1
            match llamac::batch_decode(
                ctx,
                &[(0, &self.head_tokens, 0)],
                self.n_batch as usize,
                self.n_vocab,
            ) {
                Ok(_) => {
                    llamac::mem_cp(ctx, 0, 1, 0, self.head_tokens.len() as i32);
                    let last = &self.head_tokens[self.head_tokens.len() - 1..];
                    let r = llamac::batch_decode(
                        ctx,
                        &[(1, last, self.head_tokens.len() as i32)],
                        self.n_batch as usize,
                        self.n_vocab,
                    );
                    llamac::mem_rm(ctx, 1, -1, -1);
                    if let Err(e) = &r {
                        eprintln!("snap: multi-seq probe decode failed: {e}");
                    }
                    r.is_ok()
                }
                Err(e) => {
                    eprintln!("snap: multi-seq head decode failed: {e}");
                    false
                }
            }
        };
        if ok {
            self.mctx = Some(ctx);
        } else {
            llamac::free_ctx(ctx);
        }
    }

    /// Load a calibration file produced by `snap calibrate`; refuses files
    /// bound to a different model or prompt format.
    pub fn load_calibration(&mut self, path: &str) -> Result<()> {
        let cal = crate::calibrate::Calibration::load(path)?;
        cal.check(&self.model_id)?;
        eprintln!(
            "snap: calibration {path} — {} cases fit, ece {:.3} raw / {:.3} oof",
            cal.fitted, cal.ece_before, cal.ece_oof
        );
        self.calibration = Some(cal);
        Ok(())
    }

    /// Hybrid/recurrent architectures cannot rewind the context to an arbitrary
    /// position: probe once with a tiny decode.
    fn probe_rewind(&mut self) -> bool {
        let toks = match llamac::tokenize(&self.model, " a b c d", false) {
            Ok(t) if t.len() >= 2 => t,
            _ => return false,
        };
        let n = toks.len() as i32;
        let ok = unsafe {
            llamac::mem_clear(self.ctx);
            llamac::batch_decode(
                self.ctx,
                &[(0, &toks, 0)],
                self.n_batch as usize,
                self.n_vocab,
            )
            .is_ok()
                && {
                    llamac::mem_rm(self.ctx, 0, n - 1, -1);
                    let tail = &toks[toks.len() - 1..];
                    llamac::batch_decode(
                        self.ctx,
                        &[(0, tail, n - 1)],
                        self.n_batch as usize,
                        self.n_vocab,
                    )
                    .is_ok()
                }
        };
        unsafe { llamac::mem_clear(self.ctx) };
        ok
    }

    fn render_prompt(&self, user: &str) -> Result<String> {
        let system = CString::new(prompts::SYSTEM)?;
        let user = CString::new(user)?;
        let p = unsafe { snap_chat_apply(self.tmpls, system.as_ptr(), user.as_ptr()) };
        if p.is_null() {
            bail!("chat template apply failed");
        }
        let s = unsafe { CStr::from_ptr(p).to_string_lossy().into_owned() };
        unsafe { snap_str_free(p) };
        Ok(s)
    }

    /// Tokens of the constant prompt prefix: everything before the user content.
    fn head_prompt_tokens(&self) -> Result<Vec<i32>> {
        let prompt = self.render_prompt(HEAD_SENTINEL)?;
        if !prompt.contains(HEAD_SENTINEL) {
            return Ok(vec![]);
        }
        let head = prompt.split(HEAD_SENTINEL).next().unwrap_or_default();
        let mut toks = llamac::tokenize(&self.model, head, false)?;
        toks.pop(); // drop the last token: BPE merges may cross the boundary
        Ok(toks)
    }

    fn prompt_tokens(&self, user_text: &str) -> Result<Vec<i32>> {
        llamac::tokenize(&self.model, &self.render_prompt(user_text)?, false)
    }

    fn letter_logits(&self, row: &[f32], n_slots: usize) -> Vec<f64> {
        (0..n_slots)
            .map(|i| {
                self.letter_ids[i]
                    .iter()
                    .map(|&t| row[t as usize] as f64)
                    .fold(f64::NEG_INFINITY, f64::max)
            })
            .collect()
    }

    /// Multi-sequence decode on the batched context. Suffixes run on seqs
    /// 1..=MAX_SEQS-1 in waves — expanded choice probes can exceed one wave.
    fn decide_batched(
        &mut self,
        items: &[Item],
        hstart: usize,
        plen: usize,
        temperature: f64,
    ) -> Result<(Map<String, Value>, Instant)> {
        let ctx = self.mctx.context("no mctx")?;
        let start = if plen > 0 { plen } else { hstart };
        // An item whose whole prompt is the shared prefix has no suffix to
        // decode: its answer row is the one the prefix decode itself emits.
        let mut prefix_row = None;
        unsafe {
            for seq in 1..llamac::MAX_SEQS {
                llamac::mem_rm(ctx, seq, -1, -1);
            }
            llamac::mem_rm(ctx, 0, hstart as i32, -1);
            if plen > hstart {
                prefix_row = llamac::batch_decode(
                    ctx,
                    &[(0, &items[0].toks[hstart..plen], hstart as i32)],
                    self.n_batch as usize,
                    self.n_vocab,
                )?
                .into_iter()
                .next();
            }
        }
        let t_prefill = Instant::now();
        let wave = (llamac::MAX_SEQS - 1) as usize;
        let mut rows_all: Vec<Option<Vec<f32>>> = vec![None; items.len()];
        for (w, chunk) in items.chunks(wave).enumerate() {
            let base = w * wave;
            let live: Vec<usize> = (0..chunk.len())
                .filter(|&i| chunk[i].toks.len() > start)
                .collect();
            let seqs: Vec<(i32, &[i32], i32)> = live
                .iter()
                .enumerate()
                .map(|(s, &i)| (s as i32 + 1, &chunk[i].toks[start..], start as i32))
                .collect();
            unsafe {
                for s in 0..live.len() {
                    llamac::mem_cp(ctx, 0, s as i32 + 1, 0, start as i32);
                }
                if !seqs.is_empty() {
                    let rows =
                        llamac::batch_decode(ctx, &seqs, self.n_batch as usize, self.n_vocab)?;
                    for i in 1..=live.len() as i32 {
                        llamac::mem_rm(ctx, i, -1, -1);
                    }
                    for (&i, row) in live.iter().zip(rows) {
                        rows_all[base + i] = Some(row);
                    }
                }
            }
            for (i, it) in chunk.iter().enumerate() {
                if it.toks.len() == start {
                    rows_all[base + i] = prefix_row.clone();
                }
            }
        }
        let mut answers = Map::new();
        for (it, row) in items.iter().zip(rows_all) {
            let row = row.context("item produced no logits")?;
            let logits = self.letter_logits(&row, it.slots.len());
            answers.insert(
                it.name.clone(),
                decisions::decode(&it.q, &it.slots, &logits, temperature * it.calib),
            );
        }
        Ok((answers, t_prefill))
    }

    pub fn decide(&mut self, req: &DecideRequest) -> Result<Value> {
        req.validate()?;
        let t0 = Instant::now();
        let calib = |b: &str| self.calibration.as_ref().map(|c| c.temp(b)).unwrap_or(1.0);
        let mut items = Vec::new();
        let mut expanded: Vec<(String, Question, Vec<String>)> = Vec::new();
        for (name, q) in req.questions()? {
            let mut push = |eng: &Self, name: String, q: Question, c: f64| -> Result<()> {
                let slots = prompts::slots_for(&q);
                let toks = eng.prompt_tokens(&prompts::user_message(&req.state, &q, &slots))?;
                if toks.len() as i32 > eng.n_ctx {
                    bail!(
                        "question {name:?}: prompt is {} tokens, ctx is {}",
                        toks.len(),
                        eng.n_ctx
                    );
                }
                items.push(Item {
                    name,
                    q,
                    slots,
                    toks,
                    calib: c,
                });
                Ok(())
            };
            match expand_choice(&q) {
                // expanded probes are boolean questions mechanically: the
                // boolean bucket temperature applies
                Some((subs, keys)) => {
                    for (i, sq) in subs.into_iter().enumerate() {
                        push(self, format!("{name}\u{1f}{i}"), sq, calib("boolean"))?;
                    }
                    expanded.push((name, q, keys));
                }
                None => {
                    let c = calib(crate::calibrate::bucket(q.qtype.as_str()));
                    push(self, name, q, c)?;
                }
            }
        }

        let token_lists: Vec<&[i32]> = items.iter().map(|t| t.toks.as_slice()).collect();
        let plen = if items.len() > 1 && req.mode == Mode::Shared {
            common_prefix_len(&token_lists)
        } else {
            0
        };
        // The constant prompt head is already resident: every question shares it.
        let hstart = if self.head_tokens.is_empty() {
            0
        } else {
            token_lists
                .iter()
                .map(|t| common_prefix_len(&[&self.head_tokens, t]))
                .min()
                .unwrap_or(0)
        };

        let mut suffix_mode = "sequential";
        let mut answers: Option<Map<String, Value>> = None;
        let mut t_prefill = t0;

        if self.mctx.is_some() && items.len() > 1 {
            match self.decide_batched(&items, hstart, plen, req.temperature) {
                Ok((a, tp)) => {
                    answers = Some(a);
                    t_prefill = tp;
                    suffix_mode = "batched";
                }
                Err(_) => answers = None,
            }
        }
        let mut answers = match answers {
            Some(a) => a,
            None => {
                let mut prefix_row = None;
                unsafe {
                    if self.can_rewind {
                        llamac::mem_rm(self.ctx, 0, hstart as i32, -1); // keep resident head
                    } else if let Some(st) = &self.head_state {
                        llamac::state_load(self.ctx, st);
                    } else {
                        llamac::mem_clear(self.ctx);
                    }
                    if plen > hstart {
                        prefix_row = llamac::batch_decode(
                            self.ctx,
                            &[(0, &items[0].toks[hstart..plen], hstart as i32)],
                            self.n_batch as usize,
                            self.n_vocab,
                        )?
                        .into_iter()
                        .next();
                    }
                }
                let state = if self.can_rewind {
                    None
                } else {
                    Some(unsafe { llamac::state_save(self.ctx) })
                };
                t_prefill = Instant::now();

                let mut out = Map::new();
                for it in &items {
                    let start = if plen > 0 { plen } else { hstart };
                    // whole prompt inside the shared prefix: reuse its logit row
                    let row = if it.toks.len() == start {
                        prefix_row.clone().context("missing shared-prefix logits")?
                    } else {
                        unsafe {
                            match &state {
                                None => llamac::mem_rm(self.ctx, 0, start as i32, -1),
                                Some(st) => llamac::state_load(self.ctx, st),
                            }
                            llamac::batch_decode(
                                self.ctx,
                                &[(0, &it.toks[start..], start as i32)],
                                self.n_batch as usize,
                                self.n_vocab,
                            )?
                            .remove(0)
                        }
                    };
                    let logits = self.letter_logits(&row, it.slots.len());
                    out.insert(
                        it.name.clone(),
                        decisions::decode(&it.q, &it.slots, &logits, req.temperature * it.calib),
                    );
                }
                out
            }
        };
        // fold per-option probes back into one answer per expanded choice
        for (name, q, keys) in &expanded {
            let mut p_yes = Vec::with_capacity(keys.len());
            for i in 0..keys.len() {
                let a = answers
                    .remove(&format!("{name}\u{1f}{i}"))
                    .unwrap_or_default();
                p_yes.push(a["probabilities"]["yes"].as_f64().unwrap_or(0.0));
            }
            answers.insert(name.clone(), decisions::merge_scored(q, keys, &p_yes));
        }

        let total_ms = t0.elapsed().as_secs_f64() * 1000.0;
        // tokens actually decoded: shared prefix once + every suffix
        let input_tokens = plen + token_lists.iter().map(|t| t.len() - plen).sum::<usize>();
        Ok(json!({
            "answers": answers,
            "model": self.model_id,
            "usage": {"input_tokens": input_tokens},
            "x_snap": {
                "shared_prefix_tokens": plen,
                "cached_head_tokens": hstart,
                "rewind": if self.can_rewind { "kv" } else { "snapshot" },
                "suffix_decode": suffix_mode,
                "decoded_items": items.len(),
                "prefill_ms": ((t_prefill - t0).as_secs_f64() * 1000.0 * 10.0).round() / 10.0,
                "total_ms": (total_ms * 10.0).round() / 10.0,
            }
        }))
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        if let Some(c) = self.mctx.take() {
            llamac::free_ctx(c);
        }
        llamac::free_ctx(self.ctx);
        llamac::free_model(self.model.0);
        unsafe { snap_chat_templates_free(self.tmpls) };
    }
}

/// A choice with more options than letter slots expands into independent
/// per-option probes ("is this candidate the answer?", yes/no each), merged
/// back by decisions::merge_scored. Returns the probe questions and the
/// option keys in slot order. Score/numeric stay within the letter budget.
fn expand_choice(q: &Question) -> Option<(Vec<Question>, Vec<String>)> {
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
    let subs = opts
        .iter()
        .map(|(key, desc)| {
            let cand = if key == desc {
                key.clone()
            } else {
                format!("{key} — {desc}")
            };
            Question {
                qtype: QType::Boolean,
                instructions: format!(
                    "{base}\nCandidate: {cand}\nIs this candidate the correct answer?"
                ),
                criteria: None,
                min: None,
                max: None,
                step: None,
                granularity: crate::schema::default_granularity(),
                allow_abstain: false,
            }
        })
        .collect();
    Some((subs, opts.into_iter().map(|(k, _)| k).collect()))
}

/// Token ids per letter, pooling bare and space/newline-prefixed variants.
fn build_letter_ids(model: &llamac::Model) -> Result<Vec<Vec<i32>>> {
    let mut out = Vec::with_capacity(26);
    for &b in prompts::LETTERS {
        let ch = (b as char).to_string();
        let mut ids = Vec::new();
        for s in [ch.clone(), format!(" {ch}"), format!("\n{ch}")] {
            if let Ok(t) = llamac::tokenize(model, &s, false) {
                if t.len() == 1 && !ids.contains(&t[0]) {
                    ids.push(t[0]);
                }
            }
        }
        if ids.is_empty() {
            bail!("letter {ch:?} is not a single token in this tokenizer");
        }
        out.push(ids);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::common_prefix_len;

    #[test]
    fn common_prefix() {
        assert_eq!(common_prefix_len(&[]), 0);
        assert_eq!(common_prefix_len(&[&[1, 2, 3]]), 3);
        assert_eq!(common_prefix_len(&[&[1, 2, 3], &[1, 2, 4]]), 2);
        assert_eq!(common_prefix_len(&[&[1, 2], &[1, 2, 3, 4]]), 2);
        assert_eq!(common_prefix_len(&[&[1], &[2]]), 0);
        assert_eq!(common_prefix_len(&[&[1, 2, 3], &[1, 2, 3], &[1, 2, 3]]), 3);
    }
}
