//! Engine: shared-prefix evaluation + single-token letter logits.
//!
//! Instead of generating text, each question is compiled to a prompt whose first
//! answer token must be one of A-Z. All questions in a request share the same
//! state, so their tokenized prompts share a long prefix: it is evaluated once,
//! then each question rewinds the context to the prefix boundary. The constant
//! prompt head (template header + system) is kept resident so requests skip its
//! prefill entirely.
//!
//! A second context (`mctx`) holds `n_seq_max` sequences — fork-and-discard:
//! the shared prefix is decoded on a holder seq, copied to per-item work seqs
//! via `memory_seq_cp`, and every suffix is decoded in ONE batched
//! `llama_decode` per wave. Seqs are removed whole after use — never trimmed
//! partially — so the same path works on hybrid/recurrent archs (their
//! per-seq state makes `n_seq_max` smaller there). The `[head+state]` span
//! (or `[head+catalog]` under `layout: catalog`) also caches cross-request
//! on a resident entry seq. Zero sampling, zero parsing, one logit row each.
//!
//! `layout: question_first` flips the prompt (QUESTION+OPTIONS, then STATE):
//! nothing is shared inside one request, but the question head is identical
//! across requests — it stays resident on an mctx seq (or as a ctx state
//! snapshot on hybrid archs) in a tick-LRU cache, so repeat workloads decode
//! only the state. Stolen from snapjudge, which measured it both faster and
//! more accurate on short states. `layout: catalog` instead puts a numbered
//! QUESTIONS list before the STATE and leaves only a `QUESTION i — name` +
//! OPTIONS tail per item — the question set itself caches cross-request.

use std::collections::{HashMap, HashSet};
use std::ffi::{c_char, c_void, CStr, CString};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use llama_cpp_sys_2 as sys;
use serde_json::{json, Map, Value};

use crate::decisions;
use crate::llamac;
use crate::prompts::{self, Slot};
use crate::schema::{DecideRequest, Expand, Layout, Mode, QType, Question, MAX_SLOTS};

const HEAD_SENTINEL: &str = "\u{1}snap-head\u{1}";
/// Stands in for the state when measuring where a question_first head ends.
const STATE_SENTINEL: &str = "\u{1}snap-state\u{1}";

/// `layout: auto` flips to state_first past this rendered-state length when
/// several questions share it — the document is prefilled once instead of
/// once per question (snapjudge's threshold, same idea).
const LONG_STATE_CHARS: usize = 2000;
/// Question-head snapshots kept for hybrid/fallback ctxs (bounded: each holds
/// a full KV+state copy).
const QSNAP_MAX: usize = 32;
/// Cached question heads may hold at most this fraction of the KV pool, so a
/// batched wave always has room to run.
const QCACHE_CELL_DIV: i32 = 2;
/// Slack under n_ctx when sizing waves — guards against cell-accounting
/// subtleties (shared cells, rounding); deliberately small.
const KV_SLACK: i32 = 32;
/// Free cells a state-entry install must still leave in the shared pool
const MIN_WAVE_ROOM: usize = 256;
/// Extra shared span (past the global prefix) worth one tagged cluster decode
const MIN_SUB_PREFIX: usize = 8;
/// Seq capacity of the multi-seq ctx on hybrid/recurrent archs: each seq
/// costs a full recurrent-state row (~55 MB on qwen35), so 65 seqs would
/// pin ~4 GB. 17 keeps it near 1 GB — waves cover anything larger.
const HYBRID_SEQS: i32 = 17;

/// One decodable unit: a question as written, or a per-option probe expanded
/// from an over-budget choice. `calib` is the fitted temperature for its bucket.
/// `split` marks the end of the cacheable question head in question_first
/// layout (0 elsewhere).
struct Item {
    name: String,
    q: Question,
    slots: Vec<Slot>,
    toks: Vec<i32>,
    split: usize,
    calib: f64,
}

/// Outcome of a decode path: answers plus the internals x_snap reports.
struct DecodeOut {
    answers: Map<String, Value>,
    t_prefill: Instant,
    decoded: usize, // tokens actually decoded (question_first counts heads)
    hits: usize,
    misses: usize,
}

/// Per-request counters a wave run accumulates into.
#[derive(Default)]
struct WaveStat {
    hits: usize,
    misses: usize,
    decoded: usize,
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

/// Cross-request question-head cache on mctx seqs: a head's KV stays on its
/// seq; the state suffix decodes on top and is trimmed back afterwards.
/// Tick-based LRU — simplest correct policy at ≤64 entries.
///
/// `cells` tracks the private KV cells cached heads hold (the `hstart` shared
/// prefix lives once on seq 0 and isn't counted) so waves can be packed to
/// the remaining capacity — all seqs share one n_ctx pool.
#[derive(Default)]
struct QSeqs {
    by_head: HashMap<Vec<i32>, QEnt>, // head token span -> entry
    tick: u64,
    cells: usize,
}

#[derive(Clone)]
struct QEnt {
    seq: i32,
    tick: u64,
    cells: usize,
    /// logits at the entry's last token — only prefix (state-span) entries
    /// carry one; it answers items whose whole prompt is the prefix itself
    row: Option<Vec<f32>>,
}

impl QSeqs {
    fn get(&mut self, head: &[i32]) -> Option<i32> {
        self.tick += 1;
        let e = self.by_head.get_mut(head)?;
        e.tick = self.tick;
        Some(e.seq)
    }

    /// Lookup without touching the LRU tick — used to pin heads of queued
    /// items without making them look freshly consumed.
    fn peek(&self, head: &[i32]) -> Option<i32> {
        self.by_head.get(head).map(|e| e.seq)
    }

    /// The logits row stored on a prefix entry — answers items ending
    /// exactly at the entry's tail without a decode
    fn row_of(&self, head: &[i32]) -> Option<Vec<f32>> {
        self.by_head.get(head).and_then(|e| e.row.clone())
    }

    /// A seq that is free or whose head can be evicted (not `used` this wave).
    /// `nseq` is the ctx's seq capacity — smaller than MAX_SEQS on hybrid.
    fn alloc(&mut self, nseq: i32, used: &HashSet<i32>) -> Option<i32> {
        self.tick += 1;
        let free =
            (1..nseq).find(|s| !used.contains(s) && self.by_head.values().all(|e| e.seq != *s));
        if free.is_some() {
            return free;
        }
        let victim = self
            .by_head
            .iter()
            .filter(|(_, e)| !used.contains(&e.seq))
            .min_by_key(|(_, e)| e.tick)
            .map(|(k, e)| (k.clone(), (e.seq, e.cells)));
        victim.map(|(k, (seq, cells))| {
            self.by_head.remove(&k);
            self.cells -= cells;
            seq
        })
    }

    fn insert(&mut self, head: Vec<i32>, seq: i32, cells: usize) {
        self.insert_row(head, seq, cells, None);
    }

    fn insert_row(&mut self, head: Vec<i32>, seq: i32, cells: usize, row: Option<Vec<f32>>) {
        if let Some(old) = self.by_head.get(&head) {
            self.cells -= old.cells;
        }
        self.cells += cells;
        self.by_head.insert(
            head,
            QEnt {
                seq,
                tick: self.tick,
                cells,
                row,
            },
        );
    }

    fn remove(&mut self, head: &[i32]) {
        if let Some(e) = self.by_head.remove(head) {
            self.cells -= e.cells;
        }
    }

    /// Evict LRU entries not `used` this wave until the cache fits `cap`
    /// cells; evicted seqs are wiped so their cells are actually freed.
    unsafe fn trim_to(&mut self, ctx: *mut sys::llama_context, cap: usize, used: &HashSet<i32>) {
        while self.cells > cap {
            let victim = self
                .by_head
                .iter()
                .filter(|(_, e)| !used.contains(&e.seq))
                .min_by_key(|(_, e)| e.tick)
                .map(|(k, e)| (k.clone(), (e.seq, e.cells)));
            let Some((k, (seq, cells))) = victim else {
                break;
            };
            self.by_head.remove(&k);
            self.cells -= cells;
            llamac::mem_rm(ctx, seq, -1, -1);
        }
    }

    /// Forget every entry — call whenever mctx seqs are wiped underneath the
    /// cache (decide_batched's wave reset, decode-failure cleanup).
    fn clear(&mut self) {
        self.by_head.clear();
        self.cells = 0;
    }
}

/// Same cache on single-seq ctxs (hybrid archs, or mctx init failure): a
/// question head is a whole-context state snapshot restored per use.
#[derive(Default)]
struct QSnaps {
    by_head: HashMap<Vec<i32>, (Vec<u8>, u64)>,
    tick: u64,
}

impl QSnaps {
    /// Borrow, not clone — a snapshot is a full KV+state image (tens of MB on
    /// hybrid archs), so cloning per hit was real per-request work.
    fn get(&mut self, head: &[i32]) -> Option<&Vec<u8>> {
        self.tick += 1;
        let e = self.by_head.get_mut(head)?;
        e.1 = self.tick;
        Some(&e.0)
    }

    fn insert(&mut self, head: Vec<i32>, state: Vec<u8>) {
        self.tick += 1;
        if self.by_head.len() >= QSNAP_MAX && !self.by_head.contains_key(&head) {
            let victim = self
                .by_head
                .iter()
                .min_by_key(|(_, (_, t))| *t)
                .map(|(k, _)| k.clone());
            if let Some(k) = victim {
                self.by_head.remove(&k);
            }
        }
        self.by_head.insert(head, (state, self.tick));
    }
}

pub struct Engine {
    model: llamac::Model,
    ctx: *mut sys::llama_context,
    mctx: Option<*mut sys::llama_context>, // multi-seq context: batched suffix decode
    tmpls: *mut c_void,                    // llama.cpp jinja chat templates
    letter_ids: Vec<Vec<i32>>,
    can_rewind: bool,
    hybrid: bool,
    mseqs: i32, // seq capacity of mctx (0 while mctx is absent)
    head_tokens: Vec<i32>,
    head_state: Option<Vec<u8>>,
    qseqs: QSeqs,
    qsnaps: QSnaps,
    n_vocab: usize,
    n_ctx: i32,
    n_batch: i32,
    n_threads: i32,
    pub model_id: String,
    /// Fitted per-type temperatures (`snap calibrate`); None = T 1.0 everywhere.
    pub calibration: Option<crate::calibrate::Calibration>,
}

// The engine is only ever used behind a Mutex; llama.cpp objects aren't
// internally thread-safe but exclusive access makes sharing sound.
unsafe impl Send for Engine {}

/// End of the wave starting at `lo`: at most `max_per` items and `budget`
/// total cost. A lone item over budget is an error — the caller falls back
/// to the single-seq path, which only needs the item to fit n_ctx.
fn wave_end(costs: &[usize], lo: usize, max_per: usize, budget: usize) -> Result<usize> {
    let (mut hi, mut cost) = (lo, 0usize);
    while hi < costs.len() && hi - lo < max_per && cost + costs[hi] <= budget {
        cost += costs[hi];
        hi += 1;
    }
    if hi == lo {
        bail!(
            "item {lo} needs {} KV cells, wave budget is {budget}",
            costs[lo]
        );
    }
    Ok(hi)
}

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
    pub fn new(model_path: &str, n_ctx: i32, n_batch: i32, n_threads: i32) -> Result<Self> {
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

        // 0 = auto: llama.cpp's own default is 4 threads, which starves
        // prefill on CPU-only builds (most release targets).
        let n_threads = if n_threads <= 0 {
            std::thread::available_parallelism()
                .map(|n| n.get() as i32)
                .unwrap_or(4)
        } else {
            n_threads
        };
        let hybrid = llamac::model_hybrid(&model);
        let ctx = llamac::new_ctx(&model, n_ctx, n_batch, n_batch, 1, false, n_threads)
            .context("main context")?;
        let n_vocab = llamac::n_vocab(&model) as usize;

        let mut eng = Engine {
            letter_ids: build_letter_ids(&model)?,
            can_rewind: false,
            hybrid,
            mseqs: 0,
            head_tokens: vec![],
            head_state: None,
            mctx: None,
            tmpls,
            model,
            ctx: ctx.0,
            qseqs: QSeqs::default(),
            qsnaps: QSnaps::default(),
            n_vocab,
            n_ctx,
            n_batch,
            n_threads,
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
                    &[llamac::Dec::one(&eng.head_tokens, 0)],
                    n_batch as usize,
                    n_vocab,
                )?
            };
            // every arch gets the multi-seq ctx now: fork-and-discard (cp +
            // whole-seq rm) works on hybrid memory too, so batched decoding
            // no longer needs arbitrary rewind. The single-seq ctx stays as
            // the failure fallback — hybrid still snapshots for it.
            eng.init_mctx(n_ctx, n_batch);
            if !eng.can_rewind {
                eng.head_state = Some(unsafe { llamac::state_save(eng.ctx) });
                unsafe { llamac::mem_clear(eng.ctx) };
            }
        }
        eprintln!("ready in {:.1}s", t0.elapsed().as_secs_f32());
        Ok(eng)
    }

    /// Second context with n_seq_max sequences over a unified KV cache:
    /// seq 0 keeps the resident prompt head, question suffixes are decoded in
    /// one batched call on forked seqs (cp + whole-seq rm — never a partial
    /// rewind, so hybrid/recurrent archs work too). Hybrid gets a smaller seq
    /// space: each seq reserves a recurrent-state row (~55 MB on qwen35).
    /// Any failure -> sequential fallback.
    fn init_mctx(&mut self, n_ctx: i32, n_batch: i32) {
        let nseq = if self.hybrid {
            HYBRID_SEQS
        } else {
            llamac::MAX_SEQS
        };
        let ctx = match llamac::new_ctx(
            &self.model,
            n_ctx,
            n_batch,
            n_batch,
            nseq,
            true,
            self.n_threads,
        ) {
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
                &[llamac::Dec::one(&self.head_tokens, 0)],
                self.n_batch as usize,
                self.n_vocab,
            ) {
                Ok(_) => {
                    llamac::mem_cp(ctx, 0, 1, 0, self.head_tokens.len() as i32);
                    let last = &self.head_tokens[self.head_tokens.len() - 1..];
                    let r = llamac::batch_decode(
                        ctx,
                        &[llamac::Dec::new(
                            &[1],
                            last,
                            self.head_tokens.len() as i32,
                            true,
                        )],
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
            self.mseqs = nseq;
        } else {
            llamac::free_ctx(ctx);
        }
    }

    /// Compile the backend's hot pipelines before the first real request:
    /// a multi-question shared-prefix call exercises the fused multi-seq
    /// decode (every arch now), a single-question call the seq0 ctx.
    /// State_first uses scratch seqs only, so nothing leaks into the
    /// qhead caches.
    pub fn warmup(&mut self) -> Result<()> {
        let mut questions = Map::new();
        questions.insert(
            "urgent".into(),
            json!({"type": "boolean", "instructions": "Is the order urgent?", "allow_abstain": false}),
        );
        questions.insert(
            "item".into(),
            json!({"type": "choice", "instructions": "Pick the first item.", "criteria": {"c0": "latte", "c1": "pane"}, "allow_abstain": false}),
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
            json!({"type": "boolean", "instructions": "Is the customer premium?", "allow_abstain": false}),
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
    /// position: skip the probe decode outright on them (it would only log a
    /// doomed M-RoPE/SSM position error), probe once otherwise.
    fn probe_rewind(&mut self) -> bool {
        if llamac::model_hybrid(&self.model) {
            return false;
        }
        let toks = match llamac::tokenize(&self.model, " a b c d", false) {
            Ok(t) if t.len() >= 2 => t,
            _ => return false,
        };
        let n = toks.len() as i32;
        let ok = unsafe {
            llamac::mem_clear(self.ctx);
            llamac::batch_decode(
                self.ctx,
                &[llamac::Dec::one(&toks, 0)],
                self.n_batch as usize,
                self.n_vocab,
            )
            .is_ok()
                && {
                    llamac::mem_rm(self.ctx, 0, n - 1, -1);
                    let tail = &toks[toks.len() - 1..];
                    llamac::batch_decode(
                        self.ctx,
                        &[llamac::Dec::one(tail, n - 1)],
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

    /// Fraction of the model's raw next-token mass landing on this item's
    /// letter set — snapjudge's "coverage": how much belief falls inside the
    /// allowed answers before we renormalize. Low coverage means the model
    /// wanted to say something else entirely.
    fn coverage(&self, row: &[f32], n_slots: usize) -> f64 {
        let max = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)) as f64;
        let mut ids: Vec<i32> = self.letter_ids[..n_slots]
            .iter()
            .flatten()
            .copied()
            .collect();
        ids.sort_unstable();
        let (mut all, mut hit, mut next) = (0.0f64, 0.0f64, 0usize);
        for (i, &l) in row.iter().enumerate() {
            let e = ((l as f64) - max).exp();
            all += e;
            while next < ids.len() && (ids[next] as usize) < i {
                next += 1;
            }
            if next < ids.len() && ids[next] as usize == i {
                hit += e;
            }
        }
        if all > 0.0 {
            hit / all
        } else {
            0.0
        }
    }

    /// Token index where a question_first head ends (STATE begins): render the
    /// same message with a sentinel state and take the shared prefix length —
    /// BPE may merge the separator into the first state token, so this stays a
    /// conservative boundary either way.
    fn head_split(&self, q: &Question, slots: &[Slot], toks: &[i32]) -> Result<usize> {
        let probe = prompts::user_message(STATE_SENTINEL, q, slots, Layout::QuestionFirst, &[]);
        let ptoks = self.prompt_tokens(&probe)?;
        // clamp so at least the final (answer-position) token stays a suffix
        Ok(common_prefix_len(&[toks, ptoks.as_slice()]).min(toks.len().saturating_sub(1)))
    }

    /// Token index where the cached-prefix unit ends. state_first/header
    /// cache `[head+state]` — the question-independent span before the first
    /// question block; catalog caches `[head+QUESTIONS]` — the
    /// state-independent span before the STATE marker, reusable across
    /// requests on the same question set. The boundary is measured in-context
    /// by tokenizing the real message truncated at the marker.
    fn cache_prefix_len(&self, it: &Item, umsg: &str, layout: Layout) -> usize {
        let pos = match layout {
            Layout::Catalog => umsg.find("\nSTATE\n"),
            _ => umsg.rfind(&prompts::question_block(&it.q, &it.slots).join("\n")),
        };
        let Some(pos) = pos else { return 0 };
        let Ok(probe) = self.prompt_tokens(&umsg[..pos]) else {
            return 0;
        };
        common_prefix_len(&[&it.toks, &probe])
    }

    /// seq0's invariant: it only ever holds a prefix of `head_tokens`
    /// (request tokens live on scratch seqs — holder and work seqs — so seq0
    /// should never drift). Align it to exactly `[0, end)`: a drifted seq0 is
    /// rebuilt rather than trimmed, since partial rewind isn't guaranteed on
    /// hybrid archs; a too-short seq0 gets the missing head tokens decoded.
    unsafe fn align_seq0(&self, ctx: *mut sys::llama_context, end: usize) -> Result<()> {
        let pos = llamac::mem_seq_pos_max(ctx, 0) + 1;
        if pos > end as i32 {
            llamac::mem_rm(ctx, 0, -1, -1);
            llamac::batch_decode(
                ctx,
                &[llamac::Dec::one(&self.head_tokens[..end], 0)],
                self.n_batch as usize,
                self.n_vocab,
            )?;
        } else if (pos as usize) < end {
            llamac::batch_decode(
                ctx,
                &[llamac::Dec::one(&self.head_tokens[pos as usize..end], pos)],
                self.n_batch as usize,
                self.n_vocab,
            )?;
        }
        Ok(())
    }

    /// A failed batched decode can leave seqs holding partial installs or
    /// untrimmed suffixes — poison for every later request. Heads are cheap
    /// to reinstall, so wipe the whole context; seq0 is rebuilt lazily by
    /// align_seq0 on the next entry.
    unsafe fn reset_mctx(&mut self) {
        if let Some(ctx) = self.mctx {
            llamac::mem_clear(ctx);
            self.qseqs.clear();
        }
    }

    /// Multi-sequence decode on the batched context. Suffixes run on forked
    /// work seqs in waves — expanded choice probes can exceed one wave.
    /// seq0 only ever holds the resident head: the shared prefix lives on a
    /// scratch holder seq, decoded inside the same call as the first wave's
    /// suffixes (tagged to the holder and every live seq) and copied from the
    /// holder for later waves. Fork-and-discard — every teardown is a
    /// whole-seq rm, so hybrid/recurrent archs (no partial rewind) run this
    /// path too. Scratch seqs come from the qhead cache's allocator, so
    /// cached heads survive.
    fn decide_batched(
        &mut self,
        items: &[Item],
        hstart: usize,
        plen: usize,
        s1: usize,
        temperature: f64,
    ) -> Result<(Map<String, Value>, Instant, bool)> {
        let ctx = self.mctx.context("no mctx")?;
        unsafe { self.align_seq0(ctx, hstart)? };
        // The [0,s1) span (head + state) is question-independent — it lives
        // as a qseqs cache entry, so a later request on the same state (any
        // question set) seeds holder + work seqs by copy instead of decoding
        // the state again. On a miss the span is installed inside wave 0's
        // own decode: the same tokens are simply also tagged to the entry seq.
        let mut state_seq: Option<i32> = None;
        if s1 > hstart {
            match self.qseqs.get(&items[0].toks[..s1]) {
                // only trust the seq if it holds exactly this span — residue
                // of an aborted decode is a miss, not a hit
                Some(c) if unsafe { llamac::mem_seq_pos_max(ctx, c) } == s1 as i32 - 1 => {
                    state_seq = Some(c);
                }
                Some(c) => {
                    self.qseqs.remove(&items[0].toks[..s1]);
                    unsafe { llamac::mem_rm(ctx, c, -1, -1) };
                }
                None => {}
            }
        }
        // on a miss, reserve a seq for the state entry upfront — it rides
        // wave 0's decode. A state entry duplicates [hstart,s1) permanently,
        // so only install when the extra span still leaves wave room in the
        // shared n_ctx pool; oversized states just run uncached.
        let mut pending_entry = if state_seq.is_none() && s1 > hstart {
            let need = plen.max(s1).max(hstart)
                + self.qseqs.cells
                + KV_SLACK as usize
                + (s1 - hstart)
                + MIN_WAVE_ROOM;
            if need < self.n_ctx as usize {
                self.qseqs.alloc(self.mseqs, &HashSet::new())
            } else {
                None
            }
        } else {
            None
        };
        let entry_seq = state_seq.or(pending_entry);
        // how deep copies seed each seq on wave 0; the suffix start per item
        let seeded = if state_seq.is_some() { s1 } else { hstart };
        let start = if plen > 0 {
            plen
        } else if entry_seq.is_some() {
            s1
        } else {
            hstart
        };
        // An item whose whole prompt is the shared prefix has no suffix to
        // decode: its answer row is the one the prefix decode itself emits —
        // or, on a full hit, the row stored on the state entry.
        let mut prefix_row = if state_seq.is_some() && start == s1 {
            self.qseqs.row_of(&items[0].toks[..s1])
        } else {
            None
        };
        let mut t_prefill = Instant::now();
        // seq0, the holder and a live state-entry seq are all reserved —
        // the entry sits in by_head but is pinned for the whole request
        let wave = (self.mseqs - 2 - i32::from(entry_seq.is_some())).max(1) as usize;
        // each live seq allocates its suffix cells; the holder's span, the
        // resident head and cached entries all share the n_ctx cell pool
        let costs: Vec<usize> = items.iter().map(|it| it.toks.len() - start).collect();
        let mut rows_all: Vec<Option<Vec<f32>>> = vec![None; items.len()];
        let mut lo = 0;
        let mut holder: Option<i32> = None;
        while lo < items.len() {
            let mut held = start.max(self.head_tokens.len()) + self.qseqs.cells + KV_SLACK as usize;
            if lo == 0 && pending_entry.is_some() {
                held += s1 - hstart; // the entry's cells materialize this wave
            }
            let hi = wave_end(&costs, lo, wave, (self.n_ctx as usize).saturating_sub(held))?;
            let chunk = &items[lo..hi];
            let live: Vec<usize> = (0..chunk.len())
                .filter(|&i| chunk[i].toks.len() > start)
                .collect();
            let mut used = HashSet::new();
            // the holder and the state entry are scratch/cached seqs the
            // wave's allocs must not cannibalize
            used.extend(holder);
            used.extend(entry_seq);
            let mut seqs: Vec<i32> = Vec::with_capacity(live.len());
            for _ in &live {
                let s = self
                    .qseqs
                    .alloc(self.mseqs, &used)
                    .context("question heads exceed seq capacity")?;
                used.insert(s);
                seqs.push(s);
            }
            // wave 0 decodes the shared prefix once, tagged to a holder seq +
            // every live seq; later waves copy [0,start) from the holder —
            // seq0 is never written past its resident head, so nothing here
            // needs a partial rewind. align_seq0 keeps seq0's tail exactly at
            // hstart, so copying [0,hstart) reaches the recurrent tail cell
            // even on hybrid archs.
            if lo == 0 && plen > hstart {
                let h = self
                    .qseqs
                    .alloc(self.mseqs, &used)
                    .context("question heads exceed seq capacity")?;
                used.insert(h);
                holder = Some(h);
            }
            // seq-id lists backing the Dec groups must outlive the decode —
            // collect them first so groups can borrow into span_seqs
            let mut span_seqs: Vec<Vec<i32>> = Vec::new();
            // wave 0, miss: install the state span — tagged to the entry seq
            // plus everyone who needs it (holder+works in shared mode, works
            // in direct mode). Its last-token row is kept on the entry.
            let mut install_ix = None;
            if let (0, Some(e)) = (lo, pending_entry) {
                let mut ss = vec![e];
                if plen > hstart {
                    if let Some(h) = holder {
                        ss.push(h);
                    }
                }
                ss.extend_from_slice(&seqs);
                install_ix = Some(span_seqs.len());
                span_seqs.push(ss);
            }
            // wave 0, shared: the prefix span past whatever copies/install
            // already covered — just the question boilerplate when the state
            // span was a cache hit or got installed in the same batch
            let pfx_from = if install_ix.is_some() { s1 } else { seeded };
            let mut prefix_ix = None;
            if let (0, Some(h)) = (lo, holder) {
                if plen > pfx_from {
                    let mut ps = vec![h];
                    ps.extend_from_slice(&seqs);
                    prefix_ix = Some(span_seqs.len());
                    span_seqs.push(ps);
                }
            }
            // cluster live items by shared sub-prefix past `start`: same-
            // template questions (questionnaire families, option probes)
            // share more than the global plen — their shared span decodes
            // once, tagged to every seq in the cluster. row_dst records, in
            // emission order, which items consume each emitted row.
            let mut order: Vec<usize> = (0..live.len()).collect();
            order.sort_by(|&a, &b| chunk[live[a]].toks[start..].cmp(&chunk[live[b]].toks[start..]));
            struct Cluster {
                seqs: Vec<i32>,
                share: usize,
                rep: usize,
                ends: Vec<usize>,
                rem: Vec<(usize, usize)>,
            }
            let mut clusters: Vec<Cluster> = Vec::new();
            let mut singles: Vec<(usize, usize)> = Vec::new();
            let mut o = 0;
            while o < order.len() {
                let first = live[order[o]];
                let mut e = o + 1;
                while e < order.len()
                    && common_prefix_len(&[&chunk[first].toks, &chunk[live[order[e]]].toks])
                        > start + MIN_SUB_PREFIX
                {
                    e += 1;
                }
                let share = if e - o > 1 {
                    common_prefix_len(&[&chunk[first].toks, &chunk[live[order[e - 1]]].toks])
                } else {
                    start
                };
                if share > start {
                    clusters.push(Cluster {
                        seqs: order[o..e].iter().map(|&k| seqs[k]).collect(),
                        share,
                        rep: first,
                        ends: order[o..e]
                            .iter()
                            .filter(|&&k| chunk[live[k]].toks.len() == share)
                            .map(|&k| live[k])
                            .collect(),
                        rem: order[o..e]
                            .iter()
                            .filter(|&&k| chunk[live[k]].toks.len() > share)
                            .map(|&k| (k, live[k]))
                            .collect(),
                    });
                } else {
                    singles.extend(order[o..e].iter().map(|&k| (k, live[k])));
                }
                o = e;
            }
            let mut groups: Vec<llamac::Dec> = Vec::with_capacity(span_seqs.len() + seqs.len());
            if let Some(ix) = install_ix {
                groups.push(llamac::Dec::new(
                    &span_seqs[ix],
                    &items[0].toks[hstart..s1],
                    hstart as i32,
                    true,
                ));
            }
            if let Some(ix) = prefix_ix {
                groups.push(llamac::Dec::new(
                    &span_seqs[ix],
                    &items[0].toks[pfx_from..plen],
                    pfx_from as i32,
                    true,
                ));
            }
            let mut row_dst: Vec<Vec<usize>> = Vec::new();
            for c in &clusters {
                groups.push(llamac::Dec::new(
                    &c.seqs,
                    &chunk[c.rep].toks[start..c.share],
                    start as i32,
                    !c.ends.is_empty(),
                ));
                if !c.ends.is_empty() {
                    row_dst.push(c.ends.clone());
                }
                for &(k, i) in &c.rem {
                    groups.push(llamac::Dec::new(
                        &seqs[k..k + 1],
                        &chunk[i].toks[c.share..],
                        c.share as i32,
                        true,
                    ));
                    row_dst.push(vec![i]);
                }
            }
            for &(k, i) in &singles {
                groups.push(llamac::Dec::new(
                    &seqs[k..k + 1],
                    &chunk[i].toks[start..],
                    start as i32,
                    true,
                ));
                row_dst.push(vec![i]);
            }
            unsafe {
                // seed each live seq: wave 0 copies [0,seeded) — off the
                // state entry on a hit, else the head off seq0; later waves
                // copy [0,start) off the holder (shared) or the state entry
                // (direct). The holder itself is seeded once, on wave 0.
                for &s in seqs.iter().chain(&holder.filter(|_| lo == 0)) {
                    llamac::mem_rm(ctx, s, -1, -1);
                    let (src, base) = if lo == 0 {
                        (state_seq.unwrap_or(0), seeded)
                    } else {
                        (holder.or(entry_seq).unwrap_or(0), start)
                    };
                    if base > 0 {
                        llamac::mem_cp(ctx, src, s, 0, base as i32);
                    }
                }
                if let (0, Some(e)) = (lo, pending_entry) {
                    llamac::mem_rm(ctx, e, -1, -1);
                    if hstart > 0 {
                        llamac::mem_cp(ctx, 0, e, 0, hstart as i32);
                    }
                }
                if !groups.is_empty() {
                    let mut rows =
                        llamac::batch_decode(ctx, &groups, self.n_batch as usize, self.n_vocab)?;
                    let mut entry_row = None;
                    if install_ix.is_some() {
                        entry_row = Some(rows.remove(0));
                    }
                    if prefix_ix.is_some() {
                        prefix_row = Some(rows.remove(0));
                    }
                    // degenerate items ending at the state span take its row
                    if start == s1 && prefix_row.is_none() {
                        prefix_row = entry_row.clone();
                    }
                    for (dst, row) in row_dst.iter().zip(rows) {
                        for &i in dst {
                            rows_all[lo + i] = Some(row.clone());
                        }
                    }
                    if let Some(e) = pending_entry {
                        self.qseqs.insert_row(
                            items[0].toks[..s1].to_vec(),
                            e,
                            s1 - hstart,
                            entry_row,
                        );
                        pending_entry = None;
                    }
                }
                for &s in &seqs {
                    llamac::mem_rm(ctx, s, -1, -1);
                }
            }
            if lo == 0 {
                t_prefill = Instant::now();
            }
            for (i, it) in chunk.iter().enumerate() {
                if it.toks.len() == start {
                    rows_all[lo + i] = prefix_row.clone();
                }
            }
            lo = hi;
        }
        // the holder was request-scoped scratch — wipe it whole
        if let Some(h) = holder {
            unsafe { llamac::mem_rm(ctx, h, -1, -1) };
        }
        // a still-pending entry (wave 0 never decoded) is scratch — wipe it
        if let Some(e) = pending_entry {
            unsafe { llamac::mem_rm(ctx, e, -1, -1) };
        }
        // state entries count against the cache's cell share — trim LRU
        unsafe {
            self.qseqs.trim_to(
                ctx,
                (self.n_ctx / QCACHE_CELL_DIV).max(1) as usize,
                &HashSet::new(),
            );
        }
        let mut answers = Map::new();
        for (it, row) in items.iter().zip(rows_all) {
            let row = row.context("item produced no logits")?;
            let logits = self.letter_logits(&row, it.slots.len());
            let mut ans = decisions::decode(&it.q, &it.slots, &logits, temperature * it.calib);
            ans["coverage"] = json!(r4(self.coverage(&row, it.slots.len())));
            answers.insert(it.name.clone(), ans);
        }
        Ok((answers, t_prefill, state_seq.is_some()))
    }

    /// question_first on the multi-seq ctx: every item's head lives on its own
    /// seq (cached across requests, never written past the head — partial
    /// rewind isn't guaranteed); each item's state suffix decodes on a
    /// scratch work seq forked off the head. Waves are packed to the KV
    /// budget and stop early when the seq space is exhausted — all seqs share
    /// one n_ctx cell pool, and a wave that overruns it leaves dirty seqs
    /// behind.
    fn decide_qfirst_mctx(
        &mut self,
        items: &[Item],
        hstart: usize,
        temperature: f64,
    ) -> Result<DecodeOut> {
        let ctx = self.mctx.context("no mctx")?;
        unsafe {
            self.align_seq0(ctx, self.head_tokens.len())?;
            // seqs not backing a cached head must be empty — wipe strays so
            // the wave budgets below see every occupied cell
            for s in 1..self.mseqs {
                if !self.qseqs.by_head.values().any(|e| e.seq == s) {
                    llamac::mem_rm(ctx, s, -1, -1);
                }
            }
        }
        let mut rows_all: Vec<Option<Vec<f32>>> = vec![None; items.len()];
        let mut stat = WaveStat::default();
        // worst-case new cells per item: install span + suffix (a hit costs
        // less — its head cells already exist)
        let costs: Vec<usize> = items.iter().map(|it| it.toks.len() - hstart).collect();
        let mut lo = 0;
        while lo < items.len() {
            let held = self.head_tokens.len() + self.qseqs.cells + KV_SLACK as usize;
            let hi = wave_end(
                &costs,
                lo,
                (self.mseqs - 1) as usize,
                (self.n_ctx as usize).saturating_sub(held),
            )?;
            // pin the head seqs of not-yet-processed items: without this,
            // scratch allocs and fresh installs evict exactly the heads the
            // request is about to need (cyclic access = worst-case LRU).
            // Capped so at least one seq stays allocatable as a work seq.
            let cap = (self.mseqs - 2).max(0) as usize;
            let mut pinned = HashSet::new();
            for it in &items[lo..] {
                if pinned.len() >= cap {
                    break;
                }
                let split = it.split.max(hstart);
                if split > hstart {
                    if let Some(c) = self.qseqs.peek(&it.toks[..split]) {
                        pinned.insert(c);
                    }
                }
            }
            let took = self.qfirst_wave(
                ctx,
                &items[lo..hi],
                hstart,
                &pinned,
                &mut rows_all[lo..hi],
                &mut stat,
            )?;
            lo += took;
        }
        let t_prefill = Instant::now();
        let mut answers = Map::new();
        for (it, row) in items.iter().zip(rows_all) {
            let row = row.context("item produced no logits")?;
            let logits = self.letter_logits(&row, it.slots.len());
            let mut ans = decisions::decode(&it.q, &it.slots, &logits, temperature * it.calib);
            ans["coverage"] = json!(r4(self.coverage(&row, it.slots.len())));
            answers.insert(it.name.clone(), ans);
        }
        Ok(DecodeOut {
            answers,
            t_prefill,
            decoded: stat.decoded,
            hits: stat.hits,
            misses: stat.misses,
        })
    }

    /// One wave of question_first items: assign each a scratch work seq
    /// (forked off its head seq), then decode installs and suffixes in ONE
    /// batched call — a fresh install tags its head tokens with the cache seq
    /// plus every same-head work seq (shared cells at write time), a cached
    /// or already-installed head is cloned onto the work seq. Head seqs are
    /// never written past their head, so teardown is a whole-seq rm on work
    /// seqs only — no partial rewind needed anywhere. Returns how many
    /// chunk items were consumed (the wave can stop early on seq pressure).
    /// Any error leaves dirty seqs behind — the caller must reset_mctx.
    fn qfirst_wave(
        &mut self,
        ctx: *mut sys::llama_context,
        chunk: &[Item],
        hstart: usize,
        pinned: &HashSet<i32>,
        rows: &mut [Option<Vec<f32>>],
        stat: &mut WaveStat,
    ) -> Result<usize> {
        let mut used: HashSet<i32> = pinned.clone();
        let mut works: Vec<i32> = Vec::with_capacity(chunk.len());
        // seqs that still need the resident head [0,hstart) — everyone except
        // hit-copied works (they get [0,split) from the head seq in one go)
        let mut need_base: Vec<i32> = Vec::new();
        // (head tokens beyond the base, seqs) — the cache seq plus every
        // same-head work seq share the install's cells
        let mut installs: Vec<(&[i32], Vec<i32>)> = Vec::new();
        let mut took = 0;
        for it in chunk {
            let split = it.split.max(hstart);
            let head = &it.toks[..split];
            // where does this item's head come from? a materialized cache
            // seq to copy, an in-flight install group to join, or a fresh
            // install. A used seq is a sibling source, never a direct hit.
            let mut copy_from = None;
            let mut join = None;
            let mut stale = None;
            if split > hstart {
                match self.qseqs.get(head) {
                    // c is this head's in-flight install target — its cells
                    // don't exist yet, so join the install rather than copy
                    Some(c) if installs.iter().any(|(_, ss)| ss.contains(&c)) => {
                        join = installs
                            .iter()
                            .position(|(t, _)| **t == it.toks[hstart..split]);
                    }
                    Some(c) => {
                        // only trust the seq if it holds exactly this head —
                        // residue of an aborted decode is a miss, not a hit
                        if unsafe { llamac::mem_seq_pos_max(ctx, c) } == split as i32 - 1 {
                            copy_from = Some(c);
                        } else {
                            stale = Some((head.to_vec(), c));
                        }
                    }
                    None => {}
                }
            }
            // protect the hit's source seq before allocating — an eviction
            // between lookup and copy would hand back a gutted seq
            if let Some(c) = copy_from {
                used.insert(c);
            }
            let Some(w) = self.qseqs.alloc(self.mseqs, &used) else {
                break;
            };
            unsafe { llamac::mem_rm(ctx, w, -1, -1) };
            used.insert(w);
            match (copy_from, join) {
                (Some(c), _) => {
                    stat.hits += 1;
                    unsafe { llamac::mem_cp(ctx, c, w, 0, split as i32) };
                }
                (None, Some(gi)) => {
                    stat.hits += 1;
                    need_base.push(w);
                    installs[gi].1.push(w);
                }
                (None, None) => {
                    if let Some((k, c)) = stale {
                        self.qseqs.remove(&k);
                        unsafe { llamac::mem_rm(ctx, c, -1, -1) };
                    }
                    stat.misses += 1;
                    need_base.push(w);
                    if split > hstart {
                        // cache slot for the head; if the seq space is full
                        // the item still decodes — it just isn't cached
                        if let Some(c) = self.qseqs.alloc(self.mseqs, &used) {
                            unsafe { llamac::mem_rm(ctx, c, -1, -1) };
                            used.insert(c);
                            need_base.push(c);
                            installs.push((&it.toks[hstart..split], vec![c, w]));
                            self.qseqs.insert(head.to_vec(), c, split - hstart);
                        } else {
                            installs.push((&it.toks[hstart..split], vec![w]));
                        }
                        stat.decoded += split - hstart;
                    }
                }
            }
            works.push(w);
            took += 1;
        }
        if took == 0 {
            bail!("question heads exceed seq capacity");
        }
        let chunk = &chunk[..took];
        let mut groups: Vec<llamac::Dec> = Vec::with_capacity(installs.len() + chunk.len() + 1);
        // hybrid + divergent head (hstart < head len): partial seq0 copies
        // don't reach its recurrent tail — decode the shared head tokens onto
        // this wave's seqs instead. Rare; templates normally keep hstart ==
        // head_tokens.len().
        let head_gap = self.hybrid && hstart < self.head_tokens.len() && hstart > 0;
        let mut head_gap_seqs = need_base.clone();
        head_gap_seqs.sort_unstable();
        head_gap_seqs.dedup();
        unsafe {
            if !head_gap && hstart > 0 {
                for &s in &need_base {
                    llamac::mem_cp(ctx, 0, s, 0, hstart as i32);
                }
            }
        }
        if head_gap {
            groups.push(llamac::Dec::new(
                &head_gap_seqs,
                &self.head_tokens[..hstart],
                0,
                false,
            ));
        }
        // installs first so a chunk split never orders a suffix before its head
        for (head_rest, seqs) in &installs {
            groups.push(llamac::Dec::new(
                seqs.as_slice(),
                head_rest,
                hstart as i32,
                false,
            ));
        }
        for (i, it) in chunk.iter().enumerate() {
            let split = it.split.max(hstart);
            groups.push(llamac::Dec::new(
                std::slice::from_ref(&works[i]),
                &it.toks[split..],
                split as i32,
                true,
            ));
        }
        unsafe {
            if !groups.is_empty() {
                for (i, row) in
                    llamac::batch_decode(ctx, &groups, self.n_batch as usize, self.n_vocab)?
                        .into_iter()
                        .enumerate()
                {
                    rows[i] = Some(row);
                }
            }
            // work seqs are scratch — wipe whole. Their head cells (when
            // shared with a cache seq) survive through the cache seq's tag.
            for &w in &works {
                llamac::mem_rm(ctx, w, -1, -1);
            }
            for it in chunk {
                stat.decoded += it.toks.len() - it.split.max(hstart);
            }
            // keep the head cache within its share of the pool so later
            // waves still have cells to work with
            self.qseqs
                .trim_to(ctx, (self.n_ctx / QCACHE_CELL_DIV).max(1) as usize, &used);
        }
        Ok(took)
    }

    /// question_first on the single-seq ctx: each question head is a whole-
    /// state snapshot (QSnaps LRU); the state suffix decodes on seq 0.
    fn decide_qfirst_snaps(
        &mut self,
        items: &[Item],
        hstart: usize,
        temperature: f64,
    ) -> Result<DecodeOut> {
        let (mut hits, mut misses, mut decoded) = (0usize, 0usize, 0usize);
        let mut rows: Vec<Vec<f32>> = Vec::with_capacity(items.len());
        for it in items {
            let split = it.split.max(hstart);
            let head = &it.toks[..split];
            unsafe {
                if let Some(snap) = self.qsnaps.get(head) {
                    hits += 1;
                    llamac::state_load(self.ctx, snap);
                } else {
                    misses += 1;
                    // reset ctx to head-only, decode the question head, snapshot
                    match &self.head_state {
                        Some(st) => llamac::state_load(self.ctx, st),
                        None => self.align_seq0(self.ctx, hstart)?,
                    }
                    if it.toks.len() > hstart && split > hstart {
                        llamac::batch_decode(
                            self.ctx,
                            &[llamac::Dec::one(&it.toks[hstart..split], hstart as i32)],
                            self.n_batch as usize,
                            self.n_vocab,
                        )?;
                        decoded += split - hstart;
                    }
                    let snap = llamac::state_save(self.ctx);
                    self.qsnaps.insert(head.to_vec(), snap);
                }
                let row = llamac::batch_decode(
                    self.ctx,
                    &[llamac::Dec::one(&it.toks[split..], split as i32)],
                    self.n_batch as usize,
                    self.n_vocab,
                )?
                .remove(0);
                decoded += it.toks.len() - split;
                rows.push(row);
            }
        }
        // seq0 must be left holding only resident-head tokens
        if self.head_state.is_none() {
            unsafe { llamac::mem_rm(self.ctx, 0, hstart as i32, -1) };
        }
        let t_prefill = Instant::now();
        let mut answers = Map::new();
        for (it, row) in items.iter().zip(rows) {
            let logits = self.letter_logits(&row, it.slots.len());
            let mut ans = decisions::decode(&it.q, &it.slots, &logits, temperature * it.calib);
            ans["coverage"] = json!(r4(self.coverage(&row, it.slots.len())));
            answers.insert(it.name.clone(), ans);
        }
        Ok(DecodeOut {
            answers,
            t_prefill,
            decoded,
            hits,
            misses,
        })
    }

    pub fn decide(&mut self, req: &DecideRequest) -> Result<Value> {
        req.validate()?;
        let t0 = Instant::now();
        let calib = |b: &str| self.calibration.as_ref().map(|c| c.temp(b)).unwrap_or(1.0);

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
                    let nsubs = subs.len();
                    // expanded probes are boolean questions mechanically: the
                    // boolean bucket temperature applies; paged items are real
                    // choices and take the choice bucket
                    let b = match req.expand {
                        Expand::Pages => "choice",
                        _ => "boolean",
                    };
                    for (i, sq) in subs.into_iter().enumerate() {
                        pending.push((format!("{name}\u{1f}{i}"), sq, calib(b)));
                    }
                    expanded.push((name, q, keys, nsubs));
                }
                None => {
                    let c = calib(crate::calibrate::bucket(q.qtype.as_str()));
                    pending.push((name, q, c));
                }
            }
        }

        // the state renders once per request — every item and the auto
        // heuristic share the same text
        let state_txt = if req.compact_state {
            prompts::render_state_compact(&req.state)
        } else {
            prompts::render_state(&req.state)
        };

        let layout = match req.layout {
            Layout::Auto => {
                // abstain questions read better after the evidence: the
                // __abstain__ slot before the state primes abstention
                // (measured: -5..-16pp on eval/edge across the model set)
                let state_len = state_txt.len();
                let long_multi = pending.len() > 1 && state_len > LONG_STATE_CHARS;
                // Steady-state cost: warm question_first re-decodes the state
                // per question (n×state) while state_first pays it once plus
                // every question head (state + Σheads). qf wins only when the
                // heads dominate — big rubric questions over a short state.
                // (missile game: 13 questions on a 500-char state were 3×
                // slower under qf even with all heads cached.)
                let head_sum: usize = pending
                    .iter()
                    .map(|(_, q, _)| {
                        q.instructions.len()
                            + q.criteria
                                .as_ref()
                                .map(|c| c.to_string().len())
                                .unwrap_or(0)
                    })
                    .sum();
                let n = pending.len();
                let heads_dominate = n <= 1 || head_sum > (n - 1) * state_len;
                if any_abstain || long_multi || !heads_dominate {
                    Layout::StateFirst
                } else {
                    Layout::QuestionFirst
                }
            }
            l => l,
        };
        let preamble: Vec<String> = if layout == Layout::Header {
            originals
                .iter()
                .enumerate()
                .map(|(i, (n, instr))| {
                    if instr.is_empty() {
                        format!("{}. {n}", i + 1)
                    } else {
                        format!("{}. {n} — {instr}", i + 1)
                    }
                })
                .collect()
        } else {
            Vec::new()
        };

        // the QUESTIONS catalog for Layout::Catalog — one numbered entry per
        // item (expanded probes included), built before the per-item prompts
        let slots_all: Vec<Vec<Slot>> = pending
            .iter()
            .map(|(_, q, _)| prompts::slots_for(q))
            .collect();
        let cat: Vec<String> = if layout == Layout::Catalog {
            let mut v = Vec::new();
            for (i, (name, q, _)) in pending.iter().enumerate() {
                if i > 0 {
                    v.push(String::new());
                }
                // expanded subs carry the `name\x1fi` internal tag — show the
                // question name in the catalog, the number already points
                v.extend(prompts::catalog_entry(
                    i,
                    name.split('\u{1f}').next().unwrap(),
                    q,
                ));
            }
            v
        } else {
            Vec::new()
        };

        let mut items = Vec::new();
        for (idx, ((name, q, c), slots)) in pending.into_iter().zip(slots_all).enumerate() {
            let msg = if layout == Layout::Catalog {
                let dname = name.split('\u{1f}').next().unwrap();
                prompts::catalog_message(&cat, &state_txt, idx, dname, &slots)
            } else {
                prompts::user_message(&state_txt, &q, &slots, layout, &preamble)
            };
            let toks = self.prompt_tokens(&msg)?;
            if toks.len() as i32 > self.n_ctx {
                bail!(
                    "question {name:?}: prompt is {} tokens, ctx is {}",
                    toks.len(),
                    self.n_ctx
                );
            }
            let split = if layout == Layout::QuestionFirst {
                self.head_split(&q, &slots, &toks)?
            } else {
                0
            };
            items.push(Item {
                name,
                q,
                slots,
                toks,
                split,
                calib: c,
            });
        }

        let token_lists: Vec<&[i32]> = items.iter().map(|t| t.toks.as_slice()).collect();
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

        let mut hits = 0usize;
        let mut misses = 0usize;
        let mut suffix_mode = "sequential";
        let mut state_hit = false;
        let mut t_prefill = t0;
        let mut plen = 0usize;
        let mut decoded_tokens = 0usize;

        let mut answers = if layout == Layout::QuestionFirst {
            let mut r = if self.mctx.is_some() {
                suffix_mode = "batched";
                self.decide_qfirst_mctx(&items, hstart, req.temperature)
            } else {
                self.decide_qfirst_snaps(&items, hstart, req.temperature)
            };
            if r.is_err() && self.mctx.is_some() {
                // a dead batched decode leaves poisoned seqs — wipe them and
                // answer on the single-seq path instead of failing
                if let Err(e) = &r {
                    eprintln!("snap: batched qfirst decode failed, falling back: {e}");
                }
                unsafe { self.reset_mctx() };
                suffix_mode = "fallback";
                r = self.decide_qfirst_snaps(&items, hstart, req.temperature);
            }
            let out = r?;
            t_prefill = out.t_prefill;
            hits = out.hits;
            misses = out.misses;
            decoded_tokens = out.decoded;
            out.answers
        } else {
            let plen_local = if items.len() > 1 && req.mode == Mode::Shared {
                common_prefix_len(&token_lists)
            } else {
                0
            };
            plen = plen_local;
            // The cached-prefix unit ends where the question/state-dependent
            // part starts: [head+state] for state_first/header, [head+catalog]
            // for catalog. s1<=hstart (tiny/absent unit) disables the entry.
            let s1 = if items.is_empty() {
                0
            } else {
                let umsg = match layout {
                    Layout::Catalog => prompts::catalog_message(
                        &cat,
                        &state_txt,
                        0,
                        items[0].name.split('\u{1f}').next().unwrap(),
                        &items[0].slots,
                    ),
                    _ => prompts::user_message(
                        &state_txt,
                        &items[0].q,
                        &items[0].slots,
                        layout,
                        &preamble,
                    ),
                };
                let s = self.cache_prefix_len(&items[0], &umsg, layout);
                if plen > 0 {
                    s.min(plen)
                } else {
                    s
                }
            };
            let mut answers: Option<Map<String, Value>> = None;
            if self.mctx.is_some() {
                match self.decide_batched(&items, hstart, plen, s1, req.temperature) {
                    Ok((a, tp, sh)) => {
                        answers = Some(a);
                        t_prefill = tp;
                        state_hit = sh;
                        suffix_mode = "batched";
                    }
                    // wipe poisoned seqs before the sequential fallback
                    Err(e) => {
                        eprintln!("snap: batched decode failed, falling back: {e}");
                        unsafe { self.reset_mctx() }
                    }
                }
            }
            match answers {
                Some(a) => a,
                None => {
                    self.decide_sequential(&items, hstart, plen, req.temperature, &mut t_prefill)?
                }
            }
        };

        // fold per-option probes / pages back into one answer per expanded
        // choice
        for (name, q, keys, nsubs) in &expanded {
            let mut cov = 0.0;
            let mut merged = match req.expand {
                Expand::Pages => {
                    let mut pages = Vec::with_capacity(*nsubs);
                    for i in 0..*nsubs {
                        let a = answers
                            .remove(&format!("{name}\u{1f}{i}"))
                            .unwrap_or_default();
                        cov += a["coverage"].as_f64().unwrap_or(0.0);
                        pages.push(a);
                    }
                    decisions::merge_paged(q, keys, &pages)
                }
                _ => {
                    let mut p_yes = Vec::with_capacity(keys.len());
                    for i in 0..*nsubs {
                        let a = answers
                            .remove(&format!("{name}\u{1f}{i}"))
                            .unwrap_or_default();
                        p_yes.push(a["probabilities"]["yes"].as_f64().unwrap_or(0.0));
                        cov += a["coverage"].as_f64().unwrap_or(0.0);
                    }
                    decisions::merge_scored(q, keys, &p_yes)
                }
            };
            merged["coverage"] = json!(r4(cov / (*nsubs).max(1) as f64));
            answers.insert(name.clone(), merged);
        }

        let total_ms = t0.elapsed().as_secs_f64() * 1000.0;
        // tokens actually decoded: shared prefix once + every suffix. The
        // question_first path counts real decodes (cached heads are free).
        let input_tokens = if layout == Layout::QuestionFirst {
            decoded_tokens
        } else {
            plen + token_lists.iter().map(|t| t.len() - plen).sum::<usize>()
        };
        let mut x = Map::new();
        x.insert("layout".into(), json!(layout_str(layout)));
        x.insert("shared_prefix_tokens".into(), json!(plen));
        x.insert("cached_head_tokens".into(), json!(hstart));
        if layout == Layout::QuestionFirst {
            x.insert("qhead_hits".into(), json!(hits));
            x.insert("qhead_misses".into(), json!(misses));
        } else {
            x.insert("state_hit".into(), json!(state_hit));
        }
        x.insert(
            "rewind".into(),
            json!(if self.can_rewind { "kv" } else { "snapshot" }),
        );
        x.insert("suffix_decode".into(), json!(suffix_mode));
        x.insert("decoded_items".into(), json!(items.len()));
        x.insert(
            "prefill_ms".into(),
            json!(((t_prefill - t0).as_secs_f64() * 1000.0 * 10.0).round() / 10.0),
        );
        x.insert("total_ms".into(), json!((total_ms * 10.0).round() / 10.0));
        Ok(json!({
            "answers": answers,
            "model": self.model_id,
            "usage": {"input_tokens": input_tokens},
            "x_snap": Value::Object(x),
        }))
    }

    /// Shared-prefix path on the single-seq ctx: prefill the common prefix once
    /// (seq 0), then rewind/snapshot-restore per question suffix.
    fn decide_sequential(
        &mut self,
        items: &[Item],
        hstart: usize,
        plen: usize,
        temperature: f64,
        t_prefill: &mut Instant,
    ) -> Result<Map<String, Value>> {
        let mut prefix_row = None;
        unsafe {
            if self.can_rewind {
                self.align_seq0(self.ctx, hstart)?; // keep resident head
            } else if let Some(st) = &self.head_state {
                llamac::state_load(self.ctx, st);
            } else {
                llamac::mem_clear(self.ctx);
            }
            if plen > hstart {
                prefix_row = llamac::batch_decode(
                    self.ctx,
                    &[llamac::Dec::one(
                        &items[0].toks[hstart..plen],
                        hstart as i32,
                    )],
                    self.n_batch as usize,
                    self.n_vocab,
                )?
                .into_iter()
                .next();
            }
        }
        let start = if plen > 0 { plen } else { hstart };
        // the snapshot only pays off when a second suffixed item must restore
        // the state — a lone suffix runs right on top of the prefill
        let suffixed = items.iter().filter(|it| it.toks.len() > start).count();
        let state = if self.can_rewind || suffixed < 2 {
            None
        } else {
            Some(unsafe { llamac::state_save(self.ctx) })
        };
        // ctx still holds the saved/prefilled state — the first suffixed item
        // needs no restore; only a decode dirties it
        let mut fresh = true;
        *t_prefill = Instant::now();

        let mut out = Map::new();
        for it in items {
            // whole prompt inside the shared prefix: reuse its logit row
            let row = if it.toks.len() == start {
                prefix_row.clone().context("missing shared-prefix logits")?
            } else {
                unsafe {
                    match &state {
                        Some(st) if !fresh => llamac::state_load(self.ctx, st),
                        _ if self.can_rewind => llamac::mem_rm(self.ctx, 0, start as i32, -1),
                        _ => {}
                    }
                    fresh = false;
                    llamac::batch_decode(
                        self.ctx,
                        &[llamac::Dec::one(&it.toks[start..], start as i32)],
                        self.n_batch as usize,
                        self.n_vocab,
                    )?
                    .remove(0)
                }
            };
            let logits = self.letter_logits(&row, it.slots.len());
            let mut ans = decisions::decode(&it.q, &it.slots, &logits, temperature * it.calib);
            ans["coverage"] = json!(r4(self.coverage(&row, it.slots.len())));
            out.insert(it.name.clone(), ans);
        }
        // seq0 must be left holding only resident-head tokens
        if self.can_rewind {
            unsafe { llamac::mem_rm(self.ctx, 0, hstart as i32, -1) };
        }
        Ok(out)
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
    let subs = match expand {
        // the shared stem ends at "Candidate:" so suffix clustering decodes
        // only the option text + yes/no tail per probe
        Expand::Probes => opts
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
                        "{base}\nIs this candidate the correct answer?\nCandidate: {cand}"
                    ),
                    criteria: None,
                    min: None,
                    max: None,
                    step: None,
                    granularity: crate::schema::default_granularity(),
                    allow_abstain: false,
                }
            })
            .collect(),
        // equal-size pages: within-page probabilities compare across pages,
        // so a 4-option tail page must not outscore a 26-option page
        Expand::Pages => {
            let npages = opts.len().div_ceil(MAX_SLOTS);
            let size = opts.len().div_ceil(npages);
            opts.chunks(size)
                .map(|page| {
                    let criteria: serde_json::Map<String, Value> = page
                        .iter()
                        .map(|(k, d)| (k.clone(), Value::String(d.clone())))
                        .collect();
                    Question {
                        qtype: QType::Choice,
                        instructions: format!(
                            "{base}\nThese candidates are a subset — pick the best among them."
                        ),
                        criteria: Some(Value::Object(criteria)),
                        min: None,
                        max: None,
                        step: None,
                        granularity: crate::schema::default_granularity(),
                        allow_abstain: false,
                    }
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
    use super::{common_prefix_len, wave_end, QSeqs};
    use std::collections::HashSet;

    #[test]
    fn common_prefix() {
        assert_eq!(common_prefix_len(&[]), 0);
        assert_eq!(common_prefix_len(&[&[1, 2, 3]]), 3);
        assert_eq!(common_prefix_len(&[&[1, 2, 3], &[1, 2, 4]]), 2);
        assert_eq!(common_prefix_len(&[&[1, 2], &[1, 2, 3, 4]]), 2);
        assert_eq!(common_prefix_len(&[&[1], &[2]]), 0);
        assert_eq!(common_prefix_len(&[&[1, 2, 3], &[1, 2, 3], &[1, 2, 3]]), 3);
    }

    #[test]
    fn wave_packing() {
        // count cap
        assert_eq!(wave_end(&[1; 10], 0, 4, 1000).unwrap(), 4);
        // budget cap: 5+5+5 > 12 -> two fit
        assert_eq!(wave_end(&[5, 5, 5], 0, 64, 12).unwrap(), 2);
        // exact fit ok, next item starts a new wave
        assert_eq!(wave_end(&[5, 5, 5], 0, 64, 10).unwrap(), 2);
        // mid-slice
        assert_eq!(wave_end(&[5, 5, 5], 2, 64, 12).unwrap(), 3);
        // zero-cost items only bounded by count
        assert_eq!(wave_end(&[0; 200], 0, 64, 1).unwrap(), 64);
        // a lone item over budget errors (caller falls back, doesn't hang)
        assert!(wave_end(&[11], 0, 64, 10).is_err());
        assert!(wave_end(&[1, 11], 1, 64, 10).is_err());
    }

    #[test]
    fn qseqs_bookkeeping() {
        let mut q = QSeqs::default();
        let h1 = vec![1, 2, 3];
        let h2 = vec![4, 5, 6];
        q.insert(h1.clone(), 7, 30);
        q.insert(h2.clone(), 8, 20);
        assert_eq!(q.cells, 50);
        // re-insert same head replaces cells, doesn't double count
        q.insert(h1.clone(), 9, 10);
        assert_eq!(q.cells, 30);
        assert_eq!(q.get(&h1), Some(9));
        q.remove(&h1);
        assert_eq!(q.cells, 20);
        q.clear();
        assert_eq!(q.cells, 0);
        assert!(q.get(&h2).is_none());
    }

    #[test]
    fn qseqs_alloc_free_then_lru() {
        let mut q = QSeqs::default();
        let nseq = super::llamac::MAX_SEQS;
        let used = HashSet::new();
        // free seqs come first (lowest available)
        assert_eq!(q.alloc(nseq, &used), Some(1));
        q.insert(vec![1], 1, 10);
        q.insert(vec![2], 2, 10);
        // seqs 1,2 tracked -> next free is 3
        assert_eq!(q.alloc(nseq, &used), Some(3));
        // every seq tracked/used -> evict the least recent head's seq
        let mut all_used: HashSet<i32> = (1..nseq).collect();
        for s in 3..nseq {
            q.insert(vec![100 + s], s, 1);
        }
        assert_eq!(q.cells, 10 + 10 + 62);
        all_used.remove(&2); // seq 2's head is evictable
        assert_eq!(q.alloc(nseq, &all_used), Some(2));
        assert_eq!(q.cells, 10 + 62); // head {2}'s 10 cells released
    }
}
