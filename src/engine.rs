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
//!
//! `layout: question_first` flips the prompt (QUESTION+OPTIONS, then STATE):
//! nothing is shared inside one request, but the question head is identical
//! across requests — it stays resident on an mctx seq (or as a ctx state
//! snapshot on hybrid archs) in a tick-LRU cache, so repeat workloads decode
//! only the state. Stolen from snapjudge, which measured it both faster and
//! more accurate on short states.

use std::collections::{HashMap, HashSet};
use std::ffi::{c_char, c_void, CStr, CString};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use llama_cpp_sys_2 as sys;
use serde_json::{json, Map, Value};

use crate::decisions;
use crate::llamac;
use crate::prompts::{self, Slot};
use crate::schema::{DecideRequest, Layout, Mode, QType, Question, MAX_SLOTS};

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

#[derive(Clone, Copy)]
struct QEnt {
    seq: i32,
    tick: u64,
    cells: usize,
}

impl QSeqs {
    fn get(&mut self, head: &[i32]) -> Option<i32> {
        self.tick += 1;
        let e = self.by_head.get_mut(head)?;
        e.tick = self.tick;
        Some(e.seq)
    }

    /// A seq that is free or whose head can be evicted (not `used` this wave).
    fn alloc(&mut self, used: &HashSet<i32>) -> Option<i32> {
        self.tick += 1;
        let free = (1..llamac::MAX_SEQS)
            .find(|s| !used.contains(s) && self.by_head.values().all(|e| e.seq != *s));
        if free.is_some() {
            return free;
        }
        let victim = self
            .by_head
            .iter()
            .filter(|(_, e)| !used.contains(&e.seq))
            .min_by_key(|(_, e)| e.tick)
            .map(|(k, e)| (k.clone(), *e));
        victim.map(|(k, e)| {
            self.by_head.remove(&k);
            self.cells -= e.cells;
            e.seq
        })
    }

    fn insert(&mut self, head: Vec<i32>, seq: i32, cells: usize) {
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
                .map(|(k, e)| (k.clone(), *e));
            let Some((k, e)) = victim else { break };
            self.by_head.remove(&k);
            self.cells -= e.cells;
            llamac::mem_rm(ctx, e.seq, -1, -1);
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
    fn get(&mut self, head: &[i32]) -> Option<Vec<u8>> {
        self.tick += 1;
        let e = self.by_head.get_mut(head)?;
        e.1 = self.tick;
        Some(e.0.clone())
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
    head_tokens: Vec<i32>,
    head_state: Option<Vec<u8>>,
    qseqs: QSeqs,
    qsnaps: QSnaps,
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
            qseqs: QSeqs::default(),
            qsnaps: QSnaps::default(),
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
        let probe = prompts::user_message(
            &Value::String(STATE_SENTINEL.into()),
            q,
            slots,
            Layout::QuestionFirst,
            &[],
        );
        let ptoks = self.prompt_tokens(&probe)?;
        // clamp so at least the final (answer-position) token stays a suffix
        Ok(common_prefix_len(&[toks, ptoks.as_slice()]).min(toks.len().saturating_sub(1)))
    }

    /// seq0's invariant: it only ever holds a prefix of `head_tokens`
    /// (request-token spans decoded on it are trimmed back before each
    /// function returns). Align it to exactly `[0, end)`: trim leftovers a
    /// previous request may have left, re-decode head tokens a deeper trim
    /// or a reset may have dropped.
    unsafe fn align_seq0(&self, ctx: *mut sys::llama_context, end: usize) -> Result<()> {
        let pos = llamac::mem_seq_pos_max(ctx, 0) + 1;
        if pos > end as i32 {
            llamac::mem_rm(ctx, 0, end as i32, -1);
        } else if (pos as usize) < end {
            llamac::batch_decode(
                ctx,
                &[(0, &self.head_tokens[pos as usize..end], pos)],
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
            // this path owns every seq for its own suffixes — wiping them
            // destroys the qfirst head cache, so drop its map too
            for seq in 1..llamac::MAX_SEQS {
                llamac::mem_rm(ctx, seq, -1, -1);
            }
            self.qseqs.clear();
            self.align_seq0(ctx, hstart)?;
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
        // each live seq allocates its suffix cells; seq0 holds `start`
        let costs: Vec<usize> = items.iter().map(|it| it.toks.len() - start).collect();
        let budget = (self.n_ctx - start as i32 - KV_SLACK).max(0) as usize;
        let mut rows_all: Vec<Option<Vec<f32>>> = vec![None; items.len()];
        let mut lo = 0;
        while lo < items.len() {
            let hi = wave_end(&costs, lo, wave, budget)?;
            let chunk = &items[lo..hi];
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
                        rows_all[lo + i] = Some(row);
                    }
                }
            }
            for (i, it) in chunk.iter().enumerate() {
                if it.toks.len() == start {
                    rows_all[lo + i] = prefix_row.clone();
                }
            }
            lo = hi;
        }
        // seq0 must be left holding only resident-head tokens
        unsafe { llamac::mem_rm(ctx, 0, hstart as i32, -1) };
        let mut answers = Map::new();
        for (it, row) in items.iter().zip(rows_all) {
            let row = row.context("item produced no logits")?;
            let logits = self.letter_logits(&row, it.slots.len());
            let mut ans = decisions::decode(&it.q, &it.slots, &logits, temperature * it.calib);
            ans["coverage"] = json!(r4(self.coverage(&row, it.slots.len())));
            answers.insert(it.name.clone(), ans);
        }
        Ok((answers, t_prefill))
    }

    /// question_first on the multi-seq ctx: every item's head lives on its own
    /// seq (cached across requests); all state suffixes decode in one batched
    /// call per wave, then each seq is trimmed back to its head. Waves are
    /// packed to the KV budget — all seqs share one n_ctx cell pool, and a
    /// wave that overruns it leaves dirty seqs behind.
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
            for s in 1..llamac::MAX_SEQS {
                if !self.qseqs.by_head.values().any(|e| e.seq == s) {
                    llamac::mem_rm(ctx, s, -1, -1);
                }
            }
        }
        let wave = (llamac::MAX_SEQS - 1) as usize;
        let mut rows_all: Vec<Option<Vec<f32>>> = vec![None; items.len()];
        let mut stat = WaveStat::default();
        // worst-case new cells per item: install span + suffix (a hit costs
        // less — its head cells already exist)
        let costs: Vec<usize> = items.iter().map(|it| it.toks.len() - hstart).collect();
        let mut lo = 0;
        while lo < items.len() {
            let held = self.head_tokens.len() + self.qseqs.cells + KV_SLACK as usize;
            let hi = wave_end(&costs, lo, wave, (self.n_ctx as usize).saturating_sub(held))?;
            self.qfirst_wave(
                ctx,
                &items[lo..hi],
                hstart,
                &mut rows_all[lo..hi],
                &mut stat,
            )?;
            lo = hi;
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

    /// One wave of question_first items: assign each a seq (cache hit, fresh
    /// install, or clone of a same-head sibling), decode installs then
    /// suffixes in one batched call each, trim every seq back to its head.
    /// Any error leaves untrimmed/partial seqs — the caller must reset_mctx.
    fn qfirst_wave(
        &mut self,
        ctx: *mut sys::llama_context,
        chunk: &[Item],
        hstart: usize,
        rows: &mut [Option<Vec<f32>>],
        stat: &mut WaveStat,
    ) -> Result<()> {
        let mut used: HashSet<i32> = HashSet::new();
        let mut assign: Vec<i32> = Vec::with_capacity(chunk.len());
        let mut installs: Vec<(i32, &[i32], i32)> = Vec::new();
        // (src, dst, end): a duplicated head can only be copied after its
        // source seq is materialized — i.e. after the installs decode below
        let mut copies: Vec<(i32, i32, i32)> = Vec::new();
        for it in chunk {
            let split = it.split.max(hstart);
            let head = &it.toks[..split];
            let mut dup = None;
            match self.qseqs.get(head) {
                // a hit only if no sibling this wave already owns the seq —
                // identical questions would otherwise decode two suffixes
                // onto one seq
                Some(s) if !used.contains(&s) => {
                    // and only if the seq actually holds this head — residue
                    // of an aborted decode is a miss, not a hit
                    if unsafe { llamac::mem_seq_pos_max(ctx, s) } == split as i32 - 1 {
                        stat.hits += 1;
                        used.insert(s);
                        assign.push(s);
                        continue;
                    }
                    self.qseqs.remove(head);
                    unsafe { llamac::mem_rm(ctx, s, -1, -1) };
                }
                other => dup = other.filter(|_| split > hstart),
            }
            if dup.is_some() {
                stat.hits += 1; // head exists; just cloned to a scratch seq
            } else {
                stat.misses += 1;
            }
            let s = self
                .qseqs
                .alloc(&used)
                .context("question heads exceed seq capacity")?;
            unsafe {
                llamac::mem_rm(ctx, s, -1, -1);
                match dup {
                    Some(src) => copies.push((src, s, split as i32)),
                    None => {
                        if hstart > 0 {
                            llamac::mem_cp(ctx, 0, s, 0, hstart as i32);
                        }
                        if split > hstart {
                            installs.push((s, &it.toks[hstart..split], hstart as i32));
                            stat.decoded += split - hstart;
                            self.qseqs.insert(head.to_vec(), s, split - hstart);
                        }
                    }
                }
            }
            used.insert(s);
            assign.push(s);
        }
        unsafe {
            if !installs.is_empty() {
                llamac::batch_decode(ctx, &installs, self.n_batch as usize, self.n_vocab)?;
            }
            for &(src, dst, end) in &copies {
                llamac::mem_cp(ctx, src, dst, 0, end);
            }
            let suffixes: Vec<(i32, &[i32], i32)> = chunk
                .iter()
                .zip(&assign)
                .map(|(it, &s)| {
                    (
                        s,
                        &it.toks[it.split.max(hstart)..],
                        it.split.max(hstart) as i32,
                    )
                })
                .collect();
            if !suffixes.is_empty() {
                for (i, row) in
                    llamac::batch_decode(ctx, &suffixes, self.n_batch as usize, self.n_vocab)?
                        .into_iter()
                        .enumerate()
                {
                    rows[i] = Some(row);
                }
            }
            // trim every seq back to its question head — the cache entry
            for (it, &s) in chunk.iter().zip(&assign) {
                llamac::mem_rm(ctx, s, it.split.max(hstart) as i32, -1);
                stat.decoded += it.toks.len() - it.split.max(hstart);
            }
            // keep the head cache within its share of the pool so later
            // waves still have cells to work with
            self.qseqs
                .trim_to(ctx, (self.n_ctx / QCACHE_CELL_DIV).max(1) as usize, &used);
        }
        Ok(())
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
                    llamac::state_load(self.ctx, &snap);
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
                            &[(0, &it.toks[hstart..split], hstart as i32)],
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
                    &[(0, &it.toks[split..], split as i32)],
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
        let mut expanded: Vec<(String, Question, Vec<String>)> = Vec::new();
        let mut originals: Vec<(String, String)> = Vec::new();
        let mut any_abstain = false;
        for (name, q) in req.questions()? {
            originals.push((name.clone(), q.instructions.clone()));
            any_abstain |= q.allow_abstain;
            match expand_choice(&q) {
                Some((subs, keys)) => {
                    // expanded probes are boolean questions mechanically: the
                    // boolean bucket temperature applies
                    for (i, sq) in subs.into_iter().enumerate() {
                        pending.push((format!("{name}\u{1f}{i}"), sq, calib("boolean")));
                    }
                    expanded.push((name, q, keys));
                }
                None => {
                    let c = calib(crate::calibrate::bucket(q.qtype.as_str()));
                    pending.push((name, q, c));
                }
            }
        }

        let layout = match req.layout {
            Layout::Auto => {
                // abstain questions read better after the evidence: the
                // __abstain__ slot before the state primes abstention
                // (measured: -5..-16pp on eval/edge across the model set)
                let state_len = prompts::render_state(&req.state).len();
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

        let mut items = Vec::new();
        for (name, q, c) in pending {
            let slots = prompts::slots_for(&q);
            let msg = prompts::user_message(&req.state, &q, &slots, layout, &preamble);
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
            let mut answers: Option<Map<String, Value>> = None;
            if self.mctx.is_some() && items.len() > 1 {
                match self.decide_batched(&items, hstart, plen, req.temperature) {
                    Ok((a, tp)) => {
                        answers = Some(a);
                        t_prefill = tp;
                        suffix_mode = "batched";
                    }
                    // wipe poisoned seqs before the sequential fallback
                    Err(_) => unsafe { self.reset_mctx() },
                }
            }
            match answers {
                Some(a) => a,
                None => {
                    self.decide_sequential(&items, hstart, plen, req.temperature, &mut t_prefill)?
                }
            }
        };

        // fold per-option probes back into one answer per expanded choice
        for (name, q, keys) in &expanded {
            let mut p_yes = Vec::with_capacity(keys.len());
            let mut cov = 0.0;
            for i in 0..keys.len() {
                let a = answers
                    .remove(&format!("{name}\u{1f}{i}"))
                    .unwrap_or_default();
                p_yes.push(a["probabilities"]["yes"].as_f64().unwrap_or(0.0));
                cov += a["coverage"].as_f64().unwrap_or(0.0);
            }
            let mut merged = decisions::merge_scored(q, keys, &p_yes);
            merged["coverage"] = json!(r4(cov / keys.len().max(1) as f64));
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
        *t_prefill = Instant::now();

        let mut out = Map::new();
        for it in items {
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

fn r4(v: f64) -> f64 {
    (v * 1e4).round() / 1e4
}

fn layout_str(l: Layout) -> &'static str {
    match l {
        Layout::Auto => "auto",
        Layout::StateFirst => "state_first",
        Layout::QuestionFirst => "question_first",
        Layout::Header => "header",
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
        let used = HashSet::new();
        // free seqs come first (lowest available)
        assert_eq!(q.alloc(&used), Some(1));
        q.insert(vec![1], 1, 10);
        q.insert(vec![2], 2, 10);
        // seqs 1,2 tracked -> next free is 3
        assert_eq!(q.alloc(&used), Some(3));
        // every seq tracked/used -> evict the least recent head's seq
        let mut all_used: HashSet<i32> = (1..super::llamac::MAX_SEQS).collect();
        for s in 3..super::llamac::MAX_SEQS {
            q.insert(vec![100 + s], s, 1);
        }
        assert_eq!(q.cells, 10 + 10 + 62);
        all_used.remove(&2); // seq 2's head is evictable
        assert_eq!(q.alloc(&all_used), Some(2));
        assert_eq!(q.cells, 10 + 62); // head {2}'s 10 cells released
    }
}
