//! Thin unsafe wrappers over llama-cpp-sys-2 for exactly what snap needs:
//! model load, tokenize, chat templates, batched multi-seq decode, whole-seq
//! memory ops. `Llama` is the production `kv::Backend`; everything above it
//! runs unchanged against the simulated backend in `kv::sim`, which is how
//! the KV orchestration is tested without a model. The safe llama-cpp-2
//! crate keeps the raw pointers private, so we drive the C API directly.

use std::ffi::{c_char, c_void, CStr, CString};
use std::sync::Mutex;

use anyhow::{bail, Result};
use llama_cpp_sys_2 as sys;

use crate::kv::{Backend, Dec, DecodeError};

/// Seq capacity: seq 0 keeps the resident prompt head, the rest hold cache
/// entries and per-item work seqs.
pub const MAX_SEQS: i32 = 65;
/// Seq capacity on hybrid/recurrent archs: every seq pins a full
/// recurrent-state row (~50 MiB on qwen35), so 65 would be ~3.3 GB.
pub const HYBRID_SEQS: i32 = 17;

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

// --- llama.cpp/ggml logging -------------------------------------------------
// The C API writes to stderr unless a callback is installed. Route everything
// through tracing under the `llamac` target so `--debug` / RUST_LOG decide.
// Non-CONT messages arrive newline-terminated; CONT is a continuation of the
// previous line — buffer both until a '\n' completes a line.
static LOG_PENDING: Mutex<(sys::ggml_log_level, String)> =
    Mutex::new((sys::GGML_LOG_LEVEL_INFO, String::new()));

fn emit_log(level: sys::ggml_log_level, line: &str) {
    match level {
        sys::GGML_LOG_LEVEL_ERROR => tracing::error!(target: "llamac", "{line}"),
        sys::GGML_LOG_LEVEL_WARN => tracing::warn!(target: "llamac", "{line}"),
        sys::GGML_LOG_LEVEL_INFO => tracing::info!(target: "llamac", "{line}"),
        _ => tracing::debug!(target: "llamac", "{line}"),
    }
}

unsafe extern "C" fn log_to_tracing(
    level: sys::ggml_log_level,
    text: *const c_char,
    _user_data: *mut c_void,
) {
    if text.is_null() {
        return;
    }
    let text = unsafe { CStr::from_ptr(text) }.to_string_lossy();
    let Ok(mut pending) = LOG_PENDING.lock() else {
        return;
    };
    if level != sys::GGML_LOG_LEVEL_CONT {
        pending.0 = level;
    }
    pending.1.push_str(&text);
    while let Some(nl) = pending.1.find('\n') {
        let line: String = pending.1.drain(..=nl).collect();
        emit_log(pending.0, line.trim_end());
    }
}

pub fn route_logs_to_tracing() {
    unsafe {
        sys::llama_log_set(Some(log_to_tracing), std::ptr::null_mut());
        sys::ggml_log_set(Some(log_to_tracing), std::ptr::null_mut());
    }
}

/// A loaded model plus its one multi-seq context over a unified KV cache —
/// unified because shared-prefix fan-out tags one token with many seqs.
pub struct Llama {
    model: *mut sys::llama_model,
    ctx: *mut sys::llama_context,
    tmpls: *mut c_void,
    n_vocab: usize,
    n_ctx: usize,
    n_seq: usize,
    n_batch: usize,
}

// Only ever used behind the engine's Mutex; llama.cpp objects aren't
// internally thread-safe but exclusive access makes sending them sound.
unsafe impl Send for Llama {}

impl Llama {
    pub fn load(path: &str, n_ctx: i32, n_batch: i32, n_threads: i32) -> Result<Llama> {
        unsafe { sys::llama_backend_init() };
        let cpath = CString::new(path)?;
        let mut mp = unsafe { sys::llama_model_default_params() };
        mp.n_gpu_layers = -1; // everything on GPU; llama.cpp falls back to CPU if it can't
        let model = unsafe { sys::llama_model_load_from_file(cpath.as_ptr(), mp) };
        if model.is_null() {
            bail!("failed to load model: {path}");
        }
        // owns `model` from here: Drop frees whatever got allocated
        let mut me = Llama {
            model,
            ctx: std::ptr::null_mut(),
            tmpls: std::ptr::null_mut(),
            n_vocab: unsafe { sys::llama_vocab_n_tokens(sys::llama_model_get_vocab(model)) }
                as usize,
            n_ctx: n_ctx as usize,
            n_seq: 0,
            n_batch: n_batch.max(1) as usize,
        };
        if me.meta("tokenizer.chat_template").is_none() {
            bail!("model has no tokenizer.chat_template in GGUF metadata");
        }
        me.tmpls = unsafe {
            snap_chat_templates_init(model, std::ptr::null(), std::ptr::null(), std::ptr::null())
        };
        if me.tmpls.is_null() {
            bail!("common_chat_templates_init failed (unsupported chat template?)");
        }
        // 0 = auto: llama.cpp's own default is 4 threads, which starves
        // prefill on CPU-only builds (most release targets).
        let n_threads = if n_threads > 0 {
            n_threads
        } else {
            std::thread::available_parallelism().map_or(4, |n| n.get() as i32)
        };
        let hybrid =
            unsafe { sys::llama_model_is_hybrid(model) || sys::llama_model_is_recurrent(model) };
        // fewer seqs is slower, never wrong — degrade instead of failing
        let mut nseq = if hybrid { HYBRID_SEQS } else { MAX_SEQS };
        loop {
            let mut p = unsafe { sys::llama_context_default_params() };
            p.n_ctx = n_ctx as u32;
            p.n_batch = n_batch as u32;
            p.n_ubatch = n_batch as u32;
            p.n_seq_max = nseq as u32;
            p.n_outputs_max = nseq as u32; // default would reserve n_batch vocab rows
            p.n_threads = n_threads;
            p.n_threads_batch = n_threads;
            p.flash_attn_type = sys::LLAMA_FLASH_ATTN_TYPE_ENABLED;
            p.swa_full = true; // SWA layers must keep whole prefixes to be forkable
            p.kv_unified = true;
            p.no_perf = true;
            me.ctx = unsafe { sys::llama_init_from_model(model, p) };
            if !me.ctx.is_null() {
                me.n_seq = nseq as usize;
                return Ok(me);
            }
            if nseq <= 2 {
                bail!("llama_init_from_model returned null");
            }
            nseq = nseq / 2 + 1;
            eprintln!("snap: context init failed, retrying with {nseq} seqs");
        }
    }

    /// GGUF metadata string value; grows the buffer until it fits.
    pub fn meta(&self, key: &str) -> Option<String> {
        let ckey = CString::new(key).ok()?;
        let mut size = 256i32;
        loop {
            let mut buf = vec![0 as c_char; size as usize];
            let n = unsafe {
                sys::llama_model_meta_val_str(
                    self.model,
                    ckey.as_ptr(),
                    buf.as_mut_ptr(),
                    size as usize,
                )
            };
            if n < 0 {
                return None;
            }
            if n < size {
                let bytes =
                    unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const u8, n as usize) };
                return Some(String::from_utf8_lossy(bytes).into_owned());
            }
            size = n + 1;
        }
    }

    fn memory(&self) -> sys::llama_memory_t {
        unsafe { sys::llama_get_memory(self.ctx) }
    }
}

impl Drop for Llama {
    fn drop(&mut self) {
        unsafe {
            if !self.ctx.is_null() {
                sys::llama_free(self.ctx);
            }
            if !self.tmpls.is_null() {
                snap_chat_templates_free(self.tmpls);
            }
            sys::llama_model_free(self.model);
        }
    }
}

impl Backend for Llama {
    fn n_ctx(&self) -> usize {
        self.n_ctx
    }

    fn n_seq(&self) -> usize {
        self.n_seq
    }

    fn render(&self, system: &str, user: &str) -> Result<String> {
        let (system, user) = (CString::new(system)?, CString::new(user)?);
        let p = unsafe { snap_chat_apply(self.tmpls, system.as_ptr(), user.as_ptr()) };
        if p.is_null() {
            bail!("chat template apply failed");
        }
        let s = unsafe { CStr::from_ptr(p).to_string_lossy().into_owned() };
        unsafe { snap_str_free(p) };
        Ok(s)
    }

    /// The model's own tokenizer; no BOS added (templates carry their own).
    fn tokenize(&self, text: &str, special: bool) -> Result<Vec<i32>> {
        let vocab = unsafe { sys::llama_model_get_vocab(self.model) };
        let bytes = text.as_bytes();
        let mut cap = (bytes.len() as i32 / 2) + 64;
        loop {
            let mut buf = vec![0i32; cap as usize];
            let n = unsafe {
                sys::llama_tokenize(
                    vocab,
                    bytes.as_ptr() as *const c_char,
                    bytes.len() as i32,
                    buf.as_mut_ptr(),
                    cap,
                    false,
                    special,
                )
            };
            if n >= 0 {
                buf.truncate(n as usize);
                return Ok(buf);
            }
            cap = -n + 16;
        }
    }

    /// Flatten groups, chunk to `n_batch` tokens per `llama_decode`, and hand
    /// each requested logits row to `row` while it is still valid (the next
    /// decode overwrites it) — no per-row copies.
    fn decode(
        &mut self,
        groups: &[Dec],
        row: &mut dyn FnMut(usize, &[f32]),
    ) -> Result<(), DecodeError> {
        // (group, token index within the group)
        let flat: Vec<(usize, usize)> = groups
            .iter()
            .enumerate()
            .flat_map(|(g, d)| (0..d.toks.len()).map(move |i| (g, i)))
            .collect();
        for chunk in flat.chunks(self.n_batch) {
            unsafe {
                let mut batch = sys::llama_batch_init(chunk.len() as i32, 0, MAX_SEQS);
                if batch.token.is_null() {
                    return Err(DecodeError::Failed("llama_batch_init failed".into()));
                }
                for (i, &(g, t)) in chunk.iter().enumerate() {
                    let d = &groups[g];
                    *batch.token.add(i) = d.toks[t];
                    *batch.pos.add(i) = (d.pos0 + t) as i32;
                    *batch.n_seq_id.add(i) = d.seqs.len() as i32;
                    for (k, &s) in d.seqs.iter().enumerate() {
                        *(*batch.seq_id.add(i)).add(k) = s;
                    }
                    *batch.logits.add(i) = (d.logits && t + 1 == d.toks.len()) as i8;
                }
                batch.n_tokens = chunk.len() as i32;
                let rc = sys::llama_decode(self.ctx, batch);
                sys::llama_batch_free(batch);
                match rc {
                    0 => {}
                    1 => return Err(DecodeError::NoSlot),
                    rc => return Err(DecodeError::Failed(format!("llama_decode returned {rc}"))),
                }
                for (i, &(g, t)) in chunk.iter().enumerate() {
                    if groups[g].logits && t + 1 == groups[g].toks.len() {
                        let p = sys::llama_get_logits_ith(self.ctx, i as i32);
                        row(g, std::slice::from_raw_parts(p, self.n_vocab));
                    }
                }
            }
        }
        Ok(())
    }

    fn seq_rm(&mut self, seq: i32) {
        unsafe { sys::llama_memory_seq_rm(self.memory(), seq, -1, -1) };
    }

    /// Whole-seq copy: on unified KV it only tags cells, and on recurrent
    /// memory it shares the tail state copy-on-write (which is also why
    /// `src` must hold exactly `len` tokens — recurrent seq_cp ignores ranges).
    fn seq_cp(&mut self, src: i32, dst: i32, len: usize) {
        unsafe { sys::llama_memory_seq_cp(self.memory(), src, dst, 0, len as i32) };
    }

    fn clear(&mut self) {
        unsafe { sys::llama_memory_clear(self.memory(), true) };
    }
}
