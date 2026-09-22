//! Thin unsafe wrappers over llama-cpp-sys-2 for exactly what snap needs:
//! model load, tokenize, batched decode, per-sequence memory ops, state
//! snapshots. The safe llama-cpp-2 crate keeps the raw pointers private, so
//! we drive the C API directly (mirrors the Python ctypes approach).

use std::ffi::{c_char, c_void, CStr, CString};
use std::sync::Mutex;

use anyhow::{bail, Result};
use llama_cpp_sys_2 as sys;

pub const MAX_SEQS: i32 = 65; // seq 0 owns the shared prefix; questions run on seqs 1..64

pub struct Model(pub *mut sys::llama_model);
pub struct Ctx(pub *mut sys::llama_context);

unsafe impl Send for Model {}
unsafe impl Send for Ctx {}
unsafe impl Sync for Model {}
unsafe impl Sync for Ctx {}

pub fn backend_init() {
    unsafe { sys::llama_backend_init() }
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

pub fn load_model(path: &str) -> Result<Model> {
    let cpath = CString::new(path)?;
    let mut params = unsafe { sys::llama_model_default_params() };
    params.n_gpu_layers = -1; // everything on GPU (Metal); llama.cpp falls back to CPU if it can't
    let m = unsafe { sys::llama_model_load_from_file(cpath.as_ptr(), params) };
    if m.is_null() {
        bail!("failed to load model: {path}");
    }
    Ok(Model(m))
}

pub fn new_ctx(
    model: &Model,
    n_ctx: i32,
    n_batch: i32,
    n_ubatch: i32,
    n_seq_max: i32,
    kv_unified: bool,
) -> Result<Ctx> {
    let mut p = unsafe { sys::llama_context_default_params() };
    p.n_ctx = n_ctx as u32;
    p.n_batch = n_batch as u32;
    p.n_ubatch = n_ubatch as u32;
    p.n_seq_max = n_seq_max as u32;
    p.n_outputs_max = n_seq_max as u32; // default would reserve n_batch vocab rows
    p.flash_attn_type = sys::LLAMA_FLASH_ATTN_TYPE_ENABLED;
    p.swa_full = true;
    p.kv_unified = kv_unified;
    p.no_perf = true;
    let c = unsafe { sys::llama_init_from_model(model.0, p) };
    if c.is_null() {
        bail!("llama_init_from_model returned null");
    }
    Ok(Ctx(c))
}

pub fn free_ctx(ctx: *mut sys::llama_context) {
    if !ctx.is_null() {
        unsafe { sys::llama_free(ctx) }
    }
}

pub fn free_model(model: *mut sys::llama_model) {
    if !model.is_null() {
        unsafe { sys::llama_model_free(model) }
    }
}

pub fn vocab(model: &Model) -> *const sys::llama_vocab {
    unsafe { sys::llama_model_get_vocab(model.0) }
}

pub fn n_vocab(model: &Model) -> i32 {
    unsafe { sys::llama_vocab_n_tokens(vocab(model)) }
}

/// Tokenize `text` with the model's own tokenizer (BPE). add_special adds BOS.
pub fn tokenize(model: &Model, text: &str, add_special: bool) -> Result<Vec<i32>> {
    let v = vocab(model);
    let bytes = text.as_bytes();
    let mut cap = (bytes.len() as i32 / 2) + 64;
    loop {
        let mut buf = vec![0i32; cap as usize];
        let n = unsafe {
            sys::llama_tokenize(
                v,
                bytes.as_ptr() as *const c_char,
                bytes.len() as i32,
                buf.as_mut_ptr(),
                cap,
                add_special,
                true, // parse_special: special tokens in the template matter
            )
        };
        if n >= 0 {
            buf.truncate(n as usize);
            return Ok(buf);
        }
        cap = -n + 16;
    }
}

/// GGUF metadata string value; grows the buffer until it fits.
pub fn meta_val(model: &Model, key: &str) -> Option<String> {
    let ckey = CString::new(key).ok()?;
    let mut size = 256i32;
    loop {
        let mut buf = vec![0i8; size as usize];
        let n = unsafe {
            sys::llama_model_meta_val_str(model.0, ckey.as_ptr(), buf.as_mut_ptr(), size as usize)
        };
        if n < 0 {
            return None;
        }
        if (n as usize) < size as usize {
            let bytes =
                unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const u8, n as usize) };
            return Some(String::from_utf8_lossy(bytes).into_owned());
        }
        size = n + 1;
    }
}

/// Decode (seq_id, tokens, pos0) groups, chunked to `n_batch` tokens per
/// `llama_decode`. Logits are requested on the last token of each group and
/// returned as one row per group (in order).
///
/// # Safety
/// `ctx` must be a live context whose `n_seq_max` covers every seq used here.
/// Positions must be consecutive per sequence w.r.t. what the KV already holds.
pub unsafe fn batch_decode(
    ctx: *mut sys::llama_context,
    seqs: &[(i32, &[i32], i32)],
    n_batch: usize,
    n_vocab: usize,
) -> Result<Vec<Vec<f32>>> {
    // flatten: (seq, token, pos, last-token-of-its-group, group index)
    struct Ent {
        seq: i32,
        tok: i32,
        pos: i32,
        last: bool,
        grp: usize,
    }
    let mut flat: Vec<Ent> = Vec::new();
    for (gi, (seq, toks, pos0)) in seqs.iter().enumerate() {
        for (j, &t) in toks.iter().enumerate() {
            flat.push(Ent {
                seq: *seq,
                tok: t,
                pos: pos0 + j as i32,
                last: j + 1 == toks.len(),
                grp: gi,
            });
        }
    }
    let mut rows: Vec<Option<Vec<f32>>> = vec![None; seqs.len()];
    for chunk in flat.chunks(n_batch.max(1)) {
        let mut batch = sys::llama_batch_init(chunk.len() as i32, 0, MAX_SEQS);
        if batch.token.is_null() {
            bail!("llama_batch_init failed");
        }
        for (i, e) in chunk.iter().enumerate() {
            *batch.token.add(i) = e.tok;
            *batch.pos.add(i) = e.pos;
            *batch.n_seq_id.add(i) = 1;
            *(*batch.seq_id.add(i)).add(0) = e.seq;
            *batch.logits.add(i) = e.last as i8;
        }
        batch.n_tokens = chunk.len() as i32;
        let rc = sys::llama_decode(ctx, batch);
        sys::llama_batch_free(batch);
        if rc != 0 {
            bail!("llama_decode returned {rc}");
        }
        // logits rows are valid until the next decode: read them now
        for (i, e) in chunk.iter().enumerate() {
            if e.last {
                let p = sys::llama_get_logits_ith(ctx, i as i32);
                rows[e.grp] = Some(std::slice::from_raw_parts(p, n_vocab).to_vec());
            }
        }
    }
    rows.into_iter()
        .map(|r| r.ok_or_else(|| anyhow::anyhow!("group produced no logits")))
        .collect()
}

pub unsafe fn mem_rm(ctx: *mut sys::llama_context, seq: i32, p0: i32, p1: i32) {
    let mem = sys::llama_get_memory(ctx);
    sys::llama_memory_seq_rm(mem, seq, p0, p1);
}

pub unsafe fn mem_cp(ctx: *mut sys::llama_context, src: i32, dst: i32, p0: i32, p1: i32) {
    let mem = sys::llama_get_memory(ctx);
    sys::llama_memory_seq_cp(mem, src, dst, p0, p1);
}

pub unsafe fn mem_clear(ctx: *mut sys::llama_context) {
    let mem = sys::llama_get_memory(ctx);
    sys::llama_memory_clear(mem, true);
}

/// Whole-context state snapshot (KV + recurrent state + rng). Used on hybrid
/// archs where arbitrary KV rewind is unsupported.
pub unsafe fn state_save(ctx: *mut sys::llama_context) -> Vec<u8> {
    let n = sys::llama_state_get_size(ctx);
    let mut buf = vec![0u8; n];
    let w = sys::llama_state_get_data(ctx, buf.as_mut_ptr(), n);
    buf.truncate(w);
    buf
}

pub unsafe fn state_load(ctx: *mut sys::llama_context, data: &[u8]) {
    sys::llama_state_set_data(ctx, data.as_ptr(), data.len());
}
