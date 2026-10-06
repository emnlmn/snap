//! `snap grep`: find the code that answers a question, in two stages. Recall
//! ranks every chunk of the tree lexically — no model, milliseconds — and the
//! model then reads the best candidates once each, answering one yes/no
//! letter whose probability is the score. The expensive half of that read
//! does not depend on the query, so it is kept: the first time a chunk is a
//! candidate its `[head + chunk]` memory is snapshotted into a store on disk,
//! and every later query restores it and decodes only its own question.
//!
//! The store is bound to everything that shapes that memory (snap version,
//! model file, prompt format, KV type) and lives in one directory per tree:
//! another binding opens another directory, never foreign memory. A snapshot
//! is trusted only when it holds exactly the tokens the probe expects, and one
//! that llama.cpp refuses is scored from scratch instead.
//!
//! This module is the pipeline and its reports (`--eval`, `snap index`); the
//! prompt and the scoring live in `engine.rs`, the walk in `corpus.rs`.

use std::collections::{BTreeMap, HashSet};
use std::io::{IsTerminal, Write};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::time::{Instant, UNIX_EPOCH};

use anyhow::{bail, ensure, Context, Result};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::corpus::{self, Chunk, WalkOpts};
use crate::engine::{grep_state, Engine, Probe, GREP_FORMAT};
use crate::evaluate;
use crate::kvstore::{self, Prefetch, Store};
use crate::lexical::{rrf, Doc, Index};
use crate::llamac::KvType;
use crate::prompts::PROMPT_VERSION;

/// Reciprocal rank fusion's damping: how fast a rank's vote falls off.
const RRF_K: f64 = 60.0;
/// Seconds between syncs of a long `snap index`: a kill loses at most that much.
const SYNC_EVERY: f64 = 10.0;
/// The ks of the recall@k an eval reports.
const KS: [usize; 4] = [1, 5, 10, 64];

/// A tree to search and the KV type its snapshots are held in.
pub struct Tree {
    pub root: PathBuf,
    pub walk: WalkOpts,
    pub kv: KvType,
}

/// What `snap grep` and its `--eval` share: where to look, how deep to
/// rerank, and whether snapshots are kept.
pub struct Opts {
    pub tree: Tree,
    /// regex narrowing the corpus before recall
    pub regexp: Option<String>,
    /// hits printed
    pub top: usize,
    /// recall depth reranked; 0 = every chunk
    pub candidates: usize,
    /// least P(yes) of a hit
    pub threshold: f64,
    pub store: bool,
    pub recall_only: bool,
}

impl Opts {
    fn check(&self) -> Result<()> {
        ensure!(self.top >= 1, "--top must be at least 1");
        ensure!(
            (0.0..=1.0).contains(&self.threshold),
            "--threshold is a probability: 0 to 1"
        );
        Ok(())
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Out {
    Human,
    Json,
    Files,
}

/// A search: the query, and how to print what it finds.
pub struct Search {
    pub query: String,
    pub out: Out,
    /// source lines printed per hit; 0 = locations only
    pub lines: usize,
    pub opts: Opts,
}

/// The model, loaded on demand (recall needs none), and the GGUF it came
/// from: its size and mtime bind the store.
pub type Loaded = (Engine, PathBuf);

// --- corpus and recall ------------------------------------------------------

/// The chunks of the tree after the `-e` filter, minus `skip` (root-relative
/// paths). A file named as the root is searched whatever the walk flags say,
/// like ripgrep's explicit arguments.
fn load_chunks(t: &Tree, regexp: Option<&str>, skip: &[String]) -> Result<Vec<Chunk>> {
    let mut chunks = if t.root.is_file() {
        let name = t.root.file_name().unwrap_or_default().to_string_lossy();
        let text = corpus::read_text(&t.root);
        text.map(|s| corpus::split(&name, &s)).unwrap_or_default()
    } else {
        corpus::corpus(&t.root, &t.walk)?
    };
    chunks.retain(|c| !skip.contains(&c.path));
    if let Some(re) = regexp {
        let mut keep = corpus::grep(&chunks, re)?.into_iter();
        chunks.retain(|_| keep.next() == Some(true));
    }
    Ok(chunks)
}

/// `path` as a chunk of `root` names its file: root-relative, `/`-separated.
/// None when it lies elsewhere or does not exist.
fn rel_to(root: &Path, path: &str) -> Option<String> {
    let root = root.canonicalize().ok()?;
    let path = Path::new(path).canonicalize().ok()?;
    let rel = path.strip_prefix(root).ok()?;
    let parts: Vec<_> = rel
        .components()
        .map(|c| c.as_os_str().to_string_lossy())
        .collect();
    Some(parts.join("/"))
}

/// The corpus and the two lexical views recall fuses: one document per chunk,
/// and one per file.
struct Haystack {
    chunks: Vec<Chunk>,
    by_chunk: Index,
    by_file: Index,
    /// the chunks of each document of `by_file`, adjacent as the walk emits them
    files: Vec<Range<usize>>,
}

impl Haystack {
    fn new(chunks: Vec<Chunk>) -> Haystack {
        let by_chunk = Index::build(chunks.iter().map(|c| Doc {
            path: &c.path,
            symbol: c.symbol.as_deref(),
            body: &c.text,
        }));
        let mut files = Vec::new();
        for run in chunks.chunk_by(|a, b| a.path == b.path) {
            let at = files.last().map_or(0, |r: &Range<usize>| r.end);
            files.push(at..at + run.len());
        }
        // a file is its symbols and its text: a query naming what the file is
        // about reaches the chunks that never repeat the word
        let docs: Vec<(String, String)> = files
            .iter()
            .map(|r| {
                let cs = &chunks[r.clone()];
                let symbols: Vec<&str> = cs.iter().filter_map(|c| c.symbol.as_deref()).collect();
                let texts: Vec<&str> = cs.iter().map(|c| c.text.as_str()).collect();
                (symbols.join(" "), texts.join("\n"))
            })
            .collect();
        let by_file = Index::build(files.iter().zip(&docs).map(|(r, (symbols, text))| Doc {
            path: &chunks[r.start].path,
            symbol: Some(symbols),
            body: text,
        }));
        Haystack {
            chunks,
            by_chunk,
            by_file,
            files,
        }
    }

    /// Chunk indices in recall order: the chunk ranking fused by RRF with the
    /// file ranking expanded into its files' chunks (best chunk first, then
    /// line order), cut at `candidates`. With 0 every chunk is a candidate:
    /// those no ranking holds follow in corpus order.
    fn recall(&self, query: &str, candidates: usize) -> Vec<usize> {
        let n = self.chunks.len();
        let direct = self.by_chunk.search(query, self.by_chunk.len());
        let mut score = vec![0f32; n];
        for &(i, s) in &direct {
            score[i] = s;
        }
        let expanded: Vec<usize> = self
            .by_file
            .search(query, self.by_file.len())
            .into_iter()
            .flat_map(|(f, _)| {
                let mut ids: Vec<usize> = self.files[f].clone().collect();
                ids.sort_by(|&a, &b| score[b].total_cmp(&score[a]));
                ids
            })
            .collect();
        let direct = direct.into_iter().map(|(i, _)| i).collect();
        let mut order: Vec<usize> = rrf(&[direct, expanded], RRF_K)
            .into_iter()
            .map(|(i, _)| i)
            .collect();
        if candidates == 0 {
            let held: HashSet<usize> = order.iter().copied().collect();
            order.extend((0..n).filter(|i| !held.contains(i)));
        } else {
            order.truncate(candidates);
        }
        order
    }
}

// --- the store --------------------------------------------------------------

/// Where snapshots live: `$SNAP_GREP_DIR`, else `$XDG_CACHE_HOME/snap/grep`,
/// else `~/.cache/snap/grep`, else `%LOCALAPPDATA%\snap\grep`, else the temp
/// dir. An empty variable counts as unset.
fn store_base(env: &dyn Fn(&str) -> Option<String>, temp: &Path) -> PathBuf {
    let var = |k: &str| env(k).filter(|v| !v.is_empty()).map(PathBuf::from);
    if let Some(d) = var("SNAP_GREP_DIR") {
        return d;
    }
    if let Some(d) = var("XDG_CACHE_HOME") {
        return d.join("snap").join("grep");
    }
    if let Some(h) = var("HOME") {
        return h.join(".cache").join("snap").join("grep");
    }
    if let Some(d) = var("LOCALAPPDATA") {
        return d.join("snap").join("grep");
    }
    temp.join("snap-grep")
}

/// Everything besides the chunk text that shapes a snapshot, as the lines the
/// store records and its directory name hashes: a mismatch is another store.
fn binding(model: &str, gguf: (u64, u64), kv: KvType) -> String {
    format!(
        "snap {}\nmodel {model}\ngguf {} bytes, mtime {}\nprompt v{PROMPT_VERSION}\ngrep format {GREP_FORMAT}\nkv {}\n",
        env!("CARGO_PKG_VERSION"),
        gguf.0,
        gguf.1,
        kv.as_str()
    )
}

/// The GGUF's size in bytes and mtime in seconds: the model, as far as the
/// store can tell without hashing gigabytes.
fn gguf_identity(path: &Path) -> Result<(u64, u64)> {
    let m = std::fs::metadata(path).with_context(|| format!("cannot stat {}", path.display()))?;
    let mtime = m
        .modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok());
    Ok((m.len(), mtime.map_or(0, |d| d.as_secs())))
}

/// 16 hex digits of a key: names a directory.
fn hex16(k: u128) -> String {
    format!("{:016x}", (k >> 64) as u64)
}

/// `<base>/<tree>/<binding>`: one directory per canonical root and binding.
fn store_dir(base: &Path, root: &Path, binding: &str) -> PathBuf {
    let tree = kvstore::key(root.to_string_lossy().as_bytes());
    let bound = kvstore::key(binding.as_bytes());
    base.join(hex16(tree)).join(hex16(bound))
}

/// The store location the environment names, resolved once per command.
fn default_base() -> PathBuf {
    store_base(&|k| std::env::var(k).ok(), &std::env::temp_dir())
}

/// Where this engine keeps the snapshots of the tree, and the binding the
/// directory is opened under.
fn store_spec(base: &Path, t: &Tree, gguf: &Path, model_id: &str) -> Result<(PathBuf, String)> {
    let root = t.root.canonicalize()?;
    let binding = binding(model_id, gguf_identity(gguf)?, t.kv);
    Ok((store_dir(base, &root, &binding), binding))
}

fn open_store(base: &Path, t: &Tree, gguf: &Path, model_id: &str) -> Result<Store> {
    let (dir, binding) = store_spec(base, t, gguf, model_id)?;
    tracing::debug!("snapshot store {}", dir.display());
    Store::open(&dir, &binding)
}

/// The store a search may use: one that will not open (another process holds
/// it, the disk refuses) costs speed, not answers.
fn try_store(base: &Path, o: &Opts, gguf: &Path, model_id: &str) -> Option<Store> {
    if !o.store {
        return None;
    }
    open_store(base, &o.tree, gguf, model_id)
        .map_err(|e| eprintln!("snap: no snapshot store ({e:#}); scoring from scratch"))
        .ok()
}

/// The content address of a probe's prefix.
fn prefix_key(toks: &[i32]) -> u128 {
    kvstore::key(
        &toks
            .iter()
            .flat_map(|t| t.to_le_bytes())
            .collect::<Vec<u8>>(),
    )
}

// --- rerank -----------------------------------------------------------------

/// What reranking took: candidates scored, how many of those were restored
/// from a stored snapshot (the rest had their chunk decoded in this run), and
/// the tokens the model decoded.
#[derive(Default, Clone, Copy)]
struct Cost {
    scored: usize,
    restored: usize,
    tokens: usize,
}

/// Stop after a wave when enough hits are in and the wave added none: what
/// recall ranked lower has not been reading better.
fn done(total_hits: usize, wave_hits: usize, top: usize) -> bool {
    total_hits >= top && wave_hits == 0
}

/// P(yes) of the candidates `order` names, in waves of one per parallel
/// sequence, in recall order, until the cascade stops. The result holds one P
/// per candidate scored: a prefix of `order`.
fn rerank(
    eng: &mut Engine,
    store: &mut Option<Store>,
    chunks: &[Chunk],
    order: &[usize],
    query: &str,
    top: usize,
    threshold: f64,
) -> Result<(Vec<f64>, Cost)> {
    let probes = order
        .iter()
        .map(|&i| probe_for(eng, &chunks[i], query))
        .collect::<Result<Vec<Probe>>>()?;
    let keys: Vec<u128> = probes.iter().map(|p| prefix_key(&p.toks[..p.at])).collect();
    let wave = eng.n_seq().saturating_sub(1).max(1);
    // every candidate's snapshot is asked for once: the reads of a wave
    // overlap the decode of the one before
    let mut fetch = store.as_ref().map(|s| s.prefetch(keys.clone(), wave));
    let mut cost = Cost::default();
    let (mut ps, mut hits) = (Vec::with_capacity(probes.len()), 0);
    for (probes, keys) in probes.chunks(wave).zip(keys.chunks(wave)) {
        let got = score_wave(eng, store, &mut fetch, probes, keys, &mut cost)?;
        let added = got.iter().filter(|&&p| p >= threshold).count();
        hits += added;
        ps.extend(got);
        if done(hits, added, top) {
            break;
        }
    }
    if let Some(s) = store {
        if let Err(e) = s.sync() {
            eprintln!("snap: cannot sync the snapshot store ({e:#})");
        }
    }
    cost.scored = ps.len();
    Ok((ps, cost))
}

fn probe_for(eng: &Engine, c: &Chunk, query: &str) -> Result<Probe> {
    let prose = corpus::prose_ext(&c.path).is_some();
    eng.grep_probe(&grep_state(&c.path, &c.text), query, prose)
}

/// One wave: stored snapshots where the store holds them, a fresh snapshot
/// (kept, and used from memory) where it does not, a cold decode for whatever
/// llama.cpp refuses to restore. Without a store there is nothing to keep, so
/// the copy out of the KV memory and back in is skipped.
fn score_wave(
    eng: &mut Engine,
    store: &mut Option<Store>,
    fetch: &mut Option<Prefetch>,
    probes: &[Probe],
    keys: &[u128],
    cost: &mut Cost,
) -> Result<Vec<f64>> {
    let Some(st) = store.as_mut() else {
        let refs: Vec<&Probe> = probes.iter().collect();
        let (ps, s) = eng.grep_score_cold(&refs)?;
        cost.tokens += s.decoded;
        return Ok(ps);
    };
    // a stored blob counts only if it holds as many tokens as the probe has
    // prefix: llama.cpp does not check positions, and a short one would be
    // accepted on a recurrent memory and answer wrongly
    let mut blobs: Vec<Option<Vec<u8>>> = Vec::with_capacity(probes.len());
    for (p, k) in probes.iter().zip(keys) {
        let sound = st.tokens(*k) == Some(p.at);
        blobs.push(match fetch.as_mut().and_then(|f| f.next()) {
            Some((_, Ok(Some(b)))) if sound => Some(b),
            // stored after the reads were asked for: an earlier wave of this
            // run snapshotted the same text
            _ if sound => st.read(*k).ok().flatten(),
            _ => None,
        });
    }
    let stored: Vec<bool> = blobs.iter().map(Option::is_some).collect();
    let missing: Vec<usize> = (0..probes.len()).filter(|&j| !stored[j]).collect();
    let mut write_failed = false;
    if !missing.is_empty() {
        let prefixes: Vec<&[i32]> = missing
            .iter()
            .map(|&j| &probes[j].toks[..probes[j].at])
            .collect();
        let mut fresh = vec![None; missing.len()];
        let s = eng.grep_snapshot(&prefixes, &mut |i, b| fresh[i] = Some(b))?;
        cost.tokens += s.decoded;
        for (&j, b) in missing.iter().zip(fresh) {
            // `put` keeps the first record of a key: one stored under another
            // length stays as it is, and this blob is used from memory only
            if let (Some(b), None, false) = (&b, st.tokens(keys[j]), write_failed) {
                if let Err(e) = st.put(keys[j], probes[j].at, b) {
                    eprintln!("snap: cannot write the snapshot store ({e:#}); going on without it");
                    write_failed = true;
                }
            }
            blobs[j] = b;
        }
    }
    let have: Vec<usize> = (0..probes.len()).filter(|&j| blobs[j].is_some()).collect();
    let pairs: Vec<(&Probe, &[u8])> = have
        .iter()
        .map(|&j| (&probes[j], blobs[j].as_deref().unwrap_or_default()))
        .collect();
    let (got, s) = eng.grep_score(&pairs)?;
    cost.tokens += s.decoded;
    let mut ps = vec![0.0; probes.len()];
    let mut cold: Vec<usize> = (0..probes.len()).filter(|j| !have.contains(j)).collect();
    for (&j, p) in have.iter().zip(got) {
        match p {
            Some(p) => {
                ps[j] = p;
                cost.restored += stored[j] as usize;
            }
            None => cold.push(j),
        }
    }
    if !cold.is_empty() {
        let refs: Vec<&Probe> = cold.iter().map(|&j| &probes[j]).collect();
        let (got, s) = eng.grep_score_cold(&refs)?;
        cost.tokens += s.decoded;
        for (&j, p) in cold.iter().zip(got) {
            ps[j] = p;
        }
    }
    if write_failed {
        *store = None;
        *fetch = None;
    }
    Ok(ps)
}

/// The hits: candidates at or above the threshold, best P first (recall order
/// among equals), at most `top`. `ps` covers a prefix of `order`.
fn pick(order: &[usize], ps: &[f64], top: usize, threshold: f64) -> Vec<(usize, f64)> {
    let mut hits: Vec<(usize, f64)> = order
        .iter()
        .copied()
        .zip(ps.iter().copied())
        .filter(|&(_, p)| p >= threshold)
        .collect();
    hits.sort_by(|a, b| b.1.total_cmp(&a.1));
    hits.truncate(top);
    hits
}

// --- one search -------------------------------------------------------------

#[derive(Default)]
struct Stats {
    chunks: usize,
    /// the recall list handed to the model
    candidates: usize,
    cost: Cost,
    store_bytes: Option<u64>,
    recall_ms: f64,
    rerank_ms: f64,
    total_ms: f64,
    hits: usize,
}

fn ms(t: Instant) -> f64 {
    (t.elapsed().as_secs_f64() * 1e4).round() / 10.0
}

/// 5123 -> "5.1k"
fn count(n: usize) -> String {
    match n {
        0..=999 => n.to_string(),
        _ => format!("{:.1}k", n as f64 / 1e3),
    }
}

impl Stats {
    fn json(&self) -> Value {
        let c = &self.cost;
        json!({
            "chunks": self.chunks,
            "candidates": self.candidates,
            "scored": c.scored,
            "restored": c.restored,
            "decoded": c.scored - c.restored,
            "tokens": c.tokens,
            "store_bytes": self.store_bytes,
            "recall_ms": self.recall_ms,
            "rerank_ms": self.rerank_ms,
            "total_ms": self.total_ms,
            "hits": self.hits,
        })
    }

    /// The one line `snap grep` always leaves on stderr.
    fn summary(&self, recall_only: bool) -> String {
        let c = &self.cost;
        let mut p = vec![format!("{} chunks", self.chunks)];
        if !recall_only {
            let decoded = c.scored - c.restored;
            p.push(format!(
                "{} scored ({} restored, {decoded} decoded)",
                c.scored, c.restored
            ));
            p.push(format!("{} tokens", count(c.tokens)));
            p.extend(
                self.store_bytes
                    .map(|b| format!("store {}", crate::fmt_bytes(b))),
            );
        }
        p.push(format!("recall {:.0} ms", self.recall_ms));
        if !recall_only {
            p.push(format!("rerank {:.0} ms", self.rerank_ms));
        }
        p.push(format!("total {:.0} ms", self.total_ms));
        p.push(format!("{} hits", self.hits));
        format!("snap grep: {}", p.join(", "))
    }
}

/// What a search found: the hits as (chunk, P) — P is None when only recall
/// ran — over the chunks they index.
struct Found {
    chunks: Vec<Chunk>,
    hits: Vec<(usize, Option<f64>)>,
    stats: Stats,
}

fn search(s: &Search, base: &Path, load: impl FnOnce() -> Result<Loaded>) -> Result<Found> {
    let (o, t0) = (&s.opts, Instant::now());
    o.check()?;
    let hay = Haystack::new(load_chunks(&o.tree, o.regexp.as_deref(), &[])?);
    let order = hay.recall(&s.query, o.candidates);
    let mut st = Stats {
        chunks: hay.chunks.len(),
        candidates: order.len(),
        recall_ms: ms(t0),
        ..Stats::default()
    };
    let hits = if o.recall_only {
        order.iter().take(o.top).map(|&i| (i, None)).collect()
    } else if order.is_empty() {
        Vec::new()
    } else {
        let (mut eng, gguf) = load()?;
        let mut store = try_store(base, o, &gguf, &eng.model_id);
        let t = Instant::now();
        let (ps, cost) = rerank(
            &mut eng,
            &mut store,
            &hay.chunks,
            &order,
            &s.query,
            o.top,
            o.threshold,
        )?;
        st.rerank_ms = ms(t);
        st.cost = cost;
        st.store_bytes = store.as_ref().map(Store::bytes);
        let hits = pick(&order, &ps, o.top, o.threshold);
        hits.into_iter().map(|(i, p)| (i, Some(p))).collect()
    };
    st.hits = hits.len();
    st.total_ms = ms(t0);
    Ok(Found {
        chunks: hay.chunks,
        hits,
        stats: st,
    })
}

/// `snap grep`: search, print the hits on stdout and the summary on stderr.
pub fn run(s: &Search, load: impl FnOnce() -> Result<Loaded>) -> Result<()> {
    let f = search(s, &default_base(), load)?;
    let shown = Shown::new(&s.opts.tree.root);
    let hits: Vec<Hit> = f
        .hits
        .iter()
        .enumerate()
        .map(|(rank, &(i, p))| Hit {
            chunk: &f.chunks[i],
            shown: shown.path(&f.chunks[i].path),
            p,
            rank: rank + 1,
        })
        .collect();
    let text = match s.out {
        Out::Human => human(&hits, s.lines),
        Out::Files => files(&hits),
        Out::Json => {
            let doc = json_doc(&s.query, &hits, s.lines, f.stats.json());
            format!("{}\n", serde_json::to_string_pretty(&doc)?)
        }
    };
    match std::io::stdout().lock().write_all(text.as_bytes()) {
        // `| head` closing the pipe is not a failure of the search
        Err(e) if e.kind() == std::io::ErrorKind::BrokenPipe => {}
        r => r?,
    }
    eprintln!("{}", f.stats.summary(s.opts.recall_only));
    Ok(())
}

// --- output -----------------------------------------------------------------

/// A chunk's path as the user would type it. Chunks name their file below the
/// root searched; read from where the user stands that is the root as given,
/// then the chunk's path — as ripgrep prints it, and nothing for `.`.
struct Shown {
    root: String,
    file: bool,
}

impl Shown {
    fn new(root: &Path) -> Shown {
        let r = root.to_string_lossy();
        Shown {
            root: if r == "." {
                String::new()
            } else {
                r.into_owned()
            },
            file: root.is_file(),
        }
    }

    fn path(&self, rel: &str) -> String {
        match (self.file, self.root.is_empty()) {
            (true, _) => self.root.clone(),
            (false, true) => rel.to_string(),
            (false, false) => format!("{}/{rel}", self.root.trim_end_matches(['/', '\\'])),
        }
    }
}

struct Hit<'a> {
    chunk: &'a Chunk,
    shown: String,
    /// None when only recall ran
    p: Option<f64>,
    /// 1-based position in the list
    rank: usize,
}

impl Hit<'_> {
    /// The first `lines` source lines of the chunk.
    fn head(&self, lines: usize) -> impl Iterator<Item = &str> {
        self.chunk.text.lines().take(lines)
    }
}

/// `path:start-end  0.94  symbol`, then the source with its line numbers like
/// `rg -n`, a blank line between hits. Recall-only shows the rank where the
/// probability would be.
fn human(hits: &[Hit], lines: usize) -> String {
    let mut out = String::new();
    for (i, h) in hits.iter().enumerate() {
        if i > 0 && lines > 0 {
            out.push('\n');
        }
        let c = h.chunk;
        let score =
            h.p.map_or_else(|| format!("#{}", h.rank), |p| format!("{p:.2}"));
        out.push_str(&format!("{}:{}-{}  {score}", h.shown, c.start, c.end));
        if let Some(sym) = &c.symbol {
            out.push_str(&format!("  {sym}"));
        }
        out.push('\n');
        for (k, line) in h.head(lines).enumerate() {
            out.push_str(&format!("{}:{line}\n", c.start + k));
        }
    }
    out
}

fn json_doc(query: &str, hits: &[Hit], lines: usize, x_snap: Value) -> Value {
    let hits: Vec<Value> = hits
        .iter()
        .map(|h| {
            json!({
                "path": h.shown,
                "start": h.chunk.start,
                "end": h.chunk.end,
                "symbol": h.chunk.symbol,
                "p": h.p,
                "text": h.head(lines).collect::<Vec<_>>().join("\n"),
            })
        })
        .collect();
    json!({"query": query, "hits": hits, "x_snap": x_snap})
}

/// `-l`: each path once, in the order of its first hit.
fn files(hits: &[Hit]) -> String {
    let mut seen = HashSet::new();
    hits.iter()
        .filter(|h| seen.insert(&h.shown))
        .map(|h| format!("{}\n", h.shown))
        .collect()
}

// --- snap index -------------------------------------------------------------

/// A chunk's probe for no query in particular: its prefix is the same for all.
fn probe_of(eng: &Engine, c: &Chunk) -> Result<Probe> {
    probe_for(eng, c, "")
}

/// The key and prefix length of every chunk's snapshot: what the store is
/// asked for, whichever query comes.
fn chunk_keys(eng: &Engine, chunks: &[Chunk]) -> Result<Vec<(u128, usize)>> {
    chunks
        .iter()
        .map(|c| {
            let p = probe_of(eng, c)?;
            Ok((prefix_key(&p.toks[..p.at]), p.at))
        })
        .collect()
}

/// Snapshot every chunk the store lacks, a wave at a time; returns how many
/// were added and the tokens decoded. `progress(done, total, tokens)` runs
/// after each wave.
fn build(
    eng: &mut Engine,
    store: &mut Store,
    chunks: &[Chunk],
    keys: &[(u128, usize)],
    progress: &mut dyn FnMut(usize, usize, usize),
) -> Result<(usize, usize)> {
    let mut seen = HashSet::new();
    let todo: Vec<usize> = (0..chunks.len())
        .filter(|&i| store.tokens(keys[i].0) != Some(keys[i].1) && seen.insert(keys[i].0))
        .collect();
    let wave = eng.n_seq().saturating_sub(1).max(1);
    let (mut done, mut tokens, mut synced) = (0, 0, Instant::now());
    for ids in todo.chunks(wave) {
        let probes = ids
            .iter()
            .map(|&i| probe_of(eng, &chunks[i]))
            .collect::<Result<Vec<Probe>>>()?;
        let prefixes: Vec<&[i32]> = probes.iter().map(|p| &p.toks[..p.at]).collect();
        let mut blobs = vec![None; ids.len()];
        tokens += eng
            .grep_snapshot(&prefixes, &mut |j, b| blobs[j] = Some(b))?
            .decoded;
        for (&i, blob) in ids.iter().zip(blobs) {
            let blob = blob.context("a chunk was decoded but not saved")?;
            store.put(keys[i].0, keys[i].1, &blob)?;
        }
        if synced.elapsed().as_secs_f64() >= SYNC_EVERY {
            store.sync()?;
            synced = Instant::now();
        }
        done += ids.len();
        progress(done, todo.len(), tokens);
    }
    store.sync()?;
    Ok((todo.len(), tokens))
}

/// 200 -> "3m 20s"
fn fmt_eta(secs: f64) -> String {
    let s = secs.round().max(0.0) as u64;
    match s {
        0..=59 => format!("{s}s"),
        60..=3599 => format!("{}m {:02}s", s / 60, s % 60),
        _ => format!("{}h {:02}m", s / 3600, s % 3600 / 60),
    }
}

/// `snap index`: snapshot the whole tree ahead of time, or with `stats` report
/// what the store holds of it, or with `gc` drop what the tree no longer uses.
pub fn index(t: &Tree, stats: bool, gc: bool, load: impl FnOnce() -> Result<Loaded>) -> Result<()> {
    index_in(&default_base(), t, stats, gc, load)
}

fn index_in(
    base: &Path,
    t: &Tree,
    stats: bool,
    gc: bool,
    load: impl FnOnce() -> Result<Loaded>,
) -> Result<()> {
    let t0 = Instant::now();
    let chunks = load_chunks(t, None, &[])?;
    let (mut eng, gguf) = load()?;
    let (dir, binding) = store_spec(base, t, &gguf, &eng.model_id)?;
    if (stats || gc) && !dir.exists() {
        // nothing was ever stored for this tree by this engine, and reading
        // what is not there must not create it
        println!("store      {} (none yet)", dir.display());
        println!("coverage   0/{} chunks (0%)", chunks.len());
        return Ok(());
    }
    let mut store = Store::open(&dir, &binding)?;
    let keys = chunk_keys(&eng, &chunks)?;
    let live: HashSet<u128> = keys.iter().map(|k| k.0).collect();
    if gc {
        let (bytes, n) = (store.bytes(), store.len());
        store.compact(&live)?;
        let (now, kept) = (store.bytes(), store.len());
        let (was, is) = (crate::fmt_bytes(bytes), crate::fmt_bytes(now));
        eprintln!("snap index: store {was} -> {is}, {n} -> {kept} snapshots");
    }
    if stats {
        let covered = (0..chunks.len())
            .filter(|&i| store.tokens(keys[i].0) == Some(keys[i].1))
            .count();
        let used = live.iter().filter(|&&k| store.tokens(k).is_some()).count();
        let pct = 100.0 * covered as f64 / chunks.len().max(1) as f64;
        println!("store      {}", dir.display());
        println!(
            "snapshots  {} ({} on disk)",
            store.len(),
            crate::fmt_bytes(store.bytes())
        );
        println!("coverage   {covered}/{} chunks ({pct:.0}%)", chunks.len());
        println!(
            "stale      {} (`snap index --gc` drops them)",
            store.len() - used
        );
    }
    if stats || gc {
        return Ok(());
    }
    let tty = std::io::stderr().is_terminal();
    let mut progress = |done: usize, total: usize, tokens: usize| {
        let secs = t0.elapsed().as_secs_f64().max(1e-9);
        let eta = secs / done.max(1) as f64 * total.saturating_sub(done) as f64;
        let rate = tokens as f64 / secs;
        let line = format!(
            "snap index: {done}/{total} chunks, {rate:.0} tok/s, ETA {}",
            fmt_eta(eta)
        );
        if tty {
            eprint!("\r\x1b[2K{line}");
        } else {
            eprintln!("{line}");
        }
    };
    let (added, tokens) = build(&mut eng, &mut store, &chunks, &keys, &mut progress)?;
    if tty && added > 0 {
        eprintln!();
    }
    eprintln!(
        "snap index: {} chunks, {added} snapshotted ({} tokens, {}), store {}",
        chunks.len(),
        count(tokens),
        fmt_eta(t0.elapsed().as_secs_f64()),
        crate::fmt_bytes(store.bytes())
    );
    Ok(())
}

// --- eval -------------------------------------------------------------------

/// What a case expects to find: a chunk of `path` whose text contains
/// `contains`. Anchors name code, not line numbers, so they outlive edits.
#[derive(Deserialize)]
struct Anchor {
    path: String,
    contains: String,
}

impl Anchor {
    fn met_by(&self, c: &Chunk) -> bool {
        c.path == self.path && c.text.contains(&self.contains)
    }
}

fn other() -> String {
    "other".into()
}

#[derive(Deserialize)]
struct Case {
    id: String,
    query: String,
    #[serde(default = "other")]
    kind: String,
    expect: Vec<Anchor>,
}

/// Does `c` satisfy any anchor?
fn satisfies(anchors: &[Anchor], c: &Chunk) -> bool {
    anchors.iter().any(|a| a.met_by(c))
}

/// Share of the anchors some chunk of the first `k` of `ranked` satisfies.
fn anchor_recall(chunks: &[Chunk], ranked: &[usize], anchors: &[Anchor], k: usize) -> f64 {
    let top = &ranked[..k.min(ranked.len())];
    let met = anchors
        .iter()
        .filter(|a| top.iter().any(|&i| a.met_by(&chunks[i])));
    met.count() as f64 / anchors.len() as f64
}

/// 1 / the rank of the first chunk satisfying an anchor; 0 when none does.
fn reciprocal_rank(chunks: &[Chunk], ranked: &[usize], anchors: &[Anchor]) -> f64 {
    let first = ranked.iter().position(|&i| satisfies(anchors, &chunks[i]));
    first.map_or(0.0, |r| 1.0 / (r + 1) as f64)
}

/// Nearest-rank percentile of an ascending slice.
fn percentile(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    match n {
        0 => 0.0,
        _ => sorted[((q * n as f64).ceil() as usize).clamp(1, n) - 1],
    }
}

fn mean(xs: impl Iterator<Item = f64>) -> f64 {
    let (sum, n) = xs.fold((0.0, 0), |(s, n), x| (s + x, n + 1));
    if n == 0 {
        0.0
    } else {
        sum / n as f64
    }
}

fn share(n: usize, of: usize) -> f64 {
    if of == 0 {
        0.0
    } else {
        n as f64 / of as f64
    }
}

/// How the model's order did on one case, over every candidate it scored,
/// best P first.
struct Reranked {
    hit1: bool,
    r5: f64,
    r10: f64,
    rr: f64,
    /// some candidate reached the threshold
    answered: bool,
    /// P of the candidates that satisfy an anchor, and of those that do not
    sat: Vec<f64>,
    non: Vec<f64>,
}

impl Reranked {
    fn of(
        chunks: &[Chunk],
        anchors: &[Anchor],
        order: &[usize],
        ps: &[f64],
        threshold: f64,
    ) -> Reranked {
        // best P first, recall order among equals
        let mut by_p: Vec<(usize, f64)> = order.iter().copied().zip(ps.iter().copied()).collect();
        by_p.sort_by(|a, b| b.1.total_cmp(&a.1));
        let ranked: Vec<usize> = by_p.iter().map(|x| x.0).collect();
        let of = |sat: bool| -> Vec<f64> {
            let held = by_p
                .iter()
                .filter(|x| satisfies(anchors, &chunks[x.0]) == sat);
            held.map(|x| x.1).collect()
        };
        Reranked {
            hit1: ranked
                .first()
                .is_some_and(|&i| satisfies(anchors, &chunks[i])),
            r5: anchor_recall(chunks, &ranked, anchors, 5),
            r10: anchor_recall(chunks, &ranked, anchors, 10),
            rr: reciprocal_rank(chunks, &ranked, anchors),
            answered: by_p.first().is_some_and(|x| x.1 >= threshold),
            sat: of(true),
            non: of(false),
        }
    }
}

struct Row {
    id: String,
    kind: String,
    query: String,
    /// anchor recall@k of the recall order, for the ks in `KS`
    recall: [f64; 4],
    /// rank of the first chunk satisfying an anchor in the recall order
    first: Option<usize>,
    reranked: Option<Reranked>,
    /// the model's ten best: chunk, P, whether it satisfies an anchor
    top: Vec<(usize, f64, bool)>,
    recall_ms: f64,
    rerank_ms: f64,
}

impl Row {
    fn ms(&self) -> f64 {
        self.recall_ms + self.rerank_ms
    }
}

/// recall@k as a JSON object, in the order of `KS`.
fn at_ks(recall: &[f64; 4]) -> Value {
    let at = KS.iter().zip(recall);
    Value::Object(at.map(|(k, &r)| (k.to_string(), json!(r))).collect())
}

/// Means over a set of cases.
struct Summary {
    n: usize,
    recall: [f64; 4],
    /// share of cases with every anchor in the top 64
    all64: f64,
    rerank: Option<RerankSummary>,
    median_ms: f64,
    p90_ms: f64,
}

struct RerankSummary {
    hit1: f64,
    r5: f64,
    r10: f64,
    mrr: f64,
    answered: f64,
    p_sat: Option<f64>,
    p_non: Option<f64>,
}

impl Summary {
    fn of(rows: &[&Row]) -> Summary {
        let rr: Vec<&Reranked> = rows.iter().filter_map(|r| r.reranked.as_ref()).collect();
        let sat: Vec<f64> = rr.iter().flat_map(|r| r.sat.iter().copied()).collect();
        let non: Vec<f64> = rr.iter().flat_map(|r| r.non.iter().copied()).collect();
        let mut ms: Vec<f64> = rows.iter().map(|r| r.ms()).collect();
        ms.sort_by(f64::total_cmp);
        Summary {
            n: rows.len(),
            recall: std::array::from_fn(|k| mean(rows.iter().map(|r| r.recall[k]))),
            all64: share(
                rows.iter().filter(|r| r.recall[3] >= 1.0).count(),
                rows.len(),
            ),
            rerank: (!rr.is_empty()).then(|| RerankSummary {
                hit1: share(rr.iter().filter(|r| r.hit1).count(), rr.len()),
                r5: mean(rr.iter().map(|r| r.r5)),
                r10: mean(rr.iter().map(|r| r.r10)),
                mrr: mean(rr.iter().map(|r| r.rr)),
                answered: share(rr.iter().filter(|r| r.answered).count(), rr.len()),
                p_sat: (!sat.is_empty()).then(|| mean(sat.iter().copied())),
                p_non: (!non.is_empty()).then(|| mean(non.iter().copied())),
            }),
            median_ms: percentile(&ms, 0.5),
            p90_ms: percentile(&ms, 0.9),
        }
    }

    fn json(&self) -> Value {
        json!({
            "cases": self.n,
            "recall": at_ks(&self.recall),
            "all_anchors_in_64": self.all64,
            "rerank": self.rerank.as_ref().map(|r| json!({
                "hit_at_1": r.hit1,
                "recall_at_5": r.r5,
                "recall_at_10": r.r10,
                "mrr": r.mrr,
                "answered": r.answered,
                "mean_p_satisfying": r.p_sat,
                "mean_p_other": r.p_non,
            })),
            "latency_ms": {"median": self.median_ms, "p90": self.p90_ms},
        })
    }
}

/// The compact table `snap grep --eval` prints: recall always, rerank when it ran.
fn table(groups: &[(String, Summary)], model: Option<&str>, threshold: f64) -> String {
    let mut out = String::from("recall: anchor recall@k over the candidates\n");
    out += &format!(
        "{:<11} {:>3}  {:>5} {:>5} {:>5} {:>5}  {:>6}\n",
        "kind", "n", "@1", "@5", "@10", "@64", "all@64"
    );
    for (k, s) in groups {
        let r = s.recall;
        out += &format!(
            "{k:<11} {:>3}  {:>5.2} {:>5.2} {:>5.2} {:>5.2}  {:>6.2}\n",
            s.n, r[0], r[1], r[2], r[3], s.all64
        );
    }
    match model {
        Some(m) => {
            out += &format!("\nrerank: {m}, hits at P >= {threshold:.2}\n");
            out += &format!(
                "{:<11} {:>3}  {:>5} {:>5} {:>5} {:>5} {:>8}  {:>5} {:>5}  {:>7} {:>7}\n",
                "kind",
                "n",
                "hit@1",
                "@5",
                "@10",
                "MRR",
                "answered",
                "P sat",
                "P not",
                "med ms",
                "p90 ms"
            );
            let p = |v: Option<f64>| v.map_or("-".to_string(), |v| format!("{v:.2}"));
            for (k, s) in groups {
                let Some(r) = &s.rerank else { continue };
                out += &format!(
                    "{k:<11} {:>3}  {:>5.2} {:>5.2} {:>5.2} {:>5.2} {:>8.2}  {:>5} {:>5}  {:>7.0} {:>7.0}\n",
                    s.n,
                    r.hit1,
                    r.r5,
                    r.r10,
                    r.mrr,
                    r.answered,
                    p(r.p_sat),
                    p(r.p_non),
                    s.median_ms,
                    s.p90_ms
                );
            }
        }
        None => {
            if let Some((_, s)) = groups.last() {
                out += &format!(
                    "\nlatency per case: median {:.0} ms, p90 {:.0} ms\n",
                    s.median_ms, s.p90_ms
                );
            }
        }
    }
    out
}

/// `snap grep --eval FILE`: every case's query over the tree, scored against
/// its anchors, the table on stdout and the report (create-only) at `output`.
/// The cases file and the report are left out of the corpus: each holds every
/// query verbatim, and would answer it.
pub fn eval(
    file: &str,
    output: Option<&str>,
    o: &Opts,
    load: impl FnOnce() -> Result<Loaded>,
) -> Result<()> {
    let Some(path) = output else {
        return run_eval(file, None, &default_base(), o, load).map(drop);
    };
    // the file is created before the model loads: an existing report fails first
    evaluate::create_only(path, |w| {
        let report = run_eval(file, Some(path), &default_base(), o, load)?;
        Ok(serde_json::to_writer_pretty(w, &report)?)
    })?;
    eprintln!("report written: {path}");
    Ok(())
}

fn run_eval(
    file: &str,
    output: Option<&str>,
    base: &Path,
    o: &Opts,
    load: impl FnOnce() -> Result<Loaded>,
) -> Result<Value> {
    o.check()?;
    let cases: Vec<Case> = evaluate::load_cases(file)?
        .into_iter()
        .map(|c| serde_json::from_value(c).map_err(anyhow::Error::from))
        .collect::<Result<_>>()
        .with_context(|| format!("cases in {file}"))?;
    ensure!(!cases.is_empty(), "{file}: no cases");
    if let Some(c) = cases.iter().find(|c| c.expect.is_empty()) {
        bail!("case {}: no anchors", c.id);
    }
    let skip: Vec<String> = [Some(file), output]
        .into_iter()
        .flatten()
        .filter_map(|p| rel_to(&o.tree.root, p))
        .collect();
    let hay = Haystack::new(load_chunks(&o.tree, o.regexp.as_deref(), &skip)?);
    let loaded = (!o.recall_only).then(load).transpose()?;
    let model = loaded.as_ref().map(|(e, _)| e.model_id.clone());
    let (mut eng, mut store) = match loaded {
        Some((e, gguf)) => {
            let store = try_store(base, o, &gguf, &e.model_id);
            (Some(e), store)
        }
        None => (None, None),
    };
    let mut rows = Vec::with_capacity(cases.len());
    for (n, c) in cases.iter().enumerate() {
        for a in c
            .expect
            .iter()
            .filter(|a| !hay.chunks.iter().any(|ch| a.met_by(ch)))
        {
            eprintln!(
                "snap: case {}: no chunk of {} holds {:?} — a stale anchor?",
                c.id, a.path, a.contains
            );
        }
        let t = Instant::now();
        let order = hay.recall(&c.query, o.candidates);
        let recall_ms = ms(t);
        let t = Instant::now();
        let scored = match eng.as_mut() {
            Some(e) if !order.is_empty() => Some(
                rerank(
                    e,
                    &mut store,
                    &hay.chunks,
                    &order,
                    &c.query,
                    o.top,
                    o.threshold,
                )?
                .0,
            ),
            Some(_) => Some(Vec::new()),
            None => None,
        };
        let rerank_ms = scored.as_ref().map_or(0.0, |_| ms(t));
        let top = scored.as_ref().map_or_else(Vec::new, |ps| {
            let best = pick(&order, ps, 10, 0.0);
            best.into_iter()
                .map(|(i, p)| (i, p, satisfies(&c.expect, &hay.chunks[i])))
                .collect()
        });
        let row = Row {
            id: c.id.clone(),
            kind: c.kind.clone(),
            query: c.query.clone(),
            recall: std::array::from_fn(|k| anchor_recall(&hay.chunks, &order, &c.expect, KS[k])),
            first: order
                .iter()
                .position(|&i| satisfies(&c.expect, &hay.chunks[i]))
                .map(|r| r + 1),
            reranked: scored
                .map(|ps| Reranked::of(&hay.chunks, &c.expect, &order, &ps, o.threshold)),
            top,
            recall_ms,
            rerank_ms,
        };
        eprintln!(
            "snap eval: {}/{} {} ({:.0} ms)",
            n + 1,
            cases.len(),
            row.id,
            row.ms()
        );
        rows.push(row);
    }
    let mut kinds: BTreeMap<&str, Vec<&Row>> = BTreeMap::new();
    for r in &rows {
        kinds.entry(&r.kind).or_default().push(r);
    }
    let all: Vec<&Row> = rows.iter().collect();
    let overall = Summary::of(&all);
    let mut groups: Vec<(String, Summary)> = kinds
        .iter()
        .map(|(k, v)| (k.to_string(), Summary::of(v)))
        .collect();
    groups.push(("overall".into(), overall));
    print!("{}", table(&groups, model.as_deref(), o.threshold));
    let cases_json: Vec<Value> = rows
        .iter()
        .map(|r| {
            let top: Vec<Value> = r
                .top
                .iter()
                .map(|&(i, p, sat)| {
                    let c = &hay.chunks[i];
                    json!({"path": c.path, "start": c.start, "end": c.end, "p": p, "satisfies": sat})
                })
                .collect();
            json!({
                "id": r.id,
                "kind": r.kind,
                "query": r.query,
                "recall": at_ks(&r.recall),
                "first_satisfying_rank": r.first,
                "rerank": r.reranked.as_ref().map(|m| json!({
                    "hit_at_1": m.hit1,
                    "recall_at_5": m.r5,
                    "recall_at_10": m.r10,
                    "reciprocal_rank": m.rr,
                    "answered": m.answered,
                })),
                "top": top,
                "ms": {"recall": r.recall_ms, "rerank": r.rerank_ms},
            })
        })
        .collect();
    let (overall, by_kind) =
        groups
            .split_last()
            .map_or((Value::Null, BTreeMap::new()), |((_, s), rest)| {
                (
                    s.json(),
                    rest.iter()
                        .map(|(k, s)| (k.clone(), s.json()))
                        .collect::<BTreeMap<_, _>>(),
                )
            });
    Ok(json!({
        "kind": "snap-grep-eval",
        "file": file,
        "root": o.tree.root.to_string_lossy(),
        "model": model,
        "kv": o.tree.kv.as_str(),
        "chunks": hay.chunks.len(),
        "candidates": o.candidates,
        "top": o.top,
        "threshold": o.threshold,
        "overall": overall,
        "by_kind": by_kind,
        "cases": cases_json,
    }))
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;
    use crate::kv::sim::{blob, Sim};

    /// Scratch dir that goes away with the test, pass or fail.
    struct Tmp(PathBuf);

    impl Tmp {
        fn new(name: &str) -> Tmp {
            let dir = std::env::temp_dir().join(format!("snap-grep-{}-{name}", std::process::id()));
            let _ = fs::remove_dir_all(&dir);
            fs::create_dir_all(&dir).unwrap();
            Tmp(dir)
        }

        fn at(&self, rel: &str) -> PathBuf {
            self.0.join(rel)
        }

        fn write(&self, rel: &str, text: &str) -> PathBuf {
            let p = self.at(rel);
            fs::create_dir_all(p.parent().unwrap()).unwrap();
            fs::write(&p, text).unwrap();
            p
        }
    }

    impl Drop for Tmp {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn chunk(path: &str, start: usize, symbol: Option<&str>, text: &str) -> Chunk {
        Chunk {
            path: path.into(),
            start,
            end: start + text.lines().count().max(1) - 1,
            symbol: symbol.map(Into::into),
            text: text.into(),
        }
    }

    /// An engine on the sim with `n_seq` seqs: waves of `n_seq - 1`.
    fn engine(n_seq: usize) -> Engine {
        Engine::new(Box::new(Sim::new(1 << 16, n_seq, 256)), "snap-sim".into()).unwrap()
    }

    /// `n` distinct chunks over files of three, the last file prose.
    fn synthetic(n: usize) -> Vec<Chunk> {
        (0..n)
            .map(|i| {
                let ext = if i / 3 == 3 { "md" } else { "rs" };
                chunk(
                    &format!("src/f{}.{ext}", i / 3),
                    1 + 10 * (i % 3),
                    Some(&format!("sym{i}")),
                    &format!("fn sym{i}() {{\n    let v = {i};\n    v * 2\n}}"),
                )
            })
            .collect()
    }

    fn tree(root: &Path) -> Tree {
        Tree {
            root: root.to_path_buf(),
            walk: WalkOpts::default(),
            kv: KvType::F16,
        }
    }

    fn opts(root: &Path) -> Opts {
        Opts {
            tree: tree(root),
            regexp: None,
            top: 3,
            candidates: 64,
            threshold: 0.0,
            store: true,
            recall_only: false,
        }
    }

    // --- recall

    /// slots.rs matches the query in its first chunk only; other.rs holds one
    /// chunk with a query word; zzz.rs shares nothing. (A path is searched
    /// too: no file here is named after the query.)
    fn pool_tree() -> Vec<Chunk> {
        vec![
            chunk(
                "src/slots.rs",
                1,
                Some("acquire"),
                "pub fn acquire(&self) -> Conn {\n    // check a connection out of the pool\n}",
            ),
            chunk(
                "src/slots.rs",
                5,
                Some("helper"),
                "fn helper() {\n    let x = 1;\n}",
            ),
            chunk(
                "src/slots.rs",
                9,
                Some("tidy"),
                "fn tidy() {\n    let y = 2;\n}",
            ),
            chunk(
                "src/other.rs",
                1,
                Some("render"),
                "fn render() {\n    println!(\"hello\");\n}",
            ),
            chunk(
                "src/other.rs",
                5,
                Some("parse"),
                "fn parse(s: &str) {\n    // connection strings\n}",
            ),
            chunk("src/zzz.rs", 1, Some("nothing"), "fn nothing() {}"),
        ]
    }

    #[test]
    fn a_files_other_chunks_ride_in_with_the_chunk_that_matched() {
        let hay = Haystack::new(pool_tree());
        let q = "connection pool";
        // the chunk ranking alone never sees what shares no word with the query
        let direct: Vec<usize> = hay.by_chunk.search(q, 6).into_iter().map(|h| h.0).collect();
        assert_eq!(direct, [0, 4]);
        // the file ranking adds slots.rs's other chunks, in line order, ahead
        // of the unrelated chunk of the file that matched less
        assert_eq!(hay.recall(q, 64), [0, 4, 1, 2, 3]);
        // zzz.rs is in no ranking
        assert!(!hay.recall(q, 64).contains(&5));
    }

    #[test]
    fn candidates_cut_the_ranking_and_zero_keeps_every_chunk() {
        let hay = Haystack::new(pool_tree());
        let q = "connection pool";
        assert_eq!(hay.recall(q, 2), [0, 4]);
        assert_eq!(hay.recall(q, 1), [0]);
        // every chunk: the ranked ones first, the rest in corpus order
        assert_eq!(hay.recall(q, 0), [0, 4, 1, 2, 3, 5]);
        // nothing matches: no candidates, unless every chunk is asked for
        assert!(hay.recall("zebra", 64).is_empty());
        assert_eq!(hay.recall("zebra", 0), [0, 1, 2, 3, 4, 5]);
        assert!(Haystack::new(Vec::new()).recall(q, 0).is_empty());
    }

    // --- cascade, picking, output

    #[test]
    fn the_cascade_stops_when_enough_hits_are_in_and_a_wave_adds_none() {
        assert!(!done(0, 0, 1), "no hit yet: go on");
        assert!(!done(4, 0, 5), "short of top: go on");
        assert!(!done(5, 2, 5), "the wave still added: go on");
        assert!(done(5, 0, 5));
        assert!(done(9, 0, 5));
    }

    #[test]
    fn hits_are_the_best_above_the_threshold_with_recall_order_among_equals() {
        let order = [7, 3, 9, 1, 4];
        let ps = [0.6, 0.9, 0.9, 0.2, 0.7];
        // 3 and 9 tie: 3 came first in recall
        assert_eq!(
            pick(&order, &ps, 10, 0.5),
            [(3, 0.9), (9, 0.9), (4, 0.7), (7, 0.6)]
        );
        assert_eq!(pick(&order, &ps, 2, 0.5), [(3, 0.9), (9, 0.9)]);
        // the threshold is inclusive; scores cover a prefix of the candidates
        assert_eq!(pick(&order, &ps[..3], 10, 0.9), [(3, 0.9), (9, 0.9)]);
        assert!(pick(&order, &ps, 10, 0.95).is_empty());
    }

    fn hit<'a>(c: &'a Chunk, shown: &str, p: Option<f64>, rank: usize) -> Hit<'a> {
        Hit {
            chunk: c,
            shown: shown.into(),
            p,
            rank,
        }
    }

    #[test]
    fn human_output_numbers_the_source_lines_like_rg() {
        let a = chunk("src/a.rs", 10, Some("a"), "fn a() {\n    1\n}");
        let b = chunk("src/b.rs", 3, None, "x\ny");
        let hits = [
            hit(&a, "src/a.rs", Some(0.944), 1),
            hit(&b, "src/b.rs", Some(0.5), 2),
        ];
        assert_eq!(
            human(&hits, 20),
            "src/a.rs:10-12  0.94  a\n10:fn a() {\n11:    1\n12:}\n\nsrc/b.rs:3-4  0.50\n3:x\n4:y\n"
        );
        // a cap keeps whole lines from the top
        assert_eq!(
            human(&hits, 1),
            "src/a.rs:10-12  0.94  a\n10:fn a() {\n\nsrc/b.rs:3-4  0.50\n3:x\n"
        );
        // locations only: a line per hit, nothing between
        assert_eq!(
            human(&hits, 0),
            "src/a.rs:10-12  0.94  a\nsrc/b.rs:3-4  0.50\n"
        );
        // recall only shows ranks where the probabilities would be
        let r = [hit(&a, "src/a.rs", None, 1), hit(&b, "src/b.rs", None, 2)];
        assert_eq!(human(&r, 0), "src/a.rs:10-12  #1  a\nsrc/b.rs:3-4  #2\n");
        assert_eq!(human(&[], 20), "");
    }

    #[test]
    fn json_output_is_one_document_of_hits_and_stats() {
        let a = chunk("src/a.rs", 10, Some("a"), "fn a() {\n    1\n}");
        let b = chunk("src/b.rs", 3, None, "x");
        let hits = [
            hit(&a, "./src/a.rs", Some(0.9), 1),
            hit(&b, "./src/b.rs", None, 2),
        ];
        let doc = json_doc("where?", &hits, 2, json!({"hits": 2}));
        let want = json!({
            "query": "where?",
            "hits": [
                {"path": "./src/a.rs", "start": 10, "end": 12, "symbol": "a", "p": 0.9, "text": "fn a() {\n    1"},
                {"path": "./src/b.rs", "start": 3, "end": 3, "symbol": null, "p": null, "text": "x"},
            ],
            "x_snap": {"hits": 2},
        });
        assert_eq!(doc, want);
        // locations only leaves the text empty, the field stays
        assert_eq!(json_doc("q", &hits, 0, json!({}))["hits"][0]["text"], "");
    }

    #[test]
    fn files_only_lists_each_path_once_in_hit_order() {
        let c = chunk("x", 1, None, "x");
        let hits: Vec<Hit> = ["b.rs", "a.rs", "b.rs", "c.rs", "a.rs"]
            .iter()
            .enumerate()
            .map(|(i, p)| hit(&c, p, Some(0.9), i + 1))
            .collect();
        assert_eq!(files(&hits), "b.rs\na.rs\nc.rs\n");
    }

    #[test]
    fn paths_are_shown_from_where_the_user_stands() {
        let t = Tmp::new("shown");
        let file = t.write("one.rs", "fn one() {}");
        let show = |root: &str, rel: &str| Shown::new(Path::new(root)).path(rel);
        assert_eq!(show(".", "src/a.rs"), "src/a.rs");
        assert_eq!(show("src", "a.rs"), "src/a.rs");
        assert_eq!(show("./src/", "a.rs"), "./src/a.rs");
        assert_eq!(show("../other", "src/a.rs"), "../other/src/a.rs");
        // a file searched on its own is shown as it was named
        assert_eq!(Shown::new(&file).path("one.rs"), file.to_string_lossy());
    }

    #[test]
    fn the_summary_is_one_line_of_what_the_run_did() {
        let st = Stats {
            chunks: 412,
            candidates: 64,
            cost: Cost {
                scored: 64,
                restored: 60,
                tokens: 5123,
            },
            store_bytes: Some(1_300_000_000),
            recall_ms: 12.0,
            rerank_ms: 830.0,
            total_ms: 845.0,
            hits: 7,
        };
        assert_eq!(
            st.summary(false),
            "snap grep: 412 chunks, 64 scored (60 restored, 4 decoded), 5.1k tokens, store 1.3 GB, recall 12 ms, rerank 830 ms, total 845 ms, 7 hits"
        );
        assert_eq!(
            st.summary(true),
            "snap grep: 412 chunks, recall 12 ms, total 845 ms, 7 hits"
        );
        let j = st.json();
        assert_eq!(
            (
                j["scored"].as_u64(),
                j["decoded"].as_u64(),
                j["restored"].as_u64()
            ),
            (Some(64), Some(4), Some(60))
        );
        assert_eq!(j["store_bytes"], 1_300_000_000u64);
        assert_eq!(count(999), "999");
        assert_eq!(count(5123), "5.1k");
    }

    #[test]
    fn eta_reads_as_a_person_would_say_it() {
        assert_eq!(fmt_eta(0.4), "0s");
        assert_eq!(fmt_eta(45.0), "45s");
        assert_eq!(fmt_eta(200.0), "3m 20s");
        assert_eq!(fmt_eta(3900.0), "1h 05m");
        assert_eq!(fmt_eta(-3.0), "0s");
    }

    // --- store location and binding

    fn env<'a>(pairs: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        move |k| pairs.iter().find(|p| p.0 == k).map(|p| p.1.to_string())
    }

    #[test]
    fn the_store_lives_where_the_first_set_variable_says() {
        let temp = Path::new("/t");
        let all = [
            ("SNAP_GREP_DIR", "/g"),
            ("XDG_CACHE_HOME", "/x"),
            ("HOME", "/h"),
            ("LOCALAPPDATA", "/l"),
        ];
        let from = |n: usize| store_base(&env(&all[n..]), temp);
        assert_eq!(from(0), Path::new("/g"));
        assert_eq!(from(1), Path::new("/x").join("snap").join("grep"));
        assert_eq!(
            from(2),
            Path::new("/h").join(".cache").join("snap").join("grep")
        );
        assert_eq!(from(3), Path::new("/l").join("snap").join("grep"));
        assert_eq!(from(4), Path::new("/t").join("snap-grep"));
        // an empty variable is no variable
        let blank = [
            ("SNAP_GREP_DIR", ""),
            ("XDG_CACHE_HOME", ""),
            ("HOME", "/h"),
        ];
        assert_eq!(store_base(&env(&blank), temp), from(2));
    }

    #[test]
    fn a_store_directory_is_named_by_its_root_and_its_binding() {
        let base = Path::new("/b");
        let bind = binding("snap-m", (1000, 77), KvType::F16);
        let dir = store_dir(base, Path::new("/repo"), &bind);
        // deterministic, two levels of 16 hex digits
        assert_eq!(dir, store_dir(base, Path::new("/repo"), &bind));
        let (tree, bound) = (dir.parent().unwrap(), dir.file_name().unwrap());
        for part in [tree.file_name().unwrap(), bound] {
            let s = part.to_str().unwrap();
            assert!(
                s.len() == 16 && s.bytes().all(|b| b.is_ascii_hexdigit()),
                "{s}"
            );
        }
        assert_eq!(tree.parent().unwrap(), base);
        // another root is another tree, a changed binding another leaf
        let other = store_dir(base, Path::new("/repo2"), &bind);
        assert_ne!(other.parent(), dir.parent());
        let moved = store_dir(
            base,
            Path::new("/repo"),
            &binding("snap-m", (1000, 78), KvType::F16),
        );
        assert_eq!(moved.parent(), dir.parent());
        assert_ne!(moved, dir);
        assert_eq!(
            hex16(0x0123_4567_89ab_cdef_fedc_ba98_7654_3210),
            "0123456789abcdef"
        );
    }

    #[test]
    fn the_binding_names_everything_a_snapshot_depends_on() {
        let b = binding("snap-m", (1000, 77), KvType::F16);
        assert!(b.contains(concat!("snap ", env!("CARGO_PKG_VERSION"))));
        assert!(b.contains("model snap-m"));
        assert!(b.contains("1000 bytes, mtime 77"));
        assert!(b.contains(&format!("prompt v{PROMPT_VERSION}")));
        assert!(b.contains(&format!("grep format {GREP_FORMAT}")));
        assert!(b.contains("kv f16"));
        // every one of them is a different binding
        for other in [
            binding("snap-n", (1000, 77), KvType::F16),
            binding("snap-m", (1001, 77), KvType::F16),
            binding("snap-m", (1000, 78), KvType::F16),
            binding("snap-m", (1000, 77), KvType::Q8_0),
        ] {
            assert_ne!(other, b);
        }
    }

    #[test]
    fn a_gguf_is_identified_by_size_and_mtime() {
        let t = Tmp::new("gguf");
        let p = t.write("m.gguf", "twelve bytes");
        let (len, mtime) = gguf_identity(&p).unwrap();
        assert_eq!(len, 12);
        assert!(mtime > 1_000_000_000);
        assert!(gguf_identity(&t.at("missing.gguf")).is_err());
    }

    #[test]
    fn a_store_held_elsewhere_costs_speed_not_answers() {
        let t = Tmp::new("held");
        let gguf = t.write("m.gguf", "weights");
        let root = t.at("tree");
        fs::create_dir_all(&root).unwrap();
        let o = opts(&root);
        let first = try_store(&t.at("cache"), &o, &gguf, "snap-sim").expect("opens");
        // the second opener gets none, and says so on stderr
        assert!(try_store(&t.at("cache"), &o, &gguf, "snap-sim").is_none());
        drop(first);
        assert!(try_store(&t.at("cache"), &o, &gguf, "snap-sim").is_some());
        // --no-store never opens anything
        let off = Opts {
            store: false,
            ..opts(&root)
        };
        assert!(try_store(&t.at("cache2"), &off, &gguf, "snap-sim").is_none());
        assert!(!t.at("cache2").exists());
    }

    // --- corpus

    /// A small project under `dir` of the scratch dir; returns its root.
    fn project(t: &Tmp, dir: &str) -> PathBuf {
        let w = |rel: &str, text: &str| t.write(&format!("{dir}/{rel}"), text);
        w("src/pool.rs", "pub fn acquire(&self) -> Conn {\n    // check a connection out of the pool\n    self.slots.pop()\n}\n\npub fn release(&self, conn: Conn) {\n    self.slots.push(conn)\n}\n");
        w("src/net.py", "def dial(host):\n    return socket(host)\n");
        w(
            "docs/guide.md",
            "# Guide\n\nRun the server, then ask it questions.\n",
        );
        w(".hidden/secret.txt", "tucked away\n");
        w("ignored.rs", "fn ignored() {}\n");
        w(".ignore", "ignored.rs\n");
        t.at(dir)
    }

    fn paths(chunks: &[Chunk]) -> Vec<&str> {
        let mut p: Vec<&str> = chunks.iter().map(|c| c.path.as_str()).collect();
        p.dedup();
        p
    }

    #[test]
    fn the_corpus_is_the_walk_minus_skips_narrowed_by_the_regex() {
        let t = Tmp::new("corpus");
        let root = project(&t, "tree");
        let tr = tree(&root);
        let all = load_chunks(&tr, None, &[]).unwrap();
        assert_eq!(paths(&all), ["docs/guide.md", "src/net.py", "src/pool.rs"]);
        // skipped by path, as the eval's own file is
        let skipped = load_chunks(&tr, None, &["src/net.py".into()]).unwrap();
        assert_eq!(paths(&skipped), ["docs/guide.md", "src/pool.rs"]);
        // -e keeps the chunks whose text matches (smart case)
        let re = load_chunks(&tr, Some("fn release"), &[]).unwrap();
        assert_eq!(paths(&re), ["src/pool.rs"]);
        assert!(re[0].text.contains("fn release"));
        assert_eq!(load_chunks(&tr, Some("Socket"), &[]).unwrap().len(), 0);
        assert!(load_chunks(&tr, Some("("), &[]).is_err());
        // the walk flags reach the walk
        let walk = WalkOpts {
            hidden: true,
            no_ignore: true,
            ..WalkOpts::default()
        };
        let wide = load_chunks(
            &Tree {
                walk,
                ..tree(&root)
            },
            None,
            &[],
        )
        .unwrap();
        assert!(
            paths(&wide).contains(&".hidden/secret.txt") && paths(&wide).contains(&"ignored.rs")
        );
    }

    #[test]
    fn a_file_named_as_the_root_is_searched_whatever_the_walk_says() {
        let t = Tmp::new("file");
        let root = project(&t, "tree");
        let one = tree(&root.join("ignored.rs"));
        let chunks = load_chunks(&one, None, &[]).unwrap();
        assert_eq!(paths(&chunks), ["ignored.rs"]);
        // not text: nothing to search, not an error
        t.write("blob.bin", "\0\0\0\0");
        assert!(load_chunks(&tree(&t.at("blob.bin")), None, &[])
            .unwrap()
            .is_empty());
        assert!(load_chunks(&tree(&t.at("nowhere")), None, &[]).is_err());
    }

    #[test]
    fn a_path_is_found_below_the_root_or_not_at_all() {
        let t = Tmp::new("rel");
        let f = t.write("eval/cases.jsonl", "{}");
        let p = f.to_string_lossy().into_owned();
        assert_eq!(rel_to(&t.0, &p).as_deref(), Some("eval/cases.jsonl"));
        assert_eq!(rel_to(&t.at("eval"), &p).as_deref(), Some("cases.jsonl"));
        assert_eq!(rel_to(&t.at("src"), &p), None);
        assert_eq!(rel_to(&t.0, &t.at("absent").to_string_lossy()), None);
    }

    // --- rerank on the sim

    const Q: &str = "where is the pool checked out";

    /// P(yes) of every chunk, from scratch, in `order`.
    fn cold(eng: &mut Engine, chunks: &[Chunk], order: &[usize], q: &str) -> Vec<f64> {
        rerank(eng, &mut None, chunks, order, q, usize::MAX, 0.5)
            .unwrap()
            .0
    }

    #[test]
    fn a_second_pass_restores_every_candidate_from_the_store() {
        let t = Tmp::new("warm");
        let chunks = synthetic(10);
        let order: Vec<usize> = (0..10).rev().collect();
        let mut eng = engine(4);
        let mut store = Some(Store::open(&t.0, "b").unwrap());
        let (first, c1) =
            rerank(&mut eng, &mut store, &chunks, &order, Q, usize::MAX, 0.5).unwrap();
        assert_eq!((c1.scored, c1.restored), (10, 0));
        assert_eq!(store.as_ref().unwrap().len(), 10);
        // scores are what decoding from scratch gives, on the sim exactly
        assert_eq!(first, cold(&mut eng, &chunks, &order, Q));

        let (again, c2) =
            rerank(&mut eng, &mut store, &chunks, &order, Q, usize::MAX, 0.5).unwrap();
        assert_eq!(again, first);
        assert_eq!((c2.scored, c2.restored), (10, 10));
        // what decodes is the question tail of every probe, and nothing else
        let tails: usize = order
            .iter()
            .map(|&i| {
                let p = probe_for(&eng, &chunks[i], Q).unwrap();
                p.toks.len() - p.at
            })
            .sum();
        assert_eq!(c2.tokens, tails);
        assert!(c2.tokens < c1.tokens);

        // another query reads the same snapshots, in a different order
        let backwards: Vec<usize> = (0..10).collect();
        let (_, c3) = rerank(
            &mut eng,
            &mut store,
            &chunks,
            &backwards,
            "something else",
            usize::MAX,
            0.5,
        )
        .unwrap();
        assert_eq!(c3.restored, 10);

        // and a new process finds them on disk
        drop(store);
        let mut store = Some(Store::open(&t.0, "b").unwrap());
        let mut fresh = engine(4);
        let (disk, c4) =
            rerank(&mut fresh, &mut store, &chunks, &order, Q, usize::MAX, 0.5).unwrap();
        assert_eq!((disk, c4.restored), (first, 10));
    }

    #[test]
    fn without_a_store_every_candidate_is_decoded_from_scratch() {
        let chunks = synthetic(7);
        let order: Vec<usize> = (0..7).collect();
        let mut eng = engine(4);
        let (ps, c) = rerank(&mut eng, &mut None, &chunks, &order, Q, usize::MAX, 0.5).unwrap();
        assert_eq!((ps.len(), c.scored, c.restored), (7, 7, 0));
        assert!(c.tokens > 0);
    }

    #[test]
    fn the_cascade_scores_waves_until_a_wave_adds_nothing() {
        let chunks = synthetic(12);
        let mut eng = engine(4); // waves of three
        let order: Vec<usize> = (0..12).collect();
        let all = cold(&mut eng, &chunks, &order, Q);
        // recall order = best first: the first wave holds every hit there is
        let mut best: Vec<usize> = order.clone();
        best.sort_by(|&a, &b| all[b].total_cmp(&all[a]));
        let top_p = all[best[0]];
        assert!(
            all.iter().filter(|&&p| p == top_p).count() == 1,
            "a tie would add hits"
        );
        let go = |eng: &mut Engine, top: usize, threshold: f64| {
            rerank(eng, &mut None, &chunks, &best, Q, top, threshold)
                .unwrap()
                .0
                .len()
        };
        // one hit wanted: wave one finds it, wave two adds none, stop
        assert_eq!(go(&mut eng, 1, top_p), 6);
        // more hits wanted than exist: every wave is scored
        assert_eq!(go(&mut eng, 2, top_p), 12);
        // every candidate is a hit: waves keep adding
        assert_eq!(go(&mut eng, 1, 0.0), 12);
        // no hit at all: nothing has "enough"
        assert_eq!(go(&mut eng, 1, 1.1), 12);
    }

    #[test]
    fn a_stored_snapshot_is_trusted_only_when_it_is_whole() {
        let t = Tmp::new("trust");
        let chunks = synthetic(6);
        let order: Vec<usize> = (0..6).collect();
        let mut eng = engine(4);
        let probes: Vec<Probe> = order
            .iter()
            .map(|&i| probe_for(&eng, &chunks[i], Q).unwrap())
            .collect();
        let key = |j: usize| prefix_key(&probes[j].toks[..probes[j].at]);
        let want = cold(&mut eng, &chunks, &order, Q);
        let mut store = Store::open(&t.0, "b").unwrap();
        // a blob one token short (llama.cpp would accept it on a recurrent
        // memory and answer wrongly), garbage of the right length, a whole one
        let short = probes[1].at - 1;
        store
            .put(key(1), short, &blob(&probes[1].toks[..short]))
            .unwrap();
        store.put(key(2), probes[2].at, b"garbage").unwrap();
        store
            .put(key(3), probes[3].at, &blob(&probes[3].toks[..probes[3].at]))
            .unwrap();
        let mut store = Some(store);
        let (got, c) = rerank(&mut eng, &mut store, &chunks, &order, Q, usize::MAX, 0.5).unwrap();
        assert_eq!(got, want);
        // only the whole one was restored from the store
        assert_eq!(c.restored, 1);
        // the others are left as they were, never rewritten: three new records
        let s = store.as_ref().unwrap();
        assert_eq!(
            (s.tokens(key(1)), s.tokens(key(2))),
            (Some(short), Some(probes[2].at))
        );
        assert_eq!(s.len(), 6);
    }

    #[test]
    fn a_chunk_repeated_in_the_tree_is_snapshotted_once() {
        let t = Tmp::new("dup");
        let mut chunks = synthetic(6);
        chunks[5] = chunks[0].clone();
        let order: Vec<usize> = (0..6).collect();
        let mut eng = engine(3); // waves of two: the copy comes two waves after its twin
        let mut store = Some(Store::open(&t.0, "b").unwrap());
        let (ps, c) = rerank(&mut eng, &mut store, &chunks, &order, Q, usize::MAX, 0.5).unwrap();
        assert_eq!(ps[0], ps[5]);
        assert_eq!(store.as_ref().unwrap().len(), 5);
        // the twin's snapshot, stored after the reads were asked for, serves the copy
        assert_eq!(c.restored, 1);
    }

    // --- one search

    /// What `load` hands a search: a fresh engine (a new process) and a GGUF.
    fn model(t: &Tmp) -> impl FnOnce() -> Result<Loaded> {
        let gguf = t.write("m.gguf", "weights");
        move || Ok((engine(5), gguf))
    }

    fn query(q: &str, o: Opts) -> Search {
        Search {
            query: q.into(),
            out: Out::Human,
            lines: 20,
            opts: o,
        }
    }

    #[test]
    fn a_search_reranks_the_recall_and_the_second_one_starts_warm() {
        let t = Tmp::new("search");
        let root = project(&t, "tree");
        let s = query("connection pool checkout", opts(&root));
        let base = t.at("cache");
        let cold = search(&s, &base, model(&t)).unwrap();
        assert!(cold.stats.cost.scored > 0 && cold.stats.cost.restored == 0);
        assert!(cold.hits.len() <= 3 && !cold.hits.is_empty());
        // best probability first, every one a number
        let ps: Vec<f64> = cold.hits.iter().map(|h| h.1.unwrap()).collect();
        assert!(ps.windows(2).all(|w| w[0] >= w[1]));
        assert!(cold.stats.store_bytes.unwrap() > 0 && cold.stats.rerank_ms >= 0.0);

        let warm = search(&s, &base, model(&t)).unwrap();
        assert_eq!(warm.hits, cold.hits);
        assert_eq!(warm.stats.cost.restored, warm.stats.cost.scored);
        assert!(warm.stats.cost.tokens < cold.stats.cost.tokens);
        // the same tree without a store scores the same
        let bare = query(
            &s.query,
            Opts {
                store: false,
                ..opts(&root)
            },
        );
        let bare = search(&bare, &base, model(&t)).unwrap();
        assert_eq!((bare.hits, bare.stats.store_bytes), (cold.hits, None));
    }

    #[test]
    fn recall_only_and_an_empty_recall_load_no_model() {
        let t = Tmp::new("recall");
        let root = project(&t, "tree");
        let no_model = || -> Result<Loaded> { panic!("the model was loaded") };
        let only = Opts {
            recall_only: true,
            ..opts(&root)
        };
        let found = search(&query("connection pool", only), &t.at("cache"), no_model).unwrap();
        // the ranking, best first, no probabilities
        let (first, p) = found.hits[0];
        assert_eq!(
            (found.chunks[first].path.as_str(), p),
            ("src/pool.rs", None)
        );
        assert!(found.hits.len() <= 3 && found.stats.cost.scored == 0);
        // no word in common with anything: no candidates, no model
        let none = search(&query("zebra", opts(&root)), &t.at("cache"), no_model).unwrap();
        assert!(none.hits.is_empty());
        assert!(!t.at("cache").exists());
    }

    #[test]
    fn a_search_refuses_what_makes_no_sense() {
        let t = Tmp::new("check");
        let root = project(&t, "tree");
        let go = |o: Opts| {
            let r = search(&query("pool", o), &t.at("cache"), model(&t));
            r.err().map(|e| e.to_string())
        };
        assert!(go(Opts {
            top: 0,
            ..opts(&root)
        })
        .unwrap()
        .contains("--top"));
        assert!(go(Opts {
            threshold: 1.5,
            ..opts(&root)
        })
        .unwrap()
        .contains("--threshold"));
        assert!(go(Opts {
            threshold: f64::NAN,
            ..opts(&root)
        })
        .is_some());
        let bad = go(Opts {
            regexp: Some("(".into()),
            ..opts(&root)
        });
        assert!(bad.unwrap().contains("invalid pattern"));
    }

    // --- snap index

    #[test]
    fn indexing_snapshots_what_the_store_lacks_and_gc_drops_what_the_tree_lost() {
        let t = Tmp::new("index");
        let root = project(&t, "tree");
        let (base, tr) = (t.at("cache"), tree(&root));
        let gguf = t.write("m.gguf", "weights");
        let load = || -> Result<Loaded> { Ok((engine(5), gguf.clone())) };
        let n = load_chunks(&tr, None, &[]).unwrap().len();
        index_in(&base, &tr, false, false, load).unwrap();
        let held = || open_store(&base, &tr, &gguf, "snap-sim").unwrap().len();
        assert_eq!(held(), n);
        // nothing to add the second time, stats read without writing
        index_in(&base, &tr, false, false, load).unwrap();
        index_in(&base, &tr, true, false, load).unwrap();
        assert_eq!(held(), n);
        // asking after a tree nothing was stored for creates no store
        let fresh = t.at("elsewhere");
        index_in(&fresh, &tr, true, false, load).unwrap();
        index_in(&fresh, &tr, false, true, load).unwrap();
        assert!(!fresh.exists());
        // a file goes: its snapshots stay until gc, which drops them
        let gone = load_chunks(&tr, None, &["src/net.py".into()])
            .unwrap()
            .len();
        fs::remove_file(root.join("src/net.py")).unwrap();
        index_in(&base, &tr, false, false, load).unwrap();
        assert_eq!(held(), n);
        index_in(&base, &tr, false, true, load).unwrap();
        assert_eq!(held(), gone);
    }

    #[test]
    fn build_reports_its_progress_and_is_idempotent() {
        let t = Tmp::new("build");
        let chunks = synthetic(7);
        let mut eng = engine(4);
        let keys = chunk_keys(&eng, &chunks).unwrap();
        assert_eq!(keys.len(), 7);
        let mut store = Store::open(&t.0, "b").unwrap();
        let mut seen = Vec::new();
        let mut note = |d, n, tk| seen.push((d, n, tk));
        let (added, tokens) = build(&mut eng, &mut store, &chunks, &keys, &mut note).unwrap();
        assert_eq!((added, store.len()), (7, 7));
        assert!(tokens > 0);
        // waves of three: the last one short, the total fixed
        let counts: Vec<(usize, usize)> = seen.iter().map(|s| (s.0, s.1)).collect();
        assert_eq!(counts, [(3, 7), (6, 7), (7, 7)]);
        assert!(seen.windows(2).all(|w| w[0].2 <= w[1].2));
        let idle = &mut |_, _, _| panic!("nothing to do");
        assert_eq!(
            build(&mut eng, &mut store, &chunks, &keys, idle).unwrap(),
            (0, 0)
        );
        // every snapshot is the whole prefix the probe expects
        for (c, &(k, at)) in chunks.iter().zip(&keys) {
            assert_eq!(store.tokens(k), Some(at), "{}", c.path);
        }
    }

    // --- eval

    #[test]
    fn anchors_match_on_path_and_text() {
        let c = chunk("src/a.rs", 1, None, "fn alpha() {}");
        let a = |p: &str, s: &str| Anchor {
            path: p.into(),
            contains: s.into(),
        };
        assert!(a("src/a.rs", "fn alpha(").met_by(&c));
        assert!(!a("src/b.rs", "fn alpha(").met_by(&c));
        assert!(!a("src/a.rs", "fn beta(").met_by(&c));
        assert!(satisfies(&[a("x", "y"), a("src/a.rs", "alpha")], &c));
        assert!(!satisfies(&[], &c));
    }

    fn anchored() -> (Vec<Chunk>, Vec<Anchor>) {
        let chunks = vec![
            chunk("a.rs", 1, None, "alpha beta"),
            chunk("a.rs", 5, None, "gamma"),
            chunk("b.rs", 1, None, "alpha"),
        ];
        let anchor = |p: &str| Anchor {
            path: p.into(),
            contains: "alpha".into(),
        };
        (chunks, vec![anchor("a.rs"), anchor("b.rs")])
    }

    #[test]
    fn recall_at_k_is_the_share_of_anchors_some_chunk_in_the_top_k_meets() {
        let (chunks, anchors) = anchored();
        let r = |ranked: &[usize], k| anchor_recall(&chunks, ranked, &anchors, k);
        assert_eq!(r(&[2, 0, 1], 1), 0.5);
        assert_eq!(r(&[2, 0, 1], 2), 1.0);
        assert_eq!(r(&[1, 2, 0], 1), 0.0);
        // k past the list is the list
        assert_eq!(r(&[1], 64), 0.0);
        assert_eq!(r(&[], 5), 0.0);
        assert_eq!(r(&[0, 2], 64), 1.0);
    }

    #[test]
    fn the_reciprocal_rank_is_that_of_the_first_satisfying_chunk() {
        let (chunks, anchors) = anchored();
        let rr = |ranked: &[usize]| reciprocal_rank(&chunks, ranked, &anchors);
        assert_eq!(rr(&[2, 1]), 1.0);
        assert_eq!(rr(&[1, 0]), 0.5);
        assert_eq!(rr(&[1, 1, 2]), 1.0 / 3.0);
        assert_eq!(rr(&[1]), 0.0);
        assert_eq!(rr(&[]), 0.0);
    }

    #[test]
    fn percentiles_are_nearest_rank() {
        let v: Vec<f64> = (1..=10).map(f64::from).collect();
        assert_eq!(
            (
                percentile(&v, 0.5),
                percentile(&v, 0.9),
                percentile(&v, 1.0)
            ),
            (5.0, 9.0, 10.0)
        );
        assert_eq!(percentile(&[7.0], 0.9), 7.0);
        assert_eq!(percentile(&[], 0.5), 0.0);
        assert_eq!(mean([1.0, 2.0, 6.0].into_iter()), 3.0);
        assert_eq!(mean(std::iter::empty()), 0.0);
        assert_eq!((share(1, 4), share(0, 0)), (0.25, 0.0));
    }

    #[test]
    fn the_rerank_metrics_read_the_models_order() {
        let (chunks, anchors) = anchored();
        // chunk 1 (no anchor) outscores 2 and 0, which satisfy one each
        let m = Reranked::of(&chunks, &anchors, &[0, 1, 2], &[0.2, 0.9, 0.6], 0.5);
        assert!(!m.hit1);
        assert_eq!((m.r5, m.r10, m.rr), (1.0, 1.0, 0.5));
        assert!(m.answered);
        assert_eq!((m.sat, m.non), (vec![0.6, 0.2], vec![0.9]));
        // nothing scored reaches the threshold; the best one hits
        let m = Reranked::of(&chunks, &anchors, &[0, 1], &[0.4, 0.1], 0.5);
        assert!(m.hit1 && !m.answered);
        assert_eq!((m.r5, m.rr), (0.5, 1.0));
        // nothing scored at all
        let m = Reranked::of(&chunks, &anchors, &[], &[], 0.5);
        assert!(!m.hit1 && !m.answered && (m.r5, m.rr) == (0.0, 0.0));
    }

    fn row(kind: &str, recall: [f64; 4], reranked: Option<Reranked>, ms: f64) -> Row {
        Row {
            id: "id".into(),
            kind: kind.into(),
            query: "q".into(),
            recall,
            first: None,
            reranked,
            top: Vec::new(),
            recall_ms: ms,
            rerank_ms: 0.0,
        }
    }

    #[test]
    fn summaries_average_the_cases() {
        let rr = |hit1, rr, answered, sat: &[f64], non: &[f64]| Reranked {
            hit1,
            r5: rr,
            r10: rr,
            rr,
            answered,
            sat: sat.to_vec(),
            non: non.to_vec(),
        };
        let a = row(
            "x",
            [1.0, 1.0, 1.0, 1.0],
            Some(rr(true, 1.0, true, &[0.9], &[0.3, 0.1])),
            10.0,
        );
        let b = row(
            "x",
            [0.0, 0.5, 0.5, 1.0],
            Some(rr(false, 0.5, false, &[0.5], &[])),
            30.0,
        );
        let c = row(
            "x",
            [0.0, 0.0, 0.0, 0.5],
            Some(rr(false, 0.0, true, &[], &[0.2])),
            20.0,
        );
        let s = Summary::of(&[&a, &b, &c]);
        assert_eq!(s.n, 3);
        assert_eq!(s.recall, [1.0 / 3.0, 0.5, 0.5, 2.5 / 3.0]);
        assert_eq!(s.all64, 2.0 / 3.0);
        let r = s.rerank.unwrap();
        assert_eq!((r.hit1, r.mrr, r.answered), (1.0 / 3.0, 0.5, 2.0 / 3.0));
        assert_eq!(r.p_sat, Some(0.7));
        assert!((r.p_non.unwrap() - 0.2).abs() < 1e-12);
        assert_eq!((s.median_ms, s.p90_ms), (20.0, 30.0));
        // recall-only rows have no rerank block, and an empty set no means
        let only = Summary::of(&[&row("x", [0.0; 4], None, 1.0)]);
        assert!(only.rerank.is_none() && only.json()["rerank"].is_null());
        assert_eq!(Summary::of(&[]).n, 0);
    }

    const CASES: &str = r#"{"id": "pool-1", "query": "connection pool checkout", "kind": "lexical", "expect": [{"path": "src/pool.rs", "contains": "pub fn acquire("}]}
# a comment line
{"id": "para-1", "query": "lend a database handle", "kind": "paraphrase", "expect": [{"path": "src/pool.rs", "contains": "pub fn acquire("}, {"path": "src/net.py", "contains": "def dial"}]}
{"id": "stale-1", "query": "zebra", "expect": [{"path": "src/pool.rs", "contains": "no such text"}]}
"#;

    /// A tree whose cases file sits inside it, every query verbatim.
    fn eval_tree(t: &Tmp) -> PathBuf {
        let root = project(t, "tree");
        t.write("tree/eval/cases.jsonl", CASES);
        root
    }

    #[test]
    fn eval_reports_recall_without_a_model_and_leaves_its_own_files_out() {
        let t = Tmp::new("eval");
        let root = eval_tree(&t);
        let o = Opts {
            recall_only: true,
            ..opts(&root)
        };
        let (file, out) = (root.join("eval/cases.jsonl"), t.at("report.json"));
        let (file, out) = (file.to_str().unwrap(), out.to_str().unwrap());
        let no_model = || -> Result<Loaded> { panic!("the model was loaded") };
        eval(file, Some(out), &o, no_model).unwrap();
        let rep: Value = serde_json::from_str(&fs::read_to_string(out).unwrap()).unwrap();
        // the cases file would answer every query: it is not in the corpus
        let corpus = load_chunks(&o.tree, None, &["eval/cases.jsonl".into()])
            .unwrap()
            .len();
        assert_eq!(rep["chunks"], corpus);
        assert!(load_chunks(&o.tree, None, &[]).unwrap().len() > corpus);
        assert_eq!(
            (rep["kind"].as_str(), rep["model"].is_null()),
            (Some("snap-grep-eval"), true)
        );
        assert_eq!(rep["overall"]["cases"], 3);
        assert!(rep["overall"]["rerank"].is_null());
        // recall@k reads 1, 5, 10, 64 in the file, not in the order of the strings
        let ks = rep["overall"]["recall"].as_object().unwrap();
        assert_eq!(ks.keys().collect::<Vec<_>>(), ["1", "5", "10", "64"]);
        // one group per kind, a case without a kind is `other`
        let kinds: Vec<&str> = rep["by_kind"]
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(kinds, ["lexical", "other", "paraphrase"]);
        // the lexical case finds its chunk first; the stale anchor never can
        let cases = rep["cases"].as_array().unwrap();
        assert_eq!(cases[0]["recall"]["1"], 1.0);
        assert_eq!(cases[0]["first_satisfying_rank"], 1);
        assert_eq!(cases[2]["recall"]["64"], 0.0);
        assert!(cases[2]["first_satisfying_rank"].is_null());
        // no word of the paraphrase is in the code: only its anchors can say so
        assert_eq!(cases[1]["recall"]["64"], 0.0);
        // reports are create-only
        let again = eval(file, Some(out), &o, no_model).unwrap_err();
        assert!(again.to_string().contains("exists"), "{again}");
    }

    #[test]
    fn eval_with_a_model_adds_the_rerank_stage() {
        let t = Tmp::new("eval-model");
        let root = eval_tree(&t);
        let file = root.join("eval/cases.jsonl");
        let o = opts(&root);
        let rep = run_eval(file.to_str().unwrap(), None, &t.at("cache"), &o, model(&t)).unwrap();
        assert_eq!(rep["model"], "snap-sim");
        let r = &rep["overall"]["rerank"];
        for k in ["hit_at_1", "recall_at_5", "recall_at_10", "mrr", "answered"] {
            let v = r[k].as_f64().unwrap_or(f64::NAN);
            assert!((0.0..=1.0).contains(&v), "{k} = {v}");
        }
        let lat = &rep["overall"]["latency_ms"];
        assert!(lat["p90"].as_f64().unwrap() >= lat["median"].as_f64().unwrap());
        let first = &rep["cases"][0];
        assert!(first["top"].as_array().unwrap().len() <= 10);
        assert!(first["top"][0]["p"].as_f64().is_some());
        assert!(t.at("cache").exists());
    }

    #[test]
    fn eval_refuses_cases_it_cannot_score() {
        let t = Tmp::new("eval-bad");
        let root = project(&t, "tree");
        let o = Opts {
            recall_only: true,
            ..opts(&root)
        };
        let run = |text: &str| {
            let f = t.write("cases.jsonl", text);
            let r = run_eval(f.to_str().unwrap(), None, &t.at("cache"), &o, || {
                panic!("no model")
            });
            format!("{:#}", r.unwrap_err())
        };
        assert!(run(r#"{"id": "a", "query": "q", "expect": []}"#).contains("no anchors"));
        assert!(run("").contains("no cases"));
        assert!(
            run(r#"{"id": "a", "expect": [{"path": "p", "contains": "c"}]}"#).contains("cases in")
        );
    }

    #[test]
    fn the_eval_table_has_a_row_per_kind_and_an_overall() {
        let a = row("lexical", [1.0, 1.0, 1.0, 1.0], None, 10.0);
        let b = row("concept", [0.0, 0.0, 0.5, 1.0], None, 30.0);
        let groups = vec![
            ("concept".to_string(), Summary::of(&[&b])),
            ("lexical".to_string(), Summary::of(&[&a])),
            ("overall".to_string(), Summary::of(&[&a, &b])),
        ];
        let text = table(&groups, None, 0.5);
        let lines: Vec<&str> = text.lines().collect();
        assert!(lines[0].starts_with("recall"));
        assert!(lines[1]
            .split_whitespace()
            .eq(["kind", "n", "@1", "@5", "@10", "@64", "all@64"]));
        assert!(lines[2]
            .split_whitespace()
            .eq(["concept", "1", "0.00", "0.00", "0.50", "1.00", "1.00"]));
        assert!(lines[4].starts_with("overall") && lines[4].contains("0.50"));
        assert!(text.contains("median 10 ms, p90 30 ms"), "{text}");
    }
}
