//! `snap grep`: find the code that answers a question. Recall has two sources
//! that always both run: BM25 ranks every chunk of the tree lexically — no
//! model, milliseconds — and the model descends the tree, judging folders,
//! then files, then groups of a big file's chunks with one yes/no letter
//! (`tree.rs`), which reaches what shares no word with the question. The model
//! then reads the union of their candidates once each, answering the same
//! letter, whose probability is the score. The expensive half of that read
//! does not depend on the query, so it is kept: the first time a chunk is a
//! candidate its `[head + chunk]` memory is snapshotted into a store on disk,
//! and every later query restores it and decodes only its own question.
//!
//! The store is bound to everything that shapes that memory (snap version,
//! model file, prompt format, KV type) and lives in one directory per tree:
//! another binding opens another directory, never foreign memory. A snapshot
//! is trusted only when it holds exactly the tokens the probe expects, and one
//! that llama.cpp refuses is scored from scratch instead. The cache fills only
//! as searches read candidates: a whole tree would be hundreds of GB.
//!
//! This module is the pipeline and its reports (`--eval`, `--cache`, `--gc`); the
//! walk is in `corpus.rs`, lexical recall in `lexical.rs`, the descent in
//! `tree.rs`, the store in `kvstore.rs`, the prompt and the scoring in
//! `model.rs`, the printing in `view.rs` and the flags in `cli.rs`.
//!
//! What `snap grep` takes from the rest of snap, and nothing else:
//! `engine::{Engine, Reader}` (the model, with its `kv`, `llm`, `letter_ids`,
//! `state_format`, `calibration` and `model_id`, and `probe_prompt`, which
//! compiles a one-question request as `decide` does),
//! `kv::{Kv, Backend, Job, Restore, Room, Stats}` for the whole-seq snapshot
//! calls, `llamac::KvType`, `prompts::PROMPT_VERSION`,
//! `schema::{DecideRequest, Expand, Layout, Mode}`,
//! `evaluate::{create_only, load_cases}`, and from `main.rs` `ModelArgs`,
//! `init_logs` and `fmt_bytes`. Tests also use
//! `kv::{sim, Dec, DecodeError, KV_SLACK}`, `calibrate::Calibration` and
//! `prompts`.

pub mod cli;
mod corpus;
mod kvstore;
mod lexical;
mod model;
mod tree;
mod view;

use std::collections::{BTreeMap, HashSet};
use std::io::{IsTerminal, Write};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::time::{Instant, UNIX_EPOCH};

use anyhow::{bail, ensure, Context, Result};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::engine::Engine;
use crate::evaluate;
use crate::kv::Room;
use crate::llamac::KvType;
use crate::prompts::PROMPT_VERSION;
use corpus::{Chunk, WalkOpts};
use kvstore::{Prefetch, Store};
use lexical::{rrf, Doc, Index};
use model::{grep_state, Probe, GREP_FORMAT};
use view::{files, human, json_doc, thousands, Hit, Look, Shown, DIM, JSON_LINES, SNIPPET};

/// Reciprocal rank fusion's damping: how fast a rank's vote falls off.
const RRF_K: f64 = 60.0;
/// Chunks the descent adds to BM25's list at most, interleaved with it: the
/// cascade stops after a wave, and its picks must not sit behind 64 lexical ones.
const TREE_CHUNKS: usize = 24;
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
    /// BM25 depth reranked, the descent's picks coming on top; 0 = every chunk
    pub candidates: usize,
    /// least P(yes) of a hit
    pub threshold: f64,
    pub store: bool,
    pub recall_only: bool,
    /// branches opened per level and kind by the descent
    pub beam: usize,
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
    /// source lines printed per hit; 0 = locations only, None = SNIPPET
    /// (JSON_LINES in --json)
    pub lines: Option<usize>,
    /// ANSI colors: forced on or off, or None for when stdout is a terminal
    pub color: Option<bool>,
    pub opts: Opts,
}

/// The model, loaded on demand (lexical recall needs none), and the GGUF it came
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
        // Docs, specs, mockups and schemas describe a feature in the very
        // words a question uses, and outrank the code that implements it:
        // over a whole repo they took 60 of 64 candidates. Code and the rest
        // alternate, each keeping half the list while both have chunks left.
        let (docs, code): (Vec<usize>, Vec<usize>) = order
            .into_iter()
            .partition(|&i| corpus::is_doc(&self.chunks[i].path));
        order = Vec::with_capacity(code.len() + docs.len());
        for j in 0..code.len().max(docs.len()) {
            order.extend(code.get(j));
            order.extend(docs.get(j));
        }
        if candidates == 0 {
            let held: HashSet<usize> = order.iter().copied().collect();
            order.extend((0..n).filter(|i| !held.contains(i)));
        } else {
            order.truncate(candidates);
        }
        order
    }

    /// The candidates the model reads: BM25's `lexical` list united with the
    /// chunks the model's descent reaches, and how many previews it read on
    /// the way. The previews go through the same probe as chunks.
    fn reach(
        &self,
        eng: &mut Engine,
        store: &mut Option<Store>,
        query: &str,
        o: &Opts,
        lexical: Vec<usize>,
    ) -> Result<(Vec<usize>, usize)> {
        // with every chunk a BM25 candidate the descent has nothing to add
        if o.candidates == 0 {
            return Ok((lexical, 0));
        }
        let mut previews = 0;
        let tree = tree::descend(&self.chunks, &self.files, o.beam, |ps| {
            previews += ps.len();
            let all: Vec<usize> = (0..ps.len()).collect();
            rerank(eng, store, ps, &all, query, None).map(|(p, _)| p)
        })?;
        Ok((union(lexical, &tree), previews))
    }
}

/// BM25's candidates with up to `TREE_CHUNKS` of the descent's that it did not
/// pick, in the descent's order: BM25's best first, then one of each in turn
/// while both last, then what is left of either.
fn union(lexical: Vec<usize>, tree: &[usize]) -> Vec<usize> {
    let held: HashSet<usize> = lexical.iter().copied().collect();
    let mut seen = HashSet::new();
    let extra: Vec<usize> = tree
        .iter()
        .copied()
        .filter(|i| !held.contains(i) && seen.insert(*i))
        .take(TREE_CHUNKS)
        .collect();
    let mut out = Vec::with_capacity(lexical.len() + extra.len());
    for j in 0..lexical.len().max(extra.len()) {
        out.extend(lexical.get(j));
        out.extend(extra.get(j));
    }
    out
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

/// What reranking took: candidates scored in how many waves, how many of
/// those were restored from a stored snapshot (the rest had their chunk
/// decoded in this run), and the tokens the model decoded.
#[derive(Default, Clone, Copy)]
struct Cost {
    scored: usize,
    waves: usize,
    restored: usize,
    tokens: usize,
}

/// Stop after a wave when enough hits are in and the wave added none: what
/// recall ranked lower has not been reading better.
fn done(total_hits: usize, wave_hits: usize, top: usize) -> bool {
    total_hits >= top && wave_hits == 0
}

/// A search's cascade: it stops once `top` candidates have P >= `threshold`
/// and a wave added none. An eval has none: it reads every candidate.
#[derive(Clone, Copy)]
struct Cascade {
    top: usize,
    threshold: f64,
}

/// The waves of a run: consecutive candidates, in order, as many as one
/// restore wave takes — a free seq each and cells for each whole prompt,
/// restored cells being private. That is also what sits in memory as blobs,
/// so it is what bounds them: about one KV memory's worth of bytes, never the
/// n_seq - 1 blobs a count alone allows. At ctx 8192 a wave of snap1-2b's
/// 460-token probes is 17 candidates (0.34 GB in f16) instead of 64 (1.3 GB);
/// with the prefetch queue a wave ahead, about two waves of snapshots are in
/// RAM at once. A probe that fits nowhere is still a wave of its own: the
/// decode refuses it with its own error.
fn waves(room: Room, lens: &[usize]) -> Vec<Range<usize>> {
    let mut out = Vec::new();
    let mut at = 0;
    while at < lens.len() {
        let n = room.fit(lens[at..].iter().copied()).max(1);
        out.push(at..at + n);
        at += n;
    }
    out
}

/// P(yes) of the candidates `order` names, a wave at a time in recall order,
/// until the cascade stops (never, without one). The result holds one P per
/// candidate scored: a prefix of `order`.
fn rerank(
    eng: &mut Engine,
    store: &mut Option<Store>,
    chunks: &[Chunk],
    order: &[usize],
    query: &str,
    stop: Option<Cascade>,
) -> Result<(Vec<f64>, Cost)> {
    let probes = order
        .iter()
        .map(|&i| probe_for(eng, &chunks[i], query))
        .collect::<Result<Vec<Probe>>>()?;
    let keys: Vec<u128> = probes.iter().map(|p| prefix_key(&p.toks[..p.at])).collect();
    let lens: Vec<usize> = probes.iter().map(|p| p.toks.len()).collect();
    let plan = waves(eng.grep_room(), &lens);
    // every candidate's snapshot is asked for once, as many ahead as the first
    // wave holds: the reads of a wave overlap the decode of the one before
    let depth = plan.first().map_or(1, Range::len);
    let mut fetch = store.as_ref().map(|s| s.prefetch(keys.clone(), depth));
    let mut cost = Cost::default();
    let (mut ps, mut hits) = (Vec::with_capacity(probes.len()), 0);
    // a search on a terminal shows how far the reading is: it takes seconds
    let (tty, t0) = (
        stop.is_some() && std::io::stderr().is_terminal(),
        Instant::now(),
    );
    let show = |n: usize| {
        if tty {
            let secs = t0.elapsed().as_secs_f64();
            eprint!(
                "\r\x1b[2K\x1b[2m… reading candidates · {n}/{} · {secs:.0}s\x1b[0m",
                probes.len()
            );
        }
    };
    show(0);
    for w in plan {
        let (probes, keys) = (&probes[w.clone()], &keys[w]);
        let got = score_wave(eng, store, &mut fetch, probes, keys, &mut cost)?;
        cost.waves += 1;
        let added = stop.map_or(0, |c| got.iter().filter(|&&p| p >= c.threshold).count());
        hits += added;
        ps.extend(got);
        show(ps.len());
        if stop.is_some_and(|c| done(hits, added, c.top)) {
            break;
        }
    }
    if tty {
        eprint!("\r\x1b[2K");
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

/// Hits a search shows, however few reach the threshold.
const MIN_SHOWN: usize = 3;
/// The least P of a hit shown below the threshold. snap1-2b's P(yes) runs
/// low on code: on eval/grep.jsonl the chunks that answer average 0.64, the
/// rest 0.20, and a 0.47 can be the very answer.
const FLOOR: f64 = 0.2;

/// What a search shows: the hits at or above the threshold, at most `top`;
/// when they are fewer than MIN_SHOWN, the best of the rest down to FLOOR
/// make up the number. Also how many of them reached the threshold.
fn shown(order: &[usize], ps: &[f64], top: usize, threshold: f64) -> (Vec<(usize, f64)>, usize) {
    let hits = pick(order, ps, top, threshold);
    let (n, few) = (hits.len(), MIN_SHOWN.min(top));
    if n >= few {
        return (hits, n);
    }
    (pick(order, ps, few, FLOOR.min(threshold)), n)
}

// --- one search -------------------------------------------------------------

#[derive(Default)]
struct Stats {
    chunks: usize,
    /// the candidates handed to the model: BM25's and the descent's
    candidates: usize,
    cost: Cost,
    store_bytes: Option<u64>,
    recall_ms: f64,
    /// previews the model read in the descent, and its time
    tree_previews: usize,
    tree_ms: f64,
    rerank_ms: f64,
    total_ms: f64,
    hits: usize,
}

fn ms(t: Instant) -> f64 {
    (t.elapsed().as_secs_f64() * 1e4).round() / 10.0
}

impl Stats {
    fn json(&self) -> Value {
        let c = &self.cost;
        json!({
            "chunks": self.chunks,
            "candidates": self.candidates,
            "scored": c.scored,
            "waves": c.waves,
            "restored": c.restored,
            "decoded": c.scored - c.restored,
            "tokens": c.tokens,
            "store_bytes": self.store_bytes,
            "recall_ms": self.recall_ms,
            "tree_previews": self.tree_previews,
            "tree_ms": self.tree_ms,
            "rerank_ms": self.rerank_ms,
            "total_ms": self.total_ms,
            "hits": self.hits,
        })
    }

    /// The one line `snap grep` always leaves on stderr:
    /// `10 results · 12.6 s · 1,146 chunks · 37 read by the model, 30 from cache · cache 583 MB`.
    fn summary(&self, recall_only: bool) -> String {
        let c = &self.cost;
        let mut p = vec![
            match self.hits {
                1 => "1 result".to_string(),
                n => format!("{n} results"),
            },
            match self.total_ms {
                ms if ms < 1000.0 => format!("{ms:.0} ms"),
                ms => format!("{:.1} s", ms / 1000.0),
            },
            format!("{} chunks", thousands(self.chunks)),
        ];
        if recall_only {
            p.push("lexical ranking only".into());
        } else {
            let mut read = format!("{} read by the model", c.scored);
            if c.restored > 0 {
                read += &format!(", {} from cache", c.restored);
            }
            p.push(read);
            p.push(format!(
                "tree {} previews, {:.1} s",
                self.tree_previews,
                self.tree_ms / 1000.0
            ));
            p.extend(
                self.store_bytes
                    .map(|b| format!("cache {}", crate::fmt_bytes(b))),
            );
        }
        p.join(" · ")
    }
}

/// What a search found: the hits as (chunk, P) — P is None when only recall
/// ran — over the chunks they index.
struct Found {
    chunks: Vec<Chunk>,
    hits: Vec<(usize, Option<f64>)>,
    /// how many hits reached the threshold; the rest only make up MIN_SHOWN
    confident: usize,
    /// with no hit, the candidate that came nearest the floor
    closest: Option<(usize, f64)>,
    stats: Stats,
}

fn search(s: &Search, base: &Path, load: impl FnOnce() -> Result<Loaded>) -> Result<Found> {
    let (o, t0) = (&s.opts, Instant::now());
    o.check()?;
    let hay = Haystack::new(load_chunks(&o.tree, o.regexp.as_deref(), &[])?);
    let lexical = hay.recall(&s.query, o.candidates);
    let mut st = Stats {
        chunks: hay.chunks.len(),
        candidates: lexical.len(),
        recall_ms: ms(t0),
        ..Stats::default()
    };
    let (mut closest, mut confident) = (None, 0);
    let hits = if o.recall_only {
        lexical.iter().take(o.top).map(|&i| (i, None)).collect()
    } else if hay.chunks.is_empty() {
        Vec::new()
    } else {
        let (mut eng, gguf) = load()?;
        let mut store = try_store(base, o, &gguf, &eng.model_id);
        let t = Instant::now();
        let (order, previews) = hay.reach(&mut eng, &mut store, &s.query, o, lexical)?;
        (st.candidates, st.tree_previews, st.tree_ms) = (order.len(), previews, ms(t));
        let t = Instant::now();
        let stop = Cascade {
            top: o.top,
            threshold: o.threshold,
        };
        let (ps, cost) = rerank(
            &mut eng,
            &mut store,
            &hay.chunks,
            &order,
            &s.query,
            Some(stop),
        )?;
        st.rerank_ms = ms(t);
        st.cost = cost;
        st.store_bytes = store.as_ref().map(Store::bytes);
        let hits;
        (hits, confident) = shown(&order, &ps, o.top, o.threshold);
        if hits.is_empty() {
            closest = pick(&order, &ps, 1, 0.0).pop();
        }
        hits.into_iter().map(|(i, p)| (i, Some(p))).collect()
    };
    st.hits = hits.len();
    st.total_ms = ms(t0);
    Ok(Found {
        chunks: hay.chunks,
        hits,
        confident,
        closest,
        stats: st,
    })
}

/// `snap grep`: search, print the hits on stdout and the summary on stderr.
/// False when nothing was found, the exit status grep gives it.
pub fn run(s: &Search, load: impl FnOnce() -> Result<Loaded>) -> Result<bool> {
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
            weak: p.is_some_and(|p| p < s.opts.threshold),
            rank: rank + 1,
        })
        .collect();
    let tty = std::io::stderr().is_terminal();
    let err = Look::of(tty, s.color);
    let dot = err.paint(DIM, " · ");
    let pct = s.opts.threshold * 100.0;
    if f.confident == 0 && !hits.is_empty() && hits[0].weak {
        eprintln!(
            "{} nothing reaches {pct:.0}%{dot}the closest:",
            err.paint(DIM, "◇")
        );
        if tty && s.out == Out::Human {
            eprintln!();
        }
    }
    let text = match s.out {
        Out::Human => {
            let look = Look::of(std::io::stdout().is_terminal(), s.color);
            human(&hits, s.lines.unwrap_or(SNIPPET), &s.query, look)
        }
        Out::Files => files(&hits),
        Out::Json => {
            let lines = s.lines.unwrap_or(JSON_LINES);
            let doc = json_doc(&s.query, &hits, lines, f.stats.json());
            format!("{}\n", serde_json::to_string_pretty(&doc)?)
        }
    };
    match std::io::stdout().lock().write_all(text.as_bytes()) {
        // `| head` closing the pipe is not a failure of the search
        Err(e) if e.kind() == std::io::ErrorKind::BrokenPipe => {}
        r => r?,
    }
    if let Some((i, p)) = f.closest {
        let c = &f.chunks[i];
        eprintln!(
            "{} nothing reaches {:.0}%{dot}closest {}:{}-{} at {:.0}%{dot}--threshold {:.1} shows it",
            err.paint(DIM, "◇"),
            FLOOR.min(s.opts.threshold) * 100.0,
            shown.path(&c.path),
            c.start,
            c.end,
            p * 100.0,
            (p * 10.0).floor() / 10.0
        );
    }
    if tty && s.out == Out::Human && !text.is_empty() {
        eprintln!();
    }
    eprintln!("{}", err.paint(DIM, &f.stats.summary(s.opts.recall_only)));
    Ok(!f.hits.is_empty())
}

// --- snap grep --cache, --gc -------------------------------------------------

/// A chunk's probe for no query in particular: its prefix is the same for all.
fn probe_of(eng: &Engine, c: &Chunk) -> Result<Probe> {
    probe_for(eng, c, "")
}

/// What the store is asked for a chunk, whichever query comes: the key of its
/// snapshot and the tokens that snapshot holds.
#[derive(Clone, Copy)]
struct Prefix {
    key: u128,
    at: usize,
}

fn chunk_prefixes(eng: &Engine, chunks: &[Chunk]) -> Result<Vec<Prefix>> {
    chunks
        .iter()
        .map(|c| {
            let p = probe_of(eng, c)?;
            Ok(Prefix {
                key: prefix_key(&p.toks[..p.at]),
                at: p.at,
            })
        })
        .collect()
}

/// `1 snapshot`, `2 snapshots`
fn plural(n: usize, noun: &str) -> String {
    format!("{} {noun}{}", thousands(n), if n == 1 { "" } else { "s" })
}

/// The store's place as it can be pasted into `rm -rf`: whole, `$HOME` as `~`.
fn shown_dir(dir: &Path, home: Option<&Path>) -> String {
    match home.and_then(|h| dir.strip_prefix(h).ok()) {
        Some(rest) => Path::new("~").join(rest).display().to_string(),
        None => dir.display().to_string(),
    }
}

/// `snap grep --cache` and `--gc`: what the store holds of the tree, and
/// what it drops: the snapshots no chunk uses any more and the stores of this
/// tree that other engines wrote, which nothing would open again. The gc line
/// comes first, then the report.
pub fn cache(
    t: &Tree,
    report: bool,
    gc: bool,
    load: impl FnOnce() -> Result<Loaded>,
) -> Result<()> {
    let home = std::env::var_os("HOME").map(PathBuf::from);
    for line in cache_in(&default_base(), t, report, gc, home.as_deref(), load)? {
        println!("{line}");
    }
    Ok(())
}

fn cache_in(
    base: &Path,
    t: &Tree,
    report: bool,
    gc: bool,
    home: Option<&Path>,
    load: impl FnOnce() -> Result<Loaded>,
) -> Result<Vec<String>> {
    let chunks = load_chunks(t, None, &[])?;
    let (eng, gguf) = load()?;
    let (dir, binding) = store_spec(base, t, &gguf, &eng.model_id)?;
    // reading what is not there must not create it
    let mut store = dir
        .exists()
        .then(|| Store::open(&dir, &binding))
        .transpose()?;
    let prefixes = match store {
        Some(_) => chunk_prefixes(&eng, &chunks)?,
        None => Vec::new(),
    };
    let live: HashSet<u128> = prefixes.iter().map(|p| p.key).collect();
    let mut lines = Vec::new();
    if gc {
        let mut line = match store.as_mut() {
            Some(s) => {
                let (n, bytes) = (s.len(), s.bytes());
                s.compact(&live)?;
                format!(
                    "gc: {n} → {} snapshots, {} → {}",
                    s.len(),
                    crate::fmt_bytes(bytes),
                    crate::fmt_bytes(s.bytes())
                )
            }
            None => "gc: no snapshots".into(),
        };
        let (mut gone, mut freed) = (0, 0);
        for other in kvstore::siblings(&dir) {
            if let Some(bytes) = kvstore::discard(&other)? {
                (gone, freed) = (gone + 1, freed + bytes);
            }
        }
        if gone > 0 {
            line += &format!(
                "; {} of other engines removed ({})",
                plural(gone, "store"),
                crate::fmt_bytes(freed)
            );
        }
        lines.push(line);
    }
    if report {
        let place = shown_dir(&dir, home);
        let total = chunks.len();
        let Some(s) = &store else {
            lines.push(format!(
                "cache {place} · no snapshots yet · 0 of {} chunks (0%)",
                thousands(total)
            ));
            return Ok(lines);
        };
        let covered = prefixes
            .iter()
            .filter(|p| s.tokens(p.key) == Some(p.at))
            .count();
        let used = live.iter().filter(|&&k| s.tokens(k).is_some()).count();
        lines.push(format!(
            "cache {place} · {} · {} · {covered} of {} chunks ({:.0}%) · {} stale",
            plural(s.len(), "snapshot"),
            crate::fmt_bytes(s.bytes()),
            thousands(total),
            100.0 * covered as f64 / total.max(1) as f64,
            s.len() - used
        ));
    }
    Ok(lines)
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
    /// anchor recall@k of the recall order (BM25 united with the descent's
    /// picks when the model is loaded), for the ks in `KS`
    recall: [f64; 4],
    /// the recall list, and how many of it the model scored: all of it, the
    /// eval having no cascade (None when only recall ran)
    candidates: usize,
    scored: Option<usize>,
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
        let mut order = hay.recall(&c.query, o.candidates);
        if let Some(e) = eng.as_mut() {
            order = hay.reach(e, &mut store, &c.query, o, order)?.0;
        }
        let recall_ms = ms(t);
        let t = Instant::now();
        // no cascade: hit@k and MRR are measured over every candidate, whatever
        // the size of the waves they are read in
        let scored = match eng.as_mut() {
            Some(e) if !order.is_empty() => {
                Some(rerank(e, &mut store, &hay.chunks, &order, &c.query, None)?.0)
            }
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
            candidates: order.len(),
            scored: scored.as_ref().map(Vec::len),
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
                "candidates": r.candidates,
                "scored": r.scored,
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
        "beam": o.beam,
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
    pub(super) struct Tmp(PathBuf);

    impl Tmp {
        pub(super) fn new(name: &str) -> Tmp {
            let dir = std::env::temp_dir().join(format!("snap-grep-{}-{name}", std::process::id()));
            let _ = fs::remove_dir_all(&dir);
            fs::create_dir_all(&dir).unwrap();
            Tmp(dir)
        }

        fn at(&self, rel: &str) -> PathBuf {
            self.0.join(rel)
        }

        pub(super) fn write(&self, rel: &str, text: &str) -> PathBuf {
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

    pub(super) fn chunk(path: &str, start: usize, symbol: Option<&str>, text: &str) -> Chunk {
        Chunk {
            path: path.into(),
            start,
            end: start + text.lines().count().max(1) - 1,
            symbol: symbol.map(Into::into),
            text: text.into(),
        }
    }

    /// An engine on the sim with `n_seq` seqs and a context no wave of these
    /// tests fills: waves of `n_seq - 1`.
    fn engine(n_seq: usize) -> Engine {
        engine_ctx(1 << 16, n_seq)
    }

    /// The same on a context of `n_ctx` cells, where waves are cell-bound.
    fn engine_ctx(n_ctx: usize, n_seq: usize) -> Engine {
        Engine::new(Box::new(Sim::new(n_ctx, n_seq, 256)), "snap-sim".into()).unwrap()
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
            beam: 3,
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

    #[test]
    fn docs_and_code_take_turns_while_both_last() {
        // the docs say "connection pool" more often than the code does
        let doc = |p: &str, l| {
            chunk(
                p,
                l,
                None,
                "the connection pool: a pool of connections, pooled",
            )
        };
        let hay = Haystack::new(vec![
            doc("DESIGN.md", 1),
            doc("DESIGN.md", 9),
            doc("api.yaml", 1),
            chunk(
                "src/pool.rs",
                1,
                Some("Pool"),
                "struct Pool { conns: Vec<Conn> }",
            ),
            chunk(
                "src/pool.rs",
                9,
                Some("checkout"),
                "fn checkout(&self) -> Conn",
            ),
        ]);
        let got: Vec<&str> = hay
            .recall("connection pool", 64)
            .iter()
            .map(|&i| hay.chunks[i].path.as_str())
            .collect();
        assert_eq!(
            got,
            [
                "src/pool.rs",
                "DESIGN.md",
                "src/pool.rs",
                "DESIGN.md",
                "api.yaml"
            ]
        );
        assert!(
            corpus::is_doc("a/B.HTML") && corpus::is_doc("x.md") && !corpus::is_doc("src/a.tsx")
        );
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

    #[test]
    fn fewer_than_three_hits_are_made_up_from_the_best_down_to_the_floor() {
        let order = [7, 3, 9, 1, 4];
        let ps = [0.47, 0.9, 0.3, 0.1, 0.25];
        // one reaches 0.5: the next two best above 0.2 join it
        assert_eq!(
            shown(&order, &ps, 10, 0.5),
            (vec![(3, 0.9), (7, 0.47), (9, 0.3)], 1)
        );
        // none reaches 0.95: the best three above the floor, 0.1 never
        assert_eq!(
            shown(&order, &ps, 10, 0.95),
            (vec![(3, 0.9), (7, 0.47), (9, 0.3)], 0)
        );
        assert_eq!(
            shown(&order, &[0.1, 0.15, 0.05, 0.1, 0.1], 10, 0.5),
            (vec![], 0)
        );
        // enough reach it: nothing below joins, and -n still caps
        assert_eq!(shown(&order, &ps, 10, 0.25).1, 4);
        assert_eq!(shown(&order, &ps, 10, 0.25).0.len(), 4);
        assert_eq!(shown(&order, &ps, 1, 0.95), (vec![(3, 0.9)], 0));
        // a threshold under the floor is the floor
        assert_eq!(shown(&order, &ps, 10, 0.12).0.len(), 4);
    }

    #[test]
    fn the_summary_is_one_line_of_what_the_run_did() {
        let st = Stats {
            chunks: 412,
            candidates: 64,
            cost: Cost {
                scored: 64,
                waves: 4,
                restored: 60,
                tokens: 5123,
            },
            store_bytes: Some(1_300_000_000),
            recall_ms: 12.0,
            tree_previews: 9,
            tree_ms: 1200.0,
            rerank_ms: 830.0,
            total_ms: 845.0,
            hits: 7,
        };
        assert_eq!(
            st.summary(false),
            "7 results · 845 ms · 412 chunks · 64 read by the model, 60 from cache · tree 9 previews, 1.2 s · cache 1.3 GB"
        );
        assert_eq!(
            st.summary(true),
            "7 results · 845 ms · 412 chunks · lexical ranking only"
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
        assert_eq!(
            (j["tree_previews"].as_u64(), j["tree_ms"].as_f64()),
            (Some(9), Some(1200.0))
        );
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
        // dotfiles in, what .ignore names out
        let want = [
            ".hidden/secret.txt",
            ".ignore",
            "docs/guide.md",
            "src/net.py",
            "src/pool.rs",
        ];
        assert_eq!(paths(&all), want);
        // skipped by path, as the eval's own file is
        let skipped = load_chunks(&tr, None, &["src/net.py".into()]).unwrap();
        assert_eq!(paths(&skipped), [&want[..3], &want[4..]].concat());
        // -e keeps the chunks whose text matches (smart case)
        let re = load_chunks(&tr, Some("fn release"), &[]).unwrap();
        assert_eq!(paths(&re), ["src/pool.rs"]);
        assert!(re[0].text.contains("fn release"));
        assert_eq!(load_chunks(&tr, Some("Socket"), &[]).unwrap().len(), 0);
        assert!(load_chunks(&tr, Some("("), &[]).is_err());
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
        rerank(eng, &mut None, chunks, order, q, None).unwrap().0
    }

    #[test]
    fn a_second_pass_restores_every_candidate_from_the_store() {
        let t = Tmp::new("warm");
        let chunks = synthetic(10);
        let order: Vec<usize> = (0..10).rev().collect();
        let mut eng = engine(4);
        let mut store = Some(Store::open(&t.0, "b").unwrap());
        let (first, c1) = rerank(&mut eng, &mut store, &chunks, &order, Q, None).unwrap();
        assert_eq!((c1.scored, c1.restored), (10, 0));
        assert_eq!(store.as_ref().unwrap().len(), 10);
        // scores are what decoding from scratch gives, on the sim exactly
        assert_eq!(first, cold(&mut eng, &chunks, &order, Q));

        let (again, c2) = rerank(&mut eng, &mut store, &chunks, &order, Q, None).unwrap();
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
            None,
        )
        .unwrap();
        assert_eq!(c3.restored, 10);

        // and a new process finds them on disk
        drop(store);
        let mut store = Some(Store::open(&t.0, "b").unwrap());
        let mut fresh = engine(4);
        let (disk, c4) = rerank(&mut fresh, &mut store, &chunks, &order, Q, None).unwrap();
        assert_eq!((disk, c4.restored), (first, 10));
    }

    #[test]
    fn without_a_store_every_candidate_is_decoded_from_scratch() {
        let chunks = synthetic(7);
        let order: Vec<usize> = (0..7).collect();
        let mut eng = engine(4);
        let (ps, c) = rerank(&mut eng, &mut None, &chunks, &order, Q, None).unwrap();
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
        let go = |eng: &mut Engine, stop: Option<Cascade>| {
            let ps = rerank(eng, &mut None, &chunks, &best, Q, stop).unwrap().0;
            ps.len()
        };
        let cascade = |top, threshold| Some(Cascade { top, threshold });
        // one hit wanted: wave one finds it, wave two adds none, stop
        assert_eq!(go(&mut eng, cascade(1, top_p)), 6);
        // more hits wanted than exist: every wave is scored
        assert_eq!(go(&mut eng, cascade(2, top_p)), 12);
        // every candidate is a hit: waves keep adding
        assert_eq!(go(&mut eng, cascade(1, 0.0)), 12);
        // no hit at all: nothing has "enough"
        assert_eq!(go(&mut eng, cascade(1, 1.1)), 12);
        // and with no cascade every wave is scored, hits or none, where the
        // first of these stopped at six
        assert_eq!(go(&mut eng, None), 12);
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
        let (got, c) = rerank(&mut eng, &mut store, &chunks, &order, Q, None).unwrap();
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
        let (ps, c) = rerank(&mut eng, &mut store, &chunks, &order, Q, None).unwrap();
        assert_eq!(ps[0], ps[5]);
        assert_eq!(store.as_ref().unwrap().len(), 5);
        // the twin's snapshot, stored after the reads were asked for, serves the copy
        assert_eq!(c.restored, 1);
    }

    #[test]
    fn waves_are_greedy_runs_of_what_the_room_holds() {
        let room = Room {
            seqs: 4,
            cells: 1000,
        };
        // cells bind: two 460-token probes fit 1000 cells, a third does not
        assert_eq!(waves(room, &[460; 5]), [0..2, 2..4, 4..5]);
        // seqs bind: four to a wave, however small
        assert_eq!(waves(room, &[10; 9]), [0..4, 4..8, 8..9]);
        // the sum decides, in order: a long probe closes the wave before it
        assert_eq!(waves(room, &[300, 300, 300, 600, 100]), [0..3, 3..5]);
        // a probe that fits nowhere is a wave of its own, for the decode to refuse
        assert_eq!(waves(room, &[300, 1500, 300]), [0..1, 1..2, 2..3]);
        // a room with nothing in it still moves, a probe at a time
        let none = Room { seqs: 0, cells: 0 };
        assert_eq!(waves(none, &[5, 5]), [0..1, 1..2]);
        assert!(waves(room, &[]).is_empty());
    }

    #[test]
    fn waves_tile_the_candidates_within_the_room_and_close_only_when_full() {
        // xorshift: probes of every length, rooms of every shape
        let mut x = 0x2545_f491_4f6c_dd1du64;
        let mut next = |n: u64| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x % n) as usize
        };
        for _ in 0..200 {
            let room = Room {
                seqs: 1 + next(20),
                cells: 300 + next(3000),
            };
            let lens: Vec<usize> = (0..next(60)).map(|_| 20 + next(900)).collect();
            let plan = waves(room, &lens);
            // consecutive, every candidate once
            let mut at = 0;
            for w in &plan {
                assert!(!w.is_empty() && w.start == at, "{room:?} {w:?}");
                at = w.end;
            }
            assert_eq!(at, lens.len());
            for w in &plan {
                let cells: usize = lens[w.clone()].iter().sum();
                // within the room, unless one probe alone is more than it holds
                let fits = w.len() <= room.seqs && cells <= room.cells;
                assert!(w.len() == 1 || fits, "{room:?} {w:?}");
                // greedy: a wave closes because the next probe would not fit
                if let Some(&more) = lens.get(w.end) {
                    assert!(
                        w.len() == room.seqs || cells + more > room.cells,
                        "{room:?} {w:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn a_cramped_context_forms_smaller_waves_and_the_scores_do_not_change() {
        let t = Tmp::new("cramped");
        let chunks = synthetic(10);
        let order: Vec<usize> = (0..10).collect();
        // roomy: the ten are one wave, and these are the scores to match
        let mut roomy = engine_ctx(1 << 16, 65);
        let (want, c) = rerank(&mut roomy, &mut None, &chunks, &order, Q, None).unwrap();
        assert_eq!(c.waves, 1);
        // 2000 cells hold about three probes of ~470 tokens, whatever the 64 seqs
        let mut tight = engine_ctx(2000, 65);
        let lens: Vec<usize> = order
            .iter()
            .map(|&i| probe_for(&tight, &chunks[i], Q).unwrap().toks.len())
            .collect();
        let room = tight.grep_room();
        let plan = waves(room, &lens);
        assert!(plan.len() >= 3, "{plan:?}");
        for w in &plan {
            assert!(lens[w.clone()].iter().sum::<usize>() <= room.cells);
        }
        let (ps, c) = rerank(&mut tight, &mut None, &chunks, &order, Q, None).unwrap();
        assert_eq!((ps, c.waves), (want.clone(), plan.len()));
        // with a store: the same waves cold, and warm from the snapshots
        let mut store = Some(Store::open(&t.0, "b").unwrap());
        let (cold, c) = rerank(&mut tight, &mut store, &chunks, &order, Q, None).unwrap();
        assert_eq!((cold, c.waves, c.restored), (want.clone(), plan.len(), 0));
        let (warm, c) = rerank(&mut tight, &mut store, &chunks, &order, Q, None).unwrap();
        assert_eq!((warm, c.waves, c.restored), (want, plan.len(), 10));
    }

    #[test]
    fn waves_of_different_sizes_are_read_ahead_by_the_first_one_and_answer_alike() {
        let t = Tmp::new("mixed");
        // four long chunks, then short ones: the first wave is the smallest
        let chunks: Vec<Chunk> = (0..16)
            .map(|i| {
                let body = match i {
                    0..=3 => "    let v = 1;\n".repeat(60),
                    _ => "    v\n".into(),
                };
                let text = format!("fn sym{i}() {{\n{body}}}");
                chunk(&format!("src/g{i}.rs"), 1, Some(&format!("sym{i}")), &text)
            })
            .collect();
        let order: Vec<usize> = (0..16).collect();
        let (want, _) = rerank(&mut engine(65), &mut None, &chunks, &order, Q, None).unwrap();
        let mut eng = engine_ctx(4000, 65);
        let lens: Vec<usize> = order
            .iter()
            .map(|&i| probe_for(&eng, &chunks[i], Q).unwrap().toks.len())
            .collect();
        let plan = waves(eng.grep_room(), &lens);
        let sizes: Vec<usize> = plan.iter().map(Range::len).collect();
        assert!(sizes.iter().any(|&n| n > sizes[0]), "{sizes:?}");
        // the first wave is also how far the reads run ahead: later waves are
        // bigger than the queue, and the answers are the same
        let mut store = Some(Store::open(&t.0, "b").unwrap());
        for pass in 0..2 {
            let (ps, c) = rerank(&mut eng, &mut store, &chunks, &order, Q, None).unwrap();
            assert_eq!(
                (ps, c.waves, c.restored),
                (want.clone(), plan.len(), pass * 16)
            );
        }
    }

    // --- one search

    /// What `load` hands a search: a fresh engine (a new process) and a GGUF.
    /// The GGUF is written once: its mtime is part of the binding, and a
    /// rewrite a second later would open a different store.
    fn model(t: &Tmp) -> impl FnOnce() -> Result<Loaded> {
        model_seqs(t, 5)
    }

    /// `model` on an engine of `n_seq` seqs: waves of `n_seq - 1`.
    fn model_seqs(t: &Tmp, n_seq: usize) -> impl FnOnce() -> Result<Loaded> {
        let gguf = t.at("m.gguf");
        if !gguf.exists() {
            t.write("m.gguf", "weights");
        }
        move || Ok((engine(n_seq), gguf))
    }

    fn query(q: &str, o: Opts) -> Search {
        Search {
            query: q.into(),
            out: Out::Human,
            lines: None,
            color: None,
            opts: o,
        }
    }

    #[test]
    fn a_search_reranks_the_candidates_and_the_second_one_starts_warm() {
        let t = Tmp::new("search");
        let root = project(&t, "tree");
        let s = query("connection pool checkout", opts(&root));
        let base = t.at("cache");
        let cold = search(&s, &base, model(&t)).unwrap();
        // a preview that is its chunk's own text was snapshotted by the descent
        assert!(cold.stats.cost.scored > 0 && cold.stats.tree_previews > 0);
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
        // an empty corpus has nothing to rank
        let nothing = Opts {
            regexp: Some("zebra".into()),
            ..opts(&root)
        };
        let none = search(&query("connection pool", nothing), &t.at("cache"), no_model).unwrap();
        assert!(none.hits.is_empty());
        assert!(!t.at("cache").exists());
    }

    #[test]
    fn a_query_without_a_shared_word_still_reaches_the_model_through_the_descent() {
        let t = Tmp::new("paraphrase");
        let root = project(&t, "tree");
        let s = query("zebra", opts(&root));
        assert!(Haystack::new(load_chunks(&s.opts.tree, None, &[]).unwrap())
            .recall("zebra", 64)
            .is_empty());
        let found = search(&s, &t.at("cache"), model(&t)).unwrap();
        assert!(found.stats.tree_previews > 0 && found.stats.candidates > 0);
        assert!(found.stats.cost.scored > 0);
    }

    #[test]
    fn the_union_interleaves_bm25_with_the_descents_new_picks() {
        let lexical = vec![5, 6, 7];
        // 6 is held by BM25 and 9 repeats: neither adds a candidate
        let got = union(lexical.clone(), &[6, 9, 1, 9, 2]);
        assert_eq!(got, [5, 9, 6, 1, 7, 2]);
        // the descent cut at TREE_CHUNKS, BM25's rest following when it outlasts it
        let tree: Vec<usize> = (100..100 + TREE_CHUNKS + 5).collect();
        let long: Vec<usize> = (0..64).collect();
        let got = union(long, &tree);
        assert_eq!(
            (got.len(), got[..4].to_vec()),
            (64 + TREE_CHUNKS, vec![0, 100, 1, 101])
        );
        assert_eq!(got.last(), Some(&63));
        // either side may be empty
        assert_eq!(union(lexical, &[]), [5, 6, 7]);
        assert_eq!(union(vec![], &[3, 4]), [3, 4]);
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

    // --- snap grep --cache, --gc

    #[test]
    fn the_cache_report_counts_what_a_tree_has_and_gc_drops_what_it_lost() {
        let t = Tmp::new("cache");
        let root = project(&t, "tree");
        let (base, tr) = (t.at("cache"), tree(&root));
        let gguf = t.write("m.gguf", "weights");
        let load = || -> Result<Loaded> { Ok((engine(5), gguf.clone())) };
        let run = |report, gc, home: Option<&Path>| cache_in(&base, &tr, report, gc, home, load);
        let chunks = load_chunks(&tr, None, &[]).unwrap();
        let n = chunks.len();
        // reading a tree nothing was stored for says so and creates nothing
        assert_eq!(
            run(true, false, None).unwrap()[0].split(" · ").nth(1),
            Some("no snapshots yet")
        );
        assert_eq!(run(false, true, None).unwrap(), ["gc: no snapshots"]);
        assert!(!base.exists());
        // a search snapshots what it reads: here every chunk
        let mut store = Some(open_store(&base, &tr, &gguf, "snap-sim").unwrap());
        let order: Vec<usize> = (0..n).collect();
        rerank(&mut engine(5), &mut store, &chunks, &order, Q, None).unwrap();
        let dir = store_spec(&base, &tr, &gguf, "snap-sim").unwrap().0;
        drop(store);
        let line = run(true, false, None).unwrap().remove(0);
        let parts: Vec<&str> = line.split(" · ").collect();
        assert_eq!(parts[0], format!("cache {}", dir.display()), "{line}");
        assert_eq!(parts[1], plural(n, "snapshot"));
        assert_eq!(parts[3], format!("{n} of {n} chunks (100%)"));
        assert_eq!(parts[4], "0 stale");
        // a file goes: its snapshots are stale until gc drops them
        let gone = load_chunks(&tr, None, &["src/net.py".into()])
            .unwrap()
            .len();
        fs::remove_file(root.join("src/net.py")).unwrap();
        let line = run(true, false, None).unwrap().remove(0);
        assert!(line.ends_with(&format!("{} stale", n - gone)), "{line}");
        // an engine that is not this one left a store of the tree, and another tree has its own
        let tree_dir = dir.parent().unwrap();
        let (old, elsewhere) = (
            tree_dir.join("0123456789abcdef"),
            base.join("fedcba9876543210"),
        );
        for d in [&old, &elsewhere.join("0123456789abcdef")] {
            let mut s = Store::open(d, "snap 0.0.1").unwrap();
            s.put(1, 3, &[7; 4096]).unwrap();
        }
        let out = run(true, true, None).unwrap();
        assert_eq!(out.len(), 2);
        assert!(
            out[0].starts_with(&format!("gc: {n} → {gone} snapshots, ")),
            "{}",
            out[0]
        );
        assert!(
            out[0].contains("; 1 store of other engines removed ("),
            "{}",
            out[0]
        );
        assert!(out[1].ends_with("0 stale"), "{}", out[1]);
        assert!(!old.exists() && dir.exists() && elsewhere.exists());
        // the same again frees nothing
        assert!(!run(false, true, None).unwrap()[0].contains("removed"));
    }

    #[test]
    fn the_store_is_shown_whole_under_the_home() {
        let dir = Path::new("/home/me/.cache/snap/grep/7ffe1234abcd5678/4777deadbeef0000");
        assert_eq!(
            shown_dir(dir, Some(Path::new("/home/me"))),
            "~/.cache/snap/grep/7ffe1234abcd5678/4777deadbeef0000"
        );
        assert_eq!(
            shown_dir(dir, Some(Path::new("/root"))),
            dir.to_string_lossy()
        );
        assert_eq!(plural(1, "store"), "1 store");
        assert_eq!(plural(1159, "snapshot"), "1,159 snapshots");
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
            candidates: 0,
            scored: None,
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
        // every candidate of every case was scored
        for c in rep["cases"].as_array().unwrap() {
            assert_eq!(c["scored"], c["candidates"], "{}", c["id"]);
        }
    }

    /// `n` files that all answer "pool connection": every chunk is a candidate.
    /// The cases file stays outside the tree, so a search sees just the `n`.
    fn crowd(t: &Tmp, n: usize) -> (PathBuf, PathBuf) {
        for i in 0..n {
            let text = format!(
                "pub fn pool_{i}() -> Conn {{\n    // a connection from the pool, number {i}\n}}\n"
            );
            t.write(&format!("crowd/src/m{i}.rs"), &text);
        }
        let case = r#"{"id": "c1", "query": "pool connection", "kind": "lexical", "expect": [{"path": "src/m0.rs", "contains": "fn pool_0("}]}"#;
        (t.at("crowd"), t.write("crowd-cases.jsonl", case))
    }

    #[test]
    fn an_eval_scores_every_candidate_where_a_search_would_have_stopped() {
        let t = Tmp::new("every");
        let (root, file) = crowd(&t, 12);
        let base = t.at("cache");
        // waves of two over twelve candidates, one hit wanted, no store
        let opts_at = |threshold| Opts {
            top: 1,
            threshold,
            store: false,
            ..opts(&root)
        };
        let stops = |threshold| {
            let s = query("pool connection", opts_at(threshold));
            let found = search(&s, &base, model_seqs(&t, 3)).unwrap();
            assert_eq!(found.stats.candidates, 12);
            found.stats.cost.scored < 12
        };
        // a threshold at which the cascade of a search cuts the run short
        let cut = (1..20).map(|k| f64::from(k) / 20.0).find(|&th| stops(th));
        let threshold = cut.expect("some threshold has the cascade stop early");
        // the eval, on the same options, reads all twelve: its hit@k and MRR
        // are over every candidate, whatever the wave size
        let o = opts_at(threshold);
        let rep = run_eval(file.to_str().unwrap(), None, &base, &o, model_seqs(&t, 3)).unwrap();
        let case = &rep["cases"][0];
        assert_eq!(
            (case["candidates"].as_u64(), case["scored"].as_u64()),
            (Some(12), Some(12))
        );
        assert_eq!(case["top"].as_array().unwrap().len(), 10);
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
