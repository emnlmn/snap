//! KV orchestration over one multi-seq context: a prefix cache plus a trie
//! planner.
//!
//! A job is a full prompt whose logits row is read at its last token. It
//! forks off the longest cached span it strictly extends — seq 0 holds the
//! resident template head; other entries are spans that recur across
//! requests (`[head+state]`, `[head+question]`, `[head+catalog]`: whatever
//! the job's `keep` names). The jobs of a wave then decode as one prefix
//! trie in a single batched call: every trie edge is decoded once, tagged
//! with all the seqs below it, so text shared by N questions costs one pass.
//!
//! Sequences are only ever copied whole and removed whole, never trimmed.
//! That single rule is what lets recurrent/hybrid memories — no partial
//! rewind, range-less `seq_cp` — run the same path as plain attention KV.
//!
//! Seqs and KV cells are shared by cache entries and a wave's work seqs.
//! Waves are sized to what fits, unprotected entries are evicted LRU on
//! demand, and a `NoSlot` (llama.cpp fails those atomically per call)
//! evicts or shrinks the wave and retries — the accounting here only has to
//! be good, not exact.

use std::collections::HashSet;
use std::time::Instant;

use anyhow::{bail, Result};

/// Consecutive tokens decoded at `pos0..`, tagged with every seq in `seqs`;
/// `logits` asks for the row at the last token.
pub struct Dec<'a> {
    pub seqs: Vec<i32>,
    pub toks: &'a [i32],
    pub pos0: usize,
    pub logits: bool,
}

#[derive(Debug)]
pub enum DecodeError {
    /// No free KV cells for the batch.
    NoSlot,
    Failed(String),
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            DecodeError::NoSlot => write!(f, "no free KV cells for the batch"),
            DecodeError::Failed(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for DecodeError {}

/// The model runtime: `llamac::Llama` in production, `sim::Sim` in tests.
pub trait Backend: Send {
    fn n_ctx(&self) -> usize;
    fn n_seq(&self) -> usize;
    /// Chat template over (system, user), generation prompt appended.
    fn render(&self, system: &str, user: &str) -> Result<String>;
    fn tokenize(&self, text: &str) -> Result<Vec<i32>>;
    /// Decode groups in order; `row(g, logits)` fires for each group asking
    /// for logits, while the row is still valid.
    fn decode(
        &mut self,
        groups: &[Dec],
        row: &mut dyn FnMut(usize, &[f32]),
    ) -> Result<(), DecodeError>;
    /// Remove a whole sequence.
    fn seq_rm(&mut self, seq: i32);
    /// Make the empty `dst` a copy of `src`, which holds exactly `len` tokens.
    fn seq_cp(&mut self, src: i32, dst: i32, len: usize);
    fn clear(&mut self);
}

/// A decode unit: the prompt, and the length of its prefix worth keeping
/// across requests (0 = nothing).
#[derive(Clone, Copy)]
pub struct Job<'a> {
    pub toks: &'a [i32],
    pub keep: usize,
}

#[derive(Default, Debug)]
pub struct Stats {
    /// tokens actually decoded (KV cells written)
    pub decoded: usize,
    pub waves: usize,
    /// jobs that forked off a span cached by an earlier request
    pub hits: usize,
    /// jobs whose keep span had to be decoded
    pub misses: usize,
    pub decode_ms: f64,
}

struct Entry {
    toks: Vec<i32>,
    seq: i32,
    /// cells this entry keeps alive past the span it was forked from
    cells: usize,
    tick: u64,
    born: u64,
    pinned: bool,
    /// request-scoped: dropped when the run ends
    temp: bool,
}

/// A span a wave leaves cached on a seq of its own, forked off `base`:
/// keep spans live on across requests, temp spans only until the run ends.
struct Span<'j> {
    toks: &'j [i32],
    base: (i32, usize),
    temp: bool,
}

/// What one wave takes on: its jobs (positions into `pending`, trie order),
/// the spans it caches, and the work seqs and cells that costs.
struct Plan<'j> {
    wave: Vec<usize>,
    spans: Vec<Span<'j>>,
    works: usize,
    cells: usize,
}

/// Cell slack under n_ctx when sizing waves.
const KV_SLACK: usize = 32;

pub struct Kv {
    entries: Vec<Entry>,
    tick: u64,
    /// seqs of a wave not yet cleaned up — non-empty only after an error or
    /// panic mid-wave; wiped at the start of the next run
    scratch: Vec<i32>,
}

pub(crate) fn common(a: &[i32], b: &[i32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

impl Kv {
    /// Fresh memory with `head` resident on seq 0.
    pub fn new(b: &mut dyn Backend, head: Vec<i32>) -> Result<Kv> {
        b.clear();
        let mut kv = Kv {
            entries: Vec::new(),
            tick: 0,
            scratch: Vec::new(),
        };
        if !head.is_empty() {
            let g = [Dec {
                seqs: vec![0],
                toks: &head,
                pos0: 0,
                logits: true,
            }];
            b.decode(&g, &mut |_, _| {})?;
            kv.entries.push(Entry {
                cells: head.len(),
                toks: head,
                seq: 0,
                tick: 0,
                born: 0,
                pinned: true,
                temp: false,
            });
        }
        Ok(kv)
    }

    /// Wipe the cache and rebuild the resident head — recovery after a
    /// failed decode leaves memory in an unknown state.
    pub fn reset(&mut self, b: &mut dyn Backend) -> Result<()> {
        let head = self
            .entries
            .iter()
            .find(|e| e.pinned)
            .map(|e| e.toks.clone())
            .unwrap_or_default();
        *self = Kv::new(b, head)?;
        Ok(())
    }

    /// Longest cached span `toks` strictly extends: (seq, len), or (-1, 0).
    /// Direct mode only ever forks off the resident head.
    fn base(&self, toks: &[i32], shared: bool) -> (i32, usize) {
        self.entries
            .iter()
            .filter(|e| (shared || e.pinned) && e.toks.len() < toks.len())
            .filter(|e| toks.starts_with(&e.toks))
            .max_by_key(|e| e.toks.len())
            .map_or((-1, 0), |e| (e.seq, e.toks.len()))
    }

    fn cells(&self) -> usize {
        self.entries.iter().map(|e| e.cells).sum()
    }

    fn free_seqs(&self, n_seq: usize) -> Vec<i32> {
        (1..n_seq as i32)
            .filter(|s| self.entries.iter().all(|e| e.seq != *s))
            .collect()
    }

    /// Drop the least recently used unpinned entry `allow` accepts.
    fn evict(&mut self, b: &mut dyn Backend, allow: impl Fn(&Entry) -> bool) -> bool {
        let Some(i) = (0..self.entries.len())
            .filter(|&i| !self.entries[i].pinned && allow(&self.entries[i]))
            .min_by_key(|&i| self.entries[i].tick)
        else {
            return false;
        };
        b.seq_rm(self.entries.swap_remove(i).seq);
        true
    }

    /// Decode every job; `out(job, row)` receives each job's logits row.
    pub fn run(
        &mut self,
        b: &mut dyn Backend,
        jobs: &[Job],
        shared: bool,
        out: &mut dyn FnMut(usize, &[f32]),
    ) -> Result<Stats> {
        for s in self.scratch.drain(..) {
            b.seq_rm(s);
        }
        let (n_seq, n_ctx) = (b.n_seq(), b.n_ctx());
        if let Some(j) = jobs
            .iter()
            .find(|j| j.toks.is_empty() || j.toks.len() > n_ctx)
        {
            bail!("prompt is {} tokens, ctx is {n_ctx}", j.toks.len());
        }
        let first = self.tick + 1;
        let mut st = Stats::default();
        let mut pending: Vec<usize> = (0..jobs.len()).collect();
        let mut limit = usize::MAX;
        while !pending.is_empty() {
            match self.wave(b, jobs, &pending, shared, first, limit, out, &mut st)? {
                Some(done) => {
                    pending.retain(|j| !done.contains(j));
                    limit = usize::MAX;
                }
                None => limit = (limit.min(pending.len()) / 2).max(1),
            }
        }
        while self.evict(b, |e| e.temp) {}
        // the cache keeps at most half the seqs and half the cells
        loop {
            let (n, c) = self
                .entries
                .iter()
                .filter(|e| !e.pinned)
                .fold((0, 0), |(n, c), e| (n + 1, c + e.cells));
            if (n <= n_seq / 2 && c <= n_ctx / 2) || !self.evict(b, |_| true) {
                break;
            }
        }
        Ok(st)
    }

    /// One wave of at most `limit` jobs: plan, decode, clean up. Returns the
    /// jobs answered, or None when the backend ran out of cells and the
    /// caller should retry with a smaller wave.
    #[allow(clippy::too_many_arguments)]
    fn wave(
        &mut self,
        b: &mut dyn Backend,
        jobs: &[Job],
        pending: &[usize],
        shared: bool,
        first: u64,
        limit: usize,
        out: &mut dyn FnMut(usize, &[f32]),
        st: &mut Stats,
    ) -> Result<Option<Vec<usize>>> {
        let (n_seq, n_ctx) = (b.n_seq(), b.n_ctx());
        self.tick += 1;
        let base: Vec<(i32, usize)> = pending
            .iter()
            .map(|&j| self.base(jobs[j].toks, shared))
            .collect();
        // trie order: grouped by base, then lexicographic — shared prefixes
        // and duplicates end up adjacent
        let mut order: Vec<usize> = (0..pending.len()).collect();
        order.sort_by(|&x, &y| {
            base[x]
                .0
                .cmp(&base[y].0)
                .then_with(|| jobs[pending[x]].toks.cmp(jobs[pending[y]].toks))
        });
        // entries some pending job forks from stay put while the wave forms
        let wanted: HashSet<i32> = base.iter().map(|b| b.0).collect();
        let (held, held_cells) = self
            .entries
            .iter()
            .filter(|e| e.pinned || wanted.contains(&e.seq))
            .fold((0, 0), |(n, c), e| (n + !e.pinned as usize, c + e.cells));
        let seq_cap = (n_seq - 1).saturating_sub(held);
        let cell_cap = n_ctx.saturating_sub(held_cells + KV_SLACK);
        let plan = |seqs| form(jobs, pending, &base, &order, shared, limit, seqs, cell_cap);
        let mut p = plan(seq_cap);
        // a base group spilling into the next wave: hold what all its jobs
        // share on a temp seq, so later waves fork past it instead of
        // decoding it again
        if shared && p.wave.len() < pending.len() && seq_cap > 1 {
            let bs = base[order[p.wave.len()]];
            let group: Vec<&[i32]> = order
                .iter()
                .filter(|&&x| base[x] == bs)
                .map(|&x| jobs[pending[x]].toks)
                .collect();
            let lcp = common(group[0], group[group.len() - 1]);
            let span = &group[0][..lcp];
            if lcp > bs.1 && group.iter().all(|t| t.len() > lcp) {
                let mut q = plan(seq_cap - 1);
                if q.wave.iter().any(|&x| base[x] == bs) && q.spans.iter().all(|s| s.toks != span) {
                    q.spans.push(Span {
                        toks: span,
                        base: bs,
                        temp: true,
                    });
                    p = q;
                }
            }
        }

        // make room: first evict entries no pending job needs, then ones
        // later waves wanted — never this wave's own bases
        let mine: HashSet<i32> = p.wave.iter().map(|&x| base[x].0).collect();
        let (works, cells) = (p.works, p.cells);
        let fits = |kv: &Kv, spans: usize| {
            kv.free_seqs(n_seq).len() >= works + spans && kv.cells() + cells + KV_SLACK <= n_ctx
        };
        while !fits(self, p.spans.len()) && self.evict(b, |e| !wanted.contains(&e.seq)) {}
        while !fits(self, p.spans.len()) && self.evict(b, |e| !mine.contains(&e.seq)) {}
        if self.free_seqs(n_seq).len() < works + p.spans.len() {
            p.spans.clear(); // spans are an optimization; the answers are not
        }

        // seqs: one work seq per distinct prompt, one per span, each seeded
        // with a whole copy of its base
        let wave = &p.wave;
        let mut free = self.free_seqs(n_seq);
        let mut seq_of: Vec<i32> = Vec::with_capacity(wave.len());
        for (k, &x) in wave.iter().enumerate() {
            let dup = shared
                && k > 0
                && base[wave[k - 1]] == base[x]
                && jobs[pending[wave[k - 1]]].toks == jobs[pending[x]].toks;
            let s = if dup {
                seq_of[k - 1]
            } else {
                free.pop().expect("wave sized to free seqs")
            };
            seq_of.push(s);
        }
        let span_seqs: Vec<i32> = p.spans.iter().map(|_| free.pop().unwrap()).collect();
        let mut seeds: Vec<(i32, (i32, usize))> = seq_of
            .iter()
            .zip(wave)
            .map(|(&s, &x)| (s, base[x]))
            .collect();
        seeds.dedup_by_key(|s| s.0);
        seeds.extend(span_seqs.iter().zip(&p.spans).map(|(&s, sp)| (s, sp.base)));
        self.scratch = seeds.iter().map(|s| s.0).collect();
        // free seqs are always empty: work seqs are wiped after every wave,
        // evicted entries on eviction, strays of an aborted wave on the next run
        for &(s, (bseq, blen)) in &seeds {
            if bseq >= 0 {
                b.seq_cp(bseq, s, blen);
            }
        }

        // the trie, one base group at a time
        let mut groups: Vec<Dec> = Vec::new();
        let mut rows: Vec<Vec<usize>> = Vec::new();
        let mut lo = 0;
        for part in wave.chunk_by(|&x, &y| shared && base[x] == base[y]) {
            let bs = base[part[0]];
            let leaves: Vec<Leaf> = (lo..lo + part.len())
                .map(|k| Leaf {
                    job: pending[wave[k]],
                    toks: jobs[pending[wave[k]]].toks,
                    seq: seq_of[k],
                })
                .collect();
            let tags: Vec<(&[i32], i32)> = p
                .spans
                .iter()
                .zip(&span_seqs)
                .filter(|(sp, _)| sp.base == bs)
                .map(|(sp, &s)| (sp.toks, s))
                .collect();
            node(&leaves, bs.1, &tags, &mut groups, &mut rows);
            lo += part.len();
        }

        let wrote: usize = groups.iter().map(|g| g.toks.len()).sum();
        debug_assert_eq!(wrote, cells, "wave cell estimate drifted from the trie");
        let t = Instant::now();
        let res = b.decode(&groups, &mut |g, row| {
            for &j in &rows[g] {
                out(j, row);
            }
        });
        st.decode_ms += t.elapsed().as_secs_f64() * 1000.0;
        let failed: &[i32] = if res.is_ok() { &[] } else { &span_seqs };
        for &s in seq_of.iter().chain(failed) {
            b.seq_rm(s);
        }
        self.scratch.clear();
        match res {
            Ok(()) => {}
            Err(DecodeError::NoSlot) => {
                // a failed wave never leaves a span behind; free more cells,
                // else retry smaller — a lone job that still can't fit next
                // to the head never will
                if !self.evict(b, |e| !mine.contains(&e.seq)) && wave.len() == 1 {
                    bail!(
                        "prompt of {} tokens does not fit ctx {n_ctx}",
                        jobs[pending[wave[0]]].toks.len()
                    );
                }
                return Ok(None);
            }
            Err(e) => return Err(e.into()),
        }

        for (sp, &seq) in p.spans.iter().zip(&span_seqs) {
            self.entries.push(Entry {
                toks: sp.toks.to_vec(),
                seq,
                cells: sp.toks.len() - sp.base.1,
                tick: self.tick,
                born: self.tick,
                pinned: false,
                temp: sp.temp,
            });
        }
        for e in &mut self.entries {
            if mine.contains(&e.seq) {
                e.tick = self.tick;
            }
        }
        st.decoded += wrote;
        st.waves += 1;
        for &x in wave {
            let (bseq, blen) = base[x];
            let e = self.entries.iter().find(|e| e.seq == bseq);
            st.hits += e.is_some_and(|e| !e.pinned && e.born < first) as usize;
            st.misses += (shared && jobs[pending[x]].keep > blen) as usize;
        }
        Ok(Some(wave.iter().map(|&x| pending[x]).collect()))
    }

    #[cfg(test)]
    pub(crate) fn entries(&self) -> Vec<(i32, Vec<i32>)> {
        self.entries
            .iter()
            .map(|e| (e.seq, e.toks.clone()))
            .collect()
    }
}

/// Greedy wave in trie order under a seq and a cell budget: every distinct
/// prompt needs a work seq, every new keep span an entry seq, and cells
/// grow by whatever the trie doesn't share with the previous prompt.
#[allow(clippy::too_many_arguments)]
fn form<'j>(
    jobs: &[Job<'j>],
    pending: &[usize],
    base: &[(i32, usize)],
    order: &[usize],
    shared: bool,
    limit: usize,
    seq_cap: usize,
    cell_cap: usize,
) -> Plan<'j> {
    let mut p = Plan {
        wave: Vec::new(),
        spans: Vec::new(),
        works: 0,
        cells: 0,
    };
    for &x in order {
        if p.wave.len() == limit {
            break;
        }
        let (job, (bseq, blen)) = (jobs[pending[x]], base[x]);
        let prev = p
            .wave
            .last()
            .filter(|&&q| shared && base[q].0 == bseq)
            .map(|&q| jobs[pending[q]].toks);
        let lcp = prev.map_or(0, |q| common(q, job.toks));
        let (w, c) = (
            (prev != Some(job.toks)) as usize,
            job.toks.len() - blen.max(lcp),
        );
        // new entries fork off the head only, so their cells stay countable
        let keep = (shared && bseq <= 0 && job.keep > blen && job.keep < job.toks.len())
            .then(|| &job.toks[..job.keep])
            .filter(|k| p.spans.iter().all(|s| s.toks != *k));
        let seqs = p.works + w + p.spans.len() + keep.is_some() as usize;
        if !p.wave.is_empty() && (seqs > seq_cap || p.cells + c > cell_cap) {
            break;
        }
        p.wave.push(x);
        p.works += w;
        p.cells += c;
        if let Some(toks) = keep {
            p.spans.push(Span {
                toks,
                base: (bseq, blen),
                temp: false,
            });
        }
    }
    p
}

struct Leaf<'a> {
    job: usize,
    toks: &'a [i32],
    seq: i32,
}

/// Emit the trie below `depth` for lexicographically sorted leaves that all
/// share `toks[..depth]`: the common edge first, tagged with every leaf seq
/// and every span running through it (edges split where a span ends), then
/// each child subtree.
fn node<'a>(
    ls: &[Leaf<'a>],
    depth: usize,
    spans: &[(&[i32], i32)],
    groups: &mut Vec<Dec<'a>>,
    rows: &mut Vec<Vec<usize>>,
) {
    let (first, last) = (ls[0].toks, ls[ls.len() - 1].toks);
    let mut end = depth + common(&first[depth..], &last[depth..]);
    for (k, _) in spans {
        if k.len() > depth && k.len() < end && first.starts_with(k) {
            end = k.len();
        }
    }
    if end > depth {
        let mut seqs: Vec<i32> = ls.iter().map(|l| l.seq).collect();
        seqs.dedup();
        seqs.extend(
            spans
                .iter()
                .filter(|(k, _)| k.len() >= end && first[..end] == k[..end])
                .map(|(_, s)| *s),
        );
        let ends: Vec<usize> = ls
            .iter()
            .filter(|l| l.toks.len() == end)
            .map(|l| l.job)
            .collect();
        groups.push(Dec {
            seqs,
            toks: &first[depth..end],
            pos0: depth,
            logits: !ends.is_empty(),
        });
        rows.push(ends);
    }
    // prompts ending here sort first; the rest split by their next token
    let rest = &ls[ls.iter().take_while(|l| l.toks.len() == end).count()..];
    for child in rest.chunk_by(|a, b| a.toks[end] == b.toks[end]) {
        node(child, end, spans, groups, rows);
    }
}

#[cfg(test)]
pub(crate) mod sim {
    //! Simulated llama.cpp memory: a unified pool of KV cells shared through
    //! seq tags, per-seq token histories, and the rules llama.cpp enforces on
    //! every decode call (positions, coupled seqs, capacity) — violations
    //! panic, capacity returns NoSlot atomically. A row is a hash of its
    //! sequence's whole history, so it equals `row(prompt)` exactly when the
    //! engine assembled precisely that prompt at positions 0.. on that seq.
    //! Tokens are bytes, the chat template a fixed wrapper.

    use std::collections::HashMap;

    use anyhow::Result;

    use super::{Backend, Dec, DecodeError};

    pub const N_VOCAB: usize = 256;

    pub struct Sim {
        pub n_ctx: usize,
        pub n_seq: usize,
        pub n_batch: usize,
        seqs: Vec<Vec<(i32, usize)>>,
        refs: HashMap<usize, usize>,
        next: usize,
        pub calls: usize,
        /// genuine out-of-cells failures: a planner sizing waves right never
        /// causes one
        pub no_slots: usize,
        /// make this decode call (1-based) fail with NoSlot
        pub fail_call: Option<usize>,
    }

    impl Sim {
        pub fn new(n_ctx: usize, n_seq: usize, n_batch: usize) -> Sim {
            Sim {
                n_ctx,
                n_seq,
                n_batch,
                seqs: vec![Vec::new(); n_seq],
                refs: HashMap::new(),
                next: 0,
                calls: 0,
                no_slots: 0,
                fail_call: None,
            }
        }

        pub fn history(&self, seq: i32) -> Vec<i32> {
            self.seqs[seq as usize].iter().map(|c| c.0).collect()
        }

        pub fn used(&self) -> usize {
            self.refs.len()
        }

        /// Distinct cells held by `seqs` and not by `but`.
        pub fn cells_of(&self, seqs: &[i32], but: i32) -> usize {
            let skip: std::collections::HashSet<usize> =
                self.seqs[but as usize].iter().map(|c| c.1).collect();
            let mut held: Vec<usize> = seqs
                .iter()
                .flat_map(|&s| self.seqs[s as usize].iter().map(|c| c.1))
                .filter(|c| !skip.contains(c))
                .collect();
            held.sort_unstable();
            held.dedup();
            held.len()
        }

        /// Write tokens straight onto a seq, bypassing the engine — to fake
        /// the residue of an aborted wave.
        pub fn scribble(&mut self, seq: i32, toks: &[i32]) {
            for &t in toks {
                self.next += 1;
                *self.refs.entry(self.next).or_default() += 1;
                self.seqs[seq as usize].push((t, self.next));
            }
        }
    }

    /// Deterministic pseudo-logits for a token history.
    pub fn row(hist: &[i32]) -> Vec<f32> {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for &t in hist {
            h = (h ^ t as u64).wrapping_mul(0x0100_0000_01b3);
        }
        (0..N_VOCAB)
            .map(|_| {
                h ^= h >> 12;
                h ^= h << 25;
                h ^= h >> 27;
                (h.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 40) as f32 / (1u64 << 21) as f32 - 4.0
            })
            .collect()
    }

    impl Backend for Sim {
        fn n_ctx(&self) -> usize {
            self.n_ctx
        }

        fn n_seq(&self) -> usize {
            self.n_seq
        }

        fn render(&self, system: &str, user: &str) -> Result<String> {
            Ok(format!("<s>{system}<u>{user}<a>"))
        }

        fn tokenize(&self, text: &str) -> Result<Vec<i32>> {
            Ok(text.bytes().map(i32::from).collect())
        }

        fn decode(
            &mut self,
            groups: &[Dec],
            row_out: &mut dyn FnMut(usize, &[f32]),
        ) -> Result<(), DecodeError> {
            let flat: Vec<(usize, usize)> = groups
                .iter()
                .enumerate()
                .flat_map(|(g, d)| (0..d.toks.len()).map(move |i| (g, i)))
                .collect();
            for chunk in flat.chunks(self.n_batch) {
                self.calls += 1;
                if self.fail_call == Some(self.calls) {
                    return Err(DecodeError::NoSlot);
                }
                if self.used() + chunk.len() > self.n_ctx {
                    self.no_slots += 1;
                    return Err(DecodeError::NoSlot);
                }
                // coupled seqs must hold identical histories before the call,
                // positions continue each seq's history without gaps
                let mut len: HashMap<i32, usize> = HashMap::new();
                let mut seen = std::collections::HashSet::new();
                for &(g, t) in chunk {
                    let d = &groups[g];
                    for &s in &d.seqs {
                        assert!((s as usize) < self.n_seq, "seq {s} out of range");
                        let l = len.entry(s).or_insert_with(|| self.seqs[s as usize].len());
                        assert_eq!(d.pos0 + t, *l, "seq {s}: position gap or overlap");
                        *l += 1;
                    }
                    if seen.insert(g) {
                        let h0 = self.history(d.seqs[0]);
                        for &s in &d.seqs[1..] {
                            assert_eq!(self.history(s), h0, "coupled seqs diverged");
                        }
                    }
                }
                for &(g, t) in chunk {
                    let d = &groups[g];
                    self.next += 1;
                    self.refs.insert(self.next, d.seqs.len());
                    for &s in &d.seqs {
                        self.seqs[s as usize].push((d.toks[t], self.next));
                    }
                    if d.logits && t + 1 == d.toks.len() {
                        let hist = self.history(d.seqs[0]);
                        for &s in &d.seqs[1..] {
                            assert_eq!(
                                self.history(s),
                                hist,
                                "coupled seqs hold different prompts"
                            );
                        }
                        row_out(g, &row(&hist));
                    }
                }
            }
            Ok(())
        }

        fn seq_rm(&mut self, seq: i32) {
            for (_, c) in std::mem::take(&mut self.seqs[seq as usize]) {
                let r = self.refs.get_mut(&c).unwrap();
                *r -= 1;
                if *r == 0 {
                    self.refs.remove(&c);
                }
            }
        }

        fn seq_cp(&mut self, src: i32, dst: i32, len: usize) {
            assert_eq!(
                self.seqs[src as usize].len(),
                len,
                "partial copy of seq {src}"
            );
            assert!(
                self.seqs[dst as usize].is_empty(),
                "copy onto non-empty seq {dst}"
            );
            self.seqs[dst as usize] = self.seqs[src as usize].clone();
            for (_, c) in &self.seqs[dst as usize] {
                *self.refs.get_mut(c).unwrap() += 1;
            }
        }

        fn clear(&mut self) {
            self.seqs.iter_mut().for_each(Vec::clear);
            self.refs.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::sim::{row, Sim};
    use super::*;

    /// xorshift: deterministic randomness without a dependency
    struct Rng(u64);
    impl Rng {
        fn below(&mut self, n: usize) -> usize {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 % n as u64) as usize
        }
    }

    const HEAD: [i32; 5] = [90, 91, 92, 93, 94];

    /// Prompts over a 3-token alphabet after the head: heavy prefix sharing,
    /// duplicates and prefix-of-another prompts all show up.
    fn prompts(rng: &mut Rng, n: usize) -> Vec<(Vec<i32>, usize)> {
        (0..n)
            .map(|_| {
                let mut t = HEAD.to_vec();
                let len = 1 + rng.below(24);
                t.extend((0..len).map(|_| 1 + rng.below(3) as i32));
                let keep = match rng.below(3) {
                    0 => 0,
                    _ => HEAD.len() + rng.below(len),
                };
                (t, keep)
            })
            .collect()
    }

    fn run(kv: &mut Kv, sim: &mut Sim, ps: &[(Vec<i32>, usize)], shared: bool) -> Stats {
        let jobs: Vec<Job> = ps.iter().map(|(t, k)| Job { toks: t, keep: *k }).collect();
        let mut got: Vec<Option<Vec<f32>>> = vec![None; ps.len()];
        let st = kv
            .run(sim, &jobs, shared, &mut |j, r| got[j] = Some(r.to_vec()))
            .unwrap();
        for (i, (t, _)) in ps.iter().enumerate() {
            assert_eq!(
                got[i].as_deref(),
                Some(&row(t)[..]),
                "job {i} got the wrong row"
            );
        }
        check(kv, sim);
        st
    }

    /// After a run: every entry holds exactly its span, every other seq is
    /// empty, the cache respects its budgets.
    fn check(kv: &Kv, sim: &Sim) {
        let entries = kv.entries();
        for s in 0..sim.n_seq as i32 {
            match entries.iter().find(|e| e.0 == s) {
                Some((_, toks)) => assert_eq!(&sim.history(s), toks, "entry seq {s}"),
                None => assert!(sim.history(s).is_empty(), "stray tokens on seq {s}"),
            }
        }
        let cached: Vec<i32> = entries.iter().map(|e| e.0).filter(|&s| s != 0).collect();
        assert!(cached.len() <= sim.n_seq / 2, "{} entries", cached.len());
        let cells = sim.cells_of(&cached, 0);
        assert!(cells <= sim.n_ctx / 2, "cache holds {cells} cells");
    }

    fn setup(n_ctx: usize, n_seq: usize, n_batch: usize) -> (Kv, Sim) {
        let mut sim = Sim::new(n_ctx, n_seq, n_batch);
        let kv = Kv::new(&mut sim, HEAD.to_vec()).unwrap();
        (kv, sim)
    }

    #[test]
    fn rows_match_from_scratch_decoding_everywhere() {
        // roomy, hybrid-sized, cramped, and barely-alive contexts
        for (n_ctx, n_seq, n_batch) in [(4096, 65, 64), (4096, 17, 7), (400, 6, 16), (160, 3, 5)] {
            let (mut kv, mut sim) = setup(n_ctx, n_seq, n_batch);
            let mut rng = Rng(0x5eed ^ n_ctx as u64);
            for round in 0..60 {
                let n = 1 + rng.below(40);
                let ps = prompts(&mut rng, n);
                run(&mut kv, &mut sim, &ps, round % 5 != 0);
            }
            // waves are sized from exact trie cell counts
            assert_eq!(sim.no_slots, 0, "ctx {n_ctx}: planner overfilled a wave");
        }
    }

    #[test]
    fn shared_text_is_decoded_once() {
        let (mut kv, mut sim) = setup(4096, 65, 512);
        let mut rng = Rng(7);
        let ps: Vec<(Vec<i32>, usize)> = prompts(&mut rng, 30)
            .into_iter()
            .map(|(t, _)| (t, 0))
            .collect();
        let st = run(&mut kv, &mut sim, &ps, true);
        // one wave, and exactly one decode per distinct prefix past the head
        let distinct: HashSet<&[i32]> = ps
            .iter()
            .flat_map(|(t, _)| (HEAD.len() + 1..=t.len()).map(move |i| &t[..i]))
            .collect();
        assert_eq!((st.waves, st.decoded), (1, distinct.len()));
        // direct mode shares nothing but the head
        let st = run(&mut kv, &mut sim, &ps, false);
        let total: usize = ps.iter().map(|(t, _)| t.len() - HEAD.len()).sum();
        assert_eq!(st.decoded, total);
    }

    #[test]
    fn kept_spans_are_reused_across_runs() {
        let (mut kv, mut sim) = setup(4096, 65, 64);
        let state: Vec<i32> = HEAD.iter().copied().chain(vec![7; 100]).collect();
        let ps: Vec<(Vec<i32>, usize)> = (0..8)
            .map(|q| {
                (
                    state.iter().copied().chain([q, q, q]).collect(),
                    state.len(),
                )
            })
            .collect();
        let cold = run(&mut kv, &mut sim, &ps, true);
        let warm = run(&mut kv, &mut sim, &ps, true);
        assert_eq!((cold.misses, cold.hits), (8, 0));
        assert_eq!((warm.misses, warm.hits), (0, 8));
        assert_eq!(cold.decoded - warm.decoded, 100);
        assert!(kv.entries().iter().any(|(_, t)| *t == state));
    }

    #[test]
    fn a_span_decoded_in_wave_one_serves_later_waves() {
        // 4 seqs: every wave fits two questions, the state goes in once
        let (mut kv, mut sim) = setup(4096, 4, 64);
        let state: Vec<i32> = HEAD.iter().copied().chain(vec![7; 200]).collect();
        let ps: Vec<(Vec<i32>, usize)> = (0..6)
            .map(|q| (state.iter().copied().chain([q, 1]).collect(), state.len()))
            .collect();
        let st = run(&mut kv, &mut sim, &ps, true);
        assert!(st.waves >= 3);
        assert!(st.decoded < 200 + 6 * 2 + 50, "state re-decoded: {st:?}");
    }

    #[test]
    fn later_waves_fork_past_the_shared_prefix() {
        // 3 usable seqs force several waves; the prompts share a state (kept
        // across requests) plus a boilerplate stem (held for this run only)
        let (mut kv, mut sim) = setup(4096, 4, 64);
        let state: Vec<i32> = HEAD.iter().copied().chain(vec![7; 200]).collect();
        let stem: Vec<i32> = state.iter().copied().chain([9; 6]).collect();
        let ps: Vec<(Vec<i32>, usize)> = (0..6)
            .map(|q| (stem.iter().copied().chain([q, 1]).collect(), state.len()))
            .collect();
        let st = run(&mut kv, &mut sim, &ps, true);
        assert!(st.waves >= 3);
        // state and stem decoded once, each question's own tokens once
        assert_eq!(st.decoded, 200 + 6 + 6 * 2);
        // the held stem was request-scoped
        assert!(kv.entries().iter().all(|e| e.1 != stem));
    }

    #[test]
    fn no_slot_mid_wave_is_retried() {
        let (mut kv, mut sim) = setup(4096, 65, 8);
        let mut rng = Rng(11);
        let ps = prompts(&mut rng, 20);
        sim.fail_call = Some(sim.calls + 3);
        run(&mut kv, &mut sim, &ps, true);
    }

    #[test]
    fn cache_is_evicted_to_make_room() {
        // every run caches a different long span; the context only holds a few
        let (mut kv, mut sim) = setup(600, 17, 32);
        for s in 0..12 {
            let state: Vec<i32> = HEAD.iter().copied().chain(vec![10 + s; 150]).collect();
            let ps: Vec<(Vec<i32>, usize)> = (0..3)
                .map(|q| (state.iter().copied().chain([q]).collect(), state.len()))
                .collect();
            run(&mut kv, &mut sim, &ps, true);
        }
    }

    #[test]
    fn oversized_prompt_is_an_error() {
        let (mut kv, mut sim) = setup(64, 4, 16);
        let long: Vec<i32> = HEAD.iter().copied().chain(vec![1; 100]).collect();
        let jobs = [Job {
            toks: &long,
            keep: 0,
        }];
        assert!(kv.run(&mut sim, &jobs, true, &mut |_, _| {}).is_err());
        let fits: Vec<i32> = HEAD.iter().copied().chain(vec![1; 40]).collect();
        run(&mut kv, &mut sim, &[(fits, 0)], true);
    }

    #[test]
    fn residue_of_an_aborted_wave_is_wiped() {
        let (mut kv, mut sim) = setup(4096, 8, 64);
        sim.scribble(5, &[1, 2, 3]);
        kv.scratch.push(5);
        let mut rng = Rng(3);
        let ps = prompts(&mut rng, 10);
        run(&mut kv, &mut sim, &ps, true);
    }
}
