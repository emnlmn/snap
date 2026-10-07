//! The grep side of the engine: how a chunk becomes a prompt and a score.
//!
//! A chunk is the state of a one-question `state_first` request, compiled by
//! the engine's own step, so its `[head + STATE block]` prefix does not depend
//! on the query. `grep_snapshot` decodes that prefix once and hands out
//! whole-seq snapshots, `grep_score` restores one and decodes only the question
//! tail: the same row, hence the same P(yes), as `decide` reads from scratch.
//! `GREP_FORMAT` bumps with any change to the STATE block, because the stored
//! snapshots hold exactly those tokens.

use anyhow::{Context, Result};
use serde_json::{json, Map};

use super::corpus::MAX_CHARS;
use crate::engine::{Engine, Reader};
use crate::kv::{self, Backend, Job, Kv, Restore, Room};
use crate::schema::{DecideRequest, Expand, Layout, Mode};

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
        let (toks, at) = self.probe_prompt(&grep_request(state, query, prose))?;
        Ok(Probe { toks, at })
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

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use super::*;
    use crate::calibrate::Calibration;
    use crate::kv::sim::{blob, Sim};
    use crate::kv::KV_SLACK;
    use crate::prompts;

    fn engine(n_ctx: usize, n_seq: usize, n_batch: usize) -> Engine {
        let sim = Sim::new(n_ctx, n_seq, n_batch);
        Engine::new(Box::new(sim), "snap-sim".into()).unwrap()
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
