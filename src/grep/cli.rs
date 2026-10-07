//! The command line of `snap grep`: its flags as clap `Args` and the dispatch
//! from flags to the pipeline (walk, BM25 recall and tree descent, both always,
//! the rerank of their union, snapshots): a search, `--eval`, or the cache
//! upkeep (`--cache`, `--gc`), where the one positional is the PATH. The model
//! flags are snap's own (`ModelArgs`, flattened in, with grep's default model):
//! this is the one place grep reaches back into `main.rs`, for them and for
//! the log setup.

use std::path::PathBuf;

use anyhow::Result;

use super::corpus::WalkOpts;
use super::{cache, eval, run, Opts, Out, Search, Tree};
use crate::{init_logs, llamac, ModelArgs};

/// ripgrep's walk, for the commands that read a tree
#[derive(clap::Args)]
struct WalkArgs {
    /// only paths matching this glob, or with a leading `!` never those (repeatable)
    #[arg(short = 'g', long = "glob")]
    glob: Vec<String>,
}

impl WalkArgs {
    fn opts(&self) -> WalkOpts {
        WalkOpts {
            globs: self.glob.clone(),
        }
    }
}

fn parse_kv(s: &str) -> std::result::Result<llamac::KvType, String> {
    llamac::KvType::parse(s).map_err(|e| e.to_string())
}

/// What only a search (or an eval) has a meaning for: it never goes with the
/// cache upkeep.
const SEARCH_ONLY: [&str; 13] = [
    "eval",
    "output",
    "json",
    "files_with_matches",
    "top",
    "lines",
    "threshold",
    "candidates",
    "regexp",
    "no_store",
    "recall_only",
    "beam",
    "color",
];

#[derive(clap::Args)]
pub struct GrepArgs {
    #[command(flatten)]
    m: ModelArgs,
    #[command(flatten)]
    w: WalkArgs,
    /// what to look for, in plain words (the PATH under --cache and --gc)
    #[arg(required_unless_present_any = ["eval", "cache", "gc"], conflicts_with = "eval")]
    query: Option<String>,
    /// directory (or single file) to search [default: .]
    #[arg(conflicts_with = "eval")]
    path: Option<PathBuf>,
    /// hits printed
    #[arg(short = 'n', long, default_value_t = 10)]
    top: usize,
    /// BM25 depth the model reranks, plus the descent's picks (0 = every chunk)
    #[arg(long, default_value_t = 64)]
    candidates: usize,
    /// probability of a sure hit; under 3 of them, the best others down to 20% fill in
    #[arg(long, default_value_t = 0.5)]
    threshold: f64,
    /// only chunks matching this regex (smart case, like rg -S)
    #[arg(short = 'e', long = "regexp")]
    regexp: Option<String>,
    /// one JSON document: hits, their source and run stats
    #[arg(long, conflicts_with_all = ["files_with_matches", "eval"])]
    json: bool,
    /// print only the paths of the hits
    #[arg(short = 'l', long = "files-with-matches", conflicts_with = "eval")]
    files_with_matches: bool,
    /// source lines per hit: 6, 20 in --json (0 = locations only)
    #[arg(long)]
    lines: Option<usize>,
    /// colors: auto (a terminal without NO_COLOR), always, never
    #[arg(long, value_parser = ["auto", "always", "never"], default_value = "auto")]
    color: String,
    /// keep no snapshots: every candidate is decoded from scratch
    #[arg(long)]
    no_store: bool,
    /// print the lexical ranking and stop: no model is loaded
    #[arg(long)]
    recall_only: bool,
    /// branches the descent opens per level, folders and files each
    #[arg(long, default_value_t = 3)]
    beam: usize,
    /// KV cache type, which snapshots are bound to: q8_0 keeps them near 10 MB
    /// a chunk, f16 is full precision at twice that
    #[arg(long, value_parser = parse_kv, default_value = "q8_0")]
    kv: llamac::KvType,
    /// report what the cache holds of PATH instead of searching
    #[arg(long, conflicts_with_all = SEARCH_ONLY)]
    cache: bool,
    /// drop from the cache what no chunk of PATH uses any more, and the stores of other engines
    #[arg(long, conflicts_with_all = SEARCH_ONLY)]
    gc: bool,
    /// score a cases JSONL (eval/grep.jsonl) over this tree instead of answering a query
    #[arg(long)]
    eval: Option<String>,
    /// write the full eval report JSON (create-only)
    #[arg(long)]
    output: Option<String>,
}

impl GrepArgs {
    pub fn run(&self) -> Result<()> {
        let Self {
            m,
            w,
            query,
            path,
            top,
            candidates,
            threshold,
            regexp,
            json,
            files_with_matches,
            lines,
            color,
            no_store,
            recall_only,
            beam,
            kv,
            cache: report,
            gc,
            eval: eval_file,
            output,
        } = self;
        init_logs(m.debug);
        if *report || *gc {
            // the way `rg --files src/` reads its positionals: the first is the path
            anyhow::ensure!(
                query.is_none() || path.is_none(),
                "--cache and --gc take a PATH, not a QUERY and a PATH"
            );
            let root = query.as_ref().map(PathBuf::from).or(path.clone());
            let tree = Tree {
                root: root.unwrap_or(".".into()),
                walk: w.opts(),
                kv: *kv,
            };
            return cache(&tree, *report, *gc, || m.load(*kv));
        }
        let opts = Opts {
            tree: Tree {
                root: path.clone().unwrap_or(".".into()),
                walk: w.opts(),
                kv: *kv,
            },
            regexp: regexp.clone(),
            top: *top,
            candidates: *candidates,
            threshold: *threshold,
            store: !*no_store,
            recall_only: *recall_only,
            beam: *beam,
        };
        let load = || m.load(*kv);
        match (eval_file, query) {
            (Some(file), _) => eval(file, output.as_deref(), &opts, load)?,
            (None, Some(query)) => {
                // clap waives requires = "eval" while the QUERY, which conflicts with it, is given
                anyhow::ensure!(output.is_none(), "--output is the report of --eval");
                let out = match (json, files_with_matches) {
                    (true, _) => Out::Json,
                    (_, true) => Out::Files,
                    _ => Out::Human,
                };
                let search = Search {
                    query: query.clone(),
                    out,
                    lines: *lines,
                    color: match color.as_str() {
                        "always" => Some(true),
                        "never" => Some(false),
                        _ => None,
                    },
                    opts,
                };
                // grep's status: 1 when nothing was found
                if !run(&search, load)? {
                    std::process::exit(1);
                }
            }
            (None, None) => anyhow::bail!("grep needs a QUERY (or --eval FILE)"),
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use clap::{Parser, Subcommand};

    use super::*;

    /// Just the subcommand, as `snap` mounts it.
    #[derive(Parser)]
    struct Cli {
        #[command(subcommand)]
        cmd: Cmd,
    }

    #[derive(Subcommand)]
    enum Cmd {
        Grep(GrepArgs),
    }

    fn grep(args: &[&str]) -> std::result::Result<GrepArgs, clap::Error> {
        let Cmd::Grep(a) =
            Cli::try_parse_from(std::iter::once("snap").chain(args.iter().copied()))?.cmd;
        Ok(a)
    }

    #[test]
    fn grep_takes_a_query_or_an_eval_file_never_both() {
        assert!(grep(&["grep", "where is it"]).is_ok());
        assert!(grep(&["grep", "--eval", "cases.jsonl"]).is_ok());
        assert!(grep(&["grep", "--eval", "cases.jsonl", "--output", "r.json"]).is_ok());
        assert!(grep(&["grep"]).is_err());
        assert!(grep(&["grep", "--eval", "cases.jsonl", "where"]).is_err());
        // what only a query's output has: no meaning next to --eval
        assert!(grep(&["grep", "--eval", "cases.jsonl", "--json"]).is_err());
        assert!(grep(&["grep", "--eval", "cases.jsonl", "-l"]).is_err());
        assert!(grep(&["grep", "q", "--json", "-l"]).is_err());
    }

    #[test]
    fn cache_and_gc_take_a_path_and_nothing_a_search_has() {
        for mode in ["--cache", "--gc"] {
            assert!(grep(&["grep", mode]).is_ok());
            assert!(grep(&["grep", mode, "src/"]).is_ok());
            assert!(grep(&["grep", mode, "-g", "*.rs", "--kv", "f16", "--model", "m"]).is_ok());
            for no in [
                &["--eval", "c.jsonl"][..],
                &["--output", "r.json"],
                &["--json"],
                &["-l"],
                &["-n", "3"],
                &["--lines", "2"],
                &["--threshold", "0.3"],
                &["--candidates", "8"],
                &["-e", "fn"],
                &["--no-store"],
                &["--recall-only"],
                &["--beam", "2"],
                &["--color", "never"],
            ] {
                let mut line = vec!["grep", mode];
                line.extend(no);
                assert!(grep(&line).is_err(), "{mode} {no:?}");
            }
        }
        // both at once: gc first, then the report
        let both = grep(&["grep", "--cache", "--gc", "src"]).unwrap();
        assert!(both.cache && both.gc);
        // the first positional is the query as far as clap goes; run() reads it as the path
        let a = grep(&["grep", "--gc", "src"]).unwrap();
        assert_eq!((a.query.as_deref(), a.path), (Some("src"), None));
    }

    #[test]
    fn two_positionals_under_cache_or_gc_are_refused_by_run() {
        let a = grep(&["grep", "--gc", "a", "b"]).unwrap();
        let e = a.run().unwrap_err().to_string();
        assert!(e.contains("PATH, not a QUERY and a PATH"), "{e}");
    }

    #[test]
    fn grep_flags_land_in_their_fields() {
        let line = "grep q src -n 3 --candidates 0 --threshold 0.7 -e fn -g *.rs -g !x/** \
                    -l --lines 5 --no-store --recall-only --beam 5 --kv f16 \
                    --model m.gguf";
        let GrepArgs {
            m,
            w,
            query,
            path,
            top,
            candidates,
            threshold,
            regexp,
            files_with_matches,
            lines,
            no_store,
            recall_only,
            beam,
            kv,
            ..
        } = grep(&line.split_whitespace().collect::<Vec<_>>()).unwrap();
        assert_eq!((query.as_deref(), path), (Some("q"), Some("src".into())));
        assert_eq!((top, candidates, threshold, lines), (3, 0, 0.7, Some(5)));
        assert_eq!(regexp.as_deref(), Some("fn"));
        assert!(files_with_matches && no_store && recall_only);
        assert_eq!(beam, 5);
        assert_eq!(kv, llamac::KvType::F16);
        assert_eq!(m.model, "m.gguf");
        assert_eq!(w.opts().globs, ["*.rs", "!x/**"]);
    }

    #[test]
    fn grep_defaults_to_q8_0_and_ten_hits_of_sixty_four_candidates() {
        let GrepArgs {
            top,
            candidates,
            threshold,
            lines,
            kv,
            path,
            beam,
            m,
            ..
        } = grep(&["grep", "q"]).unwrap();
        assert_eq!((top, candidates, threshold, lines), (10, 64, 0.5, None));
        assert_eq!((beam, m.model.as_str()), (3, crate::models::DEFAULT_MODEL));
        assert_eq!((kv, path), (llamac::KvType::Q8_0, None));
        assert!(grep(&["grep", "q", "--kv", "f32"]).is_err());
    }
}
