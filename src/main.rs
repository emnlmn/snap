mod api;
mod bench;
mod calibrate;
mod corpus;
mod decisions;
mod engine;
mod evaluate;
mod grep;
mod instances;
mod kv;
mod kvstore;
mod lexical;
mod llamac;
mod models;
mod prompts;
mod schema;
mod server;
mod version;

use std::io::{IsTerminal, Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{CommandFactory, Parser, Subcommand};

const DEFAULT_PORT: u16 = 8018;

#[derive(Parser)]
#[command(
    name = "snap",
    about = "snap: typed decisions from a single forward pass",
    version
)]
struct Cli {
    #[command(subcommand)]
    cmd: Option<Cmd>,
    /// print mode: one request, one response, exit (like `claude -p`)
    #[arg(short = 'p', long = "print")]
    print: bool,
    /// request for -p: inline JSON, a .json path, or '-' / empty for stdin
    request: Option<String>,
    #[command(flatten)]
    m: ModelArgs,
}

#[derive(clap::Args)]
struct ModelArgs {
    /// model name (see `snap models` — tested set only)
    #[arg(long, default_value = models::DEFAULT_MODEL)]
    model: String,
    #[arg(long, default_value_t = 8192)]
    ctx: i32,
    /// calibration file from `snap calibrate`
    #[arg(long)]
    calibration: Option<String>,
    /// decode threads (0 = all available cores; llama.cpp's own default is 4)
    #[arg(long, default_value_t = 0)]
    threads: i32,
    /// engine + llama.cpp internals on stderr
    #[arg(long)]
    debug: bool,
}

impl ModelArgs {
    /// resolve --model (pulling on first use), load the engine with its KV
    /// cache as `kv`, fit --calibration when given. The GGUF's path comes
    /// along: its size and mtime bind the snapshots of `snap grep`.
    fn load(&self, kv: llamac::KvType) -> Result<(engine::Engine, PathBuf)> {
        let path = models::resolve(&self.model)?;
        let mut eng = engine::Engine::load_kv(&path, self.ctx, 1024, self.threads, kv)?;
        if let Some(c) = &self.calibration {
            eng.load_calibration(c)?;
        }
        Ok((eng, path))
    }

    fn engine(&self) -> Result<engine::Engine> {
        Ok(self.load(llamac::KvType::F16)?.0)
    }

    /// untouched top-level flags — the -p guard's check
    fn is_default(&self) -> bool {
        self.model == models::DEFAULT_MODEL
            && self.ctx == 8192
            && self.threads == 0
            && !self.debug
            && self.calibration.is_none()
    }
}

/// ripgrep's walk, for the commands that read a tree
#[derive(clap::Args)]
struct WalkArgs {
    /// only paths matching this glob, or with a leading `!` never those (repeatable)
    #[arg(short = 'g', long = "glob")]
    glob: Vec<String>,
    /// search hidden files and directories
    #[arg(long)]
    hidden: bool,
    /// don't respect .gitignore, .ignore and .rgignore
    #[arg(long)]
    no_ignore: bool,
}

impl WalkArgs {
    fn opts(&self) -> corpus::WalkOpts {
        corpus::WalkOpts {
            hidden: self.hidden,
            no_ignore: self.no_ignore,
            globs: self.glob.clone(),
        }
    }
}

#[derive(Subcommand)]
enum Cmd {
    /// HTTP server (decisions + systemone APIs)
    Serve {
        #[command(flatten)]
        m: ModelArgs,
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        #[arg(long, default_value_t = DEFAULT_PORT)]
        port: u16,
    },
    /// find the code that answers a question: lexical recall, then the model reads each candidate once
    Grep {
        #[command(flatten)]
        m: ModelArgs,
        #[command(flatten)]
        w: WalkArgs,
        /// what to look for, in plain words
        #[arg(required_unless_present = "eval", conflicts_with = "eval")]
        query: Option<String>,
        /// directory (or single file) to search
        #[arg(default_value = ".", conflicts_with = "eval")]
        path: PathBuf,
        /// hits printed
        #[arg(short = 'n', long, default_value_t = 10)]
        top: usize,
        /// recall depth the model reranks (0 = every chunk)
        #[arg(long, default_value_t = 64)]
        candidates: usize,
        /// least probability a hit may have
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
        /// source lines per hit, in --json too (0 = locations only)
        #[arg(long, default_value_t = 20)]
        lines: usize,
        /// keep no snapshots: every candidate is decoded from scratch
        #[arg(long)]
        no_store: bool,
        /// print the lexical ranking and stop: no model is loaded
        #[arg(long)]
        recall_only: bool,
        /// KV cache type; snapshots are bound to it (f16|q8_0)
        #[arg(long, value_parser = parse_kv, default_value = "f16")]
        kv: llamac::KvType,
        /// score a cases JSONL (eval/grep.jsonl) over this tree instead of answering a query
        #[arg(long)]
        eval: Option<String>,
        /// write the full eval report JSON (create-only)
        #[arg(long)]
        output: Option<String>,
    },
    /// snapshot every chunk of a tree ahead of time, so `snap grep` over it starts warm
    Index {
        #[command(flatten)]
        m: ModelArgs,
        #[command(flatten)]
        w: WalkArgs,
        /// directory (or single file) to index
        #[arg(default_value = ".")]
        path: PathBuf,
        /// KV cache type; snapshots are bound to it (f16|q8_0)
        #[arg(long, value_parser = parse_kv, default_value = "f16")]
        kv: llamac::KvType,
        /// report what the store holds of the tree instead of indexing it
        #[arg(long)]
        stats: bool,
        /// drop the snapshots no chunk of the tree uses any more
        #[arg(long)]
        gc: bool,
    },
    /// run a JSONL benchmark and report accuracy/latency
    Evaluate {
        #[command(flatten)]
        m: ModelArgs,
        /// eval/*.jsonl
        file: String,
        /// hit a live server instead of loading a model
        #[arg(long)]
        url: Option<String>,
        #[arg(long)]
        limit: Option<usize>,
        /// force allow_abstain=false on every case
        #[arg(long)]
        no_abstain: bool,
        /// skip auto-generated stability probes (reversal, rewording, noise)
        #[arg(long)]
        no_perturb: bool,
        /// force a prompt layout for all cases (auto|state_first|question_first|header|catalog)
        #[arg(long, value_parser = parse_layout)]
        layout: Option<crate::schema::Layout>,
        /// force a state renderer for all cases (json|toon) — debug A/B knob
        #[arg(long, value_parser = parse_state_format)]
        state_format: Option<engine::StateFormat>,
        /// write full report JSON (create-only)
        #[arg(long)]
        output: Option<String>,
    },
    /// latency/throughput benchmark (single, prefix, state-size, load)
    Bench {
        #[command(flatten)]
        m: ModelArgs,
        /// hit a live server instead of loading a model
        #[arg(long)]
        url: Option<String>,
        #[arg(long, default_value_t = 20)]
        requests: usize,
        #[arg(long, default_value_t = 8)]
        concurrency: usize,
        /// write report JSON (create-only)
        #[arg(long)]
        output: Option<String>,
    },
    /// dump rendered prompts + letter slots of JSONL cases (training export)
    ExportPrompts {
        #[command(flatten)]
        m: ModelArgs,
        /// eval/training JSONL case files
        files: Vec<String>,
        /// force a prompt layout for all cases (auto|state_first|question_first|header|catalog)
        #[arg(long, value_parser = parse_layout)]
        layout: Option<crate::schema::Layout>,
        /// write JSONL (create-only)
        #[arg(long)]
        output: Option<String>,
    },
    /// fit per-type temperatures on labeled eval cases -> calibration file
    Calibrate {
        #[command(flatten)]
        m: ModelArgs,
        /// eval/*.jsonl files
        files: Vec<String>,
        /// write calibration JSON (create-only)
        #[arg(long, short)]
        output: String,
    },
    /// inspect and manage the local model cache (default: list)
    Models {
        #[command(subcommand)]
        cmd: Option<ModelCmd>,
    },
    /// ping a running server
    Check {
        #[arg(long, default_value = "http://127.0.0.1:8018")]
        url: String,
    },
    /// list running snap servers
    Ps,
    /// stop running servers (no args: the one server, else see `snap ps`)
    Stop {
        /// stop every running snap server
        #[arg(long)]
        all: bool,
        /// stop the server on this port (repeatable)
        #[arg(long, conflicts_with = "all")]
        port: Vec<u16>,
        /// stop the server with this pid (repeatable)
        #[arg(long, conflicts_with = "all")]
        pid: Vec<u32>,
        /// don't wait for a graceful drain — SIGKILL
        #[arg(long)]
        force: bool,
    },
}

#[derive(Subcommand)]
enum ModelCmd {
    /// tested models: name, size on disk, where they live
    #[command(visible_alias = "ls")]
    List,
    /// download a model's GGUF (a bare --model name pulls on first use too)
    Pull { name: String },
    /// delete a pulled model's GGUF from the cache
    #[command(visible_alias = "remove")]
    Rm { name: String },
}

/// The request behind -p / decide: inline JSON wins, then a path that
/// exists, then stdin (the unix-filter form).
fn read_request(spec: Option<&str>) -> Result<String> {
    match spec {
        None | Some("-") => {
            if std::io::stdin().is_terminal() {
                anyhow::bail!("no request: pass a JSON string, a .json path, or pipe one on stdin");
            }
            let mut s = String::new();
            std::io::stdin().read_to_string(&mut s)?;
            Ok(s)
        }
        Some(s) => {
            let t = s.trim_start();
            if t.starts_with('{') {
                Ok(t.to_string())
            } else if Path::new(t).is_file() {
                Ok(std::fs::read_to_string(t)?)
            } else {
                // neither object-shaped nor a file — let serde say what's wrong
                Ok(s.to_string())
            }
        }
    }
}

fn decide_request(m: &ModelArgs, req: api::SystemoneRequest) -> Result<()> {
    let mut eng = m.engine()?;
    let out = api::from_native(&eng.decide(&req.to_native())?);
    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}

fn parse_layout(s: &str) -> std::result::Result<crate::schema::Layout, String> {
    serde_json::from_value::<crate::schema::Layout>(serde_json::json!(s))
        .map_err(|_| "expected auto|state_first|question_first|header|catalog".to_string())
}

fn parse_state_format(s: &str) -> std::result::Result<engine::StateFormat, String> {
    engine::StateFormat::parse(s).map_err(|e| e.to_string())
}

fn parse_kv(s: &str) -> std::result::Result<llamac::KvType, String> {
    llamac::KvType::parse(s).map_err(|e| e.to_string())
}

/// `snap serve` body — model load + warmup + blocking axum loop.
fn serve(m: &ModelArgs, host: &str, port: u16) -> Result<()> {
    let mut eng = m.engine()?;
    // warm the backend pipelines before the port opens — the first
    // real request shouldn't pay shader-compile + buffer setup
    if let Err(e) = eng.warmup() {
        eprintln!("snap: warmup failed: {e}");
    }
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    rt.block_on(server::serve(eng, &m.model, m.ctx, m.threads, host, port))
}

fn print_ps(live: &[instances::Instance]) {
    println!(
        "{:<7} {:<15} {:<22} {:<8} STATE",
        "PID", "MODEL", "ADDRESS", "UPTIME"
    );
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    for i in live {
        let (uptime, model, state) = match &i.state {
            instances::State::Serving { uptime_s, model } => {
                (instances::fmt_uptime(*uptime_s), model.as_str(), "serving")
            }
            instances::State::Starting => (
                instances::fmt_uptime(now.saturating_sub(i.info.started)),
                i.info.model.as_str(),
                "starting",
            ),
            instances::State::Unresponsive => (
                instances::fmt_uptime(now.saturating_sub(i.info.started)),
                i.info.model.as_str(),
                "unresponsive",
            ),
        };
        let pid = if i.info.pid == 0 {
            "-".to_string()
        } else {
            i.info.pid.to_string()
        };
        println!(
            "{:<7} {:<15} {:<22} {:<8} {}",
            pid,
            model,
            format!("{}:{}", i.info.host, i.info.port),
            uptime,
            state
        );
    }
}

/// `snap models` — name, disk size, real path (docker-images-style).
fn print_models() {
    println!("{:<14} {:>7}  PATH", "NAME", "SIZE");
    for &(name, repo, file) in models::MODELS {
        match models::cached(repo, file) {
            Some(p) => {
                let size = p.metadata().map(|m| m.len()).unwrap_or(0);
                println!("{name:<14} {:>7}  {}", fmt_bytes(size), p.display());
            }
            None => println!("{name:<14} {:>7}  -", "-"),
        }
    }
}

fn fmt_bytes(b: u64) -> String {
    const GB: u64 = 1_000_000_000;
    if b >= GB {
        format!("{:.1} GB", b as f64 / GB as f64)
    } else {
        format!("{} MB", b / 1_000_000)
    }
}

/// `snap stop`: pick targets, SIGTERM, wait, escalate on unix.
fn stop(all: bool, port: &[u16], pid: &[u32], force: bool) -> Result<()> {
    use std::collections::BTreeSet;
    let live = instances::list();
    let mut targets: BTreeSet<u32> = BTreeSet::new();
    if all {
        targets.extend(live.iter().map(|i| i.info.pid));
    } else {
        for p in port {
            match live.iter().find(|i| i.info.port == *p) {
                Some(i) => drop(targets.insert(i.info.pid)),
                None => anyhow::bail!("no snap server on :{p}"),
            }
        }
        for p in pid {
            match live.iter().find(|i| i.info.pid == *p) {
                Some(i) => drop(targets.insert(i.info.pid)),
                None => anyhow::bail!("no snap server with pid {p}"),
            }
        }
        if port.is_empty() && pid.is_empty() {
            match live.as_slice() {
                [] => {}
                [i] => drop(targets.insert(i.info.pid)),
                _ => {
                    print_ps(&live);
                    anyhow::bail!(
                        "multiple snap servers — `snap stop --all`, or pick one with --port/--pid"
                    );
                }
            }
        }
    }
    if targets.is_empty() {
        println!("no snap servers running");
        return Ok(());
    }
    for pid in targets {
        let i = live.iter().find(|i| i.info.pid == pid).unwrap();
        if pid == 0 {
            eprintln!(
                ":{} is serving but not tracked (started outside this snap) — kill it manually",
                i.info.port
            );
            continue;
        }
        if !force && !instances::confirmed(i) {
            eprintln!(
                "pid {pid} isn't verified as a snap server (pid reuse?) — `--force` to kill anyway"
            );
            continue;
        }
        if instances::kill(pid, force)? {
            instances::unregister(pid);
            println!("stopped :{} (pid {pid})", i.info.port);
        } else {
            eprintln!("pid {pid} still alive — retry with `--force`");
        }
    }
    Ok(())
}

fn init_logs(debug: bool) {
    llamac::route_logs_to_tracing();
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                (if debug {
                    "snap=debug,llamac=debug,info"
                } else {
                    "warn"
                })
                .into()
            }),
        )
        .with_writer(std::io::stderr)
        .init();
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let Some(cmd) = &cli.cmd else {
        if !cli.print {
            if cli.request.is_none() {
                if !std::io::stdin().is_terminal() {
                    anyhow::bail!("piped input detected — did you mean `snap -p`?");
                }
                Cli::command().print_help()?;
                return Ok(());
            }
            anyhow::bail!("a request requires -p");
        }
        init_logs(cli.m.debug);
        let src = read_request(cli.request.as_deref())?;
        let req: api::SystemoneRequest = serde_json::from_str(&src)
            .context("request must be a JSON object (inline string, .json path, or stdin)")?;
        return decide_request(&cli.m, req);
    };
    if cli.print || cli.request.is_some() {
        anyhow::bail!("-p/--print takes no subcommand");
    }
    if !cli.m.is_default() {
        anyhow::bail!("model flags belong after the subcommand (top-level is -p only)");
    }
    version::nag(); // stderr only, ~once a day — never in -p mode
    match cmd {
        Cmd::Models { cmd } => match cmd {
            None | Some(ModelCmd::List) => print_models(),
            Some(ModelCmd::Pull { name }) => {
                println!("{name}: {}", models::pull(name)?.display());
            }
            Some(ModelCmd::Rm { name }) => {
                if let Some(i) = instances::list().iter().find(|i| i.info.model == *name) {
                    anyhow::bail!(
                        "{name} is serving on :{0} — `snap stop --port {0}` first",
                        i.info.port
                    );
                }
                println!("{name}: removed ({})", fmt_bytes(models::remove(name)?));
            }
        },
        Cmd::Check { url } => {
            let out: serde_json::Value =
                ureq::get(&format!("{}/healthz", url.trim_end_matches('/')))
                    .call()
                    .with_context(|| {
                        format!("no snap server at {url} — `snap ps` lists running ones")
                    })?
                    .body_mut()
                    .read_json()?;
            println!("{}", serde_json::to_string(&out)?);
        }
        Cmd::Ps => {
            let live = instances::list();
            if live.is_empty() {
                println!("no snap servers running");
            } else {
                print_ps(&live);
            }
        }
        Cmd::Stop {
            all,
            port,
            pid,
            force,
        } => stop(*all, port, pid, *force)?,
        Cmd::ExportPrompts {
            m,
            files,
            layout,
            output,
        } => {
            init_logs(m.debug);
            if files.is_empty() {
                anyhow::bail!("export-prompts needs at least one JSONL file");
            }
            let mut cases = Vec::new();
            for f in files {
                cases.extend(evaluate::load_cases(f)?);
            }
            // the model loads in here: an existing --output fails before it
            let export = |w: &mut dyn Write| {
                let eng =
                    engine::Engine::load(&models::resolve(&m.model)?, m.ctx, 1024, m.threads)?;
                evaluate::export_prompts(&eng, &cases, *layout, w)
            };
            match output {
                Some(o) => {
                    let (n, _) = evaluate::create_only(o, export)?;
                    eprintln!("{n} prompts written: {o}");
                }
                None => {
                    let mut w = std::io::BufWriter::new(std::io::stdout().lock());
                    export(&mut w)?;
                    w.flush()?;
                }
            }
        }

        Cmd::Calibrate { m, files, output } => {
            init_logs(m.debug);
            if files.is_empty() {
                anyhow::bail!("calibrate needs at least one eval/*.jsonl file");
            }
            let mut eng = m.engine()?;
            let mut cases = Vec::new();
            for f in files {
                cases.extend(evaluate::load_cases(f)?);
            }
            let cal = calibrate::fit(&mut eng, &cases)?;
            println!("fitted on {} cases ({} skipped)", cal.fitted, cal.skipped);
            for (t, temp) in &cal.temperatures {
                println!("  {t:8} T={temp:.3}");
            }
            println!(
                "ece  {:.3} raw -> {:.3} in-sample | {:.3} out-of-fold  CI95 [{:.3}, {:.3}]",
                cal.ece_before, cal.ece_after, cal.ece_oof, cal.ci_oof.0, cal.ci_oof.1
            );
            if cal.ci_separated {
                println!("gain is CI-separated from raw at 95% — real, not fitting noise");
            } else {
                println!("caveat: OOF interval overlaps raw ECE — gain not proven at this n");
            }
            calibrate::write(&cal, output)?;
        }

        Cmd::Evaluate {
            m,
            file,
            url,
            limit,
            no_abstain,
            no_perturb,
            layout,
            state_format,
            output,
        } => {
            init_logs(m.debug);
            let cases = evaluate::load_cases(file)?;
            let rep = if let Some(url) = url {
                if state_format.is_some() {
                    anyhow::bail!(
                        "--state-format is in-process only; set SNAP_STATE_FORMAT on the server"
                    );
                }
                evaluate::evaluate_url(url, &cases, *limit, *no_abstain, !*no_perturb, *layout)?
            } else {
                let mut eng = m.engine()?;
                if let Some(f) = state_format {
                    eng.state_format = Some(*f);
                }
                evaluate::evaluate(&mut eng, &cases, *limit, *no_abstain, !*no_perturb, *layout)?
            };
            evaluate::print_report(&rep);
            if let Some(o) = output {
                evaluate::write_report(&rep, o)?;
            }
        }
        Cmd::Bench {
            m,
            url,
            requests,
            concurrency,
            output,
        } => {
            init_logs(m.debug);
            let rows = if let Some(url) = url {
                bench::run_http(url, *requests, *concurrency)?
            } else {
                let mut eng = m.engine()?;
                bench::run_local(&mut eng, *requests)?
            };
            bench::print_bench(&rows);
            if let Some(o) = output {
                if Path::new(o).exists() {
                    anyhow::bail!("{o} exists (reports are create-only)");
                }
                if let Some(d) = Path::new(o).parent() {
                    std::fs::create_dir_all(d)?;
                }
                std::fs::write(o, serde_json::to_string_pretty(&rows)?)?;
                eprintln!("report written: {o}");
            }
        }
        Cmd::Grep {
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
            no_store,
            recall_only,
            kv,
            eval,
            output,
        } => {
            init_logs(m.debug);
            let opts = grep::Opts {
                tree: grep::Tree {
                    root: path.clone(),
                    walk: w.opts(),
                    kv: *kv,
                },
                regexp: regexp.clone(),
                top: *top,
                candidates: *candidates,
                threshold: *threshold,
                store: !*no_store,
                recall_only: *recall_only,
            };
            let load = || m.load(*kv);
            match (eval, query) {
                (Some(file), _) => grep::eval(file, output.as_deref(), &opts, load)?,
                (None, Some(query)) => {
                    // clap waives requires = "eval" while the QUERY, which conflicts with it, is given
                    anyhow::ensure!(output.is_none(), "--output is the report of --eval");
                    let out = match (json, files_with_matches) {
                        (true, _) => grep::Out::Json,
                        (_, true) => grep::Out::Files,
                        _ => grep::Out::Human,
                    };
                    let search = grep::Search {
                        query: query.clone(),
                        out,
                        lines: *lines,
                        opts,
                    };
                    grep::run(&search, load)?;
                }
                (None, None) => anyhow::bail!("grep needs a QUERY (or --eval FILE)"),
            }
        }
        Cmd::Index {
            m,
            w,
            path,
            kv,
            stats,
            gc,
        } => {
            init_logs(m.debug);
            let tree = grep::Tree {
                root: path.clone(),
                walk: w.opts(),
                kv: *kv,
            };
            grep::index(&tree, *stats, *gc, || m.load(*kv))?;
        }
        Cmd::Serve { m, host, port } => {
            init_logs(m.debug);
            if let Some(i) = instances::on_port(*port) {
                let who = if i.info.pid == 0 {
                    "untracked".to_string()
                } else {
                    format!("pid {}", i.info.pid)
                };
                anyhow::bail!(
                    "snap is already serving on :{port} ({who}) — \
                     `snap stop --port {port}` or pick another --port"
                );
            }
            // registered before the slow load so `snap ps` shows "starting"
            instances::register(host, *port, &m.model)?;
            let r = serve(m, host, *port);
            instances::unregister(std::process::id());
            r?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> std::result::Result<Cli, clap::Error> {
        Cli::try_parse_from(std::iter::once("snap").chain(args.iter().copied()))
    }

    #[test]
    fn the_cli_definition_is_sound() {
        Cli::command().debug_assert();
    }

    #[test]
    fn grep_takes_a_query_or_an_eval_file_never_both() {
        assert!(parse(&["grep", "where is it"]).is_ok());
        assert!(parse(&["grep", "--eval", "cases.jsonl"]).is_ok());
        assert!(parse(&["grep", "--eval", "cases.jsonl", "--output", "r.json"]).is_ok());
        assert!(parse(&["grep"]).is_err());
        assert!(parse(&["grep", "--eval", "cases.jsonl", "where"]).is_err());
        // what only a query's output has: no meaning next to --eval
        assert!(parse(&["grep", "--eval", "cases.jsonl", "--json"]).is_err());
        assert!(parse(&["grep", "--eval", "cases.jsonl", "-l"]).is_err());
        assert!(parse(&["grep", "q", "--json", "-l"]).is_err());
    }

    #[test]
    fn grep_flags_land_in_their_fields() {
        let line = "grep q src -n 3 --candidates 0 --threshold 0.7 -e fn -g *.rs -g !x/** \
                    --hidden --no-ignore -l --lines 5 --no-store --recall-only --kv q8_0 \
                    --model m.gguf";
        let cli = parse(&line.split_whitespace().collect::<Vec<_>>()).unwrap();
        let Some(Cmd::Grep {
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
            kv,
            ..
        }) = cli.cmd
        else {
            panic!("not a grep");
        };
        assert_eq!((query.as_deref(), path), (Some("q"), PathBuf::from("src")));
        assert_eq!((top, candidates, threshold, lines), (3, 0, 0.7, 5));
        assert_eq!(regexp.as_deref(), Some("fn"));
        assert!(files_with_matches && no_store && recall_only);
        assert_eq!(kv, llamac::KvType::Q8_0);
        assert_eq!(m.model, "m.gguf");
        let walk = w.opts();
        assert!(walk.hidden && walk.no_ignore);
        assert_eq!(walk.globs, ["*.rs", "!x/**"]);
    }

    #[test]
    fn grep_and_index_default_to_f16_and_ten_hits_of_sixty_four_candidates() {
        let Some(Cmd::Grep {
            top,
            candidates,
            threshold,
            lines,
            kv,
            path,
            ..
        }) = parse(&["grep", "q"]).unwrap().cmd
        else {
            panic!("not a grep");
        };
        assert_eq!((top, candidates, threshold, lines), (10, 64, 0.5, 20));
        assert_eq!((kv, path), (llamac::KvType::F16, PathBuf::from(".")));
        let Some(Cmd::Index { kv, stats, gc, .. }) = parse(&["index"]).unwrap().cmd else {
            panic!("not an index");
        };
        assert_eq!((kv, stats, gc), (llamac::KvType::F16, false, false));
        assert!(parse(&["index", "--kv", "f32"]).is_err());
    }
}
