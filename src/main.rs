mod api;
mod bench;
mod calibrate;
mod decisions;
mod engine;
mod evaluate;
mod instances;
mod kv;
mod llamac;
mod models;
mod prompts;
mod schema;
mod server;
mod version;

use std::io::{IsTerminal, Read};
use std::path::Path;

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
    /// list known model shortcuts
    Models,
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
    let path = models::resolve(&m.model)?;
    let mut eng = engine::Engine::load(path.to_string_lossy().as_ref(), m.ctx, 1024, m.threads)?;
    if let Some(c) = &m.calibration {
        eng.load_calibration(c)?;
    }
    let out = api::from_native(&eng.decide(&req.to_native())?);
    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}

fn parse_layout(s: &str) -> std::result::Result<crate::schema::Layout, String> {
    serde_json::from_value::<crate::schema::Layout>(serde_json::json!(s))
        .map_err(|_| "expected auto|state_first|question_first|header|catalog".to_string())
}

/// `snap serve` body — model load + warmup + blocking axum loop.
fn serve(m: &ModelArgs, host: &str, port: u16) -> Result<()> {
    eprintln!("snap: loading {} …", m.model);
    let path = models::resolve(&m.model)?;
    let mut eng = engine::Engine::load(path.to_string_lossy().as_ref(), m.ctx, 1024, m.threads)?;
    if let Some(c) = &m.calibration {
        eng.load_calibration(c)?;
    }
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
    if cli.m.model != models::DEFAULT_MODEL
        || cli.m.ctx != 8192
        || cli.m.threads != 0
        || cli.m.debug
        || cli.m.calibration.is_some()
    {
        anyhow::bail!("model flags belong after the subcommand (top-level is -p only)");
    }
    version::nag(); // stderr only, ~once a day — never in -p mode
    match cmd {
        Cmd::Models => {
            for (name, repo, file) in models::MODELS {
                println!("  {name:14} {repo}/{file}");
            }
        }
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
        Cmd::Calibrate { m, files, output } => {
            init_logs(m.debug);
            if files.is_empty() {
                anyhow::bail!("calibrate needs at least one eval/*.jsonl file");
            }
            let path = models::resolve(&m.model)?;
            let mut eng =
                engine::Engine::load(path.to_string_lossy().as_ref(), m.ctx, 1024, m.threads)?;
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
            output,
        } => {
            init_logs(m.debug);
            let cases = evaluate::load_cases(file)?;
            let rep = if let Some(url) = url {
                evaluate::evaluate_url(url, &cases, *limit, *no_abstain, !*no_perturb, *layout)?
            } else {
                let path = models::resolve(&m.model)?;
                let mut eng =
                    engine::Engine::load(path.to_string_lossy().as_ref(), m.ctx, 1024, m.threads)?;
                if let Some(c) = &m.calibration {
                    eng.load_calibration(c)?;
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
                let path = models::resolve(&m.model)?;
                let mut eng =
                    engine::Engine::load(path.to_string_lossy().as_ref(), m.ctx, 1024, m.threads)?;
                if let Some(c) = &m.calibration {
                    eng.load_calibration(c)?;
                }
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
