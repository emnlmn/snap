mod api;
mod bench;
mod calibrate;
mod decisions;
mod engine;
mod evaluate;
mod llamac;
mod models;
mod prompts;
mod schema;
mod server;

use std::io::{IsTerminal, Read};
use std::path::Path;

use anyhow::{Context, Result};
use clap::{CommandFactory, Parser, Subcommand};

const DEFAULT_PORT: u16 = 8018;

#[derive(Parser)]
#[command(
    name = "snap",
    about = "snap: typed decisions from a single forward pass"
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
    let mut eng = engine::Engine::new(path.to_string_lossy().as_ref(), m.ctx, 1024)?;
    if let Some(c) = &m.calibration {
        eng.load_calibration(c)?;
    }
    let out = api::from_native(&eng.decide(&req.to_native())?);
    println!("{}", serde_json::to_string_pretty(&out)?);
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
        || cli.m.debug
        || cli.m.calibration.is_some()
    {
        anyhow::bail!("model flags belong after the subcommand (top-level is -p only)");
    }
    match cmd {
        Cmd::Models => {
            for (name, repo, file) in models::MODELS {
                println!("  {name:14} {repo}/{file}");
            }
            println!(
                "  (the tested set — new candidates get evaluated, then a row in src/models.rs)"
            );
        }
        Cmd::Check { url } => {
            let out: serde_json::Value =
                ureq::get(&format!("{}/healthz", url.trim_end_matches('/')))
                    .call()?
                    .body_mut()
                    .read_json()?;
            println!("{}", serde_json::to_string(&out)?);
        }
        Cmd::Calibrate { m, files, output } => {
            init_logs(m.debug);
            if files.is_empty() {
                anyhow::bail!("calibrate needs at least one eval/*.jsonl file");
            }
            let path = models::resolve(&m.model)?;
            let mut eng = engine::Engine::new(path.to_string_lossy().as_ref(), m.ctx, 1024)?;
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
            output,
        } => {
            init_logs(m.debug);
            let cases = evaluate::load_cases(file)?;
            let rep = if let Some(url) = url {
                evaluate::evaluate_url(url, &cases, *limit, *no_abstain, !*no_perturb)?
            } else {
                let path = models::resolve(&m.model)?;
                let mut eng = engine::Engine::new(path.to_string_lossy().as_ref(), m.ctx, 1024)?;
                if let Some(c) = &m.calibration {
                    eng.load_calibration(c)?;
                }
                evaluate::evaluate(&mut eng, &cases, *limit, *no_abstain, !*no_perturb)?
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
                let mut eng = engine::Engine::new(path.to_string_lossy().as_ref(), m.ctx, 1024)?;
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
            let path = models::resolve(&m.model)?;
            let mut eng = engine::Engine::new(path.to_string_lossy().as_ref(), m.ctx, 1024)?;
            if let Some(c) = &m.calibration {
                eng.load_calibration(c)?;
            }
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?;
            rt.block_on(server::serve(eng, &m.model, m.ctx, host, *port))?;
        }
    }
    Ok(())
}
