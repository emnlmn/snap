//! Instance registry: `snap serve` drops a pidfile at startup so `ps`
//! and `stop` can find servers from another terminal — healthz alone
//! can't answer "how many are up". Files self-clean: a dead pid means
//! a dead server, so stale records are removed on every read.

use std::path::PathBuf;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// pidfile payload — enough to render `ps` without touching the socket
#[derive(Serialize, Deserialize)]
pub struct Info {
    pub pid: u32,
    pub host: String,
    pub port: u16,
    pub model: String,
    /// unix seconds at `snap serve` start
    pub started: u64,
    pub version: String,
}

/// what a live pidfile resolves to right now
pub enum State {
    /// healthz answers — uptime/model from the server itself (hot-swap aware)
    Serving { uptime_s: u64, model: String },
    /// pid alive, port not answering yet — model still loading
    Starting,
    /// pid alive but port hasn't answered for >5 min — wedged
    Unresponsive,
}

pub struct Instance {
    pub info: Info,
    pub state: State,
}

/// $SNAP_RUN_DIR, else $XDG_RUNTIME_DIR/snap (per-user tmpfs — pidfiles
/// shouldn't survive a reboot anyway), else ~/.snap/run.
fn run_dir() -> PathBuf {
    if let Ok(d) = std::env::var("SNAP_RUN_DIR") {
        return d.into();
    }
    if let Ok(d) = std::env::var("XDG_RUNTIME_DIR") {
        return PathBuf::from(d).join("snap");
    }
    for k in ["HOME", "USERPROFILE"] {
        if let Ok(h) = std::env::var(k) {
            return PathBuf::from(h).join(".snap").join("run");
        }
    }
    std::env::temp_dir().join("snap")
}

fn file_for(pid: u32) -> PathBuf {
    run_dir().join(format!("{pid}.json"))
}

fn write(info: &Info) -> Result<()> {
    let dir = run_dir();
    std::fs::create_dir_all(&dir).with_context(|| format!("cannot create {}", dir.display()))?;
    std::fs::write(file_for(info.pid), serde_json::to_string(info)?)?;
    Ok(())
}

/// Record this process — called before model load so `ps` can show
/// "starting" during the slow part.
pub fn register(host: &str, port: u16, model: &str) -> Result<()> {
    write(&Info {
        pid: std::process::id(),
        host: host.to_string(),
        port,
        model: model.to_string(),
        started: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

/// The port we actually bound — rewrites the record for `--port 0`.
pub fn bound_port(port: u16) {
    let pid = std::process::id();
    if let Ok(mut info) = read(file_for(pid)) {
        if info.port != port {
            info.port = port;
            let _ = write(&info);
        }
    }
}

pub fn unregister(pid: u32) {
    let _ = std::fs::remove_file(file_for(pid));
}

fn read(path: PathBuf) -> Result<Info> {
    Ok(serde_json::from_str(&std::fs::read_to_string(path)?)?)
}

/// All registered servers with a live pid, sorted by port. Dead or
/// corrupt records are removed as a side effect. A stray answering
/// healthz on the default port (e.g. started before the registry
/// existed) is listed too — untracked, pid 0, `stop` can't kill it.
pub fn list() -> Vec<Instance> {
    let mut out = list_in(run_dir());
    if !out.iter().any(|i| i.info.port == crate::DEFAULT_PORT) {
        if let Some((model, uptime_s)) = probe("127.0.0.1", crate::DEFAULT_PORT) {
            out.push(Instance {
                info: Info {
                    pid: 0,
                    host: "127.0.0.1".into(),
                    port: crate::DEFAULT_PORT,
                    model: model.clone(),
                    started: 0,
                    version: "?".into(),
                },
                state: State::Serving { uptime_s, model },
            });
        }
    }
    out.sort_by_key(|i| i.info.port);
    out
}

fn list_in(dir: PathBuf) -> Vec<Instance> {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return vec![];
    };
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let mut out = Vec::new();
    for e in rd.flatten() {
        if e.path().extension().is_some_and(|x| x != "json") {
            continue;
        }
        let Ok(info) = read(e.path()) else {
            let _ = std::fs::remove_file(e.path());
            continue;
        };
        if !pid_alive(info.pid) {
            let _ = std::fs::remove_file(e.path());
            continue;
        }
        let state = match probe(&info.host, info.port) {
            Some((model, uptime_s)) => State::Serving { uptime_s, model },
            None if now.saturating_sub(info.started) < 300 => State::Starting,
            None => State::Unresponsive,
        };
        out.push(Instance { info, state });
    }
    out
}

/// Live server already holding `port`? Checked before the slow model
/// load so a double-start fails in milliseconds, not minutes.
pub fn on_port(port: u16) -> Option<Instance> {
    list().into_iter().find(|i| i.info.port == port)
}

/// GET /healthz with a tight deadline. Returns (model name, uptime).
fn probe(host: &str, port: u16) -> Option<(String, u64)> {
    // can't connect to a wildcard bind — loopback stands in for it
    let host = match host {
        "0.0.0.0" | "::" | "" => "127.0.0.1",
        h => h,
    };
    let agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_millis(300)))
        .build()
        .new_agent();
    let v: serde_json::Value = agent
        .get(&format!("http://{host}:{port}/healthz"))
        .call()
        .ok()?
        .body_mut()
        .read_json()
        .ok()?;
    if v.get("status")?.as_str()? != "ok" {
        return None;
    }
    Some((
        v.get("name")
            .or_else(|| v.get("model"))
            .and_then(|x| x.as_str())
            .unwrap_or("?")
            .to_string(),
        v.get("uptime_s").and_then(|x| x.as_u64()).unwrap_or(0),
    ))
}

#[cfg(unix)]
fn pid_alive(pid: u32) -> bool {
    // kill(pid, 0): 0 = ours, EPERM = someone else's — both mean alive
    (unsafe { libc::kill(pid as i32, 0) } == 0)
        || std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

#[cfg(windows)]
fn pid_alive(pid: u32) -> bool {
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::Threading::{
        GetExitCodeProcess, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION,
    };
    const STILL_ACTIVE: u32 = 259;
    unsafe {
        let h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
        if h == 0 {
            return false;
        }
        let mut code = 0;
        let alive = GetExitCodeProcess(h, &mut code) != 0 && code == STILL_ACTIVE;
        CloseHandle(h);
        alive
    }
}

/// Is `pid` really a `snap serve`? Guards `stop` against pid reuse:
/// a stale file pointing at an innocent process must not get it killed.
/// /proc on linux, `ps` elsewhere on unix, tasklist on windows.
#[cfg(unix)]
fn is_snap(pid: u32) -> bool {
    if let Ok(cmd) = std::fs::read_to_string(format!("/proc/{pid}/cmdline")) {
        return cmdline_is_serve(&cmd.replace('\0', " "));
    }
    let Ok(out) = std::process::Command::new("ps")
        .args(["-p", &pid.to_string(), "-o", "args="])
        .output()
    else {
        return false;
    };
    cmdline_is_serve(&String::from_utf8_lossy(&out.stdout))
}

#[cfg(windows)]
fn is_snap(pid: u32) -> bool {
    let Ok(out) = std::process::Command::new("tasklist")
        .args(["/FI", &format!("PID eq {pid}"), "/FO", "CSV", "/NH"])
        .output()
    else {
        return false;
    };
    String::from_utf8_lossy(&out.stdout)
        .to_lowercase()
        .starts_with("\"snap.exe")
}

fn cmdline_is_serve(cmd: &str) -> bool {
    let mut it = cmd.split_whitespace();
    let Some(bin) = it.next() else {
        return false;
    };
    let base = bin.rsplit(['/', '\\']).next().unwrap_or(bin);
    (base == "snap" || base == "snap.exe") && it.any(|a| a == "serve")
}

/// Healthz-confirmed servers are known snap; unresponsive ones need the
/// cmdline check. Anything else requires --force to kill.
pub fn confirmed(i: &Instance) -> bool {
    matches!(i.state, State::Serving { .. }) || is_snap(i.info.pid)
}

/// SIGTERM unless force → SIGKILL. Windows has no signals —
/// TerminateProcess is the only off switch.
#[cfg(unix)]
fn terminate(pid: u32, force: bool) -> Result<()> {
    let sig = if force { libc::SIGKILL } else { libc::SIGTERM };
    if unsafe { libc::kill(pid as i32, sig) } != 0 {
        return Err(std::io::Error::last_os_error()).context("kill");
    }
    Ok(())
}

#[cfg(windows)]
fn terminate(pid: u32, _force: bool) -> Result<()> {
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::Threading::{OpenProcess, TerminateProcess, PROCESS_TERMINATE};
    unsafe {
        let h = OpenProcess(PROCESS_TERMINATE, 0, pid);
        if h == 0 {
            return Err(std::io::Error::last_os_error()).context("OpenProcess");
        }
        let ok = TerminateProcess(h, 1);
        CloseHandle(h);
        if ok == 0 {
            return Err(std::io::Error::last_os_error()).context("TerminateProcess");
        }
    }
    Ok(())
}

/// Poll until the pid is gone or the deadline hits.
fn wait_dead(pid: u32, within: Duration) -> bool {
    let t = std::time::Instant::now();
    while t.elapsed() < within && pid_alive(pid) {
        std::thread::sleep(Duration::from_millis(100));
    }
    !pid_alive(pid)
}

/// Graceful stop: SIGTERM, drain up to 10s, then SIGKILL on unix.
/// `--force` skips the drain. Returns true once the pid is gone.
pub fn kill(pid: u32, force: bool) -> Result<bool> {
    terminate(pid, force)?;
    if wait_dead(pid, Duration::from_secs(10)) {
        return Ok(true);
    }
    #[cfg(unix)]
    if !force {
        eprintln!("pid {pid} ignored SIGTERM — sending SIGKILL");
        terminate(pid, true)?;
        return Ok(wait_dead(pid, Duration::from_secs(2)));
    }
    Ok(false)
}

pub fn fmt_uptime(s: u64) -> String {
    match s {
        s if s < 60 => format!("{s}s"),
        s if s < 3600 => format!("{}m", s / 60),
        s if s < 86400 => format!("{}h {}m", s / 3600, s % 3600 / 60),
        s => format!("{}d {}h", s / 86400, s % 86400 / 3600),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serve_cmdline() {
        assert!(cmdline_is_serve("/usr/local/bin/snap serve --port 8018"));
        assert!(cmdline_is_serve("snap serve"));
        assert!(cmdline_is_serve("C:\\bin\\snap.exe serve --model x"));
        assert!(!cmdline_is_serve("/usr/bin/snapshot serve"));
        assert!(!cmdline_is_serve("snap evaluate f.jsonl"));
        assert!(!cmdline_is_serve("vim"));
    }

    #[test]
    fn uptime_fmt() {
        assert_eq!(fmt_uptime(45), "45s");
        assert_eq!(fmt_uptime(720), "12m");
        assert_eq!(fmt_uptime(7380), "2h 3m");
        assert_eq!(fmt_uptime(90000), "1d 1h");
    }

    fn fixture(dir: &std::path::Path, pid: u32, started: u64) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(
            dir.join(format!("{pid}.json")),
            serde_json::to_string(&Info {
                pid,
                host: "127.0.0.1".into(),
                port: 1,
                model: "m".into(),
                started,
                version: "t".into(),
            })
            .unwrap(),
        )
        .unwrap();
    }

    fn now() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }

    #[test]
    fn list_own_pid() {
        // a pidfile pointing at this test process reads as Starting —
        // pid alive, no healthz
        let dir = std::env::temp_dir().join(format!("snap-test-a-{}", std::process::id()));
        fixture(&dir, std::process::id(), now());
        let l = list_in(dir.clone());
        assert_eq!(l.len(), 1);
        assert!(matches!(l[0].state, State::Starting));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn stale_file_cleaned() {
        // a reaped child pid is dead — its pidfile must be dropped by list()
        let mut c = std::process::Command::new("true").spawn().unwrap();
        let dead = c.id();
        c.wait().unwrap();
        let dir = std::env::temp_dir().join(format!("snap-test-b-{}", std::process::id()));
        fixture(&dir, dead, now());
        assert!(list_in(dir.clone()).is_empty());
        assert!(!dir.join(format!("{dead}.json")).exists());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
