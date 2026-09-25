//! Update nag: once a day ask GitHub for the latest release and mention
//! it on stderr. Strictly best-effort — tight timeout, silent on any
//! failure, never on stdout (the `-p` contract), opt out with
//! SNAP_NO_UPDATE_CHECK. The cache lives in the run dir under a
//! non-.json name so instances::list_in won't eat it.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

const RELEASES_API: &str = "https://api.github.com/repos/emnlmn/snap/releases/latest";
const RELEASES_PAGE: &str = "https://github.com/emnlmn/snap/releases/latest";
const TTL_S: u64 = 24 * 3600;
const TIMEOUT: Duration = Duration::from_millis(1500);
const CACHE: &str = "version.check";

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn fetch() -> Option<String> {
    let agent = ureq::Agent::config_builder()
        .timeout_global(Some(TIMEOUT))
        .build()
        .new_agent();
    let v: serde_json::Value = agent
        .get(RELEASES_API)
        .header("user-agent", concat!("snap/", env!("CARGO_PKG_VERSION")))
        .header("accept", "application/vnd.github+json")
        .call()
        .ok()?
        .body_mut()
        .read_json()
        .ok()?;
    Some(v.get("tag_name")?.as_str()?.to_string())
}

/// (unix seconds fetched, latest tag then) — tag is None on a cached
/// failure, which rate-limits retries exactly like successes.
fn read_cache(p: &std::path::Path) -> Option<(u64, Option<String>)> {
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(p).ok()?).ok()?;
    Some((
        v.get("fetched")?.as_u64()?,
        v.get("tag").and_then(|t| t.as_str()).map(String::from),
    ))
}

fn cache_path() -> std::path::PathBuf {
    crate::instances::run_dir().join(CACHE)
}

fn write_cache(p: &std::path::Path, tag: Option<&str>) {
    if let Some(d) = p.parent() {
        let _ = std::fs::create_dir_all(d);
    }
    let _ = std::fs::write(
        p,
        serde_json::json!({ "fetched": now(), "tag": tag }).to_string(),
    );
}

/// "v0.2.0-rc1" -> (0, 2, 0); unparseable parts count as 0
fn tuple(tag: &str) -> (u32, u32, u32) {
    let v = tag.trim().trim_start_matches(['v', 'V']);
    let v = v.split(['-', '+']).next().unwrap_or(v);
    let mut n = [0u32; 3];
    for (i, p) in v.split('.').take(3).enumerate() {
        n[i] = p.parse().unwrap_or(0);
    }
    (n[0], n[1], n[2])
}

fn is_newer(tag: &str, current: &str) -> bool {
    tuple(tag) > tuple(current)
}

/// The install channel decides the upgrade words.
fn upgrade_hint() -> String {
    let exe = std::env::current_exe()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_default();
    if exe.contains("/Cellar/") || exe.contains("/linuxbrew/") {
        "brew upgrade emnlmn/snap/snap".to_string()
    } else {
        RELEASES_PAGE.to_string()
    }
}

/// Cache-gated nag: a stale cache fetches inline (bounded by TIMEOUT,
/// ~once a day), a fresh one is free. Failures cache too, so offline
/// never stalls more than once per TTL.
pub fn nag() {
    if std::env::var_os("SNAP_NO_UPDATE_CHECK").is_some() {
        return;
    }
    let cache = cache_path();
    let tag = match read_cache(&cache) {
        Some((fetched, tag)) if now().saturating_sub(fetched) < TTL_S => tag,
        _ => {
            let tag = fetch();
            write_cache(&cache, tag.as_deref());
            tag
        }
    };
    let Some(tag) = tag else { return };
    if is_newer(&tag, env!("CARGO_PKG_VERSION")) {
        eprintln!(
            "snap {tag} is out (you have {}) — {}",
            env!("CARGO_PKG_VERSION"),
            upgrade_hint()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tuple_parses_tag_forms() {
        assert_eq!(tuple("v0.1.0"), (0, 1, 0));
        assert_eq!(tuple("0.10.2"), (0, 10, 2));
        assert_eq!(tuple("v1"), (1, 0, 0));
        assert_eq!(tuple("1.2.3-rc.1"), (1, 2, 3));
        assert_eq!(tuple("junk"), (0, 0, 0));
    }

    #[test]
    fn newer_only_on_real_upgrade() {
        assert!(is_newer("v0.2.0", "0.1.9"));
        assert!(!is_newer("v0.1.0", "0.1.0"));
        assert!(!is_newer("v0.1.0", "0.2.0")); // dev build ahead of any release
        assert!(!is_newer("garbage", "0.1.0"));
    }

    #[test]
    fn cache_roundtrips() {
        let p = std::env::temp_dir().join(format!("snap-test-{}", std::process::id()));
        write_cache(&p, Some("v9.9.9"));
        let (_, tag) = read_cache(&p).unwrap();
        assert_eq!(tag.as_deref(), Some("v9.9.9"));
        write_cache(&p, None); // cached failure
        assert_eq!(read_cache(&p).unwrap().1, None);
        assert_eq!(read_cache(&p.with_file_name("missing")), None);
        let _ = std::fs::remove_file(&p);
    }
}
