//! Model resolution: the tested set only. A GGUF that *loads* is not a
//! GGUF that answers correctly — mechanical checks (chat template, single-
//! token letters) catch some failures at startup, but instruction-following
//! quality is only proven by running the eval cases. New candidates are
//! added to MODELS, evaluated, and stay if they earn it.

use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use hf_hub::{Cache, Repo};

pub const MODELS: &[(&str, &str, &str)] = &[
    (
        "minicpm5-2b",
        "openbmb/MiniCPM5-2B-GGUF",
        "MiniCPM5-2B-Q4_K_M.gguf",
    ),
    (
        "qwen3.8-4b",
        "empero-ai/Qwen3.8-4B-Distill-GGUF",
        "Qwen3.8-4B-Q4_K_M.gguf",
    ),
    (
        "spark-4b",
        "XHToken/Spark-X2.5-4B-GGUF",
        "Spark-X2.5-4B-Q8_0.gguf",
    ),
    (
        "spark-4b-q4",
        "XHToken/Spark-X2.5-4B-GGUF",
        "Spark-X2.5-4B-Q4_K_M.gguf",
    ),
    ("snap1-2b", "logitlab/snap1-2b-GGUF", "snap1-2b-q4_k_m.gguf"),
    (
        "snap1-2b-q8",
        "logitlab/snap1-2b-GGUF",
        "snap1-2b-q8_0.gguf",
    ),
    (
        "snap1-2b-bf16",
        "logitlab/snap1-2b-GGUF",
        "snap1-2b-bf16.gguf",
    ),
    (
        "winnow-e4b",
        "EldanRing/Winnow-E4B",
        "gguf/Winnow-E4B-Q8_0.gguf",
    ),
];

pub const DEFAULT_MODEL: &str = "snap1-2b";

/// The MODELS row for `name`, or the known-names error every caller shares.
fn entry(name: &str) -> Result<&'static (&'static str, &'static str, &'static str)> {
    MODELS.iter().find(|(n, _, _)| *n == name).ok_or_else(|| {
        anyhow::anyhow!(
            "unknown model {name:?} — supported: {}",
            MODELS
                .iter()
                .map(|(n, _, _)| *n)
                .collect::<Vec<_>>()
                .join(", ")
        )
    })
}

/// Where `repo` lives under the HF cache root ($HF_HOME/hub, else
/// ~/.cache/huggingface/hub).
fn repo_dir(repo: &str) -> PathBuf {
    Cache::from_env()
        .path()
        .join(Repo::model(repo.to_string()).folder_name())
}

/// The cached GGUF for a MODELS row, if pulled already.
pub(crate) fn cached(repo: &str, file: &str) -> Option<PathBuf> {
    Cache::from_env().model(repo.to_string()).get(file)
}

/// Ensure `name`'s GGUF is on disk; returns its cache path. Cache-first,
/// so an already-pulled model is a free, offline call.
pub fn pull(name: &str) -> Result<PathBuf> {
    let &(_, repo, file) = entry(name)?;
    if cached(repo, file).is_none() {
        eprintln!("snap: pulling {repo}/{file} from huggingface …");
    }
    hf_hub::api::sync::Api::new()
        .context("hf api init")?
        .model(repo.to_string())
        .get(file)
        .context("hf download failed")
}

/// Resolve a tested-model name to a local GGUF path, pulling if needed.
/// An explicit path to an existing .gguf is honored as-is — the MODELS
/// curation gates *names*, not files a caller already has on disk.
pub fn resolve(spec: &str) -> Result<PathBuf> {
    if spec.ends_with(".gguf") {
        let p = PathBuf::from(spec);
        if !p.is_file() {
            bail!("model file {spec:?} not found");
        }
        return Ok(p);
    }
    pull(spec)
}

/// Delete `name`'s GGUF from the cache: every snapshot pointer carrying
/// the file, plus each blob no remaining pointer still references (the
/// cache dir is shared — another tool may hold files `snap` never
/// pulled). Returns bytes freed.
pub fn remove(name: &str) -> Result<u64> {
    let &(_, repo, file) = entry(name)?;
    let snaps = repo_dir(repo).join("snapshots");
    if !snaps.is_dir() {
        bail!("{name:?} is not pulled");
    }
    let mut blobs = Vec::new();
    let mut freed = 0u64;
    let mut removed = false;
    for e in std::fs::read_dir(&snaps)? {
        let p = e?.path().join(file);
        let Ok(md) = std::fs::symlink_metadata(&p) else {
            continue;
        };
        removed = true;
        if md.file_type().is_symlink() {
            if let Ok(b) = p.canonicalize() {
                blobs.push(b);
            }
        } else {
            // no symlink privilege (windows): the "pointer" is the file itself
            freed += md.len();
        }
        std::fs::remove_file(&p)?;
    }
    if !removed {
        bail!("{name:?} is not pulled");
    }
    blobs.sort();
    blobs.dedup();
    'keep: for blob in blobs {
        for sha in std::fs::read_dir(&snaps)?.flatten() {
            for p in std::fs::read_dir(sha.path())?.flatten() {
                if p.path().canonicalize().ok().as_deref() == Some(blob.as_path()) {
                    continue 'keep;
                }
            }
        }
        freed += std::fs::metadata(&blob).map(|m| m.len()).unwrap_or(0);
        let mut lock = blob.clone().into_os_string();
        lock.push(".lock");
        let _ = std::fs::remove_file(&blob);
        let _ = std::fs::remove_file(lock);
    }
    Ok(freed)
}
