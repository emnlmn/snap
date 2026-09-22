//! Model resolution: the tested set only. A GGUF that *loads* is not a
//! GGUF that answers correctly — mechanical checks (chat template, single-
//! token letters) catch some failures at startup, but instruction-following
//! quality is only proven by running the eval suite. New candidates are
//! added to MODELS, evaluated, and stay if they earn it.

use std::path::PathBuf;

use anyhow::{bail, Context, Result};

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
];

pub const DEFAULT_MODEL: &str = "minicpm5-2b";

/// Resolve a tested-model name to a local GGUF path, downloading if needed.
pub fn resolve(spec: &str) -> Result<PathBuf> {
    let Some((_, repo, file)) = MODELS.iter().find(|(n, _, _)| *n == spec) else {
        bail!(
            "unknown model {spec:?} — supported: {}",
            MODELS
                .iter()
                .map(|(n, _, _)| *n)
                .collect::<Vec<_>>()
                .join(", ")
        );
    };
    if hf_hub::Cache::from_env()
        .model(repo.to_string())
        .get(file)
        .is_none()
    {
        eprintln!("downloading {repo}/{file} from huggingface …");
    }
    let api = hf_hub::api::sync::Api::new().context("hf api init")?;
    api.model(repo.to_string())
        .get(file)
        .context("hf download failed")
}
