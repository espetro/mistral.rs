//! Resolve `--checkpoint` to a local export directory, downloading Hub repos on demand.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use hf_hub::api::tokio::ApiBuilder;
use hf_hub::{Repo, RepoType};

const MANIFEST: &str = "kev.json";

/// A local directory is used as-is; anything else is `owner/repo[@revision]` on the Hub and is
/// fetched into the hf-hub cache (`HF_HOME`, token from `HF_TOKEN`), which mirrors the export layout.
pub async fn resolve(spec: &str) -> Result<PathBuf> {
    let local = Path::new(spec);
    if local.is_dir() {
        return Ok(local.to_path_buf());
    }
    let (repo_id, revision) = match spec.split_once('@') {
        Some((id, rev)) => (id, rev.to_string()),
        None => (spec, "main".to_string()),
    };
    if repo_id.matches('/').count() != 1 || repo_id.starts_with('/') || repo_id.starts_with('.') {
        bail!("{spec}: not a directory and not a Hub id (owner/repo[@revision])");
    }
    let api = ApiBuilder::from_env().with_progress(true).build()?;
    let repo = api.repo(Repo::with_revision(
        repo_id.to_string(),
        RepoType::Model,
        revision,
    ));
    let info = repo
        .info()
        .await
        .with_context(|| format!("fetching file list for {spec}"))?;
    if !info.siblings.iter().any(|s| s.rfilename == MANIFEST) {
        bail!("{spec} has no {MANIFEST}; expected a directory written by scripts/export_checkpoint.py");
    }
    let mut root = None;
    for sibling in &info.siblings {
        let path = repo
            .get(&sibling.rfilename)
            .await
            .with_context(|| format!("downloading {}", sibling.rfilename))?;
        if sibling.rfilename == MANIFEST {
            root = path.parent().map(Path::to_path_buf);
        }
    }
    root.context("kev.json downloaded without a parent directory")
}
