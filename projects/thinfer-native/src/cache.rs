//! HF cache resolution and lazy download for `FileRef`s.
//!
//! Native-only. Web has its own cache layer (OPFS) consuming the same
//! `FileRef` manifest. Keep this module free of model-specific knowledge.

use std::path::PathBuf;
use std::sync::Arc;
use thinfer_core::manifest::FileRef;

use hf_hub::api::tokio::{ApiBuilder, ApiError, Progress};
use hf_hub::{Cache, Repo, RepoType};

/// Sink for download progress. Methods may be called from any tokio worker
/// (hf-hub fans chunks across tasks) - implementors must handle concurrent
/// `update` calls. `update`'s `delta` is the byte count for that chunk, not
/// the running total.
pub trait DownloadProgress: Send + Sync + 'static {
    fn init(&self, size: u64);
    fn update(&self, delta: u64);
    fn finish(&self);
}

#[derive(Debug)]
pub enum CacheError {
    Api(ApiError),
}

impl From<ApiError> for CacheError {
    fn from(e: ApiError) -> Self {
        Self::Api(e)
    }
}

impl std::fmt::Display for CacheError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Api(e) => write!(f, "hf-hub: {e}"),
        }
    }
}

impl std::error::Error for CacheError {}

fn repo_of(file: &FileRef) -> Repo {
    match file.revision {
        Some(rev) => Repo::with_revision(file.repo.into(), RepoType::Model, rev.into()),
        None => Repo::model(file.repo.into()),
    }
}

/// Look up `file` in the HF cache without touching the network. `None` if the
/// file isn't already cached. Used by integration tests so they can skip when
/// the file isn't present rather than triggering a multi-GB download.
pub fn resolve(file: &FileRef) -> Option<PathBuf> {
    Cache::from_env().repo(repo_of(file)).get(file.path)
}

/// Cache lookup, downloading from HF if missing. Single file only; never pulls
/// the whole repo. Concurrent calls for the same file are safe — `hf-hub`
/// downloads to a temp file then renames.
pub async fn download(file: &FileRef) -> Result<PathBuf, CacheError> {
    download_with_progress(file, None).await
}

/// Like `download` but reports progress to `sink` if provided. hf-hub will
/// `init(total)`, then `update(committed)` once at the start to seed any
/// already-downloaded bytes (resume case), then `update(delta)` per chunk.
pub async fn download_with_progress(
    file: &FileRef,
    sink: Option<Arc<dyn DownloadProgress>>,
) -> Result<PathBuf, CacheError> {
    let api = ApiBuilder::from_env().build()?;
    let repo = api.repo(repo_of(file));
    match sink {
        None => repo.download(file.path).await.map_err(Into::into),
        Some(sink) => repo
            .download_with_progress(file.path, ProgressAdapter(sink))
            .await
            .map_err(Into::into),
    }
}

#[derive(Clone)]
struct ProgressAdapter(Arc<dyn DownloadProgress>);

impl Progress for ProgressAdapter {
    async fn init(&mut self, size: usize, _filename: &str) {
        self.0.init(size as u64);
    }
    async fn update(&mut self, size: usize) {
        self.0.update(size as u64);
    }
    async fn finish(&mut self) {
        self.0.finish();
    }
}

/// Resolve every file in `files`. Returns `(resolved, missing)` partition so
/// the CLI can show one consent prompt covering all missing files.
pub fn resolve_all<'a>(
    files: impl IntoIterator<Item = &'a FileRef>,
) -> (Vec<(FileRef, PathBuf)>, Vec<FileRef>) {
    let mut hit = Vec::new();
    let mut miss = Vec::new();
    for f in files {
        match resolve(f) {
            Some(p) => hit.push((*f, p)),
            None => miss.push(*f),
        }
    }
    (hit, miss)
}

/// Default cache root (`$HF_HOME` or platform default). For error messages
/// that tell the user where things would land.
pub fn cache_root() -> PathBuf {
    Cache::from_env().path().clone()
}
