//! Cache resolution + lazy download of a request's file set. The *decision* to
//! download (the CLI's interactive consent, or a server's `download_as_needed`
//! policy) stays with the caller: this module only reports what is missing and,
//! once told to proceed, fetches it. Per-file progress goes through a
//! [`DownloadReporter`] so the CLI keeps its decile logging and a server can
//! surface download progress over SSE.

use std::path::PathBuf;
use std::sync::Arc;

use thinfer_core::manifest::{FileRef, ModelManifest};
use thinfer_native::cache::{self, DownloadProgress};

/// Supplies a per-file [`DownloadProgress`] sink. Returning `None` for a file
/// downloads it without progress.
pub trait DownloadReporter {
    fn for_file(&self, file: &FileRef) -> Option<Arc<dyn DownloadProgress>>;
}

/// A reporter that attaches no progress sink.
pub struct NullReporter;

impl DownloadReporter for NullReporter {
    fn for_file(&self, _file: &FileRef) -> Option<Arc<dyn DownloadProgress>> {
        None
    }
}

/// Which of `files` are not yet in the HF cache (no network touched).
pub fn missing(files: &[FileRef]) -> Vec<FileRef> {
    cache::resolve_all(files.iter()).1
}

/// Download every file in `files` that is missing, reporting per-file progress
/// through `reporter`. Already-cached files are skipped.
pub async fn ensure(files: &[FileRef], reporter: &dyn DownloadReporter) -> Result<(), String> {
    for f in missing(files) {
        cache::download_with_progress(&f, reporter.for_file(&f))
            .await
            .map_err(|e| format!("{e:?}"))?;
    }
    Ok(())
}

/// Resolve a manifest role to its cached path (must already be present).
pub fn resolve_role(manifest: &ModelManifest, role: &str) -> Result<PathBuf, String> {
    let r = manifest
        .get(role)
        .ok_or_else(|| format!("manifest missing role {role}"))?;
    cache::resolve(r).ok_or_else(|| format!("{}/{} not in cache after download", r.repo, r.path))
}

/// Resolve a plain file reference to its cached path (must already be present).
/// For files not modeled as manifest roles (the face-swap ONNX models).
pub fn resolve_file(file: &FileRef) -> Result<PathBuf, String> {
    cache::resolve(file)
        .ok_or_else(|| format!("{}/{} not in cache after download", file.repo, file.path))
}
