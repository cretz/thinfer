//! Declarative model file manifest. Pure data; no I/O, no platform deps.
//!
//! Resolvers in `thinfer-native::cache` (HF cache + download) and (later)
//! `thinfer-web::cache` (HF Hub HTTP -> OPFS) consume `FileRef` to produce a
//! tier-appropriate handle (path, ArrayBuffer, OPFS handle).

/// One file in a model: HF repo + path inside the repo. Optional `revision`
/// pins to a commit/branch/tag for reproducibility (`None` = repo default,
/// typically `main`). No on-disk path; resolution is the caller's job.
///
/// HF-only for now. If we ever need raw HTTP / mirror / local-only sources,
/// promote this to an enum (`Hf { repo, path, revision } | Url(String) | ...`)
/// and update resolvers in `thinfer-native::cache` + `thinfer-web::cache`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FileRef {
    pub repo: &'static str,
    pub path: &'static str,
    pub revision: Option<&'static str>,
}

impl FileRef {
    pub const fn new(repo: &'static str, path: &'static str) -> Self {
        Self {
            repo,
            path,
            revision: None,
        }
    }
    pub const fn pinned(repo: &'static str, path: &'static str, rev: &'static str) -> Self {
        Self {
            repo,
            path,
            revision: Some(rev),
        }
    }
}

/// Per-model manifest: stable role -> file mapping. `Role` is a per-model
/// `&'static str` (e.g. "dit", "text_encoder/model", "vae/decoder"). Open-set
/// strings keep the type model-agnostic; per-model crates expose typed
/// accessors over the same data so callers don't typo role keys.
#[derive(Clone, Copy, Debug)]
pub struct ModelManifest {
    pub id: &'static str,
    pub files: &'static [(&'static str, FileRef)],
}

impl ModelManifest {
    pub fn get(&self, role: &str) -> Option<&'static FileRef> {
        self.files.iter().find(|(r, _)| *r == role).map(|(_, f)| f)
    }
    pub fn iter(&self) -> impl Iterator<Item = &'static (&'static str, FileRef)> + '_ {
        self.files.iter()
    }
}
