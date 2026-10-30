//! User weight sources: fills the manifest's named weight slots
//! ([`FileRef::slot`]) from a user-owned TOML file kept outside the project.
//! Which checkpoint (finetune, merge, encoder build) backs a slot is the user's
//! choice; the project only declares the slot and an upstream default.
//!
//! Lookup: `THINFER_SOURCES` (path) if set, else `<hf-cache>/thinfer-sources.toml`
//! if present, else no overrides. Read once per process.
//!
//! ```toml
//! [slots."qwen-image/dit-q8_0"]
//! repo = "owner/repo"
//! path = "dir/file.gguf"
//! # revision = "main"
//! ```

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::OnceLock;

use serde::Deserialize;
use thinfer_core::manifest::FileRef;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SourcesFile {
    #[serde(default)]
    slots: HashMap<String, SlotEntry>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SlotEntry {
    repo: String,
    path: String,
    revision: Option<String>,
}

/// Env var naming the sources file (overrides the default location).
pub const SOURCES_ENV: &str = "THINFER_SOURCES";

/// The sources file in effect, if any.
pub fn sources_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var(SOURCES_ENV)
        && !p.is_empty()
    {
        return Some(PathBuf::from(p));
    }
    let default = crate::cache::cache_root().join("thinfer-sources.toml");
    default.is_file().then_some(default)
}

/// The parsed sources, or why the file failed to load. A broken file must not
/// silently fall back to defaults (that would fetch weights the user did not
/// choose), so on error every slotted ref resolves as unfilled and
/// [`require`] surfaces the load error.
fn slots() -> &'static Result<HashMap<String, FileRef>, String> {
    static SLOTS: OnceLock<Result<HashMap<String, FileRef>, String>> = OnceLock::new();
    SLOTS.get_or_init(|| match sources_path() {
        None => Ok(HashMap::new()),
        Some(path) => load(&path).map_err(|e| format!("weight sources {}: {e}", path.display())),
    })
}

fn load(path: &std::path::Path) -> Result<HashMap<String, FileRef>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let file: SourcesFile = toml::from_str(&text).map_err(|e| e.to_string())?;
    // Leaked once per process: `FileRef` is `&'static` data end to end.
    let leak = |s: String| -> &'static str { Box::leak(s.into_boxed_str()) };
    Ok(file
        .slots
        .into_iter()
        .map(|(slot, e)| {
            let slot = leak(slot);
            let file = FileRef {
                repo: leak(e.repo),
                path: leak(e.path),
                revision: e.revision.map(leak),
                slot: Some(slot),
            };
            (slot.to_string(), file)
        })
        .collect())
}

/// The ref to actually fetch for `file`: the user's entry for its slot when one
/// is set, else `file` itself (an unslotted ref, or a slot left at its default).
/// An unfilled default-less slot comes back unchanged; check
/// [`FileRef::is_unfilled`] (or use [`require`]).
pub fn effective(file: &FileRef) -> FileRef {
    let Some(slot) = file.slot else {
        return *file;
    };
    match slots() {
        Ok(map) => map.get(slot).copied().unwrap_or(*file),
        Err(_) => FileRef::user(slot),
    }
}

/// [`effective`], erroring with a user-facing fix when the slot has no default
/// and the user's sources do not fill it.
pub fn require(file: &FileRef) -> Result<FileRef, String> {
    let eff = effective(file);
    if eff.is_unfilled() {
        if let Err(e) = slots() {
            return Err(e.clone());
        }
        let slot = file.slot.unwrap_or("?");
        let at = sources_path().map_or_else(
            || {
                crate::cache::cache_root()
                    .join("thinfer-sources.toml")
                    .display()
                    .to_string()
            },
            |p| p.display().to_string(),
        );
        return Err(format!(
            "weight slot `{slot}` has no default checkpoint; set [slots.\"{slot}\"] \
             repo/path in {at} (or point {SOURCES_ENV} at a sources file)"
        ));
    }
    Ok(eff)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_fills_slots_and_keeps_the_slot_name() {
        let dir = std::env::temp_dir().join(format!("thinfer-sources-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("sources.toml");
        std::fs::write(
            &path,
            "[slots.\"m/dit\"]\nrepo = \"o/r\"\npath = \"d/f.gguf\"\nrevision = \"v1\"\n",
        )
        .unwrap();
        let map = load(&path).unwrap();
        let f = map["m/dit"];
        assert_eq!(
            (f.repo, f.path, f.revision, f.slot),
            ("o/r", "d/f.gguf", Some("v1"), Some("m/dit"))
        );
        assert!(!f.is_unfilled());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn unknown_keys_are_rejected() {
        let dir = std::env::temp_dir().join(format!("thinfer-sources-bad-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("sources.toml");
        std::fs::write(
            &path,
            "[slots.\"m/dit\"]\nrepo = \"o/r\"\npaht = \"typo\"\n",
        )
        .unwrap();
        assert!(load(&path).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn user_slot_is_unfilled_until_sourced() {
        let f = FileRef::user("m/dit");
        assert!(f.is_unfilled());
        assert_eq!(f.slot, Some("m/dit"));
        let d = FileRef::new("o/r", "f").in_slot("m/enc");
        assert!(!d.is_unfilled());
        assert_eq!(d.slot, Some("m/enc"));
    }
}
