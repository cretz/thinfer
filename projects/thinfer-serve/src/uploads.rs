//! In-memory registry of streamed video uploads, with a TTL reaper. A browser
//! streams an mp4 to `POST /uploads` (raw body, no base64); this holds it
//! RAM-first, or seals a large clip to an encrypted on-disk spill under a
//! per-upload ephemeral key (RAM only) -- the same at-rest scheme the inline job
//! path uses, so raw plaintext never touches disk. A job later consumes the
//! upload by id (moving any spill into the job dir, where delete-on-fetch
//! governs it); anything never consumed is reaped after `ttl` so no ciphertext
//! lingers indefinitely.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use rand::RngCore;
use thinfer_app::request::VideoInput;

/// Above this size an upload spills to an encrypted on-disk blob rather than
/// staying in RAM (mirrors the inline job-spill path). 512 MiB: typical clips
/// are a few hundred MB, well under the RAM budget.
pub const VIDEO_SPILL_THRESHOLD: usize = 512 << 20;

/// Build a [`VideoInput`] from raw mp4 bytes: kept in RAM at or below
/// `threshold`, else sealed to `<dir>/input_video.enc` under a fresh ephemeral
/// AES-256-GCM key (held in the returned value, RAM only). The raw plaintext
/// never lands on disk; `dir` is created only on the spill path.
pub fn seal_video(bytes: Vec<u8>, dir: &Path, threshold: usize) -> Result<VideoInput, String> {
    if bytes.len() > threshold {
        std::fs::create_dir_all(dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
        let (key, nonce, ct) = thinfer_app::vault::ephemeral_seal(&bytes);
        drop(bytes); // wipe the plaintext copy; only ciphertext + the RAM key remain
        let path = dir.join("input_video.enc");
        std::fs::write(&path, &ct).map_err(|e| format!("write {}: {e}", path.display()))?;
        Ok(VideoInput::Encrypted { path, key, nonce })
    } else {
        Ok(VideoInput::Ram(bytes))
    }
}

struct Entry {
    input: VideoInput,
    /// The upload's own dir (holds the spill file, if any). Removed on reap.
    dir: PathBuf,
    created: Instant,
}

/// Registry of pending uploads keyed by opaque id, reaped by TTL. Cheap to
/// clone-share behind an `Arc`.
pub struct UploadStore {
    base: PathBuf,
    spill_threshold: usize,
    entries: Mutex<HashMap<String, Entry>>,
}

impl UploadStore {
    /// A store rooted at `base` (e.g. `<artifact_dir>/uploads`).
    pub fn new(base: PathBuf) -> Self {
        Self {
            base,
            spill_threshold: VIDEO_SPILL_THRESHOLD,
            entries: Mutex::new(HashMap::new()),
        }
    }

    /// Store raw upload bytes and return `(id, size)`. Large clips spill to an
    /// encrypted file under `<base>/<id>/`; small ones stay in RAM.
    pub fn put(&self, bytes: Vec<u8>) -> Result<(String, usize), String> {
        let size = bytes.len();
        let id = random_id();
        let dir = self.base.join(&id);
        let input = seal_video(bytes, &dir, self.spill_threshold)?;
        self.entries.lock().unwrap().insert(
            id.clone(),
            Entry {
                input,
                dir,
                created: Instant::now(),
            },
        );
        Ok((id, size))
    }

    /// Consume upload `id` into a job whose artifact dir is `job_dir`: remove it
    /// from the registry and, for a spilled upload, move the ciphertext into the
    /// job dir so delete-on-fetch (not the reaper) governs its lifetime from
    /// here on. Errors if the id is unknown or already expired/consumed.
    pub fn consume(&self, id: &str, job_dir: &Path) -> Result<VideoInput, String> {
        let entry = self
            .entries
            .lock()
            .unwrap()
            .remove(id)
            .ok_or("upload not found (expired, already used, or never uploaded)")?;
        let result = match entry.input {
            VideoInput::Ram(b) => VideoInput::Ram(b),
            VideoInput::Encrypted { path, key, nonce } => {
                std::fs::create_dir_all(job_dir)
                    .map_err(|e| format!("create {}: {e}", job_dir.display()))?;
                let dest = job_dir.join("input_video.enc");
                move_file(&path, &dest)?;
                VideoInput::Encrypted {
                    path: dest,
                    key,
                    nonce,
                }
            }
        };
        // The upload's own dir is now empty (spill moved out, or never existed).
        let _ = std::fs::remove_dir_all(&entry.dir);
        Ok(result)
    }

    /// Drop every entry older than `ttl`, deleting its spill dir. Returns the
    /// count reaped. Cheap when nothing is expired (a single map scan).
    pub fn reap(&self, ttl: Duration) -> usize {
        let mut reaped = Vec::new();
        {
            let mut entries = self.entries.lock().unwrap();
            entries.retain(|_, e| {
                let keep = e.created.elapsed() < ttl;
                if !keep {
                    reaped.push(e.dir.clone());
                }
                keep
            });
        }
        for dir in &reaped {
            let _ = std::fs::remove_dir_all(dir);
        }
        reaped.len()
    }
}

/// Move a file, falling back to copy+remove if `rename` fails (e.g. across
/// filesystems). Both the upload spill and the job dir live under the artifact
/// dir, so the rename normally succeeds.
fn move_file(from: &Path, to: &Path) -> Result<(), String> {
    if std::fs::rename(from, to).is_ok() {
        return Ok(());
    }
    std::fs::copy(from, to)
        .map_err(|e| format!("copy {} -> {}: {e}", from.display(), to.display()))?;
    let _ = std::fs::remove_file(from);
    Ok(())
}

/// 32 hex chars of randomness: the public, opaque upload id (and dir name).
fn random_id() -> String {
    let mut b = [0u8; 16];
    rand::thread_rng().fill_bytes(&mut b);
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// Spawn a background task reaping expired uploads on an interval bounded by
/// `ttl` (at most every 60s, at least every 1s).
pub fn spawn_reaper(store: Arc<UploadStore>, ttl: Duration) {
    let period = ttl.min(Duration::from_secs(60)).max(Duration::from_secs(1));
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(period);
        loop {
            tick.tick().await;
            let n = store.reap(ttl);
            if n > 0 {
                tracing::info!(reaped = n, "upload reaper evicted expired uploads");
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store_with(base: PathBuf, threshold: usize) -> UploadStore {
        UploadStore {
            base,
            spill_threshold: threshold,
            entries: Mutex::new(HashMap::new()),
        }
    }

    #[test]
    fn small_upload_stays_in_ram_and_consumes_bytes_intact() {
        let tmp = std::env::temp_dir().join(format!("thinfer_up_ram_{}", random_id()));
        let store = store_with(tmp.clone(), 1 << 20); // 1 MiB threshold
        let (id, size) = store.put(b"tiny-mp4-bytes".to_vec()).unwrap();
        assert_eq!(size, 14);
        // No spill dir for a RAM upload.
        assert!(!tmp.join(&id).exists());
        let job_dir = tmp.join("job");
        match store.consume(&id, &job_dir).unwrap() {
            VideoInput::Ram(b) => assert_eq!(b, b"tiny-mp4-bytes"),
            _ => panic!("expected RAM"),
        }
        // Second consume of the same id fails (removed from the registry).
        assert!(store.consume(&id, &job_dir).is_err());
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn large_upload_spills_encrypted_then_consume_moves_it_and_decrypts() {
        let tmp = std::env::temp_dir().join(format!("thinfer_up_spill_{}", random_id()));
        let store = store_with(tmp.clone(), 8); // force spill for anything > 8 bytes
        let plaintext = b"this is definitely more than eight bytes of mp4".to_vec();
        let (id, _) = store.put(plaintext.clone()).unwrap();
        // Ciphertext exists under the upload dir; it is NOT the plaintext.
        let spill = tmp.join(&id).join("input_video.enc");
        assert!(spill.exists(), "spill file should exist");
        let ct = std::fs::read(&spill).unwrap();
        assert_ne!(ct, plaintext, "on-disk bytes must be ciphertext");

        let job_dir = tmp.join("job-7");
        let input = store.consume(&id, &job_dir).unwrap();
        // The upload dir is gone; the ciphertext now lives in the job dir.
        assert!(!tmp.join(&id).exists(), "upload dir should be removed");
        let (path, key, nonce) = match &input {
            VideoInput::Encrypted { path, key, nonce } => (path.clone(), *key, nonce.clone()),
            _ => panic!("expected Encrypted"),
        };
        assert_eq!(path, job_dir.join("input_video.enc"));
        // The moved ciphertext decrypts back to the original plaintext.
        let moved_ct = std::fs::read(&path).unwrap();
        let decrypted = thinfer_app::vault::ephemeral_unseal(&key, &nonce, &moved_ct).unwrap();
        assert_eq!(decrypted, plaintext);
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn reaper_evicts_expired_entries_and_their_spill_dirs() {
        let tmp = std::env::temp_dir().join(format!("thinfer_up_reap_{}", random_id()));
        let store = store_with(tmp.clone(), 8); // force spill so a dir exists to delete
        let (id, _) = store.put(b"more than eight bytes here".to_vec()).unwrap();
        let spill_dir = tmp.join(&id);
        assert!(spill_dir.exists());

        // A generous TTL keeps it.
        assert_eq!(store.reap(Duration::from_secs(3600)), 0);
        assert!(spill_dir.exists());

        // A zero TTL evicts it and removes the spill dir.
        assert_eq!(store.reap(Duration::ZERO), 1);
        assert!(!spill_dir.exists(), "reaped spill dir should be gone");
        // The id is no longer consumable.
        assert!(store.consume(&id, &tmp.join("job")).is_err());
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
