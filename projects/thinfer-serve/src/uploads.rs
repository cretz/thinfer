//! Registry of streamed video uploads, with a TTL reaper. The client seals the
//! clip under its per-job key (`thinfer_app::jobkey` layout) BEFORE uploading,
//! so the server only ever stores ciphertext and holds no key for it: the bytes
//! are written to disk as-is. When the job runs (and its key has arrived), the
//! upload is consumed into the job dir as a decrypt-on-read input, so the frames
//! stay ciphertext until the decoder reads them. Anything never consumed is
//! reaped after `ttl`.
//!
//! [`seal_video`] remains for the inline path: a base64 clip inside an already
//! decrypted spec is held in RAM, or spilled under a fresh ephemeral key.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use rand::RngCore;
use thinfer_app::jobkey::{self, JobKey};
use thinfer_app::request::VideoInput;

/// Above this size an inline (base64-in-spec) clip spills to an encrypted
/// on-disk blob rather than staying in RAM.
pub const VIDEO_SPILL_THRESHOLD: usize = 512 << 20;

/// Build a [`VideoInput`] from raw mp4 bytes of a DECRYPTED spec: kept in RAM at
/// or below `threshold`, else sealed to `<dir>/input_video.enc` under a fresh
/// ephemeral AES-256-GCM key (held in the returned value, RAM only). The raw
/// plaintext never lands on disk; `dir` is created only on the spill path.
pub fn seal_video(bytes: Vec<u8>, dir: &Path, threshold: usize) -> Result<VideoInput, String> {
    if bytes.len() > threshold {
        std::fs::create_dir_all(dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
        let (key, nonce, ct) = thinfer_app::vault::ephemeral_seal(&bytes);
        drop(bytes); // only ciphertext + the RAM key remain
        let path = dir.join("input_video.enc");
        std::fs::write(&path, &ct).map_err(|e| format!("write {}: {e}", path.display()))?;
        Ok(VideoInput::Encrypted { path, key, nonce })
    } else {
        Ok(VideoInput::Ram(bytes))
    }
}

struct Entry {
    /// The ciphertext file (`<base>/<id>/input_video.enc`, nonce stripped).
    path: PathBuf,
    /// The client's GCM nonce (base64), split off the sealed blob.
    nonce: String,
    /// The upload's own dir. Removed on consume or reap.
    dir: PathBuf,
    created: Instant,
}

/// Registry of pending sealed uploads keyed by opaque id, reaped by TTL.
pub struct UploadStore {
    base: PathBuf,
    entries: Mutex<HashMap<String, Entry>>,
}

impl UploadStore {
    /// A store rooted at `base` (e.g. `<artifact_dir>/uploads`).
    pub fn new(base: PathBuf) -> Self {
        Self {
            base,
            entries: Mutex::new(HashMap::new()),
        }
    }

    /// Store a client-sealed upload (`[nonce][ciphertext+tag]`) and return
    /// `(id, size)`. Written to disk as ciphertext; the server has no key.
    pub fn put(&self, sealed: &[u8]) -> Result<(String, usize), String> {
        let (nonce, ct) = jobkey::split_sealed(sealed)?;
        let id = random_id();
        let dir = self.base.join(&id);
        std::fs::create_dir_all(&dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
        let path = dir.join("input_video.enc");
        std::fs::write(&path, ct).map_err(|e| format!("write {}: {e}", path.display()))?;
        self.entries.lock().unwrap().insert(
            id.clone(),
            Entry {
                path,
                nonce,
                dir,
                created: Instant::now(),
            },
        );
        Ok((id, sealed.len()))
    }

    /// Consume upload `id` into a job whose dir is `job_dir`, under the job's
    /// `key`: the ciphertext moves into the job dir (delete-on-fetch / the job
    /// reaper govern it from here) and becomes a decrypt-on-read input. Errors
    /// if the id is unknown or already expired/consumed.
    pub fn consume(&self, id: &str, job_dir: &Path, key: &JobKey) -> Result<VideoInput, String> {
        let entry = self
            .entries
            .lock()
            .unwrap()
            .remove(id)
            .ok_or("upload not found (expired, already used, or never uploaded)")?;
        std::fs::create_dir_all(job_dir)
            .map_err(|e| format!("create {}: {e}", job_dir.display()))?;
        let dest = job_dir.join("input_video.enc");
        move_file(&entry.path, &dest)?;
        let _ = std::fs::remove_dir_all(&entry.dir);
        Ok(VideoInput::Encrypted {
            path: dest,
            key: *key.bytes(),
            nonce: entry.nonce,
        })
    }

    /// Drop every entry older than `ttl`, deleting its dir. Returns the count
    /// reaped. Cheap when nothing is expired (a single map scan).
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
/// filesystems). Both the upload and the job dir live under the artifact dir,
/// so the rename normally succeeds.
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

    #[test]
    fn upload_is_stored_as_ciphertext_and_opens_with_the_job_key() {
        let tmp = std::env::temp_dir().join(format!("thinfer_up_{}", random_id()));
        let store = UploadStore::new(tmp.clone());
        let key = JobKey::generate();
        let plaintext = b"mp4 bytes that only the job key can read".to_vec();
        let (id, _) = store.put(&key.seal(&plaintext)).unwrap();
        let on_disk = std::fs::read(tmp.join(&id).join("input_video.enc")).unwrap();
        assert!(
            !on_disk.windows(8).any(|w| w == &plaintext[..8]),
            "on-disk bytes must be ciphertext"
        );

        let job_dir = tmp.join("job");
        let input = store.consume(&id, &job_dir, &key).unwrap();
        let (path, raw_key, nonce) = match &input {
            VideoInput::Encrypted { path, key, nonce } => (path.clone(), *key, nonce.clone()),
            VideoInput::Ram(_) => panic!("expected a decrypt-on-read input"),
        };
        assert!(!tmp.join(&id).exists(), "upload dir should be removed");
        let ct = std::fs::read(&path).unwrap();
        let opened = thinfer_app::vault::ephemeral_unseal(&raw_key, &nonce, &ct).unwrap();
        assert_eq!(opened, plaintext);
        assert!(store.consume(&id, &job_dir, &key).is_err(), "consumed once");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn reaper_evicts_expired_uploads_and_their_dirs() {
        let tmp = std::env::temp_dir().join(format!("thinfer_up_reap_{}", random_id()));
        let store = UploadStore::new(tmp.clone());
        let key = JobKey::generate();
        let (id, _) = store.put(&key.seal(b"some clip bytes")).unwrap();
        assert!(tmp.join(&id).exists());
        assert_eq!(store.reap(Duration::from_secs(3600)), 0);
        assert_eq!(store.reap(Duration::ZERO), 1);
        assert!(!tmp.join(&id).exists(), "reaped dir should be gone");
        assert!(store.consume(&id, &tmp.join("job"), &key).is_err());
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
