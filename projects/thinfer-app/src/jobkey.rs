//! Per-job encryption ([`thinfer_client::jobkey`]), plus the server's way of
//! opening a blob: into page-locked memory.

pub use thinfer_client::jobkey::*;

/// Opening a sealed blob into page-locked memory (the server side: job
/// plaintext must not reach swap). Use this, not `JobKey::open`, wherever the
/// plaintext feeds a job.
pub trait OpenLocked {
    /// The plaintext, page-locked and zeroed on drop: decrypted in place
    /// inside the locked buffer, never in an ordinary allocation, and not at
    /// all when the lock fails.
    fn open_locked(&self, blob: &[u8]) -> Result<thinfer_core::secmem::Locked<u8>, String>;
}

impl OpenLocked for JobKey {
    fn open_locked(&self, blob: &[u8]) -> Result<thinfer_core::secmem::Locked<u8>, String> {
        let mut out = thinfer_core::secmem::Locked::<u8>::zeroed(plaintext_len(blob)?);
        if out.lock_failed() {
            return Err("could not lock job data in RAM (not enough lockable memory)".into());
        }
        self.open_into(blob, &mut out)?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_locked_round_trips() {
        let k = JobKey::generate();
        let blob = k.seal(b"a prompt");
        assert_eq!(&*k.open_locked(&blob).unwrap(), b"a prompt");
        assert!(JobKey::generate().open_locked(&blob).is_err());
    }
}
