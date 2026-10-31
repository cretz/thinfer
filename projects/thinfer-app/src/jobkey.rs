//! Per-job encryption. The client generates one AES-256-GCM key per job and
//! seals everything that job carries with it: the job spec (prompts, images),
//! uploaded media, and the produced artifact. The server holds only ciphertext
//! until a worker is about to run the job; then it asks the client for the key,
//! decrypts in RAM right before the data feeds the model, seals the output in
//! RAM before anything touches disk, and drops (zeroes) the key and plaintext
//! when the job ends. Without the key, nothing the job left behind is readable.
//!
//! Blob layout: `[12-byte nonce][AES-GCM ciphertext + 16-byte tag]`. A fresh
//! random nonce per seal; one key seals several blobs of the same job.

use aes_gcm::aead::Aead;
use aes_gcm::{Aes256Gcm, KeyInit, Nonce};
use base64::Engine;
use zeroize::Zeroizing;

pub const KEY_LEN: usize = 32;
pub const NONCE_LEN: usize = 12;

/// One job's key. Zeroed on drop and never rendered by `Debug`.
pub struct JobKey(Zeroizing<[u8; KEY_LEN]>);

impl std::fmt::Debug for JobKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("JobKey(<redacted>)")
    }
}

fn b64() -> base64::engine::GeneralPurpose {
    base64::engine::general_purpose::STANDARD
}

impl JobKey {
    /// A fresh random key (clients: the CLI remote path, tests).
    pub fn generate() -> Self {
        let mut key = Zeroizing::new([0u8; KEY_LEN]);
        rand::RngCore::fill_bytes(&mut rand::thread_rng(), key.as_mut());
        Self(key)
    }

    /// Parse the base64 key a client delivers. The decoded copy is zeroed too.
    pub fn from_b64(s: &str) -> Result<Self, String> {
        let raw = Zeroizing::new(
            b64()
                .decode(s.trim())
                .map_err(|e| format!("job key is not valid base64: {e}"))?,
        );
        let mut key = Zeroizing::new([0u8; KEY_LEN]);
        if raw.len() != KEY_LEN {
            return Err(format!(
                "job key must be {KEY_LEN} bytes, got {}",
                raw.len()
            ));
        }
        key.copy_from_slice(&raw);
        Ok(Self(key))
    }

    /// Base64 for delivery to the server (zeroed on drop).
    pub fn to_b64(&self) -> Zeroizing<String> {
        Zeroizing::new(b64().encode(self.0.as_ref()))
    }

    /// SHA-256 of the key: the capability a client presents to fetch the
    /// job's sealed result (header `x-job-key-sha256`, base64). Reveals nothing
    /// about the key, so the server keeps it after dropping the key itself.
    pub fn proof(&self) -> [u8; 32] {
        use sha2::Digest;
        sha2::Sha256::digest(self.0.as_ref()).into()
    }

    /// The raw key, for APIs that take bytes (e.g. a decrypt-on-read input).
    pub fn bytes(&self) -> &[u8; KEY_LEN] {
        &self.0
    }

    fn cipher(&self) -> Aes256Gcm {
        Aes256Gcm::new_from_slice(self.0.as_ref()).expect("32-byte key")
    }

    /// Seal `plaintext` into a `[nonce][ciphertext+tag]` blob.
    pub fn seal(&self, plaintext: &[u8]) -> Vec<u8> {
        let mut nonce = [0u8; NONCE_LEN];
        rand::RngCore::fill_bytes(&mut rand::thread_rng(), &mut nonce);
        let body = self
            .cipher()
            .encrypt(Nonce::from_slice(&nonce), plaintext)
            .expect("aes-gcm encrypt");
        let mut out = Vec::with_capacity(NONCE_LEN + body.len());
        out.extend_from_slice(&nonce);
        out.extend_from_slice(&body);
        out
    }

    /// Open a blob from [`Self::seal`]. The plaintext is zeroed on drop.
    pub fn open(&self, blob: &[u8]) -> Result<Zeroizing<Vec<u8>>, String> {
        if blob.len() < NONCE_LEN + 16 {
            return Err("sealed blob too short".into());
        }
        let (nonce, body) = blob.split_at(NONCE_LEN);
        self.cipher()
            .decrypt(Nonce::from_slice(nonce), body)
            .map(Zeroizing::new)
            .map_err(|_| "decrypt failed (wrong job key or corrupt data)".into())
    }
}

/// Split a sealed blob into (base64 nonce, ciphertext+tag), the shape a
/// decrypt-on-read input stores (`request::VideoInput::Encrypted`).
pub fn split_sealed(blob: &[u8]) -> Result<(String, &[u8]), String> {
    if blob.len() < NONCE_LEN + 16 {
        return Err("sealed blob too short".into());
    }
    let (nonce, body) = blob.split_at(NONCE_LEN);
    Ok((b64().encode(nonce), body))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seal_open_round_trips_and_rejects_the_wrong_key() {
        let k = JobKey::generate();
        let blob = k.seal(b"a prompt");
        assert_eq!(&*k.open(&blob).unwrap(), b"a prompt");
        assert!(JobKey::generate().open(&blob).is_err());
        let k2 = JobKey::from_b64(&k.to_b64()).unwrap();
        assert_eq!(&*k2.open(&blob).unwrap(), b"a prompt");
    }

    #[test]
    fn nonces_differ_per_seal() {
        let k = JobKey::generate();
        assert_ne!(k.seal(b"x")[..NONCE_LEN], k.seal(b"x")[..NONCE_LEN]);
    }
}
