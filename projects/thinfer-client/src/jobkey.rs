//! Per-job encryption. The client generates one AES-256-GCM key per job and
//! seals everything that job carries with it: the job spec (prompts, images),
//! uploaded media, and the produced artifact. The server holds only ciphertext
//! until a worker is about to run the job; then it asks the client for the key,
//! decrypts in RAM right before the data feeds the model, seals the output in
//! RAM before anything touches disk, and drops (zeroes) the key and plaintext
//! when the job ends. Without the key, nothing the job left behind is readable.
//!
//! A text session's key (`TextSessionRef.key`) is the same kind of key.
//!
//! Blob layout: `[12-byte nonce][AES-GCM ciphertext + 16-byte tag]`. A fresh
//! random nonce per seal; one key seals several blobs of the same job.

use aes_gcm::aead::{Aead, AeadInPlace};
use aes_gcm::{Aes256Gcm, KeyInit, Nonce, Tag};
use base64::Engine;
use zeroize::Zeroizing;

use crate::Error;

pub const KEY_LEN: usize = 32;
pub const NONCE_LEN: usize = 12;
/// AES-GCM tag length (trails the ciphertext).
pub const TAG_LEN: usize = 16;

/// One job's (or session's) key. Zeroed on drop and never rendered by `Debug`.
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
    /// A fresh random key.
    pub fn generate() -> Self {
        let mut key = Zeroizing::new([0u8; KEY_LEN]);
        rand::RngCore::fill_bytes(&mut rand::thread_rng(), key.as_mut());
        Self(key)
    }

    /// Parse a base64 key. The decoded copy is zeroed too.
    pub fn from_b64(s: &str) -> Result<Self, Error> {
        let raw = Zeroizing::new(
            b64()
                .decode(s.trim())
                .map_err(|e| Error::InvalidKey(format!("not valid base64: {e}")))?,
        );
        if raw.len() != KEY_LEN {
            return Err(Error::InvalidKey(format!(
                "must be {KEY_LEN} bytes, got {}",
                raw.len()
            )));
        }
        let mut key = Zeroizing::new([0u8; KEY_LEN]);
        key.copy_from_slice(&raw);
        Ok(Self(key))
    }

    /// Base64 for delivery to the server (zeroed on drop).
    pub fn to_b64(&self) -> Zeroizing<String> {
        Zeroizing::new(b64().encode(self.0.as_ref()))
    }

    /// SHA-256 of the key: the capability a client presents to manage a job
    /// or session and fetch its sealed result (headers `x-job-key-sha256` /
    /// `x-session-key-sha256`, base64). Reveals nothing about the key, so the
    /// server keeps it after dropping the key itself.
    pub fn proof(&self) -> [u8; 32] {
        use sha2::Digest;
        sha2::Sha256::digest(self.0.as_ref()).into()
    }

    /// [`Self::proof`] in base64, as the API carries it.
    pub fn proof_b64(&self) -> String {
        b64().encode(self.proof())
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
    pub fn open(&self, blob: &[u8]) -> Result<Zeroizing<Vec<u8>>, Error> {
        let mut out = Zeroizing::new(vec![0u8; plaintext_len(blob)?]);
        self.open_into(blob, &mut out)?;
        Ok(out)
    }

    /// Open a blob into `out` (exactly [`plaintext_len`] bytes), decrypting in
    /// place there: the plaintext never lands in another allocation, so a
    /// caller can hand in page-locked memory. On failure `out` is zeroed.
    pub fn open_into(&self, blob: &[u8], out: &mut [u8]) -> Result<(), Error> {
        let n = plaintext_len(blob)?;
        assert_eq!(out.len(), n, "open_into: output length");
        let (nonce, body) = blob.split_at(NONCE_LEN);
        let (ct, tag) = body.split_at(n);
        out.copy_from_slice(ct);
        self.cipher()
            .decrypt_in_place_detached(Nonce::from_slice(nonce), &[], out, Tag::from_slice(tag))
            .map_err(|_| {
                zeroize::Zeroize::zeroize(&mut *out);
                Error::Decrypt
            })
    }
}

/// The plaintext length of a sealed blob.
pub fn plaintext_len(blob: &[u8]) -> Result<usize, Error> {
    blob.len()
        .checked_sub(NONCE_LEN + TAG_LEN)
        .ok_or(Error::Decrypt)
}

/// Split a sealed blob into (base64 nonce, ciphertext+tag), the shape a
/// decrypt-on-read input stores.
pub fn split_sealed(blob: &[u8]) -> Result<(String, &[u8]), Error> {
    plaintext_len(blob)?;
    let (nonce, body) = blob.split_at(NONCE_LEN);
    Ok((b64().encode(nonce), body))
}

/// Seal `plaintext` and base64 it (the form sealed fields take on the wire).
pub fn seal_b64(key: &JobKey, plaintext: &[u8]) -> String {
    b64().encode(key.seal(plaintext))
}

/// Open a base64 sealed field.
pub fn open_b64(key: &JobKey, sealed: &str) -> Result<Zeroizing<Vec<u8>>, Error> {
    let blob = b64()
        .decode(sealed.as_bytes())
        .map_err(|e| Error::Protocol(format!("sealed field is not base64: {e}")))?;
    key.open(&blob)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seal_open_round_trips_and_rejects_the_wrong_key() {
        let k = JobKey::generate();
        let blob = k.seal(b"a prompt");
        assert_eq!(&**k.open(&blob).unwrap(), b"a prompt");
        assert!(matches!(
            JobKey::generate().open(&blob),
            Err(Error::Decrypt)
        ));
        let k2 = JobKey::from_b64(&k.to_b64()).unwrap();
        assert_eq!(&**k2.open(&blob).unwrap(), b"a prompt");
    }

    #[test]
    fn nonces_differ_per_seal() {
        let k = JobKey::generate();
        assert_ne!(k.seal(b"x")[..NONCE_LEN], k.seal(b"x")[..NONCE_LEN]);
    }

    #[test]
    fn short_and_tampered_blobs_fail() {
        let k = JobKey::generate();
        assert!(matches!(k.open(&[0; 27]), Err(Error::Decrypt)));
        let mut blob = k.seal(b"abc");
        *blob.last_mut().unwrap() ^= 1;
        assert!(matches!(k.open(&blob), Err(Error::Decrypt)));
    }

    #[test]
    fn rejects_bad_keys() {
        assert!(JobKey::from_b64("not base64!").is_err());
        assert!(JobKey::from_b64(&b64().encode([0u8; 16])).is_err());
    }
}
