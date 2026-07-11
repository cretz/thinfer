//! Result-at-rest encryption. End-to-end by design: the client (browser)
//! generates an RSA keypair and sends only its PUBLIC key with the job; the
//! server encrypts the artifact with it and CANNOT decrypt -- only the client's
//! private key can. Hybrid scheme, because RSA can't encrypt MB-sized media:
//!
//!   - a fresh random AES-256-GCM key encrypts the artifact bytes,
//!   - RSA-OAEP(SHA-256) wraps that AES key under the client's public key.
//!
//! On-disk / on-wire layout (all big-endian framing):
//!   [u16 wrapped_key_len][wrapped AES key][12-byte nonce][AES-GCM ciphertext+tag]
//!
//! WebCrypto on the browser does the inverse: RSA-OAEP decrypt the wrapped key,
//! then AES-GCM decrypt the body.

use aes_gcm::aead::Aead;
use aes_gcm::{Aes256Gcm, KeyInit, Nonce};
use base64::Engine;
use rsa::RsaPublicKey;
use rsa::pkcs8::DecodePublicKey;
use sha2::Sha256;

const NONCE_LEN: usize = 12;

/// Encrypt `plaintext` for the holder of `public_key_b64` (base64-encoded SPKI
/// DER of an RSA public key). Returns the framed blob described in the module
/// docs.
pub fn encrypt_for(public_key_b64: &str, plaintext: &[u8]) -> Result<Vec<u8>, String> {
    let der = base64::engine::general_purpose::STANDARD
        .decode(public_key_b64.trim())
        .map_err(|e| format!("public key is not valid base64: {e}"))?;
    let public_key =
        RsaPublicKey::from_public_key_der(&der).map_err(|e| format!("bad SPKI public key: {e}"))?;

    let mut rng = rand::thread_rng();
    let mut aes_key = [0u8; 32];
    let mut nonce = [0u8; NONCE_LEN];
    rand::RngCore::fill_bytes(&mut rng, &mut aes_key);
    rand::RngCore::fill_bytes(&mut rng, &mut nonce);

    let cipher = Aes256Gcm::new_from_slice(&aes_key).map_err(|e| format!("aes key: {e}"))?;
    let body = cipher
        .encrypt(Nonce::from_slice(&nonce), plaintext)
        .map_err(|e| format!("aes-gcm encrypt: {e}"))?;

    let wrapped = public_key
        .encrypt(&mut rng, rsa::Oaep::new::<Sha256>(), &aes_key)
        .map_err(|e| format!("rsa-oaep wrap: {e}"))?;
    let wlen = u16::try_from(wrapped.len())
        .map_err(|_| "wrapped key too large".to_string())?
        .to_be_bytes();

    let mut out = Vec::with_capacity(2 + wrapped.len() + NONCE_LEN + body.len());
    out.extend_from_slice(&wlen);
    out.extend_from_slice(&wrapped);
    out.extend_from_slice(&nonce);
    out.extend_from_slice(&body);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rsa::pkcs8::EncodePublicKey;
    use rsa::{Oaep, RsaPrivateKey};

    // Round-trip through the exact wire format the browser parses: generate a
    // keypair, encrypt with the public half, then decrypt by hand with the
    // private half (RSA-OAEP unwrap -> AES-GCM open).
    #[test]
    fn encrypt_for_round_trips_via_the_wire_format() {
        let mut rng = rand::thread_rng();
        let private = RsaPrivateKey::new(&mut rng, 2048).unwrap();
        let public = private.to_public_key();
        let spki_b64 = base64::engine::general_purpose::STANDARD
            .encode(public.to_public_key_der().unwrap().as_bytes());

        let plaintext = b"the quick brown fox \x00\x01\x02 \xff and some bytes";
        let blob = encrypt_for(&spki_b64, plaintext).unwrap();

        let wlen = u16::from_be_bytes([blob[0], blob[1]]) as usize;
        let wrapped = &blob[2..2 + wlen];
        let nonce = &blob[2 + wlen..2 + wlen + NONCE_LEN];
        let body = &blob[2 + wlen + NONCE_LEN..];

        let aes_key = private.decrypt(Oaep::new::<Sha256>(), wrapped).unwrap();
        let cipher = Aes256Gcm::new_from_slice(&aes_key).unwrap();
        let decrypted = cipher.decrypt(Nonce::from_slice(nonce), body).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn rejects_garbage_public_key() {
        assert!(encrypt_for("not base64!!!", b"x").is_err());
        assert!(encrypt_for("YWJjZA==", b"x").is_err()); // valid base64, not SPKI
    }
}
