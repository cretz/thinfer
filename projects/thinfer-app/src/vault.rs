//! User-driven, encrypted-at-rest adapter (LoRA) vault. Shared by every
//! front-end: the CLI drives it directly, `thinfer-serve` wraps it in HTTP
//! endpoints, and the web UI rides those. It owns only crypto + storage +
//! download; it knows nothing about models or the GPU (a caller decrypts an
//! entry to bytes and hands it to the LoRA fold).
//!
//! STATELESS re: the key. The user's password rides in on every op; the vault
//! re-derives the key each time (Argon2id over a per-vault salt) and never
//! persists or caches anything decrypted. No recovery: lose the password, lose
//! whatever was sealed under it.
//!
//! MULTIPLE PASSWORDS, one vault. There is no vault-wide password: each entry is
//! sealed independently under a key derived from the password supplied at its
//! `add`, so different passwords partition the vault into disjoint views. `list`
//! returns only the entries the given password decrypts and SKIPS the rest, so a
//! password that unlocks nothing yields an empty list -- indistinguishable from
//! an empty vault (no oracle). `open`/`remove` still fail closed with an opaque
//! [`VaultError::Auth`] when the password can't decrypt the target entry (you can
//! only remove what you can read). The per-vault salt is shared across the user's
//! own passwords; each password still derives an independent key, and GCM's
//! per-seal random nonce keeps ciphertexts distinct. Trade-off of dropping the
//! old verifier: a typo at `add` time silently seals under the typo'd key (there
//! is nothing to reject it against), so it only reappears under the same typo.
//!
//! Adapters are scoped BY MODEL: a LoRA belongs to the model it was trained for
//! (a Krea adapter is meaningless on another DiT), so `add`/`list`/`open` all
//! take a model id and only ever see that model's entries.
//!
//! DISK INVARIANT: a reader of the vault dir learns only how many blobs exist
//! and their sizes. Never which adapters, for which models, or their metadata --
//! entry names + per-adapter settings live inside an AES-GCM `enc_meta` blob,
//! and each content blob is a random-id file holding ciphertext only. The one
//! plaintext file is `index.json`: the salt and per-entry
//! nonces/ciphertext/blob-id/size (the size already leaks via the blob's own
//! file length, so it is no new exposure).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use aes_gcm::aead::Aead;
use aes_gcm::{Aes256Gcm, KeyInit, Nonce};
use argon2::Argon2;
use base64::Engine;
use rand::RngCore;
use serde::{Deserialize, Serialize};

const B64: base64::engine::general_purpose::GeneralPurpose =
    base64::engine::general_purpose::STANDARD;
const SALT_LEN: usize = 16;
const NONCE_LEN: usize = 12;
const KEY_LEN: usize = 32;
const INDEX_FILE: &str = "index.json";
const INDEX_VERSION: u32 = 1;

/// What `list` returns and what a caller references in a generate request: an
/// opaque per-entry id, the (decrypted) display name, and the plaintext size.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "serve", derive(utoipa::ToSchema))]
#[serde(rename_all = "camelCase")]
pub struct VaultEntryInfo {
    /// Stable handle (the random blob id). Referenced in the `lora` field of a
    /// generate request; unique within a model.
    pub id: String,
    pub name: String,
    pub size: u64,
    /// Per-adapter settings the user attached at add-time (e.g. a suggested
    /// weight, a variant tag). Empty for a plain adapter. Decrypted alongside the
    /// name, so it is as secret at rest as the name is.
    #[serde(default)]
    pub extra: BTreeMap<String, String>,
}

#[derive(Debug)]
pub enum VaultError {
    Io(String),
    /// The password cannot decrypt the target entry (wrong password, or a
    /// tampered/corrupt entry). Deliberately opaque: the same variant either way,
    /// so it is not a password oracle.
    Auth,
    /// No such adapter id under this model.
    NotFound,
    Download(String),
    /// The bytes are not a usable safetensors adapter (validated by content, not
    /// by filename -- Civitai download URLs are extensionless).
    Format(String),
    Serde(String),
}

impl std::fmt::Display for VaultError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VaultError::Io(e) => write!(f, "vault io: {e}"),
            VaultError::Auth => write!(f, "invalid password or corrupt vault"),
            VaultError::NotFound => write!(f, "no such adapter"),
            VaultError::Download(e) => write!(f, "download: {e}"),
            VaultError::Format(e) => write!(f, "{e}"),
            VaultError::Serde(e) => write!(f, "vault index: {e}"),
        }
    }
}
impl std::error::Error for VaultError {}

/// One stored adapter. Every field but `blob_id`/`size` is ciphertext or a
/// nonce; `enc_meta` seals [`EntryMeta`] (the name + per-adapter settings).
#[derive(Clone, Debug, Serialize, Deserialize)]
struct Entry {
    blob_id: String,
    meta_nonce: String,
    enc_meta: String,
    content_nonce: String,
    size: u64,
}

/// The plaintext-JSON on-disk index. `salt` is vault-wide (the KDF input);
/// entries are grouped by model id. Only ciphertext, nonces, blob ids, and sizes
/// here. (Older indexes carried a `verifier`/`verifier_nonce`; serde ignores
/// those now-unknown fields, so a pre-multi-password vault still loads.)
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct Index {
    version: u32,
    salt: String,
    /// model id -> its adapters.
    models: BTreeMap<String, Vec<Entry>>,
}

/// The decrypted per-entry metadata (never written in the clear).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct EntryMeta {
    name: String,
    #[serde(default)]
    extra: BTreeMap<String, String>,
}

/// A vault rooted at a directory. Cheap to construct; holds no key material. The
/// mutation lock serializes in-process read-modify-write of the index (the salt
/// and one blob write per add); cross-process writers reload fresh each op.
pub struct Vault {
    dir: PathBuf,
    write_lock: Mutex<()>,
}

impl Vault {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self {
            dir: dir.into(),
            write_lock: Mutex::new(()),
        }
    }

    fn index_path(&self) -> PathBuf {
        self.dir.join(INDEX_FILE)
    }
    fn blob_path(&self, blob_id: &str) -> PathBuf {
        self.dir.join(format!("{blob_id}.blob"))
    }

    /// Load the index, or `None` if the vault has never been initialized (no
    /// `add` yet). A missing dir/file is "empty", not an error.
    fn load_index(&self) -> Result<Option<Index>, VaultError> {
        match std::fs::read(self.index_path()) {
            Ok(bytes) => {
                let idx: Index =
                    serde_json::from_slice(&bytes).map_err(|e| VaultError::Serde(e.to_string()))?;
                Ok(Some(idx))
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(VaultError::Io(e.to_string())),
        }
    }

    fn store_index(&self, idx: &Index) -> Result<(), VaultError> {
        std::fs::create_dir_all(&self.dir).map_err(|e| VaultError::Io(e.to_string()))?;
        let bytes = serde_json::to_vec_pretty(idx).map_err(|e| VaultError::Serde(e.to_string()))?;
        // Write via a temp file + rename so a crash mid-write can't truncate the
        // index (the blobs it points at would then be unreachable ciphertext).
        let tmp = self.index_path().with_extension("json.tmp");
        std::fs::write(&tmp, &bytes).map_err(|e| VaultError::Io(e.to_string()))?;
        std::fs::rename(&tmp, self.index_path()).map_err(|e| VaultError::Io(e.to_string()))
    }

    /// Derive the entry cipher for `password` under the vault salt. There is no
    /// verifier: whether a password is "right" is decided per entry, when its
    /// `enc_meta` (or content) is decrypted -- so different passwords derive
    /// different keys and see disjoint slices of the vault.
    fn cipher_for(&self, idx: &Index, password: &str) -> Result<Aes256Gcm, VaultError> {
        let salt = B64.decode(&idx.salt).map_err(|_| VaultError::Auth)?;
        derive_cipher(password, &salt)
    }

    /// Adapters for `model` that THIS password decrypts. An uninitialized vault
    /// (or a model with no adapters) is an empty list. Entries sealed under other
    /// passwords -- or genuinely corrupt ones -- are SKIPPED, not errors, so one
    /// unreadable entry never hides the rest and a password that unlocks nothing
    /// just returns `[]` (no oracle). Only an authentication failure is skipped;
    /// a decode error on already-authenticated plaintext (real corruption)
    /// propagates.
    pub fn list(&self, password: &str, model: &str) -> Result<Vec<VaultEntryInfo>, VaultError> {
        let Some(idx) = self.load_index()? else {
            return Ok(Vec::new());
        };
        let cipher = self.cipher_for(&idx, password)?;
        let Some(entries) = idx.models.get(model) else {
            return Ok(Vec::new());
        };
        let mut out = Vec::new();
        for e in entries {
            match decrypt_entry(&cipher, e) {
                Ok(info) => out.push(info),
                Err(VaultError::Auth) => continue,
                Err(other) => return Err(other),
            }
        }
        Ok(out)
    }

    /// Add `bytes` (a raw safetensors adapter) under `model` with display `name`
    /// and optional per-adapter `extra` settings. Mints the vault salt on first
    /// use; any password is accepted and seals a private entry readable only by
    /// that same password. Returns the new entry's info. The plaintext bytes are
    /// encrypted and dropped; only ciphertext hits disk.
    pub fn add(
        &self,
        password: &str,
        model: &str,
        name: &str,
        bytes: &[u8],
        extra: BTreeMap<String, String>,
    ) -> Result<VaultEntryInfo, VaultError> {
        let _guard = self.write_lock.lock().expect("vault write lock");
        // Load-or-init the index. First add mints the vault salt; any password is
        // accepted (there is no vault-wide verifier) and seals a private entry.
        let (mut idx, cipher) = match self.load_index()? {
            Some(idx) => {
                let cipher = self.cipher_for(&idx, password)?;
                (idx, cipher)
            }
            None => {
                let mut salt = [0u8; SALT_LEN];
                rand::thread_rng().fill_bytes(&mut salt);
                let cipher = derive_cipher(password, &salt)?;
                let idx = Index {
                    version: INDEX_VERSION,
                    salt: B64.encode(salt),
                    models: BTreeMap::new(),
                };
                (idx, cipher)
            }
        };

        let meta = EntryMeta {
            name: name.to_string(),
            extra,
        };
        let meta_json = serde_json::to_vec(&meta).map_err(|e| VaultError::Serde(e.to_string()))?;
        let (meta_nonce, enc_meta) = seal(&cipher, &meta_json)?;
        let (content_nonce, ciphertext) = seal(&cipher, bytes)?;

        let blob_id = random_id();
        std::fs::create_dir_all(&self.dir).map_err(|e| VaultError::Io(e.to_string()))?;
        // Blob holds RAW ciphertext (no base64); the index carries its nonce.
        std::fs::write(self.blob_path(&blob_id), &ciphertext)
            .map_err(|e| VaultError::Io(e.to_string()))?;

        let entry = Entry {
            blob_id: blob_id.clone(),
            meta_nonce,
            enc_meta: B64.encode(&enc_meta),
            content_nonce,
            size: bytes.len() as u64,
        };
        idx.models
            .entry(model.to_string())
            .or_default()
            .push(entry.clone());
        self.store_index(&idx)?;
        decrypt_entry(&cipher, &entry)
    }

    /// Decrypt the adapter bytes for `(model, id)`. For the use path: the caller
    /// wraps the returned bytes in an in-memory safetensors source and folds.
    pub fn open(&self, password: &str, model: &str, id: &str) -> Result<Vec<u8>, VaultError> {
        let idx = self.load_index()?.ok_or(VaultError::NotFound)?;
        let cipher = self.cipher_for(&idx, password)?;
        let entry = idx
            .models
            .get(model)
            .and_then(|es| es.iter().find(|e| e.blob_id == id))
            .ok_or(VaultError::NotFound)?;
        let ct = std::fs::read(self.blob_path(&entry.blob_id))
            .map_err(|e| VaultError::Io(e.to_string()))?;
        let nonce = B64
            .decode(&entry.content_nonce)
            .map_err(|_| VaultError::Auth)?;
        unseal(&cipher, &nonce, &ct)
    }

    /// Remove `(model, id)`: delete the blob and drop the index entry. Needs the
    /// password (so a disk-only attacker can't prune the vault). Idempotent-ish:
    /// a missing id is [`VaultError::NotFound`].
    pub fn remove(&self, password: &str, model: &str, id: &str) -> Result<(), VaultError> {
        let _guard = self.write_lock.lock().expect("vault write lock");
        let mut idx = self.load_index()?.ok_or(VaultError::NotFound)?;
        let cipher = self.cipher_for(&idx, password)?;
        let entries = idx.models.get_mut(model).ok_or(VaultError::NotFound)?;
        let pos = entries
            .iter()
            .position(|e| e.blob_id == id)
            .ok_or(VaultError::NotFound)?;
        // Authorize by ownership: you can only remove an entry your password
        // decrypts, so one password can't prune another's adapters.
        decrypt_entry(&cipher, &entries[pos])?;
        let entry = entries.remove(pos);
        if entries.is_empty() {
            idx.models.remove(model);
        }
        let _ = std::fs::remove_file(self.blob_path(&entry.blob_id));
        self.store_index(&idx)
    }
}

/// Download an adapter from `url` (a direct file link, e.g. a Civitai download
/// URL). `token` is appended as a `token` query param (Civitai's scheme) when
/// present -- never logged, never stored. Follows redirects (Civitai -> object
/// store). Returns `(filename, bytes)`; the filename prefers a
/// `Content-Disposition` and falls back to the URL's last path segment.
pub async fn download(url: &str, token: Option<&str>) -> Result<(String, Vec<u8>), VaultError> {
    let mut parsed = reqwest::Url::parse(url).map_err(|e| VaultError::Download(e.to_string()))?;
    if let Some(tok) = token {
        // Only add if the caller didn't already put a token in the URL.
        let has = parsed.query_pairs().any(|(k, _)| k == "token");
        if !has {
            parsed.query_pairs_mut().append_pair("token", tok);
        }
    }
    let resp = reqwest::Client::new()
        .get(parsed.clone())
        .send()
        .await
        .map_err(|e| VaultError::Download(e.to_string()))?
        .error_for_status()
        .map_err(|e| VaultError::Download(e.to_string()))?;

    let filename = resp
        .headers()
        .get(reqwest::header::CONTENT_DISPOSITION)
        .and_then(|v| v.to_str().ok())
        .and_then(filename_from_disposition)
        .or_else(|| {
            parsed
                .path_segments()
                .and_then(|mut s| s.next_back())
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string())
        })
        .unwrap_or_else(|| "adapter.safetensors".to_string());

    let bytes = resp
        .bytes()
        .await
        .map_err(|e| VaultError::Download(e.to_string()))?
        .to_vec();
    Ok((filename, bytes))
}

/// Pull a `filename="..."` (or bare `filename=...`) out of a Content-Disposition
/// header value. Strips surrounding quotes; ignores RFC 5987 `filename*`.
fn filename_from_disposition(v: &str) -> Option<String> {
    for part in v.split(';') {
        let part = part.trim();
        if let Some(rest) = part.strip_prefix("filename=") {
            let name = rest.trim().trim_matches('"');
            if !name.is_empty() {
                return Some(name.to_string());
            }
        }
    }
    None
}

/// Validate that `bytes` are a usable safetensors adapter, by CONTENT (the
/// header), never by filename -- Civitai download URLs are extensionless and
/// most LoRAs ship as safetensors. Front-ends call this right after download so
/// a wrong file (a pickle `.pt`/`.ckpt`, or an HTML error page when a token is
/// missing) is rejected immediately with a clear message instead of at generate
/// time. Returns the tensor count on success.
pub fn ensure_safetensors(bytes: &[u8]) -> Result<usize, VaultError> {
    match thinfer_core::format::safetensors::parse(bytes) {
        Ok(cat) if !cat.entries.is_empty() => Ok(cat.entries.len()),
        Ok(_) => Err(VaultError::Format(
            "file is a valid safetensors container but has no tensors".into(),
        )),
        Err(_) => Err(VaultError::Format(
            "not a safetensors adapter (only safetensors LoRAs are supported; \
             .pt/.ckpt pickles are not). If this is a Civitai link, check the token."
                .into(),
        )),
    }
}

/// Argon2id(password, salt) -> a ready AES-256-GCM cipher. Memory-hard KDF so a
/// stolen vault dir can't be brute-forced cheaply.
fn derive_cipher(password: &str, salt: &[u8]) -> Result<Aes256Gcm, VaultError> {
    let mut key = [0u8; KEY_LEN];
    Argon2::default()
        .hash_password_into(password.as_bytes(), salt, &mut key)
        .map_err(|_| VaultError::Auth)?;
    Aes256Gcm::new_from_slice(&key).map_err(|_| VaultError::Auth)
}

/// Encrypt `plaintext` under a fresh random nonce; returns `(b64 nonce, raw
/// ciphertext+tag)`. The caller base64s the ciphertext for the JSON index
/// (small fields) or writes it raw to a blob (large adapter bytes -- no 33%
/// base64 bloat).
fn seal(cipher: &Aes256Gcm, plaintext: &[u8]) -> Result<(String, Vec<u8>), VaultError> {
    let mut nonce = [0u8; NONCE_LEN];
    rand::thread_rng().fill_bytes(&mut nonce);
    let ct = cipher
        .encrypt(Nonce::from_slice(&nonce), plaintext)
        .map_err(|_| VaultError::Auth)?;
    Ok((B64.encode(nonce), ct))
}

/// Decrypt a `(nonce, ciphertext+tag)` pair. Any length/auth failure collapses
/// to [`VaultError::Auth`] (no oracle). The nonce size is inferred from the
/// cipher at the `decrypt` call.
fn unseal(cipher: &Aes256Gcm, nonce: &[u8], ct: &[u8]) -> Result<Vec<u8>, VaultError> {
    if nonce.len() != NONCE_LEN {
        return Err(VaultError::Auth);
    }
    cipher
        .decrypt(Nonce::from_slice(nonce), ct)
        .map_err(|_| VaultError::Auth)
}

/// Encrypt `plaintext` under a FRESH random 32-byte AES-256-GCM key (no KDF):
/// for per-request EPHEMERAL media encryption where the key lives only in RAM
/// for the request's lifetime and is thrown away with it. Returns `(key, b64
/// nonce, raw ciphertext+tag)`. Distinct from the vault's password-derived keys:
/// this is for an at-rest disk spill of an uploaded video whose key never
/// persists (see the encrypted upload path in `thinfer-serve`). The caller holds
/// `key` in RAM and hands it back to [`ephemeral_unseal`] to decrypt-on-read.
pub fn ephemeral_seal(plaintext: &[u8]) -> ([u8; KEY_LEN], String, Vec<u8>) {
    let mut key = [0u8; KEY_LEN];
    rand::thread_rng().fill_bytes(&mut key);
    let cipher = Aes256Gcm::new_from_slice(&key).expect("32-byte key is valid");
    let (nonce, ct) = seal(&cipher, plaintext).expect("aes-gcm seal");
    (key, nonce, ct)
}

/// Decrypt a blob sealed by [`ephemeral_seal`] with its RAM-held `key` + `nonce`.
/// Any auth/length failure collapses to [`VaultError::Auth`].
pub fn ephemeral_unseal(
    key: &[u8; KEY_LEN],
    nonce_b64: &str,
    ct: &[u8],
) -> Result<Vec<u8>, VaultError> {
    let cipher = Aes256Gcm::new_from_slice(key).map_err(|_| VaultError::Auth)?;
    let nonce = B64.decode(nonce_b64).map_err(|_| VaultError::Auth)?;
    unseal(&cipher, &nonce, ct)
}

fn decrypt_entry(cipher: &Aes256Gcm, e: &Entry) -> Result<VaultEntryInfo, VaultError> {
    let nonce = B64.decode(&e.meta_nonce).map_err(|_| VaultError::Auth)?;
    let ct = B64.decode(&e.enc_meta).map_err(|_| VaultError::Auth)?;
    let pt = unseal(cipher, &nonce, &ct)?;
    let meta: EntryMeta =
        serde_json::from_slice(&pt).map_err(|e| VaultError::Serde(e.to_string()))?;
    Ok(VaultEntryInfo {
        id: e.blob_id.clone(),
        name: meta.name,
        size: e.size,
        extra: meta.extra,
    })
}

/// 32 hex chars of randomness: the public, opaque blob id (and filename).
fn random_id() -> String {
    let mut b = [0u8; 16];
    rand::thread_rng().fill_bytes(&mut b);
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// Default vault dir under a base data directory. `base` is the serve/CLI data
/// root.
pub fn default_dir(base: &Path) -> PathBuf {
    base.join("vault")
}

/// Resolve the vault directory a front-end should use, so the CLI and serve
/// agree on one vault by default (an adapter added via the CLI is then usable
/// from the browser on the same box). Priority: an explicit path (a `--vault-dir`
/// flag or the `vault_dir` serve.toml key) > the `THINFER_VAULT_DIR` env var >
/// `<hf-cache>/vault` (cwd-independent, alongside the model cache).
pub fn resolve_dir(explicit: Option<&Path>) -> PathBuf {
    if let Some(p) = explicit {
        return p.to_path_buf();
    }
    if let Ok(env) = std::env::var("THINFER_VAULT_DIR")
        && !env.trim().is_empty()
    {
        return PathBuf::from(env);
    }
    default_dir(&thinfer_native::cache::cache_root())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp() -> tempfile::TempDir {
        tempfile::tempdir().unwrap()
    }

    #[test]
    fn add_list_open_round_trips_per_model() {
        let d = tmp();
        let v = Vault::new(d.path());
        let bytes = b"fake safetensors bytes \x00\x01\xff".to_vec();
        let mut extra = BTreeMap::new();
        extra.insert("weight".to_string(), "0.8".to_string());
        let info = v
            .add(
                "hunter2",
                "krea-2-turbo",
                "my-style.safetensors",
                &bytes,
                extra.clone(),
            )
            .unwrap();
        assert_eq!(info.name, "my-style.safetensors");
        assert_eq!(info.size, bytes.len() as u64);
        assert_eq!(info.extra.get("weight").map(String::as_str), Some("0.8"));

        // list sees it under its model, and NOT under another model.
        let listed = v.list("hunter2", "krea-2-turbo").unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].id, info.id);
        assert!(v.list("hunter2", "some-other-model").unwrap().is_empty());

        // open returns the exact bytes.
        let got = v.open("hunter2", "krea-2-turbo", &info.id).unwrap();
        assert_eq!(got, bytes);
    }

    #[test]
    fn wrong_password_lists_empty_and_open_remove_fail_closed() {
        let d = tmp();
        let v = Vault::new(d.path());
        v.add("correct", "m", "a", b"x", BTreeMap::new()).unwrap();
        let id = v.list("correct", "m").unwrap()[0].id.clone();
        // A password that decrypts nothing sees an empty list, not an error
        // (no oracle: indistinguishable from an empty vault).
        assert!(v.list("wrong", "m").unwrap().is_empty());
        // But open/remove of a known id fail closed under the wrong password.
        assert!(matches!(v.open("wrong", "m", &id), Err(VaultError::Auth)));
        assert!(matches!(v.remove("wrong", "m", &id), Err(VaultError::Auth)));
        // The correct password still owns it.
        assert_eq!(v.open("correct", "m", &id).unwrap(), b"x");
    }

    #[test]
    fn multiple_passwords_partition_one_vault() {
        let d = tmp();
        let v = Vault::new(d.path());
        // Two adapters under the SAME model but DIFFERENT passwords.
        let a = v
            .add("pw-a", "m", "alpha", b"AAA", BTreeMap::new())
            .unwrap();
        let b = v.add("pw-b", "m", "beta", b"BBB", BTreeMap::new()).unwrap();

        // Each password sees only its own entry.
        let la = v.list("pw-a", "m").unwrap();
        assert_eq!(la.len(), 1);
        assert_eq!(la[0].name, "alpha");
        let lb = v.list("pw-b", "m").unwrap();
        assert_eq!(lb.len(), 1);
        assert_eq!(lb[0].name, "beta");

        // Cross-password open fails; own open works.
        assert_eq!(v.open("pw-a", "m", &a.id).unwrap(), b"AAA");
        assert!(matches!(v.open("pw-a", "m", &b.id), Err(VaultError::Auth)));

        // pw-a cannot remove pw-b's adapter, but pw-b can.
        assert!(matches!(
            v.remove("pw-a", "m", &b.id),
            Err(VaultError::Auth)
        ));
        v.remove("pw-b", "m", &b.id).unwrap();
        assert!(v.list("pw-b", "m").unwrap().is_empty());
        // Removing pw-b's entry left pw-a's intact.
        assert_eq!(v.list("pw-a", "m").unwrap().len(), 1);
    }

    #[test]
    fn no_plaintext_name_on_disk() {
        let d = tmp();
        let v = Vault::new(d.path());
        v.add(
            "pw",
            "krea-2-turbo",
            "secret-adapter-name",
            b"blobbytes",
            BTreeMap::new(),
        )
        .unwrap();
        // The index must not contain the plaintext name or the model-scoped bytes.
        let index = std::fs::read(d.path().join(INDEX_FILE)).unwrap();
        assert!(!contains(&index, b"secret-adapter-name"));
        // No blob file contains the plaintext content.
        for entry in std::fs::read_dir(d.path()).unwrap() {
            let p = entry.unwrap().path();
            if p.extension().and_then(|e| e.to_str()) == Some("blob") {
                assert!(!contains(&std::fs::read(&p).unwrap(), b"blobbytes"));
            }
        }
    }

    #[test]
    fn remove_deletes_blob_and_entry() {
        let d = tmp();
        let v = Vault::new(d.path());
        let info = v.add("pw", "m", "a", b"bytes", BTreeMap::new()).unwrap();
        assert!(d.path().join(format!("{}.blob", info.id)).exists());
        v.remove("pw", "m", &info.id).unwrap();
        assert!(!d.path().join(format!("{}.blob", info.id)).exists());
        assert!(v.list("pw", "m").unwrap().is_empty());
        assert!(matches!(
            v.remove("pw", "m", &info.id),
            Err(VaultError::NotFound)
        ));
    }

    #[test]
    fn filename_from_disposition_parses() {
        assert_eq!(
            filename_from_disposition("attachment; filename=\"cool-lora.safetensors\""),
            Some("cool-lora.safetensors".to_string())
        );
        assert_eq!(
            filename_from_disposition("inline; filename=plain.bin"),
            Some("plain.bin".to_string())
        );
        assert_eq!(filename_from_disposition("attachment"), None);
    }

    fn contains(haystack: &[u8], needle: &[u8]) -> bool {
        haystack.windows(needle.len()).any(|w| w == needle)
    }
}
