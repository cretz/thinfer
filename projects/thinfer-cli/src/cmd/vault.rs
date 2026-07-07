//! `thinfer vault` -- manage the encrypted adapter (LoRA) vault from the CLI,
//! without a running server. Operates on the SAME on-disk vault serve uses (the
//! shared default dir, or `--vault-dir`), so an adapter added here is usable from
//! the web UI on this box. Adapters are scoped by `--model`.
//!
//! The password is read from a hidden interactive prompt (or `THINFER_VAULT_
//! PASSWORD` for automation), never a flag -- it stays out of shell history and
//! the process table, and is never logged.

use std::collections::BTreeMap;
use std::path::PathBuf;

use clap::{Args, Subcommand};
use thinfer_app::request::Secret;
use thinfer_app::vault::{self, Vault};

#[derive(Subcommand)]
pub enum VaultCmd {
    /// Download an adapter from a URL and store it encrypted for a model.
    Add(VaultAdd),
    /// List a model's stored adapters.
    List(VaultList),
    /// Remove a stored adapter by id.
    Remove(VaultRemove),
}

#[derive(Args)]
pub struct VaultAdd {
    /// The model these adapters apply to (e.g. `krea-2-turbo`).
    #[arg(long)]
    pub model: String,
    /// Direct download URL (a Civitai model file link, or any safetensors URL).
    #[arg(long)]
    pub url: String,
    /// Download token (Civitai). Appended as a `token` query param; not stored.
    #[arg(long)]
    pub token: Option<String>,
    /// Display name; defaults to the downloaded filename.
    #[arg(long)]
    pub name: Option<String>,
    /// Suggested blend weight, stored (encrypted) as a hint for later use.
    #[arg(long)]
    pub weight: Option<f32>,
    /// Vault directory. Defaults to the shared location (see the module docs).
    #[arg(long)]
    pub vault_dir: Option<PathBuf>,
}

#[derive(Args)]
pub struct VaultList {
    #[arg(long)]
    pub model: String,
    #[arg(long)]
    pub vault_dir: Option<PathBuf>,
}

#[derive(Args)]
pub struct VaultRemove {
    #[arg(long)]
    pub model: String,
    /// Adapter id (from `vault list`).
    #[arg(long)]
    pub id: String,
    #[arg(long)]
    pub vault_dir: Option<PathBuf>,
}

/// Read the vault password: `THINFER_VAULT_PASSWORD` if set (automation), else a
/// hidden prompt. `confirm` re-prompts and checks a match (add, where a typo on a
/// first-ever password would be unrecoverable).
pub fn read_password(confirm: bool) -> Result<Secret, String> {
    if let Ok(p) = std::env::var("THINFER_VAULT_PASSWORD")
        && !p.is_empty()
    {
        return Ok(Secret::new(p));
    }
    let p = rpassword::prompt_password("Vault password: ")
        .map_err(|e| format!("read password: {e}"))?;
    if p.is_empty() {
        return Err("password must not be empty".into());
    }
    if confirm {
        let again = rpassword::prompt_password("Confirm password: ")
            .map_err(|e| format!("read password: {e}"))?;
        if again != p {
            return Err("passwords do not match".into());
        }
    }
    Ok(Secret::new(p))
}

pub async fn run(cmd: VaultCmd) -> Result<(), String> {
    match cmd {
        VaultCmd::Add(a) => add(a).await,
        VaultCmd::List(l) => list(l),
        VaultCmd::Remove(r) => remove(r),
    }
}

async fn add(a: VaultAdd) -> Result<(), String> {
    if !thinfer_app::model::is_adapter_model(&a.model) {
        return Err(format!("{} does not support adapters", a.model));
    }
    let password = read_password(true)?;
    let vault = Vault::new(vault::resolve_dir(a.vault_dir.as_deref()));
    let model = a.model.clone();

    eprintln!("Downloading {} ...", a.url);
    let (filename, bytes) = vault::download(&a.url, a.token.as_deref())
        .await
        .map_err(|e| e.to_string())?;
    let tensors = vault::ensure_safetensors(&bytes).map_err(|e| e.to_string())?;
    let name = a.name.unwrap_or(filename);

    let mut extra = BTreeMap::new();
    if let Some(w) = a.weight {
        extra.insert("weight".to_string(), w.to_string());
    }
    let info = vault
        .add(password.expose(), &model, &name, &bytes, extra)
        .map_err(|e| e.to_string())?;
    println!(
        "added \"{}\" for {model} ({tensors} tensors, {:.1} MB) id={}",
        info.name,
        info.size as f64 / (1024.0 * 1024.0),
        info.id,
    );
    Ok(())
}

fn list(l: VaultList) -> Result<(), String> {
    let password = read_password(false)?;
    let vault = Vault::new(vault::resolve_dir(l.vault_dir.as_deref()));
    let model = l.model.to_string();
    let items = vault
        .list(password.expose(), &model)
        .map_err(|e| e.to_string())?;
    if items.is_empty() {
        println!("no adapters stored for {model}");
        return Ok(());
    }
    println!("{} adapter(s) for {model}:", items.len());
    for e in items {
        let weight = e
            .extra
            .get("weight")
            .map(|w| format!(" weight={w}"))
            .unwrap_or_default();
        println!(
            "  {}  {:>8.1} MB  {}{weight}",
            e.id,
            e.size as f64 / (1024.0 * 1024.0),
            e.name,
        );
    }
    Ok(())
}

fn remove(r: VaultRemove) -> Result<(), String> {
    let password = read_password(false)?;
    let vault = Vault::new(vault::resolve_dir(r.vault_dir.as_deref()));
    vault
        .remove(password.expose(), &r.model.to_string(), &r.id)
        .map_err(|e| e.to_string())?;
    println!("removed {}", r.id);
    Ok(())
}
