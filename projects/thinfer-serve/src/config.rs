//! Deployment configuration, loaded from a TOML file (with sane defaults). This
//! is the server/host layer: bind address, worker pool size, artifact storage,
//! retention, download policy, and the GPU/budget knobs. Per-generation defaults
//! (steps, dims) are NOT here -- they live in the model registry
//! (`thinfer_app::model`) so the CLI and API never drift.

use std::path::{Path, PathBuf};

use serde::Deserialize;
use thinfer_app::config::{BackendConfig, ResidencyBudget};
use thinfer_core::backend::PowerPreference;

#[derive(Clone, Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ServeConfig {
    /// Address to bind. Defaults to `0.0.0.0:8080` (reachable on the LAN, e.g.
    /// from a phone). Set `auth_token` before exposing it beyond a trusted
    /// network.
    pub bind: String,
    /// Optional bearer token. When set, every `/jobs*` request (and the OpenAPI
    /// doc) must carry `Authorization: Bearer <token>`; the static web UI loads
    /// unauthenticated so it can prompt for the token. `None` = open access.
    pub auth_token: Option<String>,
    /// Serve HTTPS with a self-signed cert generated at startup. Needed for the
    /// browser's WebCrypto (result encryption) over a LAN IP: WebCrypto only
    /// runs in a secure context (https or localhost). Clients see a one-time
    /// cert warning to accept. Ignored when `tls_cert`/`tls_key` are set.
    pub tls_self_signed: bool,
    /// Extra Subject Alternative Names (DNS names or IPs) to put in the
    /// self-signed cert, on top of `localhost`, `127.0.0.1`, and the
    /// auto-detected primary LAN IP. Add other addresses you reach the box by
    /// (e.g. a Tailscale IP) so they avoid a cert name-mismatch warning.
    pub tls_sans: Vec<String>,
    /// Bring-your-own TLS cert (PEM). Set with `tls_key` to serve HTTPS with a
    /// trusted cert instead of a self-signed one.
    pub tls_cert: Option<PathBuf>,
    /// Bring-your-own TLS private key (PEM); pairs with `tls_cert`.
    pub tls_key: Option<PathBuf>,
    /// Directory of static web-UI assets to serve at `/`. When unset, the
    /// compiled-in UI is served (self-contained binary); set it for an
    /// edit-reload dev loop without a rebuild.
    pub web_dir: Option<PathBuf>,
    /// GPU worker pool size. One executor (and one resident model) per worker;
    /// on a single-GPU box keep this at 1 (the DiT denoise saturates one GPU).
    pub workers: usize,
    /// Where job outputs are written (one subdir per job id).
    pub artifact_dir: PathBuf,
    /// Where the encrypted adapter (LoRA) vault lives. Unset = the shared default
    /// (`THINFER_VAULT_DIR` env, else `<hf-cache>/vault`) so an adapter added via
    /// the CLI on this box is usable from the web UI. Set it to pin a location.
    pub vault_dir: Option<PathBuf>,
    /// Artifact retention in seconds (informational sweep target in v1).
    pub retention_secs: u64,
    /// Download missing weight files automatically (no interactive consent on a
    /// server). When false, a job whose files are missing fails fast.
    pub download_as_needed: bool,
    /// `high` | `low` | `none` (wgpu power preference). Default `high`.
    pub power_preference: String,
    /// Residency budgets (e.g. `5G`, `512M`, raw bytes).
    pub ram_budget: String,
    pub vram_budget: String,
}

impl Default for ServeConfig {
    fn default() -> Self {
        Self {
            bind: "0.0.0.0:8080".into(),
            auth_token: None,
            tls_self_signed: false,
            tls_sans: Vec::new(),
            tls_cert: None,
            tls_key: None,
            web_dir: None,
            workers: 1,
            artifact_dir: PathBuf::from("thinfer-artifacts"),
            vault_dir: None,
            retention_secs: 24 * 60 * 60,
            download_as_needed: true,
            power_preference: "high".into(),
            ram_budget: "5G".into(),
            vram_budget: "5G".into(),
        }
    }
}

impl ServeConfig {
    /// Load from a TOML file, or all-defaults when `path` is `None`.
    pub fn load(path: Option<&Path>) -> Result<Self, String> {
        let Some(path) = path else {
            return Ok(Self::default());
        };
        let text =
            std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
        toml::from_str(&text).map_err(|e| format!("parse {}: {e}", path.display()))
    }

    pub fn backend_config(&self) -> BackendConfig {
        let power_preference = match self.power_preference.to_ascii_lowercase().as_str() {
            "low" | "lowpower" | "integrated" => PowerPreference::LowPower,
            "none" => PowerPreference::None,
            _ => PowerPreference::HighPerformance,
        };
        BackendConfig {
            power_preference,
            // Server runs do not drive the trace rollup.
            timestamps: false,
            // Coopmat (tensor-core) path stays ON when the GPU supports it.
            disable_coopmat: false,
        }
    }

    /// The resolved vault directory (config override, else the shared default).
    pub fn resolved_vault_dir(&self) -> PathBuf {
        thinfer_app::vault::resolve_dir(self.vault_dir.as_deref())
    }

    pub fn budget(&self) -> Result<ResidencyBudget, String> {
        Ok(ResidencyBudget {
            ram_bytes: thinfer_app::parse_budget("ram_budget", Some(&self.ram_budget))?,
            vram_bytes: thinfer_app::parse_budget("vram_budget", Some(&self.vram_budget))?,
        })
    }
}
