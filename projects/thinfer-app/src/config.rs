//! Backend + budget configuration and the end-of-run memory rollup. All knobs
//! are explicit fields here (no env reads): the CLI fills [`BackendConfig`] from
//! `THINFER_*` env vars, `thinfer-serve` from its TOML. Keeping env out of this
//! layer is what lets one process serve many jobs with one fixed config.

use std::sync::Arc;

use thinfer_core::backend::{PowerPreference, WgpuBackend, WgpuConfig};
pub use thinfer_core::policy::ResidencyBudget;
use thinfer_core::policy::parse_bytes;

/// 2 GiB default for both RAM and VRAM. Small enough to run on a low-spec
/// laptop; the residency manager pages weights, so a small budget means more
/// disk traffic, not failure. (Mirrors the historical CLI default.)
pub const DEFAULT_BUDGET_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// How to build the wgpu backend. Defaults match the CLI's historical behavior
/// (HighPerformance, timestamps off).
#[derive(Clone, Copy, Debug)]
pub struct BackendConfig {
    pub power_preference: PowerPreference,
    /// GPU timestamp queries (the trace/rollup machinery). Off in normal runs.
    pub timestamps: bool,
    /// Opt OUT of the cooperative-matrix (tensor-core) path. Default false
    /// (ON when the GPU supports it). CLI maps `THINFER_NO_COOPMAT` here.
    pub disable_coopmat: bool,
}

impl Default for BackendConfig {
    fn default() -> Self {
        // HighPerformance (not None): Vulkan drivers treat an unset preference
        // as a background-priority hint, which clamps clocks on some iGPUs and
        // slows DiT ~2.5x. See the original CLI note.
        Self {
            power_preference: PowerPreference::HighPerformance,
            timestamps: false,
            disable_coopmat: false,
        }
    }
}

/// Build the wgpu backend for the given config.
pub async fn init_backend(cfg: BackendConfig) -> Result<Arc<WgpuBackend>, String> {
    let _s = tracing::info_span!("wgpu_init").entered();
    let wcfg = WgpuConfig {
        power_preference: cfg.power_preference,
        timestamps: cfg.timestamps,
        disable_coopmat: cfg.disable_coopmat,
    };
    Ok(Arc::new(
        WgpuBackend::new_with_config(wcfg)
            .await
            .map_err(|e| format!("wgpu init: {e:?}"))?,
    ))
}

/// Parse a budget string (`8G`, `512M`, raw bytes), defaulting to
/// [`DEFAULT_BUDGET_BYTES`] when absent. `flag` names the source for errors.
pub fn parse_budget(flag: &str, raw: Option<&str>) -> Result<u64, String> {
    match raw {
        Some(s) => parse_bytes(s).map_err(|e| format!("{flag}={s:?}: {e}")),
        None => Ok(DEFAULT_BUDGET_BYTES),
    }
}

/// Print the per-category VRAM/RAM peak rollup to stderr. Caller decides whether
/// to invoke (the CLI gates on `THINFER_TRACE`).
pub fn report_mem(backend: &WgpuBackend, ram_bytes: u64, vram_bytes: u64) {
    let snap = thinfer_core::backend::Backend::mem_account(backend).snapshot();
    eprintln!(
        "[mem] vram TRUE_PEAK={} / budget {} | per-cat peaks: weights={} workspace={} staging={}",
        fmt_mib(snap.vram_total_peak),
        fmt_mib(vram_bytes),
        fmt_mib(snap.vram_weights.1),
        fmt_mib(snap.vram_workspace.1),
        fmt_mib(snap.vram_staging.1),
    );
    eprintln!(
        "[mem] ram  TRUE_PEAK={} / budget {} | per-cat peaks: upload={} readback={} other={}",
        fmt_mib(snap.ram_total_peak),
        fmt_mib(ram_bytes),
        fmt_mib(snap.ram_upload.1),
        fmt_mib(snap.ram_readback.1),
        fmt_mib(snap.ram_other.1),
    );
}

fn fmt_mib(bytes: u64) -> String {
    if bytes >= 1 << 30 {
        format!("{:.2}GiB", bytes as f64 / (1u64 << 30) as f64)
    } else {
        format!("{:.1}MiB", bytes as f64 / (1u64 << 20) as f64)
    }
}

/// Non-cryptographic seed for omitted-seed runs. Mixes nanos and pid so rapid
/// successive invocations don't collide.
pub fn random_seed() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    nanos ^ (std::process::id() as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
}
