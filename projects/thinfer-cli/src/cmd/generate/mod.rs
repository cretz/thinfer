//! `thinfer generate` subcommands. The image and video paths live in their own
//! modules (`image`, `video`); this parent holds the bits they share: backend
//! init, residency-budget parsing, download consent, output-format inference,
//! and the end-of-run mem rollup.

use std::io::{self, BufRead, IsTerminal, Write};
use std::process::ExitCode;
use std::sync::Arc;

use clap::Subcommand;
use thinfer_core::backend::{PowerPreference, WgpuBackend, WgpuConfig};
use thinfer_core::manifest::FileRef;
use thinfer_core::policy::parse_bytes;

mod image;
mod video;

use image::GenerateImage;
use video::GenerateVideo;

/// 2 GiB default for both RAM and VRAM. Chosen so a low-spec laptop can run
/// at all; larger budgets help, but the residency manager pages weights so a
/// small budget just means more disk traffic, not failure. Override with
/// `--ram-budget` / `--vram-budget`.
const DEFAULT_BUDGET_BYTES: u64 = 2 * 1024 * 1024 * 1024;

#[derive(Subcommand)]
pub enum GenerateCmd {
    /// Generate an image from a prompt.
    Image(GenerateImage),
    /// Generate a video from a prompt (t2v).
    Video(GenerateVideo),
}

pub async fn run(cmd: GenerateCmd) -> ExitCode {
    let result = match cmd {
        GenerateCmd::Image(args) => image::run_image(args).await,
        GenerateCmd::Video(args) => video::run_video(args).await,
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(1)
        }
    }
}

/// Resolve an output format: an explicit `--output-format` wins; otherwise
/// infer from the `--output` file extension. A missing or unrecognized
/// extension is a hard error so we never write (e.g.) PNG bytes into a `.jpg`.
/// `from_ext` receives the lower-cased extension; `known` lists the recognized
/// extensions for the failure message.
fn resolve_output_format<T: Copy>(
    explicit: Option<T>,
    output: &std::path::Path,
    from_ext: impl Fn(&str) -> Option<T>,
    known: &str,
) -> Result<T, String> {
    if let Some(f) = explicit {
        return Ok(f);
    }
    let ext = output.extension().and_then(|e| e.to_str()).ok_or_else(|| {
        format!(
            "cannot infer output format: {} has no file extension. Pass --output-format or use a known extension ({known}).",
            output.display(),
        )
    })?;
    from_ext(&ext.to_ascii_lowercase()).ok_or_else(|| {
        format!("cannot infer output format from extension {ext:?}; known: {known}. Pass --output-format.")
    })
}

/// Build the wgpu backend with the shared power-preference + timestamp gating.
async fn init_backend() -> Result<Arc<WgpuBackend>, String> {
    let _s = tracing::info_span!("wgpu_init").entered();
    // Default `HighPerformance` (not `None`) because Vulkan drivers treat an
    // unset preference as a background-priority hint: on Intel Arc iGPU this
    // clamps clocks / shrinks the subgroup_size range, slowing DiT by ~2.5x.
    // Users who explicitly want thin-hardware-friendly scheduling can set
    // `THINFER_POWER_PREF=low`.
    let cfg = WgpuConfig {
        power_preference: match std::env::var("THINFER_POWER_PREF")
            .ok()
            .as_deref()
            .map(str::to_ascii_lowercase)
            .as_deref()
        {
            Some("high" | "highperformance" | "discrete") => PowerPreference::HighPerformance,
            Some("low" | "lowpower" | "integrated") => PowerPreference::LowPower,
            Some("none") => PowerPreference::None,
            _ => PowerPreference::HighPerformance,
        },
        timestamps: std::env::var("THINFER_TRACE").is_ok(),
    };
    Ok(Arc::new(
        WgpuBackend::new_with_config(cfg)
            .await
            .map_err(|e| format!("wgpu init: {e:?}"))?,
    ))
}

/// An `Arc<WgpuBackend>` clone retained only when `THINFER_TRACE` is set, used
/// for the end-of-run mem rollup (same gate that enables the rollup table in
/// `main.rs`, so a single env var turns on the full report).
fn backend_for_stats(backend: &Arc<WgpuBackend>) -> Option<Arc<WgpuBackend>> {
    std::env::var_os("THINFER_TRACE").map(|_| Arc::clone(backend))
}

/// Print the per-category VRAM/RAM peak rollup. No-op unless the caller kept a
/// `backend_for_stats` handle (THINFER_TRACE).
fn report_mem(backend: &WgpuBackend, ram_bytes: u64, vram_bytes: u64) {
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

/// Non-cryptographic seed for `--seed`-omitted runs. Mixes nanos and pid so
/// rapid successive invocations don't collide.
fn random_seed() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    nanos ^ (std::process::id() as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

fn parse_budget(flag: &str, raw: Option<&str>) -> Result<u64, String> {
    match raw {
        Some(s) => parse_bytes(s).map_err(|e| format!("{flag}={s:?}: {e}")),
        None => Ok(DEFAULT_BUDGET_BYTES),
    }
}

fn validate_dim(name: &str, v: u32) -> Result<(), String> {
    // Both models narrow by vae_scale = vae_factor*2 (16). No min/max bound
    // upstream; UIs cap purely for UX.
    const VAE_SCALE: u32 = 16;
    if v == 0 {
        return Err(format!("--{name} must be > 0"));
    }
    if !v.is_multiple_of(VAE_SCALE) {
        return Err(format!(
            "--{name} must be a multiple of {VAE_SCALE} (got {v})"
        ));
    }
    Ok(())
}

/// Emits a stderr line at each 10% boundary. hf-hub fans chunks across tasks
/// and the adapter clones the `Arc` per chunk, so `update` calls are racy -
/// state lives in atomics.
struct PercentLogger {
    name: String,
    size: std::sync::atomic::AtomicU64,
    downloaded: std::sync::atomic::AtomicU64,
    last_decile: std::sync::atomic::AtomicU8,
}

impl PercentLogger {
    fn new(name: String) -> Self {
        Self {
            name,
            size: std::sync::atomic::AtomicU64::new(0),
            downloaded: std::sync::atomic::AtomicU64::new(0),
            last_decile: std::sync::atomic::AtomicU8::new(0),
        }
    }
}

impl thinfer_native::cache::DownloadProgress for PercentLogger {
    fn init(&self, size: u64) {
        use std::sync::atomic::Ordering::Relaxed;
        self.size.store(size, Relaxed);
        self.downloaded.store(0, Relaxed);
        self.last_decile.store(0, Relaxed);
        tracing::info!(
            target: thinfer_core::trace::DIAG,
            name = %self.name,
            gib = size as f64 / (1024.0 * 1024.0 * 1024.0),
            "downloading",
        );
    }
    fn update(&self, delta: u64) {
        use std::sync::atomic::Ordering::Relaxed;
        let size = self.size.load(Relaxed);
        if size == 0 {
            return;
        }
        let new = self.downloaded.fetch_add(delta, Relaxed) + delta;
        let pct = ((new.min(size) * 10) / size) as u8;
        let prev = self.last_decile.fetch_max(pct, Relaxed);
        if pct > prev {
            // Cover gaps when a single chunk crosses several deciles, and when
            // hf-hub's resume update jumps from 0 to N0% in one call.
            for p in (prev + 1)..=pct {
                tracing::info!(target: thinfer_core::trace::DIAG, name = %self.name, pct = p * 10, "download progress");
            }
        }
    }
    fn finish(&self) {
        tracing::info!(target: thinfer_core::trace::DIAG, name = %self.name, "download done");
    }
}

fn confirm_download(missing: &[FileRef], download_as_needed: bool) -> io::Result<bool> {
    eprintln!("{} file(s) not in HF cache:", missing.len());
    for f in missing {
        eprintln!("  {}/{}", f.repo, f.path);
    }
    if download_as_needed {
        return Ok(true);
    }
    let stdin = io::stdin();
    if !stdin.is_terminal() {
        eprintln!("non-interactive: use --download-as-needed to proceed");
        return Ok(false);
    }
    eprint!("download now? [y/N] ");
    io::stderr().flush()?;
    let mut line = String::new();
    stdin.lock().read_line(&mut line)?;
    Ok(matches!(line.trim(), "y" | "Y" | "yes" | "Yes"))
}
