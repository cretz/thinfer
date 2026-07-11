//! `thinfer generate` subcommands. Each modality is a clap arg struct (its own
//! module) that maps into a `thinfer_app::JobRequest`; the heavy lifting (load,
//! generate, encode) lives in `thinfer-app`. This parent holds the CLI-only
//! glue: env -> backend config, the interactive download consent prompt, decile
//! download logging, the stamped stderr progress sink, and the mem rollup.

use std::io::{self, BufRead, IsTerminal, Write};
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use clap::{Args, Subcommand};
use thinfer_app::config::BackendConfig;
use thinfer_app::download::{self, DownloadReporter};
use thinfer_app::progress::{ProgressSink, Stage};
use thinfer_app::remote::RemoteExecutor;
use thinfer_app::wire::JobSpec;
use thinfer_app::{JobRequest, JobSummary, LocalExecutor, report_mem};
use thinfer_core::backend::PowerPreference;
use thinfer_core::manifest::FileRef;
use thinfer_native::cache::DownloadProgress;

mod faceswap;
mod image;
mod video;

use faceswap::GenerateFaceSwap;
use image::GenerateImage;
use video::GenerateVideo;

#[derive(Subcommand)]
pub enum GenerateCmd {
    /// Generate an image from a prompt.
    Image(GenerateImage),
    /// Generate a video from a prompt (t2v).
    Video(GenerateVideo),
    /// Swap a face from a source image into every frame of an input video.
    #[command(name = "face-swap")]
    FaceSwap(GenerateFaceSwap),
}

pub async fn run(cmd: GenerateCmd) -> ExitCode {
    let result = match cmd {
        GenerateCmd::Image(args) => image::run_image(args).await,
        GenerateCmd::Video(args) => video::run_video(args).await,
        GenerateCmd::FaceSwap(args) => faceswap::run_faceswap(args).await,
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(1)
        }
    }
}

/// Flags shared by the generate subcommands that can run against a remote
/// `thinfer-serve` instead of this machine. Flattened into each modality's args.
#[derive(Args)]
pub struct RemoteArgs {
    /// Run the job on a remote `thinfer-serve` at this base URL (e.g.
    /// `http://box:8080`) instead of locally. The result is downloaded to
    /// `--output`. Budgets / download policy are the server's; weight files live
    /// on the server.
    #[arg(long)]
    pub remote: Option<String>,
    /// Bearer token for an `auth_token`-protected `--remote` server.
    #[arg(long, requires = "remote")]
    pub remote_token: Option<String>,
}

/// Shared remote run path: submit `spec` to the `--remote` server, tail its SSE
/// progress through the same stamped stderr sink as a local run, download the
/// result to `output`, and print the footer.
pub(super) async fn run_remote(
    remote: &RemoteArgs,
    spec: JobSpec,
    output: PathBuf,
) -> Result<(), String> {
    let url = remote
        .remote
        .as_deref()
        .expect("run_remote without --remote");
    let executor = RemoteExecutor::new(url, remote.remote_token.clone())?;
    let sink = CliSink::new();
    let summary = executor.run(&spec, &output, &sink).await?;
    sink.footer(&summary);
    Ok(())
}

/// Shared run path: consent + download missing files, build the backend, run the
/// job through a stamped stderr sink, print the footer, and (under
/// `THINFER_TRACE`) the mem rollup. `ram`/`vram` are passed through for that
/// rollup (they already live in the request budget).
pub(super) async fn run_job(
    req: JobRequest,
    files: &[FileRef],
    download_as_needed: bool,
    ram: u64,
    vram: u64,
) -> Result<(), String> {
    let missing = download::missing(files);
    if !missing.is_empty()
        && !confirm_download(&missing, download_as_needed).map_err(|e| e.to_string())?
    {
        return Err("declined download; rerun with --download-as-needed or `hf download …`".into());
    }
    download::ensure(files, &CliReporter).await?;

    let backend_cfg = backend_config_from_env();
    let executor = LocalExecutor::new(backend_cfg).await?;

    // Timer starts post-download + post-GPU-init, so it reflects generation work
    // (matches the historical CLI placement).
    let sink = CliSink::new();
    let summary = executor.run(&req, &sink).await?;
    sink.footer(&summary);

    if std::env::var_os("THINFER_TRACE").is_some() {
        report_mem(executor.backend(), ram, vram);
    }
    Ok(())
}

/// Build the wgpu backend config from `THINFER_*` env (the binary layer reads
/// env; `thinfer-app` itself never does). Default `HighPerformance`: Vulkan
/// drivers treat an unset preference as a background-priority hint, slowing DiT
/// ~2.5x on some iGPUs. `THINFER_TRACE` enables GPU timestamps for the rollup.
fn backend_config_from_env() -> BackendConfig {
    let power_preference = match std::env::var("THINFER_POWER_PREF")
        .ok()
        .as_deref()
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("high" | "highperformance" | "discrete") => PowerPreference::HighPerformance,
        Some("low" | "lowpower" | "integrated") => PowerPreference::LowPower,
        Some("none") => PowerPreference::None,
        _ => PowerPreference::HighPerformance,
    };
    BackendConfig {
        power_preference,
        timestamps: std::env::var("THINFER_TRACE").is_ok(),
        disable_coopmat: std::env::var("THINFER_NO_COOPMAT").is_ok(),
    }
}

/// Stamped stderr progress sink: every line is prefixed with elapsed-from-start
/// so per-stage durations read off directly. Structured [`Stage`]s render to the
/// historical wording; free-form `note`s pass through.
struct CliSink {
    start: Instant,
}

impl CliSink {
    fn new() -> Self {
        Self {
            start: Instant::now(),
        }
    }

    fn stamp(&self) -> String {
        format!("[{:6.1}s]", self.start.elapsed().as_secs_f64())
    }

    /// The end-of-run summary line (modality-shaped from the summary fields).
    fn footer(&self, s: &JobSummary) {
        let elapsed = self.start.elapsed().as_secs_f64();
        let out = s.output.display();
        match (s.fps, s.seed) {
            (None, seed) => eprintln!(
                "{} Wrote {out} ({}x{}{}) in {elapsed:.1}s",
                self.stamp(),
                s.width,
                s.height,
                seed.map(|v| format!(", seed {v}")).unwrap_or_default(),
            ),
            (Some(fps), Some(seed)) => eprintln!(
                "{} Wrote {out} ({}x{}, {} frames @ {fps} fps, seed {seed}) in {elapsed:.1}s",
                self.stamp(),
                s.width,
                s.height,
                s.frames,
            ),
            (Some(fps), None) => eprintln!(
                "{} Wrote {out} ({}x{}, {} frames @ {fps} fps) in {elapsed:.1}s",
                self.stamp(),
                s.width,
                s.height,
                s.frames,
            ),
        }
    }
}

impl ProgressSink for CliSink {
    fn stage(&self, stage: Stage) {
        match stage {
            Stage::TextEncode => eprintln!("{} Encoding prompt", self.stamp()),
            Stage::Step { i, n } => eprintln!("{} Denoising step {i}/{n}", self.stamp()),
            Stage::ChunkStep {
                chunk,
                num_chunks,
                step,
                num_steps,
            } => eprintln!(
                "{} Denoising chunk {chunk}/{num_chunks} step {step}/{num_steps}",
                self.stamp()
            ),
            Stage::VaeDecode => eprintln!("{} Decoding latents (VAE)", self.stamp()),
            // Throttle to every 10th frame (matches the historical face-swap log).
            Stage::FrameSwapped { done, total } => {
                if done.is_multiple_of(10) {
                    eprintln!("{} Swapped frame {done}/{total}", self.stamp());
                }
            }
        }
    }

    fn note(&self, msg: &str) {
        eprintln!("{} {msg}", self.stamp());
    }
}

/// Attaches a [`PercentLogger`] per downloaded file.
struct CliReporter;

impl DownloadReporter for CliReporter {
    fn for_file(&self, file: &FileRef) -> Option<Arc<dyn DownloadProgress>> {
        Some(Arc::new(PercentLogger::new(format!(
            "{}/{}",
            file.repo, file.path
        ))))
    }
}

/// Emits a stderr line at each 10% boundary. hf-hub fans chunks across tasks and
/// the adapter clones the `Arc` per chunk, so `update` calls are racy: state
/// lives in atomics.
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

impl DownloadProgress for PercentLogger {
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
