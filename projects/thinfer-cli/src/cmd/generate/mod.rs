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

mod audio;
mod faceswap;
mod image;
mod text;
mod video;

use audio::GenerateAudio;
use faceswap::GenerateFaceSwap;
use image::GenerateImage;
use text::GenerateText;
use video::GenerateVideo;

#[derive(Subcommand)]
pub enum GenerateCmd {
    /// Generate an image from a prompt.
    Image(GenerateImage),
    /// Generate a video from a prompt (t2v).
    Video(GenerateVideo),
    /// Generate a chat reply with an LLM (streamed to stdout).
    Text(GenerateText),
    /// Generate speech: text in a described voice or a reference clip's voice.
    Audio(GenerateAudio),
    /// Swap a face from a source image into every frame of an input video.
    #[command(name = "face-swap")]
    FaceSwap(GenerateFaceSwap),
}

pub async fn run(cmd: GenerateCmd) -> ExitCode {
    let result = match cmd {
        GenerateCmd::Image(args) => image::run_image(args).await,
        GenerateCmd::Video(args) => video::run_video(args).await,
        GenerateCmd::Text(args) => text::run_text(args).await,
        GenerateCmd::Audio(args) => audio::run_audio(args).await,
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
    /// Queue priority on the `--remote` server: higher runs first (default 0;
    /// e.g. 10 interactive, -10 background).
    #[arg(
        long,
        requires = "remote",
        default_value_t = 0,
        allow_hyphen_values = true
    )]
    pub priority: i32,
    /// Run next on the `--remote` server: when every worker is busy, a
    /// running job that can pause pauses after its current step and resumes
    /// after this one.
    #[arg(long, requires = "remote")]
    pub preempt: bool,
    /// Trust this PEM certificate for an HTTPS `--remote` (a server with
    /// its own `tls_cert`), on top of the system roots.
    #[arg(long, requires = "remote")]
    pub remote_cert: Option<PathBuf>,
    /// Accept any certificate from the `--remote` server (a `tls_self_signed`
    /// one: it is minted fresh at each start). Trusted networks only: anyone on
    /// the network path can then pose as the server and receive the job key,
    /// which decrypts the job's inputs and outputs, and the bearer token.
    #[arg(long, requires = "remote")]
    pub remote_insecure: bool,
}

impl RemoteArgs {
    /// The executor for `--remote` (which must be set).
    pub(super) fn executor(&self) -> Result<RemoteExecutor, String> {
        let url = self.remote.as_deref().expect("executor without --remote");
        let mut b =
            thinfer_client::Client::builder(url).danger_accept_invalid_certs(self.remote_insecure);
        if let Some(t) = &self.remote_token {
            b = b.token(t.as_str());
        }
        if let Some(p) = &self.remote_cert {
            let pem =
                std::fs::read(p).map_err(|e| format!("--remote-cert {}: {e}", p.display()))?;
            b = b.root_certificate_pem(&pem);
        }
        let client = b.build().map_err(|e| format!("--remote: {e}"))?;
        Ok(RemoteExecutor::new(client)
            .with_priority(self.priority)
            .with_preempt(
                self.preempt
                    .then_some(thinfer_app::checkpoint::PauseMode::StepEnd),
            ))
    }
}

/// Shared remote run path: submit `spec` to the `--remote` server, tail its SSE
/// progress through the same stamped stderr sink as a local run, download the
/// result to `output`, and print the footer.
pub(super) async fn run_remote(
    remote: &RemoteArgs,
    spec: JobSpec,
    output: PathBuf,
) -> Result<(), String> {
    let executor = remote.executor()?;
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
    let executor = local_executor(files, download_as_needed).await?;

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

/// Consent + download missing `files`, then build the local backend.
async fn local_executor(
    files: &[FileRef],
    download_as_needed: bool,
) -> Result<LocalExecutor, String> {
    ensure_files(files, download_as_needed).await?;
    new_local().await
}

/// Consent + download the missing `files`.
pub(super) async fn ensure_files(
    files: &[FileRef],
    download_as_needed: bool,
) -> Result<(), String> {
    let missing = download::missing(files)?;
    if !missing.is_empty()
        && !confirm_download(&missing, download_as_needed).map_err(|e| e.to_string())?
    {
        return Err("declined download; rerun with --download-as-needed or `hf download …`".into());
    }
    download::ensure(files, &CliReporter).await
}

/// A local backend on a fresh device (dropping it releases all its VRAM).
pub(super) async fn new_local() -> Result<LocalExecutor, String> {
    LocalExecutor::new(backend_config_from_env()).await
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
        // Native CUDA hot ops when available; `THINFER_CUDA=off` forces wgpu.
        cuda: std::env::var("THINFER_CUDA").map_or(true, |v| v != "off"),
    }
}

/// Stamped stderr progress sink: every line is prefixed with elapsed-from-start
/// so per-stage durations read off directly. Structured [`Stage`]s render to the
/// historical wording; free-form `note`s pass through.
pub(super) struct CliSink {
    start: Instant,
    /// Streamed text left stdout mid-line (break it before stderr lines).
    mid_line: std::cell::Cell<bool>,
}

impl CliSink {
    pub(super) fn new() -> Self {
        Self {
            start: Instant::now(),
            mid_line: std::cell::Cell::new(false),
        }
    }

    /// End a streamed reply's line before writing to stderr.
    fn break_line(&self) {
        if self.mid_line.replace(false) {
            println!();
        }
    }

    fn stamp(&self) -> String {
        format!("[{:6.1}s]", self.start.elapsed().as_secs_f64())
    }

    /// The end-of-run summary line (modality-shaped from the summary fields).
    pub(super) fn footer(&self, s: &JobSummary) {
        let elapsed = self.start.elapsed().as_secs_f64();
        self.break_line();
        if let Some(t) = &s.text {
            eprintln!(
                "{} {} prompt tokens, {} generated ({:.1} tok/s){} in {elapsed:.1}s",
                self.stamp(),
                t.prompt_tokens,
                t.generated_tokens,
                t.generated_tokens.saturating_sub(1) as f64 / t.decode_secs.max(1e-9),
                s.seed.map(|v| format!(", seed {v}")).unwrap_or_default(),
            );
            return;
        }
        let out = s.output.display();
        if let Some(a) = &s.audio {
            eprintln!(
                "{} Wrote {out} ({:.1} s at {} Hz{}) in {elapsed:.1}s",
                self.stamp(),
                a.seconds,
                a.sample_rate,
                s.seed.map(|v| format!(", seed {v}")).unwrap_or_default(),
            );
            return;
        }
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
            Stage::Loading => eprintln!("{} Loading model", self.stamp()),
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
            // Frames are 80 ms: this lands every 2 s of audio.
            Stage::Audio { ms } => {
                if ms.is_multiple_of(1000) {
                    eprintln!("{} Generated {} s of audio", self.stamp(), ms / 1000);
                }
            }
        }
    }

    fn note(&self, msg: &str) {
        self.break_line();
        eprintln!("{} {msg}", self.stamp());
    }

    fn text(&self, delta: &str) {
        if delta.is_empty() {
            return;
        }
        print!("{delta}");
        let _ = io::stdout().flush();
        self.mid_line.set(!delta.ends_with('\n'));
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
