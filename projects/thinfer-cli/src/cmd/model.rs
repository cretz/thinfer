use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::Subcommand;
use thinfer_core::format::safetensors::SafetensorsSource;
use thinfer_core::sanity::{self, FailReason, Outcome, Report};
use thinfer_core::tensor::Shape;
use thinfer_native::MmapFileOpener;

#[derive(Subcommand)]
pub enum ModelCmd {
    /// Validate that every tensor in the given safetensors file(s) decodes via
    /// the canonical storage->compute path.
    SanityCheck {
        #[arg(required = true)]
        files: Vec<PathBuf>,
    },
}

pub async fn run(cmd: ModelCmd) -> ExitCode {
    match cmd {
        ModelCmd::SanityCheck { files } => sanity_check(files).await,
    }
}

async fn sanity_check(files: Vec<PathBuf>) -> ExitCode {
    let mut total_tensors = 0usize;
    let mut total_pass = 0usize;
    let mut total_fail = 0usize;
    let mut any_open_err = false;

    for file in &files {
        println!("== {}", file.display());
        match check_one_file(file).await {
            Ok(report) => {
                print_report(&report);
                total_tensors += report.tensors.len();
                total_pass += report.pass_count();
                total_fail += report.fail_count();
            }
            Err(e) => {
                println!("  OPEN-ERROR  {e}");
                any_open_err = true;
            }
        }
    }

    println!(
        "TOTAL: files={} tensors={} pass={} fail={}",
        files.len(),
        total_tensors,
        total_pass,
        total_fail
    );
    if total_fail > 0 || any_open_err {
        ExitCode::from(1)
    } else {
        ExitCode::SUCCESS
    }
}

async fn check_one_file(path: &Path) -> Result<Report, String> {
    let opener = MmapFileOpener::new(path)
        .await
        .map_err(|e| format!("open {}: {e}", path.display()))?;
    let src = SafetensorsSource::open(opener)
        .await
        .map_err(|e| format!("parse {}: {e:?}", path.display()))?;
    Ok(sanity::check_source(&src).await)
}

fn print_report(report: &Report) {
    for t in &report.tensors {
        match &t.outcome {
            Outcome::Pass { decoded_bytes } => {
                println!(
                    "  PASS  {}  {}  {}  on-disk={}B  decoded={}B",
                    t.name,
                    t.encoding_label,
                    fmt_shape(&t.shape),
                    t.bytes_on_disk,
                    decoded_bytes,
                );
            }
            Outcome::Fail(reason) => {
                println!(
                    "  FAIL  {}  {}  {}  {}",
                    t.name,
                    t.encoding_label,
                    fmt_shape(&t.shape),
                    fmt_reason(reason),
                );
            }
        }
    }
    println!(
        "  -- pass={} fail={} (of {})",
        report.pass_count(),
        report.fail_count(),
        report.tensors.len()
    );
}

fn fmt_shape(shape: &Shape) -> String {
    let dims: Vec<String> = shape.0.iter().map(|d| d.to_string()).collect();
    format!("[{}]", dims.join(","))
}

fn fmt_reason(r: &FailReason) -> String {
    match r {
        FailReason::UnknownStorageDtype(s) => format!("unknown-dtype={s}"),
        FailReason::NoDecoder(enc) => format!("no-decoder={enc:?}"),
        FailReason::SizeMismatch { expected, got } => {
            format!("size-mismatch expected={expected} got={got}")
        }
        FailReason::ReadError(s) => format!("read-error={s}"),
    }
}
