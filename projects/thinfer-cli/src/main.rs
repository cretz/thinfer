use std::process::ExitCode;

use clap::{Parser, Subcommand};

mod cmd;

#[derive(Parser)]
#[command(name = "thinfer", version, about = "thinfer dev CLI")]
struct Cli {
    #[command(subcommand)]
    cmd: Top,
}

#[derive(Subcommand)]
enum Top {
    /// Model-level dev tools.
    Model {
        #[command(subcommand)]
        cmd: cmd::model::ModelCmd,
    },
    /// Generation entry points.
    Generate {
        #[command(subcommand)]
        cmd: cmd::generate::GenerateCmd,
    },
}

fn main() -> ExitCode {
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;
    // Subscriber emits to stderr. Default filter is `warn` so a normal run is
    // quiet; set `RUST_LOG=info` for pipeline lifecycle, `debug` for per-step
    // / per-dispatch, `trace` for every weight upload + eviction. Targets are
    // namespaced under `thinfer::*` (see `thinfer-core::trace`).
    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn"));
    // CLOSE events emit `time.busy` / `time.idle` per span on exit, giving
    // us per-phase wall-clock without manual Instant captures.
    let fmt_layer = tracing_subscriber::fmt::layer()
        .with_writer(std::io::stderr)
        .with_span_events(tracing_subscriber::fmt::format::FmtSpan::CLOSE);
    // `THINFER_TRACE=1` opts into the rollup layer; otherwise the SCOPE-target
    // callsites stay never-interested (single atomic load per scope!).
    let rollup_handle = match thinfer_core::trace::rollup_layer_from_env() {
        Some((layer, handle)) => {
            tracing_subscriber::registry()
                .with(env_filter)
                .with(fmt_layer)
                .with(layer)
                .init();
            Some(handle)
        }
        None => {
            tracing_subscriber::registry()
                .with(env_filter)
                .with(fmt_layer)
                .init();
            None
        }
    };
    let cli = Cli::parse();
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("build tokio runtime");
    let code = rt.block_on(async {
        match cli.cmd {
            Top::Model { cmd: sub } => cmd::model::run(sub).await,
            Top::Generate { cmd: sub } => cmd::generate::run(sub).await,
        }
    });
    if let Some(h) = rollup_handle {
        let _ = h.dump(&mut std::io::stderr());
    }
    code
}
