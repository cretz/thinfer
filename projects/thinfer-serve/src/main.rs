//! thinfer-serve: a long-running host exposing image/video/face-swap generation
//! over an async OpenAPI job API. One serial GPU per worker (default 1), a job
//! queue, in-memory metadata, on-disk artifacts.
//!
//! Usage:
//!   thinfer-serve [--config <path.toml>]      # run the server
//!   thinfer-serve --emit-openapi <path.json>  # write the OpenAPI doc and exit

use std::net::SocketAddr;
use std::process::ExitCode;
use std::sync::Arc;

use axum_server::tls_rustls::RustlsConfig;

mod api;
mod config;
mod crypto;
mod job;
mod web;
mod worker;

use api::AppState;
use config::ServeConfig;
use job::JobStore;

fn main() -> ExitCode {
    // rustls 0.23 with the no-provider feature needs a process-wide crypto
    // provider installed before any TLS config is built. Ignore the error if a
    // provider is already installed (e.g. a dependency did it first).
    let _ = rustls::crypto::ring::default_provider().install_default();

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                // Default: info, but mute the per-op parity probes
                // (`thinfer::diag`, thousands of lines + a GPU readback each per
                // generate). Re-enable with `RUST_LOG=info,thinfer::diag=info`.
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info,thinfer::diag=warn")),
        )
        .with_writer(std::io::stderr)
        .init();

    let args: Vec<String> = std::env::args().collect();
    let cli = match parse_args(&args) {
        Ok(cli) => cli,
        Err(e) => {
            eprintln!("error: {e}\n\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    let config_path = match cli {
        Cli::Help => {
            println!("{USAGE}");
            return ExitCode::SUCCESS;
        }
        Cli::EmitOpenapi { path } => return emit_openapi(&path),
        Cli::Serve { config } => config,
    };

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("build tokio runtime");
    match rt.block_on(serve(config_path)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(1)
        }
    }
}

const USAGE: &str = "\
Usage:
  thinfer-serve [--config <path.toml>]      run the server (default: built-in config)
  thinfer-serve --emit-openapi <path.json>  write the OpenAPI doc and exit
  thinfer-serve --help                      show this help";

/// Parsed command line. Unknown / positional args are rejected (not silently
/// dropped) so a misplaced config path can't boot the server with defaults.
enum Cli {
    Serve { config: Option<String> },
    EmitOpenapi { path: String },
    Help,
}

fn parse_args(args: &[String]) -> Result<Cli, String> {
    let mut config = None;
    let mut emit = None;
    let mut i = 1; // skip argv[0]
    // Consume the value following the flag at `i`, erroring if absent.
    let value_after = |i: usize, label: &str| -> Result<String, String> {
        args.get(i + 1)
            .cloned()
            .ok_or_else(|| format!("{label} requires a value"))
    };
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => return Ok(Cli::Help),
            "--config" => {
                config = Some(value_after(i, "--config")?);
                i += 2;
            }
            "--emit-openapi" => {
                emit = Some(value_after(i, "--emit-openapi")?);
                i += 2;
            }
            other => {
                return Err(format!(
                    "unknown argument: {other:?} (the config path must follow --config)"
                ));
            }
        }
    }
    if let Some(path) = emit {
        return Ok(Cli::EmitOpenapi { path });
    }
    Ok(Cli::Serve { config })
}

fn emit_openapi(path: &str) -> ExitCode {
    let doc = api::openapi_doc();
    let json = match doc.to_pretty_json() {
        Ok(j) => j,
        Err(e) => {
            eprintln!("error: serialize openapi: {e}");
            return ExitCode::from(1);
        }
    };
    match std::fs::write(path, json) {
        Ok(()) => {
            eprintln!("wrote {path}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: write {path}: {e}");
            ExitCode::from(1)
        }
    }
}

async fn serve(config_path: Option<String>) -> Result<(), String> {
    match &config_path {
        Some(p) => tracing::info!(config = %p, "loading serve config"),
        None => tracing::info!("no --config; using built-in defaults (HTTP, no TLS)"),
    }
    let config = ServeConfig::load(config_path.as_deref().map(std::path::Path::new))?;
    std::fs::create_dir_all(&config.artifact_dir)
        .map_err(|e| format!("create {}: {e}", config.artifact_dir.display()))?;

    let store = Arc::new(JobStore::default());
    let backend_cfg = config.backend_config();
    let workers = config.workers.max(1);
    for i in 0..workers {
        worker::spawn_worker(i, store.clone(), backend_cfg, config.download_as_needed);
    }
    tracing::info!(
        bind = %config.bind, workers, artifact_dir = %config.artifact_dir.display(),
        "thinfer-serve starting",
    );

    let web = web::router(config.web_dir.as_deref());
    let state = AppState {
        store,
        config: Arc::new(config.clone()),
    };
    let app = api::router(state).merge(web);

    match tls_config(&config).await? {
        Some(tls) => {
            let addr: SocketAddr = config
                .bind
                .parse()
                .map_err(|e| format!("bind addr {} (https needs ip:port): {e}", config.bind))?;
            tracing::info!(%addr, "thinfer-serve listening (https)");
            axum_server::bind_rustls(addr, tls)
                .serve(app.into_make_service())
                .await
                .map_err(|e| format!("serve tls: {e}"))
        }
        None => {
            let listener = tokio::net::TcpListener::bind(&config.bind)
                .await
                .map_err(|e| format!("bind {}: {e}", config.bind))?;
            axum::serve(listener, app)
                .with_graceful_shutdown(shutdown_signal())
                .await
                .map_err(|e| format!("serve: {e}"))
        }
    }
}

/// Build the TLS config when enabled: a bring-your-own cert (`tls_cert`+
/// `tls_key`) wins; otherwise `tls_self_signed` mints a fresh cert at startup.
/// Returns `None` for plain HTTP.
async fn tls_config(config: &ServeConfig) -> Result<Option<RustlsConfig>, String> {
    if let (Some(cert), Some(key)) = (&config.tls_cert, &config.tls_key) {
        let tls = RustlsConfig::from_pem_file(cert, key)
            .await
            .map_err(|e| format!("load tls cert/key: {e}"))?;
        return Ok(Some(tls));
    }
    if config.tls_self_signed {
        let sans = self_signed_sans(&config.tls_sans);
        let (cert_pem, key_pem) = self_signed_pem(&sans)?;
        let tls = RustlsConfig::from_pem(cert_pem.into_bytes(), key_pem.into_bytes())
            .await
            .map_err(|e| format!("self-signed tls: {e}"))?;
        tracing::warn!(
            sans = %sans.join(", "),
            "serving HTTPS with a self-signed cert; clients accept the one-time warning",
        );
        return Ok(Some(tls));
    }
    Ok(None)
}

/// The SAN list for the self-signed cert: localhost + the auto-detected primary
/// LAN IP (so reaching the box by that IP avoids a name-mismatch) + any extras
/// from config. Deduplicated, order preserved.
fn self_signed_sans(extra: &[String]) -> Vec<String> {
    let mut sans = vec!["localhost".to_string(), "127.0.0.1".to_string()];
    if let Some(ip) = primary_local_ip() {
        sans.push(ip);
    }
    sans.extend(extra.iter().cloned());
    sans.dedup();
    sans
}

/// Best-effort primary LAN IP via the default route: a UDP socket "connected"
/// to a public address sends nothing but resolves the local interface address.
/// `None` if offline / no route.
fn primary_local_ip() -> Option<String> {
    let sock = std::net::UdpSocket::bind("0.0.0.0:0").ok()?;
    sock.connect("8.8.8.8:80").ok()?;
    Some(sock.local_addr().ok()?.ip().to_string())
}

/// Generate a self-signed cert + key (PEM) for `sans` (rcgen parses IP-shaped
/// entries as IP SANs, the rest as DNS names).
fn self_signed_pem(sans: &[String]) -> Result<(String, String), String> {
    let cert = rcgen::generate_simple_self_signed(sans.to_vec())
        .map_err(|e| format!("generate self-signed cert: {e}"))?;
    Ok((cert.cert.pem(), cert.key_pair.serialize_pem()))
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
    tracing::info!("shutdown signal received");
}
