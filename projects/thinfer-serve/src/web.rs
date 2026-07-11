//! Static web UI. By default the three assets are compiled into the binary
//! (self-contained deploy); set `web_dir` in the config to serve them from a
//! directory instead (edit-reload dev loop, no rebuild). These routes are mounted
//! outside the auth layer so the page can always load and prompt for a token.

use std::path::Path;

use axum::Router;
use axum::http::header;
use axum::response::IntoResponse;
use axum::routing::get;
use tower_http::services::ServeDir;

const INDEX_HTML: &str = include_str!("../web/index.html");
const STYLE_CSS: &str = include_str!("../web/style.css");
const APP_JS: &str = include_str!("../web/app.js");

/// The UI router. With `web_dir`, files are served live from disk; otherwise the
/// compiled-in assets are served.
pub fn router(web_dir: Option<&Path>) -> Router {
    match web_dir {
        Some(dir) => Router::new().fallback_service(ServeDir::new(dir)),
        None => Router::new()
            .route("/", get(|| asset("text/html; charset=utf-8", INDEX_HTML)))
            .route(
                "/index.html",
                get(|| asset("text/html; charset=utf-8", INDEX_HTML)),
            )
            .route("/style.css", get(|| asset("text/css", STYLE_CSS)))
            .route("/app.js", get(|| asset("text/javascript", APP_JS))),
    }
}

async fn asset(content_type: &'static str, body: &'static str) -> impl IntoResponse {
    // no-store so a redeploy's HTML/JS/CSS is never served stale from cache
    // (otherwise an old app.js lingers and mis-reads new event fields).
    (
        [
            (header::CONTENT_TYPE, content_type),
            (header::CACHE_CONTROL, "no-store"),
        ],
        body,
    )
}
