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

/// Content-Security-Policy for every response. The UI is one same-origin
/// module script and stylesheet with no inline code; results and input
/// thumbnails display from `blob:` URLs the page decrypts into; every request
/// (API, `EventSource`) is same-origin. Nothing else may load, connect, frame
/// the page or be framed: injected markup cannot run script or ship plaintext
/// (prompts, decrypted results) anywhere.
const CSP: &str = "default-src 'none'; script-src 'self'; style-src 'self'; \
    img-src 'self' blob:; media-src 'self' blob:; connect-src 'self'; \
    form-action 'self'; base-uri 'none'; frame-ancestors 'none'";

/// Security headers on every response (UI and API), applied as the outermost
/// layer: the CSP above, no MIME sniffing, no referrer to other origins, an
/// isolated browsing context group, and no device / sensor permissions.
pub async fn security_headers(mut res: axum::response::Response) -> axum::response::Response {
    use axum::http::HeaderValue;
    let h = res.headers_mut();
    for (name, value) in [
        (header::CONTENT_SECURITY_POLICY, CSP),
        (header::X_CONTENT_TYPE_OPTIONS, "nosniff"),
        (header::REFERRER_POLICY, "no-referrer"),
        (
            header::HeaderName::from_static("cross-origin-opener-policy"),
            "same-origin",
        ),
        (
            header::HeaderName::from_static("permissions-policy"),
            "camera=(), microphone=(), geolocation=(), usb=(), payment=()",
        ),
    ] {
        h.insert(name, HeaderValue::from_static(value));
    }
    res
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
