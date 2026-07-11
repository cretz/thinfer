//! HTTP surface for the encrypted adapter (LoRA) vault. Thin wrappers over
//! `thinfer_app::vault`: the crypto/storage lives there so the CLI drives the
//! same vault without a server. Every op carries the password in its body (over
//! TLS); the server holds no key and caches nothing decrypted. Blocking crypto/
//! IO runs on `spawn_blocking` so a large add can't stall the async runtime.
//!
//! Adapters are scoped by `model`: `list`/`add`/`remove` only ever see one
//! model's adapters. The password is never logged (these request types do not
//! derive `Debug`).

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::Json;
use serde::{Deserialize, Serialize};
use thinfer_app::vault::{VaultEntryInfo, VaultError};
use utoipa::ToSchema;

use crate::api::{ApiError, AppState};

/// `POST /vault/adapters/add` body. `token` (Civitai) and `password` are
/// transient secrets -- never stored, never logged (no `Debug`).
#[derive(Deserialize, ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct AddAdapterRequest {
    /// Model id (kebab wire string) these adapters apply to. Image or video;
    /// validated against [`thinfer_app::model::is_adapter_model`].
    pub model: String,
    /// Direct download URL (a Civitai model file link, or any safetensors URL).
    pub url: String,
    /// Optional download token (Civitai). Appended as a `token` query param.
    #[serde(default)]
    pub token: Option<String>,
    /// Display name override; defaults to the downloaded filename.
    #[serde(default)]
    pub name: Option<String>,
    /// Suggested blend weight, stored (encrypted) as a hint for the UI.
    #[serde(default)]
    pub weight: Option<f32>,
    pub password: String,
}

/// `POST /vault/adapters/list` body.
#[derive(Deserialize, ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct ListAdaptersRequest {
    /// Model id (kebab wire string) these adapters apply to. Image or video;
    /// validated against [`thinfer_app::model::is_adapter_model`].
    pub model: String,
    pub password: String,
}

/// `POST /vault/adapters/remove` body.
#[derive(Deserialize, ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct RemoveAdapterRequest {
    /// Model id (kebab wire string) these adapters apply to. Image or video;
    /// validated against [`thinfer_app::model::is_adapter_model`].
    pub model: String,
    pub id: String,
    pub password: String,
}

/// The `list`/`add` response: this model's adapters (decrypted names/sizes).
#[derive(Serialize, ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct AdaptersResponse {
    pub adapters: Vec<VaultEntryInfo>,
}

fn map_err(e: VaultError) -> ApiError {
    let status = match e {
        VaultError::Auth => StatusCode::UNAUTHORIZED,
        VaultError::NotFound => StatusCode::NOT_FOUND,
        VaultError::Format(_) => StatusCode::BAD_REQUEST,
        VaultError::Download(_) => StatusCode::BAD_GATEWAY,
        VaultError::Io(_) | VaultError::Serde(_) => StatusCode::INTERNAL_SERVER_ERROR,
    };
    ApiError::new(status, e.to_string())
}

/// List a model's adapters. Wrong password => 401 (opaque). Uninitialized vault
/// => empty list.
#[utoipa::path(
    post, path = "/vault/adapters/list",
    request_body = ListAdaptersRequest,
    responses(
        (status = 200, body = AdaptersResponse),
        (status = 401, description = "Invalid password"),
    )
)]
pub async fn list_adapters(
    State(state): State<AppState>,
    Json(req): Json<ListAdaptersRequest>,
) -> Result<Json<AdaptersResponse>, ApiError> {
    let vault = state.vault.clone();
    let model = req.model.to_string();
    let adapters = tokio::task::spawn_blocking(move || vault.list(&req.password, &model))
        .await
        .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .map_err(map_err)?;
    Ok(Json(AdaptersResponse { adapters }))
}

/// Download an adapter from a URL, validate it is safetensors (by content, not
/// filename), encrypt it into the vault under `model`, and return the new entry.
#[utoipa::path(
    post, path = "/vault/adapters/add",
    request_body = AddAdapterRequest,
    responses(
        (status = 200, body = VaultEntryInfo),
        (status = 400, description = "Not a safetensors adapter"),
        (status = 401, description = "Invalid password"),
        (status = 502, description = "Download failed"),
    )
)]
pub async fn add_adapter(
    State(state): State<AppState>,
    Json(req): Json<AddAdapterRequest>,
) -> Result<Json<VaultEntryInfo>, ApiError> {
    if !thinfer_app::model::is_adapter_model(&req.model) {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            format!("{} does not support adapters", req.model),
        ));
    }
    // Download (async), then validate + encrypt (blocking) off the runtime.
    let (filename, bytes) = thinfer_app::vault::download(&req.url, req.token.as_deref())
        .await
        .map_err(map_err)?;
    thinfer_app::vault::ensure_safetensors(&bytes).map_err(map_err)?;

    let vault = state.vault.clone();
    let model = req.model.to_string();
    let name = req.name.unwrap_or(filename);
    let password = req.password;
    let mut extra = std::collections::BTreeMap::new();
    if let Some(w) = req.weight {
        extra.insert("weight".to_string(), w.to_string());
    }
    let info =
        tokio::task::spawn_blocking(move || vault.add(&password, &model, &name, &bytes, extra))
            .await
            .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
            .map_err(map_err)?;
    Ok(Json(info))
}

/// Remove one adapter (deletes its blob + index entry). Needs the password.
#[utoipa::path(
    post, path = "/vault/adapters/remove",
    request_body = RemoveAdapterRequest,
    responses(
        (status = 200, description = "Removed"),
        (status = 401, description = "Invalid password"),
        (status = 404, description = "No such adapter"),
    )
)]
pub async fn remove_adapter(
    State(state): State<AppState>,
    Json(req): Json<RemoveAdapterRequest>,
) -> Result<StatusCode, ApiError> {
    let vault = state.vault.clone();
    let model = req.model.to_string();
    tokio::task::spawn_blocking(move || vault.remove(&req.password, &model, &req.id))
        .await
        .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .map_err(map_err)?;
    Ok(StatusCode::OK)
}
