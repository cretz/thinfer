//! wasm-bindgen exports. Mirror of `wasm-pkg.d.ts`.
//!
//! Async exports that need `self` return `js_sys::Promise` via
//! `future_to_promise` with the state behind an `Rc` (wasm-bindgen async
//! methods can't borrow `self` across the await). The TS wrapper presents
//! them as ordinary async methods.

use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

use thinfer_core::backend::{PowerPreference, WgpuBackend, WgpuConfig};
use thinfer_core::policy::ResidencyBudget;
use thinfer_core::residency::WeightResidency;
use thinfer_core::tokenizer::{Tokenizer, TokenizerError};
use thinfer_models::z_image::manifest;
use thinfer_models::z_image::pipeline::{GenerationParams, ProgressEvent, ZImageModel};
use thinfer_models::z_image::source::{GgufOpeners, ZImageSource};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::future_to_promise;

use crate::weight_file::{JsWeightFile, WebFileOpener};

/// JSON `[{ role, repo, path, revision? }]` for a model variant id. Strings
/// are static manifest data (repo names, file paths): no escaping beyond
/// `{:?}` needed. Single source of truth with the CLI via
/// `thinfer_models::...::manifest::VARIANTS`.
#[wasm_bindgen(js_name = modelFilesJson)]
pub fn model_files_json(model_id: &str) -> Result<String, JsError> {
    let variant = manifest::variant(model_id)
        .ok_or_else(|| JsError::new(&format!("unknown model id: {model_id}")))?;
    let mut out = String::from("[");
    for (i, (role, f)) in variant.files().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&format!(
            "{{\"role\":{role:?},\"repo\":{:?},\"path\":{:?}",
            f.repo, f.path
        ));
        if let Some(rev) = f.revision {
            out.push_str(&format!(",\"revision\":{rev:?}"));
        }
        out.push('}');
    }
    out.push(']');
    Ok(out)
}

struct EngineInner {
    backend: Arc<WgpuBackend>,
    budget: ResidencyBudget,
}

#[wasm_bindgen]
pub struct WasmEngine {
    inner: Rc<EngineInner>,
}

#[wasm_bindgen]
impl WasmEngine {
    /// Adapter + device init. `power_preference` is the TS-side string
    /// (`"high-performance"` | `"low-power"`); default high-performance,
    /// matching the CLI (unset Vulkan hints read as background priority on
    /// some drivers).
    pub async fn create(
        power_preference: Option<String>,
        ram_budget_bytes: u64,
        vram_budget_bytes: u64,
    ) -> Result<WasmEngine, JsError> {
        let power_preference = match power_preference.as_deref() {
            None | Some("high-performance") => PowerPreference::HighPerformance,
            Some("low-power") => PowerPreference::LowPower,
            Some(other) => {
                return Err(JsError::new(&format!("unknown powerPreference: {other}")));
            }
        };
        let cfg = WgpuConfig {
            power_preference,
            timestamps: false,
        };
        let backend = WgpuBackend::new_with_config(cfg)
            .await
            .map_err(|e| JsError::new(&format!("wgpu init: {e:?}")))?;
        Ok(WasmEngine {
            inner: Rc::new(EngineInner {
                backend: Arc::new(backend),
                budget: ResidencyBudget {
                    ram_bytes: ram_budget_bytes,
                    vram_bytes: vram_budget_bytes,
                },
            }),
        })
    }

    /// `roles[i]` names the role of `files[i]`; each file satisfies the
    /// TS `WeightFile` duck type. Resolves to a `WasmModel`.
    #[wasm_bindgen(js_name = loadModel)]
    pub fn load_model(
        &self,
        model_id: String,
        roles: Vec<String>,
        files: Vec<JsValue>,
    ) -> js_sys::Promise {
        let engine = Rc::clone(&self.inner);
        future_to_promise(async move {
            let model = load_model_impl(&engine, &model_id, roles, files).await?;
            Ok(WasmModel {
                inner: Rc::new(model),
            }
            .into())
        })
    }
}

type WebModel = ZImageModel<ZImageSource<WebFileOpener>, WebTokenizer>;

fn js_err(context: &str, detail: impl core::fmt::Debug) -> JsValue {
    js_sys::Error::new(&format!("{context}: {detail:?}")).into()
}

async fn load_model_impl(
    engine: &EngineInner,
    model_id: &str,
    roles: Vec<String>,
    files: Vec<JsValue>,
) -> Result<WebModel, JsValue> {
    let variant = manifest::variant(model_id)
        .ok_or_else(|| js_err("loadModel", format_args!("unknown model id: {model_id}")))?;
    if roles.len() != files.len() {
        return Err(js_err(
            "loadModel",
            format_args!("{} roles but {} files", roles.len(), files.len()),
        ));
    }
    let by_role: HashMap<String, JsWeightFile> = roles
        .into_iter()
        .zip(files.into_iter().map(JsWeightFile::unchecked_from_js))
        .collect();
    let file_for = |role: &str| -> Result<JsWeightFile, JsValue> {
        by_role
            .get(role)
            .cloned()
            .ok_or_else(|| js_err("loadModel", format_args!("missing file for role {role}")))
    };

    let mut weight_openers = Vec::with_capacity(variant.weight_roles.len());
    for role in variant.weight_roles {
        weight_openers.push(WebFileOpener::new(file_for(role)?));
    }
    let gguf_openers = match (variant.dit_gguf_role, variant.te_gguf_role) {
        (Some(dit), Some(te)) => Some(GgufOpeners {
            dit: WebFileOpener::new(file_for(dit)?),
            te: WebFileOpener::new(file_for(te)?),
        }),
        (None, None) => None,
        _ => {
            return Err(js_err(
                "loadModel",
                format_args!("variant must set both gguf roles or neither"),
            ));
        }
    };
    let source = ZImageSource::open(weight_openers, gguf_openers)
        .await
        .map_err(|e| js_err("parse weight files", e))?;

    let tokenizer = WebTokenizer::load(&file_for(manifest::role::TOKENIZER_JSON)?).await?;

    let residency = WeightResidency::new(source, engine.budget);
    ZImageModel::load(Arc::clone(&engine.backend), residency, tokenizer)
        .await
        .map_err(|e| js_err("model load", e))
}

#[wasm_bindgen]
pub struct WasmModel {
    inner: Rc<WebModel>,
}

#[wasm_bindgen]
impl WasmModel {
    /// Resolves to encoded PNG bytes. `on_progress` receives
    /// `{ type: "textEncode" | "step" | "vaeDecode", i?, n? }`.
    pub fn generate(
        &self,
        prompt: String,
        width: u32,
        height: u32,
        steps: u32,
        seed: u64,
        on_progress: Option<js_sys::Function>,
    ) -> js_sys::Promise {
        let model = Rc::clone(&self.inner);
        future_to_promise(async move {
            let params = GenerationParams {
                prompt,
                height,
                width,
                steps,
                seed,
            };
            let cb = on_progress.map(|f| {
                move |ev: ProgressEvent| {
                    // A throwing callback must not abort generation; ignore.
                    let _ = f.call1(&JsValue::NULL, &progress_to_js(ev));
                }
            });
            let progress = cb.as_ref().map(|c| c as &dyn Fn(ProgressEvent));
            let png = model
                .generate(&params, progress)
                .await
                .map_err(|e| js_err("generate", e))?;
            Ok(js_sys::Uint8Array::from(png.as_slice()).into())
        })
    }
}

fn progress_to_js(ev: ProgressEvent) -> JsValue {
    let obj = js_sys::Object::new();
    let set = |k: &str, v: JsValue| {
        // Reflect::set on a fresh plain object cannot fail.
        let _ = js_sys::Reflect::set(&obj, &JsValue::from_str(k), &v);
    };
    match ev {
        ProgressEvent::TextEncode => set("type", "textEncode".into()),
        ProgressEvent::Step { i, n } => {
            set("type", "step".into());
            set("i", i.into());
            set("n", n.into());
        }
        ProgressEvent::VaeDecode => set("type", "vaeDecode".into()),
    }
    obj.into()
}

/// HF `tokenizers` (wasm build) behind the engine's `Tokenizer` trait.
/// Mirrors `thinfer_native::tokenizer::HfTokenizer`.
struct WebTokenizer {
    inner: tokenizers::Tokenizer,
}

impl WebTokenizer {
    /// Pull the whole `tokenizer.json` into wasm memory and parse. This is
    /// MBs of JSON config, not weight bytes; the no-weight-bytes rule does
    /// not apply and the parsed structure replaces the raw bytes anyway.
    async fn load(file: &JsWeightFile) -> Result<Self, JsValue> {
        let len = file.size_bytes();
        let val = wasm_bindgen_futures::JsFuture::from(file.read_at(0.0, len))
            .await
            .map_err(|e| js_err("read tokenizer.json", e))?;
        let arr: js_sys::Uint8Array = val
            .dyn_into()
            .map_err(|v| js_err("tokenizer readAt resolved to non-Uint8Array", v))?;
        let inner = tokenizers::Tokenizer::from_bytes(arr.to_vec())
            .map_err(|e| js_err("parse tokenizer.json", e.to_string()))?;
        Ok(Self { inner })
    }
}

impl Tokenizer for WebTokenizer {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, TokenizerError> {
        let enc = self
            .inner
            .encode(text, add_special_tokens)
            .map_err(|e| TokenizerError::Encode(e.to_string()))?;
        Ok(enc.get_ids().to_vec())
    }
}
