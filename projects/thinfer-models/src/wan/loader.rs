//! Residency-aware Wan2.2 DiT loader. Resolves the diffusers
//! `WanTransformer3DModel` state-dict names (see `transformer_wan.py`), looks
//! each tensor up in the source's
//! `WeightCatalog`, builds a `WeightMeta` (decode + transpose policy) and
//! registers it with `WeightResidency`. Returns `LoadedWanDitHandles`.
//!
//! No bytes flow here; decode + transpose + upload happen lazily on
//! `WeightResidency::acquire`. Mirrors `z_image/loader.rs` (same transcode +
//! transpose discipline) with the Wan-specific name layout.
//!
//! `patch_embedding` is a `Conv3d` with kernel == stride == patch, so it folds
//! to the front-door linear: the conv weight `[inner, in_ch, p_t, p_h, p_w]`
//! (already `(ic, kt, kh, kw)` row-major, see `wan/patchify.rs`) is registered
//! with its shape collapsed to 2-D `[inner, patch_in]` and `Linear2D`, which
//! transposes it to the `[patch_in, inner]` the matmul B-side wants.

use thinfer_core::residency::{
    ResidencyError, TransposePolicy, WeightHandle, WeightMeta, WeightResidency,
};
use thinfer_core::tensor::{Shape, StorageEncoding};
use thinfer_core::weight::{WeightId, WeightSource};

use crate::common::embedders::LinearBiasHandles;
use crate::wan::condition_embedder::ConditionEmbedderHandles;
use crate::wan::dit::LoadedWanDitHandles;
use crate::wan::dit_block::{WanAttnHandles, WanDitBlockHandles, WanDitConfig};

#[derive(Debug)]
pub enum LoadError {
    UnknownWeight(WeightId),
    Undecodable {
        id: WeightId,
        encoding: Option<StorageEncoding>,
        label: String,
    },
}

// ---------------------------------------------------------------------------
// Weight names (diffusers WanTransformer3DModel state-dict keys)
// ---------------------------------------------------------------------------

/// Module-level (non-block) weight names.
pub struct WanDitModelWeights {
    pub patch_weight: WeightId,
    pub patch_bias: WeightId,
    pub scale_shift_table: WeightId,
    pub proj_out_weight: WeightId,
    pub proj_out_bias: WeightId,
}

impl Default for WanDitModelWeights {
    fn default() -> Self {
        Self::new()
    }
}

impl WanDitModelWeights {
    pub fn new() -> Self {
        let id = |s: &str| WeightId(s.to_string());
        Self {
            patch_weight: id("patch_embedding.weight"),
            patch_bias: id("patch_embedding.bias"),
            scale_shift_table: id("scale_shift_table"),
            proj_out_weight: id("proj_out.weight"),
            proj_out_bias: id("proj_out.bias"),
        }
    }
}

/// One linear with bias.
struct LinearNames {
    weight: WeightId,
    bias: WeightId,
}

/// Condition embedder names. `time_embedder`/`time_proj`/`text_embedder` live
/// under `condition_embedder.*` (diffusers `WanTimeTextImageEmbedding`). The
/// T2V base has no `image_embedder` and no `fps_*` (that was SkyReels-V2-DF's
/// `inject_sample_info`).
struct ConditionEmbedderNames {
    time_linear_1: LinearNames,
    time_linear_2: LinearNames,
    time_proj: LinearNames,
    text_linear_1: LinearNames,
    text_linear_2: LinearNames,
}

impl ConditionEmbedderNames {
    fn new() -> Self {
        let lin = |w: &str, b: &str| LinearNames {
            weight: WeightId(w.to_string()),
            bias: WeightId(b.to_string()),
        };
        Self {
            time_linear_1: lin(
                "condition_embedder.time_embedder.linear_1.weight",
                "condition_embedder.time_embedder.linear_1.bias",
            ),
            time_linear_2: lin(
                "condition_embedder.time_embedder.linear_2.weight",
                "condition_embedder.time_embedder.linear_2.bias",
            ),
            time_proj: lin(
                "condition_embedder.time_proj.weight",
                "condition_embedder.time_proj.bias",
            ),
            text_linear_1: lin(
                "condition_embedder.text_embedder.linear_1.weight",
                "condition_embedder.text_embedder.linear_1.bias",
            ),
            text_linear_2: lin(
                "condition_embedder.text_embedder.linear_2.weight",
                "condition_embedder.text_embedder.linear_2.bias",
            ),
        }
    }
}

/// One self/cross attention stage's weight names.
struct AttnNames {
    q: LinearNames,
    k: LinearNames,
    v: LinearNames,
    o: LinearNames,
    norm_q: WeightId,
    norm_k: WeightId,
}

impl AttnNames {
    fn new(prefix: &str) -> Self {
        let lin = |s: &str| LinearNames {
            weight: WeightId(format!("{prefix}.{s}.weight")),
            bias: WeightId(format!("{prefix}.{s}.bias")),
        };
        Self {
            q: lin("to_q"),
            k: lin("to_k"),
            v: lin("to_v"),
            o: lin("to_out.0"),
            norm_q: WeightId(format!("{prefix}.norm_q.weight")),
            norm_k: WeightId(format!("{prefix}.norm_k.weight")),
        }
    }
}

/// Per-block diffusers weight names (`blocks.{i}.*`).
pub struct WanDitBlockWeights {
    self_attn: AttnNames,
    cross_attn: AttnNames,
    norm2: LinearNames,
    ffn_up: LinearNames,
    ffn_down: LinearNames,
    scale_shift_table: WeightId,
}

impl WanDitBlockWeights {
    pub fn new(idx: usize) -> Self {
        let p = format!("blocks.{idx}");
        Self {
            self_attn: AttnNames::new(&format!("{p}.attn1")),
            cross_attn: AttnNames::new(&format!("{p}.attn2")),
            norm2: LinearNames {
                weight: WeightId(format!("{p}.norm2.weight")),
                bias: WeightId(format!("{p}.norm2.bias")),
            },
            ffn_up: LinearNames {
                weight: WeightId(format!("{p}.ffn.net.0.proj.weight")),
                bias: WeightId(format!("{p}.ffn.net.0.proj.bias")),
            },
            ffn_down: LinearNames {
                weight: WeightId(format!("{p}.ffn.net.2.weight")),
                bias: WeightId(format!("{p}.ffn.net.2.bias")),
            },
            scale_shift_table: WeightId(format!("{p}.scale_shift_table")),
        }
    }
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Opt-in load-time i8 (Q8_0) transcode targets for the DP4A-safe matmul sites:
/// the SELF-attention qkv projections and ffn_up. Their A-side is a
/// norm-conditioned activation (no massive-activation outliers), so per-32 i8
/// act-quant is lossless enough to hold parity, and the i8xi8 `matmul_i8`
/// (dot4I8Packed) path is a pure compute win. The other matmul sites keep their
/// file encoding: cross-attn qkv's K/V project from un-normed umT5 text (i8 acts
/// overflow f16); o-proj / ffn_down A-sides (attention output, gelu) carry ~16k
/// outliers that i8 act-quant would crush.
#[derive(Clone, Copy, Default)]
pub struct WanI8Sites {
    /// Self-attention qkv only (cross-attention qkv always stays dense).
    pub qkv_self: Option<thinfer_core::quant::QuantKind>,
    pub ffn_up: Option<thinfer_core::quant::QuantKind>,
}

/// Register every Wan DiT weight with residency. `transcode`: optional
/// load-time requantize target for the matmul weights (block q/k/v/o + ffn);
/// embedders, patch, proj_out, and all norms/biases stay dense. `i8`: opt-in
/// per-site i8 transcode for the DP4A-safe sites (qkv + ffn_up), overriding
/// `transcode` for those weights only. Mirrors `z_image::loader`.
pub fn register_wan_dit_handles<S: WeightSource>(
    residency: &WeightResidency<S>,
    cfg: &WanDitConfig,
    transcode: Option<thinfer_core::quant::QuantKind>,
    i8: WanI8Sites,
) -> Result<LoadedWanDitHandles, LoadError> {
    let mw = WanDitModelWeights::new();
    let blocks = (0..cfg.num_layers)
        .map(|i| register_block(residency, &WanDitBlockWeights::new(i), transcode, i8))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(LoadedWanDitHandles {
        patch: register_conv_as_linear_bias(residency, &mw.patch_weight, &mw.patch_bias)?,
        condition: register_condition_embedder(residency, &ConditionEmbedderNames::new())?,
        blocks,
        scale_shift_table: register_passthrough(residency, &mw.scale_shift_table)?,
        proj_out: register_linear_bias(residency, &mw.proj_out_weight, &mw.proj_out_bias, None)?,
    })
}

fn register_block<S: WeightSource>(
    residency: &WeightResidency<S>,
    w: &WanDitBlockWeights,
    transcode: Option<thinfer_core::quant::QuantKind>,
    i8: WanI8Sites,
) -> Result<WanDitBlockHandles, LoadError> {
    Ok(WanDitBlockHandles {
        // Self-attn qkv takes the opt-in i8 transcode; cross-attn qkv never does
        // (its K/V project from un-normed text -> i8 acts overflow f16).
        self_attn: register_attn(residency, &w.self_attn, transcode, i8.qkv_self)?,
        cross_attn: register_attn(residency, &w.cross_attn, transcode, None)?,
        norm2_w: register_passthrough(residency, &w.norm2.weight)?,
        norm2_b: register_passthrough(residency, &w.norm2.bias)?,
        ffn_up_w: register_linear(residency, &w.ffn_up.weight, i8.ffn_up.or(transcode))?,
        ffn_up_b: register_passthrough(residency, &w.ffn_up.bias)?,
        ffn_down_w: register_linear(residency, &w.ffn_down.weight, transcode)?,
        ffn_down_b: register_passthrough(residency, &w.ffn_down.bias)?,
        scale_shift_table: register_passthrough(residency, &w.scale_shift_table)?,
    })
}

fn register_attn<S: WeightSource>(
    residency: &WeightResidency<S>,
    w: &AttnNames,
    transcode: Option<thinfer_core::quant::QuantKind>,
    qkv_i8: Option<thinfer_core::quant::QuantKind>,
) -> Result<WanAttnHandles, LoadError> {
    // q/k/v take the opt-in i8 transcode (normed A-side, DP4A-safe); o-proj keeps
    // the base encoding (its A-side is the attention output -> outlier-prone).
    Ok(WanAttnHandles {
        q_w: register_linear(residency, &w.q.weight, qkv_i8.or(transcode))?,
        q_b: register_passthrough(residency, &w.q.bias)?,
        k_w: register_linear(residency, &w.k.weight, qkv_i8.or(transcode))?,
        k_b: register_passthrough(residency, &w.k.bias)?,
        v_w: register_linear(residency, &w.v.weight, qkv_i8.or(transcode))?,
        v_b: register_passthrough(residency, &w.v.bias)?,
        o_w: register_linear(residency, &w.o.weight, transcode)?,
        o_b: register_passthrough(residency, &w.o.bias)?,
        norm_q: register_passthrough(residency, &w.norm_q)?,
        norm_k: register_passthrough(residency, &w.norm_k)?,
    })
}

fn register_condition_embedder<S: WeightSource>(
    residency: &WeightResidency<S>,
    w: &ConditionEmbedderNames,
) -> Result<ConditionEmbedderHandles, LoadError> {
    let lb = |l: &LinearNames| register_linear_bias(residency, &l.weight, &l.bias, None);
    Ok(ConditionEmbedderHandles {
        time_linear_1: lb(&w.time_linear_1)?,
        time_linear_2: lb(&w.time_linear_2)?,
        time_proj: lb(&w.time_proj)?,
        text_linear_1: lb(&w.text_linear_1)?,
        text_linear_2: lb(&w.text_linear_2)?,
    })
}

fn register_linear_bias<S: WeightSource>(
    residency: &WeightResidency<S>,
    weight: &WeightId,
    bias: &WeightId,
    transcode: Option<thinfer_core::quant::QuantKind>,
) -> Result<LinearBiasHandles, LoadError> {
    Ok(LinearBiasHandles {
        weight: register_linear(residency, weight, transcode)?,
        bias: register_passthrough(residency, bias)?,
    })
}

/// Register the patch `Conv3d` weight (5-D `[inner, in, p_t, p_h, p_w]`) as a
/// 2-D linear by collapsing the trailing dims to `[inner, patch_in]`. The bytes
/// are row-major `(inner, ic, kt, kh, kw)`, so the collapse is a pure shape
/// reinterpret; `Linear2D` then transposes to the `[patch_in, inner]` B-side.
fn register_conv_as_linear_bias<S: WeightSource>(
    residency: &WeightResidency<S>,
    weight: &WeightId,
    bias: &WeightId,
) -> Result<LinearBiasHandles, LoadError> {
    let entry = catalog_entry(residency, weight)?;
    let encoding = entry_encoding(weight, &entry)?;
    let dims = &entry.shape.0;
    assert!(
        dims.len() >= 2,
        "patch conv weight must be >= 2-D ({weight:?})"
    );
    let n = dims[0];
    let k: usize = dims[1..].iter().product();
    let (encoding, transpose, transcode) = linear_layout(weight, encoding, &entry, None)?;
    let meta = WeightMeta {
        id: weight.clone(),
        shape: Shape(vec![n, k]),
        encoding,
        on_disk_bytes: entry.size,
        transpose,
        transcode,
    };
    Ok(LinearBiasHandles {
        weight: residency.register(meta),
        bias: register_passthrough(residency, bias)?,
    })
}

/// Linear weight: bf16/f32 `[N, K]` -> `Linear2D` (transposed to `[K, N]`), or
/// requantized to the GGUF block layout when `transcode` is set, or file-native
/// quant (already `[N, K]` N-major, no transpose). Mirrors
/// `common::loader::register_linear_transcode`.
fn register_linear<S: WeightSource>(
    residency: &WeightResidency<S>,
    id: &WeightId,
    transcode: Option<thinfer_core::quant::QuantKind>,
) -> Result<WeightHandle, LoadError> {
    let entry = catalog_entry(residency, id)?;
    let encoding = entry_encoding(id, &entry)?;
    let (encoding, transpose, transcode) = linear_layout(id, encoding, &entry, transcode)?;
    Ok(residency.register(WeightMeta {
        id: id.clone(),
        shape: entry.shape.clone(),
        encoding,
        on_disk_bytes: entry.size,
        transpose,
        transcode,
    }))
}

/// Resolve `(encoding, transpose, transcode)` for a linear weight. Factored out
/// so the patch-conv path (custom shape) shares the policy.
fn linear_layout(
    id: &WeightId,
    encoding: StorageEncoding,
    entry: &thinfer_core::weight::WeightEntry,
    transcode: Option<thinfer_core::quant::QuantKind>,
) -> Result<
    (
        StorageEncoding,
        TransposePolicy,
        Option<thinfer_core::quant::QuantKind>,
    ),
    LoadError,
> {
    Ok(match encoding {
        StorageEncoding::Bf16 if transcode.is_some() => {
            assert_eq!(entry.shape.0.len(), 2, "transcode target must be 2-D");
            assert_eq!(
                entry.shape.0[1] % 32,
                0,
                "transcode requires K % 32 == 0 ({id:?})"
            );
            (encoding, TransposePolicy::None, transcode)
        }
        StorageEncoding::Bf16 | StorageEncoding::F32 => (encoding, TransposePolicy::Linear2D, None),
        StorageEncoding::Quant(_) => (encoding, TransposePolicy::None, None),
        _ => {
            return Err(LoadError::Undecodable {
                id: id.clone(),
                encoding: Some(encoding),
                label: entry.encoding_label.clone(),
            });
        }
    })
}

fn register_passthrough<S: WeightSource>(
    residency: &WeightResidency<S>,
    id: &WeightId,
) -> Result<WeightHandle, LoadError> {
    let entry = catalog_entry(residency, id)?;
    let encoding = entry_encoding(id, &entry)?;
    if thinfer_core::weight::Decoder::new(encoding).is_err() {
        return Err(LoadError::Undecodable {
            id: id.clone(),
            encoding: Some(encoding),
            label: entry.encoding_label.clone(),
        });
    }
    Ok(residency.register(WeightMeta {
        id: id.clone(),
        shape: entry.shape.clone(),
        encoding,
        on_disk_bytes: entry.size,
        transpose: TransposePolicy::None,
        transcode: None,
    }))
}

fn catalog_entry<S: WeightSource>(
    residency: &WeightResidency<S>,
    id: &WeightId,
) -> Result<thinfer_core::weight::WeightEntry, LoadError> {
    residency
        .source()
        .catalog()
        .get(id)
        .cloned()
        .ok_or_else(|| LoadError::UnknownWeight(id.clone()))
}

fn entry_encoding(
    id: &WeightId,
    entry: &thinfer_core::weight::WeightEntry,
) -> Result<StorageEncoding, LoadError> {
    entry.encoding.ok_or_else(|| LoadError::Undecodable {
        id: id.clone(),
        encoding: None,
        label: entry.encoding_label.clone(),
    })
}

impl<SE: core::fmt::Debug, BE: core::fmt::Debug> From<ResidencyError<SE, BE>> for LoadError {
    fn from(_: ResidencyError<SE, BE>) -> Self {
        unreachable!("register doesn't fail")
    }
}
