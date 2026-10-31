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
use crate::wan::dit_block::{
    QkEmbedHandles, RehyatHandles, WanAttnHandles, WanDitBlockHandles, WanDitConfig,
};

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
        Self::new("")
    }
}

impl WanDitModelWeights {
    /// `pre` namespaces every id (a per-expert prefix like `"high."` for the
    /// Wan2.2 MoE, or `""` for a single-expert model).
    pub fn new(pre: &str) -> Self {
        let id = |s: &str| WeightId(format!("{pre}{s}"));
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
    /// AnyFlow second TimestepEmbedding (registered only when the config's
    /// `delta_embedder` is set; plain Wan checkpoints don't carry it).
    delta_linear_1: LinearNames,
    delta_linear_2: LinearNames,
}

impl ConditionEmbedderNames {
    fn new(pre: &str) -> Self {
        let lin = |w: &str, b: &str| LinearNames {
            weight: WeightId(format!("{pre}{w}")),
            bias: WeightId(format!("{pre}{b}")),
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
            delta_linear_1: lin(
                "condition_embedder.delta_embedder.linear_1.weight",
                "condition_embedder.delta_embedder.linear_1.bias",
            ),
            delta_linear_2: lin(
                "condition_embedder.delta_embedder.linear_2.weight",
                "condition_embedder.delta_embedder.linear_2.bias",
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

/// mobilewan ReHyAt feature-embed names for one stage (q_embed / k_embed): the
/// two grouped 1x1 convs, each weight + bias, under `<prefix>.embed.{0,2}`.
struct QkEmbedNames {
    conv1_w: WeightId,
    conv1_b: WeightId,
    conv2_w: WeightId,
    conv2_b: WeightId,
}

impl QkEmbedNames {
    fn new(prefix: &str) -> Self {
        Self {
            conv1_w: WeightId(format!("{prefix}.embed.0.weight")),
            conv1_b: WeightId(format!("{prefix}.embed.0.bias")),
            conv2_w: WeightId(format!("{prefix}.embed.2.weight")),
            conv2_b: WeightId(format!("{prefix}.embed.2.bias")),
        }
    }
}

/// mobilewan ReHyAt embeds for one block: `attn1.{q,k}_embed.*` (siblings of
/// `attn1.base_attention`, not nested under it).
struct RehyatNames {
    q_embed: QkEmbedNames,
    k_embed: QkEmbedNames,
}

/// Per-block diffusers weight names (`blocks.{i}.*`).
pub struct WanDitBlockWeights {
    self_attn: AttnNames,
    cross_attn: AttnNames,
    /// mobilewan-only ReHyAt embeds (`Some` iff `rehyat`).
    rehyat: Option<RehyatNames>,
    norm2: LinearNames,
    ffn_up: LinearNames,
    ffn_down: LinearNames,
    scale_shift_table: WeightId,
}

impl WanDitBlockWeights {
    /// `rehyat` (mobilewan) nests self-attention under `attn1.base_attention.*`
    /// (the ReHyAt wrapper); every other Wan model has it flat under `attn1.*`.
    /// Cross-attention (`attn2`) is unchanged.
    pub fn new(pre: &str, idx: usize, rehyat: bool) -> Self {
        let p = format!("{pre}blocks.{idx}");
        let attn1 = if rehyat {
            format!("{p}.attn1.base_attention")
        } else {
            format!("{p}.attn1")
        };
        Self {
            self_attn: AttnNames::new(&attn1),
            cross_attn: AttnNames::new(&format!("{p}.attn2")),
            rehyat: rehyat.then(|| RehyatNames {
                q_embed: QkEmbedNames::new(&format!("{p}.attn1.q_embed")),
                k_embed: QkEmbedNames::new(&format!("{p}.attn1.k_embed")),
            }),
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
/// `prefix` namespaces every weight id (e.g. `"high."`/`"low."` for the Wan2.2
/// MoE so both experts' handle sets coexist in one residency; `""` otherwise).
/// The module-level matmul weights (patch + condition embedder + proj_out) always
/// stay dense bf16 (F16/F32 narrow to bf16 at upload); they run on the dedicated
/// bf16 `matmul_module` pipeline, decoupled from the (possibly quant) block sites.
pub fn register_wan_dit_handles<S: WeightSource>(
    residency: &WeightResidency<S>,
    cfg: &WanDitConfig,
    prefix: &str,
    transcode: Option<thinfer_core::quant::QuantKind>,
    i8: WanI8Sites,
) -> Result<LoadedWanDitHandles, LoadError> {
    let mw = WanDitModelWeights::new(prefix);
    let blocks = (0..cfg.num_layers)
        .map(|i| {
            register_block(
                residency,
                &WanDitBlockWeights::new(prefix, i, cfg.rehyat),
                transcode,
                i8,
                cfg,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(LoadedWanDitHandles {
        patch: register_conv_as_linear_bias(residency, &mw.patch_weight, &mw.patch_bias)?,
        condition: register_condition_embedder(
            residency,
            &ConditionEmbedderNames::new(prefix),
            cfg.delta_embedder,
        )?,
        blocks,
        scale_shift_table: register_passthrough(residency, &mw.scale_shift_table)?,
        proj_out: register_linear_bias(residency, &mw.proj_out_weight, &mw.proj_out_bias)?,
        // DreamID-V ref_conv (source-face patchify). Named literally (no diffusers
        // equivalent -> passes through the rename layer unchanged). Same conv-as-
        // linear fold as patch_embedding.
        ref_conv: cfg
            .ref_conv
            .then(|| {
                register_conv_as_linear_bias(
                    residency,
                    &WeightId(format!("{prefix}ref_conv.weight")),
                    &WeightId(format!("{prefix}ref_conv.bias")),
                )
            })
            .transpose()?,
    })
}

fn register_block<S: WeightSource>(
    residency: &WeightResidency<S>,
    w: &WanDitBlockWeights,
    transcode: Option<thinfer_core::quant::QuantKind>,
    i8: WanI8Sites,
    cfg: &WanDitConfig,
) -> Result<WanDitBlockHandles, LoadError> {
    // mobilewan: this block's PRUNED self-attn head count, from the loaded
    // `attn1.base_attention.to_q` row count (`h_i * head_dim`). The self-attn
    // matmuls run the DENSE per-block-N path, so they never take the i8
    // transcode (which prepares full-N weights); every other model keeps the
    // opt-in i8 on qkv_self.
    let (n_heads, qkv_self_i8) = if cfg.rehyat {
        let entry = catalog_entry(residency, &w.self_attn.q.weight)?;
        let rows = entry.shape.0[0];
        (rows / crate::wan::dit_block::config::HEAD_DIM, None)
    } else {
        (cfg.num_heads, i8.qkv_self)
    };
    Ok(WanDitBlockHandles {
        // Self-attn qkv takes the opt-in i8 transcode; cross-attn qkv never does
        // (its K/V project from un-normed text -> i8 acts overflow f16).
        self_attn: register_attn(residency, &w.self_attn, transcode, qkv_self_i8)?,
        cross_attn: register_attn(residency, &w.cross_attn, transcode, None)?,
        rehyat: w
            .rehyat
            .as_ref()
            .map(|r| register_rehyat(residency, r))
            .transpose()?,
        norm2_w: register_passthrough(residency, &w.norm2.weight)?,
        norm2_b: register_passthrough(residency, &w.norm2.bias)?,
        ffn_up_w: register_linear(residency, &w.ffn_up.weight, i8.ffn_up.or(transcode))?,
        ffn_up_b: register_passthrough(residency, &w.ffn_up.bias)?,
        ffn_down_w: register_linear(residency, &w.ffn_down.weight, transcode)?,
        ffn_down_b: register_passthrough(residency, &w.ffn_down.bias)?,
        scale_shift_table: register_passthrough(residency, &w.scale_shift_table)?,
        n_heads,
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

/// Register one mobilewan ReHyAt feature embed (q_embed or k_embed). The two
/// grouped 1x1 convs are read RAW by the driver's per-head grouped matmul (weight
/// `[out_ch, in_ch/groups, 1, 1]`, interpreted as `[out_ch, in_ch/groups]`), so
/// they pass through untransposed -- `register_linear`'s `[N,K]->[K,N]` transpose
/// would scramble the per-head block layout.
fn register_qk_embed<S: WeightSource>(
    residency: &WeightResidency<S>,
    w: &QkEmbedNames,
) -> Result<QkEmbedHandles, LoadError> {
    Ok(QkEmbedHandles {
        conv1_w: register_passthrough(residency, &w.conv1_w)?,
        conv1_b: register_passthrough(residency, &w.conv1_b)?,
        conv2_w: register_passthrough(residency, &w.conv2_w)?,
        conv2_b: register_passthrough(residency, &w.conv2_b)?,
    })
}

fn register_rehyat<S: WeightSource>(
    residency: &WeightResidency<S>,
    w: &RehyatNames,
) -> Result<RehyatHandles, LoadError> {
    Ok(RehyatHandles {
        q_embed: register_qk_embed(residency, &w.q_embed)?,
        k_embed: register_qk_embed(residency, &w.k_embed)?,
    })
}

fn register_condition_embedder<S: WeightSource>(
    residency: &WeightResidency<S>,
    w: &ConditionEmbedderNames,
    delta: bool,
) -> Result<ConditionEmbedderHandles, LoadError> {
    let lb = |l: &LinearNames| register_linear_bias(residency, &l.weight, &l.bias);
    Ok(ConditionEmbedderHandles {
        time_linear_1: lb(&w.time_linear_1)?,
        time_linear_2: lb(&w.time_linear_2)?,
        time_proj: lb(&w.time_proj)?,
        text_linear_1: lb(&w.text_linear_1)?,
        text_linear_2: lb(&w.text_linear_2)?,
        delta_linear_1: delta.then(|| lb(&w.delta_linear_1)).transpose()?,
        delta_linear_2: delta.then(|| lb(&w.delta_linear_2)).transpose()?,
    })
}

/// Module-level linear+bias: weight stays dense bf16 (`Linear2D [K, N]`), run on
/// the `matmul_module` pipeline. F16/F32 sources narrow to bf16 at upload.
fn register_linear_bias<S: WeightSource>(
    residency: &WeightResidency<S>,
    weight: &WeightId,
    bias: &WeightId,
) -> Result<LinearBiasHandles, LoadError> {
    Ok(LinearBiasHandles {
        weight: register_linear(residency, weight, None)?,
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
    // `linear_layout` asserts a 2-D shape for the transcode arm; the conv entry is
    // 5-D, so hand it the post-collapse [n, k] shape (the only field it inspects
    // besides the encoding).
    let mut entry2d = entry.clone();
    entry2d.shape = Shape(vec![n, k]);
    let (encoding, transpose, transcode) = linear_layout(weight, encoding, &entry2d, None)?;
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
        // Dense bf16/f16/f32 with a transcode target: requantize to the GGUF
        // block layout (N-major, no transpose). Used by the block matmul sites
        // when a load-time transcode is requested; module-level weights never
        // pass a transcode here (they stay dense bf16).
        StorageEncoding::Bf16 | StorageEncoding::F16 | StorageEncoding::F32
            if transcode.is_some() =>
        {
            assert_eq!(entry.shape.0.len(), 2, "transcode target must be 2-D");
            assert_eq!(
                entry.shape.0[1] % 32,
                0,
                "transcode requires K % 32 == 0 ({id:?})"
            );
            (encoding, TransposePolicy::None, transcode)
        }
        StorageEncoding::Bf16 | StorageEncoding::F16 | StorageEncoding::F32 => {
            (encoding, TransposePolicy::Linear2D, None)
        }
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
