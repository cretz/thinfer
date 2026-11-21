//! Qwen-Image dual-stream MMDiT. Ground truth:
//! `transformer_qwenimage.py::{QwenImageTransformerBlock,
//! QwenDoubleStreamAttnProcessor2_0, QwenImageTransformer2DModel}`.
//!
//! Each of the 60 blocks carries TWO residual streams (image `hs`, text `eh`)
//! that interact only at a single JOINT attention over `[txt ++ img]` tokens.
//! Per stream: `LayerNorm(no-affine) -> modulate(x*(1+scale)+shift) -> sublayer
//! -> gate*y residual`. Modulation is `Linear(SiLU(temb))` per stream, 6 signals
//! `[shift,scale,gate]_{msa,mlp}`. Attention has QKV bias + per-head QK-RMSNorm +
//! complex interleaved RoPE (vid freqs on img q/k, txt freqs on txt q/k). FFN is
//! plain GELU-approximate `dim->4*dim->dim`.
//!
//! Runs bf16 acts (see [`block_cfgs`]): the residual stream has large-outlier
//! channels beyond f16's +-65504, so block-wide f16 is not viable here. Block
//! matmuls are dequant-once, each weight decoded by ITS on-disk quant kind (a
//! `_K_M` file mixes Q4_K/Q5_K/Q6_K/Q8_0 per tensor; see
//! [`QwenImageDitPipelines::dense_dequant`]); img_in/txt_in/time_embed/
//! norm_out/proj_out are dense bf16 (adaln site).

use thinfer_core::backend::{Backend, BufRef, WgpuBackend, WgpuError, WgpuPipeline};
use thinfer_core::ops::{
    ActDtype, BcastAddF32, BcastFmaF32, BcastModulateF32, Bf16ToF16, F16ToBf16, GeluF32,
    LayerNormF32, Op, SiluF32, WeightDtype, WgslConfig,
};
use thinfer_core::quant::QuantKind;
use thinfer_core::residency::{GpuView, ResidencyError, WeightResidency};
use thinfer_core::trace::{self, PHASE};
use thinfer_core::weight::WeightSource;
use thinfer_core::workspace::{BatchBuf, BatchScope, Workspace};
use tracing::Instrument;

use crate::common::block::{
    ActBuf, Block, BlockPipelines, BlockWgslConfigs, DenseActSites, DequantStep, alloc_act,
    alloc_matmul_out_buf, op_rmsnorm, op_rope, op_sdpa,
};
use crate::common::embedders::{LinearBiasBufs, LinearBiasViews};
use crate::common::seq;
use crate::qwen_image::config;
use crate::qwen_image::loader::{DitBlockHandles, DitHandles};
use crate::qwen_image::rope::QwenImageRope;

/// Mixed-precision fast-attention path for the DiT block: an f16 subgroup SDPA
/// kernel plus the bf16<->f16 act casts that feed it. The residual stream stays
/// bf16 (large-outlier channels exceed f16's +-65504), but Q/K/V are O(1)
/// post-rmsnorm/rope and f16-safe, so `block_attn` casts only Q/K/V to f16, runs
/// the subgroup SDPA (the O(n^2) long pole at the edit joint), and casts the
/// output back to bf16. Present iff `i8_matmul` was requested AND the backend
/// exposes shader-f16 + subgroups; otherwise `block_attn` falls back to the
/// bf16 `op_sdpa`.
pub struct FastSdpa {
    pub sdpa: WgpuPipeline,
    pub to_f16: WgpuPipeline,
    pub to_bf16: WgpuPipeline,
    /// Lane-cluster width baked into `sdpa` (must match the dispatch).
    pub cl: u32,
}

/// `BlockPipelines` + a GELU pipeline (Qwen-Image FFN is gelu-approximate; the
/// common block set has no gelu, mirroring `WanDitPipelines`) + the optional
/// [`FastSdpa`] mixed-precision attention path.
pub struct QwenImageDitPipelines {
    pub block: BlockPipelines,
    pub gelu: WgpuPipeline,
    pub fast_sdpa: Option<FastSdpa>,
    /// Dense (dequant-once) step per quant kind. `BlockPipelines` bakes ONE
    /// scheme per site, which a mixed `_K_M` DiT (Q4_K + Q6_K within a block,
    /// Q5_K in some layers) would decode as garbage, so every block matmul
    /// picks the step of its own weight's kind.
    pub dense_dequant: std::collections::HashMap<QuantKind, DequantStep>,
}

impl QwenImageDitPipelines {
    /// `i8_matmul` gates the mixed-precision [`FastSdpa`] path: on (the default),
    /// Q/K/V run an f16 subgroup SDPA; off, the block uses the bf16 `op_sdpa`.
    /// (The name tracks the existing `--no-i8-matmul` knob; the residual + all
    /// matmuls stay bf16/Q8_0 regardless.) The vision tower passes `false` --
    /// its windowed attention rides the common-block `op_sdpa` with a real mask.
    pub async fn compile(
        backend: &WgpuBackend,
        cfgs: &BlockWgslConfigs,
        i8_matmul: bool,
    ) -> Result<Self, WgpuError> {
        let block = BlockPipelines::compile(backend, cfgs).await?;
        let gelu = backend
            .create_pipeline(
                "qwen_image_gelu",
                <GeluF32 as Op>::wgsl(&cfgs.ops),
                "main",
                <GeluF32 as Op>::layout(),
            )
            .await?;
        // f16 subgroup SDPA + bf16<->f16 casts. head_dim is 128 (% 32 == 0,
        // <= 128), so the subgroup kernel's D constraint (D % (4*CL) == 0) holds
        // for both CL=4 and CL=8.
        let (sg_min, _) = backend.subgroup_size_range();
        let fast_sdpa = if i8_matmul
            && backend.supports_shader_f16()
            && backend.supports_subgroups()
            && sg_min >= 4
        {
            let cl = if sg_min >= 8 { 8u32 } else { 4u32 };
            let sdpa_wgsl = format!(
                "{}{}",
                backend.subgroup_enable_directive(),
                thinfer_core::ops::sdpa::build_f16_sg_wgsl(cl, 1),
            );
            let sdpa = backend
                .create_pipeline(
                    "qwen_image_sdpa_f16_sg",
                    &sdpa_wgsl,
                    "main",
                    thinfer_core::ops::sdpa::sg_layout(),
                )
                .await?;
            let to_f16 = backend
                .create_pipeline(
                    "qwen_image_cast_bf16_f16",
                    <Bf16ToF16 as Op>::wgsl(&cfgs.ops),
                    "main",
                    <Bf16ToF16 as Op>::layout(),
                )
                .await?;
            let to_bf16 = backend
                .create_pipeline(
                    "qwen_image_cast_f16_bf16",
                    <F16ToBf16 as Op>::wgsl(&cfgs.ops),
                    "main",
                    <F16ToBf16 as Op>::layout(),
                )
                .await?;
            Some(FastSdpa {
                sdpa,
                to_f16,
                to_bf16,
                cl,
            })
        } else {
            None
        };
        let target = if cfgs.ops.act_dtype == ActDtype::F16 {
            thinfer_core::ops::dequant::DequantTarget::F16
        } else {
            thinfer_core::ops::dequant::DequantTarget::Bf16
        };
        let mut dense_dequant = std::collections::HashMap::new();
        for scheme in [
            QuantKind::Q8_0,
            QuantKind::Q4_K,
            QuantKind::Q5_K,
            QuantKind::Q6_K,
        ] {
            let pipeline = backend
                .create_pipeline(
                    &format!("qwen_image_dequant_{}", scheme.hint()),
                    &thinfer_core::ops::dequant::build_wgsl(scheme, target),
                    "main",
                    thinfer_core::ops::dequant::layout(),
                )
                .await?;
            dense_dequant.insert(
                scheme,
                DequantStep {
                    pipeline,
                    scheme: thinfer_core::ops::DequantScheme::Block(scheme),
                },
            );
        }
        Ok(Self {
            block,
            gelu,
            fast_sdpa,
            dense_dequant,
        })
    }
}

/// The DiT block config and the single source of truth for it (the pipeline +
/// the `dit_parity` / `dit_perf` tests all call this so the validated config
/// can't drift). bf16 acts: the qwen residual stream has large-outlier channels
/// that exceed f16's +-65504 (velocity diverges ~68% under f16), so block-wide
/// f16 -- and therefore the f16-only i8-DP4A / subgroup-SDPA block paths -- is
/// NOT viable here (unlike ideogram, whose residual stays in f16 range). The
/// fast-attention lever instead converts only the normalized (O(1), f16-safe)
/// Q/K/V to f16 for a subgroup SDPA, keeping the residual in bf16; that is gated
/// separately at pipeline compile (see `QwenImageDitPipelines`), NOT here. Block
/// matmuls are Q8_0 dequant-once; img_in/txt_in/time/norm_out/proj_out are bf16
/// (adaln site).
pub fn block_cfgs() -> BlockWgslConfigs {
    let ops = WgslConfig {
        bf16_quant_writes: false,
        act_dtype: ActDtype::Bf16,
        weight_dtype: WeightDtype::Bf16,
    };
    let q8 = WgslConfig {
        weight_dtype: WeightDtype::Quant(QuantKind::Q8_0),
        ..ops
    };
    BlockWgslConfigs {
        matmul_qkv: q8,
        matmul_qkv_self: q8,
        matmul_proj: q8,
        matmul_ffn_up: q8,
        matmul_ffn_down: q8,
        matmul_adaln: ops,
        ops,
        i8_sdpa: false,
        dense_acts: DenseActSites::default(),
        coopmat_acts: crate::common::block::CoopmatSites::default(),
        large_d_sdpa: false,
        fast_sdpa: false,
        decode_sdpa: false,
    }
}

// --- local op wrappers (mirror wan/dit_block.rs; not exported from common) ----

/// `out = layernorm(x)` (mean-subtract, no affine), eps folded into the uniform.
fn op_layernorm<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    bp: &BlockPipelines,
    x: ActBuf<'wsp>,
    dst: ActBuf<'wsp>,
    rows: u32,
    dim: u32,
    eps: f32,
) -> Result<(), WgpuError> {
    let u = scope.u32x4_uniform(rows, dim, eps.to_bits(), 0)?;
    scope.layernorm::<LayerNormF32>(&bp.layernorm, x.data, u, dst.data, rows)
}

/// `out = x * (1 + scale) + shift`, scale/shift `[dim]` activations broadcast
/// over rows (bias=1 folds the `1+`).
#[allow(clippy::too_many_arguments)]
fn op_modulate<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    bp: &BlockPipelines,
    x: ActBuf<'wsp>,
    scale: BatchBuf<'wsp>,
    shift: BatchBuf<'wsp>,
    dst: ActBuf<'wsp>,
    rows: u32,
    dim: u32,
) -> Result<(), WgpuError> {
    let u = scope.u32x4_uniform(dim, 1.0_f32.to_bits(), 0, 0)?;
    scope.bcast_modulate::<BcastModulateF32>(
        &bp.bcast_modulate,
        x.data,
        scale,
        shift,
        u,
        dst.data,
        rows * dim,
    )
}

/// `out = x + gate * y`, gate `[dim]` broadcast over rows.
#[allow(clippy::too_many_arguments)]
fn op_gate_residual<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    bp: &BlockPipelines,
    x: ActBuf<'wsp>,
    gate: BatchBuf<'wsp>,
    y: ActBuf<'wsp>,
    dst: ActBuf<'wsp>,
    rows: u32,
    dim: u32,
) -> Result<(), WgpuError> {
    let u = scope.u32x4_uniform(dim, 0, 0, 0)?;
    scope.bcast_fma::<BcastFmaF32>(&bp.bcast_fma, x.data, gate, y.data, u, dst.data, rows * dim)
}

/// One stream's `[shift, scale, gate]` for a sublayer (each `[dim]`).
#[derive(Clone, Copy)]
struct ModSig<'wsp> {
    shift: BatchBuf<'wsp>,
    scale: BatchBuf<'wsp>,
    gate: BatchBuf<'wsp>,
}

/// The image stream's modulation: `main` for every row, except that with
/// `zero_cond_t` (Edit-2511) the reference rows `[split, rows)` take `refs`,
/// the signals of the timestep-0 embedding (diffusers `modulate_index`).
#[derive(Clone, Copy)]
struct ImgSig<'wsp> {
    main: ModSig<'wsp>,
    refs: Option<ModSig<'wsp>>,
    split: u32,
}

impl<'wsp> ImgSig<'wsp> {
    /// `(row_start, rows, signals)` spans covering `[0, rows)`.
    fn spans(&self, rows: u32) -> Vec<(u32, u32, ModSig<'wsp>)> {
        match self.refs {
            Some(r) if self.split < rows => vec![
                (0, self.split, self.main),
                (self.split, rows - self.split, r),
            ],
            _ => vec![(0, rows, self.main)],
        }
    }
}

/// Row-range view of a `[rows, dim]` act buffer (row offsets of a bf16/f32
/// `dim=3072` row are 256B-aligned, as storage binding requires).
fn act_rows<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    bp: &BlockPipelines,
    buf: ActBuf<'wsp>,
    row0: u32,
    rows: u32,
    dim: u32,
) -> ActBuf<'wsp> {
    let row_b = bp.act_bytes(dim);
    ActBuf::dense(scope.subview(&buf.data, row0 as u64 * row_b, rows as u64 * row_b))
}

/// [`op_modulate`] over the image stream, per [`ImgSig`] span.
fn op_modulate_img<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    bp: &BlockPipelines,
    x: ActBuf<'wsp>,
    sig: ImgSig<'wsp>,
    dst: ActBuf<'wsp>,
    rows: u32,
    dim: u32,
) -> Result<(), WgpuError> {
    for (r0, n, s) in sig.spans(rows) {
        let xs = act_rows(scope, bp, x, r0, n, dim);
        let ds = act_rows(scope, bp, dst, r0, n, dim);
        op_modulate(scope, bp, xs, s.scale, s.shift, ds, n, dim)?;
    }
    Ok(())
}

/// [`op_gate_residual`] over the image stream, per [`ImgSig`] span.
#[allow(clippy::too_many_arguments)]
fn op_gate_residual_img<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    bp: &BlockPipelines,
    x: ActBuf<'wsp>,
    sig: ImgSig<'wsp>,
    y: ActBuf<'wsp>,
    dst: ActBuf<'wsp>,
    rows: u32,
    dim: u32,
) -> Result<(), WgpuError> {
    for (r0, n, s) in sig.spans(rows) {
        let xs = act_rows(scope, bp, x, r0, n, dim);
        let ys = act_rows(scope, bp, y, r0, n, dim);
        let ds = act_rows(scope, bp, dst, r0, n, dim);
        op_gate_residual(scope, bp, xs, s.gate, ys, ds, n, dim)?;
    }
    Ok(())
}

/// The `[shift, scale, gate]` of sublayer `half` (0 = attention, 1 = MLP) of
/// a `[1, 6*dim]` modulation-params buffer.
fn mod_sig<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    bp: &BlockPipelines,
    params: BatchBuf<'wsp>,
    half: u32,
    dim: u32,
) -> Result<ModSig<'wsp>, WgpuError> {
    let k = 3 * half;
    Ok(ModSig {
        shift: mod_signal(scope, bp, params, k, dim)?,
        scale: mod_signal(scope, bp, params, k + 1, dim)?,
        gate: mod_signal(scope, bp, params, k + 2, dim)?,
    })
}

/// Both streams' signals for sublayer `half`: `Linear(SiLU(temb))` per
/// stream; with `silu_ref` (zero_cond_t), the image stream's reference rows
/// get `img_mod(SiLU(temb(t=0)))` instead.
#[allow(clippy::too_many_arguments)]
fn block_sigs<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    bp: &BlockPipelines,
    bufs: &'wsp BlockBufs<'_>,
    silu_temb: BatchBuf<'wsp>,
    silu_ref: Option<BatchBuf<'wsp>>,
    split: u32,
    half: u32,
) -> Result<(ImgSig<'wsp>, ModSig<'wsp>), WgpuError> {
    let dim = config::DIM as u32;
    let silu = ActBuf::dense(silu_temb);
    let img_modp = qbiased(scope, bp, silu, &bufs.img_mod, 1, 6 * dim, dim, Site::Proj)?;
    let txt_modp = qbiased(scope, bp, silu, &bufs.txt_mod, 1, 6 * dim, dim, Site::Proj)?;
    let refs = match silu_ref {
        Some(s) => {
            let p = qbiased(
                scope,
                bp,
                ActBuf::dense(s),
                &bufs.img_mod,
                1,
                6 * dim,
                dim,
                Site::Proj,
            )?;
            Some(mod_sig(scope, bp, p.data, half, dim)?)
        }
        None => None,
    };
    let img = ImgSig {
        main: mod_sig(scope, bp, img_modp.data, half, dim)?,
        refs,
        split,
    };
    Ok((img, mod_sig(scope, bp, txt_modp.data, half, dim)?))
}

/// `out = x + bias[i % dim]`, bias `[dim]` resident weight broadcast over rows.
fn op_bias_add<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    bp: &BlockPipelines,
    x: ActBuf<'wsp>,
    bias: BatchBuf<'wsp>,
    dst: ActBuf<'wsp>,
    rows: u32,
    dim: u32,
) -> Result<(), WgpuError> {
    let u = scope.u32x4_uniform(dim, 0, 0, 0)?;
    scope.bcast_add::<BcastAddF32>(&bp.bcast_add, x.data, bias, u, dst.data, rows * dim)
}

/// Which compiled matmul site a block projection routes through. The site
/// selects the pipeline + weight path and, for the Quant sites, whether the
/// A-side runs the i8 DP4A path or the dequant-once bf16 path (per
/// `BlockWgslConfigs::dense_acts`). Attention q/k/v -> `Qkv`; attention out-proj
/// -> `Proj`; FFN up/down -> `FfnUp`/`FfnDown`; the per-stream AdaLN modulation
/// reuses the dense `Proj` site (its weight is Q8_0 too, but the A-side is a
/// single SiLU(temb) row, so i8 buys ~no compute and would only add quant
/// error). The F16->bf16 embedders (img_in/txt_in/time/norm_out/proj_out) ->
/// `Adaln` (bf16 weight). Under the bf16-act config every Quant site compiles
/// the same kernel, so routing is behaviour-identical there; it only diverges
/// once `dense_acts` opts the i8'd sites into DP4A under F16 acts.
#[derive(Clone, Copy)]
enum Site {
    Qkv,
    Proj,
    FfnUp,
    FfnDown,
    Adaln,
}

/// Matmul `[rows, n] = x[rows, k] @ wᵀ` through the given compiled site.
fn matmul_site<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    bp: &BlockPipelines,
    x: ActBuf<'wsp>,
    w: BatchBuf<'wsp>,
    rows: u32,
    n: u32,
    k: u32,
    site: Site,
    dq_kind: Option<&DequantStep>,
) -> Result<BatchBuf<'wsp>, WgpuError> {
    let out = alloc_matmul_out_buf(scope, bp, rows * n)?;
    let dims = scope.u32x4_uniform(rows, n, k, 0)?;
    #[allow(clippy::type_complexity)]
    let (i8, dq_i8, dq, pipe, op): (_, _, _, _, _) = match site {
        Site::Qkv => (
            bp.matmul_i8_qkv.as_ref(),
            bp.dequant_i8_qkv.as_ref(),
            bp.dequant_qkv.as_ref(),
            &bp.matmul_qkv,
            &bp.matmuls.qkv,
        ),
        Site::Proj => (
            bp.matmul_i8_proj.as_ref(),
            bp.dequant_i8_proj.as_ref(),
            bp.dequant_proj.as_ref(),
            &bp.matmul_proj,
            &bp.matmuls.proj,
        ),
        Site::FfnUp => (
            bp.matmul_i8_ffn_up.as_ref(),
            bp.dequant_i8_ffn_up.as_ref(),
            bp.dequant_ffn_up.as_ref(),
            &bp.matmul_ffn_up,
            &bp.matmuls.ffn_up,
        ),
        Site::FfnDown => (
            bp.matmul_i8_ffn_down.as_ref(),
            bp.dequant_i8_ffn_down.as_ref(),
            bp.dequant_ffn_down.as_ref(),
            &bp.matmul_ffn_down,
            &bp.matmuls.ffn_down,
        ),
        Site::Adaln => (None, None, None, &bp.matmul_adaln, &bp.matmuls.adaln),
    };
    // The weight's own kind overrides the site's baked scheme (mixed `_K_M`).
    let dq = dq_kind.or(dq);
    Block::dispatch_matmul_site(
        scope, bp, x, w, out, dims, i8, dq_i8, dq, pipe, op, rows, n, k,
    )?;
    Ok(out)
}

/// Biased projection `x @ wᵀ + bias` -> dense act `[rows, n]`, routed through
/// `site` (see [`Site`]).
#[allow(clippy::too_many_arguments)]
fn biased<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    bp: &BlockPipelines,
    x: ActBuf<'wsp>,
    w: &LinearBiasBufs,
    rows: u32,
    n: u32,
    k: u32,
    site: Site,
) -> Result<ActBuf<'wsp>, WgpuError> {
    biased_dq(scope, bp, x, w, rows, n, k, site, None)
}

#[allow(clippy::too_many_arguments)]
fn biased_dq<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    bp: &BlockPipelines,
    x: ActBuf<'wsp>,
    w: &LinearBiasBufs,
    rows: u32,
    n: u32,
    k: u32,
    site: Site,
    dq_kind: Option<&DequantStep>,
) -> Result<ActBuf<'wsp>, WgpuError> {
    let wv = scope.import_copy(w.weight);
    let pre = matmul_site(scope, bp, x, wv, rows, n, k, site, dq_kind)?;
    let bv = scope.import_copy(w.bias);
    let out = alloc_act(scope, bp, rows, n)?;
    op_bias_add(scope, bp, ActBuf::dense(pre), bv, out, rows, n)?;
    Ok(out)
}

/// A block linear's buffers + the dense dequant step of its weight's on-disk
/// kind (`None` = a float weight: it runs the bf16 dense site instead).
struct QLinBufs<'p> {
    lin: LinearBiasBufs,
    dq: Option<&'p DequantStep>,
}

/// [`biased`] for a block linear, decoded by its own quant kind.
#[allow(clippy::too_many_arguments)]
fn qbiased<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    bp: &BlockPipelines,
    x: ActBuf<'wsp>,
    w: &QLinBufs<'_>,
    rows: u32,
    n: u32,
    k: u32,
    site: Site,
) -> Result<ActBuf<'wsp>, WgpuError> {
    match w.dq {
        Some(dq) => biased_dq(scope, bp, x, &w.lin, rows, n, k, site, Some(dq)),
        None => biased(scope, bp, x, &w.lin, rows, n, k, Site::Adaln),
    }
}

// --- timestep sinusoid (CPU, diffusers `Timesteps`) ---------------------------

/// `get_timestep_embedding(t, 256, flip_sin_to_cos=True, downscale_freq_shift=0,
/// scale=1000, max_period=10000)`: `[cos(a) ++ sin(a)]`, `a_j = 1000*t*
/// exp(-ln(10000)*j/128)`, j in 0..128. Returns `[256]` f32.
fn timestep_sinusoid(t: f32) -> Vec<f32> {
    const HALF: usize = 128;
    const MAX_PERIOD: f32 = 10_000.0;
    const SCALE: f32 = 1000.0;
    let mut out = vec![0.0_f32; 2 * HALF];
    for j in 0..HALF {
        let freq = (-(MAX_PERIOD.ln()) * j as f32 / HALF as f32).exp();
        let a = SCALE * t * freq;
        out[j] = a.cos();
        out[HALF + j] = a.sin();
    }
    out
}

// --- per-block weight views ---------------------------------------------------

/// Each block linear's on-disk quant kind (see `loader::QLinearHandles`).
#[derive(Clone, Copy)]
struct BlockKinds {
    img_mod: Option<QuantKind>,
    txt_mod: Option<QuantKind>,
    to_q: Option<QuantKind>,
    to_k: Option<QuantKind>,
    to_v: Option<QuantKind>,
    to_out: Option<QuantKind>,
    add_q: Option<QuantKind>,
    add_k: Option<QuantKind>,
    add_v: Option<QuantKind>,
    to_add_out: Option<QuantKind>,
    img_mlp_0: Option<QuantKind>,
    img_mlp_2: Option<QuantKind>,
    txt_mlp_0: Option<QuantKind>,
    txt_mlp_2: Option<QuantKind>,
}

impl BlockKinds {
    fn of(h: &DitBlockHandles) -> Self {
        Self {
            img_mod: h.img_mod.kind,
            txt_mod: h.txt_mod.kind,
            to_q: h.to_q.kind,
            to_k: h.to_k.kind,
            to_v: h.to_v.kind,
            to_out: h.to_out.kind,
            add_q: h.add_q.kind,
            add_k: h.add_k.kind,
            add_v: h.add_v.kind,
            to_add_out: h.to_add_out.kind,
            img_mlp_0: h.img_mlp_0.kind,
            img_mlp_2: h.img_mlp_2.kind,
            txt_mlp_0: h.txt_mlp_0.kind,
            txt_mlp_2: h.txt_mlp_2.kind,
        }
    }
}

struct BlockViews<'a> {
    kinds: BlockKinds,
    img_mod: LinearBiasViews<'a>,
    txt_mod: LinearBiasViews<'a>,
    to_q: LinearBiasViews<'a>,
    to_k: LinearBiasViews<'a>,
    to_v: LinearBiasViews<'a>,
    to_out: LinearBiasViews<'a>,
    add_q: LinearBiasViews<'a>,
    add_k: LinearBiasViews<'a>,
    add_v: LinearBiasViews<'a>,
    to_add_out: LinearBiasViews<'a>,
    norm_q: GpuView<'a>,
    norm_k: GpuView<'a>,
    norm_added_q: GpuView<'a>,
    norm_added_k: GpuView<'a>,
    img_mlp_0: LinearBiasViews<'a>,
    img_mlp_2: LinearBiasViews<'a>,
    txt_mlp_0: LinearBiasViews<'a>,
    txt_mlp_2: LinearBiasViews<'a>,
}

struct BlockBufs<'p> {
    img_mod: QLinBufs<'p>,
    txt_mod: QLinBufs<'p>,
    to_q: QLinBufs<'p>,
    to_k: QLinBufs<'p>,
    to_v: QLinBufs<'p>,
    to_out: QLinBufs<'p>,
    add_q: QLinBufs<'p>,
    add_k: QLinBufs<'p>,
    add_v: QLinBufs<'p>,
    to_add_out: QLinBufs<'p>,
    norm_q: BufRef,
    norm_k: BufRef,
    norm_added_q: BufRef,
    norm_added_k: BufRef,
    img_mlp_0: QLinBufs<'p>,
    img_mlp_2: QLinBufs<'p>,
    txt_mlp_0: QLinBufs<'p>,
    txt_mlp_2: QLinBufs<'p>,
}

impl<'a> BlockViews<'a> {
    async fn acquire<S: WeightSource>(
        h: &DitBlockHandles,
        residency: &'a WeightResidency<S>,
        backend: &WgpuBackend,
    ) -> Result<BlockViews<'a>, ResidencyError<S::Error, WgpuError>> {
        Ok(BlockViews {
            kinds: BlockKinds::of(h),
            img_mod: h.img_mod.lin.acquire(residency, backend).await?,
            txt_mod: h.txt_mod.lin.acquire(residency, backend).await?,
            to_q: h.to_q.lin.acquire(residency, backend).await?,
            to_k: h.to_k.lin.acquire(residency, backend).await?,
            to_v: h.to_v.lin.acquire(residency, backend).await?,
            to_out: h.to_out.lin.acquire(residency, backend).await?,
            add_q: h.add_q.lin.acquire(residency, backend).await?,
            add_k: h.add_k.lin.acquire(residency, backend).await?,
            add_v: h.add_v.lin.acquire(residency, backend).await?,
            to_add_out: h.to_add_out.lin.acquire(residency, backend).await?,
            norm_q: residency.acquire(h.norm_q, backend).await?,
            norm_k: residency.acquire(h.norm_k, backend).await?,
            norm_added_q: residency.acquire(h.norm_added_q, backend).await?,
            norm_added_k: residency.acquire(h.norm_added_k, backend).await?,
            img_mlp_0: h.img_mlp_0.lin.acquire(residency, backend).await?,
            img_mlp_2: h.img_mlp_2.lin.acquire(residency, backend).await?,
            txt_mlp_0: h.txt_mlp_0.lin.acquire(residency, backend).await?,
            txt_mlp_2: h.txt_mlp_2.lin.acquire(residency, backend).await?,
        })
    }

    fn bufs<'p>(&self, pipelines: &'p QwenImageDitPipelines) -> BlockBufs<'p> {
        let q = |lin: &LinearBiasViews<'_>, kind: Option<QuantKind>| QLinBufs {
            lin: lin.bufs(),
            dq: kind.and_then(|k| pipelines.dense_dequant.get(&k)),
        };
        BlockBufs {
            img_mod: q(&self.img_mod, self.kinds.img_mod),
            txt_mod: q(&self.txt_mod, self.kinds.txt_mod),
            to_q: q(&self.to_q, self.kinds.to_q),
            to_k: q(&self.to_k, self.kinds.to_k),
            to_v: q(&self.to_v, self.kinds.to_v),
            to_out: q(&self.to_out, self.kinds.to_out),
            add_q: q(&self.add_q, self.kinds.add_q),
            add_k: q(&self.add_k, self.kinds.add_k),
            add_v: q(&self.add_v, self.kinds.add_v),
            to_add_out: q(&self.to_add_out, self.kinds.to_add_out),
            norm_q: self.norm_q.buf(),
            norm_k: self.norm_k.buf(),
            norm_added_q: self.norm_added_q.buf(),
            norm_added_k: self.norm_added_k.buf(),
            img_mlp_0: q(&self.img_mlp_0, self.kinds.img_mlp_0),
            img_mlp_2: q(&self.img_mlp_2, self.kinds.img_mlp_2),
            txt_mlp_0: q(&self.txt_mlp_0, self.kinds.txt_mlp_0),
            txt_mlp_2: q(&self.txt_mlp_2, self.kinds.txt_mlp_2),
        }
    }
}

// --- the dual-stream block forward --------------------------------------------

/// Slice the 6 modulation signals `[shift_msa, scale_msa, gate_msa, shift_mlp,
/// scale_mlp, gate_mlp]` out of a `[1, 6*dim]` act buffer (each `[dim]`).
fn mod_signal<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    bp: &BlockPipelines,
    src: BatchBuf<'wsp>,
    k: u32,
    dim: u32,
) -> Result<BatchBuf<'wsp>, WgpuError> {
    let dst = alloc_act(scope, bp, 1, dim)?;
    let row_b = bp.act_bytes(dim);
    scope.copy_buffer_to_buffer(src, k as u64 * row_b, dst.data, 0, row_b)?;
    Ok(dst.data)
}

/// Attention half of a joint DiT block: modulation + norm1 + joint SDPA + the
/// gate1 residual, writing the post-attention streams into `hs1_out`/`eh1_out`.
/// Split from the FFN half so each GPU submit's runtime stays well under the
/// Windows ~2s TDR watchdog at the edit path's large joint sequence (the 1 MP
/// reference latent pushes joint past 4.5k tokens, where a single whole-block
/// submit runs ~2s and intermittently trips the watchdog). The two halves are
/// numerically identical to the prior single-submit block: the split is only a
/// submit boundary, the MLP modulation signals are recomputed in `block_mlp`
/// from the same `silu_temb` (two tiny matmuls), so no extra state crosses it.
#[allow(clippy::too_many_arguments)]
fn block_attn<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    pipelines: &QwenImageDitPipelines,
    hs_in: BatchBuf<'wsp>,
    eh_in: BatchBuf<'wsp>,
    silu_temb: BatchBuf<'wsp>,
    silu_ref: Option<BatchBuf<'wsp>>,
    split: u32,
    vid_freqs: BatchBuf<'wsp>,
    txt_freqs: BatchBuf<'wsp>,
    mask: BatchBuf<'wsp>,
    img_seq: u32,
    txt_seq: u32,
    hs1_out: BatchBuf<'wsp>,
    eh1_out: BatchBuf<'wsp>,
    bufs: &'wsp BlockBufs<'_>,
) -> Result<(), WgpuError> {
    let bp = &pipelines.block;
    let dim = config::DIM as u32;
    let hd = config::HEAD_DIM as u32;
    let heads = config::N_HEADS as u32;
    let eps = config::NORM_EPS;
    let joint = txt_seq + img_seq;
    let scale = 1.0 / (hd as f32).sqrt();
    let hs_in = ActBuf::dense(hs_in);
    let eh_in = ActBuf::dense(eh_in);

    // --- modulation params: Linear(SiLU(temb)) -> [1, 6*dim] per stream ---
    let (i_msa, t_msa) = block_sigs(scope, bp, bufs, silu_temb, silu_ref, split, 0)?;

    // --- norm1 + modulate both streams ---
    let img_n1 = alloc_act(scope, bp, img_seq, dim)?;
    op_layernorm(scope, bp, hs_in, img_n1, img_seq, dim, eps)?;
    let img_mod1 = alloc_act(scope, bp, img_seq, dim)?;
    op_modulate_img(scope, bp, img_n1, i_msa, img_mod1, img_seq, dim)?;
    let txt_n1 = alloc_act(scope, bp, txt_seq, dim)?;
    op_layernorm(scope, bp, eh_in, txt_n1, txt_seq, dim, eps)?;
    let txt_mod1 = alloc_act(scope, bp, txt_seq, dim)?;
    op_modulate(
        scope,
        bp,
        txt_n1,
        t_msa.scale,
        t_msa.shift,
        txt_mod1,
        txt_seq,
        dim,
    )?;

    // --- joint attention ---
    // per-head QK-RMSNorm then interleaved RoPE; concat [txt ++ img] -> SDPA.
    let proj_qkn = |x: ActBuf<'wsp>,
                    w: &QLinBufs<'_>,
                    rows: u32,
                    norm_w: BufRef,
                    freqs: BatchBuf<'wsp>,
                    is_q: bool|
     -> Result<ActBuf<'wsp>, WgpuError> {
        let p = qbiased(scope, bp, x, w, rows, dim, dim, Site::Qkv)?;
        let normed = alloc_act(scope, bp, rows * heads, hd)?;
        let nw = scope.import_copy(norm_w);
        op_rmsnorm(scope, bp, p, nw, normed, rows * heads, hd, eps)?;
        let _ = is_q;
        let roped = alloc_act(scope, bp, rows, dim)?;
        op_rope(scope, bp, normed, freqs, roped, rows, heads, hd)?;
        Ok(roped)
    };
    // image q/k (qk-norm + rope), v (raw)
    let img_q = proj_qkn(img_mod1, &bufs.to_q, img_seq, bufs.norm_q, vid_freqs, true)?;
    let img_k = proj_qkn(img_mod1, &bufs.to_k, img_seq, bufs.norm_k, vid_freqs, false)?;
    let img_v = qbiased(
        scope,
        bp,
        img_mod1,
        &bufs.to_v,
        img_seq,
        dim,
        dim,
        Site::Qkv,
    )?;
    let txt_q = proj_qkn(
        txt_mod1,
        &bufs.add_q,
        txt_seq,
        bufs.norm_added_q,
        txt_freqs,
        true,
    )?;
    let txt_k = proj_qkn(
        txt_mod1,
        &bufs.add_k,
        txt_seq,
        bufs.norm_added_k,
        txt_freqs,
        false,
    )?;
    let txt_v = qbiased(
        scope,
        bp,
        txt_mod1,
        &bufs.add_v,
        txt_seq,
        dim,
        dim,
        Site::Qkv,
    )?;

    // concat [txt ++ img] along seq for q/k/v
    let cat = |a: ActBuf<'wsp>, b: ActBuf<'wsp>| -> Result<ActBuf<'wsp>, WgpuError> {
        let out = alloc_act(scope, bp, joint, dim)?;
        let abytes = bp.act_bytes(txt_seq * dim);
        let bbytes = bp.act_bytes(img_seq * dim);
        scope.copy_buffer_to_buffer(a.data, 0, out.data, 0, abytes)?;
        scope.copy_buffer_to_buffer(b.data, 0, out.data, abytes, bbytes)?;
        Ok(out)
    };
    let jq = cat(txt_q, img_q)?;
    let jk = cat(txt_k, img_k)?;
    let jv = cat(ActBuf::dense(txt_v.data), ActBuf::dense(img_v.data))?;

    let sa = alloc_act(scope, bp, joint, dim)?;
    match pipelines.fast_sdpa.as_ref() {
        // Mixed-precision path: cast the (f16-safe, post-rmsnorm/rope) Q/K/V to
        // f16, run the subgroup SDPA, cast the output back to the bf16 residual.
        // The f16 buffers are the same byte size as bf16 (2 bytes/elem), so they
        // reuse `alloc_act`'s bf16 sizing. has_mask=0 (full joint attention), so
        // the bf16-sized `mask` binding is never read -- only its size matters.
        Some(fast) => {
            let jq_f16 = alloc_act(scope, bp, joint, dim)?;
            let jk_f16 = alloc_act(scope, bp, joint, dim)?;
            let jv_f16 = alloc_act(scope, bp, joint, dim)?;
            scope.dispatch_op::<Bf16ToF16>(&fast.to_f16, &[jq.data], jq_f16.data)?;
            scope.dispatch_op::<Bf16ToF16>(&fast.to_f16, &[jk.data], jk_f16.data)?;
            scope.dispatch_op::<Bf16ToF16>(&fast.to_f16, &[jv.data], jv_f16.data)?;
            let sa_f16 = alloc_act(scope, bp, joint, dim)?;
            let u = crate::common::block::sdpa_uniform(
                scope, 1, heads, heads, joint, joint, hd, scale, 0,
            )?;
            scope.sdpa_sg(
                &fast.sdpa,
                jq_f16.data,
                jk_f16.data,
                jv_f16.data,
                mask,
                u,
                sa_f16.data,
                fast.cl,
                1,
                1,
                joint,
                heads,
                hd,
            )?;
            scope.dispatch_op::<F16ToBf16>(&fast.to_bf16, &[sa_f16.data], sa.data)?;
        }
        None => op_sdpa(
            scope, bp, jq, jk, jv, mask, sa, 1, joint, joint, heads, heads, hd, scale, 0,
        )?,
    }
    // split [txt ++ img]
    let abytes = bp.act_bytes(txt_seq * dim);
    let bbytes = bp.act_bytes(img_seq * dim);
    let txt_sa = alloc_act(scope, bp, txt_seq, dim)?;
    scope.copy_buffer_to_buffer(sa.data, 0, txt_sa.data, 0, abytes)?;
    let img_sa = alloc_act(scope, bp, img_seq, dim)?;
    scope.copy_buffer_to_buffer(sa.data, abytes, img_sa.data, 0, bbytes)?;

    let img_attn = qbiased(
        scope,
        bp,
        img_sa,
        &bufs.to_out,
        img_seq,
        dim,
        dim,
        Site::Proj,
    )?;
    let txt_attn = qbiased(
        scope,
        bp,
        txt_sa,
        &bufs.to_add_out,
        txt_seq,
        dim,
        dim,
        Site::Proj,
    )?;

    // --- gate1 residual (written to the carry buffers; FFN half resumes here) ---
    op_gate_residual_img(
        scope,
        bp,
        hs_in,
        i_msa,
        img_attn,
        ActBuf::dense(hs1_out),
        img_seq,
        dim,
    )?;
    op_gate_residual(
        scope,
        bp,
        eh_in,
        t_msa.gate,
        txt_attn,
        ActBuf::dense(eh1_out),
        txt_seq,
        dim,
    )?;
    Ok(())
}

/// FFN half of a joint DiT block: norm2 + MLP modulation + GELU MLP + gate2
/// residual per stream, resuming from the post-attention `hs1`/`eh1` streams
/// written by [`block_attn`]. The MLP modulation signals are recomputed here
/// from `silu_temb` (the same two tiny matmuls), so the only state crossing the
/// submit boundary is the two residual streams. See [`block_attn`] for why the
/// block is split.
#[allow(clippy::too_many_arguments)]
fn block_mlp<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    pipelines: &QwenImageDitPipelines,
    hs1: BatchBuf<'wsp>,
    eh1: BatchBuf<'wsp>,
    silu_temb: BatchBuf<'wsp>,
    silu_ref: Option<BatchBuf<'wsp>>,
    split: u32,
    img_seq: u32,
    txt_seq: u32,
    hs_out: BatchBuf<'wsp>,
    eh_out: BatchBuf<'wsp>,
    bufs: &'wsp BlockBufs<'_>,
) -> Result<(), WgpuError> {
    let bp = &pipelines.block;
    let dim = config::DIM as u32;
    let ffn = config::FFN_HIDDEN as u32;
    let eps = config::NORM_EPS;
    let hs1 = ActBuf::dense(hs1);
    let eh1 = ActBuf::dense(eh1);

    // Recompute the MLP modulation signals (cheap; keeps the split stateless).
    let (i_mlp, t_mlp) = block_sigs(scope, bp, bufs, silu_temb, silu_ref, split, 1)?;

    // --- norm2 + modulate + GELU MLP + gate2 residual, per stream ---
    let mlp = |x: ActBuf<'wsp>,
               w0: &QLinBufs<'_>,
               w2: &QLinBufs<'_>,
               rows: u32|
     -> Result<ActBuf<'wsp>, WgpuError> {
        let up = qbiased(scope, bp, x, w0, rows, ffn, dim, Site::FfnUp)?;
        let g = alloc_act(scope, bp, rows, ffn)?;
        scope.dispatch_op::<GeluF32>(&pipelines.gelu, &[up.data], g.data)?;
        qbiased(scope, bp, g, w2, rows, dim, ffn, Site::FfnDown)
    };

    let img_n2 = alloc_act(scope, bp, img_seq, dim)?;
    op_layernorm(scope, bp, hs1, img_n2, img_seq, dim, eps)?;
    let img_mod2 = alloc_act(scope, bp, img_seq, dim)?;
    op_modulate_img(scope, bp, img_n2, i_mlp, img_mod2, img_seq, dim)?;
    let img_mlp = mlp(img_mod2, &bufs.img_mlp_0, &bufs.img_mlp_2, img_seq)?;
    op_gate_residual_img(
        scope,
        bp,
        hs1,
        i_mlp,
        img_mlp,
        ActBuf::dense(hs_out),
        img_seq,
        dim,
    )?;

    let txt_n2 = alloc_act(scope, bp, txt_seq, dim)?;
    op_layernorm(scope, bp, eh1, txt_n2, txt_seq, dim, eps)?;
    let txt_mod2 = alloc_act(scope, bp, txt_seq, dim)?;
    op_modulate(
        scope,
        bp,
        txt_n2,
        t_mlp.scale,
        t_mlp.shift,
        txt_mod2,
        txt_seq,
        dim,
    )?;
    let txt_mlp = mlp(txt_mod2, &bufs.txt_mlp_0, &bufs.txt_mlp_2, txt_seq)?;
    op_gate_residual(
        scope,
        bp,
        eh1,
        t_mlp.gate,
        txt_mlp,
        ActBuf::dense(eh_out),
        txt_seq,
        dim,
    )?;
    Ok(())
}

// --- driver -------------------------------------------------------------------

/// Per-stage taps (parity bisection).
#[derive(Default)]
pub struct DitTaps {
    pub temb: Option<Vec<f32>>,
    pub block0_img: Option<Vec<f32>>,
    pub block0_txt: Option<Vec<f32>>,
}

#[derive(Clone, Debug)]
pub struct DitOutput {
    /// Velocity in patch-token space `[img_seq, IN_CHANNELS=64]` (proj_out).
    pub velocity: Vec<f32>,
    pub img_seq: usize,
}

#[derive(Debug)]
pub enum DitError<SE: core::fmt::Debug> {
    Wgpu(WgpuError),
    Residency(ResidencyError<SE, WgpuError>),
}

impl<SE: core::fmt::Debug> From<WgpuError> for DitError<SE> {
    fn from(e: WgpuError) -> Self {
        Self::Wgpu(e)
    }
}
impl<SE: core::fmt::Debug> From<ResidencyError<SE, WgpuError>> for DitError<SE> {
    fn from(e: ResidencyError<SE, WgpuError>) -> Self {
        Self::Residency(e)
    }
}

pub struct QwenImageDit {
    rope: QwenImageRope,
    /// Edit-2511 (`zero_cond_t`): reference tokens are modulated as if clean
    /// (timestep 0). Checkpoints mark it with an `__index_timestep_zero__`
    /// tensor (see [`ZERO_COND_T_MARKER`]).
    zero_cond_t: bool,
}

/// The empty tensor a 2511-family checkpoint carries (ComfyUI's
/// `model_detection` keys `index_timestep_zero` off it).
pub const ZERO_COND_T_MARKER: &str = "__index_timestep_zero__";

/// Upload a host f32 `[.., n]` row-major buffer as an act buffer.
fn upload_act(
    scratch: &Workspace<WgpuBackend>,
    backend: &WgpuBackend,
    bp: &BlockPipelines,
    host: &[f32],
    n: u32,
) -> Result<thinfer_core::workspace::WsBuf<WgpuBackend>, WgpuError> {
    let buf = scratch.alloc(bp.act_bytes(n))?;
    let bytes = seq::act_upload_bytes(bp.act_dtype, host);
    backend.write_buffer(buf.id, 0, &bytes)?;
    Ok(buf)
}

fn upload_freqs(
    scratch: &Workspace<WgpuBackend>,
    backend: &WgpuBackend,
    bp: &BlockPipelines,
    freqs: &[f32],
) -> Result<thinfer_core::workspace::WsBuf<WgpuBackend>, WgpuError> {
    let bytes = seq::freqs_upload_bytes(bp.act_dtype, freqs);
    let buf = scratch.alloc(bytes.len() as u64)?;
    backend.write_buffer(buf.id, 0, &bytes)?;
    Ok(buf)
}

impl QwenImageDit {
    pub fn new(zero_cond_t: bool) -> Self {
        Self {
            rope: QwenImageRope::new(),
            zero_cond_t,
        }
    }

    /// `(temb, SiLU(temb))` for timestep `t`, `temb =
    /// linear_2(SiLU(linear_1(sinusoid(t))))`, each `[1, dim]`.
    async fn silu_temb<S: WeightSource>(
        backend: &WgpuBackend,
        bp: &BlockPipelines,
        residency: &WeightResidency<S>,
        scratch: &Workspace<WgpuBackend>,
        handles: &DitHandles,
        t: f32,
    ) -> Result<
        (
            thinfer_core::workspace::WsBuf<WgpuBackend>,
            thinfer_core::workspace::WsBuf<WgpuBackend>,
        ),
        DitError<S::Error>,
    > {
        let dim = config::DIM as u32;
        let sin = timestep_sinusoid(t);
        let sin_buf = upload_act(scratch, backend, bp, &sin, 256)?;
        let temb = scratch.alloc(bp.act_bytes(dim))?;
        let silu_temb = scratch.alloc(bp.act_bytes(dim))?;
        let v1 = handles
            .top
            .time_linear_1
            .acquire(residency, backend)
            .await?;
        let v2 = handles
            .top
            .time_linear_2
            .acquire(residency, backend)
            .await?;
        let (w1, w2) = (v1.bufs(), v2.bufs());
        let scope = scratch.batch();
        let x = ActBuf::dense(scope.import_copy(sin_buf.as_buf_ref()));
        let h1 = biased(&scope, bp, x, &w1, 1, dim, 256, Site::Adaln)?;
        let s1 = alloc_act(&scope, bp, 1, dim)?;
        scope.dispatch_op::<SiluF32>(&bp.silu, &[h1.data], s1.data)?;
        let tb = biased(&scope, bp, s1, &w2, 1, dim, dim, Site::Adaln)?;
        let dst = scope.import_copy(temb.as_buf_ref());
        scope.copy_buffer_to_buffer(tb.data, 0, dst, 0, bp.act_bytes(dim))?;
        let sdst = scope.import_copy(silu_temb.as_buf_ref());
        scope.dispatch_op::<SiluF32>(&bp.silu, &[tb.data], sdst)?;
        scope.submit_void().await?;
        Ok((temb, silu_temb))
    }

    /// t2i forward: one image grid `(frame, gh, gw)`, velocity over all tokens.
    /// `img_tokens` = packed latents `[frame*gh*gw, 64]`; `txt_embeds` = encoder
    /// hidden `[txt_seq, 3584]`. bf16 acts throughout.
    #[allow(clippy::too_many_arguments)]
    pub async fn forward<S: WeightSource>(
        &self,
        backend: &WgpuBackend,
        pipelines: &QwenImageDitPipelines,
        residency: &WeightResidency<S>,
        scratch: &Workspace<WgpuBackend>,
        handles: &DitHandles,
        img_tokens: &[f32],
        txt_embeds: &[f32],
        timestep: f32,
        frame: usize,
        gh: usize,
        gw: usize,
        taps: Option<&mut DitTaps>,
    ) -> Result<DitOutput, DitError<S::Error>> {
        self.forward_multi(
            backend,
            pipelines,
            residency,
            scratch,
            handles,
            img_tokens,
            txt_embeds,
            timestep,
            &[(frame, gh, gw)],
            frame * gh * gw,
            taps,
        )
        .await
    }

    /// EDIT forward: the image stream is the concatenation of one or more grids
    /// `[(frame, h, w), ...]` (noise first, then each reference image), and the
    /// per-grid RoPE freqs are concatenated in the SAME order (`QwenEmbedRope`).
    /// `img_tokens` packs all grids' tokens in order. Only the first
    /// `velocity_tokens` patch tokens (the noise span) are returned as velocity;
    /// the reference tail is computed (it conditions attention) then dropped.
    #[allow(clippy::too_many_arguments)]
    pub async fn forward_multi<S: WeightSource>(
        &self,
        backend: &WgpuBackend,
        pipelines: &QwenImageDitPipelines,
        residency: &WeightResidency<S>,
        scratch: &Workspace<WgpuBackend>,
        handles: &DitHandles,
        img_tokens: &[f32],
        txt_embeds: &[f32],
        timestep: f32,
        grids: &[(usize, usize, usize)],
        velocity_tokens: usize,
        mut taps: Option<&mut DitTaps>,
    ) -> Result<DitOutput, DitError<S::Error>> {
        let bp = &pipelines.block;
        let dim = config::DIM as u32;
        let jd = config::JOINT_ATTENTION_DIM as u32;
        let inch = config::IN_CHANNELS as u32;
        let eps = config::NORM_EPS;
        let img_seq = grids.iter().map(|&(f, h, w)| f * h * w).sum::<usize>() as u32;
        assert!(
            velocity_tokens as u32 <= img_seq,
            "velocity_tokens {velocity_tokens} exceeds img_seq {img_seq}"
        );
        assert_eq!(
            txt_embeds.len() % config::JOINT_ATTENTION_DIM,
            0,
            "txt embeds not a multiple of joint dim"
        );
        let txt_seq = (txt_embeds.len() / config::JOINT_ATTENTION_DIM) as u32;
        assert_eq!(img_tokens.len(), (img_seq as usize) * config::IN_CHANNELS);

        let img_buf = upload_act(scratch, backend, bp, img_tokens, img_seq * inch)?;
        let txt_buf = upload_act(scratch, backend, bp, txt_embeds, txt_seq * jd)?;

        // --- img_in: [img_seq, 64] -> [img_seq, dim] (bf16 adaln site) ---
        let hs = scratch.alloc(bp.act_bytes(img_seq * dim))?;
        {
            let v = handles.top.img_in.acquire(residency, backend).await?;
            let w = v.bufs();
            let scope = scratch.batch();
            let x = ActBuf::dense(scope.import_copy(img_buf.as_buf_ref()));
            let o = biased(&scope, bp, x, &w, img_seq, dim, inch, Site::Adaln)?;
            let dst = scope.import_copy(hs.as_buf_ref());
            scope.copy_buffer_to_buffer(o.data, 0, dst, 0, bp.act_bytes(img_seq * dim))?;
            scope.submit_void().await?;
        }

        // --- eh: txt_in(rmsnorm(txt_norm, eh)) -> [txt_seq, dim] ---
        let eh = scratch.alloc(bp.act_bytes(txt_seq * dim))?;
        {
            let nv = residency.acquire(handles.top.txt_norm, backend).await?;
            let tv = handles.top.txt_in.acquire(residency, backend).await?;
            let w = tv.bufs();
            let scope = scratch.batch();
            let x = ActBuf::dense(scope.import_copy(txt_buf.as_buf_ref()));
            let nw = scope.import_copy(nv.buf());
            let normed = alloc_act(&scope, bp, txt_seq, jd)?;
            op_rmsnorm(&scope, bp, x, nw, normed, txt_seq, jd, eps)?;
            let o = biased(&scope, bp, normed, &w, txt_seq, dim, jd, Site::Adaln)?;
            let dst = scope.import_copy(eh.as_buf_ref());
            scope.copy_buffer_to_buffer(o.data, 0, dst, 0, bp.act_bytes(txt_seq * dim))?;
            scope.submit_void().await?;
        }

        // --- silu_temb = SiLU(temb(t)), shared by every block + the final layer ---
        let (temb, silu_temb) =
            Self::silu_temb(backend, bp, residency, scratch, handles, timestep).await?;
        if let Some(t) = taps.as_deref_mut() {
            let bytes = backend.read_buffer(temb.id(), 0, bp.act_bytes(dim)).await?;
            t.temb = Some(seq::act_readback_to_f32(bp.act_dtype, &bytes, dim as usize));
        }
        // zero_cond_t (Edit-2511): the reference rows `[velocity_tokens, ..)`
        // of the image stream are modulated by temb(t=0) (diffusers
        // `modulate_index` / ComfyUI `index_timestep_zero`); the text stream
        // and the final layer keep temb(t).
        let split = velocity_tokens as u32;
        let silu_ref = if self.zero_cond_t && split < img_seq {
            Some(
                Self::silu_temb(backend, bp, residency, scratch, handles, 0.0)
                    .await?
                    .1,
            )
        } else {
            None
        };

        // --- rope freqs + (unused) joint mask ---
        let vid = self.rope.vid_freqs_multi(grids);
        let txt = self.rope.txt_freqs_multi(grids, txt_seq as usize);
        let vid_buf = upload_freqs(scratch, backend, bp, &vid)?;
        let txt_buf = upload_freqs(scratch, backend, bp, &txt)?;
        let joint = txt_seq + img_seq;
        // has_mask=0 (full non-causal joint attention); content irrelevant, only
        // the binding size matters. Round the element count up so the byte size
        // is 4-aligned (storage-buffer binding requirement) -- `joint*joint` is
        // odd when both spans are odd (the edit path's noise+ref+text concat).
        let mask_elems = (joint * joint).next_multiple_of(2).max(16);
        let mask_buf = scratch.alloc(bp.act_bytes(mask_elems))?;

        // --- block loop (ping-pong both streams) ---
        // Prefetch the next block's weights concurrently with the current
        // block's FINAL submit (the FFN/`block_mlp` half), mirroring the
        // text encoder. The DiT streams the 20B weights (only a few blocks
        // resident at a 4-6G budget), so each block pages ~350MB in on
        // acquire; overlapping that acquire with the GPU-busy MLP submit
        // hides the upload latency and removes the per-block GPU-idle gap
        // (the "sawtooth"). `BlockViews::acquire` only briefly holds the
        // residency mutex (to read meta / claim a recycle victim) and then
        // streams into GPU buffers, so a `futures::join!` with the submit
        // genuinely overlaps. Purely moves WHEN the acquire happens; the
        // math, the 2-submit TDR split, and the output are unchanged.
        let mut hs_cur = hs;
        let mut eh_cur = eh;
        let blk_dbg = std::env::var_os("THINFER_VAE_MEM").is_some();
        let mut pending: Option<BlockViews<'_>> = if handles.blocks.is_empty() {
            None
        } else {
            Some(
                BlockViews::acquire(&handles.blocks[0], residency, backend)
                    .instrument(
                        tracing::debug_span!(target: PHASE, "qwen_image.dit.acquire", idx = 0),
                    )
                    .await?,
            )
        };
        for idx in 0..handles.blocks.len() {
            let _g = trace::scope!(format!("qwen_image.dit.block.{idx}")).entered();
            let views = pending.take().expect("pending acquire missing");
            let bufs = views.bufs(pipelines);
            let t_blk = std::time::Instant::now();
            let hs_nxt = scratch.alloc(bp.act_bytes(img_seq * dim))?;
            let eh_nxt = scratch.alloc(bp.act_bytes(txt_seq * dim))?;
            // Two-submit TDR split (attention, then FFN), carried across the
            // boundary by hs1/eh1. At the edit path's large joint one whole-block
            // submit runs ~2s and intermittently trips the Windows TDR watchdog;
            // two halves stay comfortably under it. The FFN recomputes its
            // modulation signals from `silu_temb`, so nothing else crosses the
            // boundary -- numerically identical to one submit.
            //
            // Kept split UNCONDITIONALLY, including under the f16 fast-attention
            // path. Merging the two halves into one submit was re-measured AFTER
            // the mixed-precision SDPA halved per-block compute (`dit_perf`, 1 MP
            // joint ~5.7k): wall was unchanged (the inter-submit bubble is within
            // run-to-run noise -- the block stays ~98% GPU-compute-bound) while a
            // single scope holds BOTH halves' transients at once (bump-allocated,
            // no intra-scope reuse), raising the workspace peak ~+350 MiB. At a
            // 4 GiB budget that extra peak forces DiT-block eviction (the
            // resident set shrank), a net loss on the 8 GiB ceiling. Same verdict
            // as the pre-SDPA pipelining attempt: the bubble is not worth the
            // VRAM. Do not merge without a workspace-peak budget that absorbs it.
            let hs1_carry = scratch.alloc(bp.act_bytes(img_seq * dim))?;
            let eh1_carry = scratch.alloc(bp.act_bytes(txt_seq * dim))?;
            {
                let scope = scratch.batch();
                block_attn(
                    &scope,
                    pipelines,
                    scope.import_copy(hs_cur.as_buf_ref()),
                    scope.import_copy(eh_cur.as_buf_ref()),
                    scope.import_copy(silu_temb.as_buf_ref()),
                    silu_ref.as_ref().map(|b| scope.import_copy(b.as_buf_ref())),
                    split,
                    scope.import_copy(vid_buf.as_buf_ref()),
                    scope.import_copy(txt_buf.as_buf_ref()),
                    scope.import_copy(mask_buf.as_buf_ref()),
                    img_seq,
                    txt_seq,
                    scope.import_copy(hs1_carry.as_buf_ref()),
                    scope.import_copy(eh1_carry.as_buf_ref()),
                    &bufs,
                )?;
                scope.submit_void().await?;
            }
            // Final submit (FFN half) overlapped with the next block's acquire
            // (mirrors the encoder; hides the ~350MB/block page-in).
            {
                let scope = scratch.batch();
                block_mlp(
                    &scope,
                    pipelines,
                    scope.import_copy(hs1_carry.as_buf_ref()),
                    scope.import_copy(eh1_carry.as_buf_ref()),
                    scope.import_copy(silu_temb.as_buf_ref()),
                    silu_ref.as_ref().map(|b| scope.import_copy(b.as_buf_ref())),
                    split,
                    img_seq,
                    txt_seq,
                    scope.import_copy(hs_nxt.as_buf_ref()),
                    scope.import_copy(eh_nxt.as_buf_ref()),
                    &bufs,
                )?;
                let next_idx = idx + 1;
                let next_acquire = async {
                    match handles.blocks.get(next_idx) {
                        Some(h) => Ok::<_, ResidencyError<S::Error, WgpuError>>(Some(
                            BlockViews::acquire(h, residency, backend).await?,
                        )),
                        None => Ok(None),
                    }
                };
                let submit_fut = scope
                    .submit_void()
                    .instrument(tracing::debug_span!(target: PHASE, "qwen_image.dit.submit", idx));
                let (submit_res, next_res) = futures::join!(submit_fut, next_acquire);
                submit_res?;
                pending = next_res?;
            }
            if blk_dbg {
                use thinfer_core::backend::Backend;
                let mem = backend.mem_account();
                eprintln!(
                    "[vae_mem]   dit.block.{} {}ms vram={}MiB (W={}MiB Ws={}MiB) joint={}",
                    idx,
                    t_blk.elapsed().as_millis(),
                    mem.vram_total_current() / (1024 * 1024),
                    mem.vram_current(thinfer_core::mem::VramCategory::Weights) / (1024 * 1024),
                    mem.vram_current(thinfer_core::mem::VramCategory::Workspace) / (1024 * 1024),
                    txt_seq + img_seq,
                );
            }
            drop(views);
            if idx == 0
                && let Some(t) = taps.as_deref_mut()
            {
                let ib = backend
                    .read_buffer(hs_nxt.id(), 0, bp.act_bytes(img_seq * dim))
                    .await?;
                t.block0_img = Some(seq::act_readback_to_f32(
                    bp.act_dtype,
                    &ib,
                    (img_seq * dim) as usize,
                ));
                let tb = backend
                    .read_buffer(eh_nxt.id(), 0, bp.act_bytes(txt_seq * dim))
                    .await?;
                t.block0_txt = Some(seq::act_readback_to_f32(
                    bp.act_dtype,
                    &tb,
                    (txt_seq * dim) as usize,
                ));
            }
            hs_cur = hs_nxt;
            eh_cur = eh_nxt;
        }

        // --- norm_out (AdaLayerNormContinuous: scale,shift) + proj_out ---
        let velocity = {
            let nv = handles.top.norm_out.acquire(residency, backend).await?;
            let pv = handles.top.proj_out.acquire(residency, backend).await?;
            let (nw, pw) = (nv.bufs(), pv.bufs());
            let vel_buf = scratch.alloc(bp.act_bytes(img_seq * inch))?;
            {
                let scope = scratch.batch();
                let silu = ActBuf::dense(scope.import_copy(silu_temb.as_buf_ref()));
                // emb = linear(SiLU(temb)) -> [1, 2*dim]; scale = emb[..dim], shift = emb[dim..]
                let emb = biased(&scope, bp, silu, &nw, 1, 2 * dim, dim, Site::Adaln)?;
                let scale = mod_signal(&scope, bp, emb.data, 0, dim)?;
                let shift = mod_signal(&scope, bp, emb.data, 1, dim)?;
                let normed = alloc_act(&scope, bp, img_seq, dim)?;
                let hsx = ActBuf::dense(scope.import_copy(hs_cur.as_buf_ref()));
                op_layernorm(&scope, bp, hsx, normed, img_seq, dim, eps)?;
                let modded = alloc_act(&scope, bp, img_seq, dim)?;
                op_modulate(&scope, bp, normed, scale, shift, modded, img_seq, dim)?;
                let vel = biased(&scope, bp, modded, &pw, img_seq, inch, dim, Site::Adaln)?;
                let dst = scope.import_copy(vel_buf.as_buf_ref());
                scope.copy_buffer_to_buffer(vel.data, 0, dst, 0, bp.act_bytes(img_seq * inch))?;
                scope.submit_void().await?;
            }
            // Velocity = the noise span only (first velocity_tokens tokens, which
            // lead the concatenated image stream); the reference tail is dropped.
            let vel_n = velocity_tokens as u32 * inch;
            let bytes = backend
                .read_buffer(vel_buf.id(), 0, bp.act_bytes(vel_n))
                .await?;
            seq::act_readback_to_f32(bp.act_dtype, &bytes, vel_n as usize)
        };
        Ok(DitOutput {
            velocity,
            img_seq: velocity_tokens,
        })
    }
}

impl Default for QwenImageDit {
    fn default() -> Self {
        Self::new(false)
    }
}
