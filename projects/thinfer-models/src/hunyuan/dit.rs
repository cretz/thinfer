//! HunyuanVideo 1.5 dual-stream MMDiT (T2V, lightx2v 4-step distill). Ground
//! truth: `hunyuanvideo_1_5_transformer.py::{MMDoubleStreamBlock,
//! HunyuanVideo_1_5_DiffusionTransformer.forward}` (T2V path: no single-stream
//! blocks, no byt5/vision streams, meanflow off).
//!
//! Forward: `img_in` (1x1x1 Conv3d 65->2048 as a flattened linear; T2V input is
//! `[noise32 | 0 | 0]`) -> tokens `(t,h,w)`; `vec = time_in(t)`; `txt =
//! cond_type[0] + SingleTokenRefiner(text, t)` ([`HunyuanRefiner`]); then 54
//! dual-stream blocks (param-less LayerNorm -> 6-chunk adaLN `modulate` ->
//! separate q/k/v, per-head qk-RMSNorm, interleaved RoPE3D on IMG q/k only,
//! joint SDPA over `[img ; txt]`, gelu-tanh FFN) -> `final_layer`
//! (adaLN(shift,scale) + linear 2048->32). Output = velocity `[n_img, 32]` in
//! `(t,h,w)` token order.
//!
//! Correctness-first: f32 acts, bf16 weights (fp16 DiT narrowed on upload); the
//! parity band absorbs the narrowing. Weights stream per-block (the 16.7GB DiT
//! never co-resides), mirroring `qwen_image/dit.rs`'s ping-pong.

pub mod ar;

use thinfer_core::backend::{Backend, BufRef, WgpuBackend, WgpuError, WgpuPipeline};
use thinfer_core::ops::{
    ActDtype, BcastAddF32, BcastFmaF32, BcastModulateF32, GeluF32, LayerNormF32, MatMulF32,
    RmsNormF32, SiluF32, WeightDtype, WgslConfig,
};
use thinfer_core::residency::{GpuView, ResidencyError, WeightHandle, WeightResidency};
use thinfer_core::weight::{WeightId, WeightSource};
use thinfer_core::workspace::{BatchBuf, BatchScope, Workspace, WsBuf};

use crate::common::block::{
    ActBuf, Block, BlockPipelines, BlockWgslConfigs, CoopmatStep, DequantStep, PreparedWeight,
    op_rope, op_sdpa_f16_win, quantize_act_paired,
};
use crate::common::loader::{
    LoadError, register_linear, register_linear_flatten, register_linear_transcode,
    register_passthrough,
};
use crate::common::rope_embedder::RopeEmbedder;
use crate::common::seq::{act_readback_to_f32, act_upload_bytes, freqs_upload_bytes};
use crate::hunyuan::config::dit as cfg;
use crate::hunyuan::refiner::{HunyuanRefiner, timestep_sinusoid};

const FREQ_DIM: usize = cfg::TIME_FREQ_DIM; // 256
const LN_EPS: f32 = 1e-6;

// ============================================================================
// Weight ids / handles / bufs
// ============================================================================

/// `weight [out,in]` (registered transposed to `[in,out]`) + `bias [out]`.
#[derive(Clone)]
struct LinW {
    weight: WeightId,
    bias: WeightId,
}

fn lin(prefix: &str) -> LinW {
    LinW {
        weight: WeightId(format!("{prefix}.weight")),
        bias: WeightId(format!("{prefix}.bias")),
    }
}

struct BlockW {
    img_mod: LinW,
    txt_mod: LinW,
    img_q: LinW,
    img_k: LinW,
    img_v: LinW,
    txt_q: LinW,
    txt_k: LinW,
    txt_v: LinW,
    img_qn: WeightId,
    img_kn: WeightId,
    txt_qn: WeightId,
    txt_kn: WeightId,
    img_proj: LinW,
    txt_proj: LinW,
    img_fc1: LinW,
    img_fc2: LinW,
    txt_fc1: LinW,
    txt_fc2: LinW,
}

impl BlockW {
    fn new(i: usize) -> Self {
        let p = format!("double_blocks.{i}");
        let norm = |n: &str| WeightId(format!("{p}.{n}.weight"));
        Self {
            img_mod: lin(&format!("{p}.img_mod.linear")),
            txt_mod: lin(&format!("{p}.txt_mod.linear")),
            img_q: lin(&format!("{p}.img_attn_q")),
            img_k: lin(&format!("{p}.img_attn_k")),
            img_v: lin(&format!("{p}.img_attn_v")),
            txt_q: lin(&format!("{p}.txt_attn_q")),
            txt_k: lin(&format!("{p}.txt_attn_k")),
            txt_v: lin(&format!("{p}.txt_attn_v")),
            img_qn: norm("img_attn_q_norm"),
            img_kn: norm("img_attn_k_norm"),
            txt_qn: norm("txt_attn_q_norm"),
            txt_kn: norm("txt_attn_k_norm"),
            img_proj: lin(&format!("{p}.img_attn_proj")),
            txt_proj: lin(&format!("{p}.txt_attn_proj")),
            img_fc1: lin(&format!("{p}.img_mlp.fc1")),
            img_fc2: lin(&format!("{p}.img_mlp.fc2")),
            txt_fc1: lin(&format!("{p}.txt_mlp.fc1")),
            txt_fc2: lin(&format!("{p}.txt_mlp.fc2")),
        }
    }
}

struct TopW {
    img_in: LinW,
    time0: LinW,
    time2: LinW,
    cond_type: WeightId,
    final_adaln: LinW,
    final_lin: LinW,
}

impl TopW {
    fn new() -> Self {
        Self {
            img_in: lin("img_in.proj"),
            time0: lin("time_in.mlp.0"),
            time2: lin("time_in.mlp.2"),
            cond_type: WeightId("cond_type_embedding.weight".into()),
            final_adaln: lin("final_layer.adaLN_modulation.1"),
            final_lin: lin("final_layer.linear"),
        }
    }
}

#[derive(Clone, Copy)]
struct LinH {
    weight: WeightHandle,
    bias: WeightHandle,
}

struct BlockH {
    img_mod: LinH,
    txt_mod: LinH,
    img_q: LinH,
    img_k: LinH,
    img_v: LinH,
    txt_q: LinH,
    txt_k: LinH,
    txt_v: LinH,
    img_qn: WeightHandle,
    img_kn: WeightHandle,
    txt_qn: WeightHandle,
    txt_kn: WeightHandle,
    img_proj: LinH,
    txt_proj: LinH,
    img_fc1: LinH,
    img_fc2: LinH,
    txt_fc1: LinH,
    txt_fc2: LinH,
}

struct TopH {
    img_in: LinH,
    time0: LinH,
    time2: LinH,
    cond_type: WeightHandle,
    final_adaln: LinH,
    final_lin: LinH,
}

struct DitH {
    top: TopH,
    blocks: Vec<BlockH>,
}

fn reg_lin<S: WeightSource>(res: &WeightResidency<S>, w: &LinW) -> Result<LinH, LoadError> {
    Ok(LinH {
        weight: register_linear(res, &w.weight)?,
        bias: register_passthrough(res, &w.bias)?,
    })
}

/// Linear whose weight transcodes to `q8` (`Some` on the i8 DP4A path) so the
/// matmul site can read it as Q8_0; `None` keeps the dense bf16 load. Bias stays
/// passthrough bf16. Used for the self-attn q/k/v + ffn-up (`fc1`) weights (i8
/// DP4A sites) and the img_mod/txt_mod linears (dequant-once adaln site), whose
/// matmul sites pin `Quant` weight_dtype under i8.
fn reg_lin_q8<S: WeightSource>(
    res: &WeightResidency<S>,
    w: &LinW,
    q8: Option<thinfer_core::quant::QuantKind>,
) -> Result<LinH, LoadError> {
    Ok(LinH {
        weight: register_linear_transcode(res, &w.weight, None, q8)?,
        bias: register_passthrough(res, &w.bias)?,
    })
}

impl DitH {
    fn register<S: WeightSource>(
        res: &WeightResidency<S>,
        i8: bool,
        coopmat: bool,
    ) -> Result<Self, LoadError> {
        // i8 DP4A weights: self-attn q/k/v + ffn-up (fc1), matching the Quant
        // sites in `hunyuan_block_cfgs`. None = dense bf16 (parity / --no-i8).
        let q8 = i8.then_some(thinfer_core::quant::QuantKind::Q8_0);
        // ffn-down (fc2) transcodes to Q8_0 only for the coopmat path (the coopmat
        // matmul dequants Quant->f16); must match `hunyuan_block_cfgs(_, _, coopmat)`.
        let q8c = coopmat.then_some(thinfer_core::quant::QuantKind::Q8_0);
        let tw = TopW::new();
        let top = TopH {
            img_in: LinH {
                weight: register_linear_flatten(res, &tw.img_in.weight)?,
                bias: register_passthrough(res, &tw.img_in.bias)?,
            },
            time0: reg_lin(res, &tw.time0)?,
            time2: reg_lin(res, &tw.time2)?,
            cond_type: register_passthrough(res, &tw.cond_type)?,
            final_adaln: reg_lin(res, &tw.final_adaln)?,
            final_lin: reg_lin(res, &tw.final_lin)?,
        };
        let mut blocks = Vec::with_capacity(cfg::DOUBLE_BLOCKS);
        for i in 0..cfg::DOUBLE_BLOCKS {
            let w = BlockW::new(i);
            blocks.push(BlockH {
                // Mod linears: Q8_0 weight-only under i8, matching the Quant
                // adaln site in `hunyuan_block_cfgs` (dequant-once dense path;
                // no i8 acts). None keeps them dense bf16 (parity / --no-i8).
                // Mod linears stay dense bf16 while the adaln Q8 path is
                // disabled (see the cfgs note in `hunyuan_block_cfgs`); the
                // registration and pipeline MUST flip together.
                img_mod: reg_lin_q8(res, &w.img_mod, None)?,
                txt_mod: reg_lin_q8(res, &w.txt_mod, None)?,
                img_q: reg_lin_q8(res, &w.img_q, q8)?,
                img_k: reg_lin_q8(res, &w.img_k, q8)?,
                img_v: reg_lin_q8(res, &w.img_v, q8)?,
                txt_q: reg_lin_q8(res, &w.txt_q, q8)?,
                txt_k: reg_lin_q8(res, &w.txt_k, q8)?,
                txt_v: reg_lin_q8(res, &w.txt_v, q8)?,
                img_qn: register_passthrough(res, &w.img_qn)?,
                img_kn: register_passthrough(res, &w.img_kn)?,
                txt_qn: register_passthrough(res, &w.txt_qn)?,
                txt_kn: register_passthrough(res, &w.txt_kn)?,
                img_proj: reg_lin_q8(res, &w.img_proj, q8c)?,
                txt_proj: reg_lin_q8(res, &w.txt_proj, q8c)?,
                img_fc1: reg_lin_q8(res, &w.img_fc1, q8)?,
                img_fc2: reg_lin_q8(res, &w.img_fc2, q8c)?,
                txt_fc1: reg_lin_q8(res, &w.txt_fc1, q8)?,
                txt_fc2: reg_lin_q8(res, &w.txt_fc2, q8c)?,
            });
        }
        Ok(Self { top, blocks })
    }
}

#[derive(Clone, Copy)]
struct LinBufs {
    weight: BufRef,
    bias: BufRef,
}

struct BlockBufs {
    img_mod: LinBufs,
    txt_mod: LinBufs,
    img_q: LinBufs,
    img_k: LinBufs,
    img_v: LinBufs,
    txt_q: LinBufs,
    txt_k: LinBufs,
    txt_v: LinBufs,
    img_qn: BufRef,
    img_kn: BufRef,
    txt_qn: BufRef,
    txt_kn: BufRef,
    img_proj: LinBufs,
    txt_proj: LinBufs,
    img_fc1: LinBufs,
    img_fc2: LinBufs,
    txt_fc1: LinBufs,
    txt_fc2: LinBufs,
}

struct TopBufs {
    img_in: LinBufs,
    time0: LinBufs,
    time2: LinBufs,
    cond_type: BufRef,
    final_adaln: LinBufs,
    final_lin: LinBufs,
}

// ============================================================================
// Pipelines
// ============================================================================

/// The shared [`BlockPipelines`] kernel set, configured for the Hunyuan
/// dual-stream block. Adopting it (vs a hand-rolled op set) gives the whole
/// fast arsenal: bf16 acts, the f16 subgroup SDPA (`op_sdpa_f16`), its
/// temporal-windowed twin, the i8 DP4A matmul sites, and the bf16<->f16 casts.
/// The Hunyuan block does NOT use [`crate::common::block::Block::forward`]
/// (that is the single-stream Z-Image/Wan shape); it composes the pipelines +
/// dispatch helpers directly for the dual-stream (separate q/k/v, joint SDPA,
/// 6-chunk dual adaLN) structure.
pub struct HunyuanDitPipelines {
    pub bp: BlockPipelines,
}

impl HunyuanDitPipelines {
    pub fn act(&self) -> ActDtype {
        self.bp.act_dtype
    }

    fn act_size(&self) -> u64 {
        self.bp.act_dtype.bytes_per_elem()
    }

    pub async fn compile_with(
        backend: &WgpuBackend,
        act: ActDtype,
        i8: bool,
    ) -> Result<Self, WgpuError> {
        // Coopmat (tensor cores) on ffn_down: only when the device exposes a usable
        // config. `BlockPipelines::compile` no-ops the coopmat step otherwise, so
        // gating the Quant ffn_down weight on the same predicate keeps the dense
        // bf16 fallback (web / no-coopmat) instead of a quality-losing i8 ffn_down.
        let coopmat = backend.coopmat().is_some();
        let bp = BlockPipelines::compile(backend, &hunyuan_block_cfgs(act, i8, coopmat)).await?;
        Ok(Self { bp })
    }
}

/// `BlockWgslConfigs` for the Hunyuan DiT block. 4a baseline = every matmul
/// site at bf16 weights with bf16 acts + mixed-precision f16 SDPA (`fast_sdpa`).
/// 4b will flip the normed A-side sites (qkv self-projection, ffn-up) to a
/// Quant weight encoding for the i8 DP4A path; the dispatch wiring already
/// routes through [`Block::dispatch_matmul_site`], so that is a per-site
/// weight-dtype change here plus the loader transcode.
fn hunyuan_block_cfgs(act: ActDtype, i8: bool, coopmat: bool) -> BlockWgslConfigs {
    let ops = WgslConfig {
        bf16_quant_writes: false,
        act_dtype: act,
        weight_dtype: WeightDtype::Bf16,
    };
    let mut cfgs = BlockWgslConfigs::uniform(ops);
    // Mixed-precision f16 self-attention: a no-op unless acts are bf16 (an F32
    // parity run falls back to the dense chunked SdpaF32 path). The joint
    // attention Q/K/V are post-qknorm/rope and O(1), so the f16 cast is safe.
    cfgs.fast_sdpa = true;
    let q8 = WeightDtype::Quant(thinfer_core::quant::QuantKind::Q8_0);
    // 4b: i8 DP4A on the two normed-A-side sites (self-attn q/k/v projection +
    // ffn-up). Both A-sides are LayerNorm+modulate outputs (no massive-activation
    // outlier rows), so per-k32 i8 activation quant is quality-neutral. proj
    // (attn-out) + ffn-down (gelu product) stay bf16 on the i8 path -- their
    // A-sides carry outliers that i8 acts would crush. The loader must transcode
    // the matching weights to Q8_0 (see `DitH::register`); pipeline + weight
    // encoding MUST agree or the matmul reads garbage.
    if i8 {
        cfgs.matmul_qkv_self.weight_dtype = q8;
        cfgs.matmul_ffn_up.weight_dtype = q8;
        // Modulation linears (img_mod/txt_mod) as Q8_0 weight-only via the
        // adaln site: DISABLED 2026-07-03 pending a fix -- dit_parity (taps
        // path) device-panics in wgpu with the adaln dequant wiring while the
        // tapless t2v_e2e passes. Re-enable by restoring
        // `cfgs.matmul_adaln.weight_dtype = q8;` here AND flipping
        // `MOD_LINEARS_Q8` in `DitH::register`; gate with dit_parity
        // (default + THINFER_DIT_TILE_ROWS=8) before shipping.
    }
    // ffn_down is the #1 DiT matmul (measured) and outlier-bound (gelu product), so
    // it is NOT i8-eligible -- but coopmat (tensor cores) only casts the A-side to
    // f16 (no per-row i8 scale), which IS outlier-safe (Wan precedent). The coopmat
    // matmul consumes a Quant->f16-dequanted weight, so pin ffn_down Quant AND opt
    // it into the coopmat site. Gated on `coopmat`: a no-coopmat device keeps
    // ffn_down dense bf16 (a Quant weight there would fall back to the bad i8 path).
    if coopmat {
        cfgs.matmul_ffn_down.weight_dtype = q8;
        cfgs.coopmat_acts.ffn_down = true;
        // proj (attention-output projection) is the next-biggest dense matmul; its
        // A-side is the bounded post-SDPA output (outlier-safe for the f16 cast,
        // Wan precedent), so coopmat it too.
        cfgs.matmul_proj.weight_dtype = q8;
        cfgs.coopmat_acts.proj = true;
    }
    cfgs
}

/// Matmul call sites in the Hunyuan block. Each maps to a `BlockPipelines`
/// per-site pipeline group so [`Block::dispatch_matmul_site`] picks the right
/// (dense / dequant-once / i8 DP4A) path. The separate q/k/v projections share
/// the `QkvSelf` site (normed A-side, i8-eligible in 4b); the two MLP linears
/// use `FfnUp`/`FfnDown`; attention output uses `Proj`. The per-block
/// modulation linears (img_mod/txt_mod) use `Mod` (the adaln site: Q8_0
/// weight + dequant-once dense matmul under i8, dense bf16 otherwise).
/// Module-level dense linears (img_in, time, final) stay bf16 via `Module`.
#[derive(Clone, Copy)]
enum Site {
    QkvSelf,
    Proj,
    FfnUp,
    FfnDown,
    Mod,
    Module,
}

type SitePipes<'a> = (
    Option<&'a CoopmatStep>,
    Option<&'a WgpuPipeline>,
    Option<&'a DequantStep>,
    Option<&'a DequantStep>,
    &'a WgpuPipeline,
    &'a MatMulF32,
);

fn site_pipes(bp: &BlockPipelines, site: Site) -> SitePipes<'_> {
    match site {
        Site::QkvSelf => (
            bp.coopmat_qkv.as_ref(),
            bp.matmul_i8_qkv_self.as_ref(),
            bp.dequant_i8_qkv_self.as_ref(),
            bp.dequant_qkv_self.as_ref(),
            &bp.matmul_qkv_self,
            &bp.matmuls.qkv_self,
        ),
        Site::Proj => (
            bp.coopmat_proj.as_ref(),
            bp.matmul_i8_proj.as_ref(),
            bp.dequant_i8_proj.as_ref(),
            bp.dequant_proj.as_ref(),
            &bp.matmul_proj,
            &bp.matmuls.proj,
        ),
        Site::FfnUp => (
            None,
            bp.matmul_i8_ffn_up.as_ref(),
            bp.dequant_i8_ffn_up.as_ref(),
            bp.dequant_ffn_up.as_ref(),
            &bp.matmul_ffn_up,
            &bp.matmuls.ffn_up,
        ),
        Site::FfnDown => (
            bp.coopmat_ffn_down.as_ref(),
            bp.matmul_i8_ffn_down.as_ref(),
            bp.dequant_i8_ffn_down.as_ref(),
            bp.dequant_ffn_down.as_ref(),
            &bp.matmul_ffn_down,
            &bp.matmuls.ffn_down,
        ),
        // Modulation linears: dequant-once dense only (dequant_adaln is Some
        // iff the i8 config pinned the adaln weight Q8_0). Deliberately no
        // i8/coopmat entries: the mod outputs gate the whole residual stream
        // and M=1, so only the weight storage quantizes.
        Site::Mod => (
            None,
            None,
            None,
            bp.dequant_adaln.as_ref(),
            &bp.matmul_adaln,
            &bp.matmuls.adaln,
        ),
        // Module-level dense bf16 linears: no quant/i8/coopmat routing.
        Site::Module => (
            None,
            None,
            None,
            None,
            &bp.matmul_module,
            &bp.matmuls.module,
        ),
    }
}

// ============================================================================
// Op wrappers (compose BlockPipelines kernels for the dual-stream block)
// ============================================================================

/// `x[rows,k] @ wᵀ + bias -> [rows,n]`, routed through the per-`site` matmul
/// dispatch (dense bf16 in 4a; the i8 DP4A / dequant-once paths light up once a
/// site's weight is pinned Quant).
#[allow(clippy::too_many_arguments)]
fn linear<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    bp: &BlockPipelines,
    x: BatchBuf<'w>,
    w: &LinBufs,
    rows: u32,
    n: u32,
    k: u32,
    site: Site,
) -> Result<BatchBuf<'w>, WgpuError> {
    linear_a(scope, bp, ActBuf::dense(x), w, rows, n, k, site, None)
}

/// As [`linear`] but takes the A-side as an [`ActBuf`], so a pre-quantized
/// paired i8 A-side (see [`quantize_act_paired`]) can be shared across the
/// separate q/k/v projections instead of each one re-running `act_quant`, and
/// an optional per-block [`PreparedWeight`] (see [`HunyuanPreparedWeights`])
/// so the tiled path skips the per-tile weight dequant inside the dispatch.
#[allow(clippy::too_many_arguments)]
fn linear_a<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    bp: &BlockPipelines,
    a: ActBuf<'w>,
    w: &LinBufs,
    rows: u32,
    n: u32,
    k: u32,
    site: Site,
    prepared: Option<PreparedWeight>,
) -> Result<BatchBuf<'w>, WgpuError> {
    let pre = scope.alloc(bp.act_bytes(rows * n))?;
    let dims = scope.u32x4_uniform(rows, n, k, 0)?;
    let wv = scope.import_copy(w.weight);
    let (cm, mi8, dqi8, dqd, mmpl, mmop) = site_pipes(bp, site);
    Block::dispatch_matmul_site_coopmat(
        scope, bp, a, wv, pre, dims, prepared, cm, mi8, dqi8, dqd, mmpl, mmop, rows, n, k,
    )?;
    let out = scope.alloc(bp.act_bytes(rows * n))?;
    let u = scope.u32x4_uniform(n, 0, 0, 0)?;
    let bv = scope.import_copy(w.bias);
    scope.bcast_add::<BcastAddF32>(&bp.bcast_add, pre, bv, u, out, rows * n)?;
    Ok(out)
}

/// Quantize the shared q/k/v A-side `m` ([rows, dim]) to a paired i8 ActBuf
/// ONCE when the qkv-self site runs the DP4A i8 path, so the three projections
/// reuse it instead of re-quantizing per call (bit-identical). On the dense
/// (`--no-i8-matmul`) path returns the dense ActBuf unchanged.
fn qkv_a_side<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    bp: &BlockPipelines,
    m: BatchBuf<'w>,
    rows: u32,
    dim: u32,
) -> Result<ActBuf<'w>, WgpuError> {
    if bp.matmul_i8_qkv_self.is_some() {
        quantize_act_paired(scope, bp, m, rows, dim)
    } else {
        Ok(ActBuf::dense(m))
    }
}

fn silu<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    bp: &BlockPipelines,
    x: BatchBuf<'w>,
    n: u32,
) -> Result<BatchBuf<'w>, WgpuError> {
    let out = scope.alloc(bp.act_bytes(n))?;
    scope.dispatch_op::<SiluF32>(&bp.silu, &[x], out)?;
    Ok(out)
}

/// Param-less LayerNorm (mean-subtract, eps), then `modulate`: `x*(1+scale)+shift`.
fn norm_modulate<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    bp: &BlockPipelines,
    x: BatchBuf<'w>,
    scale: BatchBuf<'w>,
    shift: BatchBuf<'w>,
    rows: u32,
    dim: u32,
) -> Result<BatchBuf<'w>, WgpuError> {
    let normed = scope.alloc(bp.act_bytes(rows * dim))?;
    let lu = scope.u32x4_uniform(rows, dim, LN_EPS.to_bits(), 0)?;
    scope.layernorm::<LayerNormF32>(&bp.layernorm, x, lu, normed, rows)?;
    let out = scope.alloc(bp.act_bytes(rows * dim))?;
    let mu = scope.u32x4_uniform(dim, 1.0_f32.to_bits(), 0, 0)?;
    scope.bcast_modulate::<BcastModulateF32>(
        &bp.bcast_modulate,
        normed,
        scale,
        shift,
        mu,
        out,
        rows * dim,
    )?;
    Ok(out)
}

/// `x + gate[c] * y` over `dim` (gate `[dim]` broadcast over rows).
fn gate_residual<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    bp: &BlockPipelines,
    x: BatchBuf<'w>,
    gate: BatchBuf<'w>,
    y: BatchBuf<'w>,
    rows: u32,
    dim: u32,
) -> Result<BatchBuf<'w>, WgpuError> {
    let out = scope.alloc(bp.act_bytes(rows * dim))?;
    let u = scope.u32x4_uniform(dim, 0, 0, 0)?;
    scope.bcast_fma::<BcastFmaF32>(&bp.bcast_fma, x, gate, y, u, out, rows * dim)?;
    Ok(out)
}

/// Per-head RMSNorm (affine weight `[head_dim]`, eps) over a `[rows, heads*hd]`
/// activation, then interleaved RoPE if `freqs` is `Some` (img q/k only).
#[allow(clippy::too_many_arguments)]
fn qk_norm_rope<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    bp: &BlockPipelines,
    x: BatchBuf<'w>,
    norm_w: BufRef,
    freqs: Option<BatchBuf<'w>>,
    rows: u32,
    heads: u32,
    hd: u32,
) -> Result<BatchBuf<'w>, WgpuError> {
    let normed = scope.alloc(bp.act_bytes(rows * heads * hd))?;
    let mut ru = [0u8; 16];
    ru[0..4].copy_from_slice(&(rows * heads).to_le_bytes());
    ru[4..8].copy_from_slice(&hd.to_le_bytes());
    ru[8..12].copy_from_slice(&LN_EPS.to_bits().to_le_bytes());
    let ruv = scope.write_uniform(&ru)?;
    let nw = scope.import_copy(norm_w);
    scope.rmsnorm::<RmsNormF32>(&bp.rmsnorm, x, nw, ruv, normed, rows * heads)?;
    match freqs {
        None => Ok(normed),
        Some(f) => {
            let out = scope.alloc(bp.act_bytes(rows * heads * hd))?;
            op_rope(
                scope,
                bp,
                ActBuf::dense(normed),
                f,
                ActBuf::dense(out),
                rows,
                heads,
                hd,
            )?;
            Ok(out)
        }
    }
}

/// Joint multi-head attention over `[joint, HIDDEN]` = `[img(txt_start) ; txt]`.
/// Drives the mixed-precision f16 subgroup SDPA (`op_sdpa_f16_win`) when acts are
/// bf16 (Q/K/V are post-qknorm/rope, so the f16 cast is loss-free), falling back
/// to the dense chunked `SdpaF32` otherwise. `has_mask = 0`: byt5/vision streams
/// are omitted (not masked). When `window > 0` the image queries attend only
/// keys within `±window` latent frames (`period` = tokens/frame) plus ALL text
/// keys, and text queries attend everything -- the O(frames²)→O(frames·window)
/// lever; `window = 0` is full bidirectional attention (bit-identical).
#[allow(clippy::too_many_arguments)]
fn attention<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    bp: &BlockPipelines,
    q: BatchBuf<'w>,
    k: BatchBuf<'w>,
    v: BatchBuf<'w>,
    joint: u32,
    txt_start: u32,
    period: u32,
    window: u32,
    heads: u32,
    hd: u32,
) -> Result<BatchBuf<'w>, WgpuError> {
    let out = scope.alloc(bp.act_bytes(joint * heads * hd))?;
    let mask = scope.write_uniform(&0f32.to_le_bytes())?;
    let scale = 1.0_f32 / (hd as f32).sqrt();
    op_sdpa_f16_win(
        scope,
        bp,
        ActBuf::dense(q),
        ActBuf::dense(k),
        ActBuf::dense(v),
        mask,
        ActBuf::dense(out),
        1,
        joint,
        joint,
        heads,
        heads,
        hd,
        scale,
        0,
        period,
        window,
        txt_start,
    )?;
    Ok(out)
}

/// gelu-tanh MLP `fc1 -> gelu -> fc2`. `p1`/`p2` are the optional per-block
/// prepared fc1/fc2 weights (tiled img pass; `None` keeps the inline dequant).
#[allow(clippy::too_many_arguments)]
fn mlp<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    bp: &BlockPipelines,
    x: BatchBuf<'w>,
    fc1: &LinBufs,
    fc2: &LinBufs,
    rows: u32,
    dim: u32,
    hidden: u32,
    p1: Option<PreparedWeight>,
    p2: Option<PreparedWeight>,
) -> Result<BatchBuf<'w>, WgpuError> {
    let up = linear_a(
        scope,
        bp,
        ActBuf::dense(x),
        fc1,
        rows,
        hidden,
        dim,
        Site::FfnUp,
        p1,
    )?;
    let g = scope.alloc(bp.act_bytes(rows * hidden))?;
    scope.dispatch_op::<GeluF32>(&bp.gelu, &[up], g)?;
    linear_a(
        scope,
        bp,
        ActBuf::dense(g),
        fc2,
        rows,
        dim,
        hidden,
        Site::FfnDown,
        p2,
    )
}

/// Slice modulation signal `k` ([dim]) out of a `[1, 6*dim]` buffer.
fn mod_sig<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    bp: &BlockPipelines,
    src: BatchBuf<'w>,
    k: u32,
    dim: u32,
) -> Result<BatchBuf<'w>, WgpuError> {
    let out = scope.alloc(bp.act_bytes(dim))?;
    let row = bp.act_bytes(dim);
    scope.copy_buffer_to_buffer(src, k as u64 * row, out, 0, row)?;
    Ok(out)
}

// ============================================================================
// Activation tiling (bound the per-block scope peak at high resolution)
// ============================================================================

/// Per-tile img-row target. Above one tile's worth of img tokens the block runs
/// tiled (pass A row-tiled q/k/v -> joint SDPA barrier -> pass B row-tiled
/// proj+FFN) so the heavy `[tile, mlp_hidden]` FFN transients (and the 480p
/// q/k/v/proj buffers) recycle through the pool between submits instead of all
/// co-living in one ~6GB+ scope (which OOMs an 8GB device regardless of the
/// weight budget). At or under one tile (the parity grids) the single-scope path
/// runs unchanged + bit-identical. Env override `THINFER_DIT_TILE_ROWS`
/// (diagnostics: force tiling on at a tiny grid to parity-check the tiled path).
const DIT_TILE_ROWS: u32 = 1024;

fn dit_tile_rows() -> u32 {
    use std::sync::OnceLock;
    static V: OnceLock<u32> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("THINFER_DIT_TILE_ROWS")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&n| n > 0)
            .unwrap_or(DIT_TILE_ROWS)
    })
}

/// Row range `[r0, r0+tr)` for tile `t` of `n_tiles` (remainder spread across
/// the first tiles so every tile is within one row of even).
fn tile_range(rows: u32, n_tiles: u32, t: u32) -> (u32, u32) {
    let base = rows / n_tiles;
    let rem = rows % n_tiles;
    (t * base + t.min(rem), base + if t < rem { 1 } else { 0 })
}

/// View `rows` rows of a row-major activation buffer starting at `row0`, each
/// row `row_bytes` wide.
fn row_slice(base: BufRef, row0: u32, rows: u32, row_bytes: u64) -> BufRef {
    BufRef::view(
        base.id,
        base.offset + row0 as u64 * row_bytes,
        rows as u64 * row_bytes,
    )
}

/// Depth-bounded ring of in-flight `submit_deferred` futures. Each tiled
/// movement queues its GPU work eagerly (wgpu `queue.submit` runs in
/// `submit_deferred`'s synchronous prelude) and hands back a completion future;
/// the ring holds up to `depth` of them so the CPU encodes the next movement
/// while the GPU drains the queue, instead of stalling on a full round-trip per
/// submit (the old `submit_void().await` did ~67 syncs/block). GPU-queue order
/// preserves cross-submit data deps (pass A writes the joint q/k/v, the barrier
/// reads them); the futures only gate guard-buffer recycling + wgpu error-scope
/// drain, so `depth` bounds how many tiles' transients co-live. The ring MUST be
/// fully `drain`ed before the block's weight pins drop (the deferred scopes
/// imported them).
type SubmitFut<'a> =
    core::pin::Pin<Box<dyn core::future::Future<Output = Result<(), WgpuError>> + 'a>>;

struct InFlight<'a> {
    depth: usize,
    q: std::collections::VecDeque<SubmitFut<'a>>,
}

impl<'a> InFlight<'a> {
    fn new(depth: usize) -> Self {
        Self {
            depth: depth.max(1),
            q: std::collections::VecDeque::with_capacity(depth.max(1) + 1),
        }
    }

    /// Queue a deferred submit; if the ring is over depth, await the oldest
    /// (backpressure that bounds live transients without a per-submit stall).
    async fn push(
        &mut self,
        fut: impl core::future::Future<Output = Result<(), WgpuError>> + 'a,
    ) -> Result<(), WgpuError> {
        self.q.push_back(Box::pin(fut));
        while self.q.len() > self.depth {
            self.q.pop_front().expect("nonempty").await?;
        }
        Ok(())
    }

    /// Await every outstanding submit (call before dropping imported pins).
    async fn drain(&mut self) -> Result<(), WgpuError> {
        while let Some(f) = self.q.pop_front() {
            f.await?;
        }
        Ok(())
    }
}

/// In-flight submit ring depth for the tiled block loop (env
/// `THINFER_DIT_PIPELINE_DEPTH`). Higher overlaps more CPU encode with GPU
/// execution at the cost of `depth` tiles' transients co-living.
fn dit_pipeline_depth() -> usize {
    use std::sync::OnceLock;
    static V: OnceLock<usize> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("THINFER_DIT_PIPELINE_DEPTH")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&n| n > 0)
            .unwrap_or(4)
    })
}

/// Cross-submit buffers for one block's tiled execution, allocated once and
/// reused across all blocks (identical geometry). `jq`/`jk`/`jv`/`sa` are the
/// joint `[joint, dim]` q/k/v + attention output (the only resolution-growing
/// residents -- the unavoidable cost of exact joint attention); `imod`/`tmod`
/// the per-stream `[6*dim]` modulation signals built once per block.
struct HunyuanTileBufs {
    jq: WsBuf<WgpuBackend>,
    jk: WsBuf<WgpuBackend>,
    jv: WsBuf<WgpuBackend>,
    sa: WsBuf<WgpuBackend>,
    imod: WsBuf<WgpuBackend>,
    tmod: WsBuf<WgpuBackend>,
}

impl HunyuanTileBufs {
    fn alloc(
        ws: &Workspace<WgpuBackend>,
        bp: &BlockPipelines,
        joint: u32,
        dim: u32,
    ) -> Result<Self, WgpuError> {
        let a = |n: u32| ws.alloc(bp.act_bytes(n));
        Ok(Self {
            jq: a(joint * dim)?,
            jk: a(joint * dim)?,
            jv: a(joint * dim)?,
            sa: a(joint * dim)?,
            imod: a(6 * dim)?,
            tmod: a(6 * dim)?,
        })
    }
}

/// One DP4A site's persistent dequant triple (see [`HunyuanPreparedWeights`]).
struct PreparedI8Bufs {
    i8: WsBuf<WgpuBackend>,
    scale: WsBuf<WgpuBackend>,
    qsum: WsBuf<WgpuBackend>,
}

impl PreparedI8Bufs {
    fn alloc(ws: &Workspace<WgpuBackend>, n: u64, k: u64) -> Result<Self, WgpuError> {
        Ok(Self {
            i8: ws.alloc(n * k)?,
            scale: ws.alloc(n * (k / 32) * 4)?,
            qsum: ws.alloc(n * (k / 32) * 4)?,
        })
    }

    fn weight(&self) -> PreparedWeight {
        PreparedWeight::I8 {
            i8: self.i8.as_buf_ref(),
            scale: self.scale.as_buf_ref(),
            qsum: self.qsum.as_buf_ref(),
        }
    }
}

/// Per-block prepared weights for the img-side matmul sites the tiled path
/// dispatches once per activation tile (q/k/v in pass A; proj/fc1/fc2 in pass
/// B). Allocated once per forward (site shapes are block-invariant), refilled
/// once per block by [`Self::fill`], consumed by every img tile - without this
/// the site dispatch re-runs the identical weight dequant per tile (~32x
/// redundant at 480p). The txt-side sites dispatch once per block and keep the
/// inline path; sites whose pipelines read the raw bf16 weight directly prep
/// nothing (mirrors the Wan `PreparedTileWeights` gating). Bit-identical: same
/// dequant kernels, same weight bytes, run once instead of per tile.
struct HunyuanPreparedWeights {
    /// DP4A i8 triples: img q/k/v (`[dim, dim]`, qkv_self site) and img ffn-up
    /// (`[mlp_h, dim]`).
    img_q: Option<PreparedI8Bufs>,
    img_k: Option<PreparedI8Bufs>,
    img_v: Option<PreparedI8Bufs>,
    img_fc1: Option<PreparedI8Bufs>,
    /// f16 `[N,K]` n-major images for the coopmat / dense-f16 sites: img
    /// attn-out projection (`[dim, dim]`) and img ffn-down (`[dim, mlp_h]`).
    img_proj: Option<WsBuf<WgpuBackend>>,
    img_fc2: Option<WsBuf<WgpuBackend>>,
}

impl HunyuanPreparedWeights {
    /// Allocate the persistent buffers for every site the compiled pipelines
    /// will actually prep (a DP4A site preps i8, else a coopmat/dense-dequant
    /// site preps f16, else the raw weight is read directly).
    fn alloc(
        ws: &Workspace<WgpuBackend>,
        bp: &BlockPipelines,
        dim: u64,
        mlp_h: u64,
    ) -> Result<Self, WgpuError> {
        let i8_site = |mm: &Option<WgpuPipeline>, dq: &Option<DequantStep>| {
            mm.is_some() && dq.is_some() && bp.act_quant.is_some()
        };
        let qkv_i8 = i8_site(&bp.matmul_i8_qkv_self, &bp.dequant_i8_qkv_self);
        let fc1_i8 = i8_site(&bp.matmul_i8_ffn_up, &bp.dequant_i8_ffn_up);
        let proj_f16 = (bp.coopmat_proj.is_some() || bp.dequant_proj.is_some())
            && !i8_site(&bp.matmul_i8_proj, &bp.dequant_i8_proj);
        let fc2_f16 = (bp.coopmat_ffn_down.is_some() || bp.dequant_ffn_down.is_some())
            && !i8_site(&bp.matmul_i8_ffn_down, &bp.dequant_i8_ffn_down);
        let i8_buf = |on: bool, n: u64, k: u64| -> Result<Option<PreparedI8Bufs>, WgpuError> {
            on.then(|| PreparedI8Bufs::alloc(ws, n, k)).transpose()
        };
        let f16_buf = |on: bool, n: u64, k: u64| -> Result<Option<WsBuf<WgpuBackend>>, WgpuError> {
            on.then(|| ws.alloc(n * k * 2)).transpose()
        };
        Ok(Self {
            img_q: i8_buf(qkv_i8, dim, dim)?,
            img_k: i8_buf(qkv_i8, dim, dim)?,
            img_v: i8_buf(qkv_i8, dim, dim)?,
            img_fc1: i8_buf(fc1_i8, mlp_h, dim)?,
            img_proj: f16_buf(proj_f16, dim, dim)?,
            img_fc2: f16_buf(fc2_f16, dim, mlp_h)?,
        })
    }

    /// Re-dequant this block's img-side site weights into the persistent
    /// buffers - dispatched once per block (alongside the modulation linears),
    /// consumed by every img tile in passes A and B.
    fn fill<'w>(
        &self,
        scope: &BatchScope<'w, WgpuBackend>,
        bp: &BlockPipelines,
        b: &BlockBufs,
        dim: u32,
        mlp_h: u32,
    ) -> Result<(), WgpuError> {
        let i8_fill = |dst: &Option<PreparedI8Bufs>,
                       dq: Option<&DequantStep>,
                       w: BufRef,
                       n: u32,
                       k: u32|
         -> Result<(), WgpuError> {
            if let (Some(d), Some(dq)) = (dst, dq) {
                Block::prepare_weight_i8(
                    scope,
                    dq,
                    scope.import_copy(w),
                    scope.import_copy(d.i8.as_buf_ref()),
                    scope.import_copy(d.scale.as_buf_ref()),
                    scope.import_copy(d.qsum.as_buf_ref()),
                    n,
                    k,
                )?;
            }
            Ok(())
        };
        i8_fill(
            &self.img_q,
            bp.dequant_i8_qkv_self.as_ref(),
            b.img_q.weight,
            dim,
            dim,
        )?;
        i8_fill(
            &self.img_k,
            bp.dequant_i8_qkv_self.as_ref(),
            b.img_k.weight,
            dim,
            dim,
        )?;
        i8_fill(
            &self.img_v,
            bp.dequant_i8_qkv_self.as_ref(),
            b.img_v.weight,
            dim,
            dim,
        )?;
        i8_fill(
            &self.img_fc1,
            bp.dequant_i8_ffn_up.as_ref(),
            b.img_fc1.weight,
            mlp_h,
            dim,
        )?;
        // f16 sites: the coopmat step's dequant when present, else the site's
        // dense-fallback dequant (same [N,K] n-major f16 bytes either way).
        let f16_fill = |dst: &Option<WsBuf<WgpuBackend>>,
                        dq: Option<&DequantStep>,
                        w: BufRef,
                        n: u32,
                        k: u32|
         -> Result<(), WgpuError> {
            if let (Some(d), Some(dq)) = (dst, dq) {
                Block::prepare_weight_f16(
                    scope,
                    dq,
                    scope.import_copy(w),
                    scope.import_copy(d.as_buf_ref()),
                    n,
                    k,
                )?;
            }
            Ok(())
        };
        let proj_dq = bp
            .coopmat_proj
            .as_ref()
            .map(|c| &c.dequant_f16)
            .or(bp.dequant_proj.as_ref());
        let fc2_dq = bp
            .coopmat_ffn_down
            .as_ref()
            .map(|c| &c.dequant_f16)
            .or(bp.dequant_ffn_down.as_ref());
        f16_fill(&self.img_proj, proj_dq, b.img_proj.weight, dim, dim)?;
        f16_fill(&self.img_fc2, fc2_dq, b.img_fc2.weight, dim, mlp_h)?;
        Ok(())
    }

    fn i8w(slot: &Option<PreparedI8Bufs>) -> Option<PreparedWeight> {
        slot.as_ref().map(PreparedI8Bufs::weight)
    }

    fn f16w(slot: &Option<WsBuf<WgpuBackend>>) -> Option<PreparedWeight> {
        slot.as_ref().map(|b| PreparedWeight::F16(b.as_buf_ref()))
    }
}

// ============================================================================
// DiT
// ============================================================================

pub struct HunyuanDit {
    pub pipelines: HunyuanDitPipelines,
    refiner: HunyuanRefiner,
    rope: RopeEmbedder,
    handles: DitH,
}

#[derive(Debug)]
pub enum HunyuanDitError<SE: core::fmt::Debug> {
    Wgpu(WgpuError),
    Load(LoadError),
    Residency(ResidencyError<SE, WgpuError>),
    /// The caller's `cancel` predicate returned true at a step boundary; denoise
    /// aborted cooperatively (the partial latent is discarded).
    Cancelled,
}
impl<SE: core::fmt::Debug> From<WgpuError> for HunyuanDitError<SE> {
    fn from(e: WgpuError) -> Self {
        Self::Wgpu(e)
    }
}
impl<SE: core::fmt::Debug> From<LoadError> for HunyuanDitError<SE> {
    fn from(e: LoadError) -> Self {
        Self::Load(e)
    }
}
impl<SE: core::fmt::Debug> From<ResidencyError<SE, WgpuError>> for HunyuanDitError<SE> {
    fn from(e: ResidencyError<SE, WgpuError>) -> Self {
        Self::Residency(e)
    }
}

/// Parity bisection taps (row-major `[*, dim]`).
#[derive(Default)]
pub struct HunyuanDitTaps<'a> {
    pub vec: Option<&'a mut Vec<f32>>,
    pub img_in: Option<&'a mut Vec<f32>>,
    pub txt_in: Option<&'a mut Vec<f32>>,
    pub block0_img: Option<&'a mut Vec<f32>>,
    pub block0_txt: Option<&'a mut Vec<f32>>,
}

impl HunyuanDit {
    /// `i8` MUST match the value passed to [`HunyuanDitPipelines::compile_with`]:
    /// it selects both the Q8_0 weight transcode here and the DP4A matmul
    /// pipelines there; a mismatch makes the i8 sites read garbage.
    pub fn new<S: WeightSource>(
        pipelines: HunyuanDitPipelines,
        refiner: HunyuanRefiner,
        residency: &WeightResidency<S>,
        i8: bool,
    ) -> Result<Self, LoadError> {
        // The coopmat ffn_down weight encoding follows what the pipelines actually
        // built (compile gates on the device), so they can never disagree -- a
        // no-coopmat device leaves fc2 dense bf16. Read before `pipelines` moves.
        let coopmat = pipelines.bp.coopmat_ffn_down.is_some();
        Ok(Self {
            pipelines,
            refiner,
            rope: RopeEmbedder::new(
                cfg::ROPE_THETA as f32,
                cfg::ROPE_DIM,
                // axis caps: per-axis latent grid extents stay well under these
                // (480p: t<=~21, h<=~30, w<=~52).
                [1024, 1024, 1024],
            ),
            handles: DitH::register(residency, i8, coopmat)?,
        })
    }

    /// RoPE grid ids `(t,h,w)` row-major (t outer, w inner) -- the patchify token
    /// order (`flatten(2).transpose` of the conv output).
    fn grid_ids(grid: (usize, usize, usize)) -> Vec<i32> {
        let (t, h, w) = grid;
        let mut ids = Vec::with_capacity(t * h * w * 3);
        for ti in 0..t {
            for hi in 0..h {
                for wi in 0..w {
                    ids.push(ti as i32);
                    ids.push(hi as i32);
                    ids.push(wi as i32);
                }
            }
        }
        ids
    }

    /// One T2V forward. `text [seq, 3584]`, `img_tokens [n_img, 65]` in `(t,h,w)`
    /// order (`[noise32 | cond32 | mask1]`; T2V = noise then zeros), `grid =
    /// (T,H,W)` latent extents (`T*H*W == n_img`), timestep `t`. Returns the
    /// velocity `[n_img, 32]`.
    #[allow(clippy::too_many_arguments)]
    pub async fn forward<S: WeightSource>(
        &self,
        backend: &WgpuBackend,
        residency: &WeightResidency<S>,
        ws: &Workspace<WgpuBackend>,
        text: &[f32],
        seq: usize,
        img_tokens: &[f32],
        grid: (usize, usize, usize),
        t: f32,
        window: u32,
        mut taps: Option<&mut HunyuanDitTaps<'_>>,
    ) -> Result<Vec<f32>, HunyuanDitError<S::Error>> {
        let (gt, gh, gw) = grid;
        let n_img = gt * gh * gw;
        // Temporal window period = tokens per latent frame (frame-major g_h*g_w).
        let period = (gh * gw) as u32;
        assert_eq!(
            img_tokens.len(),
            n_img * cfg::CONV_IN_CHANNELS,
            "img tokens"
        );
        let act = self.pipelines.act();
        let asz = self.pipelines.act_size();
        let dim = cfg::HIDDEN as u32;
        let heads = cfg::HEADS as u32;
        let hd = cfg::HEAD_DIM as u32;
        let mlp_h = cfg::MLP_HIDDEN as u32;
        let latent = cfg::LATENT_CHANNELS as u32;
        let conv_in = cfg::CONV_IN_CHANNELS as u32;
        let seqr = seq as u32;
        let n = n_img as u32;
        let joint = n + seqr;

        // 1) txt = SingleTokenRefiner(text, t)  (own submit + host readback).
        let txt = self
            .refiner
            .refine(backend, residency, ws, text, seq, t, None)
            .await
            .map_err(|e| match e {
                crate::hunyuan::refiner::HunyuanRefinerError::Wgpu(e) => HunyuanDitError::Wgpu(e),
                crate::hunyuan::refiner::HunyuanRefinerError::Load(e) => HunyuanDitError::Load(e),
                crate::hunyuan::refiner::HunyuanRefinerError::Residency(e) => {
                    HunyuanDitError::Residency(e)
                }
            })?;

        // Host uploads.
        let upload = |slice: &[f32]| -> Result<WsBuf<WgpuBackend>, WgpuError> {
            let bytes = act_upload_bytes(act, slice);
            let buf = ws.alloc(bytes.len() as u64)?;
            backend.write_buffer(buf.id(), 0, &bytes)?;
            Ok(buf)
        };
        let txt_up = upload(&txt)?;
        let img_up = upload(img_tokens)?;
        let tsin_up = upload(&timestep_sinusoid(t))?;
        let freqs = self.rope.lookup(&Self::grid_ids(grid));
        // Per-img-token freq stride (interleaved 3D rope packs `hd` freq values
        // per token); used to row-slice freqs per tile.
        let freq_stride = (freqs.len() / n_img) as u32;
        let freqs_up = {
            let bytes = freqs_upload_bytes(act, &freqs);
            let buf = ws.alloc(bytes.len() as u64)?;
            backend.write_buffer(buf.id(), 0, &bytes)?;
            buf
        };

        let mut pins: Vec<GpuView> = Vec::new();
        let top = self.acquire_top(residency, backend, &mut pins).await?;

        // 2) Setup submit: vec=time_in(t); silu_vec; img=img_in(tokens);
        //    txt += cond_type[0]. Persist img/txt/silu_vec for the block loop.
        let img_ws = ws.alloc((n * dim) as u64 * asz)?;
        let txt_ws = ws.alloc((seqr * dim) as u64 * asz)?;
        let silu_vec_ws = ws.alloc(dim as u64 * asz)?;
        let mut tap_vec = None;
        let mut tap_img_in = None;
        {
            let scope = ws.batch();
            let bp = &self.pipelines.bp;
            // vec = linear(time2, silu(linear(time0, sinusoid))).
            let ts = scope.import_copy(tsin_up.as_buf_ref());
            let h0 = linear(
                &scope,
                bp,
                ts,
                &top.time0,
                1,
                dim,
                FREQ_DIM as u32,
                Site::Module,
            )?;
            let h0a = silu(&scope, bp, h0, dim)?;
            let vec = linear(&scope, bp, h0a, &top.time2, 1, dim, dim, Site::Module)?;
            let silu_vec = silu(&scope, bp, vec, dim)?;
            let dst = scope.import_copy(silu_vec_ws.as_buf_ref());
            scope.copy_buffer_to_buffer(silu_vec, 0, dst, 0, dim as u64 * asz)?;
            if taps.as_ref().is_some_and(|t| t.vec.is_some()) {
                tap_vec = Some(persist(&scope, ws, vec, cfg::HIDDEN, asz)?);
            }

            // img = img_in(tokens [n,65]) -> [n, dim].
            let xi = scope.import_copy(img_up.as_buf_ref());
            let img = linear(&scope, bp, xi, &top.img_in, n, dim, conv_in, Site::Module)?;
            let dimg = scope.import_copy(img_ws.as_buf_ref());
            scope.copy_buffer_to_buffer(img, 0, dimg, 0, (n * dim) as u64 * asz)?;
            if taps.as_ref().is_some_and(|t| t.img_in.is_some()) {
                tap_img_in = Some(persist(&scope, ws, img, n_img * cfg::HIDDEN, asz)?);
            }

            // txt = uploaded refined txt + cond_type[0] (row 0 of [3, dim]).
            let txt_b = scope.import_copy(txt_up.as_buf_ref());
            let cu = scope.u32x4_uniform(dim, 0, 0, 0)?;
            let cond0 = scope.import_copy(top.cond_type); // row 0 = first `dim` elems
            let dtxt = scope.import_copy(txt_ws.as_buf_ref());
            scope.bcast_add::<BcastAddF32>(&bp.bcast_add, txt_b, cond0, cu, dtxt, seqr * dim)?;
            scope.submit_void().await?;
        }
        if let Some(t) = taps.as_deref_mut() {
            if let (Some(s), Some(b)) = (t.vec.as_deref_mut(), tap_vec.as_ref()) {
                *s = read_acts(backend, &b.as_buf_ref(), cfg::HIDDEN, act).await?;
            }
            if let (Some(s), Some(b)) = (t.img_in.as_deref_mut(), tap_img_in.as_ref()) {
                *s = read_acts(backend, &b.as_buf_ref(), n_img * cfg::HIDDEN, act).await?;
            }
            if let Some(s) = t.txt_in.as_deref_mut() {
                let bytes = backend
                    .read_buffer(txt_ws.id(), 0, (seqr * dim) as u64 * asz)
                    .await?;
                *s = act_readback_to_f32(act, &bytes, seq * cfg::HIDDEN);
            }
        }

        // 3) Block loop (stream weights; ping-pong img/txt). Above one tile's
        //    worth of img tokens, run each block activation-tiled so the 480p
        //    per-block buffers don't all co-live in one ~6GB+ submit. Diag taps
        //    are intra-block + single-scope, so they force the untiled path.
        let bp = &self.pipelines.bp;
        let n_tiles = if taps.is_some() {
            1
        } else {
            n.div_ceil(dit_tile_rows()).max(1)
        };
        let tile = if n_tiles > 1 {
            // Per-block prepared img-site weights (dequant once per block;
            // every tile reuses them). Shapes are block-invariant, so the
            // buffers are allocated once here and refilled inside the loop.
            Some((
                HunyuanTileBufs::alloc(ws, bp, joint, dim)?,
                HunyuanPreparedWeights::alloc(ws, bp, dim as u64, mlp_h as u64)?,
            ))
        } else {
            None
        };
        let mut img_cur = img_ws;
        let mut txt_cur = txt_ws;
        // Stream the 54-block DiT with one-block-ahead prefetch: acquire block 0,
        // then compute block bi while the NEXT block's weights upload concurrently.
        // compute/block (~140ms) >> upload/block (~50ms warm), so the per-step
        // weight stream (the dominant non-compute cost: ~2.8s/step serial before
        // this) hides almost entirely behind GPU compute. Two blocks' weights
        // co-reside during the overlap (~620MB, well within budget).
        let nblocks = self.handles.blocks.len();
        let mut cur_pins: Vec<GpuView> = Vec::new();
        let mut cur_b = self
            .acquire_block(&self.handles.blocks[0], residency, backend, &mut cur_pins)
            .await?;
        for bi in 0..nblocks {
            let img_nxt = ws.alloc((n * dim) as u64 * asz)?;
            let txt_nxt = ws.alloc((seqr * dim) as u64 * asz)?;
            let mut next_pins: Vec<GpuView> = Vec::new();
            let compute = self.run_block(
                ws,
                &cur_b,
                &freqs_up,
                freq_stride,
                &silu_vec_ws,
                tile.as_ref().map(|(t, p)| (t, p)),
                &img_cur,
                &txt_cur,
                &img_nxt,
                &txt_nxt,
                n,
                seqr,
                dim,
                heads,
                hd,
                mlp_h,
                n_tiles,
                period,
                window,
            );
            let prefetch = async {
                if bi + 1 < nblocks {
                    self.acquire_block(
                        &self.handles.blocks[bi + 1],
                        residency,
                        backend,
                        &mut next_pins,
                    )
                    .await
                    .map(Some)
                } else {
                    Ok(None)
                }
            };
            let (c_res, p_res) = futures::join!(compute, prefetch);
            c_res?;
            let next_b = p_res?;
            drop(cur_pins);
            if bi == 0
                && let Some(t) = taps.as_deref_mut()
            {
                if let Some(s) = t.block0_img.as_deref_mut() {
                    let bytes = backend
                        .read_buffer(img_nxt.id(), 0, (n * dim) as u64 * asz)
                        .await?;
                    *s = act_readback_to_f32(act, &bytes, n_img * cfg::HIDDEN);
                }
                if let Some(s) = t.block0_txt.as_deref_mut() {
                    let bytes = backend
                        .read_buffer(txt_nxt.id(), 0, (seqr * dim) as u64 * asz)
                        .await?;
                    *s = act_readback_to_f32(act, &bytes, seq * cfg::HIDDEN);
                }
            }
            img_cur = img_nxt;
            txt_cur = txt_nxt;
            cur_pins = next_pins;
            if let Some(b) = next_b {
                cur_b = b;
            }
        }

        // 4) final_layer: shift,scale = adaLN(silu_vec).chunk(2); modulate; linear.
        let vel_ws = ws.alloc((n * latent) as u64 * asz)?;
        {
            let scope = ws.batch();
            let bp = &self.pipelines.bp;
            let sv = scope.import_copy(silu_vec_ws.as_buf_ref());
            let emb = linear(
                &scope,
                bp,
                sv,
                &top.final_adaln,
                1,
                2 * dim,
                dim,
                Site::Module,
            )?;
            let shift = mod_sig(&scope, bp, emb, 0, dim)?;
            let scale = mod_sig(&scope, bp, emb, 1, dim)?;
            let img = scope.import_copy(img_cur.as_buf_ref());
            let modded = norm_modulate(&scope, bp, img, scale, shift, n, dim)?;
            let vel = linear(
                &scope,
                bp,
                modded,
                &top.final_lin,
                n,
                latent,
                dim,
                Site::Module,
            )?;
            let dst = scope.import_copy(vel_ws.as_buf_ref());
            scope.copy_buffer_to_buffer(vel, 0, dst, 0, (n * latent) as u64 * asz)?;
            scope.submit_void().await?;
        }
        drop(pins);
        read_acts(
            backend,
            &vel_ws.as_buf_ref(),
            n_img * cfg::LATENT_CHANNELS,
            act,
        )
        .await
        .map_err(Into::into)
    }

    /// One dual-stream block in the activation-tiled regime, numerically
    /// identical to the single-scope path. Movements, each its own submit so the
    /// pool recycles tile transients between them: (1) modulation `[1,6*dim]` per
    /// stream -> persistent `tile.imod/tmod`; (2) pass A: txt q/k/v (whole) +
    /// per-img-tile q/k/v (+qk-norm/rope) written into the joint `tile.jq/jk/jv`
    /// (`[img ; txt]` rows); (3) barrier: joint `op_sdpa_f16 -> tile.sa`
    /// (TDR-chunked inside); (4) pass B: per-img-tile (and whole-txt) o-proj +
    /// gated residual + norm/modulate + gelu MLP + gated residual into
    /// `img_nxt`/`txt_nxt`. The only resolution-growing residents are
    /// `jq/jk/jv/sa`; every heavy `[tile, *]` transient (FFN incl.) is bounded by
    /// the tile row count.
    #[allow(clippy::too_many_arguments)]
    async fn block_tiled(
        &self,
        ws: &Workspace<WgpuBackend>,
        b: &BlockBufs,
        freqs_up: &WsBuf<WgpuBackend>,
        freq_stride: u32,
        silu_vec_ws: &WsBuf<WgpuBackend>,
        tile: &HunyuanTileBufs,
        ptw: &HunyuanPreparedWeights,
        img_cur: &WsBuf<WgpuBackend>,
        txt_cur: &WsBuf<WgpuBackend>,
        img_nxt: &WsBuf<WgpuBackend>,
        txt_nxt: &WsBuf<WgpuBackend>,
        n: u32,
        seqr: u32,
        dim: u32,
        heads: u32,
        hd: u32,
        mlp_h: u32,
        n_tiles: u32,
        period: u32,
        window: u32,
    ) -> Result<(), WgpuError> {
        let bp = &self.pipelines.bp;
        let joint = n + seqr;
        let dim_row = bp.act_bytes(dim);
        let freq_row = bp.act_bytes(freq_stride);
        // Pipeline the per-movement submits: GPU-queue order keeps the data deps
        // (mod -> pass A -> attention barrier -> pass B), the ring only gates
        // transient recycling. Fully drained before return (the deferred scopes
        // imported `b`'s weight pins + the persistent tile/act buffers).
        let mut ring = InFlight::new(dit_pipeline_depth());

        // 1) Modulation signals for this block, persisted across the tile
        //    scopes, plus the once-per-block img-site weight dequants every
        //    tile below reuses.
        {
            let scope = ws.batch();
            ptw.fill(&scope, bp, b, dim, mlp_h)?;
            let sv = scope.import_copy(silu_vec_ws.as_buf_ref());
            let imodp = linear(&scope, bp, sv, &b.img_mod, 1, 6 * dim, dim, Site::Mod)?;
            let tmodp = linear(&scope, bp, sv, &b.txt_mod, 1, 6 * dim, dim, Site::Mod)?;
            let di = scope.import_copy(tile.imod.as_buf_ref());
            scope.copy_buffer_to_buffer(imodp, 0, di, 0, bp.act_bytes(6 * dim))?;
            let dt = scope.import_copy(tile.tmod.as_buf_ref());
            scope.copy_buffer_to_buffer(tmodp, 0, dt, 0, bp.act_bytes(6 * dim))?;
            ring.push(scope.submit_deferred()).await?;
        }

        // 2a) Pass A txt (whole): write tq/tk/tv into the txt region of jq/jk/jv.
        {
            let scope = ws.batch();
            let tmod = scope.import_copy(tile.tmod.as_buf_ref());
            let tsig = |k| mod_sig(&scope, bp, tmod, k, dim);
            let (t_sh1, t_sc1) = (tsig(0)?, tsig(1)?);
            let txt = scope.import_copy(txt_cur.as_buf_ref());
            let tm = norm_modulate(&scope, bp, txt, t_sc1, t_sh1, seqr, dim)?;
            let tm_a = qkv_a_side(&scope, bp, tm, seqr, dim)?;
            let tq = linear_a(
                &scope,
                bp,
                tm_a,
                &b.txt_q,
                seqr,
                dim,
                dim,
                Site::QkvSelf,
                None,
            )?;
            let tq = qk_norm_rope(&scope, bp, tq, b.txt_qn, None, seqr, heads, hd)?;
            let tk = linear_a(
                &scope,
                bp,
                tm_a,
                &b.txt_k,
                seqr,
                dim,
                dim,
                Site::QkvSelf,
                None,
            )?;
            let tk = qk_norm_rope(&scope, bp, tk, b.txt_kn, None, seqr, heads, hd)?;
            let tv = linear_a(
                &scope,
                bp,
                tm_a,
                &b.txt_v,
                seqr,
                dim,
                dim,
                Site::QkvSelf,
                None,
            )?;
            let off = n as u64 * dim_row;
            let len = seqr as u64 * dim_row;
            for (src, base) in [(tq, &tile.jq), (tk, &tile.jk), (tv, &tile.jv)] {
                let d = scope.import_copy(base.as_buf_ref());
                scope.copy_buffer_to_buffer(src, 0, d, off, len)?;
            }
            ring.push(scope.submit_deferred()).await?;
        }

        // 2b) Pass A img tiles: write iq/ik/iv into the img region of jq/jk/jv.
        for t in 0..n_tiles {
            let (r0, tr) = tile_range(n, n_tiles, t);
            let scope = ws.batch();
            let imod = scope.import_copy(tile.imod.as_buf_ref());
            let isig = |k| mod_sig(&scope, bp, imod, k, dim);
            let (i_sh1, i_sc1) = (isig(0)?, isig(1)?);
            let img_t = scope.import_copy(row_slice(img_cur.as_buf_ref(), r0, tr, dim_row));
            let im = norm_modulate(&scope, bp, img_t, i_sc1, i_sh1, tr, dim)?;
            let im_a = qkv_a_side(&scope, bp, im, tr, dim)?;
            let fq = scope.import_copy(row_slice(freqs_up.as_buf_ref(), r0, tr, freq_row));
            let pq = HunyuanPreparedWeights::i8w(&ptw.img_q);
            let iq = linear_a(&scope, bp, im_a, &b.img_q, tr, dim, dim, Site::QkvSelf, pq)?;
            let iq = qk_norm_rope(&scope, bp, iq, b.img_qn, Some(fq), tr, heads, hd)?;
            let fq2 = scope.import_copy(row_slice(freqs_up.as_buf_ref(), r0, tr, freq_row));
            let pk = HunyuanPreparedWeights::i8w(&ptw.img_k);
            let ik = linear_a(&scope, bp, im_a, &b.img_k, tr, dim, dim, Site::QkvSelf, pk)?;
            let ik = qk_norm_rope(&scope, bp, ik, b.img_kn, Some(fq2), tr, heads, hd)?;
            let pv = HunyuanPreparedWeights::i8w(&ptw.img_v);
            let iv = linear_a(&scope, bp, im_a, &b.img_v, tr, dim, dim, Site::QkvSelf, pv)?;
            let off = r0 as u64 * dim_row;
            let len = tr as u64 * dim_row;
            for (src, base) in [(iq, &tile.jq), (ik, &tile.jk), (iv, &tile.jv)] {
                let d = scope.import_copy(base.as_buf_ref());
                scope.copy_buffer_to_buffer(src, 0, d, off, len)?;
            }
            ring.push(scope.submit_deferred()).await?;
        }

        // 3) Barrier: joint self-attention over the whole [img ; txt] sequence.
        //    Image queries window to `±window` latent frames (+ all text); text
        //    queries attend everything. `window = 0` is full attention.
        {
            let scope = ws.batch();
            let jq = scope.import_copy(tile.jq.as_buf_ref());
            let jk = scope.import_copy(tile.jk.as_buf_ref());
            let jv = scope.import_copy(tile.jv.as_buf_ref());
            let sa = scope.import_copy(tile.sa.as_buf_ref());
            let mask = scope.write_uniform(&0f32.to_le_bytes())?;
            let scale = 1.0_f32 / (hd as f32).sqrt();
            op_sdpa_f16_win(
                &scope,
                bp,
                ActBuf::dense(jq),
                ActBuf::dense(jk),
                ActBuf::dense(jv),
                mask,
                ActBuf::dense(sa),
                1,
                joint,
                joint,
                heads,
                heads,
                hd,
                scale,
                0,
                period,
                window,
                n, // txt_start = n_img (text tokens follow the image rows)
            )?;
            ring.push(scope.submit_deferred()).await?;
        }

        // 4a) Pass B img tiles: o-proj + gated residual + MLP -> img_nxt.
        for t in 0..n_tiles {
            let (r0, tr) = tile_range(n, n_tiles, t);
            let scope = ws.batch();
            let imod = scope.import_copy(tile.imod.as_buf_ref());
            let isig = |k| mod_sig(&scope, bp, imod, k, dim);
            let (i_g1, i_sh2, i_sc2, i_g2) = (isig(2)?, isig(3)?, isig(4)?, isig(5)?);
            let img_t = scope.import_copy(row_slice(img_cur.as_buf_ref(), r0, tr, dim_row));
            let sa_t = scope.import_copy(row_slice(tile.sa.as_buf_ref(), r0, tr, dim_row));
            let pp = HunyuanPreparedWeights::f16w(&ptw.img_proj);
            let ip = linear_a(
                &scope,
                bp,
                ActBuf::dense(sa_t),
                &b.img_proj,
                tr,
                dim,
                dim,
                Site::Proj,
                pp,
            )?;
            let img = gate_residual(&scope, bp, img_t, i_g1, ip, tr, dim)?;
            let im2 = norm_modulate(&scope, bp, img, i_sc2, i_sh2, tr, dim)?;
            let p1 = HunyuanPreparedWeights::i8w(&ptw.img_fc1);
            let p2 = HunyuanPreparedWeights::f16w(&ptw.img_fc2);
            let imlp = mlp(
                &scope, bp, im2, &b.img_fc1, &b.img_fc2, tr, dim, mlp_h, p1, p2,
            )?;
            let img = gate_residual(&scope, bp, img, i_g2, imlp, tr, dim)?;
            let d = scope.import_copy(row_slice(img_nxt.as_buf_ref(), r0, tr, dim_row));
            scope.copy_buffer_to_buffer(img, 0, d, 0, tr as u64 * dim_row)?;
            ring.push(scope.submit_deferred()).await?;
        }

        // 4b) Pass B txt (whole): same sublayer over the txt region of sa.
        {
            let scope = ws.batch();
            let tmod = scope.import_copy(tile.tmod.as_buf_ref());
            let tsig = |k| mod_sig(&scope, bp, tmod, k, dim);
            let (t_g1, t_sh2, t_sc2, t_g2) = (tsig(2)?, tsig(3)?, tsig(4)?, tsig(5)?);
            let txt = scope.import_copy(txt_cur.as_buf_ref());
            let sa_t = scope.import_copy(row_slice(tile.sa.as_buf_ref(), n, seqr, dim_row));
            let tp = linear(&scope, bp, sa_t, &b.txt_proj, seqr, dim, dim, Site::Proj)?;
            let txt = gate_residual(&scope, bp, txt, t_g1, tp, seqr, dim)?;
            let tm2 = norm_modulate(&scope, bp, txt, t_sc2, t_sh2, seqr, dim)?;
            let tmlp = mlp(
                &scope, bp, tm2, &b.txt_fc1, &b.txt_fc2, seqr, dim, mlp_h, None, None,
            )?;
            let txt = gate_residual(&scope, bp, txt, t_g2, tmlp, seqr, dim)?;
            let d = scope.import_copy(txt_nxt.as_buf_ref());
            scope.copy_buffer_to_buffer(txt, 0, d, 0, seqr as u64 * dim_row)?;
            ring.push(scope.submit_deferred()).await?;
        }

        // Drain before return: the block's weight pins (`b`) must outlive every
        // deferred submit that imported them.
        ring.drain().await?;
        Ok(())
    }

    /// Dispatch one dual-stream block: the activation-tiled path when `tile` is
    /// `Some` (480p; bounds the per-scope peak), else the whole-block single
    /// submit ([`Self::block_single`], used for the parity-tap / tiny-grid path).
    #[allow(clippy::too_many_arguments)]
    async fn run_block(
        &self,
        ws: &Workspace<WgpuBackend>,
        b: &BlockBufs,
        freqs_up: &WsBuf<WgpuBackend>,
        freq_stride: u32,
        silu_vec_ws: &WsBuf<WgpuBackend>,
        tile: Option<(&HunyuanTileBufs, &HunyuanPreparedWeights)>,
        img_cur: &WsBuf<WgpuBackend>,
        txt_cur: &WsBuf<WgpuBackend>,
        img_nxt: &WsBuf<WgpuBackend>,
        txt_nxt: &WsBuf<WgpuBackend>,
        n: u32,
        seqr: u32,
        dim: u32,
        heads: u32,
        hd: u32,
        mlp_h: u32,
        n_tiles: u32,
        period: u32,
        window: u32,
    ) -> Result<(), WgpuError> {
        match tile {
            Some((tile, ptw)) => {
                self.block_tiled(
                    ws,
                    b,
                    freqs_up,
                    freq_stride,
                    silu_vec_ws,
                    tile,
                    ptw,
                    img_cur,
                    txt_cur,
                    img_nxt,
                    txt_nxt,
                    n,
                    seqr,
                    dim,
                    heads,
                    hd,
                    mlp_h,
                    n_tiles,
                    period,
                    window,
                )
                .await
            }
            None => {
                self.block_single(
                    ws,
                    b,
                    freqs_up,
                    silu_vec_ws,
                    img_cur,
                    txt_cur,
                    img_nxt,
                    txt_nxt,
                    n,
                    seqr,
                    dim,
                    heads,
                    hd,
                    mlp_h,
                    period,
                    window,
                )
                .await
            }
        }
    }

    /// One dual-stream block in a single submit (no activation tiling). Numerically
    /// identical to the tiled path; used only when the whole block's transients fit
    /// one scope (parity taps + tiny grids). The 480p path always tiles.
    #[allow(clippy::too_many_arguments)]
    async fn block_single(
        &self,
        ws: &Workspace<WgpuBackend>,
        b: &BlockBufs,
        freqs_up: &WsBuf<WgpuBackend>,
        silu_vec_ws: &WsBuf<WgpuBackend>,
        img_cur: &WsBuf<WgpuBackend>,
        txt_cur: &WsBuf<WgpuBackend>,
        img_nxt: &WsBuf<WgpuBackend>,
        txt_nxt: &WsBuf<WgpuBackend>,
        n: u32,
        seqr: u32,
        dim: u32,
        heads: u32,
        hd: u32,
        mlp_h: u32,
        period: u32,
        window: u32,
    ) -> Result<(), WgpuError> {
        let bp = &self.pipelines.bp;
        let asz = self.pipelines.act_size();
        let joint = n + seqr;
        let scope = ws.batch();
        let sv = scope.import_copy(silu_vec_ws.as_buf_ref());

        // modulation: [1, 6*dim] per stream from silu_vec.
        let imodp = linear(&scope, bp, sv, &b.img_mod, 1, 6 * dim, dim, Site::Mod)?;
        let tmodp = linear(&scope, bp, sv, &b.txt_mod, 1, 6 * dim, dim, Site::Mod)?;
        let isig = |k| mod_sig(&scope, bp, imodp, k, dim);
        let tsig = |k| mod_sig(&scope, bp, tmodp, k, dim);
        let (i_sh1, i_sc1, i_g1) = (isig(0)?, isig(1)?, isig(2)?);
        let (i_sh2, i_sc2, i_g2) = (isig(3)?, isig(4)?, isig(5)?);
        let (t_sh1, t_sc1, t_g1) = (tsig(0)?, tsig(1)?, tsig(2)?);
        let (t_sh2, t_sc2, t_g2) = (tsig(3)?, tsig(4)?, tsig(5)?);

        let img = scope.import_copy(img_cur.as_buf_ref());
        let txt = scope.import_copy(txt_cur.as_buf_ref());

        // --- attention sublayer ---
        let im = norm_modulate(&scope, bp, img, i_sc1, i_sh1, n, dim)?;
        let tm = norm_modulate(&scope, bp, txt, t_sc1, t_sh1, seqr, dim)?;
        let im_a = qkv_a_side(&scope, bp, im, n, dim)?;
        let tm_a = qkv_a_side(&scope, bp, tm, seqr, dim)?;
        let fq = scope.import_copy(freqs_up.as_buf_ref());
        let iq = linear_a(&scope, bp, im_a, &b.img_q, n, dim, dim, Site::QkvSelf, None)?;
        let iq = qk_norm_rope(&scope, bp, iq, b.img_qn, Some(fq), n, heads, hd)?;
        let fq2 = scope.import_copy(freqs_up.as_buf_ref());
        let ik = linear_a(&scope, bp, im_a, &b.img_k, n, dim, dim, Site::QkvSelf, None)?;
        let ik = qk_norm_rope(&scope, bp, ik, b.img_kn, Some(fq2), n, heads, hd)?;
        let iv = linear_a(&scope, bp, im_a, &b.img_v, n, dim, dim, Site::QkvSelf, None)?;
        let tq = linear_a(
            &scope,
            bp,
            tm_a,
            &b.txt_q,
            seqr,
            dim,
            dim,
            Site::QkvSelf,
            None,
        )?;
        let tq = qk_norm_rope(&scope, bp, tq, b.txt_qn, None, seqr, heads, hd)?;
        let tk = linear_a(
            &scope,
            bp,
            tm_a,
            &b.txt_k,
            seqr,
            dim,
            dim,
            Site::QkvSelf,
            None,
        )?;
        let tk = qk_norm_rope(&scope, bp, tk, b.txt_kn, None, seqr, heads, hd)?;
        let tv = linear_a(
            &scope,
            bp,
            tm_a,
            &b.txt_v,
            seqr,
            dim,
            dim,
            Site::QkvSelf,
            None,
        )?;

        // joint concat [img ; txt] for q/k/v.
        let img_bytes = (n * dim) as u64 * asz;
        let txt_bytes = (seqr * dim) as u64 * asz;
        let jq = concat2(&scope, iq, tq, img_bytes, txt_bytes)?;
        let jk = concat2(&scope, ik, tk, img_bytes, txt_bytes)?;
        let jv = concat2(&scope, iv, tv, img_bytes, txt_bytes)?;
        let sa = attention(&scope, bp, jq, jk, jv, joint, n, period, window, heads, hd)?;
        let img_sa = scope.alloc((n * dim) as u64 * asz)?;
        scope.copy_buffer_to_buffer(sa, 0, img_sa, 0, (n * dim) as u64 * asz)?;
        let txt_sa = scope.alloc((seqr * dim) as u64 * asz)?;
        scope.copy_buffer_to_buffer(
            sa,
            (n * dim) as u64 * asz,
            txt_sa,
            0,
            (seqr * dim) as u64 * asz,
        )?;

        let ip = linear(&scope, bp, img_sa, &b.img_proj, n, dim, dim, Site::Proj)?;
        let img = gate_residual(&scope, bp, img, i_g1, ip, n, dim)?;
        let tp = linear(&scope, bp, txt_sa, &b.txt_proj, seqr, dim, dim, Site::Proj)?;
        let txt = gate_residual(&scope, bp, txt, t_g1, tp, seqr, dim)?;

        // --- MLP sublayer ---
        let im2 = norm_modulate(&scope, bp, img, i_sc2, i_sh2, n, dim)?;
        let imlp = mlp(
            &scope, bp, im2, &b.img_fc1, &b.img_fc2, n, dim, mlp_h, None, None,
        )?;
        let img = gate_residual(&scope, bp, img, i_g2, imlp, n, dim)?;
        let tm2 = norm_modulate(&scope, bp, txt, t_sc2, t_sh2, seqr, dim)?;
        let tmlp = mlp(
            &scope, bp, tm2, &b.txt_fc1, &b.txt_fc2, seqr, dim, mlp_h, None, None,
        )?;
        let txt = gate_residual(&scope, bp, txt, t_g2, tmlp, seqr, dim)?;

        let di = scope.import_copy(img_nxt.as_buf_ref());
        scope.copy_buffer_to_buffer(img, 0, di, 0, (n * dim) as u64 * asz)?;
        let dt = scope.import_copy(txt_nxt.as_buf_ref());
        scope.copy_buffer_to_buffer(txt, 0, dt, 0, (seqr * dim) as u64 * asz)?;
        scope.submit_void().await?;
        Ok(())
    }

    /// 4-step (CFG-off) flow-match Euler denoise. `init_latent` is the pinned
    /// noise `[32, T, H, W]` (CTHW row-major); returns the final latent in the
    /// same layout (the VAE decoder's input). Each step packs the DiT input
    /// `[THW, 65] = [latent32 | 0 | 0]` (T2V cond block is zero), predicts the
    /// velocity, and integrates `x += dt * v`.
    #[allow(clippy::too_many_arguments)]
    pub async fn denoise<S: WeightSource>(
        &self,
        backend: &WgpuBackend,
        residency: &WeightResidency<S>,
        ws: &Workspace<WgpuBackend>,
        text: &[f32],
        seq: usize,
        init_latent: &[f32],
        grid: (usize, usize, usize),
        schedule: &crate::hunyuan::scheduler::FlowMatchSchedule,
        window: u32,
        progress: Option<&dyn Fn(u32, u32)>,
        // Cooperative cancellation: polled at each step boundary (a step is
        // ~minutes at 480p). `Some(c)` returning true aborts with
        // `HunyuanDitError::Cancelled` before the next forward.
        cancel: Option<&dyn Fn() -> bool>,
    ) -> Result<Vec<f32>, HunyuanDitError<S::Error>> {
        let (gt, gh, gw) = grid;
        let thw = gt * gh * gw;
        let lat = cfg::LATENT_CHANNELS;
        let conv_in = cfg::CONV_IN_CHANNELS;
        assert_eq!(init_latent.len(), lat * thw, "init latent size");
        let n_steps = schedule.steps() as u32;
        let mut x = init_latent.to_vec(); // [c, t, h, w] = c*thw + n
        for i in 0..schedule.steps() {
            // Stop before the (expensive) forward if the user cancelled.
            if cancel.is_some_and(|c| c()) {
                return Err(HunyuanDitError::Cancelled);
            }
            // Report the 1-based step about to run (each ~minutes at 480p, so the
            // UI must advance per step, not just once before the loop).
            if let Some(p) = progress {
                p(i as u32 + 1, n_steps);
            }
            // Pack [THW, 65]: token n channel c<32 = x[c, n]; c>=32 = 0.
            let mut img_tokens = vec![0.0f32; thw * conv_in];
            for n in 0..thw {
                for c in 0..lat {
                    img_tokens[n * conv_in + c] = x[c * thw + n];
                }
            }
            let v = self
                .forward(
                    backend,
                    residency,
                    ws,
                    text,
                    seq,
                    &img_tokens,
                    grid,
                    schedule.timesteps[i],
                    window,
                    None,
                )
                .await?;
            // v is [THW, 32] token-major; x is [32, THW] channel-major.
            let dt = schedule.dt(i);
            for n in 0..thw {
                for c in 0..lat {
                    x[c * thw + n] += dt * v[n * lat + c];
                }
            }
        }
        Ok(x)
    }

    async fn acquire_top<'r, S: WeightSource>(
        &self,
        res: &'r WeightResidency<S>,
        backend: &WgpuBackend,
        pins: &mut Vec<GpuView<'r>>,
    ) -> Result<TopBufs, ResidencyError<S::Error, WgpuError>> {
        acquire_top(&self.handles.top, res, backend, pins).await
    }

    async fn acquire_block<'r, S: WeightSource>(
        &self,
        h: &BlockH,
        res: &'r WeightResidency<S>,
        backend: &WgpuBackend,
        pins: &mut Vec<GpuView<'r>>,
    ) -> Result<BlockBufs, ResidencyError<S::Error, WgpuError>> {
        acquire_block(h, res, backend, pins).await
    }
}

/// Pin + acquire the top-level (non-block) weights.
async fn acquire_top<'r, S: WeightSource>(
    h: &TopH,
    res: &'r WeightResidency<S>,
    backend: &WgpuBackend,
    pins: &mut Vec<GpuView<'r>>,
) -> Result<TopBufs, ResidencyError<S::Error, WgpuError>> {
    Ok(TopBufs {
        img_in: acq_lin(res, backend, h.img_in, pins).await?,
        time0: acq_lin(res, backend, h.time0, pins).await?,
        time2: acq_lin(res, backend, h.time2, pins).await?,
        cond_type: acq_one(res, backend, h.cond_type, pins).await?,
        final_adaln: acq_lin(res, backend, h.final_adaln, pins).await?,
        final_lin: acq_lin(res, backend, h.final_lin, pins).await?,
    })
}

/// Pin + acquire one dual-stream block's weights.
async fn acquire_block<'r, S: WeightSource>(
    h: &BlockH,
    res: &'r WeightResidency<S>,
    backend: &WgpuBackend,
    pins: &mut Vec<GpuView<'r>>,
) -> Result<BlockBufs, ResidencyError<S::Error, WgpuError>> {
    Ok(BlockBufs {
        img_mod: acq_lin(res, backend, h.img_mod, pins).await?,
        txt_mod: acq_lin(res, backend, h.txt_mod, pins).await?,
        img_q: acq_lin(res, backend, h.img_q, pins).await?,
        img_k: acq_lin(res, backend, h.img_k, pins).await?,
        img_v: acq_lin(res, backend, h.img_v, pins).await?,
        txt_q: acq_lin(res, backend, h.txt_q, pins).await?,
        txt_k: acq_lin(res, backend, h.txt_k, pins).await?,
        txt_v: acq_lin(res, backend, h.txt_v, pins).await?,
        img_qn: acq_one(res, backend, h.img_qn, pins).await?,
        img_kn: acq_one(res, backend, h.img_kn, pins).await?,
        txt_qn: acq_one(res, backend, h.txt_qn, pins).await?,
        txt_kn: acq_one(res, backend, h.txt_kn, pins).await?,
        img_proj: acq_lin(res, backend, h.img_proj, pins).await?,
        txt_proj: acq_lin(res, backend, h.txt_proj, pins).await?,
        img_fc1: acq_lin(res, backend, h.img_fc1, pins).await?,
        img_fc2: acq_lin(res, backend, h.img_fc2, pins).await?,
        txt_fc1: acq_lin(res, backend, h.txt_fc1, pins).await?,
        txt_fc2: acq_lin(res, backend, h.txt_fc2, pins).await?,
    })
}

async fn acq_one<'r, S: WeightSource>(
    res: &'r WeightResidency<S>,
    backend: &WgpuBackend,
    h: WeightHandle,
    pins: &mut Vec<GpuView<'r>>,
) -> Result<BufRef, ResidencyError<S::Error, WgpuError>> {
    let v = res.acquire(h, backend).await?;
    let b = v.buf();
    pins.push(v);
    Ok(b)
}

async fn acq_lin<'r, S: WeightSource>(
    res: &'r WeightResidency<S>,
    backend: &WgpuBackend,
    h: LinH,
    pins: &mut Vec<GpuView<'r>>,
) -> Result<LinBufs, ResidencyError<S::Error, WgpuError>> {
    let wv = res.acquire(h.weight, backend).await?;
    let bv = res.acquire(h.bias, backend).await?;
    let bufs = LinBufs {
        weight: wv.buf(),
        bias: bv.buf(),
    };
    pins.push(wv);
    pins.push(bv);
    Ok(bufs)
}

/// Concat `a` (`a_bytes`) followed by `c` (`c_bytes`) into one buffer.
fn concat2<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    a: BatchBuf<'w>,
    c: BatchBuf<'w>,
    a_bytes: u64,
    c_bytes: u64,
) -> Result<BatchBuf<'w>, WgpuError> {
    let out = scope.alloc(a_bytes + c_bytes)?;
    scope.copy_buffer_to_buffer(a, 0, out, 0, a_bytes)?;
    scope.copy_buffer_to_buffer(c, 0, out, a_bytes, c_bytes)?;
    Ok(out)
}

fn persist<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    ws: &Workspace<WgpuBackend>,
    buf: BatchBuf<'w>,
    n: usize,
    asz: u64,
) -> Result<WsBuf<WgpuBackend>, WgpuError> {
    let dst = ws.alloc(n as u64 * asz)?;
    let d = scope.import_copy(dst.as_buf_ref());
    scope.copy_buffer_to_buffer(buf, 0, d, 0, n as u64 * asz)?;
    Ok(dst)
}

async fn read_acts(
    backend: &WgpuBackend,
    buf: &BufRef,
    n: usize,
    act: ActDtype,
) -> Result<Vec<f32>, WgpuError> {
    let asz = act.bytes_per_elem();
    let bytes = backend
        .read_buffer(buf.id, buf.offset, n as u64 * asz)
        .await?;
    Ok(act_readback_to_f32(act, &bytes, n))
}
