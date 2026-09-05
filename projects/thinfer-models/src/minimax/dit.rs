//! The MiniMax-H3 diffusion transformer: 50 uniform blocks over ONE packed
//! sequence holding text, audio and video rows at once.
//!
//! Full self-attention across the whole sequence. There is no cross-attention,
//! no per-modality block weights and (for t2va) no attention mask: the packed
//! layout emits no padding, which keeps the fast unmasked sdpa path.
//!
//! Checkpoint layout quirks, handled at READ time rather than by rewriting
//! weights (the reference does the same, and permuting quantized rows at load
//! is the kind of silent scramble that produces structured artifacts with no
//! NaN to catch it):
//!
//! - `attn.qkv_proj` output columns are PER-HEAD INTERLEAVED,
//!   `[h0: q,k,v][h1: q,k,v]...` -> [`QkvSplitHeadwiseF32`].
//! - `mlp.fc1` is a fused `[gate; value]` SwiGLU projection ->
//!   [`SiluMulFusedF32`].
//! - RoPE rotates only 96 of each head's 128 channels ->
//!   [`RopeF32HalfRotPartial`].
//!
//! Modulation never runs a gather. `build_packed_sequence` groups rows by
//! modality and every row of a modality shares one timestep, so rows sharing an
//! AdaLN table row are CONTIGUOUS ([`ModulationPlan::segments`]). Each block
//! drives the existing broadcast `bcast_modulate` / `bcast_fma` kernels once per
//! segment against an offset slice of the precomputed table. A t2va request has
//! exactly three segments.

use thinfer_core::backend::{BufRef, WgpuBackend, WgpuError, WgpuPipeline};
use thinfer_core::ops::{
    BcastFmaF32, BcastModulateF32, QkvSplitHeadwiseF32, QkvSplitOp, RopeF32HalfRotPartial, RopeOp,
    SiluMulFusedF32, SiluMulFusedOp,
};
use thinfer_core::quant::QuantKind;
use thinfer_core::residency::{
    GpuView, ResidencyError, TransposePolicy, WeightHandle, WeightResidency,
};
use thinfer_core::tensor::StorageEncoding;
use thinfer_core::weight::{WeightId, WeightSource};
use thinfer_core::workspace::{BatchBuf, BatchScope};

use std::collections::HashMap;

use crate::common::block::{
    ActBuf, Block, BlockPipelines, BlockWgslConfigs, CoopmatStep, DequantStep, alloc_act,
    alloc_matmul_out_buf, build_coopmat_step, op_add, op_rmsnorm, op_sdpa,
};
use crate::minimax::DitConfig;
use crate::minimax::adaln::{
    GATE_MLP, GATE_MSA, ModSegment, ModulationLayout, SCALE_MLP, SCALE_MSA, SHIFT_MLP, SHIFT_MSA,
};
use crate::z_image::text_encoder::{LoadError, register_one};

/// Per-site resident quant kinds. The shipped Q4_K_M DiT is mixed per tensor,
/// so a single compile-time scheme is wrong.
#[derive(Clone, Copy, Debug, Default)]
pub struct MatmulKinds {
    pub qkv: Option<QuantKind>,
    pub out: Option<QuantKind>,
    pub fc1: Option<QuantKind>,
    pub fc2: Option<QuantKind>,
}

/// Tensor names of one transformer block. These are the ORIGINAL checkpoint
/// names, which the shipped GGUF preserves verbatim (verified 2026-08-07).
struct BlockWeights {
    norm1: WeightId,
    norm2: WeightId,
    qkv_proj: WeightId,
    q_norm: WeightId,
    k_norm: WeightId,
    out_proj: WeightId,
    fc1: WeightId,
    fc2: WeightId,
}

impl BlockWeights {
    /// `prefix` is `blocks.{i}` for the main stack, `token_refiner.blocks.{i}`
    /// for the refiner. The two share a block shape; only modulation differs.
    fn new(prefix: &str) -> Self {
        let id = |s: &str| WeightId(format!("{prefix}.{s}"));
        Self {
            norm1: id("norm1.weight"),
            norm2: id("norm2.weight"),
            qkv_proj: id("attn.qkv_proj.weight"),
            q_norm: id("attn.q_norm.weight"),
            k_norm: id("attn.k_norm.weight"),
            out_proj: id("attn.out_proj.weight"),
            fc1: id("mlp.fc1.weight"),
            fc2: id("mlp.fc2.weight"),
        }
    }
}

/// One matmul site's low-rank adapter: `delta = B @ A`, `A` `[rank, K]` and
/// `B` `[N, rank]`.
#[derive(Clone, Copy, Debug)]
pub struct SiteLora {
    a: WeightHandle,
    b: WeightHandle,
    rank: u32,
}

/// The turbo distillation's adapters for one block.
///
/// These are applied at RUNTIME rather than folded into the base tensors. The
/// fold would re-encode every Q4_K site to Q8_0 (see `common::lora`), taking the
/// streamed working set from ~10.8 GiB per step to ~20.5 GiB and parking the
/// same again in the host fold cache. The runtime form costs ~5% more MACs
/// (rank 384 on the fused qkv, 128 elsewhere, against a 385M-MAC block), needs
/// no requantization, and so is also exact where a Q4_K refold would not be.
#[derive(Clone, Copy, Debug)]
pub struct BlockLoraHandles {
    qkv: SiteLora,
    out: SiteLora,
    fc1: SiteLora,
    fc2: SiteLora,
}

#[derive(Clone, Copy, Debug)]
pub struct SiteLoraBufs {
    a: BufRef,
    b: BufRef,
    rank: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct BlockLoraBufs {
    qkv: SiteLoraBufs,
    out: SiteLoraBufs,
    fc1: SiteLoraBufs,
    fc2: SiteLoraBufs,
}

struct SiteLoraViews<'a> {
    a: GpuView<'a>,
    b: GpuView<'a>,
    rank: u32,
}

pub struct BlockLoraViews<'a> {
    qkv: SiteLoraViews<'a>,
    out: SiteLoraViews<'a>,
    fc1: SiteLoraViews<'a>,
    fc2: SiteLoraViews<'a>,
}

#[derive(Clone, Copy, Debug)]
pub struct BlockHandles {
    norm1: WeightHandle,
    norm2: WeightHandle,
    qkv_proj: WeightHandle,
    q_norm: WeightHandle,
    k_norm: WeightHandle,
    out_proj: WeightHandle,
    fc1: WeightHandle,
    fc2: WeightHandle,
    kinds: MatmulKinds,
    lora: Option<BlockLoraHandles>,
}

pub struct BlockViews<'a> {
    norm1: GpuView<'a>,
    norm2: GpuView<'a>,
    qkv_proj: GpuView<'a>,
    q_norm: GpuView<'a>,
    k_norm: GpuView<'a>,
    out_proj: GpuView<'a>,
    fc1: GpuView<'a>,
    fc2: GpuView<'a>,
    lora: Option<BlockLoraViews<'a>>,
}

#[derive(Clone, Copy, Debug)]
pub struct BlockBufs {
    norm1: BufRef,
    norm2: BufRef,
    qkv_proj: BufRef,
    q_norm: BufRef,
    k_norm: BufRef,
    out_proj: BufRef,
    fc1: BufRef,
    fc2: BufRef,
    lora: Option<BlockLoraBufs>,
}

impl BlockHandles {
    /// Resident quant kind per matmul site, for the caller's dequant dispatch.
    pub fn kinds(&self) -> MatmulKinds {
        self.kinds
    }

    pub async fn acquire<'a, S: WeightSource>(
        &self,
        residency: &'a WeightResidency<S>,
        backend: &WgpuBackend,
    ) -> Result<BlockViews<'a>, ResidencyError<S::Error, WgpuError>> {
        let lora = match self.lora {
            None => None,
            Some(l) => Some(BlockLoraViews {
                qkv: l.qkv.acquire(residency, backend).await?,
                out: l.out.acquire(residency, backend).await?,
                fc1: l.fc1.acquire(residency, backend).await?,
                fc2: l.fc2.acquire(residency, backend).await?,
            }),
        };
        Ok(BlockViews {
            norm1: residency.acquire(self.norm1, backend).await?,
            norm2: residency.acquire(self.norm2, backend).await?,
            qkv_proj: residency.acquire(self.qkv_proj, backend).await?,
            q_norm: residency.acquire(self.q_norm, backend).await?,
            k_norm: residency.acquire(self.k_norm, backend).await?,
            out_proj: residency.acquire(self.out_proj, backend).await?,
            fc1: residency.acquire(self.fc1, backend).await?,
            fc2: residency.acquire(self.fc2, backend).await?,
            lora,
        })
    }
}

impl SiteLora {
    async fn acquire<'a, S: WeightSource>(
        &self,
        residency: &'a WeightResidency<S>,
        backend: &WgpuBackend,
    ) -> Result<SiteLoraViews<'a>, ResidencyError<S::Error, WgpuError>> {
        Ok(SiteLoraViews {
            a: residency.acquire(self.a, backend).await?,
            b: residency.acquire(self.b, backend).await?,
            rank: self.rank,
        })
    }
}

impl SiteLoraViews<'_> {
    fn bufs(&self) -> SiteLoraBufs {
        SiteLoraBufs {
            a: self.a.buf(),
            b: self.b.buf(),
            rank: self.rank,
        }
    }
}

impl BlockViews<'_> {
    pub fn bufs(&self) -> BlockBufs {
        BlockBufs {
            norm1: self.norm1.buf(),
            norm2: self.norm2.buf(),
            qkv_proj: self.qkv_proj.buf(),
            q_norm: self.q_norm.buf(),
            k_norm: self.k_norm.buf(),
            out_proj: self.out_proj.buf(),
            fc1: self.fc1.buf(),
            fc2: self.fc2.buf(),
            lora: self.lora.as_ref().map(|l| BlockLoraBufs {
                qkv: l.qkv.bufs(),
                out: l.out.bufs(),
                fc1: l.fc1.bufs(),
                fc2: l.fc2.bufs(),
            }),
        }
    }
}

/// Every weight the DiT needs, minus the `adaln_proj` projections: those are
/// consumed once at load to build the modulation table and then never read
/// again (see [`crate::minimax::adaln`]), which is what keeps ~13B of the 33B
/// off the streaming path.
#[derive(Clone, Debug)]
pub struct DitHandles {
    pub blocks: Vec<BlockHandles>,
    pub refiner: Vec<BlockHandles>,
    pub refiner_final_norm: WeightHandle,
    pub video_patch_proj: WeightHandle,
    pub video_patch_proj_bias: WeightHandle,
    pub audio_patch_proj: WeightHandle,
    pub audio_patch_proj_bias: WeightHandle,
    pub condition_proj: WeightHandle,
    pub condition_proj_bias: WeightHandle,
    pub final_norm: WeightHandle,
    pub video_out: WeightHandle,
    pub video_out_bias: WeightHandle,
    pub audio_out: WeightHandle,
    pub audio_out_bias: WeightHandle,
}

impl DitHandles {
    /// The same stack with every runtime adapter dropped, i.e. the base
    /// checkpoint alone. Lets a caller A/B the adapter's effect inside ONE
    /// residency rather than standing up a second GPU budget.
    pub fn without_lora(&self) -> Self {
        let strip = |b: &[BlockHandles]| -> Vec<BlockHandles> {
            b.iter()
                .map(|h| BlockHandles { lora: None, ..*h })
                .collect()
        };
        Self {
            blocks: strip(&self.blocks),
            refiner: strip(&self.refiner),
            ..self.clone()
        }
    }

    /// Blocks carrying a runtime adapter.
    pub fn lora_block_count(&self) -> usize {
        self.blocks
            .iter()
            .chain(&self.refiner)
            .filter(|h| h.lora.is_some())
            .count()
    }
}

/// Register the DiT weights. `adaln_proj` is deliberately absent: the caller
/// projects it into the modulation table at load and drops it.
pub fn register_handles<S: WeightSource>(
    cfg: &DitConfig,
    residency: &WeightResidency<S>,
    transcode: Option<QuantKind>,
) -> Result<DitHandles, LoadError> {
    let lin = |id: &WeightId| register_one(residency, id, TransposePolicy::Linear2D, transcode);
    let dense = |id: &WeightId| register_one(residency, id, TransposePolicy::None, None);
    let kind = |id: &WeightId, requant: Option<QuantKind>| -> Option<QuantKind> {
        requant.or_else(|| {
            match residency
                .source()
                .catalog()
                .get(id)
                .and_then(|e| e.encoding)
            {
                Some(StorageEncoding::Quant(k)) => Some(k),
                _ => None,
            }
        })
    };

    // A runtime adapter is present only if its tensors were merged into this
    // source (see `common::merge`); a bare checkpoint registers none and the
    // block stack runs the base weights unchanged.
    let reg_site_lora = |prefix: &str, site: &str| -> Result<Option<SiteLora>, LoadError> {
        let a_id = WeightId(format!("diffusion_model.{prefix}.{site}.lora_A.weight"));
        let b_id = WeightId(format!("diffusion_model.{prefix}.{site}.lora_B.weight"));
        let cat = residency.source().catalog();
        let (Some(ae), Some(_)) = (cat.get(&a_id), cat.get(&b_id)) else {
            return Ok(None);
        };
        // `A` is `[rank, K]`: the rank is read, never assumed.
        let rank = ae.shape.0[0] as u32;
        Ok(Some(SiteLora {
            a: register_one(residency, &a_id, TransposePolicy::Linear2D, None)?,
            b: register_one(residency, &b_id, TransposePolicy::Linear2D, None)?,
            rank,
        }))
    };
    let reg_lora = |prefix: &str| -> Result<Option<BlockLoraHandles>, LoadError> {
        let (Some(qkv), Some(out), Some(fc1), Some(fc2)) = (
            reg_site_lora(prefix, "attn.qkv_proj")?,
            reg_site_lora(prefix, "attn.out_proj")?,
            reg_site_lora(prefix, "mlp.fc1")?,
            reg_site_lora(prefix, "mlp.fc2")?,
        ) else {
            return Ok(None);
        };
        Ok(Some(BlockLoraHandles { qkv, out, fc1, fc2 }))
    };

    let reg_block = |prefix: String| -> Result<BlockHandles, LoadError> {
        let b = BlockWeights::new(&prefix);
        let lora = reg_lora(&prefix)?;
        Ok(BlockHandles {
            lora,
            norm1: dense(&b.norm1)?,
            norm2: dense(&b.norm2)?,
            qkv_proj: lin(&b.qkv_proj)?,
            q_norm: dense(&b.q_norm)?,
            k_norm: dense(&b.k_norm)?,
            out_proj: lin(&b.out_proj)?,
            fc1: lin(&b.fc1)?,
            // ffn_down is never transcoded: the massive-activation lesson.
            fc2: register_one(residency, &b.fc2, TransposePolicy::Linear2D, None)?,
            kinds: MatmulKinds {
                qkv: kind(&b.qkv_proj, transcode),
                out: kind(&b.out_proj, transcode),
                fc1: kind(&b.fc1, transcode),
                fc2: kind(&b.fc2, None),
            },
        })
    };

    let blocks = (0..cfg.n_layers)
        .map(|i| reg_block(format!("blocks.{i}")))
        .collect::<Result<Vec<_>, _>>()?;
    let refiner = (0..cfg.token_refiner_layers)
        .map(|i| reg_block(format!("token_refiner.blocks.{i}")))
        .collect::<Result<Vec<_>, _>>()?;

    let w = |s: &str| WeightId(s.to_string());
    Ok(DitHandles {
        blocks,
        refiner,
        refiner_final_norm: dense(&w("token_refiner.final_norm.weight"))?,
        video_patch_proj: lin(&w("video_patch_proj.weight"))?,
        video_patch_proj_bias: dense(&w("video_patch_proj.bias"))?,
        audio_patch_proj: lin(&w("audio_patch_proj.weight"))?,
        audio_patch_proj_bias: dense(&w("audio_patch_proj.bias"))?,
        condition_proj: lin(&w("condition_proj.weight"))?,
        condition_proj_bias: dense(&w("condition_proj.bias"))?,
        final_norm: dense(&w("final_layer.norm.weight"))?,
        video_out: lin(&w("final_layer.video_out.weight"))?,
        video_out_bias: dense(&w("final_layer.video_out.bias"))?,
        audio_out: lin(&w("final_layer.audio_out.weight"))?,
        audio_out_bias: dense(&w("final_layer.audio_out.bias"))?,
    })
}

/// Pipeline cfgs for the DiT. Quant weights pair with F16 acts where the
/// adapter exposes `SHADER_F16`, else F32 (see `plan-details`); the residual
/// itself stays wide because H3's is a 5376-wide single stream carrying every
/// modality.
/// Sites production runs on the tensor cores. Currently NONE.
///
/// The plumbing is complete and roughly halves the DiT matmul time (96s -> 48s
/// at 759 rows), but the `dit_health` gate measures a ~9% divergence from the
/// dense reference on EVERY site independently -- a common-mode effect of the
/// f16 A-side cast rather than a per-site overflow. Nine percent is far more
/// than rounding should cost, and H3 has NO parity reference yet, so nothing
/// can say which side is right. Enabling it would trade unknown quality for
/// speed, which is backwards. Revisit once a pyref DiT parity gate exists.
pub const COOPMAT_SITES: crate::common::block::CoopmatSites = crate::common::block::CoopmatSites {
    qkv: false,
    proj: false,
    ffn_up: false,
    // Unavailable regardless: this site's A-side IS the SwiGLU output, whose
    // channels demonstrably exceed the f16 range on this model (that is why
    // `ffn_up` needs the f32-output variant), and an f32 output cannot rescue
    // an input-side clamp.
    ffn_down: false,
};

/// `coopmat` selects which matmul sites take the tensor-core path. Production
/// passes [`COOPMAT_SITES`]; the conformance gate varies it so it can hold each
/// site against the reference dense matmul independently.
pub fn dit_cfgs(
    weight_dtype: thinfer_core::ops::WeightDtype,
    act_dtype: thinfer_core::ops::ActDtype,
    coopmat: crate::common::block::CoopmatSites,
) -> BlockWgslConfigs {
    use thinfer_core::ops::WgslConfig;
    let ops = WgslConfig {
        bf16_quant_writes: false,
        act_dtype,
        weight_dtype: thinfer_core::ops::WeightDtype::Bf16,
    };
    // `weight_dtype` here selects the B-operand CONVENTION, not what the kernel
    // reads: a `Quant` mm config means "the site pre-dequants to a dense
    // N-major workspace", which is exactly what the per-kind dequant below
    // produces. Declaring `Bf16` instead would select the K-major layout and
    // silently transpose every weight -- no NaN, just a scrambled model.
    let mm = WgslConfig {
        weight_dtype,
        ..ops
    };
    BlockWgslConfigs {
        matmul_qkv: mm,
        matmul_qkv_self: mm,
        matmul_proj: mm,
        matmul_ffn_up: mm,
        matmul_ffn_down: mm,
        matmul_adaln: ops,
        ops,
        i8_sdpa: false,
        dense_acts: crate::common::block::DenseActSites {
            qkv: true,
            qkv_self: true,
            proj: true,
            ffn_up: true,
            ffn_down: true,
        },
        // Tensor cores on the three sites whose A-side stays inside the f16
        // range: `qkv` and `ffn_up` read the modulated pre-norm (O(1); the
        // precomputed modulation peaks around 17) and `proj` reads the SDPA
        // output (a convex combination of V). `ffn_up` additionally needs the
        // f32-OUTPUT variant: its 28672-wide pre-SwiGLU activation carries
        // channels past the f16 max, and one inf there becomes a NaN once
        // SwiGLU multiplies it (observed: the whole decode went non-finite).
        //
        // `ffn_down` stays dense, and not for lack of trying: its A-side IS
        // that SwiGLU output, so the f16 A-side cast would reintroduce exactly
        // the overflow the f32 output was added to dodge. An f32 output cannot
        // help an input-side clamp.
        coopmat_acts: coopmat,
        // `head_dim` is exactly 128, so the wide-head kernel is never routed to
        // (dispatch takes it only above 128) and it does not build under bf16
        // acts. Keep it off and let the act dtype decide the SDPA path.
        large_d_sdpa: false,
        // Mixed-precision SDPA on a bf16 block: the post-norm Q/K/V are cast to
        // f16 to reach the subgroup kernel, which is otherwise F16-acts-only.
        // Safe HERE specifically because both are per-head RMSNormed just
        // before, so they are O(1) and nowhere near the f16 range -- unlike the
        // residual stream, which stays bf16 for exactly that reason.
        fast_sdpa: act_dtype == thinfer_core::ops::ActDtype::Bf16,
        decode_sdpa: false,
    }
}

/// `BlockPipelines` plus the three H3-specific kernels and the per-kind dequant.
pub struct DitPipelines {
    pub block: BlockPipelines,
    qkv_split_headwise: WgpuPipeline,
    rope_partial: WgpuPipeline,
    silu_mul_fused: WgpuPipeline,
    dense_dequant: HashMap<QuantKind, DequantStep>,
    /// Tensor-core matmul, keyed by quant kind.
    ///
    /// H3 is the video-gen hot path coopmat exists for, but the shipped
    /// `Q4_K_M` checkpoint mixes schemes PER TENSOR, so one compile-time scheme
    /// (what `BlockPipelines` bakes per site) would silently misdecode every
    /// tensor of the other kind. Keyed by kind, each matmul routes by its own
    /// catalog entry. Empty when the device has no usable coopmat config or the
    /// block is not running the bf16<->f16 casts, in which case every site
    /// falls back to the dense path.
    coopmat: HashMap<QuantKind, CoopmatStep>,
    /// The same, writing an f32 output. Used by `ffn_up`, whose 28672-wide
    /// pre-SwiGLU activation carries channels that overflow f16.
    coopmat_f32: HashMap<QuantKind, CoopmatStep>,
    /// Which sites are allowed on the tensor-core path.
    ///
    /// Coopmat computes in f16, so a site is only eligible if BOTH its A-side
    /// and its output stay inside the f16 range. That is a per-site property,
    /// not a per-model one: `fc1` emits the 28672-wide pre-SwiGLU activation,
    /// which is exactly where a big DiT's outlier channels live, and one value
    /// past 65504 becomes an inf that SwiGLU turns into a NaN.
    coopmat_sites: crate::common::block::CoopmatSites,
}

impl DitPipelines {
    pub async fn compile(
        backend: &WgpuBackend,
        cfgs: &BlockWgslConfigs,
    ) -> Result<Self, WgpuError> {
        use thinfer_core::backend::Backend;
        let block = BlockPipelines::compile(backend, cfgs).await?;
        let qkv_split_headwise = backend
            .create_pipeline(
                "minimax_qkv_split_headwise",
                <QkvSplitHeadwiseF32 as QkvSplitOp>::wgsl(&cfgs.ops),
                "main",
                <QkvSplitHeadwiseF32 as QkvSplitOp>::layout(),
            )
            .await?;
        let rope_partial = backend
            .create_pipeline(
                "minimax_rope_halfrot_partial",
                <RopeF32HalfRotPartial as RopeOp>::wgsl(&cfgs.ops),
                "main",
                <RopeF32HalfRotPartial as RopeOp>::layout(),
            )
            .await?;
        let silu_mul_fused = backend
            .create_pipeline(
                "minimax_silu_mul_fused",
                <SiluMulFusedF32 as SiluMulFusedOp>::wgsl(&cfgs.ops),
                "main",
                <SiluMulFusedF32 as SiluMulFusedOp>::layout(),
            )
            .await?;

        use thinfer_core::ops::dequant::{DequantTarget, build_wgsl, layout};
        let dq_layout = layout();
        let mut dense_dequant = HashMap::new();
        for scheme in [
            QuantKind::Q8_0,
            QuantKind::Q4_K,
            QuantKind::Q5_K,
            QuantKind::Q6_K,
        ] {
            let wgsl = build_wgsl(scheme, DequantTarget::Bf16);
            let pipeline = backend
                .create_pipeline(
                    &format!("minimax_dit_dequant_{scheme:?}"),
                    &wgsl,
                    "main",
                    dq_layout,
                )
                .await?;
            dense_dequant.insert(scheme, DequantStep { pipeline, scheme });
        }
        // The tensor-core path needs the bf16<->f16 act casts, which only exist
        // on a bf16 block that opted into the mixed-precision SDPA.
        use thinfer_core::ops::matmul_coopmat::CoopmatOut;
        let mut coopmat = HashMap::new();
        let mut coopmat_f32 = HashMap::new();
        if cfgs.fast_sdpa && cfgs.ops.act_dtype == thinfer_core::ops::ActDtype::Bf16 {
            for scheme in [QuantKind::Q4_K, QuantKind::Q6_K, QuantKind::Q8_0] {
                for (out, map) in [
                    (CoopmatOut::F16, &mut coopmat),
                    (CoopmatOut::F32, &mut coopmat_f32),
                ] {
                    let label = format!(
                        "minimax_{scheme:?}_{}",
                        if matches!(out, CoopmatOut::F32) {
                            "f32"
                        } else {
                            "f16"
                        }
                    );
                    if let Some(step) = build_coopmat_step(backend, &label, scheme, out).await? {
                        map.insert(scheme, step);
                    }
                }
            }
        }
        tracing::info!(
            target: thinfer_core::trace::ADAPTER,
            coopmat_kinds = coopmat.len(),
            "minimax dit pipelines",
        );
        Ok(Self {
            block,
            qkv_split_headwise,
            rope_partial,
            silu_mul_fused,
            dense_dequant,
            coopmat,
            coopmat_f32,
            coopmat_sites: cfgs.coopmat_acts,
        })
    }

    fn dequant_for(&self, kind: Option<QuantKind>) -> Option<&DequantStep> {
        kind.and_then(|k| self.dense_dequant.get(&k))
    }

    /// `wide` selects the f32-output variant, for a site whose values can
    /// leave the f16 range.
    fn coopmat_for(
        &self,
        kind: Option<QuantKind>,
        site_enabled: bool,
        wide: bool,
    ) -> Option<&CoopmatStep> {
        if !site_enabled {
            return None;
        }
        let map = if wide {
            &self.coopmat_f32
        } else {
            &self.coopmat
        };
        kind.and_then(|k| map.get(&k))
    }
}

/// The precomputed modulation table, resident for the whole run.
///
/// One buffer per block, `[num_timesteps * 3, 6 * hidden]`, plus the final
/// layer's `[num_timesteps, 2 * hidden]`. Tiny (~48 MB at 4 steps) next to the
/// ~7 GB of `adaln_proj` it replaces.
pub struct ModulationTable {
    pub layout: ModulationLayout,
    /// One per transformer block, in block order.
    pub blocks: Vec<BufRef>,
    /// `[num_timesteps, 2 * hidden]`: shift then scale.
    pub final_layer: BufRef,
}

impl ModulationTable {
    /// Byte view of one parameter for one table row.
    ///
    /// Every offset here is a multiple of `hidden * bytes` (5376 * 4 = 21504 =
    /// 84 * 256), so bindings stay inside WebGPU's storage-offset alignment.
    fn param(&self, block: usize, table_row: u32, param: usize, elem_bytes: u64) -> BufRef {
        let hidden = self.layout.hidden as u64;
        let row_w = self.layout.row_width() as u64;
        let base = self.blocks[block];
        let off = (table_row as u64 * row_w + param as u64 * hidden) * elem_bytes;
        BufRef::view(base.id, base.offset + off, hidden * elem_bytes)
    }

    fn final_param(&self, timestep_index: u32, param: usize, elem_bytes: u64) -> BufRef {
        let hidden = self.layout.hidden as u64;
        let off = (timestep_index as u64 * 2 * hidden + param as u64 * hidden) * elem_bytes;
        BufRef::view(
            self.final_layer.id,
            self.final_layer.offset + off,
            hidden * elem_bytes,
        )
    }
}

/// Shape of one forward.
#[derive(Clone, Copy, Debug)]
pub struct DitShape {
    pub cfg: DitConfig,
    /// Packed sequence length. No padding is emitted.
    pub rows: usize,
}

impl DitShape {
    fn sdpa_scale(&self) -> f32 {
        1.0 / (self.cfg.head_dim as f32).sqrt()
    }
}

/// One transformer block: pre-norm modulated attention -> gated residual ->
/// pre-norm modulated SwiGLU -> gated residual.
///
/// `segments` is empty for the token refiner, which runs the same block shape
/// with NO modulation and NO rope (it refines the projected text stream).
#[allow(clippy::too_many_arguments)]
fn block_forward<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    pipelines: &DitPipelines,
    shape: DitShape,
    x_in: BatchBuf<'wsp>,
    freqs_in: Option<BatchBuf<'wsp>>,
    y_out: BatchBuf<'wsp>,
    bufs: &'wsp BlockBufs,
    kinds: MatmulKinds,
    modulation: Option<(&ModulationTable, usize, &[ModSegment])>,
) -> Result<(), WgpuError> {
    let p = &pipelines.block;
    let c = shape.cfg;
    let rows = shape.rows as u32;
    let dim = c.hidden as u32;
    let inner = c.inner_dim() as u32;
    let hd = c.head_dim as u32;
    let heads = c.n_heads as u32;
    let ffn = c.ffn_hidden as u32;
    let elem_bytes = p.act_bytes(1);

    let x_in = ActBuf::dense(x_in);
    let y_out = ActBuf::dense(y_out);
    let norm1_w = scope.import(&bufs.norm1);
    let norm2_w = scope.import(&bufs.norm2);
    let qkv_w = scope.import(&bufs.qkv_proj);
    let qn_w = scope.import(&bufs.q_norm);
    let kn_w = scope.import(&bufs.k_norm);
    let out_w = scope.import(&bufs.out_proj);
    let fc1_w = scope.import(&bufs.fc1);
    let fc2_w = scope.import(&bufs.fc2);

    // A dense bf16 matmul with no i8 path and no dequant: the adapter tensors
    // are bf16 already, and the rank-side intermediate is NOT normed, so
    // quantizing its activations would reintroduce exactly the outlier problem
    // the i8 siting rules exist to avoid.
    let dense_mm = |src: ActBuf<'wsp>,
                    w: BatchBuf<'wsp>,
                    dst: BatchBuf<'wsp>,
                    out_w: u32,
                    in_w: u32|
     -> Result<(), WgpuError> {
        let dims = scope.u32x4_uniform(rows, out_w, in_w, 0)?;
        Block::dispatch_matmul_site(
            scope,
            p,
            src,
            w,
            dst,
            dims,
            None,
            None,
            None,
            // Dense bf16 adapter weights: the MODULE pipeline, which reads a
            // `Linear2D [K, N]` weight directly. The block pipeline expects the
            // N-major workspace the per-site dequant produces, so feeding it a
            // dense weight would transpose it silently.
            &p.matmul_module,
            &p.matmuls.module,
            rows,
            out_w,
            in_w,
        )
    };

    let mm = |src: ActBuf<'wsp>,
              w,
              out_w: u32,
              in_w: u32,
              kind: Option<QuantKind>,
              coopmat_ok: bool,
              wide_out: bool,
              lora: Option<SiteLoraBufs>|
     -> Result<BatchBuf<'wsp>, WgpuError> {
        let scratch = alloc_matmul_out_buf(scope, p, rows * out_w)?;
        let dims = scope.u32x4_uniform(rows, out_w, in_w, 0)?;
        // Tensor cores when this tensor's quant kind has a coopmat step, the
        // dense per-kind dequant otherwise. `dispatch_matmul_site_coopmat`
        // itself falls back for a sub-WM dispatch (which device-loses).
        Block::dispatch_matmul_site_coopmat(
            scope,
            p,
            src,
            w,
            scratch,
            dims,
            None,
            pipelines.coopmat_for(kind, coopmat_ok, wide_out),
            p.matmul_i8_qkv.as_ref(),
            p.dequant_i8_qkv.as_ref(),
            pipelines.dequant_for(kind),
            // The B-operand layout follows the site, not the model: a QUANT
            // site pre-dequants to an N-major workspace (the block pipeline),
            // while an unquantized one (the F16 token refiner, and the F16
            // patch/head projections) is read as a dense `Linear2D [K, N]` (the
            // module pipeline). Using one pipeline for both transposes half the
            // weights with no NaN to show for it.
            if kind.is_some() {
                &p.matmul_qkv
            } else {
                &p.matmul_module
            },
            if kind.is_some() {
                &p.matmuls.qkv
            } else {
                &p.matmuls.module
            },
            rows,
            out_w,
            in_w,
        )?;
        let Some(l) = lora else {
            return Ok(scratch);
        };
        // `src @ A^T` then `@ B^T`, added to the base output. The sum lands in
        // a THIRD buffer: one buffer cannot be both the read and the read_write
        // operand of a single dispatch.
        let mid = alloc_matmul_out_buf(scope, p, rows * l.rank)?;
        dense_mm(src, scope.import_copy(l.a), mid, l.rank, in_w)?;
        let delta = alloc_matmul_out_buf(scope, p, rows * out_w)?;
        dense_mm(
            ActBuf::dense(mid),
            scope.import_copy(l.b),
            delta,
            out_w,
            l.rank,
        )?;
        let sum = alloc_matmul_out_buf(scope, p, rows * out_w)?;
        op_add(
            scope,
            p,
            ActBuf::dense(scratch),
            ActBuf::dense(delta),
            ActBuf::dense(sum),
        )?;
        Ok(sum)
    };

    // Apply `x * (1 + scale[seg]) + shift[seg]` per contiguous segment.
    let modulate = |src: &ActBuf<'wsp>,
                    dst: &ActBuf<'wsp>,
                    scale_param: usize,
                    shift_param: usize|
     -> Result<(), WgpuError> {
        let Some((table, block_idx, segments)) = modulation else {
            return Ok(());
        };
        for seg in segments {
            let off = seg.start as u64 * dim as u64 * elem_bytes;
            let len = seg.len as u64 * dim as u64 * elem_bytes;
            let (x_r, o_r) = (src.data.sub(off, len), dst.data.sub(off, len));
            let s_r = table.param(block_idx, seg.table_row, scale_param, elem_bytes);
            let t_r = table.param(block_idx, seg.table_row, shift_param, elem_bytes);
            let x = scope.import(&x_r);
            let o = scope.import(&o_r);
            let s = scope.import(&s_r);
            let t = scope.import(&t_r);
            // bias = 1.0 folds the `(1 + scale)` of AdaLN into the kernel.
            let u = scope.u32x4_uniform(dim, 1.0_f32.to_bits(), 0, 0)?;
            scope.bcast_modulate::<BcastModulateF32>(
                &p.bcast_modulate,
                x,
                s,
                t,
                u,
                o,
                seg.len * dim,
            )?;
        }
        Ok(())
    };

    // Apply `x + gate[seg] * y` per contiguous segment.
    let gate_add = |x: &ActBuf<'wsp>,
                    y: &ActBuf<'wsp>,
                    dst: &ActBuf<'wsp>,
                    gate_param: usize|
     -> Result<(), WgpuError> {
        let Some((table, block_idx, segments)) = modulation else {
            return op_add(scope, p, *x, *y, *dst);
        };
        for seg in segments {
            let off = seg.start as u64 * dim as u64 * elem_bytes;
            let len = seg.len as u64 * dim as u64 * elem_bytes;
            let x_r = x.data.sub(off, len);
            let y_r = y.data.sub(off, len);
            let o_r = dst.data.sub(off, len);
            let g_r = table.param(block_idx, seg.table_row, gate_param, elem_bytes);
            let xb = scope.import(&x_r);
            let yb = scope.import(&y_r);
            let ob = scope.import(&o_r);
            let g = scope.import(&g_r);
            let u = scope.u32x4_uniform(dim, 0, 0, 0)?;
            scope.bcast_fma::<BcastFmaF32>(&p.bcast_fma, xb, g, yb, u, ob, seg.len * dim)?;
        }
        Ok(())
    };

    // --- attention ---
    let n1 = alloc_act(scope, p, rows, dim)?;
    op_rmsnorm(scope, p, x_in, norm1_w, n1, rows, dim, c.norm_eps)?;
    let h1 = if modulation.is_some() {
        let h = alloc_act(scope, p, rows, dim)?;
        modulate(&n1, &h, SCALE_MSA, SHIFT_MSA)?;
        h
    } else {
        n1
    };

    let qkv = mm(
        h1,
        qkv_w,
        3 * inner,
        dim,
        kinds.qkv,
        pipelines.coopmat_sites.qkv,
        false,
        bufs.lora.map(|l| l.qkv),
    )?;
    let q = alloc_act(scope, p, rows, inner)?;
    let k = alloc_act(scope, p, rows, inner)?;
    let v = alloc_act(scope, p, rows, inner)?;
    {
        let u = scope.u32x4_uniform(rows, inner, hd, 0)?;
        scope.qkv_split::<QkvSplitHeadwiseF32>(
            &pipelines.qkv_split_headwise,
            qkv,
            q.data,
            k.data,
            v.data,
            u,
            rows * inner,
        )?;
    }

    // Per-head RMSNorm over head_dim, before rope.
    let qn = alloc_act(scope, p, rows * heads, hd)?;
    op_rmsnorm(scope, p, q, qn_w, qn, rows * heads, hd, c.qk_norm_eps)?;
    let kn = alloc_act(scope, p, rows * heads, hd)?;
    op_rmsnorm(scope, p, k, kn_w, kn, rows * heads, hd, c.qk_norm_eps)?;

    let (qr, kr) = match freqs_in {
        None => (qn, kn),
        Some(freqs) => {
            // Partial half-rot: 96 of 128 channels rotate, the rest pass
            // through. `rot_pairs` rides in the uniform's fourth slot.
            let rot_pairs = (c.rotary_dim() / 2) as u32;
            let qr = alloc_act(scope, p, rows, heads * hd)?;
            let u = scope.u32x4_uniform(rows, heads, hd / 2, rot_pairs)?;
            scope.rope::<RopeF32HalfRotPartial>(
                &pipelines.rope_partial,
                qn.data,
                freqs,
                u,
                qr.data,
                rows,
                heads,
                hd / 2,
            )?;
            let kr = alloc_act(scope, p, rows, heads * hd)?;
            let u2 = scope.u32x4_uniform(rows, heads, hd / 2, rot_pairs)?;
            scope.rope::<RopeF32HalfRotPartial>(
                &pipelines.rope_partial,
                kn.data,
                freqs,
                u2,
                kr.data,
                rows,
                heads,
                hd / 2,
            )?;
            (qr, kr)
        }
    };

    let sa = alloc_act(scope, p, rows, heads * hd)?;
    // Padless sequence -> no mask. The dummy binding is never read.
    let dummy_mask = scope.alloc(elem_bytes.max(4))?;
    op_sdpa(
        scope,
        p,
        qr,
        kr,
        v,
        dummy_mask,
        sa,
        1,
        rows,
        rows,
        heads,
        heads,
        hd,
        shape.sdpa_scale(),
        0,
    )?;

    let proj = mm(
        sa,
        out_w,
        dim,
        inner,
        kinds.out,
        pipelines.coopmat_sites.proj,
        false,
        bufs.lora.map(|l| l.out),
    )?;
    let after_attn = alloc_act(scope, p, rows, dim)?;
    gate_add(&x_in, &ActBuf::dense(proj), &after_attn, GATE_MSA)?;

    // --- feed-forward ---
    let n2 = alloc_act(scope, p, rows, dim)?;
    op_rmsnorm(scope, p, after_attn, norm2_w, n2, rows, dim, c.norm_eps)?;
    let h2 = if modulation.is_some() {
        let h = alloc_act(scope, p, rows, dim)?;
        modulate(&n2, &h, SCALE_MLP, SHIFT_MLP)?;
        h
    } else {
        n2
    };

    // fc1 emits the fused [gate; value]; the SwiGLU reads it in place.
    let fused = mm(
        h2,
        fc1_w,
        2 * ffn,
        dim,
        kinds.fc1,
        pipelines.coopmat_sites.ffn_up,
        true,
        bufs.lora.map(|l| l.fc1),
    )?;
    let gu = alloc_act(scope, p, rows, ffn)?;
    {
        let u = scope.u32x4_uniform(rows, ffn, 0, 0)?;
        scope.silu_mul_fused::<SiluMulFusedF32>(
            &pipelines.silu_mul_fused,
            fused,
            u,
            gu.data,
            rows * ffn,
        )?;
    }
    let down = mm(
        gu,
        fc2_w,
        dim,
        ffn,
        kinds.fc2,
        pipelines.coopmat_sites.ffn_down,
        false,
        bufs.lora.map(|l| l.fc2),
    )?;
    gate_add(&after_attn, &ActBuf::dense(down), &y_out, GATE_MLP)?;
    Ok(())
}

/// One transformer block of the main stack.
#[allow(clippy::too_many_arguments)]
pub fn forward_block<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    pipelines: &DitPipelines,
    shape: DitShape,
    x_in: BatchBuf<'wsp>,
    freqs_in: BatchBuf<'wsp>,
    y_out: BatchBuf<'wsp>,
    bufs: &'wsp BlockBufs,
    kinds: MatmulKinds,
    table: &ModulationTable,
    block_idx: usize,
    segments: &[ModSegment],
) -> Result<(), WgpuError> {
    block_forward(
        scope,
        pipelines,
        shape,
        x_in,
        Some(freqs_in),
        y_out,
        bufs,
        kinds,
        Some((table, block_idx, segments)),
    )
}

/// One token-refiner block: same shape, no AdaLN and no rotary.
#[allow(clippy::too_many_arguments)]
pub fn forward_refiner_block<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    pipelines: &DitPipelines,
    shape: DitShape,
    x_in: BatchBuf<'wsp>,
    y_out: BatchBuf<'wsp>,
    bufs: &'wsp BlockBufs,
    kinds: MatmulKinds,
) -> Result<(), WgpuError> {
    block_forward(
        scope, pipelines, shape, x_in, None, y_out, bufs, kinds, None,
    )
}

/// `final_layer`: shared modulated RMSNorm, then the two output heads run over
/// EVERY row. The caller selects each modality's rows afterwards.
///
/// The final modulation is keyed by timestep alone (not modality), so its
/// segments come from the per-row timestep index rather than the AdaLN rows.
#[allow(clippy::too_many_arguments)]
pub fn forward_final_norm<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    pipelines: &DitPipelines,
    shape: DitShape,
    x_in: BatchBuf<'wsp>,
    norm_w: &'wsp BufRef,
    y_out: BatchBuf<'wsp>,
    table: &ModulationTable,
    timestep_segments: &[ModSegment],
) -> Result<(), WgpuError> {
    let p = &pipelines.block;
    let c = shape.cfg;
    let rows = shape.rows as u32;
    let dim = c.hidden as u32;
    let elem_bytes = p.act_bytes(1);

    let normed = alloc_act(scope, p, rows, dim)?;
    let nw = scope.import(norm_w);
    op_rmsnorm(
        scope,
        p,
        ActBuf::dense(x_in),
        nw,
        normed,
        rows,
        dim,
        c.final_norm_eps,
    )?;

    for seg in timestep_segments {
        let off = seg.start as u64 * dim as u64 * elem_bytes;
        let len = seg.len as u64 * dim as u64 * elem_bytes;
        let x_r = normed.data.sub(off, len);
        let o_r = y_out.sub(off, len);
        // The projection's two halves are shift then scale.
        let shift_r = table.final_param(seg.table_row, 0, elem_bytes);
        let scale_r = table.final_param(seg.table_row, 1, elem_bytes);
        let x = scope.import(&x_r);
        let o = scope.import(&o_r);
        let shift = scope.import(&shift_r);
        let scale = scope.import(&scale_r);
        let u = scope.u32x4_uniform(dim, 1.0_f32.to_bits(), 0, 0)?;
        scope.bcast_modulate::<BcastModulateF32>(
            &p.bcast_modulate,
            x,
            scale,
            shift,
            u,
            o,
            seg.len * dim,
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_weight_names_match_the_checkpoint() {
        let b = BlockWeights::new("blocks.7");
        assert_eq!(b.qkv_proj.0, "blocks.7.attn.qkv_proj.weight");
        assert_eq!(b.out_proj.0, "blocks.7.attn.out_proj.weight");
        assert_eq!(b.fc1.0, "blocks.7.mlp.fc1.weight");
        assert_eq!(b.norm1.0, "blocks.7.norm1.weight");
    }

    #[test]
    fn refiner_shares_the_block_layout_under_its_own_prefix() {
        let b = BlockWeights::new("token_refiner.blocks.1");
        assert_eq!(b.fc2.0, "token_refiner.blocks.1.mlp.fc2.weight");
    }

    /// Every modulation binding offset must satisfy WebGPU's 256-byte storage
    /// offset alignment, at both act widths.
    #[test]
    fn modulation_offsets_are_binding_aligned() {
        let c = DitConfig::h3_33b();
        let layout = ModulationLayout::new(5, c.hidden);
        for elem_bytes in [2u64, 4] {
            for row in 0..layout.rows() as u32 {
                for param in 0..6 {
                    let off = (row as u64 * layout.row_width() as u64
                        + param as u64 * c.hidden as u64)
                        * elem_bytes;
                    assert_eq!(off % 256, 0, "row {row} param {param} at {elem_bytes}B");
                }
            }
        }
    }

    /// Per-segment activation slices must be aligned too.
    #[test]
    fn segment_offsets_are_binding_aligned() {
        let c = DitConfig::h3_33b();
        for elem_bytes in [2u64, 4] {
            for start in [0u64, 1, 37, 1008, 37_296] {
                assert_eq!((start * c.hidden as u64 * elem_bytes) % 256, 0);
            }
        }
    }

    /// The fused SwiGLU and headwise split both need even widths to stay
    /// word-aligned on the packed act paths.
    #[test]
    fn packed_paths_stay_word_aligned() {
        let c = DitConfig::h3_33b();
        assert_eq!(c.ffn_hidden % 2, 0);
        assert_eq!(c.head_dim % 2, 0);
        assert_eq!(c.inner_dim() % 2, 0);
        assert_eq!(c.hidden % 2, 0);
    }
}
