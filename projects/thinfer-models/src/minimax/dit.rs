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
//! - `attn.qkv_proj` in the GGUF is CONTIGUOUS `[q; k; v]` rows: its converter
//!   already undid the original checkpoint's per-head interleave (measured vs the
//!   official FL2VA tensor: corr 0.997 contiguous, 0.014 interleaved). Reading it
//!   interleaved scrambles every attention layer yet still matches a reference
//!   that makes the same assumption -> plain contiguous split.
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

use thinfer_core::backend::nvfp4_fuses_lora_rank;
use thinfer_core::backend::{BufRef, WgpuBackend, WgpuError, WgpuPipeline};
use thinfer_core::ops::{
    BcastFmaF32, BcastModulateF32, RopeF32HalfRotPartial, RopeOp, SiluMulFusedF32, SiluMulFusedOp,
};
use thinfer_core::quant::QuantKind;
use thinfer_core::residency::{
    GpuView, ResidencyError, TransposePolicy, WeightHandle, WeightResidency,
};
use thinfer_core::tensor::StorageEncoding;
use thinfer_core::weight::{WeightId, WeightSource};
use thinfer_core::workspace::{BatchBuf, BatchScope, GemmEpi};

use std::collections::HashMap;

use crate::common::block::{
    ActBuf, Block, BlockPipelines, BlockWgslConfigs, CoopmatStep, DequantStep, alloc_act,
    alloc_matmul_out_buf, build_coopmat_step, op_add, op_qkv_split, op_rmsnorm, op_sdpa_f16,
};
use crate::minimax::DitConfig;
use crate::minimax::adaln::{
    GATE_MLP, GATE_MSA, ModSegment, ModulationLayout, SCALE_MLP, SCALE_MSA, SHIFT_MLP, SHIFT_MSA,
};
use crate::z_image::text_encoder::{LoadError, register_one};
use thinfer_core::ops::DequantScheme;

/// Per-site resident dequant schemes. The shipped Q4_K_M DiT is mixed per
/// tensor and FastH3 ships NVFP4 composites, so a single compile-time scheme is
/// wrong. `None` = an unquantized (dense) site.
#[derive(Clone, Copy, Debug, Default)]
pub struct MatmulKinds {
    pub qkv: Option<DequantScheme>,
    pub out: Option<DequantScheme>,
    pub fc1: Option<DequantScheme>,
    pub fc2: Option<DequantScheme>,
    pub gate: Option<DequantScheme>,
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
    gate: WeightId,
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
            gate: id("attn.gate_proj.weight"),
        }
    }
}

/// One matmul site's low-rank adapter: `delta = B @ A`, held as `A^T`
/// `[K, rank]` and `B^T` `[rank, N]` (the dense matmul's `[K, N]` operands).
#[derive(Clone, Copy, Debug)]
pub struct SiteLora {
    a: WeightHandle,
    b: WeightHandle,
    rank: u32,
}

/// One block's runtime adapters (the turbo distillation and/or user LoRAs,
/// already stacked into one pair per site), as handles, views or buffers.
/// A site is `None` when no adapter targets it.
///
/// These are applied at RUNTIME rather than folded into the base tensors. The
/// fold would re-encode every Q4_K site to Q8_0 (see `common::lora`), taking the
/// streamed working set from ~10.8 GiB per step to ~20.5 GiB and parking the
/// same again in the host fold cache, and an NVFP4 site cannot be refolded at
/// all. The runtime form costs a few % more MACs (rank 384 on the turbo's fused
/// qkv, 128 elsewhere, against a 385M-MAC block), needs no requantization, and
/// so is also exact where a refold would not be.
#[derive(Clone, Copy, Debug)]
pub struct BlockLora<T> {
    qkv: Option<T>,
    out: Option<T>,
    fc1: Option<T>,
    fc2: Option<T>,
}

impl<T> BlockLora<T> {
    fn sites(&self) -> [&Option<T>; 4] {
        [&self.qkv, &self.out, &self.fc1, &self.fc2]
    }

    fn map<U>(&self, mut f: impl FnMut(&T) -> U) -> BlockLora<U> {
        BlockLora {
            qkv: self.qkv.as_ref().map(&mut f),
            out: self.out.as_ref().map(&mut f),
            fc1: self.fc1.as_ref().map(&mut f),
            fc2: self.fc2.as_ref().map(&mut f),
        }
    }
}

pub type BlockLoraHandles = BlockLora<SiteLora>;
pub type BlockLoraViews<'a> = BlockLora<SiteLoraViews<'a>>;
pub type BlockLoraBufs = BlockLora<SiteLoraBufs>;

#[derive(Clone, Copy, Debug)]
pub struct SiteLoraBufs {
    a: BufRef,
    b: BufRef,
    rank: u32,
}

pub struct SiteLoraViews<'a> {
    a: GpuView<'a>,
    b: GpuView<'a>,
    rank: u32,
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
    /// VSA compression gate (FastH3); absent on the dense checkpoints.
    gate: Option<WeightHandle>,
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
    gate: Option<GpuView<'a>>,
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
    gate: Option<BufRef>,
    lora: Option<BlockLoraBufs>,
}

impl BlockHandles {
    /// Resident quant kind per matmul site, for the caller's dequant dispatch.
    pub fn kinds(&self) -> MatmulKinds {
        self.kinds
    }

    /// OS readahead for this block's matmul weights (see
    /// [`WeightResidency::hint_read`]): issued a block early, the cold reads
    /// overlap the current block's GPU work.
    pub async fn hint_read<S: WeightSource>(
        &self,
        residency: &WeightResidency<S>,
    ) -> Result<(), S::Error> {
        for h in [self.qkv_proj, self.out_proj, self.fc1, self.fc2]
            .into_iter()
            .chain(self.gate)
        {
            residency.hint_read(h).await?;
        }
        Ok(())
    }

    pub async fn acquire<'a, S: WeightSource>(
        &self,
        residency: &'a WeightResidency<S>,
        backend: &WgpuBackend,
    ) -> Result<BlockViews<'a>, ResidencyError<S::Error, WgpuError>> {
        let lora = match self.lora {
            None => None,
            Some(l) => Some(BlockLoraViews {
                qkv: SiteLora::acquire_opt(l.qkv, residency, backend).await?,
                out: SiteLora::acquire_opt(l.out, residency, backend).await?,
                fc1: SiteLora::acquire_opt(l.fc1, residency, backend).await?,
                fc2: SiteLora::acquire_opt(l.fc2, residency, backend).await?,
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
            gate: match self.gate {
                Some(h) => Some(residency.acquire(h, backend).await?),
                None => None,
            },
            lora,
        })
    }
}

impl SiteLora {
    async fn acquire_opt<'a, S: WeightSource>(
        site: Option<Self>,
        residency: &'a WeightResidency<S>,
        backend: &WgpuBackend,
    ) -> Result<Option<SiteLoraViews<'a>>, ResidencyError<S::Error, WgpuError>> {
        match site {
            Some(s) => s.acquire(residency, backend).await.map(Some),
            None => Ok(None),
        }
    }

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
            gate: self.gate.as_ref().map(|g| g.buf()),
            lora: self.lora.as_ref().map(|l| l.map(SiteLoraViews::bufs)),
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
    /// The output heads, one per sampling step for a PDD checkpoint (a fused
    /// head per step, see `pdd`), else a single entry every step uses.
    pub heads: Vec<OutHeads>,
}

#[derive(Clone, Copy, Debug)]
pub struct OutHeads {
    pub video_out: WeightHandle,
    pub video_out_bias: WeightHandle,
    pub audio_out: WeightHandle,
    pub audio_out_bias: WeightHandle,
}

impl DitHandles {
    /// The heads step `step` decodes with.
    pub fn heads_for(&self, step: usize) -> OutHeads {
        self.heads[step.min(self.heads.len() - 1)]
    }

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
    let kind = |id: &WeightId, requant: Option<QuantKind>| -> Option<DequantScheme> {
        requant.map(DequantScheme::Block).or_else(|| {
            match residency
                .source()
                .catalog()
                .get(id)
                .and_then(|e| e.encoding)
            {
                Some(StorageEncoding::Quant(k)) => Some(DequantScheme::Block(k)),
                Some(StorageEncoding::Nvfp4) => Some(DequantScheme::Nvfp4),
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
        // `minimax::lora` emits `A^T` `[K, rank]` and `B^T` `[rank, N]`, the
        // dense matmul's `[K, N]` operand layout: uploaded as-is (no GPU
        // transpose per upload). The rank is read, never assumed.
        let rank = ae.shape.0[1] as u32;
        Ok(Some(SiteLora {
            a: register_one(residency, &a_id, TransposePolicy::None, None)?,
            b: register_one(residency, &b_id, TransposePolicy::None, None)?,
            rank,
        }))
    };
    let reg_lora = |prefix: &str| -> Result<Option<BlockLoraHandles>, LoadError> {
        let l = BlockLoraHandles {
            qkv: reg_site_lora(prefix, "attn.qkv_proj")?,
            out: reg_site_lora(prefix, "attn.out_proj")?,
            fc1: reg_site_lora(prefix, "mlp.fc1")?,
            fc2: reg_site_lora(prefix, "mlp.fc2")?,
        };
        Ok(l.sites().iter().any(|s| s.is_some()).then_some(l))
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
            gate: match residency.source().catalog().get(&b.gate) {
                Some(_) => Some(lin(&b.gate)?),
                None => None,
            },
            kinds: MatmulKinds {
                qkv: kind(&b.qkv_proj, transcode),
                out: kind(&b.out_proj, transcode),
                fc1: kind(&b.fc1, transcode),
                fc2: kind(&b.fc2, None),
                gate: kind(&b.gate, None),
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
        heads: match cfg.pdd {
            None => vec![String::new()],
            Some(p) => (0..p.steps).map(|k| format!(".step{k}")).collect(),
        }
        .iter()
        .map(|s| {
            Ok(OutHeads {
                video_out: lin(&w(&format!("final_layer.video_out{s}.weight")))?,
                video_out_bias: dense(&w(&format!("final_layer.video_out{s}.bias")))?,
                audio_out: lin(&w(&format!("final_layer.audio_out{s}.weight")))?,
                audio_out_bias: dense(&w(&format!("final_layer.audio_out{s}.bias")))?,
            })
        })
        .collect::<Result<_, LoadError>>()?,
    })
}

/// Pipeline cfgs for the DiT. Quant weights pair with F16 acts where the
/// adapter exposes `SHADER_F16`, else F32; the residual
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
    rope_partial: WgpuPipeline,
    silu_mul_fused: WgpuPipeline,
    dense_dequant: HashMap<DequantScheme, DequantStep>,
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
            dense_dequant.insert(
                DequantScheme::Block(scheme),
                DequantStep {
                    pipeline,
                    scheme: DequantScheme::Block(scheme),
                },
            );
        }
        let nvfp4 = backend
            .create_pipeline(
                "minimax_dit_dequant_nvfp4",
                &thinfer_core::ops::nvfp4::build_wgsl(DequantTarget::Bf16),
                "main",
                dq_layout,
            )
            .await?;
        dense_dequant.insert(
            DequantScheme::Nvfp4,
            DequantStep {
                pipeline: nvfp4,
                scheme: DequantScheme::Nvfp4,
            },
        );
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
            rope_partial,
            silu_mul_fused,
            dense_dequant,
            coopmat,
            coopmat_f32,
            coopmat_sites: cfgs.coopmat_acts,
        })
    }

    fn dequant_for(&self, kind: Option<DequantScheme>) -> Option<&DequantStep> {
        kind.and_then(|k| self.dense_dequant.get(&k))
    }

    /// `wide` selects the f32-output variant, for a site whose values can
    /// leave the f16 range.
    fn coopmat_for(
        &self,
        kind: Option<DequantScheme>,
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
        kind.and_then(DequantScheme::block)
            .and_then(|k| map.get(&k))
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
    pub fn sdpa_scale(&self) -> f32 {
        1.0 / (self.cfg.head_dim as f32).sqrt()
    }
}

/// Intermediates a [`BlockProbe`] captures, in order, with their element count
/// for a `rows`-row block. Parity diagnostics only.
pub fn probe_layout(shape: DitShape) -> [(&'static str, usize); 9] {
    let c = shape.cfg;
    let r = shape.rows;
    [
        ("attn_in", r * c.hidden),
        ("qkv", r * 3 * c.inner_dim()),
        ("q_normed", r * c.inner_dim()),
        ("attn_out", r * c.inner_dim()),
        ("attn_proj", r * c.hidden),
        ("after_attn", r * c.hidden),
        ("ffn_up", r * 2 * c.ffn_hidden),
        ("swiglu", r * c.ffn_hidden),
        ("ffn_down", r * c.hidden),
    ]
}

/// Caller-owned buffers a block copies its intermediates into, one per
/// [`probe_layout`] entry. Lets a parity gate localize a divergence INSIDE a
/// block; production never passes one.
pub struct BlockProbe<'a> {
    pub bufs: &'a [BufRef],
}

/// One transformer block: pre-norm modulated attention -> gated residual ->
/// pre-norm modulated SwiGLU -> gated residual.
///
/// `segments` is empty for the token refiner, which runs the same block shape
/// with NO modulation and NO rope (it refines the projected text stream).
/// The whole main-stack block on the accelerator as ONE deferred sequence
/// (FastH3: NVFP4 sites, VSA with its gate): no backend round trips between
/// ops, and the caller overlaps the next block's weight upload with it before
/// the scope's submit waits. Same math and bf16 rounding points as
/// [`block_forward`], whose ops these replace one for one (a runtime adapter
/// adds its two dense GEMMs and the sum after its site). `false` = not covered
/// (another checkpoint, a probe, no accelerator, a declined shape): nothing
/// was enqueued and the caller runs the backend path.
#[allow(clippy::too_many_arguments)]
fn block_forward_accel<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    pipelines: &DitPipelines,
    shape: DitShape,
    x_in: BatchBuf<'wsp>,
    freqs: BatchBuf<'wsp>,
    y_out: BatchBuf<'wsp>,
    bufs: &'wsp BlockBufs,
    kinds: MatmulKinds,
    (table, block_idx, segments): (&ModulationTable, usize, &[ModSegment]),
    vsa: &'wsp VsaPlan,
) -> Result<bool, WgpuError> {
    use thinfer_core::backend::{
        ElementwiseOp, MAX_SEGMENTS, Nvfp4Rows, RowSegments, VsaAddrs, VsaPart, VsaShape, VsaSlab,
    };
    let p = &pipelines.block;
    let c = shape.cfg;
    let rows = shape.rows as u32;
    let (dim, inner, ffn) = (c.hidden as u32, c.inner_dim() as u32, c.ffn_hidden as u32);
    let (heads, hd) = (c.n_heads as u32, c.head_dim as u32);
    let nvfp4 = Some(DequantScheme::Nvfp4);
    // The learned compression gate, where the config runs it.
    let gate_w = match (c.vsa_gate, bufs.gate.as_ref()) {
        (false, _) => None,
        (true, Some(w)) => Some(w),
        (true, None) => return Ok(false),
    };
    // Segments must tile the rows in order (they index the table per row).
    let tiles = segments.len() <= MAX_SEGMENTS
        && segments
            .iter()
            .try_fold(0u32, |at, s| (s.start == at).then_some(at + s.len))
            == Some(rows);
    if p.act_dtype != thinfer_core::ops::ActDtype::Bf16
        || !tiles
        || [kinds.qkv, kinds.out, kinds.fc1, kinds.fc2, kinds.gate]
            .iter()
            .any(|k| *k != nvfp4)
    {
        return Ok(false);
    }
    let Some(acc) = scope.accelerator() else {
        return Ok(false);
    };
    let gemms = [
        (3 * inner, dim),
        (inner, dim),
        (dim, inner),
        (2 * ffn, dim),
        (dim, ffn),
    ];
    // Row-local work runs over row chunks (attention alone needs every row), so
    // the widest intermediates (qkv, fused SwiGLU, adapter deltas) are sized by
    // the chunk, not the sequence: what lets the packed sequence pass the old
    // 16k-row ceiling. Every op is row-independent, so chunking is exact. The
    // chunks are the VSA slabs, whole tiles each, so queries can follow them.
    let chunks: Vec<(u32, u32)> = vsa
        .slabs
        .iter()
        .map(|s| (s.rows.start as u32, s.rows.len() as u32))
        .collect();
    // Slabbed queries re-run the q rows of the qkv GEMM with the qkv adapter
    // fused in; an adapter past the fused rank (separate q / k / v pieces
    // stack to 3x theirs) re-runs the whole projection instead.
    let query_slabs = vsa.query_slabs;
    let q_rows_only = bufs
        .lora
        .and_then(|l| l.qkv)
        .is_none_or(|s| nvfp4_fuses_lora_rank(s.rank));
    let vsa_shape = if query_slabs {
        vsa.shape
    } else {
        VsaShape {
            slab_tiles: vsa.shape.n_tiles,
            ..vsa.shape
        }
    };
    let max_chunk = chunks.iter().map(|&(_, n)| n).max().unwrap_or(0);
    let gemm_ms: Vec<u32> = {
        let mut m: Vec<u32> = chunks.iter().map(|&(_, n)| n).collect();
        m.dedup();
        m
    };
    // An adapter site adds `x @ A^T` (N = rank) then `@ B^T` on the dense
    // module pipeline, as `block_forward` does.
    let lora_ok = bufs.lora.is_none_or(|l| {
        [
            (l.qkv, 3 * inner),
            (l.out, dim),
            (l.fc1, 2 * ffn),
            (l.fc2, dim),
        ]
        .iter()
        .all(|(site, n)| {
            site.is_none_or(|s| {
                s.rank.is_multiple_of(8)
                    && gemm_ms.iter().all(|&m| {
                        scope.accepts_gemm(&p.matmul_module, m, s.rank)
                            && scope.accepts_gemm(&p.matmul_module, m, *n)
                    })
            })
        })
    });
    let Some(vsa_need) = acc.accepts_vsa(&vsa_shape) else {
        return Ok(false);
    };
    if !acc.accepts_elementwise()
        || !lora_ok
        || gemms.iter().any(|&(n, k)| {
            gemm_ms
                .iter()
                .any(|&m| acc.accepts_gemm_nvfp4(m, n, k).is_none())
        })
        || !scope.accel_begin()?
    {
        return Ok(false);
    }

    // Past this point work is enqueued, so a decline is a broken accept
    // contract, not a fallback.
    let ran = |ok: bool, what: &str| assert!(ok, "accelerator declined {what} after accepting it");
    let addr = |b: BatchBuf<'wsp>| {
        scope
            .accel_addr(b)
            .expect("workspace and weights are accelerator-addressable")
    };
    let weight = |r: &'wsp BufRef| addr(scope.import(r));
    let elem_bytes = p.act_bytes(1);
    let act = |elems: u32| scope.alloc(elems as u64 * elem_bytes);
    // Rows `[r0, r0 + n)` of a `[rows, width]` activation.
    let rows_of = |b: BatchBuf<'wsp>, r0: u32, n: u32, width: u32| {
        scope.subview(
            &b,
            r0 as u64 * width as u64 * elem_bytes,
            n as u64 * width as u64 * elem_bytes,
        )
    };
    // The modulation segments over rows `[r0, r0 + n)`, ends rebased to the
    // chunk (they index the table per row).
    let seg = |a: usize, b: Option<usize>, r0: u32, n: u32| -> RowSegments {
        let mut s = RowSegments::default();
        let param = |row: u32, param: usize| {
            addr(scope.import_copy(table.param(block_idx, row, param, elem_bytes)))
        };
        for sg in segments {
            let (lo, hi) = (sg.start.max(r0), (sg.start + sg.len).min(r0 + n));
            if lo >= hi {
                continue;
            }
            let i = s.n as usize;
            s.end[i] = hi - r0;
            s.a[i] = param(sg.table_row, a);
            if let Some(b) = b {
                s.b[i] = param(sg.table_row, b);
            }
            s.n += 1;
        }
        s
    };

    // Chunk-sized scratch, allocated once and reused chunk after chunk (the
    // accelerator runs the sequence in order).
    let max_rank = bufs.lora.map_or(0, |l| {
        [l.qkv, l.out, l.fc1, l.fc2]
            .iter()
            .filter_map(|s| s.map(|s| s.rank))
            .max()
            .unwrap_or(0)
    });
    // All of them are dead while a whole-sequence VSA runs (qkv is done, out
    // projection not started), so they share one arena that VSA reuses for
    // its INT8 Q/K (~0.7 GB at 15s 832x480): the arena is sized to the larger
    // of the two, so the INT8 path's peak is a max, not a sum. (Slabbed
    // queries interleave the arena's work with attention: their INT8 Q/K get
    // a region of their own below.)
    let chunk_bytes = [dim, 3 * inner, dim, dim, 2 * ffn, ffn, max_rank]
        .map(|w| (max_chunk * w) as u64 * elem_bytes);
    let at = |i: usize| -> u64 {
        chunk_bytes[..i]
            .iter()
            .map(|b| b.next_multiple_of(256))
            .sum()
    };
    let arena_qk = if query_slabs { 0 } else { vsa_need.qk_i8 };
    let arena = scope.alloc(at(chunk_bytes.len()).max(arena_qk))?;
    let part = |i: usize| scope.subview(&arena, at(i), chunk_bytes[i]);
    let (h, qkv, t, after_attn, fused_buf, gu) =
        (part(0), part(1), part(2), part(3), part(4), part(5));
    let mid = (max_rank > 0).then(|| part(6));

    // A site's adapter, `base += (src @ A^T) @ B^T` over `m` rows. The first
    // GEMM runs here (`mid`); the second rides the site's NVFP4 GEMM when
    // `nvfp4_fuses_lora_rank` (`fused`), else it accumulates into
    // `base` afterwards in its own GEMM epilogue (`lora_tail`). Neither form
    // makes a separate pass over `[m, n]`.
    type Term<'a> = (BatchBuf<'a>, BatchBuf<'a>, u32);
    let lora_head = |src: BatchBuf<'wsp>,
                     m: u32,
                     k: u32,
                     site: Option<SiteLoraBufs>|
     -> Result<Option<Term<'wsp>>, WgpuError> {
        let (Some(l), Some(mid)) = (site, mid) else {
            return Ok(None);
        };
        let mid = scope.subview(&mid, 0, (m * l.rank) as u64 * elem_bytes);
        ran(
            scope.try_gemm(
                &p.matmul_module,
                src,
                scope.import_copy(l.a),
                mid,
                m,
                l.rank,
                k,
            )?,
            "adapter A",
        );
        Ok(Some((mid, scope.import_copy(l.b), l.rank)))
    };
    let fused = |t: Option<Term<'wsp>>| t.filter(|&(_, _, r)| nvfp4_fuses_lora_rank(r));
    let lora_tail = |t: Option<Term<'wsp>>, base: BatchBuf<'wsp>, m: u32, n: u32| {
        let Some((mid, bt, rank)) = t.filter(|&(_, _, r)| !nvfp4_fuses_lora_rank(r)) else {
            return Ok::<_, WgpuError>(());
        };
        let epi = GemmEpi {
            residual: Some(base),
            ..Default::default()
        };
        ran(
            scope.try_gemm_fused(&p.matmul_module, mid, bt, base, m, n, rank, epi)?,
            "adapter B",
        );
        Ok(())
    };
    let site = |pick: fn(BlockLoraBufs) -> Option<SiteLoraBufs>| bufs.lora.and_then(pick);

    // --- attention ---
    // Keys and values for every row. Queries and the gate either for every
    // row as well, or (`query_slabs`) one slab at a time: pass 1 pools each
    // slab's queries for the tile scores and drops them, pass 2 recomputes
    // them (the q rows of the qkv GEMM) right before the slab attends. The
    // VSA tables and INT8 K persist in `parts` between.
    let (k, v) = (act(rows * inner)?, act(rows * inner)?);
    let q_rows = if query_slabs { max_chunk } else { rows };
    let q = act(q_rows * inner)?;
    let gate = match gate_w {
        Some(_) => Some(act(q_rows * inner)?),
        None => None,
    };
    // Chunk `[r0, r0 + n)` of a query-side buffer (slab-local when slabbed).
    let q_side =
        |b: BatchBuf<'wsp>, r0: u32, n: u32| rows_of(b, if query_slabs { 0 } else { r0 }, n, inner);
    let rot_pairs = (c.rotary_dim() / 2) as u32;
    let geometry = scope.import(&vsa.geometry);
    let parts = match query_slabs {
        true => Some((
            scope.alloc(vsa_need.scratch)?,
            match vsa_need.qk_i8 {
                0 => None,
                bytes => Some(scope.alloc(bytes)?),
            },
        )),
        false => None,
    };
    let vsa_part = |part: VsaPart, r0: u32, n: u32| {
        let (scratch, qk_i8) = parts.expect("slabbed queries own the VSA scratch");
        let addrs = VsaAddrs {
            q: addr(q_side(q, r0, n)),
            k: addr(k),
            v: addr(v),
            gate: gate.map_or(0, |g| addr(q_side(g, r0, n))),
            out: addr(q_side(q, r0, n)),
            geometry: addr(geometry),
            scratch: addr(scratch),
            qk_i8: qk_i8.map_or(0, addr),
        };
        ran(scope.accel_vsa_part(&vsa_shape, part, &addrs), "vsa part");
    };
    let slab = |ci: usize| {
        let s = &vsa.slabs[ci];
        VsaSlab {
            tile0: s.tiles.start as u32,
            tiles: s.tiles.len() as u32,
            row0: s.rows.start as u32,
        }
    };
    let norm1 = |r0: u32, n: u32| {
        let h1 = rows_of(h, 0, n, dim);
        ran(
            scope.accel_elementwise(&ElementwiseOp::NormModulate {
                x: addr(rows_of(x_in, r0, n, dim)),
                w: weight(&bufs.norm1),
                out: addr(h1),
                rows: n,
                dim,
                eps: c.norm_eps,
                seg: seg(SCALE_MSA, Some(SHIFT_MSA), r0, n),
            }),
            "norm1",
        );
        h1
    };
    let gate_proj = |h1: BatchBuf<'wsp>, r0: u32, n: u32| -> Result<(), WgpuError> {
        if let (Some(gate_w), Some(gate)) = (gate_w, gate) {
            ran(
                scope.try_gemm_nvfp4(
                    h1,
                    scope.import(gate_w),
                    q_side(gate, r0, n),
                    n,
                    inner,
                    dim,
                    c.nvfp4_format,
                )?,
                "gate",
            );
        }
        Ok(())
    };
    // The whole qkv projection of chunk `[r0, r0 + n)` (from its normed
    // `h1`), normed and rotated into q (query side) and k / v.
    let qkv_proj = |h1: BatchBuf<'wsp>, r0: u32, n: u32| -> Result<(), WgpuError> {
        let qkv = rows_of(qkv, 0, n, 3 * inner);
        let lt = lora_head(h1, n, dim, site(|l| l.qkv))?;
        ran(
            scope.try_gemm_nvfp4_lora(
                h1,
                scope.import(&bufs.qkv_proj),
                qkv,
                n,
                3 * inner,
                dim,
                fused(lt),
                c.nvfp4_format,
            )?,
            "qkv",
        );
        lora_tail(lt, qkv, n, 3 * inner)?;
        ran(
            scope.accel_elementwise(&ElementwiseOp::QkNormRope {
                headwise: false,
                q_only: false,
                qkv: addr(qkv),
                q_norm: weight(&bufs.q_norm),
                k_norm: weight(&bufs.k_norm),
                freqs: addr(rows_of(freqs, r0, n, 2 * rot_pairs)),
                q: addr(q_side(q, r0, n)),
                k: addr(rows_of(k, r0, n, inner)),
                v: addr(rows_of(v, r0, n, inner)),
                rows: n,
                heads,
                head_dim: hd,
                rot_pairs,
                eps: c.qk_norm_eps,
            }),
            "qk norm + rope",
        );
        Ok(())
    };
    for (ci, &(r0, n)) in chunks.iter().enumerate() {
        let h1 = norm1(r0, n);
        if !query_slabs {
            gate_proj(h1, r0, n)?;
        }
        qkv_proj(h1, r0, n)?;
        if query_slabs {
            vsa_part(VsaPart::Pool(slab(ci)), r0, n);
        }
    }
    // The attention output overwrites q in place: each VSA query tile reads
    // only its own q rows (into registers, before any store) and writes only
    // those rows, and the pooled means were taken by an earlier launch. One
    // `rows x inner` buffer less, which is what lets a full 15s clip fit.
    if query_slabs {
        vsa_part(VsaPart::Select, 0, chunks[0].1);
    } else {
        ran(
            scope.try_vsa(&vsa_shape, q, k, v, gate, q, geometry, Some(arena))?,
            "vsa",
        );
    }

    // --- per chunk: (slabbed) its queries and attention, then the out
    // projection + feed-forward ---
    for (ci, &(r0, n)) in chunks.iter().enumerate() {
        if query_slabs {
            // The slab's queries again, bit-identical to pass 1's (same input,
            // same weight rows, same adapter term): the q rows of the
            // projection alone, or the whole projection (rewriting k / v with
            // the bits they hold) when the adapter is past the fused rank.
            let h1 = norm1(r0, n);
            gate_proj(h1, r0, n)?;
            if q_rows_only {
                let qp = rows_of(qkv, 0, n, inner);
                let lt = lora_head(h1, n, dim, site(|l| l.qkv))?;
                ran(
                    scope.try_gemm_nvfp4_rows(
                        h1,
                        scope.import(&bufs.qkv_proj),
                        qp,
                        n,
                        inner,
                        dim,
                        fused(lt),
                        c.nvfp4_format,
                        Some(Nvfp4Rows {
                            base: 0,
                            total: 3 * inner,
                        }),
                    )?,
                    "q",
                );
                ran(
                    scope.accel_elementwise(&ElementwiseOp::QkNormRope {
                        headwise: false,
                        q_only: true,
                        qkv: addr(qp),
                        q_norm: weight(&bufs.q_norm),
                        k_norm: 0,
                        freqs: addr(rows_of(freqs, r0, n, 2 * rot_pairs)),
                        q: addr(q_side(q, r0, n)),
                        k: 0,
                        v: 0,
                        rows: n,
                        heads,
                        head_dim: hd,
                        rot_pairs,
                        eps: c.qk_norm_eps,
                    }),
                    "q norm + rope",
                );
            } else {
                qkv_proj(h1, r0, n)?;
            }
            vsa_part(VsaPart::Attend(slab(ci)), r0, n);
        }
        let sa_c = q_side(q, r0, n);
        let proj = rows_of(t, 0, n, dim);
        let lt = lora_head(sa_c, n, inner, site(|l| l.out))?;
        ran(
            scope.try_gemm_nvfp4_lora(
                sa_c,
                scope.import(&bufs.out_proj),
                proj,
                n,
                dim,
                inner,
                fused(lt),
                c.nvfp4_format,
            )?,
            "out",
        );
        lora_tail(lt, proj, n, dim)?;
        let after = rows_of(after_attn, 0, n, dim);
        ran(
            scope.accel_elementwise(&ElementwiseOp::GateAdd {
                x: addr(rows_of(x_in, r0, n, dim)),
                y: addr(proj),
                out: addr(after),
                rows: n,
                dim,
                seg: seg(GATE_MSA, None, r0, n),
            }),
            "attn gate",
        );
        let h2 = rows_of(h, 0, n, dim);
        ran(
            scope.accel_elementwise(&ElementwiseOp::NormModulate {
                x: addr(after),
                w: weight(&bufs.norm2),
                out: addr(h2),
                rows: n,
                dim,
                eps: c.norm_eps,
                seg: seg(SCALE_MLP, Some(SHIFT_MLP), r0, n),
            }),
            "norm2",
        );
        let fused_c = rows_of(fused_buf, 0, n, 2 * ffn);
        let lt = lora_head(h2, n, dim, site(|l| l.fc1))?;
        ran(
            scope.try_gemm_nvfp4_lora(
                h2,
                scope.import(&bufs.fc1),
                fused_c,
                n,
                2 * ffn,
                dim,
                fused(lt),
                c.nvfp4_format,
            )?,
            "fc1",
        );
        lora_tail(lt, fused_c, n, 2 * ffn)?;
        let gu = rows_of(gu, 0, n, ffn);
        ran(
            scope.accel_elementwise(&ElementwiseOp::SiluMulFused {
                x: addr(fused_c),
                out: addr(gu),
                rows: n,
                h: ffn,
            }),
            "swiglu",
        );
        let down = rows_of(t, 0, n, dim);
        let lt = lora_head(gu, n, ffn, site(|l| l.fc2))?;
        ran(
            scope.try_gemm_nvfp4_lora(
                gu,
                scope.import(&bufs.fc2),
                down,
                n,
                dim,
                ffn,
                fused(lt),
                c.nvfp4_format,
            )?,
            "fc2",
        );
        lora_tail(lt, down, n, dim)?;
        ran(
            scope.accel_elementwise(&ElementwiseOp::GateAdd {
                x: addr(after),
                y: addr(down),
                out: addr(rows_of(y_out, r0, n, dim)),
                rows: n,
                dim,
                seg: seg(GATE_MLP, None, r0, n),
            }),
            "mlp gate",
        );
    }
    Ok(true)
}

/// Row chunk size for the accelerated block's row-local ops: big enough that
/// every GEMM stays efficient, small enough that the widest intermediate (the
/// fused SwiGLU, `chunk x 28672` bf16) is ~235 MiB.
pub const ROW_CHUNK: usize = 4096;

/// Packed rows above which the accelerated block holds queries one slab at a
/// time ([`VsaPlan::query_slabs`]): it re-runs the q third of the qkv GEMM per
/// slab to keep two `rows x inner` buffers out of VRAM (~0.9 GB each at a 15s
/// clip with two references), which shorter sequences do not need.
pub const QUERY_SLAB_ROWS: usize = 24 * 1024;

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
    vsa: Option<&'wsp VsaPlan>,
    probe: Option<&BlockProbe>,
) -> Result<(), WgpuError> {
    if let (Some(m), Some(plan), Some(f), None) = (modulation, vsa, freqs_in, probe)
        && block_forward_accel(
            scope, pipelines, shape, x_in, f, y_out, bufs, kinds, m, plan,
        )?
    {
        return Ok(());
    }
    let p = &pipelines.block;
    let c = shape.cfg;
    let rows = shape.rows as u32;
    let dim = c.hidden as u32;
    let inner = c.inner_dim() as u32;
    let hd = c.head_dim as u32;
    let heads = c.n_heads as u32;
    let ffn = c.ffn_hidden as u32;
    let elem_bytes = p.act_bytes(1);
    let layout = probe_layout(shape);
    let capture = |stage: usize, src: BatchBuf<'wsp>| -> Result<(), WgpuError> {
        let Some(pr) = probe else {
            return Ok(());
        };
        let dst = scope.import_copy(pr.bufs[stage]);
        scope.copy_buffer_to_buffer(src, 0, dst, 0, layout[stage].1 as u64 * elem_bytes)
    };

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

    // `src @ A^T` then `@ B^T`, added to the site's base output. The sum lands
    // in a THIRD buffer: one buffer cannot be both the read and the read_write
    // operand of a single dispatch.
    let add_lora = |src: ActBuf<'wsp>,
                    base: BatchBuf<'wsp>,
                    out_w: u32,
                    in_w: u32,
                    lora: Option<SiteLoraBufs>|
     -> Result<BatchBuf<'wsp>, WgpuError> {
        let Some(l) = lora else {
            return Ok(base);
        };
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
            ActBuf::dense(base),
            ActBuf::dense(delta),
            ActBuf::dense(sum),
        )?;
        Ok(sum)
    };

    let mm = |src: ActBuf<'wsp>,
              w,
              out_w: u32,
              in_w: u32,
              kind: Option<DequantScheme>,
              coopmat_ok: bool,
              wide_out: bool,
              lora: Option<SiteLoraBufs>|
     -> Result<BatchBuf<'wsp>, WgpuError> {
        let scratch = alloc_matmul_out_buf(scope, p, rows * out_w)?;
        // NVFP4 runs natively (W4A4, the checkpoint's own numerics) when the
        // accelerator takes it; everything else, and any declined shape,
        // dequantizes below.
        if kind == Some(DequantScheme::Nvfp4)
            && src.scale.is_none()
            && scope.try_gemm_nvfp4(src.data, w, scratch, rows, out_w, in_w, c.nvfp4_format)?
        {
            return add_lora(src, scratch, out_w, in_w, lora);
        }
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
        add_lora(src, scratch, out_w, in_w, lora)
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
    capture(0, h1.data)?;

    let qkv = mm(
        h1,
        qkv_w,
        3 * inner,
        dim,
        kinds.qkv,
        pipelines.coopmat_sites.qkv,
        false,
        bufs.lora.and_then(|l| l.qkv),
    )?;
    capture(1, qkv)?;
    let q = alloc_act(scope, p, rows, inner)?;
    let k = alloc_act(scope, p, rows, inner)?;
    let v = alloc_act(scope, p, rows, inner)?;
    op_qkv_split(scope, p, ActBuf::dense(qkv), q, k, v, rows, inner)?;

    // Per-head RMSNorm over head_dim, before rope.
    let qn = alloc_act(scope, p, rows * heads, hd)?;
    op_rmsnorm(scope, p, q, qn_w, qn, rows * heads, hd, c.qk_norm_eps)?;
    let kn = alloc_act(scope, p, rows * heads, hd)?;
    op_rmsnorm(scope, p, k, kn_w, kn, rows * heads, hd, c.qk_norm_eps)?;
    capture(2, qn.data)?;

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
    // Video sparse attention where the checkpoint was trained with it: the
    // learned compression gate reads the same normed input as q/k/v.
    let sparse = match (vsa, bufs.gate.as_ref()) {
        (Some(plan), Some(gate_w)) if c.vsa_gate => {
            let gate = mm(
                h1,
                scope.import(gate_w),
                inner,
                dim,
                kinds.gate,
                false,
                false,
                None,
            )?;
            scope.try_vsa(
                &plan.shape,
                qr.data,
                kr.data,
                v.data,
                Some(gate),
                sa.data,
                scope.import(&plan.geometry),
                None,
            )?
        }
        _ => false,
    };
    if !sparse {
        // Padless sequence -> no mask. The dummy binding is never read.
        let dummy_mask = scope.alloc(elem_bytes.max(4))?;
        // The `_f16` entry is what `fast_sdpa` was compiled for: it casts
        // Q/K/V to f16 for the subgroup kernel. Plain `op_sdpa` on this bf16
        // block once fed that f16 kernel raw bf16 bits (the noise bug). V is
        // not normed here, but the parity gate measures it O(100) at most, far
        // inside f16.
        op_sdpa_f16(
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
    }
    capture(3, sa.data)?;

    let proj = mm(
        sa,
        out_w,
        dim,
        inner,
        kinds.out,
        pipelines.coopmat_sites.proj,
        false,
        bufs.lora.and_then(|l| l.out),
    )?;
    let after_attn = alloc_act(scope, p, rows, dim)?;
    gate_add(&x_in, &ActBuf::dense(proj), &after_attn, GATE_MSA)?;
    capture(4, proj)?;
    capture(5, after_attn.data)?;

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
        bufs.lora.and_then(|l| l.fc1),
    )?;
    capture(6, fused)?;
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
        bufs.lora.and_then(|l| l.fc2),
    )?;
    capture(7, gu.data)?;
    capture(8, down)?;
    gate_add(&after_attn, &ActBuf::dense(down), &y_out, GATE_MLP)?;
    Ok(())
}

/// One transformer block of the main stack, updating the residual stream `x`
/// in place. The accelerated block consumes each row chunk of `x` before it
/// writes that chunk's output, so it runs on `x` directly: one sequence-sized
/// buffer less at the peak. The backend block reads `x` whole while writing,
/// so it goes through a temporary.
#[allow(clippy::too_many_arguments)]
pub fn forward_block<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    pipelines: &DitPipelines,
    shape: DitShape,
    x: BatchBuf<'wsp>,
    freqs_in: BatchBuf<'wsp>,
    bufs: &'wsp BlockBufs,
    kinds: MatmulKinds,
    table: &ModulationTable,
    block_idx: usize,
    segments: &[ModSegment],
    vsa: Option<&'wsp VsaPlan>,
    probe: Option<&BlockProbe>,
) -> Result<(), WgpuError> {
    let modulation = (table, block_idx, segments);
    if let (Some(plan), None) = (vsa, probe)
        && block_forward_accel(
            scope, pipelines, shape, x, freqs_in, x, bufs, kinds, modulation, plan,
        )?
    {
        return Ok(());
    }
    let y = scope.alloc(x.len())?;
    block_forward(
        scope,
        pipelines,
        shape,
        x,
        Some(freqs_in),
        y,
        bufs,
        kinds,
        Some(modulation),
        vsa,
        probe,
    )?;
    scope.copy_buffer_to_buffer(y, 0, x, 0, x.len())
}

/// One run's video sparse attention inputs: the device tile table
/// ([`super::vsa::VsaGeometry::bytes`]), its shape, and the slabs the
/// accelerated block chunks its rows by ([`super::vsa::VsaGeometry::slabs`]
/// at [`ROW_CHUNK`]).
#[derive(Clone, Debug)]
pub struct VsaPlan {
    pub geometry: BufRef,
    pub shape: thinfer_core::backend::VsaShape,
    pub slabs: Vec<super::vsa::Slab>,
    /// Hold queries (and the gate) one slab at a time instead of for every
    /// row (see `block_forward_accel`); `shape.slab_tiles` is sized for it.
    pub query_slabs: bool,
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
    probe: Option<&BlockProbe>,
) -> Result<(), WgpuError> {
    block_forward(
        scope, pipelines, shape, x_in, None, y_out, bufs, kinds, None, None, probe,
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
