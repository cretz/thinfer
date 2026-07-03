//! Shared DiT transformer block: the `BlockPipelines` kernel set, the
//! activation/op primitives, and a `Block::forward` for the Z-Image-style
//! block (both `modulation` flavors). Reused across models (Z-Image drives
//! `forward`; Wan composes the same pipelines/ops/`dispatch_matmul_site`).
//!
//! Forward sequence per upstream `ZImageTransformerBlock.forward`
//! (`src/zimage/transformer.py`):
//!
//! ```text
//! attn_in  = norm1(x) * (1 + scale_msa)?     // *(1+scale) only if modulated
//! attn_out = attention(attn_in, mask, freqs)
//! x = x + (gate_msa? *) norm2(attn_out)
//! x = x + (gate_mlp? *) ffn_norm2(ffn(ffn_norm1(x) * (1 + scale_mlp)?))
//! ```
//!
//! All transient buffers (activations + per-call uniforms) come from
//! `BatchScope`.

use thinfer_core::Backend;
use thinfer_core::backend::{BufRef, WgpuBackend, WgpuError, WgpuPipeline};
use thinfer_core::cache::KernelKey;
use thinfer_core::ops::{
    ActDtype, AddF32, BcastAddF32, BcastAddOp, BcastAffineF32, BcastAffineOp, BcastFmaF32,
    BcastFmaOp, BcastModulateF32, BcastModulateOp, BcastMulF32, Bf16ToF16, F16ToBf16, GeluF32,
    LayerNormF32, LayerNormOp, MatMulConfig, MatMulF32, MatmulOp, MulF32, Op, QkvSplitF32,
    QkvSplitOp, RmsNormF32, RmsNormOp, RopeF32, RopeF32HalfRot, RopeOp, ScatterPadRowsF32,
    ScatterPadRowsOp, SdpaDecode, SdpaF32, SdpaF32LargeD, SdpaOp, SiluF32, SiluMulF32, TanhF32,
    WeightDtype, WgslConfig,
};
use thinfer_core::residency::{GpuView, ResidencyError, WeightHandle, WeightResidency};
use thinfer_core::trace;
use thinfer_core::weight::WeightSource;
use thinfer_core::workspace::{BatchBuf, BatchScope, ScopePacker};

#[derive(Clone, Copy, Debug)]
pub struct BlockConfig {
    pub dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub ffn_hidden: usize,
    pub batch: usize,
    pub seq: usize,
    pub norm_eps: f32,
    pub adaln_embed_dim: usize,
    pub modulation: bool,
    /// RoPE convention for q/k. `false` (default) drives the interleaved-pair
    /// `op_rope` (Z-Image / Wan). `true` drives the half-rot `op_rope_halfrot`
    /// (HF `(k, k+D/2)` pairing) used by Ideogram-4's MRoPE and Qwen3.
    pub rope_halfrot: bool,
}

impl BlockConfig {
    pub fn rows(&self) -> usize {
        self.batch * self.seq
    }
    pub fn sdpa_scale(&self) -> f32 {
        1.0 / (self.head_dim as f32).sqrt()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct AdaLnBufs {
    pub weight: BufRef,
    pub bias: BufRef,
}

#[derive(Clone, Copy, Debug)]
pub struct BlockWeightBufs {
    pub attention_norm1: BufRef,
    pub attention_norm2: BufRef,
    pub ffn_norm1: BufRef,
    pub ffn_norm2: BufRef,
    /// Fused upstream-canonical QKV weight (one matmul producing `[rows, 3*H]`,
    /// then `qkv_split` peels off three contiguous `[rows, H]` slabs).
    pub attn_qkv: BufRef,
    pub attn_to_out: BufRef,
    pub attn_norm_q: BufRef,
    pub attn_norm_k: BufRef,
    pub ffn_w1: BufRef,
    pub ffn_w2: BufRef,
    pub ffn_w3: BufRef,
    pub adaln: Option<AdaLnBufs>,
}

/// Per-regime matmul instances for a DiT block. Five distinct call sites in
/// `Block::forward` map to five potentially-distinct kernel shapes:
/// QKV projections, attention output, FFN up (w1/w3), FFN down (w2), and the
/// AdaLN linear. Each config is sized for its shape; the pipeline cache
/// compiles one shader per distinct config.
pub struct BlockMatmuls {
    pub qkv: MatMulF32,
    /// Self-attention QKV, split from `qkv` (cross-attention) so its weight can
    /// take a distinct (e.g. i8/Q8_0) encoding. Same shape as `qkv`; identical to
    /// it unless [`BlockWgslConfigs::matmul_qkv_self`] overrides the weight dtype.
    pub qkv_self: MatMulF32,
    pub proj: MatMulF32,
    pub ffn_up: MatMulF32,
    pub ffn_down: MatMulF32,
    pub adaln: MatMulF32,
    /// Module-level dense linears (patch embed, condition embedder, proj_out):
    /// weights that are NOT folded into the quant block set and so stay bf16
    /// (`Linear2D [K, N]`). Always bf16-weight, independent of whatever encoding
    /// the block matmul sites take, so these sites never touch the per-site
    /// dequant/i8 block pipeline. Square 64x64 tiling (the patch/proj_out M can
    /// be large, unlike adaln's M=1).
    pub module: MatMulF32,
}

impl BlockMatmuls {
    /// Per-kernel GPU timestamps swept across DEFAULT/8x8/32x64/2x4/
    /// 32x128/2x8/64x64/4x4/64x128/4x8/128x64/4x4: 64x64/4x4 wins.
    /// Register-blocked bigger tiles win on this iGPU until the cliff
    /// at tm*tn=32 acc regs/thread (64x128/4x8 doubled ffn ms). The
    /// worklog "iGPU is occupancy-bound, shrink WG" gotcha is *wrong*
    /// for this kernel. Hard constraint: WG threads <= 256 (WebGPU),
    /// so 128x64/4x4 is invalid (512 threads). adaln stays DEFAULT:
    /// M=1 makes register blocking pointless.
    ///
    /// `bk` per matmul is selected from each kernel's `weight_dtype`
    /// so quant schemes satisfy `bk % block_size == 0`. Bf16/f32 use
    /// bk=16 (matches f32 acts, 16 KiB shared at bm=bn=64). Q8_0 uses
    /// bk=block_size=32 (one block per K-step). Bigger bk costs more
    /// per-WG shared memory which hurts occupancy on Intel iGPU more
    /// than the reduced t-loop count saves; the WG-level B-load is
    /// kept saturated via the cooperative dequant (TPB=4 threads per
    /// block, see matmul.rs Quant arm).
    pub fn for_cfgs(cfgs: &BlockWgslConfigs) -> Self {
        // bk per matmul. Bf16/f32 use bk=16. Q8_0 uses bk=32 (= block_size,
        // one block per K-step). K-family (Q4_K/Q5_K/Q6_K, block_size=256)
        // uses bk=64 via SUB-BLOCK dequant: 4 K-strips per 256-block per
        // column, full 256-thread WG saturated during dequant
        // (TPB=4, slot_threads=4*64=256). bk=128 was measured 13% slower
        // (TPB=2 → only 128 threads do dequant work, half occupancy).
        let bk_for = |wd: WeightDtype| -> u32 {
            match wd {
                WeightDtype::Quant(k) if k.block_size() >= 128 => 64,
                WeightDtype::Quant(k) => k.block_size(),
                _ => 16,
            }
        };
        // Square 64x64/4x4 across all weight dtypes. K-family at bk=64
        // f16 tiles: shared = (64+64)*64*2 = 16 KiB, half budget.
        // When weight is Quant, the layer pre-dequants to a bf16 dense
        // N-major workspace; the matmul reads via the bf16 path with
        // `b_nmajor=true`, so we pick bk for the Bf16 case (bk=16) and
        // set b_nmajor on the config.
        let square = |wd: WeightDtype| {
            let (bk, b_nmajor) = match wd {
                WeightDtype::Quant(_) => (16, true),
                _ => (bk_for(wd), false),
            };
            MatMulConfig {
                bm: 64,
                bn: 64,
                bk,
                tm: 4,
                tn: 4,
                b_nmajor,
            }
        };
        // Wide-K 128x32/4x4 for FFN-down (K=10240). Quant goes through
        // the dequant-once path same as `square`, so it uses 64x64 + bf16
        // bk=16 + b_nmajor=true. Non-quant uses the wide-K shape.
        let wide_k = |wd: WeightDtype| match wd {
            WeightDtype::Quant(_) => square(wd),
            _ => MatMulConfig {
                bm: 128,
                bn: 32,
                bk: bk_for(wd),
                tm: 4,
                tn: 4,
                b_nmajor: false,
            },
        };
        Self {
            qkv: MatMulF32::new(square(cfgs.matmul_qkv.weight_dtype)),
            qkv_self: MatMulF32::new(square(cfgs.matmul_qkv_self.weight_dtype)),
            proj: MatMulF32::new(square(cfgs.matmul_proj.weight_dtype)),
            ffn_up: MatMulF32::new(square(cfgs.matmul_ffn_up.weight_dtype)),
            ffn_down: MatMulF32::new(wide_k(cfgs.matmul_ffn_down.weight_dtype)),
            // tn=2 (not DEFAULT tn=1) so AdaLN output can land in packed-bf16
            // storage when `WgslConfig.act_dtype = Bf16`. Output cols are
            // 6*ADALN_EMBED_DIM = 1536 (even) so pairing is clean. M=1 keeps
            // register blocking pointless, so bm/tm stay at DEFAULT. AdaLN
            // weight stays bf16 even in the quant-DiT case.
            adaln: MatMulF32::new(MatMulConfig {
                tn: 2,
                ..MatMulConfig::DEFAULT
            }),
            // Bf16 square (b_nmajor=false): module weights are dense bf16
            // `Linear2D [K, N]`, never the N-major dequant workspace.
            module: MatMulF32::new(square(WeightDtype::Bf16)),
        }
    }
}

/// One dequant-once-per-matmul step. Carries the pipeline plus the source
/// scheme (needed by dispatch to know block_size for workgroup count).
/// Present when the layer's weight for that matmul site is Quant; the matmul
/// pipeline alongside it is built with `weight_dtype = Bf16, b_nmajor = true`
/// so it reads the dense dequanted workspace.
pub struct DequantStep {
    pub pipeline: WgpuPipeline,
    pub scheme: thinfer_core::quant::QuantKind,
}

/// A cooperative-matrix (tensor-core) matmul site. `Some` only for an opt-in
/// Wan-DiT site when the device exposes a usable coopmat config AND the block
/// runs the fast_mixed (bf16<->f16) casts. Bundles: an f16-target dequant of
/// the Quant weight into `[N,K]` nmajor f16, the coopmat matmul pipeline, and
/// its config. Dispatch (`dispatch_matmul_site_coopmat`) casts the bf16 A-side
/// to f16, dequants the weight to f16, runs the coopmat matmul (f16 in, f32
/// accumulate, f16 out), then casts the output back to the bf16 residual.
pub struct CoopmatStep {
    pub dequant_f16: DequantStep,
    pub matmul: WgpuPipeline,
    pub cfg: thinfer_core::ops::matmul_coopmat::CoopmatMatmulConfig,
}

pub struct BlockPipelines {
    pub matmuls: BlockMatmuls,
    pub matmul_qkv: WgpuPipeline,
    /// Self-attention QKV pipeline (cross-attention uses `matmul_qkv`). Compiled
    /// from [`BlockWgslConfigs::matmul_qkv_self`]; identical to `matmul_qkv`
    /// unless that slot pins a distinct (i8/Q8_0) weight encoding.
    pub matmul_qkv_self: WgpuPipeline,
    pub matmul_proj: WgpuPipeline,
    pub matmul_ffn_up: WgpuPipeline,
    pub matmul_ffn_down: WgpuPipeline,
    pub matmul_adaln: WgpuPipeline,
    /// Module-level dense bf16 matmul (patch embed, condition embedder,
    /// proj_out). Always reads a bf16 `Linear2D [K, N]` weight directly, so the
    /// module sites are decoupled from the block's per-site quant pipeline.
    pub matmul_module: WgpuPipeline,
    /// Per-site dequant pre-pass. `Some` iff the corresponding matmul's
    /// weight_dtype is Quant. When present, the layer forward dequants the
    /// quant weight into a workspace buffer, then runs the bf16-nmajor
    /// matmul against the dense workspace. None means the matmul reads its
    /// weight buffer directly (bf16 or f32 path).
    pub dequant_qkv: Option<DequantStep>,
    pub dequant_qkv_self: Option<DequantStep>,
    pub dequant_proj: Option<DequantStep>,
    pub dequant_ffn_up: Option<DequantStep>,
    pub dequant_ffn_down: Option<DequantStep>,
    /// DP4A int8 path: per-Quant-site `(dequant_i8 pipeline + scheme,
    /// matmul_i8 pipeline)`. `Some` iff the backend exposes
    /// `WgslLanguageFeatures::Packed4x8IntegerDotProduct` AND the site's
    /// weight is Quant. Takes precedence over `dequant_<site>` when present:
    /// `block.rs` forward chooses I8 path when these are Some, falling back
    /// to the F16-workspace dequant path otherwise. Lean and independent;
    /// the legacy F16 matmul pipeline (`matmul_<site>` below) is still
    /// compiled in this case but goes unused on the I8 site.
    pub dequant_i8_qkv: Option<DequantStep>,
    pub dequant_i8_qkv_self: Option<DequantStep>,
    pub dequant_i8_proj: Option<DequantStep>,
    pub dequant_i8_ffn_up: Option<DequantStep>,
    pub dequant_i8_ffn_down: Option<DequantStep>,
    pub matmul_i8_qkv: Option<WgpuPipeline>,
    pub matmul_i8_qkv_self: Option<WgpuPipeline>,
    pub matmul_i8_proj: Option<WgpuPipeline>,
    pub matmul_i8_ffn_up: Option<WgpuPipeline>,
    pub matmul_i8_ffn_down: Option<WgpuPipeline>,
    /// Shared activation-quantizer pipeline (f16 acts -> packed i8 + per-(M,
    /// K/32) f32 params). One pipeline serves every Quant matmul site since
    /// the kernel is K-agnostic. `Some` when any DP4A matmul site is in use
    /// (the matmul-site transcode) or when `sdpa_i8` is built (the post-rope
    /// q/k/v quantize).
    pub act_quant: Option<WgpuPipeline>,
    /// Tile shape for the DP4A matmul (`matmul_i8_<site>` pipelines were
    /// built with this cfg). Same shape for all sites today (DEFAULT).
    pub matmul_i8_cfg: thinfer_core::ops::matmul_i8::MatMulI8Config,
    /// I8-acts × bf16-weights matmul. Compiled when this pipeline set has
    /// `act_dtype = I8` AND at least one main-matmul site's weight is Bf16
    /// (the DiT refiners, t_embedder under I8 routing). One pipeline serves
    /// every applicable site since the kernel is K-agnostic.
    pub matmul_i8_bf16: Option<WgpuPipeline>,
    pub matmul_i8_bf16_cfg: thinfer_core::ops::matmul_i8_bf16::MatMulI8Bf16Config,
    /// bf16-block-sum precompute, paired with `matmul_i8_bf16`. Produces
    /// `b_sum[n, t] = Σ_{k in block t} b[n, k]` (f32) per dispatch into
    /// scope; consumed by matmul_i8_bf16 as the asymmetric correction-term
    /// factor on bf16 weights.
    pub bf16_block_sum: Option<WgpuPipeline>,
    pub rmsnorm: WgpuPipeline,
    pub layernorm: WgpuPipeline,
    pub rope: WgpuPipeline,
    pub rope_halfrot: WgpuPipeline,
    pub sdpa: WgpuPipeline,
    /// Subgroup small-D sdpa. `Some` iff F16 acts AND the backend exposes
    /// subgroups (min size >= 4). Dispatch prefers it when `head_dim % 32 == 0
    /// && head_dim <= 128`, else falls back to `sdpa`.
    pub sdpa_sg: Option<WgpuPipeline>,
    /// Lane-cluster width (CL) baked into the `sdpa_sg` kernel: 8 when the
    /// adapter's subgroup min size >= 8, else 4. With `sdpa_sg_qr` it sets
    /// BR = WG/CL*R at dispatch.
    pub sdpa_sg_cl: u32,
    /// Q rows per lane cluster (R) baked into the `sdpa_sg` kernels
    /// (Q-register blocking; bit-exact vs R=1). Default 1;
    /// `THINFER_SDPA_SG_QR` overrides for perf A/B.
    pub sdpa_sg_qr: u32,
    /// Temporal sliding-window twin of `sdpa_sg` (`build_f16_sg_windowed_wgsl`).
    /// Built alongside `sdpa_sg` whenever the subgroup f16 SDPA is. Dispatched
    /// in place of `sdpa_sg` only when a run sets `attn_window > 0` (video DiT
    /// self-attention); restricts each query to keys within +-W latent frames,
    /// breaking the O(frames^2) cost. Changes the output, so it is opt-in.
    pub sdpa_sg_win: Option<WgpuPipeline>,
    /// Large-D SDPA (`SdpaF32LargeD`, one workgroup per Q row). `Some` iff
    /// `BlockWgslConfigs::large_d_sdpa`; dispatch routes `head_dim > 128` here.
    pub sdpa_large_d: Option<WgpuPipeline>,
    /// Decode SDPA (`SdpaDecode`, one workgroup per Q row, the workgroup
    /// cooperates over the KV length). `Some` iff `BlockWgslConfigs::decode_sdpa`.
    /// For single-query decode against a long KV cache this is ~1-2 orders faster
    /// than `SdpaF32` (which runs the whole per-head softmax in one thread); the
    /// caller opts in per-call via [`op_sdpa_decode`]. `head_dim <= 128`.
    pub sdpa_decode: Option<WgpuPipeline>,
    pub qkv_split: WgpuPipeline,
    pub silu: WgpuPipeline,
    pub silu_mul: WgpuPipeline,
    /// gelu-tanh activation. Built always (cheap), used by the Hunyuan
    /// dual-stream block's gelu-tanh FFN; Z-Image/Wan FFNs are SwiGLU and
    /// drive `silu_mul` instead.
    pub gelu: WgpuPipeline,
    pub add: WgpuPipeline,
    pub mul: WgpuPipeline,
    pub tanh: WgpuPipeline,
    pub bcast_affine: WgpuPipeline,
    pub bcast_fma: WgpuPipeline,
    pub bcast_modulate: WgpuPipeline,
    pub bcast_add: WgpuPipeline,
    pub bcast_mul: WgpuPipeline,
    pub scatter_pad_rows: WgpuPipeline,
    /// Opt-in i8 attention (the only i8 activation storage outside matmul
    /// internals). `Some` iff `BlockWgslConfigs::i8_sdpa`. When enabled the
    /// forward quantizes q/k/v once after the F16 rope (`act_quant` into
    /// fused `[data || scale]` pairs), runs `sdpa_i8`, and feeds its paired
    /// output straight into the attn-proj matmul A-side. Everything else
    /// (residual carry, norms, modulate, FFN glue) stays dense at
    /// `act_dtype`: per-32-block i8 of the residual stream is numerically
    /// unsound (outlier channels annihilate their block neighbors and the
    /// error compounds across all 30 blocks; see worklog 2026-06-04).
    pub sdpa_i8: Option<WgpuPipeline>,
    /// Mixed-precision f16 SDPA casts (`Bf16ToF16` / `F16ToBf16`). `Some` iff
    /// `BlockWgslConfigs::fast_sdpa` AND the acts are bf16 AND the backend has
    /// subgroups + SHADER_F16. When present, `op_sdpa` casts the (normed,
    /// f16-safe) Q/K/V to f16, runs the subgroup SDPA, and casts the output
    /// back to the bf16 residual -- the per-block speedup without exposing the
    /// outlier-prone residual stream to f16. An F16-act block leaves these
    /// `None` and feeds f16 Q/K/V to `sdpa_sg` directly.
    pub cast_to_f16: Option<WgpuPipeline>,
    pub cast_to_bf16: Option<WgpuPipeline>,
    /// Opt-in coopmat (tensor-core) matmul on the outlier-bound bf16 sites that
    /// i8 can't touch (proj = attn-out, ffn_down = gelu product, qkv = cross
    /// from un-normed text). `Some` only when the device exposes a usable
    /// coopmat config, the block runs fast_mixed, and the site is opted in via
    /// [`BlockWgslConfigs::coopmat_acts`]. Takes precedence over the dense/i8
    /// path in `dispatch_matmul_site_coopmat`; falls back when `None`.
    pub coopmat_proj: Option<CoopmatStep>,
    pub coopmat_ffn_down: Option<CoopmatStep>,
    pub coopmat_qkv: Option<CoopmatStep>,
    /// On-GPU activation storage dtype for buffers compiled against this set
    /// of pipelines. Drives byte sizing of every transient alloc through the
    /// DiT block forward pass.
    pub act_dtype: ActDtype,
}

/// Per-block WGSL configurations. The 5 matmul kernels can each take a
/// distinct `WgslConfig` (so a quant `weight_dtype` can be pinned to the
/// main projections while keeping AdaLN at bf16). Every other op shares
/// [`Self::ops`]. All six configs must agree on `act_dtype` and
/// `bf16_quant_writes` since they read/write the same activation buffers;
/// the constructor validates this.
#[derive(Clone, Copy, Debug)]
pub struct BlockWgslConfigs {
    pub matmul_qkv: WgslConfig,
    /// Self-attention QKV. Split from `matmul_qkv` (which now serves only
    /// cross-attention) so the self-attn projection can pin a distinct weight
    /// encoding -- e.g. Q8_0 for the DP4A path -- while cross-attention, whose
    /// K/V project from un-normed text and overflow under i8 acts, stays dense.
    /// Set equal to `matmul_qkv` to keep the two identical (the default).
    pub matmul_qkv_self: WgslConfig,
    pub matmul_proj: WgslConfig,
    pub matmul_ffn_up: WgslConfig,
    pub matmul_ffn_down: WgslConfig,
    pub matmul_adaln: WgslConfig,
    pub ops: WgslConfig,
    /// Opt-in i8 attention: quantize q/k/v once post-rope, run `sdpa_i8`,
    /// feed its paired output to the proj matmul A-side. Requires F16 ops
    /// (SHADER_F16). Never affects the residual carry or elementwise ops.
    pub i8_sdpa: bool,
    /// Per-site opt-out of the i8 activation path (see [`DenseActSites`]).
    pub dense_acts: DenseActSites,
    /// Per-site opt-in of the coopmat tensor-core matmul (see [`CoopmatSites`]).
    pub coopmat_acts: CoopmatSites,
    /// Compile the large-D SDPA kernel (`SdpaF32LargeD`, one workgroup per Q
    /// row, D split across threads) so a block with `head_dim > 128` (e.g.
    /// Ideogram-4's 256) has a real attention path instead of overrunning the
    /// `SdpaF32` `MAX_D=128` shared tile. Only F32/F16 acts (the kernel has no
    /// bf16-packed variant); leave `false` for `head_dim <= 128`.
    pub large_d_sdpa: bool,
    /// Opt-in mixed-precision f16 SDPA for a bf16-act block: cast the normed,
    /// f16-safe Q/K/V to f16, run the subgroup SDPA, cast the output back to the
    /// bf16 residual. Only takes effect when `ops.act_dtype == Bf16` and the
    /// backend exposes subgroups + SHADER_F16 (an F16-act block already feeds
    /// f16 Q/K/V to `sdpa_sg`); a no-op otherwise. Set it only on blocks whose
    /// SDPA Q/K/V are post-rmsnorm/rope and O(1) (DiT self-attention); leave it
    /// off for raw-residual encoder attention (umT5 et al.).
    pub fast_sdpa: bool,
    /// Compile the decode SDPA kernel (`SdpaDecode`, one workgroup per Q row, the
    /// workgroup cooperates over the KV length). Set on autoregressive generators
    /// (the rewriter LM) so single-token decode against a long KV cache is not
    /// bottlenecked by `SdpaF32`'s one-thread-per-head softmax. `head_dim <= 128`,
    /// F32/Bf16 acts. The caller opts into it per-call via [`op_sdpa_decode`].
    pub decode_sdpa: bool,
}

/// Per-site DP4A opt-out: a site set here keeps its A-side dense at the
/// block act dtype and runs the dequant-once matmul even when the device
/// has DP4A. For sites whose A-side has no preceding norm and can carry
/// massive-activation outlier rows (Qwen3 attention-sink token at
/// proj/ffn_down, max-abs ~16k vs ~1 median): per-32 i8 act_quant crushes
/// the outlier's 31 block neighbors and corrupts that token's entire
/// output row. The weight encoding is unchanged; only the activation
/// quantization is bypassed.
/// Per-site opt-IN of the coopmat (tensor-core) matmul. Only the outlier-bound
/// bf16 sites that the i8 DP4A path can't handle are candidates: `proj`
/// (attn-out), `ffn_down` (gelu product), and `qkv` (cross-attn from un-normed
/// text -- the riskiest for f16, enable last). A set site is honored only when
/// the device exposes a usable coopmat config and the block runs fast_mixed;
/// otherwise it silently falls back to the dense/i8 path. Narrowly scoped to
/// the Wan video DiT by design (see memory feedback_coopmat_scope).
#[derive(Clone, Copy, Debug, Default)]
pub struct CoopmatSites {
    pub proj: bool,
    pub ffn_down: bool,
    pub qkv: bool,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct DenseActSites {
    pub qkv: bool,
    /// Self-attention QKV (Wan's split site; the shared `Block::forward` never
    /// uses it). Distinct from `qkv` so a model can i8 the normed self-attn
    /// projection while keeping cross-attn QKV (un-normed text K/V) dense.
    pub qkv_self: bool,
    pub proj: bool,
    pub ffn_up: bool,
    pub ffn_down: bool,
}

impl BlockWgslConfigs {
    /// All six configs identical, i8_sdpa off. Existing call sites that
    /// don't mix weight encodings within a block use this.
    pub fn uniform(cfg: WgslConfig) -> Self {
        Self {
            matmul_qkv: cfg,
            matmul_qkv_self: cfg,
            matmul_proj: cfg,
            matmul_ffn_up: cfg,
            matmul_ffn_down: cfg,
            matmul_adaln: cfg,
            ops: cfg,
            i8_sdpa: false,
            dense_acts: DenseActSites::default(),
            coopmat_acts: CoopmatSites::default(),
            large_d_sdpa: false,
            fast_sdpa: false,
            decode_sdpa: false,
        }
    }

    fn validate(&self) {
        let a = self.ops.act_dtype;
        let q = self.ops.bf16_quant_writes;
        // I8 is a matmul/sdpa-internal storage form, never a block-wide ops
        // dtype: per-32-block i8 of the residual stream is numerically
        // unsound (outlier channels; see worklog 2026-06-04).
        assert_ne!(
            a,
            ActDtype::I8,
            "BlockWgslConfigs: ActDtype::I8 is not a valid ops act_dtype"
        );
        // All five matmuls share the block act_dtype: their outputs and
        // inputs are block-stream activations.
        for c in [
            self.matmul_qkv,
            self.matmul_qkv_self,
            self.matmul_proj,
            self.matmul_ffn_up,
            self.matmul_ffn_down,
            self.matmul_adaln,
        ] {
            assert_eq!(
                c.act_dtype, a,
                "BlockWgslConfigs: matmul act_dtype must match ops.act_dtype"
            );
            assert_eq!(
                c.bf16_quant_writes, q,
                "BlockWgslConfigs: matmul bf16_quant_writes must match ops"
            );
        }
        if self.i8_sdpa {
            assert!(
                a == ActDtype::F16 || (a == ActDtype::Bf16 && self.fast_sdpa),
                "BlockWgslConfigs: i8_sdpa requires F16 ops, or Bf16 ops with fast_sdpa \
                 (the mixed-precision casts feed act_quant and the dense-bf16 output mode)"
            );
        }
        if self.large_d_sdpa {
            assert!(
                matches!(a, ActDtype::F32 | ActDtype::F16),
                "BlockWgslConfigs: large_d_sdpa supports only F32/F16 acts (no bf16-packed variant)"
            );
        }
    }
}

impl BlockPipelines {
    /// Bytes for `n` activation elements at this pipeline set's dtype.
    pub fn act_bytes(&self, n: u32) -> u64 {
        n as u64 * self.act_dtype.bytes_per_elem()
    }

    /// Bytes for the per-(rows, dim/32) f32 params buffer that pairs with an
    /// i8-quantized `rows * dim` activation (sdpa_i8 I/O, matmul-site
    /// transcode scratch).
    pub fn i8_scale_bytes(rows: u32, dim: u32) -> u64 {
        ActDtype::I8.scale_bytes(rows as u64, dim as u64)
    }

    /// True iff this pipeline set runs the opt-in i8 attention path.
    pub fn i8_sdpa(&self) -> bool {
        self.sdpa_i8.is_some()
    }

    pub async fn compile(
        backend: &WgpuBackend,
        cfgs: &BlockWgslConfigs,
    ) -> Result<Self, WgpuError> {
        cfgs.validate();
        let cfg = &cfgs.ops;
        let cfg_compat = cfg;
        let matmuls = BlockMatmuls::for_cfgs(cfgs);
        let mm_layout = <MatMulF32 as MatmulOp>::layout();
        // When weight_dtype is Quant, the matmul pipeline is built against a
        // pre-dequanted dense workspace (see `BlockMatmuls::for_cfgs`
        // square/wide_k closures: those configs already set b_nmajor=true).
        // Override weight_dtype for the matmul WGSL build to the workspace's
        // storage dtype, and compile a parallel dequant pipeline matching.
        //
        // Workspace target tracks act_dtype: F16 acts pair with the native
        // f16 workspace (zero-conversion B-load fast path); F32/Bf16 acts
        // fall back to the bf16-packed workspace. The act_dtype is uniform
        // across the block by construction (pipeline.rs picks one per DiT).
        let dequant_target = if cfg.act_dtype == ActDtype::F16 {
            thinfer_core::ops::dequant::DequantTarget::F16
        } else {
            thinfer_core::ops::dequant::DequantTarget::Bf16
        };
        let workspace_weight_dtype = match dequant_target {
            thinfer_core::ops::dequant::DequantTarget::F16 => WeightDtype::F16,
            thinfer_core::ops::dequant::DequantTarget::Bf16 => WeightDtype::Bf16,
        };
        let matmul_cfg_for = |cfg: WgslConfig| -> WgslConfig {
            if matches!(cfg.weight_dtype, WeightDtype::Quant(_)) {
                WgslConfig {
                    weight_dtype: workspace_weight_dtype,
                    ..cfg
                }
            } else {
                cfg
            }
        };
        let qkv_mm_cfg = matmul_cfg_for(cfgs.matmul_qkv);
        let qkv_self_mm_cfg = matmul_cfg_for(cfgs.matmul_qkv_self);
        let proj_mm_cfg = matmul_cfg_for(cfgs.matmul_proj);
        let ffn_up_mm_cfg = matmul_cfg_for(cfgs.matmul_ffn_up);
        let ffn_down_mm_cfg = matmul_cfg_for(cfgs.matmul_ffn_down);
        let qkv_wgsl = matmuls.qkv.wgsl(&qkv_mm_cfg);
        let qkv_self_wgsl = matmuls.qkv_self.wgsl(&qkv_self_mm_cfg);
        let proj_wgsl = matmuls.proj.wgsl(&proj_mm_cfg);
        let ffn_up_wgsl = matmuls.ffn_up.wgsl(&ffn_up_mm_cfg);
        let ffn_down_wgsl = matmuls.ffn_down.wgsl(&ffn_down_mm_cfg);
        let adaln_wgsl = matmuls.adaln.wgsl(&cfgs.matmul_adaln);
        // Module-level dense linears: force a bf16 weight regardless of the
        // block's quant choice (the patch/embedder/proj_out weights stay bf16).
        let module_cfg = WgslConfig {
            weight_dtype: WeightDtype::Bf16,
            ..*cfg
        };
        let module_wgsl = matmuls.module.wgsl(&module_cfg);
        // Build dequant pipelines for sites whose source weight is Quant.
        let dq_layout = thinfer_core::ops::dequant::layout();
        let build_dq =
            async |label: &str, wd: WeightDtype| -> Result<Option<DequantStep>, WgpuError> {
                match wd {
                    WeightDtype::Quant(scheme) => {
                        let wgsl = thinfer_core::ops::dequant::build_wgsl(scheme, dequant_target);
                        let pipeline = backend
                            .create_pipeline(label, &wgsl, "main", dq_layout)
                            .await?;
                        Ok(Some(DequantStep { pipeline, scheme }))
                    }
                    _ => Ok(None),
                }
            };
        let dequant_qkv = build_dq("dequant_qkv", cfgs.matmul_qkv.weight_dtype).await?;
        let dequant_qkv_self =
            build_dq("dequant_qkv_self", cfgs.matmul_qkv_self.weight_dtype).await?;
        let dequant_proj = build_dq("dequant_proj", cfgs.matmul_proj.weight_dtype).await?;
        let dequant_ffn_up = build_dq("dequant_ffn_up", cfgs.matmul_ffn_up.weight_dtype).await?;
        let dequant_ffn_down =
            build_dq("dequant_ffn_down", cfgs.matmul_ffn_down.weight_dtype).await?;
        // DP4A int8 path. Gated on the WGSL packed_4x8_integer_dot_product
        // language feature (queried on the wgpu Instance). When present we
        // build a per-site (dequant_i8, matmul_i8) pair for each Quant
        // matmul; block.rs forward dispatches act_quant -> dequant_i8 ->
        // matmul_i8 (DP4A) and skips the F16-dequant matmul path. When
        // absent (Firefox WebGPU WIP, some Safari, older drivers), these
        // stay None and the F16-dequant path above runs unchanged. The
        // DP4A path also requires SHADER_F16 because the matmul output is
        // paired vec2<f16> (matching the rest of the F16-act DiT block);
        // when SHADER_F16 is absent, the I8 path is also disabled.
        let (sg_min, sg_max) = backend.subgroup_size_range();
        // Mixed-precision opt-in (`fast_sdpa`) for a bf16-act block: builds the
        // bf16<->f16 act casts so the f16 subgroup SDPA AND the i8 DP4A matmul
        // (both compute in f16/i8 and accumulate back into the bf16 residual)
        // can run even though the block acts + residual stay bf16. Needs
        // subgroups + SHADER_F16, same as the native f16 path.
        let fast_mixed = cfgs.fast_sdpa
            && cfg.act_dtype == ActDtype::Bf16
            && backend.supports_subgroups()
            && backend.supports_shader_f16()
            && sg_min >= 4;
        // DP4A int8 matmul: native on an F16-act block; on a bf16 block it rides
        // the fast_mixed casts (A-side bf16->f16->i8 in, i8 matmul f16 out->bf16).
        let use_dp4a = backend.supports_packed_int_dot()
            && backend.supports_shader_f16()
            && (cfg.act_dtype == ActDtype::F16 || fast_mixed);
        // Matmul subgroups: the ORT-style register-resident kernel branches
        // at runtime on `sg_size >= 16` (shuffle path) with a broadcast
        // shared-read fallback, so the flag only requires the feature; any
        // reported size range is safe.
        // Matmul subgroups stay OFF: the shuffle path measured ~30% SLOWER
        // than the broadcast shared-read path on NVIDIA sg=32 (a vec4
        // subgroupShuffle lowers to 4 SHFL, ~one per dp4a; broadcast reads
        // are served to all 16 lanes in one transaction). ORT gates its
        // shuffle path to Intel sg=16; revisit per-vendor if a browser
        // measurement on Intel ever justifies it. sdpa subgroups (configured
        // below) are unaffected and stay on.
        let i8_cfg = thinfer_core::ops::matmul_i8::MatMulI8Config::DEFAULT;
        // Subgroup small-D sdpa: a CL-lane cluster owns each Q row, so CL must
        // divide the ACTUAL runtime subgroup size. Pick CL = min(8, sg_min),
        // which divides any power-of-2 size >= the reported floor: native
        // (sg_min=32) -> CL=8 (unchanged); web/mobile, where the browser reports
        // the spec floor of 4, -> CL=4. (sg_min >= 4 guards pathological adapters
        // that expose subgroups but report a sub-spec floor.)
        // `THINFER_SDPA_SG_CL` (native perf A/B) overrides within [4, sg_min]:
        // smaller CL = fewer shuffle-reduce hops per key but more rows in
        // flight per workgroup. NOT bit-exact across CL values (the per-key
        // dot's f32 reduce order changes); any default change gates on the
        // e2e parity bands like a kernel change.
        let cl_default = if sg_min >= 8 { 8u32 } else { 4u32 };
        let sdpa_sg_cl = std::env::var("THINFER_SDPA_SG_CL")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .filter(|c| c.is_power_of_two() && *c >= 4 && *c <= sg_min.max(4))
            .unwrap_or(cl_default);
        // Q-register blocking: R Q rows per CL-lane cluster, amortizing the
        // per-key K/V shared-tile loads + f16->f32 converts across R rows.
        // BIT-EXACT vs R=1 (each row's f32 accumulation order is unchanged);
        // costs R x the q/o register set, so occupancy decides the win.
        // `THINFER_SDPA_SG_QR` A/B override; default 1 until measured.
        let sdpa_sg_qr = std::env::var("THINFER_SDPA_SG_QR")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .filter(|r| matches!(r, 1 | 2 | 4))
            .unwrap_or(1);
        let use_sdpa_sg =
            cfg.act_dtype == ActDtype::F16 && backend.supports_subgroups() && sg_min >= 4;
        let build_sdpa_sg = use_sdpa_sg || fast_mixed;
        tracing::info!(
            target: thinfer_core::trace::ADAPTER,
            use_dp4a = use_dp4a,
            sdpa_use_subgroup = use_sdpa_sg,
            fast_mixed_sdpa = fast_mixed,
            sdpa_sg_cl = sdpa_sg_cl,
            sdpa_sg_qr = sdpa_sg_qr,
            matmul_i8_tile = i8_cfg.tile,
            matmul_i8_use_subgroup = i8_cfg.use_subgroup,
            shader_f16 = backend.supports_shader_f16(),
            packed_int_dot = backend.supports_packed_int_dot(),
            subgroups = backend.supports_subgroups(),
            subgroup_min_size = sg_min,
            subgroup_max_size = sg_max,
            act_dtype = ?cfg.act_dtype,
            "z-image block pipeline config",
        );
        let dq_i8_layout = thinfer_core::ops::dequant_i8::layout();
        let mm_i8_layout = thinfer_core::ops::matmul_i8::layout();
        let build_i8 =
            async |site: &str,
                   wd: WeightDtype,
                   dense_acts: bool|
                   -> Result<(Option<DequantStep>, Option<WgpuPipeline>), WgpuError> {
                // dense_acts sites skip the i8 pair entirely: dispatch falls
                // through to the dequant-once dense matmul built above.
                if !use_dp4a || dense_acts {
                    return Ok((None, None));
                }
                match wd {
                    WeightDtype::Quant(scheme) => {
                        let dq_wgsl = thinfer_core::ops::dequant_i8::build_wgsl(scheme);
                        let dq_pipe = backend
                            .create_pipeline(
                                &format!("dequant_i8_{site}"),
                                &dq_wgsl,
                                "main",
                                dq_i8_layout,
                            )
                            .await?;
                        // Subgroup-using shader: prepend `enable subgroups;` on
                        // the web (Tint) backend; native (naga) returns "".
                        let mm_wgsl = format!(
                            "{}{}",
                            backend.subgroup_enable_directive(),
                            thinfer_core::ops::matmul_i8::build_wgsl(&i8_cfg),
                        );
                        let mm_pipe = backend
                            .create_pipeline(
                                &format!("matmul_i8_{site}"),
                                &mm_wgsl,
                                "main",
                                mm_i8_layout,
                            )
                            .await?;
                        Ok((
                            Some(DequantStep {
                                pipeline: dq_pipe,
                                scheme,
                            }),
                            Some(mm_pipe),
                        ))
                    }
                    _ => Ok((None, None)),
                }
            };
        let (dequant_i8_qkv, matmul_i8_qkv) =
            build_i8("qkv", cfgs.matmul_qkv.weight_dtype, cfgs.dense_acts.qkv).await?;
        let (dequant_i8_qkv_self, matmul_i8_qkv_self) = build_i8(
            "qkv_self",
            cfgs.matmul_qkv_self.weight_dtype,
            cfgs.dense_acts.qkv_self,
        )
        .await?;
        let (dequant_i8_proj, matmul_i8_proj) =
            build_i8("proj", cfgs.matmul_proj.weight_dtype, cfgs.dense_acts.proj).await?;
        let (dequant_i8_ffn_up, matmul_i8_ffn_up) = build_i8(
            "ffn_up",
            cfgs.matmul_ffn_up.weight_dtype,
            cfgs.dense_acts.ffn_up,
        )
        .await?;
        let (dequant_i8_ffn_down, matmul_i8_ffn_down) = build_i8(
            "ffn_down",
            cfgs.matmul_ffn_down.weight_dtype,
            cfgs.dense_acts.ffn_down,
        )
        .await?;
        // Coopmat (tensor-core) path: opt-in per site, only when the device
        // exposes a usable square f16/f32 config, the subgroup width is uniform
        // (baked as the kernel workgroup size), and the block runs fast_mixed
        // (so the bf16<->f16 act casts exist). The weight is dequanted to f16
        // [N,K] nmajor and consumed column-major (b_col_major).
        let use_coopmat = fast_mixed && backend.coopmat().is_some() && sg_min == sg_max;
        let build_coopmat = async |label: &str,
                                   wd: WeightDtype,
                                   opt_in: bool|
               -> Result<Option<CoopmatStep>, WgpuError> {
            if !use_coopmat || !opt_in {
                return Ok(None);
            }
            let WeightDtype::Quant(scheme) = wd else {
                // Coopmat needs an f16 weight; today only the Quant->f16 dequant
                // path is wired (the Wan DiT weights are GGUF Quant).
                return Ok(None);
            };
            let dq_wgsl = thinfer_core::ops::dequant::build_wgsl(
                scheme,
                thinfer_core::ops::dequant::DequantTarget::F16,
            );
            let dq_pipe = backend
                .create_pipeline(
                    &format!("dequant_f16_{label}"),
                    &dq_wgsl,
                    "main",
                    thinfer_core::ops::dequant::layout(),
                )
                .await?;
            let cm = backend.coopmat().expect("use_coopmat gates on coopmat()");
            let mut cfg = thinfer_core::ops::matmul_coopmat::CoopmatMatmulConfig::new(
                cm.tile,
                sg_min,
                thinfer_core::ops::matmul_coopmat::CoopmatOut::F16,
            );
            cfg.b_col_major = true;
            let mm_wgsl = thinfer_core::ops::matmul_coopmat::build_wgsl(&cfg);
            let mm_pipe = backend
                .create_pipeline(
                    &format!("matmul_coopmat_{label}"),
                    &mm_wgsl,
                    "main",
                    thinfer_core::ops::matmul_coopmat::layout(),
                )
                .await?;
            Ok(Some(CoopmatStep {
                dequant_f16: DequantStep {
                    pipeline: dq_pipe,
                    scheme,
                },
                matmul: mm_pipe,
                cfg,
            }))
        };
        let coopmat_proj = build_coopmat(
            "proj",
            cfgs.matmul_proj.weight_dtype,
            cfgs.coopmat_acts.proj,
        )
        .await?;
        let coopmat_ffn_down = build_coopmat(
            "ffn_down",
            cfgs.matmul_ffn_down.weight_dtype,
            cfgs.coopmat_acts.ffn_down,
        )
        .await?;
        let coopmat_qkv =
            build_coopmat("qkv", cfgs.matmul_qkv.weight_dtype, cfgs.coopmat_acts.qkv).await?;
        // act_quant serves two consumers: the matmul-site dense->paired
        // transcode on every i8 site, and the post-rope q/k/v quantize
        // when i8 attention is enabled.
        let any_i8_site = [
            matmul_i8_qkv.is_some(),
            matmul_i8_qkv_self.is_some(),
            matmul_i8_proj.is_some(),
            matmul_i8_ffn_up.is_some(),
            matmul_i8_ffn_down.is_some(),
        ]
        .into_iter()
        .any(|s| s);
        let act_quant = if any_i8_site || cfgs.i8_sdpa {
            let wgsl = thinfer_core::ops::act_quant::build_wgsl();
            Some(
                backend
                    .create_pipeline(
                        "act_quant",
                        &wgsl,
                        "main",
                        thinfer_core::ops::act_quant::layout(),
                    )
                    .await?,
            )
        } else {
            None
        };
        let sdpa_i8 = if cfgs.i8_sdpa {
            let wgsl = thinfer_core::ops::sdpa_i8::build_wgsl();
            Some(
                backend
                    .create_pipeline(
                        "sdpa_i8",
                        &wgsl,
                        "main",
                        thinfer_core::ops::sdpa_i8::layout(),
                    )
                    .await?,
            )
        } else {
            None
        };
        // Paired-acts × bf16 weights mixed matmul. Only the attn-proj site
        // can see a paired A-side (the sdpa_i8 output); built when i8_sdpa
        // is on AND that site keeps Bf16 weights.
        let needs_i8_bf16 =
            cfgs.i8_sdpa && matches!(cfgs.matmul_proj.weight_dtype, WeightDtype::Bf16);
        let i8_bf16_cfg = thinfer_core::ops::matmul_i8_bf16::MatMulI8Bf16Config::DEFAULT;
        let matmul_i8_bf16 = if needs_i8_bf16 {
            let wgsl = thinfer_core::ops::matmul_i8_bf16::build_wgsl(&i8_bf16_cfg);
            Some(
                backend
                    .create_pipeline(
                        "matmul_i8_bf16",
                        &wgsl,
                        "main",
                        thinfer_core::ops::matmul_i8_bf16::layout(),
                    )
                    .await?,
            )
        } else {
            None
        };
        // Pair the bf16-block-sum precompute with the i8×bf16 matmul. The
        // sum carries the asymmetric correction-term factor and is computed
        // per dispatch into scope (mirrors dequant_i8 producing b_qsum for
        // the Quant-weight path).
        let bf16_block_sum = if needs_i8_bf16 {
            let wgsl = thinfer_core::ops::bf16_block_sum::build_wgsl();
            Some(
                backend
                    .create_pipeline(
                        "bf16_block_sum",
                        &wgsl,
                        "main",
                        thinfer_core::ops::bf16_block_sum::layout(),
                    )
                    .await?,
            )
        } else {
            None
        };
        Ok(Self {
            matmul_qkv: backend
                .create_pipeline("matmul_qkv", &qkv_wgsl, "main", mm_layout)
                .await?,
            matmul_qkv_self: backend
                .create_pipeline("matmul_qkv_self", &qkv_self_wgsl, "main", mm_layout)
                .await?,
            matmul_proj: backend
                .create_pipeline("matmul_proj", &proj_wgsl, "main", mm_layout)
                .await?,
            matmul_ffn_up: backend
                .create_pipeline("matmul_ffn_up", &ffn_up_wgsl, "main", mm_layout)
                .await?,
            matmul_ffn_down: backend
                .create_pipeline("matmul_ffn_down", &ffn_down_wgsl, "main", mm_layout)
                .await?,
            matmul_adaln: backend
                .create_pipeline("matmul_adaln", &adaln_wgsl, "main", mm_layout)
                .await?,
            matmul_module: backend
                .create_pipeline("matmul_module", &module_wgsl, "main", mm_layout)
                .await?,
            dequant_qkv,
            dequant_qkv_self,
            dequant_proj,
            dequant_ffn_up,
            dequant_ffn_down,
            dequant_i8_qkv,
            dequant_i8_qkv_self,
            dequant_i8_proj,
            dequant_i8_ffn_up,
            dequant_i8_ffn_down,
            matmul_i8_qkv,
            matmul_i8_qkv_self,
            matmul_i8_proj,
            matmul_i8_ffn_up,
            matmul_i8_ffn_down,
            act_quant,
            matmul_i8_cfg: i8_cfg,
            matmul_i8_bf16,
            matmul_i8_bf16_cfg: i8_bf16_cfg,
            bf16_block_sum,
            matmuls,
            rmsnorm: backend
                .create_pipeline(
                    "rmsnorm",
                    <RmsNormF32 as RmsNormOp>::wgsl(cfg_compat),
                    "main",
                    <RmsNormF32 as RmsNormOp>::layout(),
                )
                .await?,
            layernorm: backend
                .create_pipeline(
                    "layernorm",
                    <LayerNormF32 as LayerNormOp>::wgsl(cfg_compat),
                    "main",
                    <LayerNormF32 as LayerNormOp>::layout(),
                )
                .await?,
            rope: backend
                .create_pipeline(
                    "rope",
                    <RopeF32 as RopeOp>::wgsl(cfg_compat),
                    "main",
                    <RopeF32 as RopeOp>::layout(),
                )
                .await?,
            rope_halfrot: backend
                .create_pipeline(
                    "rope_halfrot",
                    <RopeF32HalfRot as RopeOp>::wgsl(cfg_compat),
                    "main",
                    <RopeF32HalfRot as RopeOp>::layout(),
                )
                .await?,
            sdpa: backend
                .create_pipeline(
                    "sdpa",
                    <SdpaF32 as SdpaOp>::wgsl(cfg_compat),
                    "main",
                    <SdpaF32 as SdpaOp>::layout(),
                )
                .await?,
            sdpa_sg: if build_sdpa_sg {
                // Subgroup-using shader: prepend `enable subgroups;` on the web
                // (Tint) backend; native (naga) returns "". CL/R are baked into
                // the kernel here and must match `sdpa_sg_cl`/`sdpa_sg_qr` at
                // dispatch.
                let sdpa_sg_wgsl = format!(
                    "{}{}",
                    backend.subgroup_enable_directive(),
                    thinfer_core::ops::sdpa::build_f16_sg_wgsl(sdpa_sg_cl, sdpa_sg_qr),
                );
                Some(
                    backend
                        .create_pipeline(
                            "sdpa_sg",
                            &sdpa_sg_wgsl,
                            "main",
                            thinfer_core::ops::sdpa::sg_layout(),
                        )
                        .await?,
                )
            } else {
                None
            },
            sdpa_sg_win: if build_sdpa_sg {
                // Same subgroup kernel as `sdpa_sg` plus the temporal-window
                // tile-skip + per-key fold. Built unconditionally with `sdpa_sg`
                // (cheap) so a run can switch it on via the uniform at dispatch.
                let win_wgsl = format!(
                    "{}{}",
                    backend.subgroup_enable_directive(),
                    thinfer_core::ops::sdpa::build_f16_sg_windowed_wgsl(sdpa_sg_cl, sdpa_sg_qr),
                );
                Some(
                    backend
                        .create_pipeline(
                            "sdpa_sg_win",
                            &win_wgsl,
                            "main",
                            thinfer_core::ops::sdpa::sg_layout(),
                        )
                        .await?,
                )
            } else {
                None
            },
            sdpa_sg_cl,
            sdpa_sg_qr,
            sdpa_large_d: if cfgs.large_d_sdpa {
                Some(
                    backend
                        .create_pipeline(
                            "sdpa_large_d",
                            <SdpaF32LargeD as SdpaOp>::wgsl(cfg_compat),
                            "main",
                            <SdpaF32LargeD as SdpaOp>::layout(),
                        )
                        .await?,
                )
            } else {
                None
            },
            sdpa_decode: if cfgs.decode_sdpa {
                Some(
                    backend
                        .create_pipeline(
                            "sdpa_decode",
                            <thinfer_core::ops::SdpaDecode as SdpaOp>::wgsl(cfg_compat),
                            "main",
                            <thinfer_core::ops::SdpaDecode as SdpaOp>::layout(),
                        )
                        .await?,
                )
            } else {
                None
            },
            qkv_split: backend
                .create_pipeline(
                    "qkv_split",
                    <QkvSplitF32 as QkvSplitOp>::wgsl(cfg_compat),
                    "main",
                    <QkvSplitF32 as QkvSplitOp>::layout(),
                )
                .await?,
            silu: backend
                .create_pipeline("silu", SiluF32::wgsl(cfg_compat), "main", SiluF32::layout())
                .await?,
            silu_mul: backend
                .create_pipeline(
                    "silu_mul",
                    SiluMulF32::wgsl(cfg_compat),
                    "main",
                    SiluMulF32::layout(),
                )
                .await?,
            gelu: backend
                .create_pipeline("gelu", GeluF32::wgsl(cfg_compat), "main", GeluF32::layout())
                .await?,
            add: backend
                .create_pipeline("add", AddF32::wgsl(cfg_compat), "main", AddF32::layout())
                .await?,
            mul: backend
                .create_pipeline("mul", MulF32::wgsl(cfg_compat), "main", MulF32::layout())
                .await?,
            tanh: backend
                .create_pipeline("tanh", TanhF32::wgsl(cfg_compat), "main", TanhF32::layout())
                .await?,
            bcast_affine: backend
                .create_pipeline(
                    "bcast_affine",
                    <BcastAffineF32 as BcastAffineOp>::wgsl(cfg_compat),
                    "main",
                    <BcastAffineF32 as BcastAffineOp>::layout(),
                )
                .await?,
            bcast_fma: backend
                .create_pipeline(
                    "bcast_fma",
                    <BcastFmaF32 as BcastFmaOp>::wgsl(cfg_compat),
                    "main",
                    <BcastFmaF32 as BcastFmaOp>::layout(),
                )
                .await?,
            bcast_modulate: backend
                .create_pipeline(
                    "bcast_modulate",
                    <BcastModulateF32 as BcastModulateOp>::wgsl(cfg_compat),
                    "main",
                    <BcastModulateF32 as BcastModulateOp>::layout(),
                )
                .await?,
            bcast_add: backend
                .create_pipeline(
                    "bcast_add",
                    <BcastAddF32 as BcastAddOp>::wgsl(cfg_compat),
                    "main",
                    <BcastAddF32 as BcastAddOp>::layout(),
                )
                .await?,
            bcast_mul: backend
                .create_pipeline(
                    "bcast_mul",
                    <BcastMulF32 as BcastAddOp>::wgsl(cfg_compat),
                    "main",
                    <BcastMulF32 as BcastAddOp>::layout(),
                )
                .await?,
            scatter_pad_rows: backend
                .create_pipeline(
                    "scatter_pad_rows",
                    <ScatterPadRowsF32 as ScatterPadRowsOp>::wgsl(cfg_compat),
                    "main",
                    <ScatterPadRowsF32 as ScatterPadRowsOp>::layout(),
                )
                .await?,
            sdpa_i8,
            cast_to_f16: if fast_mixed {
                Some(
                    backend
                        .create_pipeline(
                            "cast_bf16_f16",
                            <Bf16ToF16 as Op>::wgsl(&cfgs.ops),
                            "main",
                            <Bf16ToF16 as Op>::layout(),
                        )
                        .await?,
                )
            } else {
                None
            },
            cast_to_bf16: if fast_mixed {
                Some(
                    backend
                        .create_pipeline(
                            "cast_f16_bf16",
                            <F16ToBf16 as Op>::wgsl(&cfgs.ops),
                            "main",
                            <F16ToBf16 as Op>::layout(),
                        )
                        .await?,
                )
            } else {
                None
            },
            coopmat_proj,
            coopmat_ffn_down,
            coopmat_qkv,
            act_dtype: cfg.act_dtype,
        })
    }
}

pub struct Block {
    pub cfg: BlockConfig,
}

/// Reference pair for an activation-bearing tap. Under I8 ops the caller
/// must populate `scale` with a `(rows, dim/32) * 4`-byte BufRef; under
/// non-I8 modes `scale` is `None` and only `data` is meaningful. Mirrors
/// `ActBufRef` at the BlockDebugTaps surface.
#[derive(Clone)]
pub struct ActTapBufRef {
    pub data: BufRef,
    pub scale: Option<BufRef>,
}

#[derive(Default, Clone)]
pub struct BlockDebugTaps {
    pub adaln_input: Option<BufRef>,
    pub adaln_pre: Option<BufRef>,
    pub adaln_full: Option<BufRef>,
    pub scale_msa: Option<BufRef>,
    pub gate_msa: Option<BufRef>,
    pub scale_mlp: Option<BufRef>,
    pub gate_mlp: Option<BufRef>,
    pub attn_norm1_out: Option<ActTapBufRef>,
    pub modulated_attn_in: Option<ActTapBufRef>,
    pub attn_q: Option<ActTapBufRef>,
    pub attn_k: Option<ActTapBufRef>,
    pub attn_v: Option<ActTapBufRef>,
    pub attn_q_norm: Option<ActTapBufRef>,
    pub attn_k_norm: Option<ActTapBufRef>,
    pub attn_q_rope: Option<ActTapBufRef>,
    pub attn_k_rope: Option<ActTapBufRef>,
    pub attn_sdpa: Option<ActTapBufRef>,
    pub attn_out: Option<ActTapBufRef>,
    pub attn_norm2_out: Option<ActTapBufRef>,
    pub x_mid: Option<ActTapBufRef>,
    pub ffn_norm1_out: Option<ActTapBufRef>,
    pub modulated_ffn_in: Option<ActTapBufRef>,
    pub ffn_raw: Option<ActTapBufRef>,
    pub ffn_norm2_out: Option<ActTapBufRef>,
    /// Pre-act_quant snapshot of the f16 qkv matmul scratch. Single buffer,
    /// no paired scale (the scratch is dense f16 sized `rows * n_qkv * 2`
    /// bytes). Only populated under I8 modes where act_quant is the next op.
    pub attn_qkv_f16_pre_quant: Option<BufRef>,
    /// Same idea for the attention output projection scratch
    /// (`rows * dim * 2` f16 bytes).
    pub attn_proj_f16_pre_quant: Option<BufRef>,
    /// Pre-act_quant f16 scratch from FFN w1 matmul (`rows * hid * 2` bytes).
    pub ffn_h1_f16_pre_quant: Option<BufRef>,
    /// Pre-act_quant f16 scratch from FFN w3 matmul (`rows * hid * 2` bytes).
    pub ffn_h3_f16_pre_quant: Option<BufRef>,
    /// Pre-act_quant f16 scratch from FFN w2 matmul (`rows * dim * 2` bytes).
    pub ffn_h2_f16_pre_quant: Option<BufRef>,
    /// DIAG: raw byte snapshot of `sa.data` (sdpa output paired data) at the
    /// proj matmul input, for offline decode against `attn_sdpa`. Size set
    /// by Block0TapBufs allocator (a few KiB).
    pub proj_sa_data_head: Option<BufRef>,
    /// DIAG: raw byte snapshot of `sa.scale` (sdpa output paired scale).
    pub proj_sa_scale_head: Option<BufRef>,
    /// DIAG: raw byte snapshot of dequanted W_o i8 (first cols x first K).
    pub proj_wo_b_i8_head: Option<BufRef>,
    /// DIAG: raw byte snapshot of W_o b_scale (first cols x first K-blocks).
    pub proj_wo_b_scale_head: Option<BufRef>,
    /// DIAG: raw byte snapshots at the QKV-matmul site (block-26 audit).
    /// `qkv_attn_in_data_head` is the i8 acts going INTO matmul_i8;
    /// `qkv_attn_in_params_head` is the paired (s,z) vec2<f16>;
    /// `qkv_b_i8_head`/`qkv_b_scale_head`/`qkv_b_qsum_head` are the
    /// dequant_i8 outputs (weight i8, weight f16 scale per K-block,
    /// weight f32 qsum per K-block).
    pub qkv_attn_in_data_head: Option<BufRef>,
    pub qkv_attn_in_params_head: Option<BufRef>,
    pub qkv_b_i8_head: Option<BufRef>,
    pub qkv_b_scale_head: Option<BufRef>,
    pub qkv_b_qsum_head: Option<BufRef>,
    /// DIAG: per-K-block GPU trace from `matmul_i8` for ONE target cell
    /// (block-26 audit). Hardcoded target inside `forward_taps_packed`
    /// (`m=287, n=255` Q col). Layout: (k/32) K-blocks * 8 f32 (= 960 f32
    /// at dim=3840) + 16-f32 probe area. Per block: (sa, za, sb, qsum, dot,
    /// main, corr, acc_running).
    pub qkv_dbg_trace_head: Option<BufRef>,
}

impl BlockDebugTaps {
    pub const EMPTY: Self = Self {
        adaln_input: None,
        adaln_pre: None,
        adaln_full: None,
        scale_msa: None,
        gate_msa: None,
        scale_mlp: None,
        gate_mlp: None,
        attn_norm1_out: None,
        modulated_attn_in: None,
        attn_q: None,
        attn_k: None,
        attn_v: None,
        attn_q_norm: None,
        attn_k_norm: None,
        attn_q_rope: None,
        attn_k_rope: None,
        attn_sdpa: None,
        attn_out: None,
        attn_norm2_out: None,
        x_mid: None,
        ffn_norm1_out: None,
        modulated_ffn_in: None,
        ffn_raw: None,
        ffn_norm2_out: None,
        attn_qkv_f16_pre_quant: None,
        attn_proj_f16_pre_quant: None,
        ffn_h1_f16_pre_quant: None,
        ffn_h3_f16_pre_quant: None,
        ffn_h2_f16_pre_quant: None,
        proj_sa_data_head: None,
        proj_sa_scale_head: None,
        proj_wo_b_i8_head: None,
        proj_wo_b_scale_head: None,
        qkv_attn_in_data_head: None,
        qkv_attn_in_params_head: None,
        qkv_b_i8_head: None,
        qkv_b_scale_head: None,
        qkv_b_qsum_head: None,
        qkv_dbg_trace_head: None,
    };
}

impl Block {
    pub fn new(cfg: BlockConfig) -> Self {
        assert_eq!(
            cfg.n_heads * cfg.head_dim,
            cfg.dim,
            "dim must equal n_heads * head_dim"
        );
        if cfg.modulation {
            assert_eq!(
                cfg.batch, 1,
                "v1 modulation path is single-batch only (bcast ops are channel-broadcast)"
            );
        }
        Self { cfg }
    }

    pub fn kernel_keys() -> [KernelKey; 15] {
        [
            kk(<MatMulF32 as MatmulOp>::KERNEL_ID),
            kk(<RmsNormF32 as RmsNormOp>::KERNEL_ID),
            kk(<LayerNormF32 as LayerNormOp>::KERNEL_ID),
            kk(<RopeF32 as RopeOp>::KERNEL_ID),
            kk(<SdpaF32 as SdpaOp>::KERNEL_ID),
            kk(<QkvSplitF32 as QkvSplitOp>::KERNEL_ID),
            kk(<SiluF32 as Op>::KERNEL_ID),
            kk(<AddF32 as Op>::KERNEL_ID),
            kk(<MulF32 as Op>::KERNEL_ID),
            kk(<TanhF32 as Op>::KERNEL_ID),
            kk(<BcastAffineF32 as BcastAffineOp>::KERNEL_ID),
            kk(<BcastFmaF32 as BcastFmaOp>::KERNEL_ID),
            kk(<BcastModulateF32 as BcastModulateOp>::KERNEL_ID),
            kk(<BcastAddF32 as BcastAddOp>::KERNEL_ID),
            kk(<BcastMulF32 as BcastAddOp>::KERNEL_ID),
        ]
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward<'wsp>(
        &self,
        scope: &BatchScope<'wsp, WgpuBackend>,
        pipelines: &BlockPipelines,
        x_in: ActBuf<'wsp>,
        freqs_in: BatchBuf<'wsp>,
        mask_in: BatchBuf<'wsp>,
        adaln_input: Option<BatchBuf<'wsp>>,
        y_out: ActBuf<'wsp>,
        bufs: &'wsp BlockWeightBufs,
    ) -> Result<(), WgpuError> {
        self.forward_taps(
            scope,
            pipelines,
            x_in,
            freqs_in,
            mask_in,
            adaln_input,
            y_out,
            bufs,
            &BlockDebugTaps::EMPTY,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_taps<'wsp>(
        &self,
        scope: &BatchScope<'wsp, WgpuBackend>,
        pipelines: &BlockPipelines,
        x_in: ActBuf<'wsp>,
        freqs_in: BatchBuf<'wsp>,
        mask_in: BatchBuf<'wsp>,
        adaln_input: Option<BatchBuf<'wsp>>,
        y_out: ActBuf<'wsp>,
        bufs: &'wsp BlockWeightBufs,
        taps: &BlockDebugTaps,
    ) -> Result<(), WgpuError> {
        debug_assert_eq!(
            self.cfg.modulation,
            bufs.adaln.is_some(),
            "adaln bufs presence must match cfg.modulation"
        );
        let cfg = self.cfg;
        let rows = cfg.rows() as u32;
        let dim = cfg.dim as u32;
        let hd = cfg.head_dim as u32;
        let hq = cfg.n_heads as u32;
        let hkv = cfg.n_kv_heads as u32;
        let hid = cfg.ffn_hidden as u32;
        let eps = cfg.norm_eps;
        let scale = cfg.sdpa_scale();
        let b = cfg.batch as u32;
        let s = cfg.seq as u32;
        let ad = cfg.adaln_embed_dim as u32;

        // AdaLN prep: produces F32 chunks under I8 ops, otherwise act-dtype
        // chunks. Each chunk is `b * dim` elements of the AdaLN-output dtype.
        let ada: Option<AdaLnChunks<'wsp>> = match (cfg.modulation, &bufs.adaln, adaln_input) {
            (true, Some(w), Some(input)) => {
                copy_tap(
                    scope,
                    input,
                    taps.adaln_input.as_ref(),
                    pipelines.act_bytes(b * ad),
                )?;
                Some(self.prepare_adaln(scope, pipelines, *w, input, taps)?)
            }
            (false, None, None) => None,
            _ => panic!("modulation/adaln_input/adaln-bufs mismatch"),
        };
        let chunk_bytes = adaln_chunk_bytes(pipelines, b, dim);
        if let Some(a) = ada.as_ref() {
            copy_tap(scope, a.scale_msa, taps.scale_msa.as_ref(), chunk_bytes)?;
            copy_tap(scope, a.gate_msa, taps.gate_msa.as_ref(), chunk_bytes)?;
            copy_tap(scope, a.scale_mlp, taps.scale_mlp.as_ref(), chunk_bytes)?;
            copy_tap(scope, a.gate_mlp, taps.gate_mlp.as_ref(), chunk_bytes)?;
        }

        // ==================== Attention: pre-norm + modulate ====================
        let t1 = alloc_act(scope, pipelines, rows, dim)?;
        let an1 = scope.import_copy(bufs.attention_norm1);
        op_rmsnorm(scope, pipelines, x_in, an1, t1, rows, dim, eps)?;
        copy_tap_act(
            scope,
            pipelines,
            t1,
            taps.attn_norm1_out.as_ref(),
            rows,
            dim,
        )?;
        let attn_in: ActBuf<'wsp> = match ada.as_ref() {
            Some(a) => {
                let dst = alloc_act(scope, pipelines, rows, dim)?;
                op_bcast_affine(scope, pipelines, t1, a.scale_msa, dst, rows, dim, 1.0)?;
                dst
            }
            None => t1,
        };
        copy_tap_act(
            scope,
            pipelines,
            attn_in,
            taps.modulated_attn_in.as_ref(),
            rows,
            dim,
        )?;

        // ==================== Attention: fused QKV matmul + split ====================
        let (q, k, v) = {
            let _g = trace::scope!("attn_qkv").entered();
            debug_assert_eq!(
                hq, hkv,
                "fused QKV currently assumes hq == hkv (Z-Image); GQA needs schema rework"
            );
            let h = hq * hd;
            let n_qkv = 3 * h;
            let qkv_scratch = alloc_matmul_out_buf(scope, pipelines, rows * n_qkv)?;
            let dims_qkv = scope.u32x4_uniform(rows, n_qkv, dim, 0)?;
            let w_qkv = scope.import_copy(bufs.attn_qkv);
            Self::dispatch_matmul_site(
                scope,
                pipelines,
                attn_in,
                w_qkv,
                qkv_scratch,
                dims_qkv,
                pipelines.matmul_i8_qkv.as_ref(),
                pipelines.dequant_i8_qkv.as_ref(),
                pipelines.dequant_qkv.as_ref(),
                &pipelines.matmul_qkv,
                &pipelines.matmuls.qkv,
                rows,
                n_qkv,
                dim,
            )?;
            // Telemetry: snapshot the raw matmul output (kept for parity
            // tooling continuity; identical to the qkv output stream).
            if let Some(dst) = taps.attn_qkv_f16_pre_quant {
                let dst_b = scope.import_copy(dst);
                let bytes = pipelines.act_bytes(rows * n_qkv);
                scope.copy_buffer_to_buffer(qkv_scratch, 0, dst_b, 0, bytes)?;
            }
            let qkv_fused = ActBuf::dense(qkv_scratch);
            let q = alloc_act(scope, pipelines, rows, h)?;
            let k = alloc_act(scope, pipelines, rows, h)?;
            let v = alloc_act(scope, pipelines, rows, h)?;
            op_qkv_split(scope, pipelines, qkv_fused, q, k, v, rows, h)?;
            copy_tap_act(scope, pipelines, q, taps.attn_q.as_ref(), rows, h)?;
            copy_tap_act(scope, pipelines, k, taps.attn_k.as_ref(), rows, h)?;
            copy_tap_act(scope, pipelines, v, taps.attn_v.as_ref(), rows, h)?;
            (q, k, v)
        };

        // ==================== Attention: q/k norm + rope + sdpa ====================
        let sa = {
            let _g = trace::scope!("attn_sdpa").entered();
            let qn = alloc_act(scope, pipelines, rows * hq, hd)?;
            let kn = alloc_act(scope, pipelines, rows * hkv, hd)?;
            let nq = scope.import_copy(bufs.attn_norm_q);
            op_rmsnorm(scope, pipelines, q, nq, qn, rows * hq, hd, eps)?;
            copy_tap_act(
                scope,
                pipelines,
                qn,
                taps.attn_q_norm.as_ref(),
                rows * hq,
                hd,
            )?;
            let nk = scope.import_copy(bufs.attn_norm_k);
            op_rmsnorm(scope, pipelines, k, nk, kn, rows * hkv, hd, eps)?;
            copy_tap_act(
                scope,
                pipelines,
                kn,
                taps.attn_k_norm.as_ref(),
                rows * hkv,
                hd,
            )?;

            let qr = alloc_act(scope, pipelines, rows, hq * hd)?;
            let kr = alloc_act(scope, pipelines, rows, hkv * hd)?;
            if cfg.rope_halfrot {
                op_rope_halfrot(scope, pipelines, qn, freqs_in, qr, rows, hq, hd)?;
            } else {
                op_rope(scope, pipelines, qn, freqs_in, qr, rows, hq, hd)?;
            }
            copy_tap_act(
                scope,
                pipelines,
                qr,
                taps.attn_q_rope.as_ref(),
                rows,
                hq * hd,
            )?;
            if cfg.rope_halfrot {
                op_rope_halfrot(scope, pipelines, kn, freqs_in, kr, rows, hkv, hd)?;
            } else {
                op_rope(scope, pipelines, kn, freqs_in, kr, rows, hkv, hd)?;
            }
            copy_tap_act(
                scope,
                pipelines,
                kr,
                taps.attn_k_rope.as_ref(),
                rows,
                hkv * hd,
            )?;

            // i8 attention opt-in: quantize q/k/v ONCE, post-rope, into the
            // fused paired sdpa_i8 I/O slots. Otherwise sdpa runs dense.
            let (qx, kx, vx, sa) = if pipelines.i8_sdpa() {
                (
                    quant_for_sdpa(scope, pipelines, qr, rows, hq * hd)?,
                    quant_for_sdpa(scope, pipelines, kr, rows, hkv * hd)?,
                    quant_for_sdpa(scope, pipelines, v, rows, hq * hd)?,
                    alloc_act_sdpa_io(scope, pipelines, rows, hq * hd)?,
                )
            } else {
                (qr, kr, v, alloc_act(scope, pipelines, rows, hq * hd)?)
            };
            op_sdpa(
                scope, pipelines, qx, kx, vx, mask_in, sa, b, s, s, hq, hkv, hd, scale, 1,
            )?;
            copy_tap_act(scope, pipelines, sa, taps.attn_sdpa.as_ref(), rows, hq * hd)?;
            sa
        };

        // ==================== Attention: out projection + post-norm ====================
        let t2 = {
            let _g = trace::scope!("attn_proj").entered();
            let k_proj = hq * hd;
            let proj_scratch = alloc_matmul_out_buf(scope, pipelines, rows * dim)?;
            let dims_proj = scope.u32x4_uniform(rows, dim, k_proj, 0)?;
            let wo = scope.import_copy(bufs.attn_to_out);
            Self::dispatch_matmul_site_diag(
                scope,
                pipelines,
                sa,
                wo,
                proj_scratch,
                dims_proj,
                pipelines.matmul_i8_proj.as_ref(),
                pipelines.dequant_i8_proj.as_ref(),
                pipelines.dequant_proj.as_ref(),
                &pipelines.matmul_proj,
                &pipelines.matmuls.proj,
                rows,
                dim,
                k_proj,
                None,
                None,
                taps.proj_wo_b_i8_head,
                taps.proj_wo_b_scale_head,
                None,
                None,
                None,
            )?;
            if let Some(dst) = taps.attn_proj_f16_pre_quant {
                let dst_b = scope.import_copy(dst);
                let bytes = pipelines.act_bytes(rows * dim);
                scope.copy_buffer_to_buffer(proj_scratch, 0, dst_b, 0, bytes)?;
            }
            let proj = ActBuf::dense(proj_scratch);
            copy_tap_act(scope, pipelines, proj, taps.attn_out.as_ref(), rows, dim)?;

            let t2 = alloc_act(scope, pipelines, rows, dim)?;
            let an2 = scope.import_copy(bufs.attention_norm2);
            op_rmsnorm(scope, pipelines, proj, an2, t2, rows, dim, eps)?;
            copy_tap_act(
                scope,
                pipelines,
                t2,
                taps.attn_norm2_out.as_ref(),
                rows,
                dim,
            )?;
            t2
        };

        // ==================== Residual 1 ====================
        let x1 = alloc_act(scope, pipelines, rows, dim)?;
        self.residual(
            scope,
            pipelines,
            x_in,
            t2,
            ada.as_ref().map(|a| a.gate_msa),
            x1,
            rows,
            dim,
        )?;
        copy_tap_act(scope, pipelines, x1, taps.x_mid.as_ref(), rows, dim)?;

        // ==================== FFN: pre-norm + modulate ====================
        let t4 = {
            let _g = trace::scope!("ffn").entered();
            let t3 = alloc_act(scope, pipelines, rows, dim)?;
            let fn1 = scope.import_copy(bufs.ffn_norm1);
            op_rmsnorm(scope, pipelines, x1, fn1, t3, rows, dim, eps)?;
            copy_tap_act(scope, pipelines, t3, taps.ffn_norm1_out.as_ref(), rows, dim)?;
            let ffn_in: ActBuf<'wsp> = match ada.as_ref() {
                Some(a) => {
                    let dst = alloc_act(scope, pipelines, rows, dim)?;
                    op_bcast_affine(scope, pipelines, t3, a.scale_mlp, dst, rows, dim, 1.0)?;
                    dst
                }
                None => t3,
            };
            copy_tap_act(
                scope,
                pipelines,
                ffn_in,
                taps.modulated_ffn_in.as_ref(),
                rows,
                dim,
            )?;

            // ==================== FFN: w1 + w3 + silu_mul ====================
            let h1_scratch = alloc_matmul_out_buf(scope, pipelines, rows * hid)?;
            let h3_scratch = alloc_matmul_out_buf(scope, pipelines, rows * hid)?;
            let dims_ffn1 = scope.u32x4_uniform(rows, hid, dim, 0)?;
            let dims_ffn3 = scope.u32x4_uniform(rows, hid, dim, 0)?;
            let w1 = scope.import_copy(bufs.ffn_w1);
            let w3 = scope.import_copy(bufs.ffn_w3);
            Self::dispatch_matmul_site(
                scope,
                pipelines,
                ffn_in,
                w1,
                h1_scratch,
                dims_ffn1,
                pipelines.matmul_i8_ffn_up.as_ref(),
                pipelines.dequant_i8_ffn_up.as_ref(),
                pipelines.dequant_ffn_up.as_ref(),
                &pipelines.matmul_ffn_up,
                &pipelines.matmuls.ffn_up,
                rows,
                hid,
                dim,
            )?;
            Self::dispatch_matmul_site(
                scope,
                pipelines,
                ffn_in,
                w3,
                h3_scratch,
                dims_ffn3,
                pipelines.matmul_i8_ffn_up.as_ref(),
                pipelines.dequant_i8_ffn_up.as_ref(),
                pipelines.dequant_ffn_up.as_ref(),
                &pipelines.matmul_ffn_up,
                &pipelines.matmuls.ffn_up,
                rows,
                hid,
                dim,
            )?;
            let h1 = ActBuf::dense(h1_scratch);
            let h3 = ActBuf::dense(h3_scratch);
            let h13 = alloc_act(scope, pipelines, rows, hid)?;
            op_silu_mul(scope, pipelines, h1, h3, h13)?;

            // ==================== FFN: w2 + post-norm ====================
            let h2_scratch = alloc_matmul_out_buf(scope, pipelines, rows * dim)?;
            let dims_ffn2 = scope.u32x4_uniform(rows, dim, hid, 0)?;
            let w2 = scope.import_copy(bufs.ffn_w2);
            Self::dispatch_matmul_site(
                scope,
                pipelines,
                h13,
                w2,
                h2_scratch,
                dims_ffn2,
                pipelines.matmul_i8_ffn_down.as_ref(),
                pipelines.dequant_i8_ffn_down.as_ref(),
                pipelines.dequant_ffn_down.as_ref(),
                &pipelines.matmul_ffn_down,
                &pipelines.matmuls.ffn_down,
                rows,
                dim,
                hid,
            )?;
            let h2 = ActBuf::dense(h2_scratch);
            copy_tap_act(scope, pipelines, h2, taps.ffn_raw.as_ref(), rows, dim)?;

            let t4 = alloc_act(scope, pipelines, rows, dim)?;
            let fn2 = scope.import_copy(bufs.ffn_norm2);
            op_rmsnorm(scope, pipelines, h2, fn2, t4, rows, dim, eps)?;
            copy_tap_act(scope, pipelines, t4, taps.ffn_norm2_out.as_ref(), rows, dim)?;
            t4
        };

        // ==================== Residual 2 (writes y_out) ====================
        self.residual(
            scope,
            pipelines,
            x1,
            t4,
            ada.as_ref().map(|a| a.gate_mlp),
            y_out,
            rows,
            dim,
        )?;

        Ok(())
    }

    /// Estimated peak workspace bytes per phase for the packer.
    ///
    /// Phases:
    ///  - 0: AttnPreQkv (rmsnorm + optional AdaLN prep + optional modulate +
    ///    qkv matmul + qkv_split). Carries q, k, v (+ ada chunks).
    ///  - 1: AttnSdpaProj (sdpa norms+rope+sdpa + attn_proj matmul + post-norm
    ///    + residual1). Carries x1 (+ scale_mlp, gate_mlp).
    ///  - 2: Ffn1 (ffn-norm + optional modulate + w1 + w3 + silu_mul).
    ///    Carries h13, x1 (+ gate_mlp).
    ///  - 3: Ffn2 (w2 + ffn-norm-post + residual2). Writes y_out.
    ///
    /// Each phase's value is the SUM of every workspace alloc made inside its
    /// scope (BatchScope holds all guards until submit, so peak = sum).
    /// Conservative upper bound — assumes worst-case matmul path per site
    /// based on which pipeline fields are populated (DP4A > dequant-once > bf16).
    pub fn phase_peaks(&self, pipelines: &BlockPipelines) -> [u64; 4] {
        let cfg = self.cfg;
        let rows = cfg.rows() as u64;
        let dim = cfg.dim as u64;
        let hd = cfg.head_dim as u64;
        let hq = cfg.n_heads as u64;
        let hkv = cfg.n_kv_heads as u64;
        let hid = cfg.ffn_hidden as u64;
        let b = cfg.batch as u64;
        let h = hq * hd;
        let abe = pipelines.act_dtype.bytes_per_elem();
        let i8_sdpa = pipelines.i8_sdpa();
        // Dense activation / matmul-output scratch bytes.
        let act_b = |m: u64, d: u64| -> u64 { m * d * abe };
        // Fused paired sdpa_i8 I/O slot: packed i8 data + per-block scale.
        let sdpa_pair_b = |m: u64, d: u64| -> u64 { m * d + m * (d / 32) * 4 };
        // Matmul-site additional allocs (inside `dispatch_matmul_site`).
        // DP4A: b dequant (i8 + scale + qsum) plus the dense->paired A-side
        // transcode (skipped when A arrives paired, but counted always as a
        // conservative upper bound).
        let mm_site =
            |m: u64, n: u64, k: u64, mm_i8_available: bool, dq_dense_available: bool| -> u64 {
                if mm_i8_available && pipelines.act_quant.is_some() {
                    let b_side = n * k + 2 * (n * (k / 32) * 4);
                    let a_side = m * k + m * (k / 32) * 4;
                    b_side + a_side
                } else if dq_dense_available {
                    n * k * 2
                } else {
                    0
                }
            };
        let act = act_b(rows, dim);
        let q_b = act_b(rows, h);
        let kv_b = act_b(rows, hkv * hd);
        let hid_b = act_b(rows, hid);
        let modulated = cfg.modulation;
        let chunk_b = b * dim * abe;
        let ada_full_b = b * 4 * dim * abe;
        // AdaLN prep allocs: pre + full + 6 * chunk (scale_msa, gate_msa_pre,
        // scale_mlp, gate_mlp_pre, gate_msa, gate_mlp). All live through phase 1.
        let ada_full = if modulated {
            2 * ada_full_b + 6 * chunk_b
        } else {
            0
        };

        // Phase 0: ada_full + t1 + (modulate?attn_in:0) + qkv_scratch + q + 2*kv + mm_site
        let p0 = ada_full
            + act
            + if modulated { act } else { 0 }
            + act_b(rows, 3 * h)
            + mm_site(
                rows,
                3 * h,
                dim,
                pipelines.matmul_i8_qkv.is_some(),
                pipelines.dequant_qkv.is_some(),
            )
            + q_b
            + 2 * kv_b;

        // Phase 1: qn + kn + qr + kr + sa + (i8_sdpa? 3 quant pairs + paired
        // sa) + proj_scratch + mm_site + t2 + x1
        let sdpa_extra = if i8_sdpa {
            2 * sdpa_pair_b(rows, h) + sdpa_pair_b(rows, hkv * hd) + sdpa_pair_b(rows, h)
        } else {
            0
        };
        let p1 = q_b
            + kv_b
            + q_b
            + kv_b
            + if i8_sdpa { 0 } else { q_b }
            + sdpa_extra
            + act_b(rows, dim)
            + mm_site(
                rows,
                dim,
                h,
                pipelines.matmul_i8_proj.is_some(),
                pipelines.dequant_proj.is_some(),
            )
            + act
            + act;

        // Phase 2: t3 + (modulate?ffn_in:0) + 2*hid_scratch + 2*mm_site +
        // h13. Two separate b dequants under DP4A (w1, w3 differ).
        let p2 = act
            + if modulated { act } else { 0 }
            + 2 * act_b(rows, hid)
            + 2 * mm_site(
                rows,
                hid,
                dim,
                pipelines.matmul_i8_ffn_up.is_some(),
                pipelines.dequant_ffn_up.is_some(),
            )
            + hid_b;

        // Phase 3: h2_scratch + mm_site + t4
        let p3 = act_b(rows, dim)
            + mm_site(
                rows,
                dim,
                hid,
                pipelines.matmul_i8_ffn_down.is_some(),
                pipelines.dequant_ffn_down.is_some(),
            )
            + act;

        [p0, p1, p2, p3]
    }

    /// Packer-driven variant of [`Self::forward_taps`]. Inputs are `BufRef`s
    /// (re-imported into each sub-scope as needed) and the work is sliced
    /// into 4 phases. The packer cuts a scope whenever the next phase's
    /// estimated peak workspace would push live bytes past its budget;
    /// otherwise consecutive phases share a scope (zero overhead at small
    /// dims). See [`Self::phase_peaks`].
    #[allow(clippy::too_many_arguments)]
    pub async fn forward_taps_packed<'wsp>(
        &self,
        packer: &mut ScopePacker<'wsp, WgpuBackend>,
        pipelines: &BlockPipelines,
        x_in_ref: ActBufRef,
        freqs_in_ref: BufRef,
        mask_in_ref: BufRef,
        adaln_input_ref: Option<BufRef>,
        y_out_ref: ActBufRef,
        bufs: &BlockWeightBufs,
        taps: &BlockDebugTaps,
    ) -> Result<(), WgpuError> {
        debug_assert_eq!(
            self.cfg.modulation,
            bufs.adaln.is_some(),
            "adaln bufs presence must match cfg.modulation"
        );
        let cfg = self.cfg;
        let rows = cfg.rows() as u32;
        let dim = cfg.dim as u32;
        let hd = cfg.head_dim as u32;
        let hq = cfg.n_heads as u32;
        let hkv = cfg.n_kv_heads as u32;
        let hid = cfg.ffn_hidden as u32;
        let eps = cfg.norm_eps;
        let scale = cfg.sdpa_scale();
        let b = cfg.batch as u32;
        let s = cfg.seq as u32;
        let ad = cfg.adaln_embed_dim as u32;
        let h = hq * hd;
        let chunk_bytes = adaln_chunk_bytes(pipelines, b, dim);
        let peaks = self.phase_peaks(pipelines);
        packer.charge(peaks[0]);
        let modulated = cfg.modulation;

        // ============================ Phase 0 (attn pre + qkv) ============================
        let p1_in: Vec<BatchBuf<'wsp>> = {
            let _g = trace::scope!("attn_qkv").entered();
            let scope = packer.scope();
            let x_in = import_act_ref(scope, x_in_ref);

            let ada: Option<AdaLnChunks<'wsp>> = match (modulated, &bufs.adaln, adaln_input_ref) {
                (true, Some(w), Some(input_ref)) => {
                    let input = scope.import_copy(input_ref);
                    copy_tap(
                        scope,
                        input,
                        taps.adaln_input.as_ref(),
                        pipelines.act_bytes(b * ad),
                    )?;
                    Some(self.prepare_adaln(scope, pipelines, *w, input, taps)?)
                }
                (false, None, None) => None,
                _ => panic!("modulation/adaln_input/adaln-bufs mismatch"),
            };
            if let Some(a) = ada.as_ref() {
                copy_tap(scope, a.scale_msa, taps.scale_msa.as_ref(), chunk_bytes)?;
                copy_tap(scope, a.gate_msa, taps.gate_msa.as_ref(), chunk_bytes)?;
                copy_tap(scope, a.scale_mlp, taps.scale_mlp.as_ref(), chunk_bytes)?;
                copy_tap(scope, a.gate_mlp, taps.gate_mlp.as_ref(), chunk_bytes)?;
            }

            let t1 = alloc_act(scope, pipelines, rows, dim)?;
            let an1 = scope.import_copy(bufs.attention_norm1);
            op_rmsnorm(scope, pipelines, x_in, an1, t1, rows, dim, eps)?;
            copy_tap_act(
                scope,
                pipelines,
                t1,
                taps.attn_norm1_out.as_ref(),
                rows,
                dim,
            )?;
            let attn_in: ActBuf<'wsp> = match ada.as_ref() {
                Some(a) => {
                    let dst = alloc_act(scope, pipelines, rows, dim)?;
                    op_bcast_affine(scope, pipelines, t1, a.scale_msa, dst, rows, dim, 1.0)?;
                    dst
                }
                None => t1,
            };
            copy_tap_act(
                scope,
                pipelines,
                attn_in,
                taps.modulated_attn_in.as_ref(),
                rows,
                dim,
            )?;

            debug_assert_eq!(
                hq, hkv,
                "fused QKV currently assumes hq == hkv (Z-Image); GQA needs schema rework"
            );
            let n_qkv = 3 * h;
            let qkv_scratch = alloc_matmul_out_buf(scope, pipelines, rows * n_qkv)?;
            let dims_qkv = scope.u32x4_uniform(rows, n_qkv, dim, 0)?;
            let w_qkv = scope.import_copy(bufs.attn_qkv);
            // DIAG block-26 matmul_i8 audit: the i8 acts actually consumed by
            // matmul_i8 (a_i8/a_params, captured post-transcode inside
            // dispatch_matmul_site_diag) and the dequant_i8 outputs (b_i8,
            // b_scale, b_qsum) so the e2e_parity test can CPU-recompute one
            // output element from the actual bytes.
            Self::dispatch_matmul_site_diag(
                scope,
                pipelines,
                attn_in,
                w_qkv,
                qkv_scratch,
                dims_qkv,
                pipelines.matmul_i8_qkv.as_ref(),
                pipelines.dequant_i8_qkv.as_ref(),
                pipelines.dequant_qkv.as_ref(),
                &pipelines.matmul_qkv,
                &pipelines.matmuls.qkv,
                rows,
                n_qkv,
                dim,
                taps.qkv_attn_in_data_head,
                taps.qkv_attn_in_params_head,
                taps.qkv_b_i8_head,
                taps.qkv_b_scale_head,
                taps.qkv_b_qsum_head,
                taps.qkv_dbg_trace_head,
                // Hardcoded trace target. NOTE: must be a cell that exists in
                // THIS dispatch: m < rows (288 at 256x256), n < 3*dim. The
                // prior (323, 255) target silently matched NO workgroup.
                taps.qkv_dbg_trace_head.map(|_| (287u32, 255u32)),
            )?;
            // Telemetry: snapshot the raw matmul output (kept for parity
            // tooling continuity; identical to the qkv output stream).
            if let Some(dst) = taps.attn_qkv_f16_pre_quant {
                let dst_b = scope.import_copy(dst);
                let bytes = pipelines.act_bytes(rows * n_qkv);
                scope.copy_buffer_to_buffer(qkv_scratch, 0, dst_b, 0, bytes)?;
            }
            let qkv_fused = ActBuf::dense(qkv_scratch);
            let q = alloc_act(scope, pipelines, rows, h)?;
            let k = alloc_act(scope, pipelines, rows, h)?;
            let v = alloc_act(scope, pipelines, rows, h)?;
            op_qkv_split(scope, pipelines, qkv_fused, q, k, v, rows, h)?;
            copy_tap_act(scope, pipelines, q, taps.attn_q.as_ref(), rows, h)?;
            copy_tap_act(scope, pipelines, k, taps.attn_k.as_ref(), rows, h)?;
            copy_tap_act(scope, pipelines, v, taps.attn_v.as_ref(), rows, h)?;
            let mut carry: Vec<BatchBuf<'wsp>> = Vec::new();
            q.push_carry(&mut carry);
            k.push_carry(&mut carry);
            v.push_carry(&mut carry);
            if let Some(a) = ada {
                carry.push(a.scale_msa);
                carry.push(a.gate_msa);
                carry.push(a.scale_mlp);
                carry.push(a.gate_mlp);
            }
            carry
        };

        // ---- Advance to phase 1 (sdpa + proj + residual1) ----
        let p1_carry = packer.advance(&p1_in, peaks[1]).await?;
        let mut idx = 0usize;
        let q = pop_act(&p1_carry, &mut idx);
        let k = pop_act(&p1_carry, &mut idx);
        let v = pop_act(&p1_carry, &mut idx);
        // Ada chunks follow as flat BatchBufs.
        let gate_msa = if modulated {
            Some(p1_carry[idx + 1])
        } else {
            None
        };
        let scale_mlp_p1 = if modulated {
            Some(p1_carry[idx + 2])
        } else {
            None
        };
        let gate_mlp_p1 = if modulated {
            Some(p1_carry[idx + 3])
        } else {
            None
        };

        // ============================ Phase 1 (sdpa + proj + residual1) ============================
        let p2_in: Vec<BatchBuf<'wsp>> = {
            let scope = packer.scope();
            let x_in = import_act_ref(scope, x_in_ref);
            let freqs_in = scope.import_copy(freqs_in_ref);
            let mask_in = scope.import_copy(mask_in_ref);

            let sa = {
                let _g = trace::scope!("attn_sdpa").entered();
                let qn = alloc_act(scope, pipelines, rows * hq, hd)?;
                let kn = alloc_act(scope, pipelines, rows * hkv, hd)?;
                let nq = scope.import_copy(bufs.attn_norm_q);
                op_rmsnorm(scope, pipelines, q, nq, qn, rows * hq, hd, eps)?;
                copy_tap_act(
                    scope,
                    pipelines,
                    qn,
                    taps.attn_q_norm.as_ref(),
                    rows * hq,
                    hd,
                )?;
                let nk = scope.import_copy(bufs.attn_norm_k);
                op_rmsnorm(scope, pipelines, k, nk, kn, rows * hkv, hd, eps)?;
                copy_tap_act(
                    scope,
                    pipelines,
                    kn,
                    taps.attn_k_norm.as_ref(),
                    rows * hkv,
                    hd,
                )?;

                let qr = alloc_act(scope, pipelines, rows, hq * hd)?;
                let kr = alloc_act(scope, pipelines, rows, hkv * hd)?;
                if cfg.rope_halfrot {
                    op_rope_halfrot(scope, pipelines, qn, freqs_in, qr, rows, hq, hd)?;
                } else {
                    op_rope(scope, pipelines, qn, freqs_in, qr, rows, hq, hd)?;
                }
                copy_tap_act(
                    scope,
                    pipelines,
                    qr,
                    taps.attn_q_rope.as_ref(),
                    rows,
                    hq * hd,
                )?;
                if cfg.rope_halfrot {
                    op_rope_halfrot(scope, pipelines, kn, freqs_in, kr, rows, hkv, hd)?;
                } else {
                    op_rope(scope, pipelines, kn, freqs_in, kr, rows, hkv, hd)?;
                }
                copy_tap_act(
                    scope,
                    pipelines,
                    kr,
                    taps.attn_k_rope.as_ref(),
                    rows,
                    hkv * hd,
                )?;

                // i8 attention opt-in: quantize q/k/v ONCE, post-rope, into
                // fused paired sdpa_i8 I/O slots. Otherwise sdpa runs dense.
                let (qx, kx, vx, sa) = if pipelines.i8_sdpa() {
                    (
                        quant_for_sdpa(scope, pipelines, qr, rows, hq * hd)?,
                        quant_for_sdpa(scope, pipelines, kr, rows, hkv * hd)?,
                        quant_for_sdpa(scope, pipelines, v, rows, hq * hd)?,
                        alloc_act_sdpa_io(scope, pipelines, rows, hq * hd)?,
                    )
                } else {
                    (qr, kr, v, alloc_act(scope, pipelines, rows, hq * hd)?)
                };
                op_sdpa(
                    scope, pipelines, qx, kx, vx, mask_in, sa, b, s, s, hq, hkv, hd, scale, 1,
                )?;
                copy_tap_act(scope, pipelines, sa, taps.attn_sdpa.as_ref(), rows, hq * hd)?;
                sa
            };

            let t2 = {
                let _g = trace::scope!("attn_proj").entered();
                let k_proj = hq * hd;
                let proj_scratch = alloc_matmul_out_buf(scope, pipelines, rows * dim)?;
                let dims_proj = scope.u32x4_uniform(rows, dim, k_proj, 0)?;
                let wo = scope.import_copy(bufs.attn_to_out);
                Self::dispatch_matmul_site_diag(
                    scope,
                    pipelines,
                    sa,
                    wo,
                    proj_scratch,
                    dims_proj,
                    pipelines.matmul_i8_proj.as_ref(),
                    pipelines.dequant_i8_proj.as_ref(),
                    pipelines.dequant_proj.as_ref(),
                    &pipelines.matmul_proj,
                    &pipelines.matmuls.proj,
                    rows,
                    dim,
                    k_proj,
                    None,
                    None,
                    taps.proj_wo_b_i8_head,
                    taps.proj_wo_b_scale_head,
                    None,
                    None,
                    None,
                )?;
                if let Some(dst) = taps.attn_proj_f16_pre_quant {
                    let dst_b = scope.import_copy(dst);
                    let bytes = pipelines.act_bytes(rows * dim);
                    scope.copy_buffer_to_buffer(proj_scratch, 0, dst_b, 0, bytes)?;
                }
                // DIAG sa raw heads (paired only under i8_sdpa).
                if let Some(ss) = sa.scale {
                    if let Some(dst) = taps.proj_sa_data_head {
                        let dst_b = scope.import_copy(dst);
                        scope.copy_buffer_to_buffer(sa.data, 0, dst_b, 0, dst.len)?;
                    }
                    if let Some(dst) = taps.proj_sa_scale_head {
                        let dst_b = scope.import_copy(dst);
                        scope.copy_buffer_to_buffer(ss, 0, dst_b, 0, dst.len)?;
                    }
                }
                let proj = ActBuf::dense(proj_scratch);
                copy_tap_act(scope, pipelines, proj, taps.attn_out.as_ref(), rows, dim)?;

                let t2 = alloc_act(scope, pipelines, rows, dim)?;
                let an2 = scope.import_copy(bufs.attention_norm2);
                op_rmsnorm(scope, pipelines, proj, an2, t2, rows, dim, eps)?;
                copy_tap_act(
                    scope,
                    pipelines,
                    t2,
                    taps.attn_norm2_out.as_ref(),
                    rows,
                    dim,
                )?;
                t2
            };

            let x1 = alloc_act(scope, pipelines, rows, dim)?;
            self.residual(scope, pipelines, x_in, t2, gate_msa, x1, rows, dim)?;
            copy_tap_act(scope, pipelines, x1, taps.x_mid.as_ref(), rows, dim)?;

            let mut carry: Vec<BatchBuf<'wsp>> = Vec::new();
            x1.push_carry(&mut carry);
            if let (Some(s), Some(g)) = (scale_mlp_p1, gate_mlp_p1) {
                carry.push(s);
                carry.push(g);
            }
            carry
        };

        // ---- Advance to phase 2 (ffn1) ----
        let p2_carry = packer.advance(&p2_in, peaks[2]).await?;
        let mut idx = 0usize;
        let x1 = pop_act(&p2_carry, &mut idx);
        let scale_mlp = if modulated { Some(p2_carry[idx]) } else { None };
        let gate_mlp_p2 = if modulated {
            Some(p2_carry[idx + 1])
        } else {
            None
        };

        // ============================ Phase 2 (ffn1) ============================
        let p3_in: Vec<BatchBuf<'wsp>> = {
            let _g = trace::scope!("ffn1").entered();
            let scope = packer.scope();
            let t3 = alloc_act(scope, pipelines, rows, dim)?;
            let fn1 = scope.import_copy(bufs.ffn_norm1);
            op_rmsnorm(scope, pipelines, x1, fn1, t3, rows, dim, eps)?;
            copy_tap_act(scope, pipelines, t3, taps.ffn_norm1_out.as_ref(), rows, dim)?;
            let ffn_in: ActBuf<'wsp> = match scale_mlp {
                Some(sm) => {
                    let dst = alloc_act(scope, pipelines, rows, dim)?;
                    op_bcast_affine(scope, pipelines, t3, sm, dst, rows, dim, 1.0)?;
                    dst
                }
                None => t3,
            };
            copy_tap_act(
                scope,
                pipelines,
                ffn_in,
                taps.modulated_ffn_in.as_ref(),
                rows,
                dim,
            )?;

            let h1_scratch = alloc_matmul_out_buf(scope, pipelines, rows * hid)?;
            let h3_scratch = alloc_matmul_out_buf(scope, pipelines, rows * hid)?;
            let dims_ffn1 = scope.u32x4_uniform(rows, hid, dim, 0)?;
            let dims_ffn3 = scope.u32x4_uniform(rows, hid, dim, 0)?;
            let w1 = scope.import_copy(bufs.ffn_w1);
            let w3 = scope.import_copy(bufs.ffn_w3);
            Self::dispatch_matmul_site(
                scope,
                pipelines,
                ffn_in,
                w1,
                h1_scratch,
                dims_ffn1,
                pipelines.matmul_i8_ffn_up.as_ref(),
                pipelines.dequant_i8_ffn_up.as_ref(),
                pipelines.dequant_ffn_up.as_ref(),
                &pipelines.matmul_ffn_up,
                &pipelines.matmuls.ffn_up,
                rows,
                hid,
                dim,
            )?;
            Self::dispatch_matmul_site(
                scope,
                pipelines,
                ffn_in,
                w3,
                h3_scratch,
                dims_ffn3,
                pipelines.matmul_i8_ffn_up.as_ref(),
                pipelines.dequant_i8_ffn_up.as_ref(),
                pipelines.dequant_ffn_up.as_ref(),
                &pipelines.matmul_ffn_up,
                &pipelines.matmuls.ffn_up,
                rows,
                hid,
                dim,
            )?;
            if let Some(dst) = taps.ffn_h1_f16_pre_quant {
                let dst_b = scope.import_copy(dst);
                let bytes = pipelines.act_bytes(rows * hid);
                scope.copy_buffer_to_buffer(h1_scratch, 0, dst_b, 0, bytes)?;
            }
            if let Some(dst) = taps.ffn_h3_f16_pre_quant {
                let dst_b = scope.import_copy(dst);
                let bytes = pipelines.act_bytes(rows * hid);
                scope.copy_buffer_to_buffer(h3_scratch, 0, dst_b, 0, bytes)?;
            }
            let h1 = ActBuf::dense(h1_scratch);
            let h3 = ActBuf::dense(h3_scratch);
            let h13 = alloc_act(scope, pipelines, rows, hid)?;
            op_silu_mul(scope, pipelines, h1, h3, h13)?;
            let mut carry: Vec<BatchBuf<'wsp>> = Vec::new();
            h13.push_carry(&mut carry);
            x1.push_carry(&mut carry);
            if let Some(g) = gate_mlp_p2 {
                carry.push(g);
            }
            carry
        };

        // ---- Advance to phase 3 (ffn2 + residual2) ----
        let p3_carry = packer.advance(&p3_in, peaks[3]).await?;
        let mut idx = 0usize;
        let h13 = pop_act(&p3_carry, &mut idx);
        let x1 = pop_act(&p3_carry, &mut idx);
        let gate_mlp = if modulated { Some(p3_carry[idx]) } else { None };

        // ============================ Phase 3 (ffn2 + residual2) ============================
        {
            let _g = trace::scope!("ffn2").entered();
            let scope = packer.scope();
            let h2_scratch = alloc_matmul_out_buf(scope, pipelines, rows * dim)?;
            let dims_ffn2 = scope.u32x4_uniform(rows, dim, hid, 0)?;
            let w2 = scope.import_copy(bufs.ffn_w2);
            Self::dispatch_matmul_site(
                scope,
                pipelines,
                h13,
                w2,
                h2_scratch,
                dims_ffn2,
                pipelines.matmul_i8_ffn_down.as_ref(),
                pipelines.dequant_i8_ffn_down.as_ref(),
                pipelines.dequant_ffn_down.as_ref(),
                &pipelines.matmul_ffn_down,
                &pipelines.matmuls.ffn_down,
                rows,
                dim,
                hid,
            )?;
            if let Some(dst) = taps.ffn_h2_f16_pre_quant {
                let dst_b = scope.import_copy(dst);
                let bytes = pipelines.act_bytes(rows * dim);
                scope.copy_buffer_to_buffer(h2_scratch, 0, dst_b, 0, bytes)?;
            }
            let h2 = ActBuf::dense(h2_scratch);
            copy_tap_act(scope, pipelines, h2, taps.ffn_raw.as_ref(), rows, dim)?;
            let t4 = alloc_act(scope, pipelines, rows, dim)?;
            let fn2 = scope.import_copy(bufs.ffn_norm2);
            op_rmsnorm(scope, pipelines, h2, fn2, t4, rows, dim, eps)?;
            copy_tap_act(scope, pipelines, t4, taps.ffn_norm2_out.as_ref(), rows, dim)?;
            let y_out = import_act_ref(scope, y_out_ref);
            self.residual(scope, pipelines, x1, t4, gate_mlp, y_out, rows, dim)?;
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn residual<'wsp>(
        &self,
        scope: &BatchScope<'wsp, WgpuBackend>,
        pipelines: &BlockPipelines,
        x: ActBuf<'wsp>,
        y: ActBuf<'wsp>,
        gate: Option<BatchBuf<'wsp>>,
        out: ActBuf<'wsp>,
        rows: u32,
        dim: u32,
    ) -> Result<(), WgpuError> {
        match gate {
            Some(g) => op_bcast_fma(scope, pipelines, x, g, y, out, rows, dim),
            None => op_add(scope, pipelines, x, y, out),
        }
    }

    /// Dispatch one matmul site at the right level of the path stack:
    ///   - DP4A (`weight` is Quant + adapter has DP4A): act_quant the dense
    ///     `a.data` to (i8, params) — or consume `a`'s pair directly when it
    ///     is already paired (sdpa_i8 output) — dequant weight; matmul_i8.
    ///   - Paired `a` + Bf16 weight: matmul_i8_bf16 mixed kernel.
    ///   - Non-DP4A but Quant weight: dequant once to dense; standard matmul.
    ///   - Non-Quant weight: standard matmul direct on weight buffer.
    ///
    /// `out_scratch` receives the raw matmul output, always dense at the
    /// block act dtype (the DP4A kernels write f16 == the F16 act dtype they
    /// are gated on). Wrap it with `ActBuf::dense` to feed the next op.
    ///
    /// Associated fn (no receiver): shared by the DiT block and the Qwen3
    /// text-encoder block, which routes its 7 per-layer matmuls through the
    /// same site logic.
    #[allow(clippy::too_many_arguments)]
    /// Coopmat-aware matmul site: when `coopmat` is `Some` and the A-side is
    /// dense (not a paired-i8 sdpa output), run the tensor-core f16 path:
    /// dequant the Quant weight to f16 `[N,K]` nmajor, cast the bf16 A-side to
    /// f16, coopmat (f16 in, f32 accumulate, f16 out), cast the output back to
    /// the bf16 residual. Otherwise fall through to [`Self::dispatch_matmul_site`]
    /// (dense / dequant-once / DP4A). Keeps coopmat isolated to its opt-in
    /// callers (Wan `lin`) without touching the shared dispatch signature.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dispatch_matmul_site_coopmat<'wsp>(
        scope: &BatchScope<'wsp, WgpuBackend>,
        pipelines: &BlockPipelines,
        a: ActBuf<'wsp>,
        b_weight: BatchBuf<'wsp>,
        out_scratch: BatchBuf<'wsp>,
        dims: BatchBuf<'wsp>,
        coopmat: Option<&CoopmatStep>,
        matmul_i8: Option<&WgpuPipeline>,
        dequant_i8: Option<&DequantStep>,
        dequant_dense: Option<&DequantStep>,
        matmul_pipeline: &WgpuPipeline,
        matmul_op: &MatMulF32,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), WgpuError> {
        // Coopmat only when the dispatch spans at least one full workgroup row
        // block (`M >= WM`). A sub-WM dispatch (e.g. a short text stream, M=16 <
        // WM=32) DEVICE-LOSES on the RTX 5070 despite the kernel's ragged/robust
        // design -- the degenerate single-partial-workgroup case. The dense
        // fallback is the reference path and such tensors are tiny anyway (the
        // hot large-M image stream always clears WM).
        if let Some(cm) = coopmat
            && a.scale.is_none()
            && m >= cm.cfg.wm()
        {
            // (1) dequant Quant weight -> f16 [N,K] nmajor.
            let b_f16 = scope.alloc(pipelines.act_bytes(n * k))?;
            let dq_dims = scope.u32x4_uniform(n, k, 0, 0)?;
            scope.dequant(
                &cm.dequant_f16.pipeline,
                cm.dequant_f16.scheme,
                b_weight,
                b_f16,
                dq_dims,
                n,
                k,
            )?;
            // (2) cast the bf16 A-side to f16.
            let a_f16 = scope.alloc(pipelines.act_bytes(m * k))?;
            let to_f16 = pipelines
                .cast_to_f16
                .as_ref()
                .expect("cast_to_f16 built whenever a coopmat step exists (fast_mixed)");
            scope.dispatch_op::<Bf16ToF16>(to_f16, &[a.data], a_f16)?;
            // (3) coopmat f16 matmul -> f16 out.
            let mm_out = scope.alloc(pipelines.act_bytes(m * n))?;
            scope.coopmat(&cm.matmul, &cm.cfg, a_f16, b_f16, mm_out, dims, m, n)?;
            // (4) cast f16 output back to the bf16 residual.
            let to_bf16 = pipelines
                .cast_to_bf16
                .as_ref()
                .expect("cast_to_bf16 built whenever a coopmat step exists (fast_mixed)");
            scope.dispatch_op::<F16ToBf16>(to_bf16, &[mm_out], out_scratch)?;
            return Ok(());
        }
        Self::dispatch_matmul_site(
            scope,
            pipelines,
            a,
            b_weight,
            out_scratch,
            dims,
            matmul_i8,
            dequant_i8,
            dequant_dense,
            matmul_pipeline,
            matmul_op,
            m,
            n,
            k,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dispatch_matmul_site<'wsp>(
        scope: &BatchScope<'wsp, WgpuBackend>,
        pipelines: &BlockPipelines,
        a: ActBuf<'wsp>,
        b_weight: BatchBuf<'wsp>,
        out_scratch: BatchBuf<'wsp>,
        dims: BatchBuf<'wsp>,
        matmul_i8: Option<&WgpuPipeline>,
        dequant_i8: Option<&DequantStep>,
        dequant_dense: Option<&DequantStep>,
        matmul_pipeline: &WgpuPipeline,
        matmul_op: &MatMulF32,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), WgpuError> {
        Self::dispatch_matmul_site_diag(
            scope,
            pipelines,
            a,
            b_weight,
            out_scratch,
            dims,
            matmul_i8,
            dequant_i8,
            dequant_dense,
            matmul_pipeline,
            matmul_op,
            m,
            n,
            k,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch_matmul_site_diag<'wsp>(
        scope: &BatchScope<'wsp, WgpuBackend>,
        pipelines: &BlockPipelines,
        a: ActBuf<'wsp>,
        b_weight: BatchBuf<'wsp>,
        out_scratch: BatchBuf<'wsp>,
        dims: BatchBuf<'wsp>,
        matmul_i8: Option<&WgpuPipeline>,
        dequant_i8: Option<&DequantStep>,
        dequant_dense: Option<&DequantStep>,
        matmul_pipeline: &WgpuPipeline,
        matmul_op: &MatMulF32,
        m: u32,
        n: u32,
        k: u32,
        diag_a_i8_head: Option<BufRef>,
        diag_a_params_head: Option<BufRef>,
        diag_b_i8_head: Option<BufRef>,
        diag_b_scale_head: Option<BufRef>,
        diag_b_qsum_head: Option<BufRef>,
        diag_dbg_trace_head: Option<BufRef>,
        diag_dbg_target: Option<(u32, u32)>,
    ) -> Result<(), WgpuError> {
        // DP4A path: matmul_i8 + (paired or transcoded) input.
        if let (Some(mm_i8), Some(dq_i8), Some(aq)) =
            (matmul_i8, dequant_i8, pipelines.act_quant.as_ref())
        {
            // Dequant the quant weight into (i8, scale, qsum). `qsum[n, t] =
            // Σ_{k in block} qb[n, k]` carries the asymmetric correction-term
            // factor consumed by matmul_i8 to subtract the activation zero-
            // point bias from the DP4A main path.
            let b_i8 = scope.alloc(n as u64 * k as u64)?;
            let b_sc = scope.alloc(n as u64 * (k as u64 / 32) * 4)?;
            let b_qs = scope.alloc(n as u64 * (k as u64 / 32) * 4)?;
            let dq_dims = scope.u32x4_uniform(n, k, 0, 0)?;
            scope.dequant_i8(
                &dq_i8.pipeline,
                dq_i8.scheme,
                b_weight,
                b_i8,
                b_sc,
                b_qs,
                dq_dims,
                n,
                k,
            )?;
            if let Some(dst) = diag_b_i8_head {
                let dst_b = scope.import_copy(dst);
                let n = dst.len.min(b_i8.len());
                scope.copy_buffer_to_buffer(b_i8, 0, dst_b, 0, n)?;
            }
            if let Some(dst) = diag_b_scale_head {
                let dst_b = scope.import_copy(dst);
                let n = dst.len.min(b_sc.len());
                scope.copy_buffer_to_buffer(b_sc, 0, dst_b, 0, n)?;
            }
            if let Some(dst) = diag_b_qsum_head {
                let dst_b = scope.import_copy(dst);
                let n = dst.len.min(b_qs.len());
                scope.copy_buffer_to_buffer(b_qs, 0, dst_b, 0, n)?;
            }
            // On a bf16-act block the i8 path rides the fast_mixed casts:
            // act_quant reads vec2<f16>, so the dense bf16 A-side is cast to
            // f16 first (the A-sides routed here are normed/modulated and O(1),
            // so the f16 clamp never bites), and matmul_i8's f16 output is cast
            // back to bf16 below. An F16-act block feeds f16 straight through.
            let bf16_block = pipelines.act_dtype == ActDtype::Bf16;
            // Acquire (a_i8, a_params) — direct from `a` when it is already
            // paired (sdpa_i8 output feeding proj), else transcode via
            // act_quant from the dense `a.data` buffer.
            let (a_i8, a_p) = match a.scale {
                Some(s) => (a.data, s),
                None => {
                    let a_src = if bf16_block {
                        let a_f16 = scope.alloc(pipelines.act_bytes(m * k))?;
                        let to_f16 = pipelines
                            .cast_to_f16
                            .as_ref()
                            .expect("cast_to_f16 built for the bf16 i8 matmul path");
                        scope.dispatch_op::<Bf16ToF16>(to_f16, &[a.data], a_f16)?;
                        a_f16
                    } else {
                        a.data
                    };
                    let a_i8 = scope.alloc(m as u64 * k as u64)?;
                    let a_p = scope.alloc(m as u64 * (k as u64 / 32) * 4)?;
                    let aq_dims = scope.u32x4_uniform(m, k, 0, 0)?;
                    scope.act_quant(aq, a_src, a_i8, a_p, aq_dims, m, k)?;
                    (a_i8, a_p)
                }
            };
            // DIAG: raw byte snapshots of the i8 acts actually consumed by
            // matmul_i8 (block-26 audit). Captured here, post-transcode, so
            // the audit sees the true A-side regardless of input form.
            if let Some(dst) = diag_a_i8_head {
                let dst_b = scope.import_copy(dst);
                let n = dst.len.min(a_i8.len());
                scope.copy_buffer_to_buffer(a_i8, 0, dst_b, 0, n)?;
            }
            if let Some(dst) = diag_a_params_head {
                let dst_b = scope.import_copy(dst);
                let n = dst.len.min(a_p.len());
                scope.copy_buffer_to_buffer(a_p, 0, dst_b, 0, n)?;
            }
            // DIAG trace bindings for the new (slot 7) per-K-block dbg_out
            // and (slot 8) dbg uniform. When `diag_dbg_target` is None the
            // dbg uniform sets `enable = 0` and the kernel takes no trace
            // path. With Some, dbg_out is sized for k/32 * 8 f32s; that
            // single thread writes the trace and we copy dbg_out into the
            // diag head buffer right after the matmul dispatch.
            let dbg_target = diag_dbg_target.unwrap_or((0, 0));
            let dbg = scope.u32x4_uniform(
                dbg_target.0,
                dbg_target.1,
                diag_dbg_target.is_some() as u32,
                0,
            )?;
            // Bind the trace head DIRECTLY as slot 7 (no intermediate alloc +
            // copy: removes one suspect from the writes-don't-land chain).
            // Pre-fill with sentinel -777.0 so the readback distinguishes
            // "kernel never wrote index i" (sentinel) from "readback hit a
            // different buffer" (garbage, no sentinel anywhere).
            let dbg_out = if let Some(dst) = diag_dbg_trace_head {
                let h = scope.import_copy(dst);
                let sentinel: Vec<u8> =
                    std::iter::repeat_n((-777.0f32).to_le_bytes(), (dst.len / 4) as usize)
                        .flatten()
                        .collect();
                scope.write_bytes(h, 0, &sentinel)?;
                h
            } else {
                scope.alloc(4)?
            };
            // matmul_i8 writes vec2<f16>. On a bf16 block, land it in an f16
            // temp and cast back to the bf16 `out_scratch` the caller expects.
            let mm_out = if bf16_block {
                scope.alloc(pipelines.act_bytes(m * n))?
            } else {
                out_scratch
            };
            scope.matmul_i8(
                mm_i8,
                &pipelines.matmul_i8_cfg,
                a_i8,
                a_p,
                b_i8,
                b_sc,
                b_qs,
                mm_out,
                dims,
                dbg_out,
                dbg,
                m,
                n,
            )?;
            if bf16_block {
                let to_bf16 = pipelines
                    .cast_to_bf16
                    .as_ref()
                    .expect("cast_to_bf16 built for the bf16 i8 matmul path");
                scope.dispatch_op::<F16ToBf16>(to_bf16, &[mm_out], out_scratch)?;
            }
            return Ok(());
        }
        // Paired acts × bf16 weights mixed matmul: the proj site consuming
        // sdpa_i8's paired output when its weight stays at full bf16. The
        // kernel consumes the pair directly — no act_quant transcode, no
        // bf16-dequant scratch — and writes vec2<f16> like matmul_i8.
        if a.scale.is_some()
            && matmul_i8.is_none()
            && let Some(mm_bf16) = pipelines.matmul_i8_bf16.as_ref()
        {
            let (a_i8, a_p) = a.paired_unchecked();
            // Precompute b_sum[n, t] from the bf16 weight. Same architectural
            // pattern as dequant_i8 → b_qsum on the Quant path.
            let bsum_pipe = pipelines
                .bf16_block_sum
                .as_ref()
                .expect("bf16_block_sum pipeline must be built when matmul_i8_bf16 is built");
            let b_sum = scope.alloc(n as u64 * (k as u64 / 32) * 4)?;
            let bsum_dims = scope.u32x4_uniform(n, k, 0, 0)?;
            scope.bf16_block_sum(bsum_pipe, b_weight, b_sum, bsum_dims, n, k)?;
            scope.matmul_i8_bf16(
                mm_bf16,
                &pipelines.matmul_i8_bf16_cfg,
                a_i8,
                a_p,
                b_weight,
                b_sum,
                out_scratch,
                dims,
                m,
                n,
            )?;
            return Ok(());
        }
        // Non-DP4A, non-mixed path: the dense matmul reads either the raw
        // weight buffer (non-quant) or a pre-dequanted workspace (Quant
        // weight, F16-fallback path). A paired A-side cannot fall through
        // here — it requires one of the i8-consuming paths above.
        debug_assert!(
            a.scale.is_none(),
            "paired A-side requires the DP4A or mixed-bf16 matmul path; dispatch_matmul_site fell through"
        );
        let b_dense = match dequant_dense {
            Some(dq) => {
                let dense = scope.alloc(n as u64 * k as u64 * 2)?;
                let dq_dims = scope.u32x4_uniform(n, k, 0, 0)?;
                scope.dequant(&dq.pipeline, dq.scheme, b_weight, dense, dq_dims, n, k)?;
                dense
            }
            None => b_weight,
        };
        scope.matmul(
            matmul_pipeline,
            matmul_op,
            a.data,
            b_dense,
            dims,
            out_scratch,
            m,
            n,
        )
    }

    fn prepare_adaln<'wsp>(
        &self,
        scope: &BatchScope<'wsp, WgpuBackend>,
        pipelines: &BlockPipelines,
        w: AdaLnBufs,
        adaln_input: BatchBuf<'wsp>,
        taps: &BlockDebugTaps,
    ) -> Result<AdaLnChunks<'wsp>, WgpuError> {
        let cfg = self.cfg;
        let dim = cfg.dim as u32;
        let b = cfg.batch as u32;
        let ad = cfg.adaln_embed_dim as u32;
        let four_dim = 4 * dim;
        let chunk_bytes = adaln_chunk_bytes(pipelines, b, dim);
        let full_bytes = adaln_full_bytes(pipelines, b, dim);

        let pre = scope.alloc(full_bytes)?;
        let dims_g = scope.u32x4_uniform(b, four_dim, ad, 0)?;
        let aw = scope.import_copy(w.weight);
        scope.matmul(
            &pipelines.matmul_adaln,
            &pipelines.matmuls.adaln,
            adaln_input,
            aw,
            dims_g,
            pre,
            b,
            four_dim,
        )?;
        copy_tap(scope, pre, taps.adaln_pre.as_ref(), full_bytes)?;
        let full = scope.alloc(full_bytes)?;
        let ab = scope.import_copy(w.bias);
        let ab_u = bcast_add_uniform(scope, four_dim)?;
        scope.bcast_add::<BcastAddF32>(&pipelines.bcast_add, pre, ab, ab_u, full, b * four_dim)?;
        copy_tap(scope, full, taps.adaln_full.as_ref(), full_bytes)?;

        let scale_msa = scope.alloc(chunk_bytes)?;
        let gate_msa_pre = scope.alloc(chunk_bytes)?;
        let scale_mlp = scope.alloc(chunk_bytes)?;
        let gate_mlp_pre = scope.alloc(chunk_bytes)?;
        for (i, c) in [scale_msa, gate_msa_pre, scale_mlp, gate_mlp_pre]
            .iter()
            .enumerate()
        {
            scope.copy_buffer_to_buffer(full, i as u64 * chunk_bytes, *c, 0, chunk_bytes)?;
        }
        let gate_msa = scope.alloc(chunk_bytes)?;
        let gate_mlp = scope.alloc(chunk_bytes)?;
        scope.dispatch_op::<TanhF32>(&pipelines.tanh, &[gate_msa_pre], gate_msa)?;
        scope.dispatch_op::<TanhF32>(&pipelines.tanh, &[gate_mlp_pre], gate_mlp)?;

        Ok(AdaLnChunks {
            scale_msa,
            gate_msa,
            scale_mlp,
            gate_mlp,
        })
    }
}

struct AdaLnChunks<'wsp> {
    scale_msa: BatchBuf<'wsp>,
    gate_msa: BatchBuf<'wsp>,
    scale_mlp: BatchBuf<'wsp>,
    gate_mlp: BatchBuf<'wsp>,
}

/// Paired-or-dense activation handle threaded through the DiT block.
///
/// In F16/F32/Bf16 ops modes `scale` is `None` and `data` is the activation
/// itself. In I8 ops mode `data` holds the packed `array<u32>` of i8 values
/// and `scale` is `Some` holding the per-`(rows, dim/32)` f32 scale buffer.
/// Every transient residual-stream allocation, every model input, every
/// model output is wrapped this way so the block forward branches once on
/// dtype (inside the op-wrappers below) rather than at every call site.
#[derive(Clone, Copy)]
pub struct ActBuf<'wsp> {
    pub data: BatchBuf<'wsp>,
    pub scale: Option<BatchBuf<'wsp>>,
}

impl<'wsp> ActBuf<'wsp> {
    pub fn dense(data: BatchBuf<'wsp>) -> Self {
        Self { data, scale: None }
    }
    pub fn paired(data: BatchBuf<'wsp>, scale: BatchBuf<'wsp>) -> Self {
        Self {
            data,
            scale: Some(scale),
        }
    }
    /// Push the data and (if i8) scale BatchBufs of this ActBuf into a packer
    /// carry vec, in that order. Inverse of `pop_act`.
    fn push_carry(self, carry: &mut Vec<BatchBuf<'wsp>>) {
        carry.push(self.data);
        if let Some(s) = self.scale {
            carry.push(s);
        }
    }
    /// Unwrap into (data, scale) asserting paired mode. Panics if not paired.
    #[inline]
    fn paired_unchecked(self) -> (BatchBuf<'wsp>, BatchBuf<'wsp>) {
        (
            self.data,
            self.scale
                .expect("ActBuf: expected paired (I8) but got dense"),
        )
    }
}

/// BufRef pair mirroring `ActBuf` shape. Callers of `forward_taps_packed`
/// pass `ActBufRef::dense` under non-I8 modes and `ActBufRef::paired` under
/// I8 (paired plumbing lands when dit.rs allocates the scale companion).
#[derive(Clone, Copy)]
pub struct ActBufRef {
    pub data: BufRef,
    pub scale: Option<BufRef>,
}

impl ActBufRef {
    pub fn dense(data: BufRef) -> Self {
        Self { data, scale: None }
    }
    pub fn paired(data: BufRef, scale: BufRef) -> Self {
        Self {
            data,
            scale: Some(scale),
        }
    }
}

/// Import an `ActBufRef` into the given scope.
fn import_act_ref<'wsp>(scope: &BatchScope<'wsp, WgpuBackend>, r: ActBufRef) -> ActBuf<'wsp> {
    ActBuf::dense(scope.import_copy(r.data))
}

/// Pop one ActBuf out of a packer carry vec, advancing `idx`. Mirrors
/// `ActBuf::push_carry` (phase-crossing acts are always dense).
fn pop_act<'wsp>(carry: &[BatchBuf<'wsp>], idx: &mut usize) -> ActBuf<'wsp> {
    let data = carry[*idx];
    *idx += 1;
    ActBuf::dense(data)
}

/// Allocate a dense ActBuf sized for `rows * dim` activation elements.
pub(crate) fn alloc_act<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    pipelines: &BlockPipelines,
    rows: u32,
    dim: u32,
) -> Result<ActBuf<'wsp>, WgpuError> {
    Ok(ActBuf::dense(scope.alloc(pipelines.act_bytes(rows * dim))?))
}

/// Quantize a dense (bf16/f16) A-side to the paired i8 form `[m, k]` ONCE, so
/// several DP4A matmul sites that share the SAME A-side (the separate q/k/v
/// projections, whose input is the one modulated stream) consume it without
/// each re-running `act_quant` (+ the bf16->f16 cast) internally. The result is
/// BIT-IDENTICAL to letting every `dispatch_matmul_site` quantize on its own
/// (act_quant is deterministic), so it preserves parity exactly. ONLY valid
/// when the consuming sites take the DP4A path (Quant weight, so `act_quant`/
/// the i8 matmul are built); on a dense path pass the dense `ActBuf` instead.
/// Mirrors the dense->paired transcode inside `dispatch_matmul_site_diag`.
pub(crate) fn quantize_act_paired<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    pipelines: &BlockPipelines,
    a: BatchBuf<'wsp>,
    m: u32,
    k: u32,
) -> Result<ActBuf<'wsp>, WgpuError> {
    let aq = pipelines
        .act_quant
        .as_ref()
        .expect("quantize_act_paired called off the i8 path (no act_quant pipeline)");
    // bf16 block: act_quant reads vec2<f16>, so cast the bf16 A-side to f16
    // first (these A-sides are normed/modulated and O(1), so the clamp never
    // bites). An f16-act block feeds straight through.
    let a_src = if pipelines.act_dtype == ActDtype::Bf16 {
        let a_f16 = scope.alloc(pipelines.act_bytes(m * k))?;
        let to_f16 = pipelines
            .cast_to_f16
            .as_ref()
            .expect("cast_to_f16 built on the bf16 i8 path");
        scope.dispatch_op::<Bf16ToF16>(to_f16, &[a], a_f16)?;
        a_f16
    } else {
        a
    };
    let a_i8 = scope.alloc(m as u64 * k as u64)?;
    let a_p = scope.alloc(m as u64 * (k as u64 / 32) * 4)?;
    let aq_dims = scope.u32x4_uniform(m, k, 0, 0)?;
    scope.act_quant(aq, a_src, a_i8, a_p, aq_dims, m, k)?;
    Ok(ActBuf::paired(a_i8, a_p))
}

/// Allocate an ActBuf for an sdpa_i8 input/output slot. The data half is
/// packed i8 (`rows * dim` bytes); data and scale halves are co-located in
/// one underlying buffer via `alloc_pair` (data at offset 0, scale right
/// after) so `sdpa_i8` can bind both halves through one storage binding per
/// role (it derives the scale word-offset in-kernel from B/S/H/D).
fn alloc_act_sdpa_io<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    _pipelines: &BlockPipelines,
    rows: u32,
    dim: u32,
) -> Result<ActBuf<'wsp>, WgpuError> {
    let (_fused, data, scale) = scope.alloc_pair(
        rows as u64 * dim as u64,
        BlockPipelines::i8_scale_bytes(rows, dim),
    )?;
    Ok(ActBuf {
        data,
        scale: Some(scale),
    })
}

/// Quantize a dense act into a fused paired sdpa_i8 I/O slot via `act_quant`.
/// Only called when `pipelines.i8_sdpa()`. On a bf16-act block the source is
/// cast to f16 first (act_quant reads vec2<f16>); valid ONLY for normed/roped
/// O(1) Q/K/V, the same contract as `fast_sdpa`.
pub(crate) fn quant_for_sdpa<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    pipelines: &BlockPipelines,
    src: ActBuf<'wsp>,
    rows: u32,
    dim: u32,
) -> Result<ActBuf<'wsp>, WgpuError> {
    let dst = alloc_act_sdpa_io(scope, pipelines, rows, dim)?;
    let (dd, ds) = dst.paired_unchecked();
    let aq = pipelines
        .act_quant
        .as_ref()
        .expect("act_quant pipeline must be built when sdpa_i8 is");
    let src_f16 = if pipelines.act_dtype == ActDtype::Bf16 {
        let f16 = scope.alloc(pipelines.act_bytes(rows * dim))?;
        let to_f16 = pipelines
            .cast_to_f16
            .as_ref()
            .expect("cast_to_f16 built on the bf16 i8_sdpa path");
        scope.dispatch_op::<Bf16ToF16>(to_f16, &[src.data], f16)?;
        f16
    } else {
        src.data
    };
    let u = scope.u32x4_uniform(rows, dim, 0, 0)?;
    scope.act_quant(aq, src_f16, dd, ds, u, rows, dim)?;
    Ok(dst)
}

// ============================================================================
// Per-op ActBuf wrappers. All elementwise/norm/rope/split ops run dense at
// the block act dtype; only `op_sdpa` branches (i8 attention opt-in).
// ============================================================================

#[allow(clippy::too_many_arguments)]
pub(crate) fn op_rmsnorm<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    pipelines: &BlockPipelines,
    src: ActBuf<'wsp>,
    w: BatchBuf<'wsp>,
    dst: ActBuf<'wsp>,
    rows: u32,
    dim: u32,
    eps: f32,
) -> Result<(), WgpuError> {
    let u = rmsnorm_uniform(scope, rows, dim, eps)?;
    scope.rmsnorm::<RmsNormF32>(&pipelines.rmsnorm, src.data, w, u, dst.data, rows)
}

pub(crate) fn op_silu_mul<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    pipelines: &BlockPipelines,
    a: ActBuf<'wsp>,
    b: ActBuf<'wsp>,
    dst: ActBuf<'wsp>,
) -> Result<(), WgpuError> {
    scope.dispatch_op::<SiluMulF32>(&pipelines.silu_mul, &[a.data, b.data], dst.data)
}

pub(crate) fn op_add<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    pipelines: &BlockPipelines,
    a: ActBuf<'wsp>,
    b: ActBuf<'wsp>,
    dst: ActBuf<'wsp>,
) -> Result<(), WgpuError> {
    scope.dispatch_op::<AddF32>(&pipelines.add, &[a.data, b.data], dst.data)
}

#[allow(clippy::too_many_arguments)]
fn op_bcast_affine<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    pipelines: &BlockPipelines,
    x: ActBuf<'wsp>,
    s: BatchBuf<'wsp>,
    dst: ActBuf<'wsp>,
    rows: u32,
    dim: u32,
    bias: f32,
) -> Result<(), WgpuError> {
    let u = bcast_affine_uniform(scope, dim, bias)?;
    scope.bcast_affine::<BcastAffineF32>(
        &pipelines.bcast_affine,
        x.data,
        s,
        u,
        dst.data,
        rows * dim,
    )
}

#[allow(clippy::too_many_arguments)]
fn op_bcast_fma<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    pipelines: &BlockPipelines,
    x: ActBuf<'wsp>,
    s: BatchBuf<'wsp>,
    y: ActBuf<'wsp>,
    dst: ActBuf<'wsp>,
    rows: u32,
    dim: u32,
) -> Result<(), WgpuError> {
    {
        let u = bcast_fma_uniform(scope, dim)?;
        scope.bcast_fma::<BcastFmaF32>(
            &pipelines.bcast_fma,
            x.data,
            s,
            y.data,
            u,
            dst.data,
            rows * dim,
        )
    }
}

fn op_qkv_split<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    pipelines: &BlockPipelines,
    src: ActBuf<'wsp>,
    q: ActBuf<'wsp>,
    k: ActBuf<'wsp>,
    v: ActBuf<'wsp>,
    rows: u32,
    h: u32,
) -> Result<(), WgpuError> {
    let n_words = match pipelines.act_dtype {
        ActDtype::F32 => rows * h,
        ActDtype::Bf16 | ActDtype::F16 => rows * (h / 2),
        ActDtype::I8 => unreachable!("I8 is never a block act_dtype"),
    };
    let u = qkv_split_uniform(scope, rows, h)?;
    scope.qkv_split::<QkvSplitF32>(
        &pipelines.qkv_split,
        src.data,
        q.data,
        k.data,
        v.data,
        u,
        n_words,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn op_rope<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    pipelines: &BlockPipelines,
    src: ActBuf<'wsp>,
    freqs: BatchBuf<'wsp>,
    dst: ActBuf<'wsp>,
    rows: u32,
    heads: u32,
    head_dim: u32,
) -> Result<(), WgpuError> {
    let pairs = head_dim / 2;
    let u = scope.u32x4_uniform(rows, heads, pairs, 0)?;
    scope.rope::<RopeF32>(
        &pipelines.rope,
        src.data,
        freqs,
        u,
        dst.data,
        rows,
        heads,
        pairs,
    )
}

/// Half-rotation rope (HF Qwen3 `(k, k+D/2)` pairing) at the block act
/// dtype. Mirrors `op_rope` but drives the `rope_halfrot` pipeline.
#[allow(clippy::too_many_arguments)]
pub(crate) fn op_rope_halfrot<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    pipelines: &BlockPipelines,
    src: ActBuf<'wsp>,
    freqs: BatchBuf<'wsp>,
    dst: ActBuf<'wsp>,
    rows: u32,
    heads: u32,
    head_dim: u32,
) -> Result<(), WgpuError> {
    let pairs = head_dim / 2;
    let u = scope.u32x4_uniform(rows, heads, pairs, 0)?;
    scope.rope::<RopeF32HalfRot>(
        &pipelines.rope_halfrot,
        src.data,
        freqs,
        u,
        dst.data,
        rows,
        heads,
        pairs,
    )
}

/// SDPA with the bf16 residual path. The default for every site; see
/// [`op_sdpa_f16`] for the self-attention mixed-precision variant.
#[allow(clippy::too_many_arguments)]
pub(crate) fn op_sdpa<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    pipelines: &BlockPipelines,
    q: ActBuf<'wsp>,
    k: ActBuf<'wsp>,
    v: ActBuf<'wsp>,
    mask: BatchBuf<'wsp>,
    dst: ActBuf<'wsp>,
    b: u32,
    s_q: u32,
    s_k: u32,
    h_q: u32,
    h_kv: u32,
    head_dim: u32,
    scale: f32,
    has_mask: u32,
) -> Result<(), WgpuError> {
    op_sdpa_impl(
        scope, pipelines, q, k, v, mask, dst, b, s_q, s_k, h_q, h_kv, head_dim, scale, has_mask,
        false, 0, 0, 0,
    )
}

/// Decode-optimized SDPA: dispatches `SdpaDecode` (one workgroup per query row,
/// the workgroup cooperates over the KV length via per-tile barriers). For
/// single-token decode (`s_q == 1`) against a long KV cache this replaces
/// `SdpaF32`'s one-thread-per-head serial softmax with a full workgroup, ~1-2
/// orders faster. Requires the `sdpa_decode` pipeline (`decode_sdpa` config);
/// falls back to [`op_sdpa`] when it was not built. Bit-equivalent to `op_sdpa`
/// (identical f32 online-softmax math). Whole-tensor single dispatch: at `s_q`
/// small the per-row cost is `~O(s_k)` on a full workgroup, well under the TDR.
#[allow(clippy::too_many_arguments)]
pub(crate) fn op_sdpa_decode<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    pipelines: &BlockPipelines,
    q: ActBuf<'wsp>,
    k: ActBuf<'wsp>,
    v: ActBuf<'wsp>,
    mask: BatchBuf<'wsp>,
    dst: ActBuf<'wsp>,
    b: u32,
    s_q: u32,
    s_k: u32,
    h_q: u32,
    h_kv: u32,
    head_dim: u32,
    scale: f32,
    has_mask: u32,
) -> Result<(), WgpuError> {
    let Some(dec) = pipelines.sdpa_decode.as_ref() else {
        return op_sdpa(
            scope, pipelines, q, k, v, mask, dst, b, s_q, s_k, h_q, h_kv, head_dim, scale, has_mask,
        );
    };
    let q = q.data;
    let k = k.data;
    let v = v.data;
    let dst = dst.data;
    let u = sdpa_uniform(scope, b, h_q, h_kv, s_q, s_k, head_dim, scale, has_mask)?;
    let m_c = scope.subview(&mask, 0, mask.len());
    scope.sdpa::<SdpaDecode>(dec, q, k, v, m_c, u, dst, b, s_q, h_q)
}

/// SDPA that opts into the mixed-precision f16 path when the block compiled it
/// (`BlockWgslConfigs::fast_sdpa` on a bf16-act block). Use ONLY where Q/K/V are
/// post-rmsnorm/rope and O(1) (DiT self-attention). Falls back to [`op_sdpa`]'s
/// behavior when the f16 pipelines were not built.
#[allow(clippy::too_many_arguments)]
pub(crate) fn op_sdpa_f16<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    pipelines: &BlockPipelines,
    q: ActBuf<'wsp>,
    k: ActBuf<'wsp>,
    v: ActBuf<'wsp>,
    mask: BatchBuf<'wsp>,
    dst: ActBuf<'wsp>,
    b: u32,
    s_q: u32,
    s_k: u32,
    h_q: u32,
    h_kv: u32,
    head_dim: u32,
    scale: f32,
    has_mask: u32,
) -> Result<(), WgpuError> {
    op_sdpa_impl(
        scope, pipelines, q, k, v, mask, dst, b, s_q, s_k, h_q, h_kv, head_dim, scale, has_mask,
        true, 0, 0, 0,
    )
}

/// Mixed-precision f16 SDPA restricted to a temporal sliding window: each query
/// attends only to keys within `±window` latent frames, where `period` is the
/// token count per latent frame (frame-major `(f, h, w)` layout). `txt_start` is
/// the first joint-token index that is text (always-in-window; image queries
/// attend all text, text queries attend everything) -- pure-video callers pass
/// `s_k`. Falls back to the full [`op_sdpa_f16`] behavior when `window == 0` or
/// the windowed pipeline was not built. Use ONLY for video DiT self-attention
/// (Q/K/V normed/roped).
#[allow(clippy::too_many_arguments)]
pub(crate) fn op_sdpa_f16_win<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    pipelines: &BlockPipelines,
    q: ActBuf<'wsp>,
    k: ActBuf<'wsp>,
    v: ActBuf<'wsp>,
    mask: BatchBuf<'wsp>,
    dst: ActBuf<'wsp>,
    b: u32,
    s_q: u32,
    s_k: u32,
    h_q: u32,
    h_kv: u32,
    head_dim: u32,
    scale: f32,
    has_mask: u32,
    period: u32,
    window: u32,
    txt_start: u32,
) -> Result<(), WgpuError> {
    op_sdpa_impl(
        scope, pipelines, q, k, v, mask, dst, b, s_q, s_k, h_q, h_kv, head_dim, scale, has_mask,
        true, period, window, txt_start,
    )
}

#[allow(clippy::too_many_arguments)]
fn op_sdpa_impl<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    pipelines: &BlockPipelines,
    q: ActBuf<'wsp>,
    k: ActBuf<'wsp>,
    v: ActBuf<'wsp>,
    mask: BatchBuf<'wsp>,
    dst: ActBuf<'wsp>,
    b: u32,
    s_q: u32,
    s_k: u32,
    h_q: u32,
    h_kv: u32,
    head_dim: u32,
    scale: f32,
    has_mask: u32,
    allow_f16: bool,
    period: u32,
    window: u32,
    txt_start: u32,
) -> Result<(), WgpuError> {
    // The i8 paired path (opt-in): taken only when the caller actually
    // quantized Q/K/V into paired slots (a block may compile sdpa_i8 for its
    // long self-attention while keeping short cross-attn calls dense). The
    // output form follows `dst`: paired (feeds a paired-A matmul, z_image
    // proj) or dense at the act dtype (no output quantize; Wan's Q8 proj).
    if let Some(sdpa_i8) = pipelines.sdpa_i8.as_ref()
        && q.scale.is_some()
    {
        let (qd, qs) = q.paired_unchecked();
        let (kd, ks) = k.paired_unchecked();
        let (vd, vs) = v.paired_unchecked();
        let q_fused = scope.fuse_pair(qd, qs);
        let k_fused = scope.fuse_pair(kd, ks);
        let v_fused = scope.fuse_pair(vd, vs);
        let (o_buf, out_mode) = match dst.scale {
            Some(_) => {
                let (dd, ds) = dst.paired_unchecked();
                (scope.fuse_pair(dd, ds), 0u32)
            }
            None => (
                dst.data,
                match pipelines.act_dtype {
                    ActDtype::F16 => 1,
                    ActDtype::Bf16 => 2,
                    a => unreachable!("sdpa_i8 dense out unsupported at act dtype {a:?}"),
                },
            ),
        };
        // Q-row chunking under the ~2s Windows GPU watchdog, same scheme as
        // the f16/dense paths below; each heavy chunk flushes into its own
        // submit. Bit-exact: every kernel offset derives from the GLOBAL row
        // index (`u.row0 + local`) and all buffers stay bound whole.
        let max_rows = if b == 1 && has_mask == 0 {
            sdpa_chunk_rows_i8(s_k)
        } else {
            s_q.max(1)
        };
        let chunked = max_rows < s_q;
        let mut r0 = 0u32;
        while r0 < s_q {
            let rows = (s_q - r0).min(max_rows);
            let u = sdpa_uniform_i8(
                scope, b, h_q, h_kv, s_q, s_k, head_dim, scale, has_mask, r0, out_mode,
            )?;
            scope.sdpa_i8(
                sdpa_i8, q_fused, k_fused, v_fused, mask, o_buf, u, b, rows, h_q, head_dim,
            )?;
            r0 += rows;
            if chunked {
                scope.flush()?;
            }
        }
        return Ok(());
    }

    // Mixed-precision f16 SDPA (opt-in via fast_sdpa; bf16-act self-attention
    // only). The residual stream stays bf16 (outlier channels exceed f16's
    // +-65504), but these Q/K/V are normed/roped and O(1), so casting them to
    // f16 for the subgroup kernel is loss-free; the output casts straight back
    // to bf16. K/V cast whole (the kernel reads them entirely per query tile);
    // Q and output cast per chunk so the f16 scratch stays bounded at long
    // clips. Same TDR row-chunking as the SdpaF32 path -- a single 32k-row f16
    // dispatch still trips the 2s watchdog; the f16 kernel is ~2x faster, so a
    // chunk carries ~2x the rows under the same budget. b>1 / masked calls take
    // the whole-tensor path.
    if allow_f16
        && let (Some(sdpa_sg), Some(to_f16), Some(to_bf16)) = (
            pipelines
                .sdpa_sg
                .as_ref()
                .filter(|_| head_dim.is_multiple_of(32) && head_dim <= 128),
            pipelines.cast_to_f16.as_ref(),
            pipelines.cast_to_bf16.as_ref(),
        )
    {
        let q = q.data;
        let k = k.data;
        let v = v.data;
        let dst = dst.data;
        let kv_bytes = pipelines.act_bytes(s_k * h_kv * head_dim);
        let k_f16 = scope.alloc(kv_bytes)?;
        let v_f16 = scope.alloc(kv_bytes)?;
        scope.dispatch_op::<Bf16ToF16>(to_f16, &[scope.subview(&k, 0, kv_bytes)], k_f16)?;
        scope.dispatch_op::<Bf16ToF16>(to_f16, &[scope.subview(&v, 0, kv_bytes)], v_f16)?;
        let row_elems = h_q * head_dim;
        let max_rows = if b == 1 && has_mask == 0 {
            sdpa_chunk_rows_f16(s_k)
        } else {
            s_q.max(1)
        };
        // Temporal sliding window: dispatch the windowed twin kernel when a run
        // requested it (`window > 0`) and it compiled. `row_off = r0` lets a
        // chunked Q range recover each query's GLOBAL latent frame.
        let windowed =
            window > 0 && period > 0 && b == 1 && has_mask == 0 && pipelines.sdpa_sg_win.is_some();
        let sg_pipe = if windowed {
            pipelines.sdpa_sg_win.as_ref().unwrap()
        } else {
            sdpa_sg
        };
        let chunked = max_rows < s_q;
        let mut r0 = 0u32;
        while r0 < s_q {
            let rows = (s_q - r0).min(max_rows);
            let off = pipelines.act_bytes(r0 * row_elems);
            let len = pipelines.act_bytes(rows * row_elems);
            let q_f16 = scope.alloc(len)?;
            scope.dispatch_op::<Bf16ToF16>(to_f16, &[scope.subview(&q, off, len)], q_f16)?;
            let o_f16 = scope.alloc(len)?;
            let m_c = scope.subview(&mask, 0, mask.len());
            let u = if windowed {
                sdpa_uniform_win(
                    scope, b, h_q, h_kv, rows, s_k, head_dim, scale, has_mask, period, window, r0,
                    txt_start,
                )?
            } else {
                sdpa_uniform(scope, b, h_q, h_kv, rows, s_k, head_dim, scale, has_mask)?
            };
            scope.sdpa_sg(
                sg_pipe,
                q_f16,
                k_f16,
                v_f16,
                m_c,
                u,
                o_f16,
                pipelines.sdpa_sg_cl,
                pipelines.sdpa_sg_qr,
                b,
                rows,
                h_q,
            )?;
            scope.dispatch_op::<F16ToBf16>(to_bf16, &[o_f16], scope.subview(&dst, off, len))?;
            r0 += rows;
            if chunked {
                scope.flush()?;
            }
        }
        return Ok(());
    }
    // Dense path. A single flash-attention SDPA dispatch is ~O(s_q * s_k); on a
    // large token grid (video DiT self-attention) it runs long enough to trip
    // the ~2s Windows GPU watchdog (TDR), which resets the engine mid-dispatch
    // and surfaces as "device lost" (nvlddmkm Event 153). Each BR=64 query tile
    // is an independent workgroup, so splitting the query range across several
    // dispatches is BIT-EXACT: query row `r0+sq` of a chunk bound at row offset
    // `r0` lands at the same global element (q_off is independent of s_q at
    // b==1), and K/V/keys stay whole. We chunk only the small-D `SdpaF32` path
    // (the measured offender); the `large_d` (VAE) and subgroup-f16 (LTX) paths
    // and any b>1 / masked call take the whole-tensor path unchanged.
    let q = q.data;
    let k = k.data;
    let v = v.data;
    let dst = dst.data;
    // The SdpaF32 small-D fallback is selected iff large-D (head_dim>128) and
    // the subgroup-f16 path (sg present, head_dim%32==0) both don't apply.
    let uses_small_f32 =
        head_dim <= 128 && !(pipelines.sdpa_sg.is_some() && head_dim.is_multiple_of(32));
    let max_rows = if b == 1 && has_mask == 0 && uses_small_f32 {
        sdpa_chunk_rows(s_k)
    } else {
        s_q.max(1)
    };
    let row_elems = h_q * head_dim;
    // When we split, the OS GPU watchdog (a per-submit timer) is only respected
    // if each heavy chunk lands in its OWN submit: dispatches inside one command
    // buffer are not preempted, so 4 chunks in one submit still trips the 2s TDR.
    // Flush after each chunk to break the scope's submit at chunk boundaries.
    let chunked = max_rows < s_q;
    let mut r0 = 0u32;
    while r0 < s_q {
        let rows = (s_q - r0).min(max_rows);
        let off = pipelines.act_bytes(r0 * row_elems);
        let len = pipelines.act_bytes(rows * row_elems);
        let q_c = scope.subview(&q, off, len);
        let o_c = scope.subview(&dst, off, len);
        let k_c = scope.subview(&k, 0, k.len());
        let v_c = scope.subview(&v, 0, v.len());
        let m_c = scope.subview(&mask, 0, mask.len());
        let u = sdpa_uniform(scope, b, h_q, h_kv, rows, s_k, head_dim, scale, has_mask)?;
        dispatch_sdpa_dense(
            scope, pipelines, q_c, k_c, v_c, m_c, o_c, u, b, rows, h_q, head_dim,
        )?;
        r0 += rows;
        if chunked {
            scope.flush()?;
        }
    }
    Ok(())
}

/// Queries-per-dispatch cap that keeps one `SdpaF32` flash-attention dispatch
/// under the ~2s Windows GPU watchdog. Cost is ~O(rows * s_k); bound the product
/// to ~10M, measured ~0.6s for the bf16 kernel on the 8GB RTX 5070 Laptop
/// (~3x under the 2s reset). Rounds down to a BR=64 multiple (the workgroup tile)
/// so chunk offsets stay storage-binding aligned; min one tile.
fn sdpa_chunk_rows(s_k: u32) -> u32 {
    const MAX_QK: u64 = 10_000_000;
    let rows = (MAX_QK / s_k.max(1) as u64) as u32;
    (rows / 64 * 64).max(64)
}

/// Queries-per-dispatch cap for the f16 subgroup SDPA. The kernel is ~2x faster
/// than `SdpaF32`, so a 2x-larger QK budget keeps each chunk under the 2s
/// watchdog with about half the submits. Each output row is independent, so any
/// row count is bit-exact; rounding to a 64 multiple keeps chunk offsets
/// storage-binding aligned.
fn sdpa_chunk_rows_f16(s_k: u32) -> u32 {
    const MAX_QK: u64 = 20_000_000;
    let rows = (MAX_QK / s_k.max(1) as u64) as u32;
    (rows / 64 * 64).max(64)
}

/// Queries-per-dispatch cap for the i8 SDPA. Same 20M QK budget as the f16
/// subgroup path: measured headroom there is ~10x under the 2s watchdog, so
/// even if the i8 kernel lands slower per QK pair it stays well clear. Rounds
/// to the BR=64 workgroup tile.
fn sdpa_chunk_rows_i8(s_k: u32) -> u32 {
    const MAX_QK: u64 = 20_000_000;
    let rows = (MAX_QK / s_k.max(1) as u64) as u32;
    (rows / 64 * 64).max(64)
}

/// Dense SDPA variant selection (no i8): large-D (VAE), subgroup-f16 (LTX), or
/// the `SdpaF32` small-D fallback. Shared by the whole-tensor and chunked paths.
#[allow(clippy::too_many_arguments)]
fn dispatch_sdpa_dense<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    pipelines: &BlockPipelines,
    q: BatchBuf<'wsp>,
    k: BatchBuf<'wsp>,
    v: BatchBuf<'wsp>,
    mask: BatchBuf<'wsp>,
    out: BatchBuf<'wsp>,
    u: BatchBuf<'wsp>,
    b: u32,
    s_q: u32,
    h_q: u32,
    head_dim: u32,
) -> Result<(), WgpuError> {
    if let Some(sdpa_large_d) = pipelines.sdpa_large_d.as_ref().filter(|_| head_dim > 128) {
        scope.sdpa::<SdpaF32LargeD>(sdpa_large_d, q, k, v, mask, u, out, b, s_q, h_q)
    } else if let Some(sdpa_sg) = pipelines
        .sdpa_sg
        .as_ref()
        .filter(|_| head_dim.is_multiple_of(32) && head_dim <= 128)
    {
        scope.sdpa_sg(
            sdpa_sg,
            q,
            k,
            v,
            mask,
            u,
            out,
            pipelines.sdpa_sg_cl,
            pipelines.sdpa_sg_qr,
            b,
            s_q,
            h_q,
        )
    } else {
        scope.sdpa::<SdpaF32>(&pipelines.sdpa, q, k, v, mask, u, out, b, s_q, h_q)
    }
}

/// Copy-tap an ActBuf into an `ActTapBufRef`. Dense sources copy
/// `act_bytes`; paired sources (sdpa_i8 I/O) copy the packed-i8 data plus
/// the scale companion (the tap must carry a matching scale BufRef). Cheap
/// no-op when `tap` is `None`.
fn copy_tap_act<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    pipelines: &BlockPipelines,
    src: ActBuf<'wsp>,
    tap: Option<&ActTapBufRef>,
    rows: u32,
    dim: u32,
) -> Result<(), WgpuError> {
    let Some(t) = tap else {
        return Ok(());
    };
    let dst_d = scope.import_copy(t.data);
    match src.scale {
        None => {
            scope.copy_buffer_to_buffer(src.data, 0, dst_d, 0, pipelines.act_bytes(rows * dim))?;
        }
        Some(ss) => {
            scope.copy_buffer_to_buffer(src.data, 0, dst_d, 0, rows as u64 * dim as u64)?;
            let ts = t
                .scale
                .expect("copy_tap_act: paired source needs a paired tap (sdpa_i8 slots)");
            let dst_s = scope.import_copy(ts);
            scope.copy_buffer_to_buffer(
                ss,
                0,
                dst_s,
                0,
                BlockPipelines::i8_scale_bytes(rows, dim),
            )?;
        }
    }
    Ok(())
}

/// Byte size of one AdaLN chunk (`b * dim` elements at the block act dtype).
fn adaln_chunk_bytes(pipelines: &BlockPipelines, b: u32, dim: u32) -> u64 {
    pipelines.act_bytes(b * dim)
}

/// Byte size of the AdaLN matmul + post-bias `full` tensor (`b * 4*dim`).
fn adaln_full_bytes(pipelines: &BlockPipelines, b: u32, dim: u32) -> u64 {
    pipelines.act_bytes(b * 4 * dim)
}

/// Allocate the scratch buffer that a matmul writes its raw output into
/// (always the native act dtype; the DP4A kernels write f16 == the F16 act
/// dtype they are gated on).
pub(crate) fn alloc_matmul_out_buf<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    pipelines: &BlockPipelines,
    n_elems: u32,
) -> Result<BatchBuf<'wsp>, WgpuError> {
    scope.alloc(pipelines.act_bytes(n_elems))
}

#[derive(Clone, Debug)]
pub struct BlockHandles {
    pub attention_norm1: WeightHandle,
    pub attention_norm2: WeightHandle,
    pub ffn_norm1: WeightHandle,
    pub ffn_norm2: WeightHandle,
    pub attn_qkv: WeightHandle,
    pub attn_to_out: WeightHandle,
    pub attn_norm_q: WeightHandle,
    pub attn_norm_k: WeightHandle,
    pub ffn_w1: WeightHandle,
    pub ffn_w2: WeightHandle,
    pub ffn_w3: WeightHandle,
    pub adaln: Option<AdaLnHandles>,
}

#[derive(Clone, Debug)]
pub struct AdaLnHandles {
    pub weight: WeightHandle,
    pub bias: WeightHandle,
}

pub struct BlockViews<'a> {
    pub attention_norm1: GpuView<'a>,
    pub attention_norm2: GpuView<'a>,
    pub ffn_norm1: GpuView<'a>,
    pub ffn_norm2: GpuView<'a>,
    pub attn_qkv: GpuView<'a>,
    pub attn_to_out: GpuView<'a>,
    pub attn_norm_q: GpuView<'a>,
    pub attn_norm_k: GpuView<'a>,
    pub ffn_w1: GpuView<'a>,
    pub ffn_w2: GpuView<'a>,
    pub ffn_w3: GpuView<'a>,
    pub adaln: Option<AdaLnViews<'a>>,
}

pub struct AdaLnViews<'a> {
    pub weight: GpuView<'a>,
    pub bias: GpuView<'a>,
}

impl BlockHandles {
    pub async fn acquire<'a, S: WeightSource>(
        &self,
        residency: &'a WeightResidency<S>,
        backend: &WgpuBackend,
    ) -> Result<BlockViews<'a>, ResidencyError<S::Error, WgpuError>> {
        let adaln = match &self.adaln {
            Some(a) => Some(AdaLnViews {
                weight: residency.acquire(a.weight, backend).await?,
                bias: residency.acquire(a.bias, backend).await?,
            }),
            None => None,
        };
        Ok(BlockViews {
            attention_norm1: residency.acquire(self.attention_norm1, backend).await?,
            attention_norm2: residency.acquire(self.attention_norm2, backend).await?,
            ffn_norm1: residency.acquire(self.ffn_norm1, backend).await?,
            ffn_norm2: residency.acquire(self.ffn_norm2, backend).await?,
            attn_qkv: residency.acquire(self.attn_qkv, backend).await?,
            attn_to_out: residency.acquire(self.attn_to_out, backend).await?,
            attn_norm_q: residency.acquire(self.attn_norm_q, backend).await?,
            attn_norm_k: residency.acquire(self.attn_norm_k, backend).await?,
            ffn_w1: residency.acquire(self.ffn_w1, backend).await?,
            ffn_w2: residency.acquire(self.ffn_w2, backend).await?,
            ffn_w3: residency.acquire(self.ffn_w3, backend).await?,
            adaln,
        })
    }

    /// Load every weight in this block to GPU without pinning. Drives all
    /// per-tensor uploads serially within this future, but the future as a
    /// whole runs concurrently with the previous block's GPU submit when the
    /// caller `join!`s it.
    pub async fn prefetch<S: WeightSource>(
        &self,
        residency: &WeightResidency<S>,
        backend: &WgpuBackend,
    ) -> Result<(), ResidencyError<S::Error, WgpuError>> {
        if let Some(a) = &self.adaln {
            residency.prefetch(a.weight, backend).await?;
            residency.prefetch(a.bias, backend).await?;
        }
        residency.prefetch(self.attention_norm1, backend).await?;
        residency.prefetch(self.attention_norm2, backend).await?;
        residency.prefetch(self.ffn_norm1, backend).await?;
        residency.prefetch(self.ffn_norm2, backend).await?;
        residency.prefetch(self.attn_qkv, backend).await?;
        residency.prefetch(self.attn_to_out, backend).await?;
        residency.prefetch(self.attn_norm_q, backend).await?;
        residency.prefetch(self.attn_norm_k, backend).await?;
        residency.prefetch(self.ffn_w1, backend).await?;
        residency.prefetch(self.ffn_w2, backend).await?;
        residency.prefetch(self.ffn_w3, backend).await?;
        Ok(())
    }
}

impl BlockViews<'_> {
    pub fn bufs(&self) -> BlockWeightBufs {
        BlockWeightBufs {
            attention_norm1: self.attention_norm1.buf(),
            attention_norm2: self.attention_norm2.buf(),
            ffn_norm1: self.ffn_norm1.buf(),
            ffn_norm2: self.ffn_norm2.buf(),
            attn_qkv: self.attn_qkv.buf(),
            attn_to_out: self.attn_to_out.buf(),
            attn_norm_q: self.attn_norm_q.buf(),
            attn_norm_k: self.attn_norm_k.buf(),
            ffn_w1: self.ffn_w1.buf(),
            ffn_w2: self.ffn_w2.buf(),
            ffn_w3: self.ffn_w3.buf(),
            adaln: self.adaln.as_ref().map(|a| AdaLnBufs {
                weight: a.weight.buf(),
                bias: a.bias.buf(),
            }),
        }
    }
}

pub(crate) fn copy_tap<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    src: BatchBuf<'wsp>,
    dst: Option<&BufRef>,
    bytes: u64,
) -> Result<(), WgpuError> {
    if let Some(d) = dst {
        let d_h = scope.import_copy(*d);
        scope.copy_buffer_to_buffer(src, 0, d_h, 0, bytes)?;
    }
    Ok(())
}

fn kk(kernel_id: &'static str) -> KernelKey {
    KernelKey {
        kernel_id,
        hint: String::new(),
    }
}

fn rmsnorm_uniform<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    n_rows: u32,
    d: u32,
    eps: f32,
) -> Result<BatchBuf<'wsp>, WgpuError> {
    let mut bytes = [0u8; 16];
    bytes[0..4].copy_from_slice(&n_rows.to_le_bytes());
    bytes[4..8].copy_from_slice(&d.to_le_bytes());
    bytes[8..12].copy_from_slice(&eps.to_le_bytes());
    scope.write_uniform(&bytes)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn sdpa_uniform<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    b: u32,
    h_q: u32,
    h_kv: u32,
    s_q: u32,
    s_k: u32,
    d: u32,
    scale: f32,
    has_mask: u32,
) -> Result<BatchBuf<'wsp>, WgpuError> {
    let mut bytes = [0u8; 32];
    bytes[0..4].copy_from_slice(&b.to_le_bytes());
    bytes[4..8].copy_from_slice(&h_q.to_le_bytes());
    bytes[8..12].copy_from_slice(&h_kv.to_le_bytes());
    bytes[12..16].copy_from_slice(&s_q.to_le_bytes());
    bytes[16..20].copy_from_slice(&s_k.to_le_bytes());
    bytes[20..24].copy_from_slice(&d.to_le_bytes());
    bytes[24..28].copy_from_slice(&scale.to_le_bytes());
    bytes[28..32].copy_from_slice(&has_mask.to_le_bytes());
    scope.write_uniform(&bytes)
}

/// Uniform for the windowed subgroup SDPA: the 8 base fields plus `period`
/// (tokens per latent frame), `window` (radius in frames), `row_off` (the
/// global row index of this dispatch's first query, so a chunked Q range still
/// recovers each query's global frame), and `txt_start` (first joint-token index
/// that is text -- always-in-window; pure-video callers pass `s_k`). 48 bytes
/// (12 u32) keeps the uniform 16-byte aligned.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sdpa_uniform_win<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    b: u32,
    h_q: u32,
    h_kv: u32,
    s_q: u32,
    s_k: u32,
    d: u32,
    scale: f32,
    has_mask: u32,
    period: u32,
    window: u32,
    row_off: u32,
    txt_start: u32,
) -> Result<BatchBuf<'wsp>, WgpuError> {
    let mut bytes = [0u8; 48];
    bytes[0..4].copy_from_slice(&b.to_le_bytes());
    bytes[4..8].copy_from_slice(&h_q.to_le_bytes());
    bytes[8..12].copy_from_slice(&h_kv.to_le_bytes());
    bytes[12..16].copy_from_slice(&s_q.to_le_bytes());
    bytes[16..20].copy_from_slice(&s_k.to_le_bytes());
    bytes[20..24].copy_from_slice(&d.to_le_bytes());
    bytes[24..28].copy_from_slice(&scale.to_le_bytes());
    bytes[28..32].copy_from_slice(&has_mask.to_le_bytes());
    bytes[32..36].copy_from_slice(&period.to_le_bytes());
    bytes[36..40].copy_from_slice(&window.to_le_bytes());
    bytes[40..44].copy_from_slice(&row_off.to_le_bytes());
    bytes[44..48].copy_from_slice(&txt_start.to_le_bytes());
    scope.write_uniform(&bytes)
}

/// Uniform for the i8 SDPA: the 8 base fields plus `row0` (the global row
/// index of this dispatch's first query, for Q-row chunking) and `out_mode`
/// (0 = paired i8 out, 1 = dense packed f16, 2 = dense packed bf16). 48 bytes
/// (12 u32) keeps the uniform 16-byte aligned.
#[allow(clippy::too_many_arguments)]
fn sdpa_uniform_i8<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    b: u32,
    h_q: u32,
    h_kv: u32,
    s_q: u32,
    s_k: u32,
    d: u32,
    scale: f32,
    has_mask: u32,
    row0: u32,
    out_mode: u32,
) -> Result<BatchBuf<'wsp>, WgpuError> {
    let mut bytes = [0u8; 48];
    bytes[0..4].copy_from_slice(&b.to_le_bytes());
    bytes[4..8].copy_from_slice(&h_q.to_le_bytes());
    bytes[8..12].copy_from_slice(&h_kv.to_le_bytes());
    bytes[12..16].copy_from_slice(&s_q.to_le_bytes());
    bytes[16..20].copy_from_slice(&s_k.to_le_bytes());
    bytes[20..24].copy_from_slice(&d.to_le_bytes());
    bytes[24..28].copy_from_slice(&scale.to_le_bytes());
    bytes[28..32].copy_from_slice(&has_mask.to_le_bytes());
    bytes[32..36].copy_from_slice(&row0.to_le_bytes());
    bytes[36..40].copy_from_slice(&out_mode.to_le_bytes());
    scope.write_uniform(&bytes)
}

fn bcast_affine_uniform<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    c: u32,
    bias: f32,
) -> Result<BatchBuf<'wsp>, WgpuError> {
    let mut bytes = [0u8; 16];
    bytes[0..4].copy_from_slice(&c.to_le_bytes());
    bytes[4..8].copy_from_slice(&bias.to_le_bytes());
    scope.write_uniform(&bytes)
}

fn bcast_fma_uniform<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    c: u32,
) -> Result<BatchBuf<'wsp>, WgpuError> {
    let mut bytes = [0u8; 16];
    bytes[0..4].copy_from_slice(&c.to_le_bytes());
    scope.write_uniform(&bytes)
}

fn qkv_split_uniform<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    rows: u32,
    h: u32,
) -> Result<BatchBuf<'wsp>, WgpuError> {
    let mut bytes = [0u8; 16];
    bytes[0..4].copy_from_slice(&rows.to_le_bytes());
    bytes[4..8].copy_from_slice(&h.to_le_bytes());
    scope.write_uniform(&bytes)
}

fn bcast_add_uniform<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    c: u32,
) -> Result<BatchBuf<'wsp>, WgpuError> {
    let mut bytes = [0u8; 16];
    bytes[0..4].copy_from_slice(&c.to_le_bytes());
    scope.write_uniform(&bytes)
}
