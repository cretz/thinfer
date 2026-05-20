//! Z-Image transformer block forward, both `modulation` flavors.
//!
//! Sequence per upstream `ZImageTransformerBlock.forward`
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
    BcastFmaOp, LayerNormF32, LayerNormOp, MatMulConfig, MatMulF32, MatmulOp, MulF32, Op,
    RmsNormF32, RmsNormOp, RopeF32, RopeF32HalfRot, RopeOp, ScatterPadRowsF32, ScatterPadRowsOp,
    SdpaF32, SdpaOp, SiluF32, SiluMulF32, TanhF32, WgslConfig,
};
use thinfer_core::residency::{GpuView, ResidencyError, WeightHandle, WeightResidency};
use thinfer_core::trace;
use thinfer_core::weight::WeightSource;
use thinfer_core::workspace::{BatchBuf, BatchScope};

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
    pub attn_to_q: BufRef,
    pub attn_to_k: BufRef,
    pub attn_to_v: BufRef,
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
    pub proj: MatMulF32,
    pub ffn_up: MatMulF32,
    pub ffn_down: MatMulF32,
    pub adaln: MatMulF32,
}

impl Default for BlockMatmuls {
    fn default() -> Self {
        // Per-kernel GPU timestamps swept across DEFAULT/8x8/32x64/2x4/
        // 32x128/2x8/64x64/4x4/64x128/4x8/128x64/4x4: 64x64/4x4 wins.
        // Register-blocked bigger tiles win on this iGPU until the cliff
        // at tm*tn=32 acc regs/thread (64x128/4x8 doubled ffn ms). The
        // worklog "iGPU is occupancy-bound, shrink WG" gotcha is *wrong*
        // for this kernel. Hard constraint: WG threads <= 256 (WebGPU),
        // so 128x64/4x4 is invalid (512 threads). adaln stays DEFAULT:
        // M=1 makes register blocking pointless.
        let blocked = MatMulConfig {
            bm: 64,
            bn: 64,
            bk: 16,
            tm: 4,
            tn: 4,
        };
        Self {
            qkv: MatMulF32::new(blocked),
            proj: MatMulF32::new(blocked),
            ffn_up: MatMulF32::new(blocked),
            ffn_down: MatMulF32::new(blocked),
            // tn=2 (not DEFAULT tn=1) so AdaLN output can land in packed-bf16
            // storage when `WgslConfig.act_dtype = Bf16`. Output cols are
            // 6*ADALN_EMBED_DIM = 1536 (even) so pairing is clean. M=1 keeps
            // register blocking pointless, so bm/tm stay at DEFAULT.
            adaln: MatMulF32::new(MatMulConfig {
                tn: 2,
                ..MatMulConfig::DEFAULT
            }),
        }
    }
}

pub struct BlockPipelines {
    pub matmuls: BlockMatmuls,
    pub matmul_qkv: WgpuPipeline,
    pub matmul_proj: WgpuPipeline,
    pub matmul_ffn_up: WgpuPipeline,
    pub matmul_ffn_down: WgpuPipeline,
    pub matmul_adaln: WgpuPipeline,
    pub rmsnorm: WgpuPipeline,
    pub layernorm: WgpuPipeline,
    pub rope: WgpuPipeline,
    pub rope_halfrot: WgpuPipeline,
    pub sdpa: WgpuPipeline,
    pub silu: WgpuPipeline,
    pub silu_mul: WgpuPipeline,
    pub add: WgpuPipeline,
    pub mul: WgpuPipeline,
    pub tanh: WgpuPipeline,
    pub bcast_affine: WgpuPipeline,
    pub bcast_fma: WgpuPipeline,
    pub bcast_add: WgpuPipeline,
    pub scatter_pad_rows: WgpuPipeline,
    /// On-GPU activation storage dtype for buffers compiled against this set
    /// of pipelines. Drives byte sizing of every transient alloc through the
    /// DiT block forward pass.
    pub act_dtype: ActDtype,
}

impl BlockPipelines {
    /// Bytes for `n` activation elements at this pipeline set's dtype.
    pub fn act_bytes(&self, n: u32) -> u64 {
        n as u64 * self.act_dtype.bytes_per_elem()
    }

    pub async fn compile(backend: &WgpuBackend, cfg: &WgslConfig) -> Result<Self, WgpuError> {
        let matmuls = BlockMatmuls::default();
        let mm_layout = <MatMulF32 as MatmulOp>::layout();
        let qkv_wgsl = matmuls.qkv.wgsl(cfg);
        let proj_wgsl = matmuls.proj.wgsl(cfg);
        let ffn_up_wgsl = matmuls.ffn_up.wgsl(cfg);
        let ffn_down_wgsl = matmuls.ffn_down.wgsl(cfg);
        let adaln_wgsl = matmuls.adaln.wgsl(cfg);
        Ok(Self {
            matmul_qkv: backend
                .create_pipeline(&qkv_wgsl, "main", mm_layout)
                .await?,
            matmul_proj: backend
                .create_pipeline(&proj_wgsl, "main", mm_layout)
                .await?,
            matmul_ffn_up: backend
                .create_pipeline(&ffn_up_wgsl, "main", mm_layout)
                .await?,
            matmul_ffn_down: backend
                .create_pipeline(&ffn_down_wgsl, "main", mm_layout)
                .await?,
            matmul_adaln: backend
                .create_pipeline(&adaln_wgsl, "main", mm_layout)
                .await?,
            matmuls,
            rmsnorm: backend
                .create_pipeline(
                    <RmsNormF32 as RmsNormOp>::wgsl(cfg),
                    "main",
                    <RmsNormF32 as RmsNormOp>::layout(),
                )
                .await?,
            layernorm: backend
                .create_pipeline(
                    <LayerNormF32 as LayerNormOp>::wgsl(cfg),
                    "main",
                    <LayerNormF32 as LayerNormOp>::layout(),
                )
                .await?,
            rope: backend
                .create_pipeline(
                    <RopeF32 as RopeOp>::wgsl(cfg),
                    "main",
                    <RopeF32 as RopeOp>::layout(),
                )
                .await?,
            rope_halfrot: backend
                .create_pipeline(
                    <RopeF32HalfRot as RopeOp>::wgsl(cfg),
                    "main",
                    <RopeF32HalfRot as RopeOp>::layout(),
                )
                .await?,
            sdpa: backend
                .create_pipeline(
                    <SdpaF32 as SdpaOp>::wgsl(cfg),
                    "main",
                    <SdpaF32 as SdpaOp>::layout(),
                )
                .await?,
            silu: backend
                .create_pipeline(SiluF32::wgsl(cfg), "main", SiluF32::layout())
                .await?,
            silu_mul: backend
                .create_pipeline(SiluMulF32::wgsl(cfg), "main", SiluMulF32::layout())
                .await?,
            add: backend
                .create_pipeline(AddF32::wgsl(cfg), "main", AddF32::layout())
                .await?,
            mul: backend
                .create_pipeline(MulF32::wgsl(cfg), "main", MulF32::layout())
                .await?,
            tanh: backend
                .create_pipeline(TanhF32::wgsl(cfg), "main", TanhF32::layout())
                .await?,
            bcast_affine: backend
                .create_pipeline(
                    <BcastAffineF32 as BcastAffineOp>::wgsl(cfg),
                    "main",
                    <BcastAffineF32 as BcastAffineOp>::layout(),
                )
                .await?,
            bcast_fma: backend
                .create_pipeline(
                    <BcastFmaF32 as BcastFmaOp>::wgsl(cfg),
                    "main",
                    <BcastFmaF32 as BcastFmaOp>::layout(),
                )
                .await?,
            bcast_add: backend
                .create_pipeline(
                    <BcastAddF32 as BcastAddOp>::wgsl(cfg),
                    "main",
                    <BcastAddF32 as BcastAddOp>::layout(),
                )
                .await?,
            scatter_pad_rows: backend
                .create_pipeline(
                    <ScatterPadRowsF32 as ScatterPadRowsOp>::wgsl(cfg),
                    "main",
                    <ScatterPadRowsF32 as ScatterPadRowsOp>::layout(),
                )
                .await?,
            act_dtype: cfg.act_dtype,
        })
    }
}

pub struct Block {
    pub cfg: BlockConfig,
}

#[derive(Default, Clone, Copy)]
pub struct BlockDebugTaps {
    pub scale_msa: Option<BufRef>,
    pub gate_msa: Option<BufRef>,
    pub scale_mlp: Option<BufRef>,
    pub gate_mlp: Option<BufRef>,
    pub attn_norm1_out: Option<BufRef>,
    pub modulated_attn_in: Option<BufRef>,
    pub attn_q: Option<BufRef>,
    pub attn_k: Option<BufRef>,
    pub attn_v: Option<BufRef>,
    pub attn_q_norm: Option<BufRef>,
    pub attn_k_norm: Option<BufRef>,
    pub attn_q_rope: Option<BufRef>,
    pub attn_k_rope: Option<BufRef>,
    pub attn_sdpa: Option<BufRef>,
    pub attn_out: Option<BufRef>,
    pub attn_norm2_out: Option<BufRef>,
    pub x_mid: Option<BufRef>,
    pub ffn_norm1_out: Option<BufRef>,
    pub modulated_ffn_in: Option<BufRef>,
    pub ffn_raw: Option<BufRef>,
    pub ffn_norm2_out: Option<BufRef>,
}

impl BlockDebugTaps {
    pub const EMPTY: Self = Self {
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

    pub fn kernel_keys() -> [KernelKey; 12] {
        [
            kk(<MatMulF32 as MatmulOp>::KERNEL_ID),
            kk(<RmsNormF32 as RmsNormOp>::KERNEL_ID),
            kk(<LayerNormF32 as LayerNormOp>::KERNEL_ID),
            kk(<RopeF32 as RopeOp>::KERNEL_ID),
            kk(<SdpaF32 as SdpaOp>::KERNEL_ID),
            kk(<SiluF32 as Op>::KERNEL_ID),
            kk(<AddF32 as Op>::KERNEL_ID),
            kk(<MulF32 as Op>::KERNEL_ID),
            kk(<TanhF32 as Op>::KERNEL_ID),
            kk(<BcastAffineF32 as BcastAffineOp>::KERNEL_ID),
            kk(<BcastFmaF32 as BcastFmaOp>::KERNEL_ID),
            kk(<BcastAddF32 as BcastAddOp>::KERNEL_ID),
        ]
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward<'wsp>(
        &self,
        scope: &BatchScope<'wsp, WgpuBackend>,
        pipelines: &BlockPipelines,
        x_in: BatchBuf<'wsp>,
        freqs_in: BatchBuf<'wsp>,
        mask_in: BatchBuf<'wsp>,
        adaln_input: Option<BatchBuf<'wsp>>,
        y_out: BatchBuf<'wsp>,
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
        x_in: BatchBuf<'wsp>,
        freqs_in: BatchBuf<'wsp>,
        mask_in: BatchBuf<'wsp>,
        adaln_input: Option<BatchBuf<'wsp>>,
        y_out: BatchBuf<'wsp>,
        bufs: &'wsp BlockWeightBufs,
        taps: &'wsp BlockDebugTaps,
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

        let act_bytes = pipelines.act_bytes(rows * dim);
        let q_bytes = pipelines.act_bytes(rows * hq * hd);
        let kv_bytes = pipelines.act_bytes(rows * hkv * hd);
        let hid_bytes = pipelines.act_bytes(rows * hid);

        let ada: Option<AdaLnChunks<'wsp>> = match (cfg.modulation, &bufs.adaln, adaln_input) {
            (true, Some(w), Some(input)) => Some(self.prepare_adaln(scope, pipelines, w, input)?),
            (false, None, None) => None,
            _ => panic!("modulation/adaln_input/adaln-bufs mismatch"),
        };
        let chunk_bytes = pipelines.act_bytes(b * dim);
        if let Some(a) = ada.as_ref() {
            copy_tap(scope, a.scale_msa, &taps.scale_msa, chunk_bytes)?;
            copy_tap(scope, a.gate_msa, &taps.gate_msa, chunk_bytes)?;
            copy_tap(scope, a.scale_mlp, &taps.scale_mlp, chunk_bytes)?;
            copy_tap(scope, a.gate_mlp, &taps.gate_mlp, chunk_bytes)?;
        }

        let t1 = scope.alloc(act_bytes)?;
        let u_rmsnorm_pre = rmsnorm_uniform(scope, rows, dim, eps)?;
        let an1 = scope.import(&bufs.attention_norm1);
        scope.rmsnorm::<RmsNormF32>(&pipelines.rmsnorm, x_in, an1, u_rmsnorm_pre, t1, rows)?;
        copy_tap(scope, t1, &taps.attn_norm1_out, act_bytes)?;
        let attn_in: BatchBuf<'wsp> = match ada.as_ref() {
            Some(a) => {
                let dst = scope.alloc(act_bytes)?;
                let u_ba = bcast_affine_uniform(scope, dim, 1.0)?;
                scope.bcast_affine::<BcastAffineF32>(
                    &pipelines.bcast_affine,
                    t1,
                    a.scale_msa,
                    u_ba,
                    dst,
                    rows * dim,
                )?;
                dst
            }
            None => t1,
        };
        copy_tap(scope, attn_in, &taps.modulated_attn_in, act_bytes)?;

        let (q, k, v) = {
            let _g = trace::scope!("attn_qkv").entered();
            let q = scope.alloc(q_bytes)?;
            let k = scope.alloc(kv_bytes)?;
            let v = scope.alloc(kv_bytes)?;
            let dims_q = scope.u32x4_uniform(rows, hq * hd, dim, 0)?;
            let wq = scope.import(&bufs.attn_to_q);
            scope.matmul(
                &pipelines.matmul_qkv,
                &pipelines.matmuls.qkv,
                attn_in,
                wq,
                dims_q,
                q,
                rows,
                hq * hd,
            )?;
            copy_tap(scope, q, &taps.attn_q, q_bytes)?;
            let dims_k = scope.u32x4_uniform(rows, hkv * hd, dim, 0)?;
            let wk = scope.import(&bufs.attn_to_k);
            scope.matmul(
                &pipelines.matmul_qkv,
                &pipelines.matmuls.qkv,
                attn_in,
                wk,
                dims_k,
                k,
                rows,
                hkv * hd,
            )?;
            copy_tap(scope, k, &taps.attn_k, kv_bytes)?;
            let dims_v = scope.u32x4_uniform(rows, hkv * hd, dim, 0)?;
            let wv = scope.import(&bufs.attn_to_v);
            scope.matmul(
                &pipelines.matmul_qkv,
                &pipelines.matmuls.qkv,
                attn_in,
                wv,
                dims_v,
                v,
                rows,
                hkv * hd,
            )?;
            copy_tap(scope, v, &taps.attn_v, kv_bytes)?;
            (q, k, v)
        };

        let sa = {
            let _g = trace::scope!("attn_sdpa").entered();
            let qn = scope.alloc(q_bytes)?;
            let kn = scope.alloc(kv_bytes)?;
            let u_rms_q = rmsnorm_uniform(scope, rows * hq, hd, eps)?;
            let nq = scope.import(&bufs.attn_norm_q);
            scope.rmsnorm::<RmsNormF32>(&pipelines.rmsnorm, q, nq, u_rms_q, qn, rows * hq)?;
            copy_tap(scope, qn, &taps.attn_q_norm, q_bytes)?;
            let u_rms_k = rmsnorm_uniform(scope, rows * hkv, hd, eps)?;
            let nk = scope.import(&bufs.attn_norm_k);
            scope.rmsnorm::<RmsNormF32>(&pipelines.rmsnorm, k, nk, u_rms_k, kn, rows * hkv)?;
            copy_tap(scope, kn, &taps.attn_k_norm, kv_bytes)?;

            let qr = scope.alloc(q_bytes)?;
            let kr = scope.alloc(kv_bytes)?;
            let pairs = hd / 2;
            let u_rope_q = scope.u32x4_uniform(rows, hq, pairs, 0)?;
            scope.rope::<RopeF32>(&pipelines.rope, qn, freqs_in, u_rope_q, qr, rows, hq, pairs)?;
            copy_tap(scope, qr, &taps.attn_q_rope, q_bytes)?;
            let u_rope_k = scope.u32x4_uniform(rows, hkv, pairs, 0)?;
            scope.rope::<RopeF32>(
                &pipelines.rope,
                kn,
                freqs_in,
                u_rope_k,
                kr,
                rows,
                hkv,
                pairs,
            )?;
            copy_tap(scope, kr, &taps.attn_k_rope, kv_bytes)?;

            let sa = scope.alloc(q_bytes)?;
            let u_sdpa = sdpa_uniform(scope, b, hq, hkv, s, s, hd, scale, 1)?;
            scope.sdpa::<SdpaF32>(&pipelines.sdpa, qr, kr, v, mask_in, u_sdpa, sa, b, s, hq)?;
            copy_tap(scope, sa, &taps.attn_sdpa, q_bytes)?;
            sa
        };

        let t2 = {
            let _g = trace::scope!("attn_proj").entered();
            let proj = scope.alloc(act_bytes)?;
            let dims_proj = scope.u32x4_uniform(rows, dim, hq * hd, 0)?;
            let wo = scope.import(&bufs.attn_to_out);
            scope.matmul(
                &pipelines.matmul_proj,
                &pipelines.matmuls.proj,
                sa,
                wo,
                dims_proj,
                proj,
                rows,
                dim,
            )?;
            copy_tap(scope, proj, &taps.attn_out, act_bytes)?;

            let t2 = scope.alloc(act_bytes)?;
            let u_rms_post = rmsnorm_uniform(scope, rows, dim, eps)?;
            let an2 = scope.import(&bufs.attention_norm2);
            scope.rmsnorm::<RmsNormF32>(&pipelines.rmsnorm, proj, an2, u_rms_post, t2, rows)?;
            copy_tap(scope, t2, &taps.attn_norm2_out, act_bytes)?;
            t2
        };

        let x1 = scope.alloc(act_bytes)?;
        self.residual(
            scope,
            pipelines,
            x_in,
            t2,
            ada.as_ref().map(|a| a.gate_msa),
            x1,
            rows * dim,
            dim,
        )?;
        copy_tap(scope, x1, &taps.x_mid, act_bytes)?;

        let t4 = {
            let _g = trace::scope!("ffn").entered();
            let t3 = scope.alloc(act_bytes)?;
            let u_rms_ffn1 = rmsnorm_uniform(scope, rows, dim, eps)?;
            let fn1 = scope.import(&bufs.ffn_norm1);
            scope.rmsnorm::<RmsNormF32>(&pipelines.rmsnorm, x1, fn1, u_rms_ffn1, t3, rows)?;
            copy_tap(scope, t3, &taps.ffn_norm1_out, act_bytes)?;
            let ffn_in: BatchBuf<'wsp> = match ada.as_ref() {
                Some(a) => {
                    let dst = scope.alloc(act_bytes)?;
                    let u_ba = bcast_affine_uniform(scope, dim, 1.0)?;
                    scope.bcast_affine::<BcastAffineF32>(
                        &pipelines.bcast_affine,
                        t3,
                        a.scale_mlp,
                        u_ba,
                        dst,
                        rows * dim,
                    )?;
                    dst
                }
                None => t3,
            };
            copy_tap(scope, ffn_in, &taps.modulated_ffn_in, act_bytes)?;

            let h1 = scope.alloc(hid_bytes)?;
            let h3 = scope.alloc(hid_bytes)?;
            let dims_ffn1 = scope.u32x4_uniform(rows, hid, dim, 0)?;
            let w1 = scope.import(&bufs.ffn_w1);
            scope.matmul(
                &pipelines.matmul_ffn_up,
                &pipelines.matmuls.ffn_up,
                ffn_in,
                w1,
                dims_ffn1,
                h1,
                rows,
                hid,
            )?;
            let dims_ffn3 = scope.u32x4_uniform(rows, hid, dim, 0)?;
            let w3 = scope.import(&bufs.ffn_w3);
            scope.matmul(
                &pipelines.matmul_ffn_up,
                &pipelines.matmuls.ffn_up,
                ffn_in,
                w3,
                dims_ffn3,
                h3,
                rows,
                hid,
            )?;

            let h13 = scope.alloc(hid_bytes)?;
            scope.dispatch_op::<SiluMulF32>(&pipelines.silu_mul, &[h1, h3], h13)?;

            let h2 = scope.alloc(act_bytes)?;
            let dims_ffn2 = scope.u32x4_uniform(rows, dim, hid, 0)?;
            let w2 = scope.import(&bufs.ffn_w2);
            scope.matmul(
                &pipelines.matmul_ffn_down,
                &pipelines.matmuls.ffn_down,
                h13,
                w2,
                dims_ffn2,
                h2,
                rows,
                dim,
            )?;
            copy_tap(scope, h2, &taps.ffn_raw, act_bytes)?;

            let t4 = scope.alloc(act_bytes)?;
            let u_rms_ffn2 = rmsnorm_uniform(scope, rows, dim, eps)?;
            let fn2 = scope.import(&bufs.ffn_norm2);
            scope.rmsnorm::<RmsNormF32>(&pipelines.rmsnorm, h2, fn2, u_rms_ffn2, t4, rows)?;
            copy_tap(scope, t4, &taps.ffn_norm2_out, act_bytes)?;
            t4
        };

        self.residual(
            scope,
            pipelines,
            x1,
            t4,
            ada.as_ref().map(|a| a.gate_mlp),
            y_out,
            rows * dim,
            dim,
        )?;

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn residual<'wsp>(
        &self,
        scope: &BatchScope<'wsp, WgpuBackend>,
        pipelines: &BlockPipelines,
        x: BatchBuf<'wsp>,
        y: BatchBuf<'wsp>,
        gate: Option<BatchBuf<'wsp>>,
        out: BatchBuf<'wsp>,
        n_elems: u32,
        dim: u32,
    ) -> Result<(), WgpuError> {
        match gate {
            Some(g) => {
                let u = bcast_fma_uniform(scope, dim)?;
                scope.bcast_fma::<BcastFmaF32>(&pipelines.bcast_fma, x, g, y, u, out, n_elems)
            }
            None => scope.dispatch_op::<AddF32>(&pipelines.add, &[x, y], out),
        }
    }

    fn prepare_adaln<'wsp>(
        &self,
        scope: &BatchScope<'wsp, WgpuBackend>,
        pipelines: &BlockPipelines,
        w: &'wsp AdaLnBufs,
        adaln_input: BatchBuf<'wsp>,
    ) -> Result<AdaLnChunks<'wsp>, WgpuError> {
        let cfg = self.cfg;
        let dim = cfg.dim as u32;
        let b = cfg.batch as u32;
        let ad = cfg.adaln_embed_dim as u32;
        let four_dim = 4 * dim;
        let chunk_bytes = pipelines.act_bytes(b * dim);
        let full_bytes = pipelines.act_bytes(b * four_dim);

        let pre = scope.alloc(full_bytes)?;
        let dims_g = scope.u32x4_uniform(b, four_dim, ad, 0)?;
        let aw = scope.import(&w.weight);
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
        let full = scope.alloc(full_bytes)?;
        let ab = scope.import(&w.bias);
        let ab_u = bcast_add_uniform(scope, four_dim)?;
        scope.bcast_add::<BcastAddF32>(&pipelines.bcast_add, pre, ab, ab_u, full, b * four_dim)?;

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

#[derive(Clone, Debug)]
pub struct BlockHandles {
    pub attention_norm1: WeightHandle,
    pub attention_norm2: WeightHandle,
    pub ffn_norm1: WeightHandle,
    pub ffn_norm2: WeightHandle,
    pub attn_to_q: WeightHandle,
    pub attn_to_k: WeightHandle,
    pub attn_to_v: WeightHandle,
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
    pub attn_to_q: GpuView<'a>,
    pub attn_to_k: GpuView<'a>,
    pub attn_to_v: GpuView<'a>,
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
            attn_to_q: residency.acquire(self.attn_to_q, backend).await?,
            attn_to_k: residency.acquire(self.attn_to_k, backend).await?,
            attn_to_v: residency.acquire(self.attn_to_v, backend).await?,
            attn_to_out: residency.acquire(self.attn_to_out, backend).await?,
            attn_norm_q: residency.acquire(self.attn_norm_q, backend).await?,
            attn_norm_k: residency.acquire(self.attn_norm_k, backend).await?,
            ffn_w1: residency.acquire(self.ffn_w1, backend).await?,
            ffn_w2: residency.acquire(self.ffn_w2, backend).await?,
            ffn_w3: residency.acquire(self.ffn_w3, backend).await?,
            adaln,
        })
    }
}

impl BlockViews<'_> {
    pub fn bufs(&self) -> BlockWeightBufs {
        BlockWeightBufs {
            attention_norm1: self.attention_norm1.buf(),
            attention_norm2: self.attention_norm2.buf(),
            ffn_norm1: self.ffn_norm1.buf(),
            ffn_norm2: self.ffn_norm2.buf(),
            attn_to_q: self.attn_to_q.buf(),
            attn_to_k: self.attn_to_k.buf(),
            attn_to_v: self.attn_to_v.buf(),
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

fn copy_tap<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    src: BatchBuf<'wsp>,
    dst: &'wsp Option<BufRef>,
    bytes: u64,
) -> Result<(), WgpuError> {
    if let Some(d) = dst.as_ref() {
        let d_h = scope.import(d);
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
fn sdpa_uniform<'wsp>(
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

fn bcast_add_uniform<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    c: u32,
) -> Result<BatchBuf<'wsp>, WgpuError> {
    let mut bytes = [0u8; 16];
    bytes[0..4].copy_from_slice(&c.to_le_bytes());
    scope.write_uniform(&bytes)
}
