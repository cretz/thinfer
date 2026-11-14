//! RelayVSR "Flash" decoder (ComfyUI `RelayFlash`, the VAE slot of RelayVSR):
//! a 10-block, 512-wide per-frame transformer that turns the keyframe latents
//! on either side of a frame plus that frame's LR pixels into the upscaled
//! frame `bicubic(lr, up) + residual`.
//!
//! Per keyframe latent `[48, h, w]`: `condition_encoder` (1x1 conv + RMSNorm) +
//! condition type embedding, then every block's `condition` pass (48x48 window
//! self-attention, cut at the frame border) keeps that block's roped keys and
//! values. Per frame of the window between keyframes `a` and `b`: `lr_encoder`
//! (3 strided 3x3 convs + 1x1) + video type embedding, then every block's
//! `video` pass attends `[cond_a (48x48), cond_b (48x48), past frame (24x24,
//! rope t=-1), this frame (24x24)]`; the head (`spatial_head` -> SiLU ->
//! `residual_projection`) pixel-shuffles by 16 into the residual.
//!
//! Runs as an f32-act island (bf16 weights): the model is ~32M parameters and
//! its cost is the window attention, not the linears.

use thinfer_core::backend::{Backend, BufRef, ElementwiseOp, WgpuBackend, WgpuError, WgpuPipeline};
use thinfer_core::ops::conv2d::Conv2dConfig;
use thinfer_core::ops::window_attn::{self, WindowAttnUniform};
use thinfer_core::ops::{
    ActDtype, BcastAddF32, SiluF32, Transpose12F32, Transpose12Op, WeightDtype, WgslConfig,
};
use thinfer_core::residency::{GpuView, ResidencyError, WeightHandle, WeightResidency};
use thinfer_core::weight::{WeightId, WeightSource};
use thinfer_core::workspace::{BatchBuf, BatchScope, GemmEpi, Workspace, WsBuf};

use crate::common::autoencoder_kl::{ConvPipeline, conv2d_uniform};
use crate::common::block::{
    ActBuf, BlockPipelines, alloc_act, op_add, op_rmsnorm, op_rope, op_silu_mul,
};
use crate::common::embedders::{LinearBiasBufs, bcast_add_uniform, linear_bias, linear_no_bias};
use crate::common::seq::act_upload_bytes;
use crate::wan::dit::read_into_f32;
use crate::wan::dit_block::accel_op;
use crate::wan::loader::{
    LoadError, register_conv_as_linear_bias, register_linear, register_linear_bias,
    register_passthrough,
};
use crate::wan::pipeline::{SiteOverride, block_cfgs};

pub const DIM: u32 = 512;
pub const HEADS: u32 = 8;
pub const KV_HEADS: u32 = 2;
pub const HEAD_DIM: u32 = 64;
pub const FFN: u32 = 1408;
pub const DEPTH: usize = 10;
pub const EPS: f32 = 1e-5;
pub const COND_WINDOW: (u32, u32) = (48, 48);
pub const VIDEO_WINDOW: (u32, u32) = (24, 24);
/// Residual head width: `3 * 16 * 16` (pixel-shuffled by 16).
pub const HEAD_OUT: u32 = 768;
/// Head hidden width.
pub const HEAD_HIDDEN: u32 = 1024;
/// Latent channels of the keyframe model (Wan2.2 VAE z-dim).
pub const LATENT_CH: u32 = 48;
const ROPE_AXES: [usize; 3] = [16, 24, 24];
const ROPE_THETA: f64 = 10_000.0;

struct LinearH {
    weight: WeightHandle,
    bias: Option<WeightHandle>,
}

struct ConvH {
    weight: WeightHandle,
    bias: WeightHandle,
    cout: u32,
    cin: u32,
    stride: u32,
}

struct BlockH {
    attn_norm: WeightHandle,
    query: LinearH,
    key: LinearH,
    value: LinearH,
    cond_key: LinearH,
    cond_value: LinearH,
    key_norm: WeightHandle,
    value_norm: WeightHandle,
    output: LinearH,
    ffn_norm: WeightHandle,
    gate: LinearH,
    up: LinearH,
    down: LinearH,
}

pub struct FlashHandles {
    lr_convs: [ConvH; 3],
    lr_proj: LinearH,
    lr_norm: WeightHandle,
    cond_proj: LinearH,
    cond_norm: WeightHandle,
    video_type: WeightHandle,
    cond_type: WeightHandle,
    blocks: Vec<BlockH>,
    head_hidden: LinearH,
    head_out: LinearH,
}

fn id(s: &str) -> WeightId {
    WeightId(s.to_string())
}

fn lin_b<S: WeightSource>(r: &WeightResidency<S>, stem: &str) -> Result<LinearH, LoadError> {
    let h = register_linear_bias(
        r,
        &id(&format!("{stem}.weight")),
        &id(&format!("{stem}.bias")),
    )?;
    Ok(LinearH {
        weight: h.weight,
        bias: Some(h.bias),
    })
}

fn lin_nb<S: WeightSource>(r: &WeightResidency<S>, stem: &str) -> Result<LinearH, LoadError> {
    Ok(LinearH {
        weight: register_linear(r, &id(&format!("{stem}.weight")), None)?,
        bias: None,
    })
}

/// Register the Flash decoder (`FlashSplitSource` names: the fused
/// `gate_up_proj` is served as `gate_proj` + `up_proj`). `up` = 2 or 4.
pub fn register_flash<S: WeightSource>(
    r: &WeightResidency<S>,
    up: u32,
) -> Result<FlashHandles, LoadError> {
    let conv = |i: usize, cin: u32, cout: u32, stride: u32| -> Result<ConvH, LoadError> {
        Ok(ConvH {
            weight: register_passthrough(r, &id(&format!("lr_encoder.layers.{i}.weight")))?,
            bias: register_passthrough(r, &id(&format!("lr_encoder.layers.{i}.bias")))?,
            cout,
            cin,
            stride,
        })
    };
    let ls = 16 / up;
    let lr6 = register_conv_as_linear_bias(
        r,
        &id("lr_encoder.layers.6.weight"),
        &id("lr_encoder.layers.6.bias"),
    )?;
    let cp = register_conv_as_linear_bias(
        r,
        &id("condition_encoder.projection.weight"),
        &id("condition_encoder.projection.bias"),
    )?;
    let blocks = (0..DEPTH)
        .map(|i| {
            let p = format!("backbone.blocks.{i}");
            Ok(BlockH {
                attn_norm: register_passthrough(r, &id(&format!("{p}.attention_norm.weight")))?,
                query: lin_b(r, &format!("{p}.attention.query"))?,
                key: lin_b(r, &format!("{p}.attention.key"))?,
                value: lin_b(r, &format!("{p}.attention.value"))?,
                cond_key: lin_b(r, &format!("{p}.attention.condition_key"))?,
                cond_value: lin_b(r, &format!("{p}.attention.condition_value"))?,
                key_norm: register_passthrough(r, &id(&format!("{p}.attention.key_norm.weight")))?,
                value_norm: register_passthrough(
                    r,
                    &id(&format!("{p}.attention.value_norm.weight")),
                )?,
                output: lin_b(r, &format!("{p}.attention.output"))?,
                ffn_norm: register_passthrough(r, &id(&format!("{p}.feed_forward_norm.weight")))?,
                gate: lin_nb(r, &format!("{p}.feed_forward.gate_proj"))?,
                up: lin_nb(r, &format!("{p}.feed_forward.up_proj"))?,
                down: lin_nb(r, &format!("{p}.feed_forward.down"))?,
            })
        })
        .collect::<Result<Vec<_>, LoadError>>()?;
    Ok(FlashHandles {
        lr_convs: [
            conv(0, 3, 64, 2)?,
            conv(2, 64, 128, 2)?,
            conv(4, 128, 128, ls / 4)?,
        ],
        lr_proj: LinearH {
            weight: lr6.weight,
            bias: Some(lr6.bias),
        },
        lr_norm: register_passthrough(r, &id("lr_encoder.norm.weight"))?,
        cond_proj: LinearH {
            weight: cp.weight,
            bias: Some(cp.bias),
        },
        cond_norm: register_passthrough(r, &id("condition_encoder.norm.weight"))?,
        video_type: register_passthrough(r, &id("video_type_embedding"))?,
        cond_type: register_passthrough(r, &id("condition_type_embedding"))?,
        blocks,
        head_hidden: lin_b(r, "spatial_head.0")?,
        head_out: lin_b(r, "residual_projection")?,
    })
}

/// The kernel set at the island's act dtype (f16 where the device has
/// shader-f16, so the window attention runs on the accelerator; else f32).
pub struct FlashPipelines {
    pub act: ActDtype,
    pub block: BlockPipelines,
    pub conv: ConvPipeline,
    pub window_attn: WgpuPipeline,
    pub transpose: WgpuPipeline,
}

impl FlashPipelines {
    pub async fn compile(backend: &WgpuBackend) -> Result<Self, WgpuError> {
        let act = if backend.supports_shader_f16() {
            ActDtype::F16
        } else {
            ActDtype::F32
        };
        let cfgs = block_cfgs(
            WeightDtype::Bf16,
            act,
            SiteOverride::default(),
            false,
            false,
        );
        let wcfg = WgslConfig {
            bf16_quant_writes: false,
            act_dtype: act,
            weight_dtype: WeightDtype::Bf16,
        };
        Ok(Self {
            act,
            block: BlockPipelines::compile(backend, &cfgs).await?,
            conv: ConvPipeline::compile(backend, "relay_flash_conv", &wcfg, Conv2dConfig::DEFAULT)
                .await?,
            window_attn: backend
                .create_pipeline(
                    "relay_flash_window",
                    window_attn::wgsl(act),
                    "main",
                    window_attn::layout(),
                )
                .await?,
            transpose: backend
                .create_pipeline(
                    "relay_flash_transpose",
                    <Transpose12F32 as Transpose12Op>::wgsl(&wcfg),
                    "main",
                    <Transpose12F32 as Transpose12Op>::layout(),
                )
                .await?,
        })
    }
}

/// Interleaved `(cos, sin)` rope table `[h * w, HEAD_DIM]` for ids `(t, y, x)`
/// (flux `EmbedND`, axes 16/24/24, theta 1e4; angles in f64 like the reference).
pub fn grid_rope(t: f64, h: usize, w: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(h * w * HEAD_DIM as usize);
    for y in 0..h {
        for x in 0..w {
            for (axis, &d) in ROPE_AXES.iter().enumerate() {
                let pos = [t, y as f64, x as f64][axis];
                for j in 0..d / 2 {
                    let omega = 1.0 / ROPE_THETA.powf((2 * j) as f64 / d as f64);
                    let a = pos * omega;
                    out.push(a.cos() as f32);
                    out.push(a.sin() as f32);
                }
            }
        }
    }
    out
}

/// Acquired views of every Flash weight (held for one decode).
struct Views<'a> {
    v: Vec<GpuView<'a>>,
}

impl Views<'_> {
    fn buf(&self, i: usize) -> BufRef {
        self.v[i].buf()
    }
}

fn add(handles: &mut Vec<WeightHandle>, h: WeightHandle) -> usize {
    handles.push(h);
    handles.len() - 1
}

fn add_lin(handles: &mut Vec<WeightHandle>, l: &LinearH) -> LinI {
    let w = add(handles, l.weight);
    let b = l.bias.map(|b| add(handles, b));
    LinI { w, b }
}

#[derive(Clone, Copy)]
struct LinI {
    w: usize,
    b: Option<usize>,
}

#[derive(Clone, Copy)]
struct BlockI {
    attn_norm: usize,
    query: LinI,
    key: LinI,
    value: LinI,
    cond_key: LinI,
    cond_value: LinI,
    key_norm: usize,
    value_norm: usize,
    output: LinI,
    ffn_norm: usize,
    gate: LinI,
    up: LinI,
    down: LinI,
}

struct ConvI {
    w: usize,
    b: usize,
    cout: u32,
    cin: u32,
    stride: u32,
}

struct LayoutI {
    lr_convs: Vec<ConvI>,
    lr_proj: LinI,
    lr_norm: usize,
    cond_proj: LinI,
    cond_norm: usize,
    video_type: usize,
    cond_type: usize,
    blocks: Vec<BlockI>,
    head_hidden: LinI,
    head_out: LinI,
}

/// Every keyframe's per-block condition keys (roped) and values, keyframe
/// planes stacked: `[keyframes, plane, 128]` per block.
pub struct Conditions {
    kv: Vec<(WsBuf<WgpuBackend>, WsBuf<WgpuBackend>)>,
    count: u32,
}

/// The previous step's per-block unroped `(k, v)` of the batch's frames,
/// `[b, plane, 128]` per block. Empty at a window start.
pub type History = Vec<(WsBuf<WgpuBackend>, WsBuf<WgpuBackend>)>;

/// One key plane of one batch element: `(k, v, byte offset)`.
type PlaneRef<'w> = (BatchBuf<'w>, BatchBuf<'w>, u64);

/// The Flash decoder over a batch of independent keyframe windows: batch
/// element `i` is window `i` (between keyframes `i` and `i + 1`; history
/// resets per window, so windows share nothing but the weights).
pub struct FlashDecoder<'a> {
    backend: &'a WgpuBackend,
    p: &'a FlashPipelines,
    scratch: &'a Workspace<WgpuBackend>,
    views: Views<'a>,
    l: LayoutI,
    /// Latent grid (`h, w`) = tokens per frame plane.
    pub h: u32,
    pub w: u32,
    /// Most planes one pass carries (rope tables are repeated this often).
    max_batch: u32,
    /// Blocks and the head run as deferred accelerator scopes.
    deferred: bool,
    freqs: WsBuf<WgpuBackend>,
    freqs_past: WsBuf<WgpuBackend>,
}

impl<'a> FlashDecoder<'a> {
    /// Acquire every Flash weight and build the rope tables for up to
    /// `max_batch` planes of an `h x w` latent grid.
    #[allow(clippy::too_many_arguments)]
    pub async fn new<S: WeightSource>(
        backend: &'a WgpuBackend,
        p: &'a FlashPipelines,
        scratch: &'a Workspace<WgpuBackend>,
        residency: &'a WeightResidency<S>,
        hd: &FlashHandles,
        h: u32,
        w: u32,
        max_batch: u32,
    ) -> Result<Self, ResidencyError<S::Error, WgpuError>> {
        let mut hs: Vec<WeightHandle> = Vec::new();
        let layout_convs = hd
            .lr_convs
            .iter()
            .map(|c| ConvI {
                w: add(&mut hs, c.weight),
                b: add(&mut hs, c.bias),
                cout: c.cout,
                cin: c.cin,
                stride: c.stride,
            })
            .collect();
        let lr_proj = add_lin(&mut hs, &hd.lr_proj);
        let lr_norm = add(&mut hs, hd.lr_norm);
        let cond_proj = add_lin(&mut hs, &hd.cond_proj);
        let cond_norm = add(&mut hs, hd.cond_norm);
        let video_type = add(&mut hs, hd.video_type);
        let cond_type = add(&mut hs, hd.cond_type);
        let blocks = hd
            .blocks
            .iter()
            .map(|b| BlockI {
                attn_norm: add(&mut hs, b.attn_norm),
                query: add_lin(&mut hs, &b.query),
                key: add_lin(&mut hs, &b.key),
                value: add_lin(&mut hs, &b.value),
                cond_key: add_lin(&mut hs, &b.cond_key),
                cond_value: add_lin(&mut hs, &b.cond_value),
                key_norm: add(&mut hs, b.key_norm),
                value_norm: add(&mut hs, b.value_norm),
                output: add_lin(&mut hs, &b.output),
                ffn_norm: add(&mut hs, b.ffn_norm),
                gate: add_lin(&mut hs, &b.gate),
                up: add_lin(&mut hs, &b.up),
                down: add_lin(&mut hs, &b.down),
            })
            .collect();
        let head_hidden = add_lin(&mut hs, &hd.head_hidden);
        let head_out = add_lin(&mut hs, &hd.head_out);
        let handles = hs;
        let mut v = Vec::with_capacity(handles.len());
        for hnd in handles {
            v.push(residency.acquire(hnd, backend).await?);
        }
        let upload = |x: &[f32]| -> Result<WsBuf<WgpuBackend>, WgpuError> {
            let b = act_upload_bytes(p.act, x);
            let buf = scratch.alloc(b.len() as u64)?;
            backend.write_buffer(buf.id(), 0, &b)?;
            Ok(buf)
        };
        let tiled = |t: f64| grid_rope(t, h as usize, w as usize).repeat(max_batch as usize);
        let freqs = upload(&tiled(0.0)).map_err(ResidencyError::Backend)?;
        let freqs_past = upload(&tiled(-1.0)).map_err(ResidencyError::Backend)?;
        Ok(Self {
            backend,
            p,
            scratch,
            views: Views { v },
            l: LayoutI {
                lr_convs: layout_convs,
                lr_proj,
                lr_norm,
                cond_proj,
                cond_norm,
                video_type,
                cond_type,
                blocks,
                head_hidden,
                head_out,
            },
            h,
            w,
            max_batch,
            deferred: p.act == ActDtype::F16 && backend.accelerator().is_some(),
            freqs,
            freqs_past,
        })
    }

    /// Tokens per frame plane.
    fn plane(&self) -> u32 {
        self.h * self.w
    }

    fn kv_plane_bytes(&self) -> u64 {
        self.bp().act_bytes(self.plane() * KV_HEADS * HEAD_DIM)
    }

    fn bp(&self) -> &BlockPipelines {
        &self.p.block
    }

    fn linear<'w>(
        &self,
        scope: &BatchScope<'w, WgpuBackend>,
        x: BatchBuf<'w>,
        l: LinI,
        rows: u32,
        k: u32,
        n: u32,
    ) -> Result<ActBuf<'w>, WgpuError> {
        if scope.is_deferred() {
            // The accelerator GEMM straight on the raw bf16 weight, the bias
            // in its epilogue.
            let out = alloc_act(scope, self.bp(), rows, n)?;
            let w = scope.import_copy(self.views.buf(l.w));
            let epi = GemmEpi {
                bias: l.b.map(|b| scope.import_copy(self.views.buf(b))),
                residual: None,
                gate: None,
            };
            let pipe = &self.bp().matmul_qkv;
            if scope.try_gemm_fused(pipe, x, w, out.data, rows, n, k, epi)? {
                return Ok(out);
            }
            return Err(WgpuError::AcceleratorDeclined {
                op: "relay_flash linear".into(),
            });
        }
        let bufs = LinearBiasBufs {
            weight: self.views.buf(l.w),
            bias: self.views.buf(l.b.unwrap_or(l.w)),
        };
        match l.b {
            Some(_) => {
                let out = alloc_act(scope, self.bp(), rows, n)?;
                linear_bias(scope, self.bp(), x, &bufs, rows, k, n, out.data)?;
                Ok(out)
            }
            None => Ok(ActBuf::dense(linear_no_bias(
                scope,
                self.bp(),
                x,
                &bufs,
                rows,
                k,
                n,
            )?)),
        }
    }

    fn rms<'w>(
        &self,
        scope: &BatchScope<'w, WgpuBackend>,
        x: ActBuf<'w>,
        w: usize,
        rows: u32,
        dim: u32,
    ) -> Result<ActBuf<'w>, WgpuError> {
        let out = alloc_act(scope, self.bp(), rows, dim)?;
        let wb = scope.import_copy(self.views.buf(w));
        if scope.is_deferred() {
            accel_op(scope, "relay_flash rmsnorm", |a| {
                Some(ElementwiseOp::RmsNormF16 {
                    x: a(x.data)?,
                    w: a(wb)?,
                    out: a(out.data)?,
                    rows,
                    dim,
                    eps: EPS,
                })
            })?;
            return Ok(out);
        }
        op_rmsnorm(scope, self.bp(), x, wb, out, rows, dim, EPS)?;
        Ok(out)
    }

    fn silu<'w>(
        &self,
        scope: &BatchScope<'w, WgpuBackend>,
        x: ActBuf<'w>,
        n: u32,
    ) -> Result<ActBuf<'w>, WgpuError> {
        let out = alloc_act(scope, self.bp(), n, 1)?;
        if scope.is_deferred() {
            accel_op(scope, "relay_flash silu", |a| {
                Some(ElementwiseOp::SiluF16 {
                    x: a(x.data)?,
                    out: a(out.data)?,
                    n,
                })
            })?;
            return Ok(out);
        }
        scope.dispatch_op::<SiluF32>(&self.bp().silu, &[x.data], out.data)?;
        Ok(out)
    }

    fn add_emb<'w>(
        &self,
        scope: &BatchScope<'w, WgpuBackend>,
        x: ActBuf<'w>,
        emb: usize,
        rows: u32,
    ) -> Result<ActBuf<'w>, WgpuError> {
        let out = alloc_act(scope, self.bp(), rows, DIM)?;
        let e = scope.import_copy(self.views.buf(emb));
        if scope.is_deferred() {
            accel_op(scope, "relay_flash add_emb", |a| {
                Some(ElementwiseOp::AddBiasF16 {
                    x: a(x.data)?,
                    bias: a(e)?,
                    out: a(out.data)?,
                    rows,
                    dim: DIM,
                })
            })?;
            return Ok(out);
        }
        let u = bcast_add_uniform(scope, DIM)?;
        scope.bcast_add::<BcastAddF32>(&self.bp().bcast_add, x.data, e, u, out.data, rows * DIM)?;
        Ok(out)
    }

    /// `out = x + y` over `[rows, dim]`.
    fn add<'w>(
        &self,
        scope: &BatchScope<'w, WgpuBackend>,
        x: ActBuf<'w>,
        y: ActBuf<'w>,
        rows: u32,
        dim: u32,
    ) -> Result<ActBuf<'w>, WgpuError> {
        let out = alloc_act(scope, self.bp(), rows, dim)?;
        if scope.is_deferred() {
            accel_op(scope, "relay_flash add", |a| {
                Some(ElementwiseOp::AddF16 {
                    x: a(x.data)?,
                    y: a(y.data)?,
                    out: a(out.data)?,
                    n: rows * dim,
                })
            })?;
            return Ok(out);
        }
        op_add(scope, self.bp(), x, y, out)?;
        Ok(out)
    }

    /// `bytes` from `src` (at `src_off`) to `dst` (at `dst_off`), in order
    /// with the scope's other work (deferred: on the accelerator stream).
    fn copy<'w>(
        &self,
        scope: &BatchScope<'w, WgpuBackend>,
        src: BatchBuf<'w>,
        src_off: u64,
        dst: BatchBuf<'w>,
        dst_off: u64,
        bytes: u64,
    ) -> Result<(), WgpuError> {
        if scope.is_deferred() {
            return accel_op(scope, "relay_flash copy", |a| {
                Some(ElementwiseOp::Copy {
                    src: a(src)? + src_off,
                    dst: a(dst)? + dst_off,
                    bytes,
                })
            });
        }
        scope.copy_buffer_to_buffer(src, src_off, dst, dst_off, bytes)
    }

    /// Rope `b` planes of `x` (the tables repeat per plane).
    fn rope<'w>(
        &self,
        scope: &BatchScope<'w, WgpuBackend>,
        x: ActBuf<'w>,
        freqs: &WsBuf<WgpuBackend>,
        heads: u32,
        b: u32,
    ) -> Result<ActBuf<'w>, WgpuError> {
        debug_assert!(b <= self.max_batch);
        let rows = b * self.plane();
        let out = alloc_act(scope, self.bp(), rows, heads * HEAD_DIM)?;
        let f = scope.import_copy(freqs.as_buf_ref());
        if scope.is_deferred() {
            accel_op(scope, "relay_flash rope", |a| {
                Some(ElementwiseOp::RopeF16 {
                    x: a(x.data)?,
                    freqs: a(f)?,
                    out: a(out.data)?,
                    rows,
                    heads,
                    pairs: HEAD_DIM / 2,
                })
            })?;
            return Ok(out);
        }
        op_rope(scope, self.bp(), x, f, out, rows, heads, HEAD_DIM)?;
        Ok(out)
    }

    /// Projected `(q roped, k normed, v normed)` of `b` planes; `k` unroped.
    fn project<'w>(
        &self,
        scope: &BatchScope<'w, WgpuBackend>,
        xn: ActBuf<'w>,
        blk: &BlockI,
        cond: bool,
        b: u32,
    ) -> Result<(ActBuf<'w>, ActBuf<'w>, ActBuf<'w>), WgpuError> {
        let rows = b * self.plane();
        let kvw = KV_HEADS * HEAD_DIM;
        let q = self.linear(scope, xn.data, blk.query, rows, DIM, DIM)?;
        let q = self.rope(scope, q, &self.freqs, HEADS, b)?;
        let (kl, vl) = if cond {
            (blk.cond_key, blk.cond_value)
        } else {
            (blk.key, blk.value)
        };
        let k = self.linear(scope, xn.data, kl, rows, DIM, kvw)?;
        let v = self.linear(scope, xn.data, vl, rows, DIM, kvw)?;
        let k = self.rms(scope, k, blk.key_norm, rows * KV_HEADS, HEAD_DIM)?;
        let v = self.rms(scope, v, blk.value_norm, rows * KV_HEADS, HEAD_DIM)?;
        Ok((q, k, v))
    }

    /// Window attention of the `b` query planes in `q`: batch element `i`
    /// attends its key planes `planes[i]` (each `[plane, 128]` at a byte
    /// offset), copied into one `[b, tk, plane, 128]` stack.
    fn attend<'w>(
        &self,
        scope: &BatchScope<'w, WgpuBackend>,
        q: ActBuf<'w>,
        planes: &[Vec<PlaneRef<'w>>],
        windows: &[(u32, u32)],
    ) -> Result<ActBuf<'w>, WgpuError> {
        let b = planes.len() as u32;
        let plane_bytes = self.kv_plane_bytes();
        let tk = planes[0].len() as u32;
        debug_assert!(planes.iter().all(|p| p.len() == tk as usize));
        let k = scope.alloc(plane_bytes * (b * tk) as u64)?;
        let v = scope.alloc(plane_bytes * (b * tk) as u64)?;
        for (bi, element) in planes.iter().enumerate() {
            for (i, &(pk, pv, src)) in element.iter().enumerate() {
                let off = plane_bytes * (bi as u64 * tk as u64 + i as u64);
                self.copy(scope, pk, src, k, off, plane_bytes)?;
                self.copy(scope, pv, src, v, off, plane_bytes)?;
            }
        }
        let mut win = WindowAttnUniform::same_window(
            windows[windows.len() - 1].0,
            windows[windows.len() - 1].1,
        );
        win[..windows.len()].copy_from_slice(windows);
        let u = WindowAttnUniform {
            b,
            tq: 1,
            tk,
            h: self.h,
            w: self.w,
            nh: HEADS,
            kvh: KV_HEADS,
            hd: HEAD_DIM,
            temporal_window: tk,
            clamp: false,
            scale: (HEAD_DIM as f32).powf(-0.5),
            windows: win,
        };
        let out = alloc_act(scope, self.bp(), b * self.plane(), DIM)?;
        let on_accel =
            self.p.act == ActDtype::F16 && scope.try_window_attn(q.data, k, v, out.data, &u)?;
        if !on_accel && scope.is_deferred() {
            return Err(WgpuError::AcceleratorDeclined {
                op: "relay_flash window_attn".into(),
            });
        }
        if !on_accel {
            let ub = scope.write_uniform(&u.to_bytes())?;
            scope.dispatch(
                &self.p.window_attn,
                &[q.data, k, v, out.data, ub],
                window_attn::workgroups(&u),
            )?;
        }
        Ok(out)
    }

    fn ffn<'w>(
        &self,
        scope: &BatchScope<'w, WgpuBackend>,
        x: ActBuf<'w>,
        blk: &BlockI,
        rows: u32,
    ) -> Result<ActBuf<'w>, WgpuError> {
        let xn = self.rms(scope, x, blk.ffn_norm, rows, DIM)?;
        let g = self.linear(scope, xn.data, blk.gate, rows, DIM, FFN)?;
        let u = self.linear(scope, xn.data, blk.up, rows, DIM, FFN)?;
        let a = alloc_act(scope, self.bp(), rows, FFN)?;
        if scope.is_deferred() {
            accel_op(scope, "relay_flash silu_mul", |ad| {
                Some(ElementwiseOp::SiluMulF16 {
                    g: ad(g.data)?,
                    u: ad(u.data)?,
                    out: ad(a.data)?,
                    n: rows * FFN,
                })
            })?;
        } else {
            op_silu_mul(scope, self.bp(), g, u, a)?;
        }
        let d = self.linear(scope, a.data, blk.down, rows, FFN, DIM)?;
        self.add(scope, x, d, rows, DIM)
    }

    fn persist<'w>(
        &self,
        scope: &BatchScope<'w, WgpuBackend>,
        x: ActBuf<'w>,
        bytes: u64,
    ) -> Result<WsBuf<WgpuBackend>, WgpuError> {
        let keep = self.scratch.alloc(bytes)?;
        let dst = scope.import_copy(keep.as_buf_ref());
        self.copy(scope, x.data, 0, dst, 0, bytes)?;
        Ok(keep)
    }

    /// A scope for one block (or the head): deferred onto the accelerator
    /// when every op runs there (f16 acts), so a block is one stream of
    /// kernels with no per-op drain.
    fn block_scope(&self) -> Result<BatchScope<'a, WgpuBackend>, WgpuError> {
        let scope = self.scratch.batch();
        if self.deferred {
            scope.accel_begin()?;
        }
        Ok(scope)
    }

    async fn submit(&self, scope: BatchScope<'a, WgpuBackend>) -> Result<(), WgpuError> {
        if self.deferred {
            scope.submit_deferred().await
        } else {
            scope.submit_void().await
        }
    }

    /// The `condition` pass of every keyframe latent (each `[48, h, w]`,
    /// channel-major) as one batch of planes.
    pub async fn conditions(&self, latents: &[Vec<f32>]) -> Result<Conditions, WgpuError> {
        let n = latents.len() as u32;
        let plane = self.plane();
        let rows = n * plane;
        let kv_plane = self.kv_plane_bytes();
        // channel-major -> token rows `[n, h*w, 48]`
        let hw = plane as usize;
        let ch = LATENT_CH as usize;
        let mut tok = vec![0f32; latents.len() * hw * ch];
        for (p, latent) in latents.iter().enumerate() {
            let dst = &mut tok[p * hw * ch..][..hw * ch];
            for c in 0..ch {
                for i in 0..hw {
                    dst[i * ch + c] = latent[c * hw + i];
                }
            }
        }
        let tb = act_upload_bytes(self.p.act, &tok);
        let tin = self.scratch.alloc(tb.len() as u64)?;
        self.backend.write_buffer(tin.id(), 0, &tb)?;
        let x_bytes = self.bp().act_bytes(rows * DIM);
        // One scope per block: a scope keeps every intermediate until its
        // submit, so a whole batched pass in one would hold ten blocks' worth.
        let mut x = {
            let scope = self.block_scope()?;
            let x = self.linear(
                &scope,
                scope.import_copy(tin.as_buf_ref()),
                self.l.cond_proj,
                rows,
                LATENT_CH,
                DIM,
            )?;
            let x = self.rms(&scope, x, self.l.cond_norm, rows, DIM)?;
            let x = self.add_emb(&scope, x, self.l.cond_type, rows)?;
            let keep = self.persist(&scope, x, x_bytes)?;
            self.submit(scope).await?;
            keep
        };
        let mut kv = Vec::with_capacity(DEPTH);
        for blk in &self.l.blocks {
            let scope = self.block_scope()?;
            let xin = ActBuf::dense(scope.import_copy(x.as_buf_ref()));
            let xn = self.rms(&scope, xin, blk.attn_norm, rows, DIM)?;
            let (q, k, v) = self.project(&scope, xn, blk, true, n)?;
            let kr = self.rope(&scope, k, &self.freqs, KV_HEADS, n)?;
            let planes: Vec<_> = (0..n as u64)
                .map(|p| vec![(kr.data, v.data, p * kv_plane)])
                .collect();
            let a = self.attend(&scope, q, &planes, &[COND_WINDOW])?;
            let o = self.linear(&scope, a.data, blk.output, rows, DIM, DIM)?;
            let x1 = self.add(&scope, xin, o, rows, DIM)?;
            let x2 = self.ffn(&scope, x1, blk, rows)?;
            let next = self.persist(&scope, x2, x_bytes)?;
            kv.push((
                self.persist(&scope, kr, kv_plane * n as u64)?,
                self.persist(&scope, v, kv_plane * n as u64)?,
            ));
            self.submit(scope).await?;
            x = next;
        }
        Ok(Conditions { kv, count: n })
    }

    /// One step of the `video` pass for the first `b` windows: batch element
    /// `i` decodes a frame of window `i` (between keyframes `i` and `i + 1`,
    /// clamped to the last). `lr` is the `b` padded LR frames NCHW `[b, 3,
    /// lr_h, lr_w]` in [-1, 1]; `history` carries each window's previous frame
    /// (empty at the windows' start) and is replaced. Returns the head output
    /// `[b, h*w, 768]` (pre pixel-shuffle).
    #[allow(clippy::too_many_arguments)]
    pub async fn frames(
        &self,
        lr: &[f32],
        b: u32,
        lr_h: u32,
        lr_w: u32,
        conds: &Conditions,
        history: &mut History,
    ) -> Result<Vec<f32>, WgpuError> {
        debug_assert!(b >= 1 && b <= self.max_batch);
        let plane = self.plane();
        let rows = b * plane;
        let kv_plane = self.kv_plane_bytes();
        let lb = act_upload_bytes(self.p.act, lr);
        let lin = self.scratch.alloc(lb.len() as u64)?;
        self.backend.write_buffer(lin.id(), 0, &lb)?;
        let head_out = self.scratch.alloc(self.bp().act_bytes(rows * HEAD_OUT))?;
        let mut next_hist = Vec::with_capacity(DEPTH);
        // Keyframe planes of each window: (start, end), clamped.
        let last = conds.count - 1;
        let ends: Vec<(u64, u64)> = (0..b)
            .map(|i| (i.min(last) as u64, (i + 1).min(last) as u64))
            .collect();
        let x_bytes = self.bp().act_bytes(rows * DIM);
        // One scope per stage (see `conditions`).
        let mut xs = {
            let scope = self.scratch.batch();
            // lr_encoder: 3 strided 3x3 convs (SiLU each), NCHW.
            let mut x = scope.import_copy(lin.as_buf_ref());
            let (mut ch, mut hh, mut ww) = (3u32, lr_h, lr_w);
            for c in &self.l.lr_convs {
                debug_assert_eq!(c.cin, ch);
                let ho = (hh + 2 - 3) / c.stride + 1;
                let wo = (ww + 2 - 3) / c.stride + 1;
                let out = scope.alloc(self.bp().act_bytes(b * c.cout * ho * wo))?;
                let u = conv2d_uniform(
                    &scope,
                    (b, ch, hh, ww),
                    (c.cout, ho, wo),
                    (3, 3, 1, 1, c.stride, c.stride),
                )?;
                let wv = scope.import_copy(self.views.buf(c.w));
                let bv = scope.import_copy(self.views.buf(c.b));
                scope.conv2d(
                    &self.p.conv.pipeline,
                    &self.p.conv.op,
                    x,
                    wv,
                    bv,
                    u,
                    out,
                    c.cout,
                    ho * wo,
                    b,
                )?;
                let s = self.silu(&scope, ActBuf::dense(out), b * c.cout * ho * wo)?;
                x = s.data;
                (ch, hh, ww) = (c.cout, ho, wo);
            }
            debug_assert_eq!((hh, ww), (self.h, self.w));
            // NCHW [b, 128, hw] -> rows [b, hw, 128]
            let t = scope.alloc(self.bp().act_bytes(ch * rows))?;
            let tu = scope.u32x4_uniform(b, ch, plane, 1)?;
            scope.transpose12::<Transpose12F32>(&self.p.transpose, x, tu, t, ch * rows)?;
            let x = self.linear(&scope, t, self.l.lr_proj, rows, ch, DIM)?;
            let x = self.rms(&scope, x, self.l.lr_norm, rows, DIM)?;
            let x = self.add_emb(&scope, x, self.l.video_type, rows)?;
            let keep = self.persist(&scope, x, x_bytes)?;
            scope.submit_void().await?;
            keep
        };
        for (i, blk) in self.l.blocks.iter().enumerate() {
            let scope = self.block_scope()?;
            let x = ActBuf::dense(scope.import_copy(xs.as_buf_ref()));
            let xn = self.rms(&scope, x, blk.attn_norm, rows, DIM)?;
            let (q, k, v) = self.project(&scope, xn, blk, false, b)?;
            let (ck, cv) = &conds.kv[i];
            let (ck, cv) = (
                scope.import_copy(ck.as_buf_ref()),
                scope.import_copy(cv.as_buf_ref()),
            );
            let mut planes: Vec<Vec<PlaneRef<'_>>> = ends
                .iter()
                .map(|&(s, e)| vec![(ck, cv, s * kv_plane), (ck, cv, e * kv_plane)])
                .collect();
            let mut windows = vec![COND_WINDOW, COND_WINDOW];
            if let Some((pk, pv)) = history.get(i) {
                let pk = ActBuf::dense(scope.import_copy(pk.as_buf_ref()));
                let pkr = self.rope(&scope, pk, &self.freqs_past, KV_HEADS, b)?;
                let pv = scope.import_copy(pv.as_buf_ref());
                for (bi, p) in planes.iter_mut().enumerate() {
                    p.push((pkr.data, pv, bi as u64 * kv_plane));
                }
                windows.push(VIDEO_WINDOW);
            }
            let kr = self.rope(&scope, k, &self.freqs, KV_HEADS, b)?;
            for (bi, p) in planes.iter_mut().enumerate() {
                p.push((kr.data, v.data, bi as u64 * kv_plane));
            }
            windows.push(VIDEO_WINDOW);
            let a = self.attend(&scope, q, &planes, &windows)?;
            let o = self.linear(&scope, a.data, blk.output, rows, DIM, DIM)?;
            let x1 = self.add(&scope, x, o, rows, DIM)?;
            let x2 = self.ffn(&scope, x1, blk, rows)?;
            let next = self.persist(&scope, x2, x_bytes)?;
            next_hist.push((
                self.persist(&scope, k, kv_plane * b as u64)?,
                self.persist(&scope, v, kv_plane * b as u64)?,
            ));
            self.submit(scope).await?;
            xs = next;
        }
        {
            let scope = self.block_scope()?;
            let x = scope.import_copy(xs.as_buf_ref());
            let hdn = self.linear(&scope, x, self.l.head_hidden, rows, DIM, HEAD_HIDDEN)?;
            let hdn = self.silu(&scope, hdn, rows * HEAD_HIDDEN)?;
            let o = self.linear(
                &scope,
                hdn.data,
                self.l.head_out,
                rows,
                HEAD_HIDDEN,
                HEAD_OUT,
            )?;
            let dst = scope.import_copy(head_out.as_buf_ref());
            self.copy(
                &scope,
                o.data,
                0,
                dst,
                0,
                self.bp().act_bytes(rows * HEAD_OUT),
            )?;
            self.submit(scope).await?;
        }
        *history = next_hist;
        let mut out = Vec::new();
        read_into_f32(
            self.backend,
            &head_out.as_buf_ref(),
            (rows * HEAD_OUT) as usize,
            self.p.act,
            &mut out,
        )
        .await?;
        Ok(out)
    }
}

/// `pixel_shuffle(r, 16)` of the head output `[h*w, 768]` plus
/// `bicubic(lr, up)`, cropped to `out_h x out_w`, as u8 RGB `[out_h, out_w, 3]`
/// (`(x + 1) / 2` clamped). `lr` is the padded NCHW `[3, lr_h, lr_w]` frame in
/// [-1, 1].
#[allow(clippy::too_many_arguments)]
pub fn compose_frame(
    head: &[f32],
    h: usize,
    w: usize,
    lr: &[f32],
    lr_h: usize,
    lr_w: usize,
    up: usize,
    out_h: usize,
    out_w: usize,
    detail: f32,
) -> Vec<u8> {
    let bic = bicubic_up(lr, 3, lr_h, lr_w, up);
    let (bh, bw) = (lr_h * up, lr_w * up);
    debug_assert_eq!((bh, bw), (h * 16, w * 16));
    let mut out = vec![0u8; out_h * out_w * 3];
    for y in 0..out_h {
        let (ty, iy) = (y / 16, y % 16);
        for x in 0..out_w {
            let (tx, ix) = (x / 16, x % 16);
            let row = &head[(ty * w + tx) * 768..][..768];
            for c in 0..3 {
                let r = row[c * 256 + iy * 16 + ix];
                let v = bic[(c * bh + y) * bw + x] + r * detail;
                out[(y * out_w + x) * 3 + c] =
                    (((v + 1.0) * 0.5).clamp(0.0, 1.0) * 255.0).round() as u8;
            }
        }
    }
    out
}

/// torch `F.interpolate(mode="bicubic", align_corners=False)` (A = -0.75,
/// border-clamped taps) of NCHW `[c, h, w]` by an integer factor.
pub fn bicubic_up(x: &[f32], c: usize, h: usize, w: usize, up: usize) -> Vec<f32> {
    fn weights(t: f32) -> [f32; 4] {
        const A: f32 = -0.75;
        let w1 = |d: f32| ((A + 2.0) * d - (A + 3.0)) * d * d + 1.0;
        let w2 = |d: f32| ((A * d - 5.0 * A) * d + 8.0 * A) * d - 4.0 * A;
        [w2(t + 1.0), w1(t), w1(1.0 - t), w2(2.0 - t)]
    }
    let (oh, ow) = (h * up, w * up);
    let taps = |o: usize, n: usize| -> ([usize; 4], [f32; 4]) {
        let src = (o as f32 + 0.5) / up as f32 - 0.5;
        let i0 = src.floor();
        let wt = weights(src - i0);
        let i0 = i0 as i64;
        let idx = [-1, 0, 1, 2].map(|d| (i0 + d).clamp(0, n as i64 - 1) as usize);
        (idx, wt)
    };
    let ytaps: Vec<_> = (0..oh).map(|y| taps(y, h)).collect();
    let xtaps: Vec<_> = (0..ow).map(|x| taps(x, w)).collect();
    let mut out = vec![0f32; c * oh * ow];
    let threads = std::thread::available_parallelism()
        .map_or(1, |n| n.get())
        .min(8);
    let rows_per = (c * oh).div_ceil(threads).max(1);
    std::thread::scope(|s| {
        for (ci, chunk) in out.chunks_mut(rows_per * ow).enumerate() {
            let (ytaps, xtaps) = (&ytaps, &xtaps);
            s.spawn(move || {
                for (r, orow) in chunk.chunks_mut(ow).enumerate() {
                    let gr = ci * rows_per + r;
                    let (ch, y) = (gr / oh, gr % oh);
                    let (yi, yw) = ytaps[y];
                    let plane = &x[ch * h * w..][..h * w];
                    for (xo, o) in orow.iter_mut().enumerate() {
                        let (xi, xw) = xtaps[xo];
                        let mut acc = 0f32;
                        for a in 0..4 {
                            let prow = &plane[yi[a] * w..][..w];
                            let mut racc = 0f32;
                            for b in 0..4 {
                                racc += prow[xi[b]] * xw[b];
                            }
                            acc += racc * yw[a];
                        }
                        *o = acc;
                    }
                }
            });
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A constant image stays constant (weights sum to 1, clamped borders).
    #[test]
    fn bicubic_preserves_constants() {
        let x = vec![0.25f32; 3 * 5 * 7];
        let y = bicubic_up(&x, 3, 5, 7, 2);
        assert_eq!(y.len(), 3 * 10 * 14);
        assert!(y.iter().all(|v| (v - 0.25).abs() < 1e-6));
    }

    /// Rope table: t=0 rows rotate the t pairs by 0 (cos 1, sin 0).
    #[test]
    fn rope_t0_identity_on_time_axis() {
        let r = grid_rope(0.0, 2, 3);
        assert_eq!(r.len(), 6 * 64);
        for p in 0..8 {
            assert_eq!(r[2 * p], 1.0);
            assert_eq!(r[2 * p + 1], 0.0);
        }
    }
}
