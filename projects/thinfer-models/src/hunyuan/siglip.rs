//! SigLIP so400m-patch14-384 vision tower (the FLUX.1-Redux image encoder,
//! `Comfy-Org/sigclip_vision_384` F16 safetensors). Encodes the I2V
//! conditioning image into `[729, 1152]` tokens (`last_hidden_state`, i.e.
//! post-layernorm encoder output; the pooling head is unused). Ground truth:
//! HF `SiglipVisionModel` (the exact class minWM's `VisionEncoder` wraps).
//!
//! Arch: patch conv 14x14/14 (run as an im2col matmul over host-extracted
//! patches) + learned 729-token position embedding; 27 pre-LN blocks (affine
//! LN -> q/k/v (16 heads x 72, NO qk-norm) -> SDPA -> out_proj -> residual;
//! affine LN -> fc1 -> gelu-tanh -> fc2 -> residual); final post_layernorm.
//! head_dim 72 fails the f16-subgroup `D % 32` gate, so `op_sdpa` runs the
//! dense SdpaF32 path automatically -- fine for a one-shot 729-token encode.
//!
//! Reuses the DiT's compiled [`BlockPipelines`] (same act dtype); all matmuls
//! are `Module`-site dense bf16.

use thinfer_core::backend::{Backend, WgpuBackend, WgpuError};
use thinfer_core::ops::AddF32;
use thinfer_core::residency::{GpuView, ResidencyError, WeightHandle, WeightResidency};
use thinfer_core::weight::{WeightId, WeightSource};
use thinfer_core::workspace::{BatchBuf, BatchScope, Workspace};

use crate::common::block::{ActBuf, BlockPipelines, op_sdpa};
use crate::common::loader::{
    LoadError, register_linear, register_linear_flatten, register_passthrough,
};
use crate::common::seq::{act_readback_to_f32, act_upload_bytes};

/// Input image side (the processor resizes to 384x384) and derived dims.
pub const IMAGE_SIZE: usize = 384;
pub const PATCH: usize = 14;
pub const GRID: usize = IMAGE_SIZE / PATCH; // 27
pub const TOKENS: usize = GRID * GRID; // 729
pub const HIDDEN: usize = 1152;
const PATCH_K: usize = 3 * PATCH * PATCH; // 588
const LAYERS: usize = 27;
const HEADS: u32 = 16;
const HEAD_DIM: u32 = 72;
const MLP_HIDDEN: u32 = 4304;
const LN_EPS: f32 = 1e-6;

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

struct LayerW {
    ln1: LinW,
    q: LinW,
    k: LinW,
    v: LinW,
    out: LinW,
    ln2: LinW,
    fc1: LinW,
    fc2: LinW,
}

impl LayerW {
    fn new(i: usize) -> Self {
        let p = format!("vision_model.encoder.layers.{i}");
        Self {
            ln1: lin(&format!("{p}.layer_norm1")),
            q: lin(&format!("{p}.self_attn.q_proj")),
            k: lin(&format!("{p}.self_attn.k_proj")),
            v: lin(&format!("{p}.self_attn.v_proj")),
            out: lin(&format!("{p}.self_attn.out_proj")),
            ln2: lin(&format!("{p}.layer_norm2")),
            fc1: lin(&format!("{p}.mlp.fc1")),
            fc2: lin(&format!("{p}.mlp.fc2")),
        }
    }
}

#[derive(Clone, Copy)]
struct LinH {
    weight: WeightHandle,
    bias: WeightHandle,
}

struct LayerH {
    ln1: LinH,
    q: LinH,
    k: LinH,
    v: LinH,
    out: LinH,
    ln2: LinH,
    fc1: LinH,
    fc2: LinH,
}

pub struct SiglipEncoder {
    patch: LinH,
    pos: WeightHandle,
    layers: Vec<LayerH>,
    post_ln: LinH,
}

#[derive(Debug)]
pub enum SiglipError<SE: core::fmt::Debug> {
    Wgpu(WgpuError),
    Residency(ResidencyError<SE, WgpuError>),
}
impl<SE: core::fmt::Debug> From<WgpuError> for SiglipError<SE> {
    fn from(e: WgpuError) -> Self {
        Self::Wgpu(e)
    }
}
impl<SE: core::fmt::Debug> From<ResidencyError<SE, WgpuError>> for SiglipError<SE> {
    fn from(e: ResidencyError<SE, WgpuError>) -> Self {
        Self::Residency(e)
    }
}

fn reg_lin<S: WeightSource>(res: &WeightResidency<S>, w: &LinW) -> Result<LinH, LoadError> {
    Ok(LinH {
        weight: register_linear(res, &w.weight)?,
        bias: register_passthrough(res, &w.bias)?,
    })
}

fn reg_ln<S: WeightSource>(res: &WeightResidency<S>, w: &LinW) -> Result<LinH, LoadError> {
    Ok(LinH {
        weight: register_passthrough(res, &w.weight)?,
        bias: register_passthrough(res, &w.bias)?,
    })
}

impl SiglipEncoder {
    pub fn new<S: WeightSource>(res: &WeightResidency<S>) -> Result<Self, LoadError> {
        let patch = lin("vision_model.embeddings.patch_embedding");
        let mut layers = Vec::with_capacity(LAYERS);
        for i in 0..LAYERS {
            let w = LayerW::new(i);
            layers.push(LayerH {
                ln1: reg_ln(res, &w.ln1)?,
                q: reg_lin(res, &w.q)?,
                k: reg_lin(res, &w.k)?,
                v: reg_lin(res, &w.v)?,
                out: reg_lin(res, &w.out)?,
                ln2: reg_ln(res, &w.ln2)?,
                fc1: reg_lin(res, &w.fc1)?,
                fc2: reg_lin(res, &w.fc2)?,
            });
        }
        Ok(Self {
            patch: LinH {
                // conv [1152, 3, 14, 14] flattened to a [588 -> 1152] linear.
                weight: register_linear_flatten(res, &patch.weight)?,
                bias: register_passthrough(res, &patch.bias)?,
            },
            pos: register_passthrough(
                res,
                &WeightId("vision_model.embeddings.position_embedding.weight".into()),
            )?,
            layers,
            post_ln: reg_ln(res, &lin("vision_model.post_layernorm"))?,
        })
    }

    /// Host im2col matching the conv weight reshape: token (gy,gx) row =
    /// `[c=0..3][ky=0..14][kx=0..14]` of `pixels` (`[3, 384, 384]`, already
    /// resized + normalized `(px/255 - 0.5) / 0.5`).
    fn im2col(pixels: &[f32]) -> Vec<f32> {
        assert_eq!(
            pixels.len(),
            3 * IMAGE_SIZE * IMAGE_SIZE,
            "pixels [3,384,384]"
        );
        let mut out = vec![0.0f32; TOKENS * PATCH_K];
        for gy in 0..GRID {
            for gx in 0..GRID {
                let row = (gy * GRID + gx) * PATCH_K;
                for c in 0..3 {
                    for ky in 0..PATCH {
                        let y = gy * PATCH + ky;
                        let src = c * IMAGE_SIZE * IMAGE_SIZE + y * IMAGE_SIZE + gx * PATCH;
                        let dst = row + c * PATCH * PATCH + ky * PATCH;
                        out[dst..dst + PATCH].copy_from_slice(&pixels[src..src + PATCH]);
                    }
                }
            }
        }
        out
    }

    /// Encode a preprocessed `[3, 384, 384]` image to `[729, 1152]` tokens.
    pub async fn encode<S: WeightSource>(
        &self,
        backend: &WgpuBackend,
        residency: &WeightResidency<S>,
        ws: &Workspace<WgpuBackend>,
        bp: &BlockPipelines,
        pixels: &[f32],
    ) -> Result<Vec<f32>, SiglipError<S::Error>> {
        let act = bp.act_dtype;
        let asz = act.bytes_per_elem();
        let dim = HIDDEN as u32;
        let n = TOKENS as u32;

        let patches = Self::im2col(pixels);
        let bytes = act_upload_bytes(act, &patches);
        let patches_up = ws.alloc(bytes.len() as u64)?;
        backend.write_buffer(patches_up.id(), 0, &bytes)?;

        // Embeddings: patches @ conv + bias + position embedding.
        let x_ws = ws.alloc((n * dim) as u64 * asz)?;
        {
            let pw = residency.acquire(self.patch.weight, backend).await?;
            let pb = residency.acquire(self.patch.bias, backend).await?;
            let pos = residency.acquire(self.pos, backend).await?;
            let scope = ws.batch();
            let xin = scope.import_copy(patches_up.as_buf_ref());
            let emb = lin_fwd(&scope, bp, xin, pw.buf(), pb.buf(), n, dim, PATCH_K as u32)?;
            let posv = scope.import_copy(pos.buf());
            let x = scope.alloc(bp.act_bytes(n * dim))?;
            scope.dispatch_op::<AddF32>(&bp.add, &[emb, posv], x)?;
            let dst = scope.import_copy(x_ws.as_buf_ref());
            scope.copy_buffer_to_buffer(x, 0, dst, 0, (n * dim) as u64 * asz)?;
            scope.submit_void().await?;
        }

        // Encoder layers (weights acquired per layer; the whole tower is small).
        let mut cur = x_ws;
        for lw in &self.layers {
            let mut lp: Vec<GpuView> = Vec::new();
            let l = acquire_layer(lw, residency, backend, &mut lp).await?;
            let nxt = ws.alloc((n * dim) as u64 * asz)?;
            {
                let scope = ws.batch();
                let x = scope.import_copy(cur.as_buf_ref());
                // Attention sublayer.
                let h1 = affine_ln(&scope, bp, x, &l.ln1, n, dim)?;
                let q = lin_fwd(&scope, bp, h1, l.q.0, l.q.1, n, dim, dim)?;
                let k = lin_fwd(&scope, bp, h1, l.k.0, l.k.1, n, dim, dim)?;
                let v = lin_fwd(&scope, bp, h1, l.v.0, l.v.1, n, dim, dim)?;
                let sa = scope.alloc(bp.act_bytes(n * dim))?;
                let mask = scope.write_uniform(&0f32.to_le_bytes())?;
                let scale = 1.0_f32 / (HEAD_DIM as f32).sqrt();
                op_sdpa(
                    &scope,
                    bp,
                    ActBuf::dense(q),
                    ActBuf::dense(k),
                    ActBuf::dense(v),
                    mask,
                    ActBuf::dense(sa),
                    1,
                    n,
                    n,
                    HEADS,
                    HEADS,
                    HEAD_DIM,
                    scale,
                    0,
                )?;
                let o = lin_fwd(&scope, bp, sa, l.out.0, l.out.1, n, dim, dim)?;
                let x1 = scope.alloc(bp.act_bytes(n * dim))?;
                scope.dispatch_op::<AddF32>(&bp.add, &[x, o], x1)?;
                // MLP sublayer.
                let h2 = affine_ln(&scope, bp, x1, &l.ln2, n, dim)?;
                let up = lin_fwd(&scope, bp, h2, l.fc1.0, l.fc1.1, n, MLP_HIDDEN, dim)?;
                let g = scope.alloc(bp.act_bytes(n * MLP_HIDDEN))?;
                scope.dispatch_op::<thinfer_core::ops::GeluF32>(&bp.gelu, &[up], g)?;
                let down = lin_fwd(&scope, bp, g, l.fc2.0, l.fc2.1, n, dim, MLP_HIDDEN)?;
                let x2 = scope.alloc(bp.act_bytes(n * dim))?;
                scope.dispatch_op::<AddF32>(&bp.add, &[x1, down], x2)?;
                let dst = scope.import_copy(nxt.as_buf_ref());
                scope.copy_buffer_to_buffer(x2, 0, dst, 0, (n * dim) as u64 * asz)?;
                scope.submit_void().await?;
            }
            cur = nxt;
        }

        // post_layernorm -> readback.
        let out_ws = ws.alloc((n * dim) as u64 * asz)?;
        {
            let mut lp: Vec<GpuView> = Vec::new();
            let w = acq_lin_bufs(self.post_ln, residency, backend, &mut lp).await?;
            let scope = ws.batch();
            let x = scope.import_copy(cur.as_buf_ref());
            let y = affine_ln(&scope, bp, x, &w, n, dim)?;
            let dst = scope.import_copy(out_ws.as_buf_ref());
            scope.copy_buffer_to_buffer(y, 0, dst, 0, (n * dim) as u64 * asz)?;
            scope.submit_void().await?;
        }
        let bytes = backend
            .read_buffer(out_ws.id(), 0, (n * dim) as u64 * asz)
            .await?;
        Ok(act_readback_to_f32(act, &bytes, TOKENS * HIDDEN))
    }
}

/// `(weight, bias)` GPU views for one linear/LN.
type LinBufs2 = (thinfer_core::backend::BufRef, thinfer_core::backend::BufRef);

struct LayerBufs {
    ln1: LinBufs2,
    q: LinBufs2,
    k: LinBufs2,
    v: LinBufs2,
    out: LinBufs2,
    ln2: LinBufs2,
    fc1: LinBufs2,
    fc2: LinBufs2,
}

async fn acq_lin_bufs<'r, S: WeightSource>(
    h: LinH,
    res: &'r WeightResidency<S>,
    backend: &WgpuBackend,
    pins: &mut Vec<GpuView<'r>>,
) -> Result<LinBufs2, ResidencyError<S::Error, WgpuError>> {
    let w = res.acquire(h.weight, backend).await?;
    let b = res.acquire(h.bias, backend).await?;
    let out = (w.buf(), b.buf());
    pins.push(w);
    pins.push(b);
    Ok(out)
}

async fn acquire_layer<'r, S: WeightSource>(
    lw: &LayerH,
    res: &'r WeightResidency<S>,
    backend: &WgpuBackend,
    pins: &mut Vec<GpuView<'r>>,
) -> Result<LayerBufs, ResidencyError<S::Error, WgpuError>> {
    Ok(LayerBufs {
        ln1: acq_lin_bufs(lw.ln1, res, backend, pins).await?,
        q: acq_lin_bufs(lw.q, res, backend, pins).await?,
        k: acq_lin_bufs(lw.k, res, backend, pins).await?,
        v: acq_lin_bufs(lw.v, res, backend, pins).await?,
        out: acq_lin_bufs(lw.out, res, backend, pins).await?,
        ln2: acq_lin_bufs(lw.ln2, res, backend, pins).await?,
        fc1: acq_lin_bufs(lw.fc1, res, backend, pins).await?,
        fc2: acq_lin_bufs(lw.fc2, res, backend, pins).await?,
    })
}

/// Dense module-site linear `x @ wT + b` using the shared block kernels.
#[allow(clippy::too_many_arguments)]
fn lin_fwd<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    bp: &BlockPipelines,
    x: BatchBuf<'w>,
    w: thinfer_core::backend::BufRef,
    b: thinfer_core::backend::BufRef,
    rows: u32,
    n: u32,
    k: u32,
) -> Result<BatchBuf<'w>, WgpuError> {
    use thinfer_core::ops::{BcastAddF32, MatMulF32};
    let pre = scope.alloc(bp.act_bytes(rows * n))?;
    let dims = scope.u32x4_uniform(rows, n, k, 0)?;
    let wv = scope.import_copy(w);
    scope.matmul::<MatMulF32>(
        &bp.matmul_module,
        &bp.matmuls.module,
        x,
        wv,
        dims,
        pre,
        rows,
        n,
    )?;
    let out = scope.alloc(bp.act_bytes(rows * n))?;
    let u = scope.u32x4_uniform(n, 0, 0, 0)?;
    let bv = scope.import_copy(b);
    scope.bcast_add::<BcastAddF32>(&bp.bcast_add, pre, bv, u, out, rows * n)?;
    Ok(out)
}

/// Affine LayerNorm `LN(x)*w[c]+b[c]`.
fn affine_ln<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    bp: &BlockPipelines,
    x: BatchBuf<'w>,
    w: &LinBufs2,
    rows: u32,
    dim: u32,
) -> Result<BatchBuf<'w>, WgpuError> {
    use thinfer_core::ops::{BcastAddF32, BcastMulF32, LayerNormF32};
    let normed = scope.alloc(bp.act_bytes(rows * dim))?;
    let lu = scope.u32x4_uniform(rows, dim, LN_EPS.to_bits(), 0)?;
    scope.layernorm::<LayerNormF32>(&bp.layernorm, x, lu, normed, rows)?;
    let scaled = scope.alloc(bp.act_bytes(rows * dim))?;
    let au = scope.u32x4_uniform(dim, 0, 0, 0)?;
    let wv = scope.import_copy(w.0);
    scope.bcast_add::<BcastMulF32>(&bp.bcast_mul, normed, wv, au, scaled, rows * dim)?;
    let out = scope.alloc(bp.act_bytes(rows * dim))?;
    let bu = scope.u32x4_uniform(dim, 0, 0, 0)?;
    let bv = scope.import_copy(w.1);
    scope.bcast_add::<BcastAddF32>(&bp.bcast_add, scaled, bv, bu, out, rows * dim)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn im2col_layout() {
        // A single lit pixel at (c=1, y=15, x=2) lands in token (gy=1, gx=0)
        // at k = 1*196 + 1*14 + 2 (ky=1, kx=2).
        let mut px = vec![0.0f32; 3 * IMAGE_SIZE * IMAGE_SIZE];
        px[IMAGE_SIZE * IMAGE_SIZE + 15 * IMAGE_SIZE + 2] = 7.0;
        let cols = SiglipEncoder::im2col(&px);
        let row = GRID * PATCH_K; // token (1, 0)
        assert_eq!(cols[row + 196 + 14 + 2], 7.0);
        assert_eq!(cols.iter().filter(|&&v| v != 0.0).count(), 1);
    }
}
