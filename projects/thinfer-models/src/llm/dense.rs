//! A dense Qwen3 decoder stack on the LLM kernels: bf16 matrices read in
//! place, f32 activations, f16 KV caches. Shared by the Qwen3-TTS talker /
//! code predictor (one row per step) and the MOSS-SoundEffect text encoder
//! (one prefill, every row's final-normed hidden state).

use std::sync::Arc;

use thinfer_core::Backend;
use thinfer_core::backend::WgpuBackend;
use thinfer_core::residency::{GpuView, WeightResidency};
use thinfer_core::tensor::StorageEncoding;
use thinfer_core::weight::{WeightId, WeightReader, WeightSource};
use thinfer_core::workspace::WsBuf;

use crate::llm::decoder::{Buf, LlmError, OwnedBuf, Registry, Scope, W, acquire, mat};
use crate::llm::kernels::{Act, LlmPipelines, act_bytes};

/// Shape of one Qwen3 decoder stack.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StackDims {
    pub hidden: u32,
    pub layers: u32,
    pub heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub ffn: u32,
    pub eps: f32,
    pub rope_theta: f32,
}

impl StackDims {
    pub const fn kv_width(&self) -> u32 {
        self.kv_heads * self.head_dim
    }
}

/// One decoder layer's weights (matrices in place, norms widened to f32
/// buffers the stack owns).
struct Layer {
    q: W,
    k: W,
    v: W,
    o: W,
    gate: W,
    up: W,
    down: W,
    in_norm: OwnedBuf,
    post_norm: OwnedBuf,
    q_norm: OwnedBuf,
    k_norm: OwnedBuf,
}

/// Matrices per layer, in [`DenseStack::weights`] order.
const PER_LAYER: usize = 7;

/// Which rows of the final norm a forward writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rows {
    /// The last row only (`[1, hidden]`): next-token steps.
    Last,
    /// Every row (`[t, hidden]`): an encoder's hidden states.
    All,
}

/// A decoder stack: layers + final norm.
pub struct DenseStack {
    pub dims: StackDims,
    layers: Vec<Layer>,
    norm: OwnedBuf,
}

/// Bytes -> f32 of a bf16 tensor read whole (norms, biases).
pub async fn bf16_vec<S: WeightSource>(src: &S, name: &str) -> Result<Vec<f32>, LlmError> {
    let id = WeightId(name.to_string());
    let e = src
        .catalog()
        .get(&id)
        .cloned()
        .ok_or_else(|| LlmError::Weights(format!("missing tensor {name}")))?;
    if e.encoding != Some(StorageEncoding::Bf16) {
        return Err(LlmError::Weights(format!("{name}: expected bf16")));
    }
    let mut r = src
        .open(&id)
        .await
        .map_err(|e| LlmError::Weights(format!("{e:?}")))?;
    let mut b = vec![0u8; e.size as usize];
    r.read_at(0, &mut b)
        .await
        .map_err(|e| LlmError::Weights(format!("{e:?}")))?;
    Ok(widen(&b))
}

/// bf16 bytes -> f32.
pub fn widen(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
        .collect()
}

/// Read row `idx` of a bf16 `[n, out.len()]` table (token embeddings) into
/// `out`, widened.
pub async fn bf16_row<S: WeightSource>(
    src: &S,
    name: &str,
    idx: u32,
    out: &mut [f32],
) -> Result<(), LlmError> {
    let id = WeightId(name.to_string());
    let mut r = src
        .open(&id)
        .await
        .map_err(|e| LlmError::Weights(format!("{e:?}")))?;
    let mut b = vec![0u8; out.len() * 2];
    r.read_at(idx as u64 * b.len() as u64, &mut b)
        .await
        .map_err(|e| LlmError::Weights(format!("{e:?}")))?;
    for (o, c) in out.iter_mut().zip(b.chunks_exact(2)) {
        *o = f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16);
    }
    Ok(())
}

/// NeoX rope `[rows, half, (cos, sin)]` for positions `pos0..`.
pub fn rope_rows(d: &StackDims, pos0: u32, rows: u32) -> Vec<f32> {
    let half = d.head_dim / 2;
    let mut v = Vec::with_capacity((rows * half * 2) as usize);
    for r in 0..rows {
        let p = (pos0 + r) as f64;
        for i in 0..half {
            let inv = 1.0f64 / (d.rope_theta as f64).powf((2 * i) as f64 / d.head_dim as f64);
            let a = p * inv;
            v.push(a.cos() as f32);
            v.push(a.sin() as f32);
        }
    }
    v
}

/// One layer's K and V caches (`[ctx, kv_width]` f16 each).
pub type KvPair = [WsBuf<WgpuBackend>; 2];

impl DenseStack {
    /// Register `{prefix}.layers.N.*` (bf16 in place) and read the norms.
    pub async fn load<S: WeightSource>(
        backend: &Arc<WgpuBackend>,
        residency: &WeightResidency<S>,
        prefix: &str,
        dims: StackDims,
    ) -> Result<Self, LlmError> {
        let reg = Registry { residency };
        let src = residency.source();
        let norm = async |name: String| -> Result<OwnedBuf, LlmError> {
            OwnedBuf::f32s(backend, &bf16_vec(src, &name).await?)
        };
        let mut layers = Vec::with_capacity(dims.layers as usize);
        for i in 0..dims.layers {
            let p = format!("{prefix}.layers.{i}");
            let d = |n: &str| reg.dense(&format!("{p}.{n}.weight"));
            layers.push(Layer {
                q: d("self_attn.q_proj")?,
                k: d("self_attn.k_proj")?,
                v: d("self_attn.v_proj")?,
                o: d("self_attn.o_proj")?,
                gate: d("mlp.gate_proj")?,
                up: d("mlp.up_proj")?,
                down: d("mlp.down_proj")?,
                in_norm: norm(format!("{p}.input_layernorm.weight")).await?,
                post_norm: norm(format!("{p}.post_attention_layernorm.weight")).await?,
                q_norm: norm(format!("{p}.self_attn.q_norm.weight")).await?,
                k_norm: norm(format!("{p}.self_attn.k_norm.weight")).await?,
            });
        }
        Ok(Self {
            dims,
            layers,
            norm: norm(format!("{prefix}.norm.weight")).await?,
        })
    }

    /// Every matrix, in the order [`Self::encode`] reads `views`.
    pub fn weights(&self) -> Vec<W> {
        self.layers
            .iter()
            .flat_map(|l| [l.q, l.k, l.v, l.o, l.gate, l.up, l.down])
            .collect()
    }

    /// Acquire [`Self::weights`] (resident for the views' lifetime).
    pub async fn views<'r, S: WeightSource>(
        &self,
        residency: &'r WeightResidency<S>,
        backend: &Arc<WgpuBackend>,
    ) -> Result<Vec<GpuView<'r>>, LlmError> {
        let mut v = Vec::with_capacity(self.layers.len() * PER_LAYER);
        for w in self.weights() {
            v.push(acquire(residency, backend, w).await?);
        }
        Ok(v)
    }

    /// Allocate one KV pair per layer for `ctx` positions.
    pub fn kv(
        &self,
        ws: &thinfer_core::workspace::Workspace<WgpuBackend>,
        ctx: u32,
    ) -> Result<Vec<KvPair>, LlmError> {
        let b = ctx as u64 * self.dims.kv_width() as u64 * 2;
        (0..self.dims.layers)
            .map(|_| Ok([ws.alloc(b)?, ws.alloc(b)?]))
            .collect()
    }

    /// One forward of `t` rows at `pos` from `x` (`[t, hidden]`): the KV of
    /// the rows lands in `kv` at `pos..`, and the final-normed `rows` go to
    /// `out`.
    #[allow(clippy::too_many_arguments)]
    pub fn encode<'s>(
        &self,
        s: &Scope<'s>,
        p: &LlmPipelines,
        v: &[GpuView<'_>],
        kv: &[KvPair],
        rope: Buf<'s>,
        mut x: Buf<'s>,
        t: u32,
        pos: u32,
        out: Buf<'s>,
        rows: Rows,
    ) -> Result<(), LlmError> {
        let d = self.dims;
        let (hd, nh, nkv, dh, ff) = (d.hidden, d.heads, d.kv_heads, d.head_dim, d.ffn);
        let (qw, kvw) = (nh * dh, d.kv_width());
        let imp = |b: &OwnedBuf| s.import_copy(b.buf_ref());
        for (li, l) in self.layers.iter().enumerate() {
            let lv = &v[li * PER_LAYER..(li + 1) * PER_LAYER];
            let h = s.alloc(act_bytes(t, hd))?;
            p.rmsnorm(
                s,
                Act::dense(x, hd),
                Some(imp(&l.in_norm)),
                Act::dense(h, hd),
                None,
                t,
                hd,
                d.eps,
            )?;
            let q = s.alloc(act_bytes(t, qw))?;
            let k = s.alloc(act_bytes(t, kvw))?;
            let vv = s.alloc(act_bytes(t, kvw))?;
            for (wi, w, y, n) in [(0, l.q, q, qw), (1, l.k, k, kvw), (2, l.v, vv, kvw)] {
                p.gemv(
                    s,
                    mat(s, &lv[wi], w.fmt),
                    Act::dense(h, hd),
                    Act::dense(y, n),
                    None,
                    t,
                    n,
                    hd,
                )?;
            }
            let qn = s.alloc(act_bytes(t * nh, dh))?;
            p.rmsnorm(
                s,
                Act::dense(q, dh),
                Some(imp(&l.q_norm)),
                Act::dense(qn, dh),
                None,
                t * nh,
                dh,
                d.eps,
            )?;
            let kn = s.alloc(act_bytes(t * nkv, dh))?;
            p.rmsnorm(
                s,
                Act::dense(k, dh),
                Some(imp(&l.k_norm)),
                Act::dense(kn, dh),
                None,
                t * nkv,
                dh,
                d.eps,
            )?;
            p.rope(s, qn, rope, t, nh, qw, dh, dh / 2)?;
            p.rope(s, kn, rope, t, nkv, kvw, dh, dh / 2)?;
            let kc = s.import_copy(kv[li][0].as_buf_ref());
            let vc = s.import_copy(kv[li][1].as_buf_ref());
            p.kv_store(s, Act::dense(kn, kvw), kc, t, kvw, pos, 0)?;
            p.kv_store(s, Act::dense(vv, kvw), vc, t, kvw, pos, 0)?;
            let ao = s.alloc(act_bytes(t, qw))?;
            p.attention(
                s,
                Act::dense(qn, qw),
                dh,
                kc,
                vc,
                None,
                Act::dense(ao, qw),
                t,
                nh,
                nkv,
                dh,
                pos,
                0,
                0,
                1.0 / (dh as f32).sqrt(),
            )?;
            let x2 = s.alloc(act_bytes(t, hd))?;
            p.gemv(
                s,
                mat(s, &lv[3], l.o.fmt),
                Act::dense(ao, qw),
                Act::dense(x2, hd),
                Some(x),
                t,
                hd,
                qw,
            )?;
            let hn = s.alloc(act_bytes(t, hd))?;
            p.rmsnorm(
                s,
                Act::dense(x2, hd),
                Some(imp(&l.post_norm)),
                Act::dense(hn, hd),
                None,
                t,
                hd,
                d.eps,
            )?;
            let g = s.alloc(act_bytes(t, ff))?;
            let u = s.alloc(act_bytes(t, ff))?;
            p.gemv(
                s,
                mat(s, &lv[4], l.gate.fmt),
                Act::dense(hn, hd),
                Act::dense(g, ff),
                None,
                t,
                ff,
                hd,
            )?;
            p.gemv(
                s,
                mat(s, &lv[5], l.up.fmt),
                Act::dense(hn, hd),
                Act::dense(u, ff),
                None,
                t,
                ff,
                hd,
            )?;
            let a = s.alloc(act_bytes(t, ff))?;
            p.glu(s, g, u, a, t * ff, false)?;
            let x3 = s.alloc(act_bytes(t, hd))?;
            p.gemv(
                s,
                mat(s, &lv[6], l.down.fmt),
                Act::dense(a, ff),
                Act::dense(x3, hd),
                Some(x2),
                t,
                hd,
                ff,
            )?;
            for b in [h, q, k, vv, qn, kn, ao, hn, g, u, a, x2] {
                s.recycle(b);
            }
            if li > 0 {
                s.recycle(x);
            }
            x = x3;
        }
        let (from, n) = match rows {
            Rows::Last => ((t - 1) * hd, 1),
            Rows::All => (0, t),
        };
        p.rmsnorm(
            s,
            Act::dense(x, hd).at(from),
            Some(imp(&self.norm)),
            Act::dense(out, hd),
            None,
            n,
            hd,
            d.eps,
        )?;
        s.recycle(x);
        Ok(())
    }
}

/// Host <-> device transfers that work on both paths: in stream order on an
/// accelerator scope, else queue writes (land before the scope's submit) and
/// reads after it.
pub struct Io<'b> {
    pub backend: &'b WgpuBackend,
}

impl Io<'_> {
    pub fn put<'s>(
        &self,
        s: &Scope<'s>,
        dst: &WsBuf<WgpuBackend>,
        bytes: &[u8],
    ) -> Result<(), LlmError> {
        if s.is_deferred() {
            let b = s.import_copy(dst.as_buf_ref());
            let (Some(a), Some(acc)) = (s.accel_addr(b), s.accelerator()) else {
                return Err(LlmError::Weights("write: not device-addressable".into()));
            };
            acc.write(a, bytes).map_err(LlmError::Weights)?;
        } else {
            self.backend.write_buffer(dst.id(), 0, bytes)?;
        }
        Ok(())
    }

    pub fn get_deferred<'s>(
        s: &Scope<'s>,
        src: &WsBuf<WgpuBackend>,
        n: usize,
    ) -> Result<Vec<f32>, LlmError> {
        let b = s.import_copy(src.as_buf_ref());
        let (Some(a), Some(acc)) = (s.accel_addr(b), s.accelerator()) else {
            return Err(LlmError::Weights("read: not device-addressable".into()));
        };
        let mut v = vec![0f32; n];
        acc.read(a, bytemuck::cast_slice_mut(&mut v))
            .map_err(LlmError::Weights)?;
        Ok(v)
    }

    /// Read `n` f32 of `src` after `s` (submitting it): in stream order on an
    /// accelerator scope, else a readback after the submit.
    pub async fn submit_read<'s>(
        &self,
        s: Scope<'s>,
        src: &WsBuf<WgpuBackend>,
        n: usize,
    ) -> Result<Vec<f32>, LlmError> {
        if s.is_deferred() {
            let v = Self::get_deferred(&s, src, n)?;
            s.submit_void().await?;
            Ok(v)
        } else {
            s.submit_void().await?;
            let bytes = self.backend.read_buffer(src.id(), 0, n as u64 * 4).await?;
            Ok(bytemuck::cast_slice::<u8, f32>(&bytes).to_vec())
        }
    }
}
