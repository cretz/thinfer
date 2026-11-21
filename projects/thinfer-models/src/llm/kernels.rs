//! Compiled LLM pipelines and typed dispatch helpers over a [`BatchScope`].
//! The WGSL lives in `thinfer_core::ops::llm`; this module owns one pipeline
//! per (kernel, weight format) and builds each dispatch's uniform + grid.

use std::collections::HashMap;

use thinfer_core::Backend;
use thinfer_core::backend::WgpuError as E;
use thinfer_core::backend::{WgpuBackend, WgpuError, WgpuPipeline};
use thinfer_core::ops::llm::{self as k, LlmOp, LlmWeight};
use thinfer_core::quant::QuantKind;
use thinfer_core::workspace::{BatchBuf, BatchScope};

/// Every weight format the LLM matmuls read in place.
pub const FORMATS: [LlmWeight; 8] = [
    LlmWeight::F32,
    LlmWeight::Quant(QuantKind::Q8_0),
    LlmWeight::Quant(QuantKind::Q4_0),
    LlmWeight::Quant(QuantKind::Q5_0),
    LlmWeight::Quant(QuantKind::Q5_1),
    LlmWeight::Quant(QuantKind::Q4_K),
    LlmWeight::Quant(QuantKind::Q5_K),
    LlmWeight::Quant(QuantKind::Q6_K),
];

pub struct LlmPipelines {
    gemv: HashMap<LlmWeight, WgpuPipeline>,
    rmsnorm: WgpuPipeline,
    rope: WgpuPipeline,
    kv_store: WgpuPipeline,
    /// Attention by dims-per-thread (head_dim <= 256 * dpt).
    attention: HashMap<u32, WgpuPipeline>,
    conv1d_silu: WgpuPipeline,
    delta_gates: WgpuPipeline,
    delta_net: WgpuPipeline,
    router: WgpuPipeline,
    glu: WgpuPipeline,
    moe_combine: WgpuPipeline,
}

type Scope<'s> = BatchScope<'s, WgpuBackend>;
type Buf<'s> = BatchBuf<'s>;

/// A weight matrix view: buffer + its encoding. `slot_bytes` is the stride
/// between expert slots (0 for a plain matrix).
#[derive(Clone, Copy)]
pub struct Mat<'s> {
    pub buf: Buf<'s>,
    pub fmt: LlmWeight,
    pub slot_bytes: u32,
}

/// Row-strided view of an f32 activation buffer.
#[derive(Clone, Copy)]
pub struct Act<'s> {
    pub buf: Buf<'s>,
    pub stride: u32,
    pub off: u32,
}

impl<'s> Act<'s> {
    pub fn dense(buf: Buf<'s>, cols: u32) -> Self {
        Self {
            buf,
            stride: cols,
            off: 0,
        }
    }

    pub fn at(self, off: u32) -> Self {
        Self {
            off: self.off + off,
            ..self
        }
    }
}

impl LlmPipelines {
    pub async fn compile(b: &WgpuBackend) -> Result<Self, WgpuError> {
        let mut gemv = HashMap::new();
        for f in FORMATS {
            let p = b
                .create_pipeline("llm_gemv", &k::gemv_wgsl(f), "main", &k::GEMV_LAYOUT)
                .await?;
            gemv.insert(f, p);
        }
        let mut attention = HashMap::new();
        for dpt in [1u32, 2] {
            let p = b
                .create_pipeline(
                    "llm_attention",
                    &k::attention_wgsl(dpt),
                    "main",
                    &k::ATTN_LAYOUT,
                )
                .await?;
            attention.insert(dpt, p);
        }
        Ok(Self {
            gemv,
            rmsnorm: b
                .create_pipeline("llm_rmsnorm", k::RMSNORM_WGSL, "main", &k::RMSNORM_LAYOUT)
                .await?,
            rope: b
                .create_pipeline("llm_rope", k::ROPE_WGSL, "main", &k::ROPE_LAYOUT)
                .await?,
            kv_store: b
                .create_pipeline(
                    "llm_kv_store",
                    k::KV_STORE_WGSL,
                    "main",
                    &k::KV_STORE_LAYOUT,
                )
                .await?,
            attention,
            conv1d_silu: b
                .create_pipeline(
                    "llm_conv1d_silu",
                    k::CONV1D_SILU_WGSL,
                    "main",
                    &k::CONV1D_SILU_LAYOUT,
                )
                .await?,
            delta_gates: b
                .create_pipeline(
                    "llm_delta_gates",
                    k::DELTA_GATES_WGSL,
                    "main",
                    &k::DELTA_GATES_LAYOUT,
                )
                .await?,
            delta_net: b
                .create_pipeline(
                    "llm_delta_net",
                    k::DELTA_NET_WGSL,
                    "main",
                    &k::DELTA_NET_LAYOUT,
                )
                .await?,
            router: b
                .create_pipeline(
                    "llm_router_topk",
                    k::ROUTER_TOPK_WGSL,
                    "main",
                    &k::ROUTER_TOPK_LAYOUT,
                )
                .await?,
            glu: b
                .create_pipeline("llm_glu", k::GLU_WGSL, "main", &k::GLU_LAYOUT)
                .await?,
            moe_combine: b
                .create_pipeline(
                    "llm_moe_combine",
                    k::MOE_COMBINE_WGSL,
                    "main",
                    &k::MOE_COMBINE_LAYOUT,
                )
                .await?,
        })
    }

    fn uniform<'s, T: Copy>(&self, s: &Scope<'s>, u: &T) -> Result<Buf<'s>, WgpuError> {
        s.write_uniform(&k::uniform_bytes(u))
    }

    /// `y[r, n] = W[n, :] . x[r, :]` for `rows` rows (+ `res` when given).
    #[allow(clippy::too_many_arguments)]
    pub fn gemv<'s>(
        &self,
        s: &Scope<'s>,
        w: Mat<'s>,
        x: Act<'s>,
        y: Act<'s>,
        res: Option<Buf<'s>>,
        rows: u32,
        n: u32,
        kdim: u32,
    ) -> Result<(), WgpuError> {
        let u = k::GemvUniform {
            n,
            k: kdim,
            x_stride: x.stride,
            x_off: x.off,
            y_stride: y.stride,
            y_off: y.off,
            slot_bytes: 0,
            flags: if res.is_some() { k::GEMV_RESIDUAL } else { 0 },
        };
        if s.is_deferred() {
            return accel(
                s,
                LlmOp::Gemv {
                    u,
                    fmt: w.fmt,
                    w: addr(s, w.buf)?,
                    x: addr(s, x.buf)?,
                    y: addr(s, y.buf)?,
                    res: res.map(|r| addr(s, r)).transpose()?.unwrap_or(0),
                    rows,
                },
            );
        }
        let ub = self.uniform(s, &u)?;
        s.dispatch(
            &self.gemv[&w.fmt],
            &[w.buf, x.buf, y.buf, ub, x.buf, res.unwrap_or(x.buf)],
            [n.div_ceil(k::GEMV_ROWS_PER_WG), rows, 1],
        )
    }

    /// Indirect gemv over `(slot, x_row, y_row)` triples in `table`.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_gather<'s>(
        &self,
        s: &Scope<'s>,
        w: Mat<'s>,
        x: Act<'s>,
        y: Act<'s>,
        table: Buf<'s>,
        pairs: u32,
        n: u32,
        kdim: u32,
    ) -> Result<(), WgpuError> {
        let u = k::GemvUniform {
            n,
            k: kdim,
            x_stride: x.stride,
            x_off: x.off,
            y_stride: y.stride,
            y_off: y.off,
            slot_bytes: w.slot_bytes,
            flags: k::GEMV_INDIRECT,
        };
        let ub = self.uniform(s, &u)?;
        s.dispatch(
            &self.gemv[&w.fmt],
            &[w.buf, x.buf, y.buf, ub, table, x.buf],
            [n.div_ceil(k::GEMV_ROWS_PER_WG), pairs, 1],
        )
    }

    /// Accelerator-only routed-expert gemv: pair `p` uses the matrix at device
    /// address `table[ids[p]]` (u64 per expert), x row `p / x_div`, y row `p`.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_experts<'s>(
        &self,
        s: &Scope<'s>,
        fmt: LlmWeight,
        table: Buf<'s>,
        ids: Buf<'s>,
        x: Act<'s>,
        y: Act<'s>,
        pairs: u32,
        x_div: u32,
        n: u32,
        kdim: u32,
    ) -> Result<(), WgpuError> {
        let u = k::GemvUniform {
            n,
            k: kdim,
            x_stride: x.stride,
            x_off: x.off,
            y_stride: y.stride,
            y_off: y.off,
            slot_bytes: 0,
            flags: 0,
        };
        accel(
            s,
            LlmOp::GemvExperts {
                u,
                fmt,
                table: addr(s, table)?,
                ids: addr(s, ids)?,
                x: addr(s, x.buf)?,
                y: addr(s, y.buf)?,
                pairs,
                x_div,
            },
        )
    }

    /// RMSNorm of `rows` rows of `dim` (weight `w`, or none), optionally
    /// adding `res` (same layout as `y`).
    #[allow(clippy::too_many_arguments)]
    pub fn rmsnorm<'s>(
        &self,
        s: &Scope<'s>,
        x: Act<'s>,
        w: Option<Buf<'s>>,
        y: Act<'s>,
        res: Option<Buf<'s>>,
        rows: u32,
        dim: u32,
        eps: f32,
    ) -> Result<(), WgpuError> {
        self.rmsnorm_with(s, x, w, y, res, rows, dim, eps, false)
    }

    /// [`Self::rmsnorm`] (weighted), then times `w[dim]`: the weight buffer
    /// carries a per-layer output scale after its `dim` weights.
    #[allow(clippy::too_many_arguments)]
    pub fn rmsnorm_out_scaled<'s>(
        &self,
        s: &Scope<'s>,
        x: Act<'s>,
        w: Buf<'s>,
        y: Act<'s>,
        res: Option<Buf<'s>>,
        rows: u32,
        dim: u32,
        eps: f32,
    ) -> Result<(), WgpuError> {
        self.rmsnorm_with(s, x, Some(w), y, res, rows, dim, eps, true)
    }

    #[allow(clippy::too_many_arguments)]
    fn rmsnorm_with<'s>(
        &self,
        s: &Scope<'s>,
        x: Act<'s>,
        w: Option<Buf<'s>>,
        y: Act<'s>,
        res: Option<Buf<'s>>,
        rows: u32,
        dim: u32,
        eps: f32,
        out_scale: bool,
    ) -> Result<(), WgpuError> {
        let mut flags = 0;
        if out_scale {
            flags |= k::RMSNORM_OUT_SCALE;
        }
        if w.is_none() {
            flags |= k::RMSNORM_NO_WEIGHT;
        }
        if res.is_some() {
            flags |= k::RMSNORM_RESIDUAL;
        }
        let u = k::RmsNormUniform {
            rows,
            dim,
            x_stride: x.stride,
            x_off: x.off,
            y_stride: y.stride,
            y_off: y.off,
            eps,
            flags,
        };
        if s.is_deferred() {
            return accel(
                s,
                LlmOp::RmsNorm {
                    u,
                    x: addr(s, x.buf)?,
                    w: w.map(|w| addr(s, w)).transpose()?.unwrap_or(0),
                    y: addr(s, y.buf)?,
                    res: res.map(|r| addr(s, r)).transpose()?.unwrap_or(0),
                },
            );
        }
        let ub = self.uniform(s, &u)?;
        s.dispatch(
            &self.rmsnorm,
            &[x.buf, w.unwrap_or(x.buf), y.buf, ub, res.unwrap_or(x.buf)],
            [rows, 1, 1],
        )
    }

    /// In-place NeoX rope over the first `2 * half` dims of each head.
    #[allow(clippy::too_many_arguments)]
    pub fn rope<'s>(
        &self,
        s: &Scope<'s>,
        x: Buf<'s>,
        cos_sin: Buf<'s>,
        rows: u32,
        heads: u32,
        row_stride: u32,
        head_stride: u32,
        half: u32,
    ) -> Result<(), WgpuError> {
        let u = k::RopeUniform {
            rows,
            heads,
            row_stride,
            head_stride,
            off: 0,
            half,
            ..Default::default()
        };
        if s.is_deferred() {
            return accel(
                s,
                LlmOp::Rope {
                    u,
                    x: addr(s, x)?,
                    cos_sin: addr(s, cos_sin)?,
                },
            );
        }
        let ub = self.uniform(s, &u)?;
        s.dispatch(
            &self.rope,
            &[x, cos_sin, ub],
            k::grid_1d(rows * heads * half),
        )
    }

    /// Append `rows` rows of width `width` from `src` into an f16 cache at
    /// positions `pos0..`; a `ring` of that many rows wraps (0 = linear).
    #[allow(clippy::too_many_arguments)]
    pub fn kv_store<'s>(
        &self,
        s: &Scope<'s>,
        src: Act<'s>,
        cache: Buf<'s>,
        rows: u32,
        width: u32,
        pos0: u32,
        ring: u32,
    ) -> Result<(), WgpuError> {
        let u = k::KvStoreUniform {
            rows,
            width,
            src_stride: src.stride,
            src_off: src.off,
            pos0,
            ring,
            ..Default::default()
        };
        if s.is_deferred() {
            return accel(
                s,
                LlmOp::KvStore {
                    u,
                    src: addr(s, src.buf)?,
                    cache: addr(s, cache)?,
                },
            );
        }
        let ub = self.uniform(s, &u)?;
        s.dispatch(
            &self.kv_store,
            &[src.buf, cache, ub],
            k::grid_1d(rows * width / 2),
        )
    }

    /// Causal attention of `rows` query rows (at absolute `pos0..`) over the
    /// caches (a `ring` of that many rows, 0 = linear); `gate` multiplies by
    /// `sigmoid(gate)` when given.
    #[allow(clippy::too_many_arguments)]
    pub fn attention<'s>(
        &self,
        s: &Scope<'s>,
        q: Act<'s>,
        q_head_stride: u32,
        kc: Buf<'s>,
        vc: Buf<'s>,
        gate: Option<(Act<'s>, u32)>,
        out: Act<'s>,
        rows: u32,
        heads: u32,
        kv_heads: u32,
        head_dim: u32,
        pos0: u32,
        window: u32,
        ring: u32,
        scale: f32,
    ) -> Result<(), WgpuError> {
        debug_assert!(
            ring == 0 || window == 0 || ring >= window + rows,
            "KV ring too small for this forward"
        );
        let (g, ghs) = gate.unwrap_or((q, q_head_stride));
        let u = k::AttnUniform {
            rows,
            heads,
            kv_heads,
            head_dim,
            pos0,
            window,
            scale,
            flags: if gate.is_some() { k::ATTN_GATE } else { 0 },
            q_stride: q.stride,
            q_head_stride,
            q_off: q.off,
            out_stride: out.stride,
            gate_stride: g.stride,
            gate_head_stride: ghs,
            gate_off: g.off,
            ring,
        };
        if s.is_deferred() {
            return accel(
                s,
                LlmOp::Attention {
                    u,
                    q: addr(s, q.buf)?,
                    k_cache: addr(s, kc)?,
                    v_cache: addr(s, vc)?,
                    gate: addr(s, g.buf)?,
                    out: addr(s, out.buf)?,
                },
            );
        }
        let ub = self.uniform(s, &u)?;
        let dpt = head_dim.div_ceil(256);
        s.dispatch(
            &self.attention[&dpt],
            &[q.buf, kc, vc, g.buf, out.buf, ub],
            [heads, rows, 1],
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn conv1d_silu<'s>(
        &self,
        s: &Scope<'s>,
        x: Buf<'s>,
        w: Buf<'s>,
        state: Buf<'s>,
        y: Buf<'s>,
        rows: u32,
        channels: u32,
        kernel: u32,
    ) -> Result<(), WgpuError> {
        let u = k::Conv1dUniform {
            rows,
            channels,
            kernel,
            _pad: 0,
        };
        if s.is_deferred() {
            return accel(
                s,
                LlmOp::Conv1dSilu {
                    u,
                    x: addr(s, x)?,
                    w: addr(s, w)?,
                    state: addr(s, state)?,
                    y: addr(s, y)?,
                },
            );
        }
        let ub = self.uniform(s, &u)?;
        s.dispatch(
            &self.conv1d_silu,
            &[x, w, state, y, ub],
            [channels.div_ceil(256), 1, 1],
        )
    }

    pub fn delta_gates<'s>(
        &self,
        s: &Scope<'s>,
        ab: Buf<'s>,
        a_neg: Buf<'s>,
        dt_bias: Buf<'s>,
        gb: Buf<'s>,
        rows: u32,
        heads: u32,
    ) -> Result<(), WgpuError> {
        let u = k::DeltaGatesUniform {
            rows,
            heads,
            ..Default::default()
        };
        if s.is_deferred() {
            return accel(
                s,
                LlmOp::DeltaGates {
                    u,
                    ab: addr(s, ab)?,
                    a_neg: addr(s, a_neg)?,
                    dt_bias: addr(s, dt_bias)?,
                    gb: addr(s, gb)?,
                },
            );
        }
        let ub = self.uniform(s, &u)?;
        s.dispatch(
            &self.delta_gates,
            &[ab, a_neg, dt_bias, gb, ub],
            [(rows * heads).div_ceil(256), 1, 1],
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn delta_net<'s>(
        &self,
        s: &Scope<'s>,
        qkv: Buf<'s>,
        gb: Buf<'s>,
        z: Buf<'s>,
        norm_w: Buf<'s>,
        state: Buf<'s>,
        out: Buf<'s>,
        u: k::DeltaNetUniform,
    ) -> Result<(), WgpuError> {
        let heads = u.v_heads;
        if s.is_deferred() {
            return accel(
                s,
                LlmOp::DeltaNet {
                    u,
                    qkv: addr(s, qkv)?,
                    gb: addr(s, gb)?,
                    z: addr(s, z)?,
                    norm_w: addr(s, norm_w)?,
                    state: addr(s, state)?,
                    out: addr(s, out)?,
                },
            );
        }
        let ub = self.uniform(s, &u)?;
        s.dispatch(
            &self.delta_net,
            &[qkv, gb, z, norm_w, state, out, ub],
            [heads, 1, 1],
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn router<'s>(
        &self,
        s: &Scope<'s>,
        logits: Buf<'s>,
        scale: Option<Buf<'s>>,
        ids: Buf<'s>,
        weights: Buf<'s>,
        rows: u32,
        experts: u32,
        top_k: u32,
        renorm: bool,
    ) -> Result<(), WgpuError> {
        let mut flags = 0;
        if renorm {
            flags |= k::ROUTER_RENORM;
        }
        if scale.is_some() {
            flags |= k::ROUTER_EXPERT_SCALE;
        }
        let u = k::RouterUniform {
            rows,
            experts,
            top_k,
            flags,
        };
        if s.is_deferred() {
            return accel(
                s,
                LlmOp::Router {
                    u,
                    logits: addr(s, logits)?,
                    scale: scale.map(|b| addr(s, b)).transpose()?.unwrap_or(0),
                    ids: addr(s, ids)?,
                    weights: addr(s, weights)?,
                },
            );
        }
        let ub = self.uniform(s, &u)?;
        s.dispatch(
            &self.router,
            &[logits, scale.unwrap_or(logits), ids, weights, ub],
            [rows, 1, 1],
        )
    }

    /// `y = act(a) * b` over `n` elements (SiLU, or GELU-tanh when `gelu`).
    pub fn glu<'s>(
        &self,
        s: &Scope<'s>,
        a: Buf<'s>,
        b: Buf<'s>,
        y: Buf<'s>,
        n: u32,
        gelu: bool,
    ) -> Result<(), WgpuError> {
        if s.is_deferred() {
            return accel(
                s,
                LlmOp::Glu {
                    n,
                    gelu,
                    a: addr(s, a)?,
                    b: addr(s, b)?,
                    y: addr(s, y)?,
                },
            );
        }
        let u: [u32; 4] = [n, if gelu { k::GLU_GELU_TANH } else { 0 }, 0, 0];
        let ub = self.uniform(s, &u)?;
        s.dispatch(&self.glu, &[a, b, y, ub], k::grid_1d(n))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn moe_combine<'s>(
        &self,
        s: &Scope<'s>,
        experts: Buf<'s>,
        weights: Buf<'s>,
        shared: Option<(Buf<'s>, Buf<'s>)>,
        res: Option<Buf<'s>>,
        y: Buf<'s>,
        rows: u32,
        dim: u32,
        top_k: u32,
    ) -> Result<(), WgpuError> {
        let mut flags = 0;
        if shared.is_some() {
            flags |= k::MOE_SHARED;
        }
        if res.is_some() {
            flags |= k::MOE_RESIDUAL;
        }
        let (sh, sg) = shared.unwrap_or((experts, weights));
        let u = k::MoeCombineUniform {
            rows,
            dim,
            top_k,
            flags,
        };
        if s.is_deferred() {
            return accel(
                s,
                LlmOp::MoeCombine {
                    u,
                    experts: addr(s, experts)?,
                    weights: addr(s, weights)?,
                    shared: addr(s, sh)?,
                    sg: addr(s, sg)?,
                    res: res.map(|r| addr(s, r)).transpose()?.unwrap_or(0),
                    y: addr(s, y)?,
                },
            );
        }
        let ub = self.uniform(s, &u)?;
        s.dispatch(
            &self.moe_combine,
            &[experts, weights, sh, sg, res.unwrap_or(experts), y, ub],
            k::grid_1d(rows * dim),
        )
    }
}

/// Device address of `b` (deferred accelerator mode).
fn addr<'s>(s: &Scope<'s>, b: Buf<'s>) -> Result<u64, WgpuError> {
    s.accel_addr(b).ok_or_else(|| E::AcceleratorDeclined {
        op: "llm: buffer is not device-addressable".into(),
    })
}

/// Enqueue `op` on the accelerator; a decline is an error (deferred work has
/// no backend fallback).
fn accel(s: &Scope<'_>, op: LlmOp) -> Result<(), WgpuError> {
    if s.accel_llm(&op) {
        Ok(())
    } else {
        Err(E::AcceleratorDeclined {
            op: format!("llm {op:?}"),
        })
    }
}

/// Bytes of a `[rows, cols]` f32 activation.
pub fn act_bytes(rows: u32, cols: u32) -> u64 {
    rows as u64 * cols as u64 * 4
}
