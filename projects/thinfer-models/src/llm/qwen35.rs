//! Qwen3.5 / Qwen3.6 MoE (`qwen35moe`): hybrid Gated-DeltaNet + gated full
//! attention decoder with routed + shared experts, over the GGUF the engine
//! loads (llama.cpp layout: linear-attn V heads tiled, zero-centered norms
//! stored `w + 1`, `ssm_a = -exp(A_log)`). The forward is the shared
//! [`Decoder`]; this file is the architecture's [`Blocks`].

use std::sync::Arc;

use thinfer_core::backend::WgpuBackend;
use thinfer_core::ops::llm::DeltaNetUniform;
use thinfer_core::policy::ResidencyBudget;
use thinfer_core::residency::{GpuView, WeightResidency};
use thinfer_core::weight::WeightSource;
use thinfer_core::workspace::WsBuf;

use crate::llm::decoder::{
    Blocks, Buf, Combine, Decoder, Dims, LayerBuf, Mid, Registry, Scope, StateBuf, W, mat,
};
pub use crate::llm::decoder::{LlmError, LlmState, Output, PhaseTimes, Want};
use crate::llm::experts::ExpertTensors;
use crate::llm::kernels::{Act, LlmPipelines, act_bytes};

#[derive(Clone, Debug)]
pub struct Qwen35Config {
    pub hidden: u32,
    pub layers: u32,
    /// `true` = Gated DeltaNet layer, `false` = gated full attention.
    pub linear: Vec<bool>,
    pub heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    /// Rotary dims per head (`head_dim * partial_rotary_factor`).
    pub rot_dim: u32,
    pub rope_theta: f32,
    pub lin_k_heads: u32,
    pub lin_v_heads: u32,
    pub lin_head_dim: u32,
    pub conv_kernel: u32,
    pub experts: u32,
    pub top_k: u32,
    pub moe_ff: u32,
    pub shared_ff: u32,
    pub vocab: u32,
    pub eps: f32,
}

impl Qwen35Config {
    /// From the upstream `config.json` (`text_config` when nested).
    pub fn from_json(v: &serde_json::Value) -> Result<Self, String> {
        let t = v.get("text_config").unwrap_or(v);
        let u = |k: &str| -> Result<u32, String> {
            t.get(k)
                .and_then(|x| x.as_u64())
                .map(|x| x as u32)
                .ok_or_else(|| format!("config.json: missing {k}"))
        };
        let rope = t.get("rope_parameters").unwrap_or(t);
        let theta = rope
            .get("rope_theta")
            .and_then(|x| x.as_f64())
            .ok_or("config.json: missing rope_theta")? as f32;
        let partial = rope
            .get("partial_rotary_factor")
            .or_else(|| t.get("partial_rotary_factor"))
            .and_then(|x| x.as_f64())
            .unwrap_or(1.0);
        let layer_types = t
            .get("layer_types")
            .and_then(|x| x.as_array())
            .ok_or("config.json: missing layer_types")?;
        let linear = layer_types
            .iter()
            .map(|x| x.as_str() == Some("linear_attention"))
            .collect::<Vec<_>>();
        let head_dim = u("head_dim")?;
        let lin_head_dim = u("linear_key_head_dim")?;
        if u("linear_value_head_dim")? != lin_head_dim || lin_head_dim != 128 {
            return Err("qwen35: linear-attention head dims must be 128".into());
        }
        Ok(Self {
            hidden: u("hidden_size")?,
            layers: u("num_hidden_layers")?,
            linear,
            heads: u("num_attention_heads")?,
            kv_heads: u("num_key_value_heads")?,
            head_dim,
            rot_dim: (head_dim as f64 * partial) as u32,
            rope_theta: theta,
            lin_k_heads: u("linear_num_key_heads")?,
            lin_v_heads: u("linear_num_value_heads")?,
            lin_head_dim,
            conv_kernel: u("linear_conv_kernel_dim")?,
            experts: u("num_experts")?,
            top_k: u("num_experts_per_tok")?,
            moe_ff: u("moe_intermediate_size")?,
            shared_ff: u("shared_expert_intermediate_size")?,
            vocab: u("vocab_size")?,
            eps: t
                .get("rms_norm_eps")
                .and_then(|x| x.as_f64())
                .unwrap_or(1e-6) as f32,
        })
    }

    fn lin_kdim(&self) -> u32 {
        self.lin_k_heads * self.lin_head_dim
    }

    fn lin_vdim(&self) -> u32 {
        self.lin_v_heads * self.lin_head_dim
    }

    /// Conv channels = q + k + v widths.
    fn conv_dim(&self) -> u32 {
        2 * self.lin_kdim() + self.lin_vdim()
    }

    /// KV-cache bytes per token over all full-attention layers (f16 K + V).
    pub fn kv_bytes_per_token(&self) -> u64 {
        let full = self.linear.iter().filter(|l| !**l).count() as u64;
        full * 2 * self.kv_heads as u64 * self.head_dim as u64 * 2
    }

    /// Fixed recurrent-state bytes (conv + delta state) over linear layers.
    pub fn recurrent_state_bytes(&self) -> u64 {
        let lin = self.linear.iter().filter(|l| **l).count() as u64;
        let conv = self.conv_dim() as u64 * (self.conv_kernel as u64 - 1) * 4;
        let rec = self.lin_v_heads as u64 * 128 * 128 * 4;
        lin * (conv + rec)
    }
}

enum Mixer {
    Linear {
        qkv: W,
        z: W,
        alpha: W,
        beta: W,
        a_neg: W,
        dt_bias: W,
        conv: W,
        norm: W,
        out: W,
    },
    Full {
        q: W,
        k: W,
        v: W,
        o: W,
        q_norm: W,
        k_norm: W,
    },
}

impl Mixer {
    fn weights(&self) -> Vec<W> {
        match self {
            Self::Linear {
                qkv,
                z,
                alpha,
                beta,
                a_neg,
                dt_bias,
                conv,
                norm,
                out,
            } => vec![
                *qkv, *z, *alpha, *beta, *a_neg, *dt_bias, *conv, *norm, *out,
            ],
            Self::Full {
                q,
                k,
                v,
                o,
                q_norm,
                k_norm,
            } => vec![*q, *k, *v, *o, *q_norm, *k_norm],
        }
    }
}

struct Layer {
    attn_norm: W,
    post_norm: W,
    mixer: Mixer,
    router: W,
    sh_gate: W,
    sh_up: W,
    sh_down: W,
    sh_inp: W,
}

// Layer view indexes ([`Layer::weights`] order); the mixer's follow at MIX.
const ATTN_NORM: usize = 0;
const POST_NORM: usize = 1;
const ROUTER: usize = 2;
const SH_GATE: usize = 3;
const SH_UP: usize = 4;
const SH_DOWN: usize = 5;
const SH_INP: usize = 6;
const MIX: usize = 7;

impl Layer {
    fn weights(&self) -> Vec<W> {
        let mut v = vec![
            self.attn_norm,
            self.post_norm,
            self.router,
            self.sh_gate,
            self.sh_up,
            self.sh_down,
            self.sh_inp,
        ];
        v.extend(self.mixer.weights());
        v
    }
}

pub struct Qwen35 {
    pub cfg: Qwen35Config,
    dims: Dims,
    layers: Vec<Layer>,
    out_norm: W,
    lm_head: W,
}

pub type Qwen35Model<S> = Decoder<S, Qwen35>;
pub type Qwen35State = LlmState;

impl<S: WeightSource> Decoder<S, Qwen35> {
    /// Register every dense weight (pinned while it fits) and size the routed
    /// expert pool from what the VRAM budget leaves.
    pub fn load(
        backend: &Arc<WgpuBackend>,
        source: S,
        cfg: Qwen35Config,
        budget: ResidencyBudget,
        max_ctx: u32,
    ) -> Result<Self, LlmError> {
        let residency = WeightResidency::new(source, budget);
        let reg = Registry {
            residency: &residency,
        };
        let mut layers = Vec::with_capacity(cfg.layers as usize);
        let mut experts = Vec::with_capacity(cfg.layers as usize);
        for i in 0..cfg.layers as usize {
            let b = |s: &str| format!("blk.{i}.{s}");
            let d = |s: &str| reg.dense(&b(s));
            let mixer = if cfg.linear[i] {
                Mixer::Linear {
                    qkv: d("attn_qkv.weight")?,
                    z: d("attn_gate.weight")?,
                    alpha: d("ssm_alpha.weight")?,
                    beta: d("ssm_beta.weight")?,
                    a_neg: d("ssm_a")?,
                    dt_bias: d("ssm_dt.bias")?,
                    conv: d("ssm_conv1d.weight")?,
                    norm: d("ssm_norm.weight")?,
                    out: d("ssm_out.weight")?,
                }
            } else {
                Mixer::Full {
                    q: d("attn_q.weight")?,
                    k: d("attn_k.weight")?,
                    v: d("attn_v.weight")?,
                    o: d("attn_output.weight")?,
                    q_norm: d("attn_q_norm.weight")?,
                    k_norm: d("attn_k_norm.weight")?,
                }
            };
            layers.push(Layer {
                attn_norm: d("attn_norm.weight")?,
                post_norm: d("post_attention_norm.weight")?,
                mixer,
                router: d("ffn_gate_inp.weight")?,
                sh_gate: d("ffn_gate_shexp.weight")?,
                sh_up: d("ffn_up_shexp.weight")?,
                sh_down: d("ffn_down_shexp.weight")?,
                sh_inp: d("ffn_gate_inp_shexp.weight")?,
            });
            experts.push(ExpertTensors::separate([
                reg.experts(&b("ffn_gate_exps.weight"))?,
                reg.experts(&b("ffn_up_exps.weight"))?,
                reg.experts(&b("ffn_down_exps.weight"))?,
            ]));
        }
        let blocks = Qwen35 {
            dims: Dims {
                hidden: cfg.hidden,
                layers: cfg.layers,
                experts: cfg.experts,
                top_k: cfg.top_k,
                moe_ff: cfg.moe_ff,
                vocab: cfg.vocab,
                eps: cfg.eps,
                gelu: false,
            },
            out_norm: reg.dense("output_norm.weight")?,
            lm_head: reg.dense("output.weight")?,
            layers,
            cfg,
        };
        Decoder::new(backend, residency, blocks, experts, max_ctx)
    }
}

impl Blocks for Qwen35 {
    fn dims(&self) -> &Dims {
        &self.dims
    }

    fn layer_weights(&self, li: usize) -> Vec<W> {
        self.layers[li].weights()
    }

    fn head_weights(&self) -> Vec<W> {
        vec![self.out_norm, self.lm_head]
    }

    fn state_layout(&self, li: usize) -> Vec<StateBuf> {
        let c = &self.cfg;
        if c.linear[li] {
            vec![
                StateBuf::Fixed(c.conv_dim() as u64 * (c.conv_kernel as u64 - 1) * 4),
                StateBuf::Fixed(c.lin_v_heads as u64 * 128 * 128 * 4),
            ]
        } else {
            let row = StateBuf::Rows(c.kv_heads as u64 * c.head_dim as u64 * 2);
            vec![row, row]
        }
    }

    /// NeoX rope table `[rows, rot_dim / 2, (cos, sin)]` for positions `pos0..`.
    fn rope_tables(&self, pos0: u32, rows: u32) -> Vec<Vec<f32>> {
        let half = self.cfg.rot_dim / 2;
        let rd = self.cfg.rot_dim as f32;
        let mut v = Vec::with_capacity((rows * half * 2) as usize);
        for r in 0..rows {
            let p = (pos0 + r) as f32;
            for i in 0..half {
                let inv = 1.0f32 / self.cfg.rope_theta.powf((2 * i) as f32 / rd);
                let a = (p * inv) as f64;
                v.push(a.cos() as f32);
                v.push(a.sin() as f32);
            }
        }
        vec![v]
    }

    /// Token mixer (linear or full attention), post norm, router + top-k and
    /// shared expert of layer `li` on residual `x`.
    fn encode_mixer<'s>(
        &self,
        s: &Scope<'s>,
        p: &LlmPipelines,
        state: &[LayerBuf],
        pos: u32,
        li: usize,
        v: &[GpuView<'_>],
        x: Buf<'s>,
        rope: &[WsBuf<WgpuBackend>],
        t: u32,
        ids_out: Option<Buf<'s>>,
    ) -> Result<Mid<'s>, LlmError> {
        let c = &self.cfg;
        let hd = c.hidden;
        let kk = c.top_k;
        let pairs = t * kk;
        let l = &self.layers[li];
        let wb = |v: &GpuView<'_>| s.import_copy(v.buf());
        let xi = x;
        let h = s.alloc(act_bytes(t, hd))?;
        p.rmsnorm(
            s,
            Act::dense(xi, hd),
            Some(wb(&v[ATTN_NORM])),
            Act::dense(h, hd),
            None,
            t,
            hd,
            c.eps,
        )?;
        let x2 = s.alloc(act_bytes(t, hd))?;
        match &l.mixer {
            Mixer::Linear {
                qkv,
                z,
                alpha,
                beta,
                out,
                ..
            } => {
                let (vd, cd) = (c.lin_vdim(), c.conv_dim());
                let nv = c.lin_v_heads;
                let qkv_o = s.alloc(act_bytes(t, cd))?;
                p.gemv(
                    s,
                    mat(s, &v[MIX], qkv.fmt),
                    Act::dense(h, hd),
                    Act::dense(qkv_o, cd),
                    None,
                    t,
                    cd,
                    hd,
                )?;
                let zo = s.alloc(act_bytes(t, vd))?;
                p.gemv(
                    s,
                    mat(s, &v[MIX + 1], z.fmt),
                    Act::dense(h, hd),
                    Act::dense(zo, vd),
                    None,
                    t,
                    vd,
                    hd,
                )?;
                let ab = s.alloc(act_bytes(t, 2 * nv))?;
                p.gemv(
                    s,
                    mat(s, &v[MIX + 2], alpha.fmt),
                    Act::dense(h, hd),
                    Act::dense(ab, 2 * nv),
                    None,
                    t,
                    nv,
                    hd,
                )?;
                p.gemv(
                    s,
                    mat(s, &v[MIX + 3], beta.fmt),
                    Act::dense(h, hd),
                    Act::dense(ab, 2 * nv).at(nv),
                    None,
                    t,
                    nv,
                    hd,
                )?;
                let conv_st = s.import_copy(state[0].buf.as_buf_ref());
                let qkv_c = s.alloc(act_bytes(t, cd))?;
                p.conv1d_silu(
                    s,
                    qkv_o,
                    wb(&v[MIX + 6]),
                    conv_st,
                    qkv_c,
                    t,
                    cd,
                    c.conv_kernel,
                )?;
                let gb = s.alloc(act_bytes(t, 2 * nv))?;
                p.delta_gates(s, ab, wb(&v[MIX + 4]), wb(&v[MIX + 5]), gb, t, nv)?;
                let rec = s.import_copy(state[1].buf.as_buf_ref());
                let o = s.alloc(act_bytes(t, vd))?;
                p.delta_net(
                    s,
                    qkv_c,
                    gb,
                    zo,
                    wb(&v[MIX + 7]),
                    rec,
                    o,
                    DeltaNetUniform {
                        rows: t,
                        v_heads: nv,
                        k_heads: c.lin_k_heads,
                        conv_width: cd,
                        eps: c.eps,
                        ..Default::default()
                    },
                )?;
                p.gemv(
                    s,
                    mat(s, &v[MIX + 8], out.fmt),
                    Act::dense(o, vd),
                    Act::dense(x2, hd),
                    Some(xi),
                    t,
                    hd,
                    vd,
                )?;
                for b in [xi, h, qkv_o, zo, ab, qkv_c, gb, o] {
                    s.recycle(b);
                }
            }
            Mixer::Full { q, k, v: vw, o, .. } => {
                let (nh, nkv, d) = (c.heads, c.kv_heads, c.head_dim);
                let qw = nh * d * 2;
                let kvw = nkv * d;
                let qg = s.alloc(act_bytes(t, qw))?;
                p.gemv(
                    s,
                    mat(s, &v[MIX], q.fmt),
                    Act::dense(h, hd),
                    Act::dense(qg, qw),
                    None,
                    t,
                    qw,
                    hd,
                )?;
                let ko = s.alloc(act_bytes(t, kvw))?;
                p.gemv(
                    s,
                    mat(s, &v[MIX + 1], k.fmt),
                    Act::dense(h, hd),
                    Act::dense(ko, kvw),
                    None,
                    t,
                    kvw,
                    hd,
                )?;
                let vo = s.alloc(act_bytes(t, kvw))?;
                p.gemv(
                    s,
                    mat(s, &v[MIX + 2], vw.fmt),
                    Act::dense(h, hd),
                    Act::dense(vo, kvw),
                    None,
                    t,
                    kvw,
                    hd,
                )?;
                // Per-head q (first half of each 2*d group) and k norms.
                let qn = s.alloc(act_bytes(t * nh, d))?;
                p.rmsnorm(
                    s,
                    Act {
                        buf: qg,
                        stride: 2 * d,
                        off: 0,
                    },
                    Some(wb(&v[MIX + 4])),
                    Act::dense(qn, d),
                    None,
                    t * nh,
                    d,
                    c.eps,
                )?;
                let kn = s.alloc(act_bytes(t * nkv, d))?;
                p.rmsnorm(
                    s,
                    Act::dense(ko, d),
                    Some(wb(&v[MIX + 5])),
                    Act::dense(kn, d),
                    None,
                    t * nkv,
                    d,
                    c.eps,
                )?;
                let csb = s.import_copy(rope[0].as_buf_ref());
                let half = c.rot_dim / 2;
                p.rope(s, qn, csb, t, nh, nh * d, d, half)?;
                p.rope(s, kn, csb, t, nkv, kvw, d, half)?;
                let kc = s.import_copy(state[0].buf.as_buf_ref());
                let vc = s.import_copy(state[1].buf.as_buf_ref());
                p.kv_store(s, Act::dense(kn, kvw), kc, t, kvw, pos, 0)?;
                p.kv_store(s, Act::dense(vo, kvw), vc, t, kvw, pos, 0)?;
                let ao = s.alloc(act_bytes(t, nh * d))?;
                p.attention(
                    s,
                    Act::dense(qn, nh * d),
                    d,
                    kc,
                    vc,
                    Some((Act::dense(qg, qw).at(d), 2 * d)),
                    Act::dense(ao, nh * d),
                    t,
                    nh,
                    nkv,
                    d,
                    pos,
                    0,
                    0,
                    1.0 / (d as f32).sqrt(),
                )?;
                p.gemv(
                    s,
                    mat(s, &v[MIX + 3], o.fmt),
                    Act::dense(ao, nh * d),
                    Act::dense(x2, hd),
                    Some(xi),
                    t,
                    hd,
                    nh * d,
                )?;
                for b in [xi, h, qg, ko, vo, qn, kn, ao] {
                    s.recycle(b);
                }
            }
        }
        // post-attention norm, router, shared expert
        let hn = s.alloc(act_bytes(t, hd))?;
        p.rmsnorm(
            s,
            Act::dense(x2, hd),
            Some(wb(&v[POST_NORM])),
            Act::dense(hn, hd),
            None,
            t,
            hd,
            c.eps,
        )?;
        let logits = s.alloc(act_bytes(t, c.experts))?;
        p.gemv(
            s,
            mat(s, &v[ROUTER], l.router.fmt),
            Act::dense(hn, hd),
            Act::dense(logits, c.experts),
            None,
            t,
            c.experts,
            hd,
        )?;
        let ids = match ids_out {
            Some(b) => b,
            None => s.alloc(pairs as u64 * 4)?,
        };
        let wts = s.alloc(pairs as u64 * 4)?;
        p.router(s, logits, None, ids, wts, t, c.experts, kk, true)?;
        let sf = c.shared_ff;
        let sgo = s.alloc(act_bytes(t, sf))?;
        let suo = s.alloc(act_bytes(t, sf))?;
        for (w, fmt, y) in [(SH_GATE, l.sh_gate.fmt, sgo), (SH_UP, l.sh_up.fmt, suo)] {
            p.gemv(
                s,
                mat(s, &v[w], fmt),
                Act::dense(hn, hd),
                Act::dense(y, sf),
                None,
                t,
                sf,
                hd,
            )?;
        }
        let sact = s.alloc(act_bytes(t, sf))?;
        p.glu(s, sgo, suo, sact, t * sf, false)?;
        s.recycle(sgo);
        s.recycle(suo);
        let sh = s.alloc(act_bytes(t, hd))?;
        p.gemv(
            s,
            mat(s, &v[SH_DOWN], l.sh_down.fmt),
            Act::dense(sact, sf),
            Act::dense(sh, hd),
            None,
            t,
            hd,
            sf,
        )?;
        s.recycle(sact);
        let sg = s.alloc(act_bytes(t, 1))?;
        p.gemv(
            s,
            mat(s, &v[SH_INP], l.sh_inp.fmt),
            Act::dense(hn, hd),
            Act::dense(sg, 1),
            None,
            t,
            1,
            hd,
        )?;
        Ok(Mid {
            hn,
            ids,
            wts,
            logits,
            combine: Combine {
                x2,
                wts,
                aux: vec![sh, sg],
            },
        })
    }

    /// Layer `li`'s post-attention norm + router on the previous layer's
    /// post-mixer residual: the residual moves little across one mixer, so
    /// most picks carry over.
    fn encode_prediction<'s>(
        &self,
        s: &Scope<'s>,
        p: &LlmPipelines,
        li: usize,
        v: &[GpuView<'_>],
        mid: &Mid<'s>,
        t: u32,
        ids: Buf<'s>,
    ) -> Result<(), LlmError> {
        let c = &self.cfg;
        let hd = c.hidden;
        let hn = s.alloc(act_bytes(t, hd))?;
        p.rmsnorm(
            s,
            Act::dense(mid.combine.x2, hd),
            Some(s.import_copy(v[POST_NORM].buf())),
            Act::dense(hn, hd),
            None,
            t,
            hd,
            c.eps,
        )?;
        let logits = s.alloc(act_bytes(t, c.experts))?;
        p.gemv(
            s,
            mat(s, &v[ROUTER], self.layers[li].router.fmt),
            Act::dense(hn, hd),
            Act::dense(logits, c.experts),
            None,
            t,
            c.experts,
            hd,
        )?;
        let wts = s.alloc(t as u64 * c.top_k as u64 * 4)?;
        p.router(s, logits, None, ids, wts, t, c.experts, c.top_k, true)?;
        for b in [hn, logits, wts] {
            s.recycle(b);
        }
        Ok(())
    }

    /// Weighted routed outputs + sigmoid-gated shared expert + residual.
    fn encode_combine<'s>(
        &self,
        s: &Scope<'s>,
        p: &LlmPipelines,
        _li: usize,
        _v: &[GpuView<'_>],
        c: &Combine<'s>,
        ed: Buf<'s>,
        t: u32,
    ) -> Result<Buf<'s>, LlmError> {
        let (hd, kk) = (self.cfg.hidden, self.cfg.top_k);
        let (sh, sg) = (c.aux[0], c.aux[1]);
        let xn = s.alloc(act_bytes(t, hd))?;
        p.moe_combine(s, ed, c.wts, Some((sh, sg)), Some(c.x2), xn, t, hd, kk)?;
        for b in [sh, sg, c.x2] {
            s.recycle(b);
        }
        Ok(xn)
    }

    fn encode_head<'s>(
        &self,
        s: &Scope<'s>,
        p: &LlmPipelines,
        v: &[GpuView<'_>],
        x: Act<'s>,
    ) -> Result<Buf<'s>, LlmError> {
        let c = &self.cfg;
        let hd = c.hidden;
        let xn = s.alloc(act_bytes(1, hd))?;
        p.rmsnorm(
            s,
            x,
            Some(s.import_copy(v[0].buf())),
            Act::dense(xn, hd),
            None,
            1,
            hd,
            c.eps,
        )?;
        let lg = s.alloc(act_bytes(1, c.vocab))?;
        p.gemv(
            s,
            mat(s, &v[1], self.lm_head.fmt),
            Act::dense(xn, hd),
            Act::dense(lg, c.vocab),
            None,
            1,
            c.vocab,
            hd,
        )?;
        Ok(lg)
    }
}
