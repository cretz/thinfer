//! Gemma 4 MoE (`gemma4`, e.g. 26B-A4B): sliding-window (local) and global
//! attention layers, each followed by a dense GELU MLP and a routed-expert
//! FFN in parallel, over the GGUF llama.cpp writes (norm weights plain,
//! gate + up experts merged, `ffn_down_exps.scale` = per-expert scale,
//! `ffn_gate_inp.scale` = router input scale). The forward is the shared
//! [`Decoder`]; this file is the architecture's [`Blocks`]. Text only: the
//! vision / audio towers are not loaded.
//!
//! Per layer (HF `Gemma4TextDecoderLayer`):
//! `a = x + post_attn_norm(attn(attn_norm(x)))`;
//! `m1 = post_ffw_norm_1(mlp(ffw_norm(a)))`;
//! `m2 = post_ffw_norm_2(experts(pre_ffw_norm_2(a)))` routed on
//! `rms(a) * router_scale / sqrt(hidden)`;
//! `out = (a + post_ffw_norm(m1 + m2)) * layer_scale`.
//! Attention: q/k RMS-normed per head (weighted), v RMS-normed unweighted,
//! scale 1, global layers share k as v (`attention_k_eq_v`) and rotate a
//! quarter of their 512 dims ("proportional" rope).

use std::sync::Arc;

use thinfer_core::backend::WgpuBackend;
use thinfer_core::policy::ResidencyBudget;
use thinfer_core::residency::{GpuView, WeightResidency};
use thinfer_core::weight::WeightSource;
use thinfer_core::workspace::WsBuf;

use crate::llm::decoder::{
    Blocks, Buf, Combine, Decoder, Dims, LayerBuf, Mid, OwnedBuf, Registry, Scope, StateBuf, W, mat,
};
pub use crate::llm::decoder::{LlmError, LlmState, Output, PhaseTimes, Want};
use crate::llm::experts::ExpertTensors;
use crate::llm::kernels::{Act, LlmPipelines, act_bytes};

#[derive(Clone, Debug)]
pub struct Gemma4Config {
    pub hidden: u32,
    pub layers: u32,
    /// `true` = sliding-window (local) attention, `false` = global.
    pub sliding: Vec<bool>,
    pub heads: u32,
    pub kv_heads: u32,
    pub global_kv_heads: u32,
    pub head_dim: u32,
    pub global_head_dim: u32,
    pub window: u32,
    pub rope_theta: f32,
    pub global_rope_theta: f32,
    /// Rotated fraction of a global head (the rest pass through).
    pub global_rotary: f32,
    pub ff: u32,
    pub experts: u32,
    pub top_k: u32,
    pub moe_ff: u32,
    pub vocab: u32,
    pub eps: f32,
    pub logit_softcap: Option<f32>,
}

impl Gemma4Config {
    /// From the upstream `config.json` (`text_config` when nested).
    pub fn from_json(v: &serde_json::Value) -> Result<Self, String> {
        let t = v.get("text_config").unwrap_or(v);
        let u = |k: &str| -> Result<u32, String> {
            t.get(k)
                .and_then(|x| x.as_u64())
                .map(|x| x as u32)
                .ok_or_else(|| format!("config.json: missing {k}"))
        };
        let f = |v: Option<&serde_json::Value>| v.and_then(|x| x.as_f64());
        let rope = t
            .get("rope_parameters")
            .ok_or("config.json: missing rope_parameters")?;
        let local = rope
            .get("sliding_attention")
            .ok_or("rope: no sliding_attention")?;
        let global = rope
            .get("full_attention")
            .ok_or("rope: no full_attention")?;
        let sliding = t
            .get("layer_types")
            .and_then(|x| x.as_array())
            .ok_or("config.json: missing layer_types")?
            .iter()
            .map(|x| x.as_str() == Some("sliding_attention"))
            .collect::<Vec<_>>();
        if t.get("attention_k_eq_v").and_then(|x| x.as_bool()) != Some(true) {
            return Err("gemma4: only attention_k_eq_v checkpoints are supported".into());
        }
        if u("hidden_size_per_layer_input").unwrap_or(0) != 0 {
            return Err("gemma4: per-layer input embeddings are not supported".into());
        }
        if u("num_kv_shared_layers").unwrap_or(0) != 0 {
            return Err("gemma4: shared-KV layers are not supported".into());
        }
        Ok(Self {
            hidden: u("hidden_size")?,
            layers: u("num_hidden_layers")?,
            sliding,
            heads: u("num_attention_heads")?,
            kv_heads: u("num_key_value_heads")?,
            global_kv_heads: u("num_global_key_value_heads")?,
            head_dim: u("head_dim")?,
            global_head_dim: u("global_head_dim")?,
            window: u("sliding_window")?,
            rope_theta: f(local.get("rope_theta")).ok_or("rope: sliding theta")? as f32,
            global_rope_theta: f(global.get("rope_theta")).ok_or("rope: full theta")? as f32,
            global_rotary: f(global.get("partial_rotary_factor")).unwrap_or(1.0) as f32,
            ff: u("intermediate_size")?,
            experts: u("num_experts")?,
            top_k: u("top_k_experts")?,
            moe_ff: u("moe_intermediate_size")?,
            vocab: u("vocab_size")?,
            eps: f(t.get("rms_norm_eps")).unwrap_or(1e-6) as f32,
            logit_softcap: f(t.get("final_logit_softcapping")).map(|x| x as f32),
        })
    }

    /// `(kv heads, head dim)` of layer `li`.
    fn attn_shape(&self, li: usize) -> (u32, u32) {
        if self.sliding[li] {
            (self.kv_heads, self.head_dim)
        } else {
            (self.global_kv_heads, self.global_head_dim)
        }
    }
}

struct Layer {
    /// `None` on global layers (k doubles as v).
    v: Option<W>,
    /// Registered weights in view order (see the index constants).
    dense: [W; DENSE],
    /// `ffn_gate_inp.scale / sqrt(hidden)`: the router input's norm weight.
    router_in: OwnedBuf,
    /// `post_ffw_norm` weights followed by `layer_output_scale`.
    post_ffw: OwnedBuf,
}

// Layer view indexes; global layers have no V view.
const ATTN_NORM: usize = 0;
const Q: usize = 1;
const K: usize = 2;
const O: usize = 3;
const Q_NORM: usize = 4;
const K_NORM: usize = 5;
const POST_ATTN: usize = 6;
const FFW_NORM: usize = 7;
const GATE: usize = 8;
const UP: usize = 9;
const DOWN: usize = 10;
const POST_FFW_1: usize = 11;
const PRE_FFW_2: usize = 12;
const POST_FFW_2: usize = 13;
const ROUTER: usize = 14;
const EXPERT_SCALE: usize = 15;
const DENSE: usize = 16;
const V: usize = DENSE;

const DENSE_NAMES: [&str; DENSE] = [
    "attn_norm.weight",
    "attn_q.weight",
    "attn_k.weight",
    "attn_output.weight",
    "attn_q_norm.weight",
    "attn_k_norm.weight",
    "post_attention_norm.weight",
    "ffn_norm.weight",
    "ffn_gate.weight",
    "ffn_up.weight",
    "ffn_down.weight",
    "post_ffw_norm_1.weight",
    "pre_ffw_norm_2.weight",
    "post_ffw_norm_2.weight",
    "ffn_gate_inp.weight",
    "ffn_down_exps.scale",
];

pub struct Gemma4 {
    pub cfg: Gemma4Config,
    dims: Dims,
    layers: Vec<Layer>,
    out_norm: W,
    /// Tied: the token embedding matrix.
    lm_head: W,
}

pub type Gemma4Model<S> = Decoder<S, Gemma4>;

impl<S: WeightSource> Decoder<S, Gemma4> {
    /// Register every dense weight (pinned while it fits), fold the router
    /// input scale and the layer output scale into small owned buffers, and
    /// size the routed expert pool from what the VRAM budget leaves.
    pub async fn load(
        backend: &Arc<WgpuBackend>,
        source: S,
        cfg: Gemma4Config,
        budget: ResidencyBudget,
        max_ctx: u32,
    ) -> Result<Self, LlmError> {
        let residency = WeightResidency::new(source, budget);
        let reg = Registry {
            residency: &residency,
        };
        let inv_root = 1.0 / (cfg.hidden as f32).sqrt();
        let mut layers = Vec::with_capacity(cfg.layers as usize);
        let mut experts = Vec::with_capacity(cfg.layers as usize);
        for i in 0..cfg.layers as usize {
            let b = |s: &str| format!("blk.{i}.{s}");
            let mut dense = Vec::with_capacity(DENSE);
            for n in DENSE_NAMES {
                dense.push(reg.dense(&b(n))?);
            }
            let mut router_in = reg.f32_vec(&b("ffn_gate_inp.scale")).await?;
            router_in.iter_mut().for_each(|x| *x *= inv_root);
            let mut post_ffw = reg.f32_vec(&b("post_ffw_norm.weight")).await?;
            post_ffw.extend(reg.f32_vec(&b("layer_output_scale.weight")).await?);
            layers.push(Layer {
                v: match cfg.sliding[i] {
                    true => Some(reg.dense(&b("attn_v.weight"))?),
                    false => None,
                },
                dense: dense.try_into().expect("DENSE weights"),
                router_in: OwnedBuf::f32s(backend, &router_in)?,
                post_ffw: OwnedBuf::f32s(backend, &post_ffw)?,
            });
            experts.push(ExpertTensors::merged_gate_up(
                reg.experts(&b("ffn_gate_up_exps.weight"))?,
                reg.experts(&b("ffn_down_exps.weight"))?,
            ));
        }
        let blocks = Gemma4 {
            dims: Dims {
                hidden: cfg.hidden,
                layers: cfg.layers,
                experts: cfg.experts,
                top_k: cfg.top_k,
                moe_ff: cfg.moe_ff,
                vocab: cfg.vocab,
                eps: cfg.eps,
                gelu: true,
            },
            out_norm: reg.dense("output_norm.weight")?,
            lm_head: reg.dense("token_embd.weight")?,
            layers,
            cfg,
        };
        Decoder::new(backend, residency, blocks, experts, max_ctx)
    }
}

impl Gemma4 {
    /// `y = rms(x) * w (+ res)` over `t` rows of `hidden`.
    #[allow(clippy::too_many_arguments)]
    fn norm<'s>(
        &self,
        s: &Scope<'s>,
        p: &LlmPipelines,
        x: Buf<'s>,
        w: Option<Buf<'s>>,
        res: Option<Buf<'s>>,
        t: u32,
    ) -> Result<Buf<'s>, LlmError> {
        let hd = self.cfg.hidden;
        let y = s.alloc(act_bytes(t, hd))?;
        p.rmsnorm(
            s,
            Act::dense(x, hd),
            w,
            Act::dense(y, hd),
            res,
            t,
            hd,
            self.cfg.eps,
        )?;
        Ok(y)
    }

    /// Router logits + top-k (renormed, per-expert scaled) of layer `li` on
    /// the residual `a`.
    #[allow(clippy::too_many_arguments)]
    fn route<'s>(
        &self,
        s: &Scope<'s>,
        p: &LlmPipelines,
        li: usize,
        v: &[GpuView<'_>],
        a: Buf<'s>,
        t: u32,
        ids: Buf<'s>,
        wts: Buf<'s>,
    ) -> Result<Buf<'s>, LlmError> {
        let c = &self.cfg;
        let l = &self.layers[li];
        let rn = self.norm(s, p, a, Some(s.import_copy(l.router_in.buf_ref())), None, t)?;
        let logits = s.alloc(act_bytes(t, c.experts))?;
        p.gemv(
            s,
            mat(s, &v[ROUTER], l.dense[ROUTER].fmt),
            Act::dense(rn, c.hidden),
            Act::dense(logits, c.experts),
            None,
            t,
            c.experts,
            c.hidden,
        )?;
        s.recycle(rn);
        let scale = Some(s.import_copy(v[EXPERT_SCALE].buf()));
        p.router(s, logits, scale, ids, wts, t, c.experts, c.top_k, true)?;
        Ok(logits)
    }
}

impl Blocks for Gemma4 {
    fn dims(&self) -> &Dims {
        &self.dims
    }

    fn layer_weights(&self, li: usize) -> Vec<W> {
        let l = &self.layers[li];
        let mut w = l.dense.to_vec();
        w.extend(l.v);
        w
    }

    fn head_weights(&self) -> Vec<W> {
        vec![self.out_norm, self.lm_head]
    }

    fn state_layout(&self, li: usize) -> Vec<StateBuf> {
        let (kvh, d) = self.cfg.attn_shape(li);
        let row = kvh as u64 * d as u64 * 2;
        let kv = match self.cfg.sliding[li] {
            true => StateBuf::Window {
                row,
                window: self.cfg.window,
            },
            false => StateBuf::Rows(row),
        };
        vec![kv, kv]
    }

    /// `[sliding, global]` NeoX tables `[rows, head_dim / 2, (cos, sin)]`.
    /// Global heads rotate the first `global_rotary * dim / 2` frequency
    /// pairs (exponents over the full head dim); the rest get angle 0.
    fn rope_tables(&self, pos0: u32, rows: u32) -> Vec<Vec<f32>> {
        let c = &self.cfg;
        let table = |dim: u32, theta: f32, rotated: u32| {
            let half = dim / 2;
            let mut v = Vec::with_capacity((rows * half * 2) as usize);
            for r in 0..rows {
                let p = (pos0 + r) as f32;
                for i in 0..half {
                    let inv = match i < rotated {
                        true => 1.0f32 / theta.powf((2 * i) as f32 / dim as f32),
                        false => 0.0,
                    };
                    let a = (p * inv) as f64;
                    v.push(a.cos() as f32);
                    v.push(a.sin() as f32);
                }
            }
            v
        };
        let gd = c.global_head_dim;
        let rotated = (c.global_rotary * gd as f32) as u32 / 2;
        vec![
            table(c.head_dim, c.rope_theta, c.head_dim / 2),
            table(gd, c.global_rope_theta, rotated),
        ]
    }

    /// `sqrt(hidden)` rounded to bf16, as HF applies it (`embed_scale` cast
    /// to the bf16 weight dtype).
    fn embed_scale(&self) -> f32 {
        half::bf16::from_f32((self.cfg.hidden as f32).sqrt()).to_f32()
    }

    fn logit_softcap(&self) -> Option<f32> {
        self.cfg.logit_softcap
    }

    /// Attention, dense MLP, router + top-k and the experts' input norm of
    /// layer `li` on residual `x`.
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
        let l = &self.layers[li];
        let hd = c.hidden;
        let wb = |v: &GpuView<'_>| s.import_copy(v.buf());
        let fmt = |i: usize| l.dense[i].fmt;
        let (nh, (nkv, d)) = (c.heads, c.attn_shape(li));
        let (qw, kvw) = (nh * d, nkv * d);

        // Attention.
        let h = self.norm(s, p, x, Some(wb(&v[ATTN_NORM])), None, t)?;
        let proj = |i: usize, f, n: u32| -> Result<Buf<'s>, LlmError> {
            let y = s.alloc(act_bytes(t, n))?;
            p.gemv(
                s,
                mat(s, &v[i], f),
                Act::dense(h, hd),
                Act::dense(y, n),
                None,
                t,
                n,
                hd,
            )?;
            Ok(y)
        };
        let qo = proj(Q, fmt(Q), qw)?;
        let ko = proj(K, fmt(K), kvw)?;
        let vo = match l.v {
            Some(w) => proj(V, w.fmt, kvw)?,
            None => ko,
        };
        let head_norm = |src: Buf<'s>, w: Option<Buf<'s>>, heads: u32| {
            let y = s.alloc(act_bytes(t * heads, d))?;
            p.rmsnorm(
                s,
                Act::dense(src, d),
                w,
                Act::dense(y, d),
                None,
                t * heads,
                d,
                c.eps,
            )?;
            Ok::<_, LlmError>(y)
        };
        let qn = head_norm(qo, Some(wb(&v[Q_NORM])), nh)?;
        let kn = head_norm(ko, Some(wb(&v[K_NORM])), nkv)?;
        let vn = head_norm(vo, None, nkv)?;
        let cs = s.import_copy(rope[usize::from(!c.sliding[li])].as_buf_ref());
        p.rope(s, qn, cs, t, nh, qw, d, d / 2)?;
        p.rope(s, kn, cs, t, nkv, kvw, d, d / 2)?;
        let (kc, vc) = (
            s.import_copy(state[0].buf.as_buf_ref()),
            s.import_copy(state[1].buf.as_buf_ref()),
        );
        let ring = state[0].ring;
        p.kv_store(s, Act::dense(kn, kvw), kc, t, kvw, pos, ring)?;
        p.kv_store(s, Act::dense(vn, kvw), vc, t, kvw, pos, ring)?;
        let ao = s.alloc(act_bytes(t, qw))?;
        let window = if c.sliding[li] { c.window } else { 0 };
        p.attention(
            s,
            Act::dense(qn, qw),
            d,
            kc,
            vc,
            None,
            Act::dense(ao, qw),
            t,
            nh,
            nkv,
            d,
            pos,
            window,
            ring,
            1.0,
        )?;
        let o = s.alloc(act_bytes(t, hd))?;
        p.gemv(
            s,
            mat(s, &v[O], fmt(O)),
            Act::dense(ao, qw),
            Act::dense(o, hd),
            None,
            t,
            hd,
            qw,
        )?;
        let a = self.norm(s, p, o, Some(wb(&v[POST_ATTN])), Some(x), t)?;
        for b in [h, qo, ko, qn, kn, vn, ao, o, x] {
            s.recycle(b);
        }
        if l.v.is_some() {
            s.recycle(vo);
        }

        // Dense MLP.
        let n1 = self.norm(s, p, a, Some(wb(&v[FFW_NORM])), None, t)?;
        let ff = c.ff;
        let g = s.alloc(act_bytes(t, ff))?;
        let u = s.alloc(act_bytes(t, ff))?;
        for (i, y) in [(GATE, g), (UP, u)] {
            p.gemv(
                s,
                mat(s, &v[i], fmt(i)),
                Act::dense(n1, hd),
                Act::dense(y, ff),
                None,
                t,
                ff,
                hd,
            )?;
        }
        let act = s.alloc(act_bytes(t, ff))?;
        p.glu(s, g, u, act, t * ff, true)?;
        let mlp = s.alloc(act_bytes(t, hd))?;
        p.gemv(
            s,
            mat(s, &v[DOWN], fmt(DOWN)),
            Act::dense(act, ff),
            Act::dense(mlp, hd),
            None,
            t,
            hd,
            ff,
        )?;
        let m1 = self.norm(s, p, mlp, Some(wb(&v[POST_FFW_1])), None, t)?;
        for b in [n1, g, u, act, mlp] {
            s.recycle(b);
        }

        // Router (on the residual) and the experts' input.
        let pairs = t * c.top_k;
        let ids = match ids_out {
            Some(b) => b,
            None => s.alloc(pairs as u64 * 4)?,
        };
        let wts = s.alloc(pairs as u64 * 4)?;
        let logits = self.route(s, p, li, v, a, t, ids, wts)?;
        let hn = self.norm(s, p, a, Some(wb(&v[PRE_FFW_2])), None, t)?;
        Ok(Mid {
            hn,
            ids,
            wts,
            logits,
            combine: Combine {
                x2: a,
                wts,
                aux: vec![m1],
            },
        })
    }

    /// Layer `li`'s router on the previous layer's post-attention residual.
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
        let wts = s.alloc(t as u64 * self.cfg.top_k as u64 * 4)?;
        let logits = self.route(s, p, li, v, mid.combine.x2, t, ids, wts)?;
        s.recycle(logits);
        s.recycle(wts);
        Ok(())
    }

    /// `out = (a + post_ffw_norm(m1 + post_ffw_norm_2(sum_k w_k e_k))) *
    /// layer_scale`.
    fn encode_combine<'s>(
        &self,
        s: &Scope<'s>,
        p: &LlmPipelines,
        li: usize,
        v: &[GpuView<'_>],
        c: &Combine<'s>,
        ed: Buf<'s>,
        t: u32,
    ) -> Result<Buf<'s>, LlmError> {
        let cfg = &self.cfg;
        let hd = cfg.hidden;
        let moe = s.alloc(act_bytes(t, hd))?;
        p.moe_combine(s, ed, c.wts, None, None, moe, t, hd, cfg.top_k)?;
        let m1 = c.aux[0];
        let sum = self.norm(
            s,
            p,
            moe,
            Some(s.import_copy(v[POST_FFW_2].buf())),
            Some(m1),
            t,
        )?;
        let out = s.alloc(act_bytes(t, hd))?;
        p.rmsnorm_out_scaled(
            s,
            Act::dense(sum, hd),
            s.import_copy(self.layers[li].post_ffw.buf_ref()),
            Act::dense(out, hd),
            Some(c.x2),
            t,
            hd,
            cfg.eps,
        )?;
        for b in [moe, m1, sum, c.x2] {
            s.recycle(b);
        }
        Ok(out)
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
