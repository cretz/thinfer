//! The 12 Hz speech codec: [`CodecDecoder`] (codes -> samples) and
//! [`CodecEncoder`] (samples -> codes, for in-context voice cloning). Both
//! are causal conv stacks around a sliding-window transformer with layer
//! scale; every weight stays f32 (the codec ships and runs in f32). Convs run
//! in `[C, L]` on the bigvgan kernels, the transformers in rows `[T, C]` on
//! the LLM kernels, with host transposes between (small next to the convs).

pub mod decoder;
pub mod encoder;

use std::collections::HashMap;
use std::sync::Arc;

use thinfer_core::Backend;
use thinfer_core::backend::{BindingKind, BindingLayout, BufRef, WgpuBackend, WgpuPipeline};
use thinfer_core::ops::WeightDtype;
use thinfer_core::ops::llm::LlmWeight;
use thinfer_core::residency::{GpuView, WeightHandle, WeightResidency};
use thinfer_core::tensor::StorageEncoding;
use thinfer_core::weight::{WeightId, WeightReader, WeightSource};
use thinfer_core::workspace::{BatchBuf, BatchScope, Workspace};

pub use decoder::CodecDecoder;
pub use encoder::CodecEncoder;

use crate::common::bigvgan::{self as bv, BigVganPipelines, Cl};
use crate::llm::decoder::{LlmError, OwnedBuf, W, acquire, mat};
use crate::llm::kernels::{Act, LlmPipelines, act_bytes};

type Scope<'s> = BatchScope<'s, WgpuBackend>;

/// One scope's worth of conv work on a `[C, L]` activation.
type Stage<'a> = Box<dyn for<'s> Fn(&Scope<'s>, Cl<'s>, &Acq<'_>) -> Result<Cl<'s>, LlmError> + 'a>;

async fn read_f32<S: WeightSource>(src: &S, name: &str) -> Result<Vec<f32>, LlmError> {
    let id = WeightId(name.to_string());
    let e = src
        .catalog()
        .get(&id)
        .cloned()
        .ok_or_else(|| LlmError::Weights(format!("missing tensor {name}")))?;
    if e.encoding != Some(StorageEncoding::F32) {
        return Err(LlmError::Weights(format!("{name}: expected f32")));
    }
    let mut r = src
        .open(&id)
        .await
        .map_err(|e| LlmError::Weights(format!("{e:?}")))?;
    let mut b = vec![0u8; e.size as usize];
    r.read_at(0, &mut b)
        .await
        .map_err(|e| LlmError::Weights(format!("{e:?}")))?;
    Ok(bytemuck::cast_slice(&b).to_vec())
}

/// Register an f32 tensor to stay f32 on the GPU.
fn reg_f32<S: WeightSource>(
    res: &WeightResidency<S>,
    name: &str,
) -> Result<WeightHandle, LlmError> {
    crate::common::loader::register_keep_f32(res, &WeightId(name.to_string()))
        .map_err(|e| LlmError::Weights(format!("{name}: {e:?}")))
}

fn mat_f32<S: WeightSource>(res: &WeightResidency<S>, name: &str) -> Result<W, LlmError> {
    Ok(W {
        h: reg_f32(res, name)?,
        fmt: LlmWeight::F32,
    })
}

/// A causal 1D conv: weight (+ bias) handles and geometry.
struct Conv {
    w: WeightHandle,
    b: Option<WeightHandle>,
    cout: u32,
    k: u32,
    stride: u32,
    dilation: u32,
    groups: u32,
    /// Pad with the edge values instead of zeros.
    replicate: bool,
}

impl Conv {
    #[allow(clippy::too_many_arguments)]
    fn new<S: WeightSource>(
        res: &WeightResidency<S>,
        prefix: &str,
        cout: u32,
        k: u32,
        stride: u32,
        dilation: u32,
        groups: u32,
        bias: bool,
    ) -> Result<Self, LlmError> {
        Ok(Self {
            w: reg_f32(res, &format!("{prefix}.weight"))?,
            b: bias
                .then(|| reg_f32(res, &format!("{prefix}.bias")))
                .transpose()?,
            cout,
            k,
            stride,
            dilation,
            groups,
            replicate: false,
        })
    }

    fn replicate(self) -> Self {
        Self {
            replicate: true,
            ..self
        }
    }
}

/// Kernels the shared op set lacks.
struct Glue {
    ln: WgpuPipeline,
    gelu: WgpuPipeline,
    elu: WgpuPipeline,
    scale_add: WgpuPipeline,
}

/// Uniform of [`GLUE_WGSL`].
fn glue_u(n: u32, c: u32, inner: u32, eps: f32, cs: u32, ps: u32, positions: u32) -> [u32; 8] {
    [n, c, inner, eps.to_bits(), cs, ps, positions, 0]
}

const GLUE_WGSL: &str = r#"
struct U { n: u32, c: u32, inner: u32, eps: f32, cs: u32, ps: u32, positions: u32, _p: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> u: U;

// LayerNorm over the c channels at each position (element (ch, p) at
// ch * cs + p * ps): y = (x - mean) / sqrt(var + eps) * a[ch] + b[ch].
@compute @workgroup_size(64)
fn ln(@builtin(global_invocation_id) g: vec3<u32>) {
    let p = g.x;
    if (p >= u.positions) { return; }
    var m = 0.0;
    for (var ch = 0u; ch < u.c; ch++) { m += x[ch * u.cs + p * u.ps]; }
    m /= f32(u.c);
    var v = 0.0;
    for (var ch = 0u; ch < u.c; ch++) { let d = x[ch * u.cs + p * u.ps] - m; v += d * d; }
    let r = inverseSqrt(v / f32(u.c) + u.eps);
    for (var ch = 0u; ch < u.c; ch++) {
        let i = ch * u.cs + p * u.ps;
        y[i] = (x[i] - m) * r * a[ch] + b[ch];
    }
}

// Exact-erf GELU (erf by Abramowitz-Stegun 7.1.26, |err| < 1.5e-7).
fn erf_(z: f32) -> f32 {
    let s = sign(z);
    let t = 1.0 / (1.0 + 0.3275911 * abs(z));
    let p = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    return s * (1.0 - p * exp(-z * z));
}
@compute @workgroup_size(64)
fn gelu(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = g.y * 65535u * 64u + g.x;
    if (i >= u.n) { return; }
    let v = x[i];
    y[i] = 0.5 * v * (1.0 + erf_(v * 0.7071067811865476));
}

@compute @workgroup_size(64)
fn elu(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = g.y * 65535u * 64u + g.x;
    if (i >= u.n) { return; }
    let v = x[i];
    y[i] = select(exp(v) - 1.0, v, v > 0.0);
}

// y = b + a[channel] * x, channel = (i / inner) % c.
@compute @workgroup_size(64)
fn scale_add(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = g.y * 65535u * 64u + g.x;
    if (i >= u.n) { return; }
    y[i] = b[i] + a[(i / u.inner) % u.c] * x[i];
}
"#;

fn grid(n: u32) -> [u32; 3] {
    let wg = n.div_ceil(64);
    [wg.min(65535), wg.div_ceil(65535), 1]
}

/// The conv-side kernels (f32 weights) + glue, shared by both directions.
struct Convs {
    bv: BigVganPipelines,
    glue: Glue,
    /// Zero bias for bias-free convs (sized for the widest).
    zero: OwnedBuf,
}

impl Convs {
    async fn new(backend: &Arc<WgpuBackend>, label: &str) -> Result<Self, LlmError> {
        use BindingKind::{StorageRead as R, StorageReadWrite as Wr, Uniform as U};
        let layout: Vec<BindingLayout> = [R, R, R, Wr, U]
            .iter()
            .enumerate()
            .map(|(i, &kind)| BindingLayout {
                slot: i as u32,
                kind,
            })
            .collect();
        let pipe = |entry: &'static str| {
            let layout = layout.clone();
            async move {
                backend
                    .create_pipeline(&format!("{label}_{entry}"), GLUE_WGSL, entry, &layout)
                    .await
            }
        };
        Ok(Self {
            bv: BigVganPipelines::compile_with(backend, label, WeightDtype::F32).await?,
            glue: Glue {
                ln: pipe("ln").await?,
                gelu: pipe("gelu").await?,
                elu: pipe("elu").await?,
                scale_add: pipe("scale_add").await?,
            },
            zero: OwnedBuf::f32s(backend, &[0f32; 4096])?,
        })
    }

    fn dispatch<'s>(
        &self,
        s: &Scope<'s>,
        pipe: &WgpuPipeline,
        bufs: [BatchBuf<'s>; 4],
        u: [u32; 8],
        threads: u32,
    ) -> Result<(), LlmError> {
        let ub = s.write_uniform(bytemuck::cast_slice(&u))?;
        let [x, a, b, y] = bufs;
        s.dispatch(pipe, &[x, a, b, y, ub], grid(threads))?;
        Ok(())
    }

    /// Elementwise `pipe` (gelu / elu) of `x` into a new buffer.
    fn unary<'s>(&self, s: &Scope<'s>, pipe: &WgpuPipeline, x: Cl<'s>) -> Result<Cl<'s>, LlmError> {
        let y = s.alloc(x.bytes())?;
        self.dispatch(
            s,
            pipe,
            [x.buf, x.buf, x.buf, y],
            glue_u(x.n(), x.c, x.l, 0.0, 0, 0, 0),
            x.n(),
        )?;
        Ok(Cl { buf: y, ..x })
    }

    /// Causal conv as the reference pads it: the receptive field's overlap
    /// on the left, then just enough on the right for whole output frames
    /// (zeros, or edge values when `replicate`).
    fn conv<'s>(
        &self,
        s: &Scope<'s>,
        c: &Conv,
        x: Cl<'s>,
        a: &Acq<'_>,
    ) -> Result<Cl<'s>, LlmError> {
        let reach = (c.k - 1) * c.dilation + 1;
        let pad = reach - c.stride;
        // n_frames = ceil((L - reach + pad) / stride + 1) - 1; out = n_frames + 1.
        let n_frames = (x.l + pad - reach).div_ceil(c.stride);
        let lout = n_frames + 1;
        let bias = c.b.map_or(self.zero.buf_ref(), |b| a.get(b));
        if c.replicate {
            let extra = n_frames * c.stride + reach - pad - x.l;
            let xp = bv::pad(s, &self.bv, x, pad, extra)?;
            return Ok(bv::conv1d(
                s,
                &self.bv,
                xp,
                a.get(c.w),
                bias,
                c.cout,
                c.k,
                c.stride,
                c.dilation,
                0,
                c.groups,
            )?);
        }
        // Symmetric zero padding covers the right side too (`pad >= extra`
        // for every conv here); keep only the reference's outputs.
        Ok(bv::conv1d_head(
            s,
            &self.bv,
            x,
            a.get(c.w),
            bias,
            c.cout,
            c.k,
            c.stride,
            c.dilation,
            pad,
            c.groups,
            lout,
        )?)
    }

    /// Causal transposed conv (trailing `k - stride` samples dropped).
    fn conv_up<'s>(
        &self,
        s: &Scope<'s>,
        c: &Conv,
        x: Cl<'s>,
        a: &Acq<'_>,
    ) -> Result<Cl<'s>, LlmError> {
        let bias = c.b.map_or(self.zero.buf_ref(), |b| a.get(b));
        // `(l - 1) * stride + k` samples less the trailing `k - stride`.
        Ok(bv::conv_transpose1d_head(
            s,
            &self.bv,
            x,
            a.get(c.w),
            bias,
            c.cout,
            c.k,
            c.stride,
            x.l * c.stride,
        )?)
    }

    /// Upload `x` (`[c, l]`), run each stage in its own scope (the
    /// activation persists between them; a wgpu scope keeps every
    /// intermediate alive until its submit), read the result back.
    #[allow(clippy::too_many_arguments)]
    async fn stages<S: WeightSource>(
        &self,
        backend: &Arc<WgpuBackend>,
        res: &WeightResidency<S>,
        ws: &Workspace<WgpuBackend>,
        convs: &[&Conv],
        x: &[f32],
        (c, l): (u32, u32),
        stages: &[Stage<'_>],
    ) -> Result<(Vec<f32>, u32, u32), LlmError> {
        let acq = Acq::new(res, backend, convs).await?;
        let mut cur = ws.alloc((x.len() * 4) as u64)?;
        backend.write_buffer(cur.id(), 0, bytemuck::cast_slice(x))?;
        let (mut c, mut l) = (c, l);
        for st in stages {
            let s = ws.batch();
            let xi = Cl {
                buf: s.import_copy(cur.as_buf_ref()),
                c,
                l,
            };
            let y = st(&s, xi, &acq)?;
            (c, l) = (y.c, y.l);
            cur = s.submit(y.buf).await?;
        }
        let bytes = backend.read_buffer(cur.id(), 0, (c * l) as u64 * 4).await?;
        Ok((bytemuck::cast_slice::<u8, f32>(&bytes).to_vec(), c, l))
    }
}

/// Acquired conv weights of one stage run (views kept alive across it).
struct Acq<'r> {
    views: HashMap<WeightHandle, GpuView<'r>>,
}

impl<'r> Acq<'r> {
    async fn new<S: WeightSource>(
        res: &'r WeightResidency<S>,
        backend: &Arc<WgpuBackend>,
        convs: &[&Conv],
    ) -> Result<Self, LlmError> {
        let mut views = HashMap::new();
        for c in convs {
            for h in std::iter::once(c.w).chain(c.b) {
                let v = res
                    .acquire(h, &**backend)
                    .await
                    .map_err(|e| LlmError::Weights(format!("{e:?}")))?;
                views.insert(h, v);
            }
        }
        Ok(Self { views })
    }

    fn get(&self, h: WeightHandle) -> BufRef {
        self.views[&h].buf()
    }
}

fn transpose(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut y = vec![0f32; x.len()];
    for r in 0..rows {
        for c in 0..cols {
            y[c * rows + r] = x[r * cols + c];
        }
    }
    y
}

/// NeoX rope `[t, d / 2, (cos, sin)]` for positions `0..t`.
fn rope_rows(t: u32, d: u32, theta: f64) -> Vec<f32> {
    let half = d / 2;
    let mut v = Vec::with_capacity((t * half * 2) as usize);
    for r in 0..t {
        for i in 0..half {
            let a = r as f64 / theta.powf((2 * i) as f64 / d as f64);
            v.push(a.cos() as f32);
            v.push(a.sin() as f32);
        }
    }
    v
}

/// Norm flavour of a codec transformer.
enum Norm {
    /// RMSNorm (weight).
    Rms(OwnedBuf),
    /// LayerNorm (weight, bias).
    Layer(OwnedBuf, OwnedBuf),
}

/// MLP flavour: SwiGLU `down(silu(gate) * up)` or GELU `fc2(gelu(fc1))`.
enum Mlp {
    SwiGlu { gate: W, up: W, down: W },
    Gelu { fc1: W, fc2: W },
}

struct XfLayer {
    q: W,
    k: W,
    v: W,
    o: W,
    mlp: Mlp,
    in_norm: Norm,
    post_norm: Norm,
    attn_scale: OwnedBuf,
    mlp_scale: OwnedBuf,
}

/// Shapes of a codec transformer.
#[derive(Clone, Copy)]
struct XfDims {
    hidden: u32,
    heads: u32,
    head_dim: u32,
    ffn: u32,
    window: u32,
    eps: f32,
    theta: f64,
}

/// A causal sliding-window transformer with layer scale (no qk norm, MHA).
struct Xf {
    layers: Vec<XfLayer>,
    d: XfDims,
}

/// `y = W a` over `t` rows (f32 weight), into a new buffer.
#[allow(clippy::too_many_arguments)]
fn lin<'s>(
    s: &Scope<'s>,
    p: &LlmPipelines,
    v: &GpuView<'_>,
    a: BatchBuf<'s>,
    t: u32,
    k: u32,
    n: u32,
) -> Result<BatchBuf<'s>, LlmError> {
    let y = s.alloc(act_bytes(t, n))?;
    p.gemv(
        s,
        mat(s, v, LlmWeight::F32),
        Act::dense(a, k),
        Act::dense(y, n),
        None,
        t,
        n,
        k,
    )?;
    Ok(y)
}

/// `(weight, bias)` of a linear the transformer runs before / after its
/// layers, and its output width.
struct Lin {
    w: W,
    bias: Option<Vec<f32>>,
    out: u32,
}

impl Xf {
    #[allow(clippy::too_many_arguments)]
    async fn load<S: WeightSource>(
        backend: &Arc<WgpuBackend>,
        res: &WeightResidency<S>,
        prefix: &str,
        n: usize,
        layer_norm: bool,
        swiglu: bool,
        d: XfDims,
    ) -> Result<Self, LlmError> {
        let src = res.source();
        let buf =
            |name: String| async move { OwnedBuf::f32s(backend, &read_f32(src, &name).await?) };
        let norm = |p: String| async move {
            Ok::<_, LlmError>(if layer_norm {
                Norm::Layer(
                    buf(format!("{p}.weight")).await?,
                    buf(format!("{p}.bias")).await?,
                )
            } else {
                Norm::Rms(buf(format!("{p}.weight")).await?)
            })
        };
        let mut layers = Vec::with_capacity(n);
        for i in 0..n {
            let p = format!("{prefix}.layers.{i}");
            let m = |name: &str| mat_f32(res, &format!("{p}.{name}.weight"));
            layers.push(XfLayer {
                q: m("self_attn.q_proj")?,
                k: m("self_attn.k_proj")?,
                v: m("self_attn.v_proj")?,
                o: m("self_attn.o_proj")?,
                mlp: if swiglu {
                    Mlp::SwiGlu {
                        gate: m("mlp.gate_proj")?,
                        up: m("mlp.up_proj")?,
                        down: m("mlp.down_proj")?,
                    }
                } else {
                    Mlp::Gelu {
                        fc1: m("mlp.fc1")?,
                        fc2: m("mlp.fc2")?,
                    }
                },
                in_norm: norm(format!("{p}.input_layernorm")).await?,
                post_norm: norm(format!("{p}.post_attention_layernorm")).await?,
                attn_scale: buf(format!("{p}.self_attn_layer_scale.scale")).await?,
                mlp_scale: buf(format!("{p}.mlp_layer_scale.scale")).await?,
            });
        }
        Ok(Self { layers, d })
    }

    fn norm<'s>(
        &self,
        s: &Scope<'s>,
        cv: &Convs,
        p: &LlmPipelines,
        n: &Norm,
        x: BatchBuf<'s>,
        t: u32,
    ) -> Result<BatchBuf<'s>, LlmError> {
        let hd = self.d.hidden;
        let y = s.alloc(act_bytes(t, hd))?;
        match n {
            Norm::Rms(w) => p.rmsnorm(
                s,
                Act::dense(x, hd),
                Some(s.import_copy(w.buf_ref())),
                Act::dense(y, hd),
                None,
                t,
                hd,
                self.d.eps,
            )?,
            Norm::Layer(w, b) => cv.dispatch(
                s,
                &cv.glue.ln,
                [x, s.import_copy(w.buf_ref()), s.import_copy(b.buf_ref()), y],
                glue_u(t * hd, hd, 1, self.d.eps, 1, hd, t),
                t,
            )?,
        }
        Ok(y)
    }

    /// `rows` (`[t, in]`) through `pre`, the layers, `final_norm` and `post`.
    #[allow(clippy::too_many_arguments)]
    async fn run<S: WeightSource>(
        &self,
        backend: &Arc<WgpuBackend>,
        res: &WeightResidency<S>,
        p: &LlmPipelines,
        cv: &Convs,
        ws: &Workspace<WgpuBackend>,
        rows: &[f32],
        t: u32,
        pre: Option<&Lin>,
        final_norm: Option<&Norm>,
        post: Option<&Lin>,
    ) -> Result<Vec<f32>, LlmError> {
        let (hd, qw, ff) = (self.d.hidden, self.d.heads * self.d.head_dim, self.d.ffn);
        let up = |v: &[f32]| -> Result<_, LlmError> {
            let b = ws.alloc((v.len() * 4) as u64)?;
            backend.write_buffer(b.id(), 0, bytemuck::cast_slice(v))?;
            Ok(b)
        };
        let in_w = (rows.len() / t as usize) as u32;
        let x0 = up(rows)?;
        let rope = up(&rope_rows(t, self.d.head_dim, self.d.theta))?;
        let bias_rows = |l: Option<&Lin>| -> Result<_, LlmError> {
            l.and_then(|l| l.bias.as_ref())
                .map(|b| up(&b.repeat(t as usize)))
                .transpose()
        };
        let (b_pre, b_post) = (bias_rows(pre)?, bias_rows(post)?);
        let kv: Vec<_> = (0..self.layers.len())
            .map(|_| {
                let b = t as u64 * qw as u64 * 2;
                Ok::<_, LlmError>([ws.alloc(b)?, ws.alloc(b)?])
            })
            .collect::<Result<_, _>>()?;
        let mut views: Vec<Vec<GpuView<'_>>> = Vec::new();
        for l in &self.layers {
            let mlp = match &l.mlp {
                Mlp::SwiGlu { gate, up, down } => vec![*gate, *up, *down],
                Mlp::Gelu { fc1, fc2 } => vec![*fc1, *fc2],
            };
            let mut v = Vec::new();
            for w in [l.q, l.k, l.v, l.o].into_iter().chain(mlp) {
                v.push(acquire(res, backend, w).await?);
            }
            views.push(v);
        }
        let vpre = match pre {
            Some(l) => Some(acquire(res, backend, l.w).await?),
            None => None,
        };
        let vpost = match post {
            Some(l) => Some(acquire(res, backend, l.w).await?),
            None => None,
        };
        let s = ws.batch();
        let imp = |b: &OwnedBuf| s.import_copy(b.buf_ref());
        let f32m = LlmWeight::F32;
        let mut x = s.import_copy(x0.as_buf_ref());
        if let (Some(l), Some(v)) = (pre, &vpre) {
            let y = s.alloc(act_bytes(t, l.out))?;
            let b = b_pre.as_ref().map(|b| s.import_copy(b.as_buf_ref()));
            p.gemv(
                &s,
                mat(&s, v, f32m),
                Act::dense(x, in_w),
                Act::dense(y, l.out),
                b,
                t,
                l.out,
                in_w,
            )?;
            x = y;
        }
        let rp = s.import_copy(rope.as_buf_ref());
        let gemv = |v: &GpuView<'_>, a, k, n| lin(&s, p, v, a, t, k, n);
        for (li, l) in self.layers.iter().enumerate() {
            let v = &views[li];
            let h = self.norm(&s, cv, p, &l.in_norm, x, t)?;
            let q = gemv(&v[0], h, hd, qw)?;
            let k = gemv(&v[1], h, hd, qw)?;
            let vv = gemv(&v[2], h, hd, qw)?;
            let half = self.d.head_dim / 2;
            p.rope(&s, q, rp, t, self.d.heads, qw, self.d.head_dim, half)?;
            p.rope(&s, k, rp, t, self.d.heads, qw, self.d.head_dim, half)?;
            let kc = s.import_copy(kv[li][0].as_buf_ref());
            let vc = s.import_copy(kv[li][1].as_buf_ref());
            p.kv_store(&s, Act::dense(k, qw), kc, t, qw, 0, 0)?;
            p.kv_store(&s, Act::dense(vv, qw), vc, t, qw, 0, 0)?;
            let ao = s.alloc(act_bytes(t, qw))?;
            p.attention(
                &s,
                Act::dense(q, qw),
                self.d.head_dim,
                kc,
                vc,
                None,
                Act::dense(ao, qw),
                t,
                self.d.heads,
                self.d.heads,
                self.d.head_dim,
                0,
                self.d.window,
                0,
                1.0 / (self.d.head_dim as f32).sqrt(),
            )?;
            let at = gemv(&v[3], ao, qw, hd)?;
            let x2 = s.alloc(act_bytes(t, hd))?;
            cv.dispatch(
                &s,
                &cv.glue.scale_add,
                [at, imp(&l.attn_scale), x, x2],
                glue_u(t * hd, hd, 1, 0.0, 0, 0, 0),
                t * hd,
            )?;
            let hn = self.norm(&s, cv, p, &l.post_norm, x2, t)?;
            let m = match l.mlp {
                Mlp::SwiGlu { .. } => {
                    let g = gemv(&v[4], hn, hd, ff)?;
                    let u = gemv(&v[5], hn, hd, ff)?;
                    let a = s.alloc(act_bytes(t, ff))?;
                    p.glu(&s, g, u, a, t * ff, false)?;
                    gemv(&v[6], a, ff, hd)?
                }
                Mlp::Gelu { .. } => {
                    let h1 = gemv(&v[4], hn, hd, ff)?;
                    let a = cv.unary(
                        &s,
                        &cv.glue.gelu,
                        Cl {
                            buf: h1,
                            c: t,
                            l: ff,
                        },
                    )?;
                    gemv(&v[5], a.buf, ff, hd)?
                }
            };
            let x3 = s.alloc(act_bytes(t, hd))?;
            cv.dispatch(
                &s,
                &cv.glue.scale_add,
                [m, imp(&l.mlp_scale), x2, x3],
                glue_u(t * hd, hd, 1, 0.0, 0, 0, 0),
                t * hd,
            )?;
            x = x3;
        }
        if let Some(n) = final_norm {
            x = self.norm(&s, cv, p, n, x, t)?;
        }
        let mut width = hd;
        if let (Some(l), Some(v)) = (post, &vpost) {
            let y = s.alloc(act_bytes(t, l.out))?;
            let b = b_post.as_ref().map(|b| s.import_copy(b.as_buf_ref()));
            p.gemv(
                &s,
                mat(&s, v, f32m),
                Act::dense(x, hd),
                Act::dense(y, l.out),
                b,
                t,
                l.out,
                hd,
            )?;
            x = y;
            width = l.out;
        }
        let out = s.submit(x).await?;
        let bytes = backend
            .read_buffer(out.id(), 0, act_bytes(t, width))
            .await?;
        Ok(bytemuck::cast_slice::<u8, f32>(&bytes).to_vec())
    }
}
