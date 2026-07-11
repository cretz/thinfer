//! LTX-2.3 DiT timestep / AdaLN conditioning (`AdaLayerNormSingle`, the 8
//! `*_adaln_single` modules) + the per-block table-add that produces the block's
//! `StreamMod` vectors. Ground truth: `transformer/{adaln,timestep_embedding}.py`
//! + the `get_ada_values` / `get_av_ca_ada_values` table-add in `transformer.py`.
//!
//! Per AdaLN module the chain is (sigma a scalar, B=1):
//! ```text
//! s   = sinusoid(sigma * 1000)            // [256], flip_sin_to_cos, host
//! emb = L2(silu(L1(s)))                    // embedded_timestep [D]
//! ts  = linear(silu(emb)) + bias           // the per-param "timestep" block
//! ```
//! The block then adds its per-block `scale_shift_table` to `ts`
//! (`get_ada_values`: `table[p] + ts[p*D..]`). The output stage uses `emb`
//! directly (`scale_shift_table_out[k] + emb`).
//!
//! The 8 modules: video/audio `adaln_single` (9*D), `prompt_adaln_single` (2*D,
//! on the text KV), `av_ca_{video,audio}_scale_shift` (4*D), and
//! `av_ca_{a2v,v2a}_gate` (1*D). The av-cross gate module reads the CROSS
//! stream's sigma (video gate <- audio sigma, audio gate <- video sigma); for a
//! uniform timestep these coincide. Weights are BF16 in the GGUF (bf16 matmul);
//! F32 acts, rows=1 (cheap; computed once per denoise step, table-add per block).

use thinfer_core::backend::{Backend, BufRef, WgpuBackend, WgpuError};
use thinfer_core::ops::{ActDtype, BcastAddF32, SiluF32};
use thinfer_core::residency::{TransposePolicy, WeightHandle, WeightResidency};
use thinfer_core::weight::{WeightId, WeightSource};
use thinfer_core::workspace::{BatchScope, Workspace};

use super::LtxVariant;
use super::config as dit;
use super::dit::{BlockHandles, DitError, DitPipelines, HostStreamMod};
use crate::common::block::{ActBuf, alloc_act, alloc_matmul_out_buf};
use crate::common::embedders::bcast_add_uniform;
use crate::z_image::text_encoder::{LoadError, register_one};

/// Sinusoidal timestep dim (`PixArtAlphaCombinedTimestepSizeEmbeddings`,
/// `time_proj` num_channels=256).
const SINUSOID_DIM: usize = 256;

type WsBuf = thinfer_core::workspace::WsBuf<WgpuBackend>;

// ---------------------------------------------------------------------------
// Weight handles
// ---------------------------------------------------------------------------

/// One `AdaLayerNormSingle`: the timestep embedder (`linear_1`/`linear_2`) +
/// the final `linear` projecting to `coeff*D`. All BF16 weights, F32 biases.
#[derive(Clone, Copy, Debug)]
pub struct AdalnHandles {
    l1_w: WeightHandle,
    l1_b: WeightHandle,
    l2_w: WeightHandle,
    l2_b: WeightHandle,
    lin_w: WeightHandle,
    lin_b: WeightHandle,
}

#[derive(Clone, Copy, Debug)]
pub struct TimestepHandles {
    pub video: AdalnHandles,
    pub audio: AdalnHandles,
    /// Prompt-AdaLN (continuous-sigma text-KV modulation). `None` on 19B.
    pub prompt_video: Option<AdalnHandles>,
    pub prompt_audio: Option<AdalnHandles>,
    pub av_video_ss: AdalnHandles,
    pub av_audio_ss: AdalnHandles,
    pub av_a2v_gate: AdalnHandles,
    pub av_v2a_gate: AdalnHandles,
}

fn register_adaln<S: WeightSource>(
    residency: &WeightResidency<S>,
    prefix: &str,
) -> Result<AdalnHandles, LoadError> {
    let lin = |s: String| register_one(residency, &WeightId(s), TransposePolicy::Linear2D, None);
    let den = |s: String| register_one(residency, &WeightId(s), TransposePolicy::None, None);
    let emb = format!("{prefix}.emb.timestep_embedder");
    Ok(AdalnHandles {
        l1_w: lin(format!("{emb}.linear_1.weight"))?,
        l1_b: den(format!("{emb}.linear_1.bias"))?,
        l2_w: lin(format!("{emb}.linear_2.weight"))?,
        l2_b: den(format!("{emb}.linear_2.bias"))?,
        lin_w: lin(format!("{prefix}.linear.weight"))?,
        lin_b: den(format!("{prefix}.linear.bias"))?,
    })
}

pub fn register_timestep<S: WeightSource>(
    residency: &WeightResidency<S>,
    variant: LtxVariant,
) -> Result<TimestepHandles, LoadError> {
    let prompt = |p: &str| -> Result<Option<AdalnHandles>, LoadError> {
        Ok(if variant.prompt_adaln {
            Some(register_adaln(residency, p)?)
        } else {
            None
        })
    };
    Ok(TimestepHandles {
        video: register_adaln(residency, "adaln_single")?,
        audio: register_adaln(residency, "audio_adaln_single")?,
        prompt_video: prompt("prompt_adaln_single")?,
        prompt_audio: prompt("audio_prompt_adaln_single")?,
        av_video_ss: register_adaln(residency, "av_ca_video_scale_shift_adaln_single")?,
        av_audio_ss: register_adaln(residency, "av_ca_audio_scale_shift_adaln_single")?,
        av_a2v_gate: register_adaln(residency, "av_ca_a2v_gate_adaln_single")?,
        av_v2a_gate: register_adaln(residency, "av_ca_v2a_gate_adaln_single")?,
    })
}

// ---------------------------------------------------------------------------
// Sinusoidal embedding (host)
// ---------------------------------------------------------------------------

/// `get_timestep_embedding(t, 256, flip_sin_to_cos=True, downscale_freq_shift=0,
/// max_period=10000)`: `emb = [cos(t*w_i), sin(t*w_i)]`, `w_i = exp(-ln(10000)*
/// i/128)`, `i in 0..128`. f32 (the upstream `time_proj` is f32).
fn sinusoid_256(t: f32) -> Vec<f32> {
    let half = SINUSOID_DIM / 2;
    let mut out = vec![0.0f32; SINUSOID_DIM];
    let log_max = (10000.0f64).ln();
    for i in 0..half {
        let w = (-log_max * i as f64 / half as f64).exp();
        let e = t as f64 * w;
        out[i] = e.cos() as f32; // flip_sin_to_cos -> cos first
        out[half + i] = e.sin() as f32;
    }
    out
}

// ---------------------------------------------------------------------------
// On-device AdaLN module forward
// ---------------------------------------------------------------------------

struct AdalnBufs {
    l1_w: BufRef,
    l1_b: BufRef,
    l2_w: BufRef,
    l2_b: BufRef,
    lin_w: BufRef,
    lin_b: BufRef,
}

/// `out = x @ w^T + bias` (bf16 weight, rows>=1), reusing the DiT bf16 matmul.
fn linear_bf16<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    pipes: &DitPipelines,
    x: ActBuf<'wsp>,
    w: BufRef,
    b: BufRef,
    rows: u32,
    n: u32,
    k: u32,
) -> Result<ActBuf<'wsp>, WgpuError> {
    let bp = &pipes.block;
    let pre = alloc_matmul_out_buf(scope, bp, rows * n)?;
    let dims = scope.u32x4_uniform(rows, n, k, 0)?;
    let wv = scope.import_copy(w);
    scope.matmul(
        &pipes.block.matmul_adaln,
        &pipes.block.matmuls.adaln,
        x.data,
        wv,
        dims,
        pre,
        rows,
        n,
    )?;
    let bv = scope.import_copy(b);
    let out = alloc_act(scope, bp, rows, n)?;
    let u = bcast_add_uniform(scope, n)?;
    scope.bcast_add::<BcastAddF32>(&bp.bcast_add, pre, bv, u, out.data, rows * n)?;
    Ok(out)
}

/// One AdaLN module forward (`emb = L2(silu(L1(s)))`, `ts = lin(silu(emb))+b`).
/// `sinusoid` is the on-GPU `[256]` input; `d` is the module's hidden width.
/// Returns `(emb [d], ts [out])` as scope buffers.
fn adaln_forward<'wsp>(
    scope: &BatchScope<'wsp, WgpuBackend>,
    pipes: &DitPipelines,
    sinusoid: ActBuf<'wsp>,
    w: &AdalnBufs,
    d: u32,
    out: u32,
) -> Result<(ActBuf<'wsp>, ActBuf<'wsp>), WgpuError> {
    let bp = &pipes.block;
    let h1 = linear_bf16(
        scope,
        pipes,
        sinusoid,
        w.l1_w,
        w.l1_b,
        1,
        d,
        SINUSOID_DIM as u32,
    )?;
    let s1 = alloc_act(scope, bp, 1, d)?;
    scope.dispatch_op::<SiluF32>(&pipes.silu, &[h1.data], s1.data)?;
    let emb = linear_bf16(scope, pipes, s1, w.l2_w, w.l2_b, 1, d, d)?;
    let s2 = alloc_act(scope, bp, 1, d)?;
    scope.dispatch_op::<SiluF32>(&pipes.silu, &[emb.data], s2.data)?;
    let ts = linear_bf16(scope, pipes, s2, w.lin_w, w.lin_b, 1, out, d)?;
    Ok((emb, ts))
}

// ---------------------------------------------------------------------------
// Shared timestep blocks (computed once per denoise step)
// ---------------------------------------------------------------------------

/// The adaln-linear outputs ("timestep" blocks the block adds its tables to) +
/// the embedded_timestep (for the output stage), per stream. Host f32.
#[derive(Clone, Debug)]
pub struct SharedTimestep {
    pub v_main: Vec<f32>,    // [9*DIM]
    pub v_prompt: Vec<f32>,  // [2*DIM]
    pub v_av_ss: Vec<f32>,   // [4*DIM]
    pub v_av_gate: Vec<f32>, // [1*DIM]
    pub v_emb: Vec<f32>,     // [DIM] embedded_timestep
    pub a_main: Vec<f32>,    // [9*AUDIO_DIM]
    pub a_prompt: Vec<f32>,  // [2*AUDIO_DIM]
    pub a_av_ss: Vec<f32>,   // [4*AUDIO_DIM]
    pub a_av_gate: Vec<f32>, // [1*AUDIO_DIM]
    pub a_emb: Vec<f32>,     // [AUDIO_DIM]
}

/// Compute the adaln modules for `(sigma_video, sigma_audio)` (av-cross gate reads
/// the cross stream's sigma). One GPU submit (rows=1), read back to host. 22B runs
/// 8 modules (9-way main + 2-way prompt + av); 19B runs 6 (6-way main + av, no
/// prompt), selected by `variant`.
#[allow(clippy::too_many_arguments)]
pub async fn compute_shared_timestep<S: WeightSource>(
    backend: &WgpuBackend,
    pipes: &DitPipelines,
    residency: &WeightResidency<S>,
    workspace: &Workspace<WgpuBackend>,
    th: &TimestepHandles,
    sigma_video: f32,
    sigma_audio: f32,
    variant: LtxVariant,
) -> Result<SharedTimestep, DitError<S::Error>> {
    let vd = dit::DIM as u32;
    let ad = dit::AUDIO_DIM as u32;
    let nmod = variant.n_block_mod as u32;
    let mul = dit::TIMESTEP_SCALE_MULTIPLIER;
    // The adaln matmuls run on the shared block pipelines, so their buffers must
    // use the block act dtype (f32 parity path, or f16 when the perf path is on).
    let act = pipes.block.act_dtype;
    let ab = super::dit::act_bytes(act);

    // sinusoids (host -> GPU). av-cross gate uses the CROSS stream's sigma.
    let sin_v = sinusoid_256(sigma_video * mul);
    let sin_a = sinusoid_256(sigma_audio * mul);
    let up = |d: &[f32]| -> Result<_, WgpuError> {
        let b = workspace.alloc(d.len() as u64 * ab)?;
        backend.write_buffer(b.id(), 0, &crate::common::seq::act_upload_bytes(act, d))?;
        Ok(b)
    };
    let sv = up(&sin_v)?;
    let sa = up(&sin_a)?;

    // Acquire every adaln weight (held across the scope).
    macro_rules! acq {
        ($m:expr) => {{
            let m = $m;
            (
                residency.acquire(m.l1_w, backend).await?,
                residency.acquire(m.l1_b, backend).await?,
                residency.acquire(m.l2_w, backend).await?,
                residency.acquire(m.l2_b, backend).await?,
                residency.acquire(m.lin_w, backend).await?,
                residency.acquire(m.lin_b, backend).await?,
            )
        }};
    }
    let v = acq!(&th.video);
    let a = acq!(&th.audio);
    let pv = if let Some(m) = th.prompt_video {
        Some(acq!(&m))
    } else {
        None
    };
    let pa = if let Some(m) = th.prompt_audio {
        Some(acq!(&m))
    } else {
        None
    };
    let avs = acq!(&th.av_video_ss);
    let aas = acq!(&th.av_audio_ss);
    let a2vg = acq!(&th.av_a2v_gate);
    let v2ag = acq!(&th.av_v2a_gate);
    let bufs = |t: &(
        thinfer_core::residency::GpuView,
        thinfer_core::residency::GpuView,
        thinfer_core::residency::GpuView,
        thinfer_core::residency::GpuView,
        thinfer_core::residency::GpuView,
        thinfer_core::residency::GpuView,
    )| AdalnBufs {
        l1_w: t.0.buf(),
        l1_b: t.1.buf(),
        l2_w: t.2.buf(),
        l2_b: t.3.buf(),
        lin_w: t.4.buf(),
        lin_b: t.5.buf(),
    };

    // Output buffers (workspace allocs) for readback (block act dtype).
    let mk = |n: u32| -> Result<_, WgpuError> { workspace.alloc(n as u64 * ab) };
    let v_main_b = mk(nmod * vd)?;
    let v_prompt_b = mk(2 * vd)?;
    let v_av_ss_b = mk(4 * vd)?;
    let v_av_gate_b = mk(vd)?;
    let v_emb_b = mk(vd)?;
    let a_main_b = mk(nmod * ad)?;
    let a_prompt_b = mk(2 * ad)?;
    let a_av_ss_b = mk(4 * ad)?;
    let a_av_gate_b = mk(ad)?;
    let a_emb_b = mk(ad)?;

    {
        let scope = workspace.batch();
        let sv_h = ActBuf::dense(scope.import_copy(sv.as_buf_ref()));
        let sa_h = ActBuf::dense(scope.import_copy(sa.as_buf_ref()));
        macro_rules! copy {
            ($src:expr, $dst:expr, $n:expr) => {{
                let d = scope.import_copy($dst.as_buf_ref());
                scope.copy_buffer_to_buffer($src.data, 0, d, 0, $n as u64 * ab)?;
            }};
        }

        let (v_emb, v_ts) = adaln_forward(&scope, pipes, sv_h, &bufs(&v), vd, nmod * vd)?;
        copy!(v_ts, &v_main_b, nmod * vd);
        copy!(v_emb, &v_emb_b, vd);
        if let Some(pv) = &pv {
            let (_, vp_ts) = adaln_forward(&scope, pipes, sv_h, &bufs(pv), vd, 2 * vd)?;
            copy!(vp_ts, &v_prompt_b, 2 * vd);
        }
        let (_, vss_ts) = adaln_forward(&scope, pipes, sv_h, &bufs(&avs), vd, 4 * vd)?;
        copy!(vss_ts, &v_av_ss_b, 4 * vd);
        // video av gate reads the AUDIO sigma sinusoid.
        let (_, vg_ts) = adaln_forward(&scope, pipes, sa_h, &bufs(&a2vg), vd, vd)?;
        copy!(vg_ts, &v_av_gate_b, vd);

        let (a_emb, a_ts) = adaln_forward(&scope, pipes, sa_h, &bufs(&a), ad, nmod * ad)?;
        copy!(a_ts, &a_main_b, nmod * ad);
        copy!(a_emb, &a_emb_b, ad);
        if let Some(pa) = &pa {
            let (_, ap_ts) = adaln_forward(&scope, pipes, sa_h, &bufs(pa), ad, 2 * ad)?;
            copy!(ap_ts, &a_prompt_b, 2 * ad);
        }
        let (_, ass_ts) = adaln_forward(&scope, pipes, sa_h, &bufs(&aas), ad, 4 * ad)?;
        copy!(ass_ts, &a_av_ss_b, 4 * ad);
        // audio av gate reads the VIDEO sigma sinusoid.
        let (_, ag_ts) = adaln_forward(&scope, pipes, sv_h, &bufs(&v2ag), ad, ad)?;
        copy!(ag_ts, &a_av_gate_b, ad);

        scope.submit_void().await?;
    }

    let prompt_v = if variant.prompt_adaln {
        read_ws(backend, &v_prompt_b, (2 * vd) as usize, act).await?
    } else {
        Vec::new()
    };
    let prompt_a = if variant.prompt_adaln {
        read_ws(backend, &a_prompt_b, (2 * ad) as usize, act).await?
    } else {
        Vec::new()
    };
    Ok(SharedTimestep {
        v_main: read_ws(backend, &v_main_b, (nmod * vd) as usize, act).await?,
        v_prompt: prompt_v,
        v_av_ss: read_ws(backend, &v_av_ss_b, 4 * vd as usize, act).await?,
        v_av_gate: read_ws(backend, &v_av_gate_b, vd as usize, act).await?,
        v_emb: read_ws(backend, &v_emb_b, vd as usize, act).await?,
        a_main: read_ws(backend, &a_main_b, (nmod * ad) as usize, act).await?,
        a_prompt: prompt_a,
        a_av_ss: read_ws(backend, &a_av_ss_b, 4 * ad as usize, act).await?,
        a_av_gate: read_ws(backend, &a_av_gate_b, ad as usize, act).await?,
        a_emb: read_ws(backend, &a_emb_b, ad as usize, act).await?,
    })
}

/// Read a workspace activation buffer (`n` elems, `act` dtype) back to host f32.
async fn read_ws(
    backend: &WgpuBackend,
    b: &WsBuf,
    n: usize,
    act: ActDtype,
) -> Result<Vec<f32>, WgpuError> {
    let bytes = backend
        .read_buffer(b.id(), 0, n as u64 * super::dit::act_bytes(act))
        .await?;
    Ok(crate::common::seq::act_readback_to_f32(act, &bytes, n))
}

// ---------------------------------------------------------------------------
// Per-block table-add (host) -> StreamMod vectors
// ---------------------------------------------------------------------------

/// The 6 per-block AdaLN tables, read back to host (bf16-rounded as stored).
#[derive(Clone, Debug)]
pub struct BlockTables {
    pub scale_shift: Vec<f32>,       // [9*DIM]
    pub audio_scale_shift: Vec<f32>, // [9*AUDIO_DIM]
    pub prompt: Vec<f32>,            // [2*DIM]
    pub audio_prompt: Vec<f32>,      // [2*AUDIO_DIM]
    pub a2v_ca_video: Vec<f32>,      // [5*DIM]
    pub a2v_ca_audio: Vec<f32>,      // [5*AUDIO_DIM]
}

/// Read one block's 6 modulation tables back to host (the F32 GGUF tensors
/// upload bf16, so they read back bf16-rounded -- matching the pyref).
pub async fn read_block_tables<S: WeightSource>(
    backend: &WgpuBackend,
    residency: &WeightResidency<S>,
    handles: &BlockHandles,
    variant: LtxVariant,
) -> Result<BlockTables, DitError<S::Error>> {
    let t = &handles.tables;
    let rd = |h: WeightHandle, n: usize| async move {
        let view = residency.acquire(h, backend).await?;
        let bytes = backend
            .read_buffer(view.buf().id, 0, (n * 2) as u64)
            .await?;
        Ok::<_, DitError<S::Error>>(crate::common::seq::act_readback_to_f32(
            ActDtype::Bf16,
            &bytes,
            n,
        ))
    };
    let d = dit::DIM;
    let ad = dit::AUDIO_DIM;
    let nmod = variant.n_block_mod;
    // Prompt tables exist only on 22B (`prompt_scale_shift` handles are None on 19B).
    let prompt = match t.prompt_scale_shift {
        Some(h) => rd(h, 2 * d).await?,
        None => Vec::new(),
    };
    let audio_prompt = match t.audio_prompt_scale_shift {
        Some(h) => rd(h, 2 * ad).await?,
        None => Vec::new(),
    };
    Ok(BlockTables {
        scale_shift: rd(t.scale_shift, nmod * d).await?,
        audio_scale_shift: rd(t.audio_scale_shift, nmod * ad).await?,
        prompt,
        audio_prompt,
        a2v_ca_video: rd(t.a2v_ca_video, 5 * d).await?,
        a2v_ca_audio: rd(t.a2v_ca_audio, 5 * ad).await?,
    })
}

/// `table[p*D..(p+1)*D] + ts[p*D..]` for param row `p` (one `get_ada_values`
/// param slice).
fn row_add(table: &[f32], ts: &[f32], p: usize, d: usize) -> Vec<f32> {
    let tb = &table[p * d..(p + 1) * d];
    let tv = &ts[p * d..(p + 1) * d];
    tb.iter().zip(tv).map(|(a, b)| a + b).collect()
}

#[allow(clippy::too_many_arguments)]
fn assemble_stream(
    main: &[f32],
    prompt_ts: &[f32],
    av_ss: &[f32],
    av_gate: &[f32],
    table: &[f32],
    prompt_table: &[f32],
    av_table: &[f32],
    d: usize,
    cross_adaln: bool,
) -> HostStreamMod {
    // get_ada_values order per slice: row p -> (shift @0, scale @1, gate @2).
    // 22B: msa(0,1,2) mlp(3,4,5) cross-q(6,7,8) + prompt cross-kv(0,1). 19B: only
    // msa + mlp (6-way table); cross-attn is raw so cq/ckv are unused -> zeros
    // (block_forward never reads them on the 19B path, but the upload path walks
    // all 16 vectors, so keep them d-sized rather than empty).
    let (cq_shift, cq_scale, cq_gate, ckv_shift, ckv_scale) = if cross_adaln {
        (
            row_add(table, main, 6, d),
            row_add(table, main, 7, d),
            row_add(table, main, 8, d),
            row_add(prompt_table, prompt_ts, 0, d),
            row_add(prompt_table, prompt_ts, 1, d),
        )
    } else {
        let z = || vec![0.0f32; d];
        (z(), z(), z(), z(), z())
    };
    HostStreamMod {
        msa_shift: row_add(table, main, 0, d),
        msa_scale: row_add(table, main, 1, d),
        msa_gate: row_add(table, main, 2, d),
        mlp_shift: row_add(table, main, 3, d),
        mlp_scale: row_add(table, main, 4, d),
        mlp_gate: row_add(table, main, 5, d),
        cq_shift,
        cq_scale,
        cq_gate,
        ckv_shift,
        ckv_scale,
        // av_ca scale/shift: rows 0,1 = a2v (scale,shift); 2,3 = v2a (scale,shift).
        a2v_scale: row_add(av_table, av_ss, 0, d),
        a2v_shift: row_add(av_table, av_ss, 1, d),
        v2a_scale: row_add(av_table, av_ss, 2, d),
        v2a_shift: row_add(av_table, av_ss, 3, d),
        // av_ca gate: table row 4 + the 1*D gate timestep.
        av_gate: av_table[4 * d..5 * d]
            .iter()
            .zip(av_gate)
            .map(|(a, b)| a + b)
            .collect(),
    }
}

/// Combine the shared timestep blocks with one block's tables into the block's
/// `(video, audio)` `HostStreamMod` (the `get_ada_values` table-add).
pub fn assemble_block_mod(
    shared: &SharedTimestep,
    tables: &BlockTables,
    variant: LtxVariant,
) -> (HostStreamMod, HostStreamMod) {
    let d = dit::DIM;
    let ad = dit::AUDIO_DIM;
    let ca = variant.cross_adaln;
    let vmod = assemble_stream(
        &shared.v_main,
        &shared.v_prompt,
        &shared.v_av_ss,
        &shared.v_av_gate,
        &tables.scale_shift,
        &tables.prompt,
        &tables.a2v_ca_video,
        d,
        ca,
    );
    let amod = assemble_stream(
        &shared.a_main,
        &shared.a_prompt,
        &shared.a_av_ss,
        &shared.a_av_gate,
        &tables.audio_scale_shift,
        &tables.audio_prompt,
        &tables.a2v_ca_audio,
        ad,
        ca,
    );
    (vmod, amod)
}
