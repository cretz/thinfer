//! HunyuanVideo 1.5 causal AR I2V DiT (minWM WorldPlay `HY15/TI2V/dmd`).
//! Ground truth: `minWM/HY15/hy15_inference.py::run_inference_rollout` +
//! `hyvideo/models/transformers/worldplay_1_5_transformer.py::{forward_txt,
//! forward_vision}`.
//!
//! Same 54-block dual-stream MMDiT as the parent T2V module, run causally:
//! - **Text pass (once, t=0)**: the txt-side stream `[vision_in(SigLIP)+cond2 ;
//!   refiner(text)+cond0]` runs through all blocks with txt-only self-attention;
//!   each block's post-qknorm K/V are cached. ByT5 is omitted (upstream drops
//!   zero-masked tokens before the block loop, and `byt5_model=None` is a
//!   sanctioned upstream path), matching the T2V port's validated policy.
//! - **Chunk forwards**: each chunk of `CHUNK_LATENT_FRAMES` latent frames runs
//!   img-only block passes; per block the SDPA reads `kv = [txt ; committed
//!   vision ; this chunk]` (dense, no mask: causality is the cache contents,
//!   the LongLive scheme). RoPE at ABSOLUTE frame positions. 4 flow-match Euler
//!   steps (shift 5.0) + one t=0 recache forward that commits the chunk's
//!   post-rope K/V to the cache.
//! - **KV cache** is host-staged (54 layers x K+V x bf16 = ~442KB/latent-token;
//!   a 77-frame clip is ~14GB -- it cannot live in 8GB VRAM). Per block per
//!   forward the prefix is host-written into a ping-pong pair of VRAM buffers
//!   (prefetched alongside the next block's weights), the chunk K/V are
//!   appended in-scope, and only the recache pass reads chunk K/V back.
//!
//! I2V conditioning rides the `img_in` concat channels: `[noisy32 | cond32 |
//! mask1]` with the VAE-encoded first frame + mask=1 on (global) frame 0 only.

use super::*;
use crate::hunyuan::config::ar as arcfg;
use thinfer_core::ops::BcastMulF32;

/// Per-block host-staged KV prefix: raw act-encoded (bf16) rows `[rows, dim]`
/// in cache order `[txt ; committed vision]`. `rows` tracks both (k and v are
/// always appended together).
struct BlockKv {
    k: Vec<u8>,
    v: Vec<u8>,
    rows: usize,
}

/// SigLIP `vision_in` projection weights: `LN(1152) -> Linear(1152,1152) ->
/// GELU -> Linear(1152,2048) -> LN(2048)` (`VisionProjection`).
struct VisW {
    ln0: LinW,
    lin1: LinW,
    lin3: LinW,
    ln4: LinW,
}

impl VisW {
    fn new() -> Self {
        Self {
            ln0: lin("vision_in.proj.0"),
            lin1: lin("vision_in.proj.1"),
            lin3: lin("vision_in.proj.3"),
            ln4: lin("vision_in.proj.4"),
        }
    }
}

struct VisH {
    ln0: LinH,
    lin1: LinH,
    lin3: LinH,
    ln4: LinH,
}

struct VisBufs {
    ln0: LinBufs,
    lin1: LinBufs,
    lin3: LinBufs,
    ln4: LinBufs,
}

/// LN params registered passthrough (weight [dim] must NOT go through the
/// linear transpose path).
fn reg_ln<S: WeightSource>(res: &WeightResidency<S>, w: &LinW) -> Result<LinH, LoadError> {
    Ok(LinH {
        weight: register_passthrough(res, &w.weight)?,
        bias: register_passthrough(res, &w.bias)?,
    })
}

/// Affine LayerNorm `LN(x)*w[c]+b[c]` composed from the shared block kernels
/// (mirrors `refiner::affine_ln`; `BcastMulF32` is bf16-weight-aware).
fn affine_ln<'w>(
    scope: &BatchScope<'w, WgpuBackend>,
    bp: &BlockPipelines,
    x: BatchBuf<'w>,
    w: &LinBufs,
    rows: u32,
    dim: u32,
) -> Result<BatchBuf<'w>, WgpuError> {
    let normed = scope.alloc(bp.act_bytes(rows * dim))?;
    let lu = scope.u32x4_uniform(rows, dim, LN_EPS.to_bits(), 0)?;
    scope.layernorm::<LayerNormF32>(&bp.layernorm, x, lu, normed, rows)?;
    let scaled = scope.alloc(bp.act_bytes(rows * dim))?;
    let au = scope.u32x4_uniform(dim, 0, 0, 0)?;
    let wv = scope.import_copy(w.weight);
    scope.bcast_add::<BcastMulF32>(&bp.bcast_mul, normed, wv, au, scaled, rows * dim)?;
    let out = scope.alloc(bp.act_bytes(rows * dim))?;
    let bu = scope.u32x4_uniform(dim, 0, 0, 0)?;
    let bv = scope.import_copy(w.bias);
    scope.bcast_add::<BcastAddF32>(&bp.bcast_add, scaled, bv, bu, out, rows * dim)?;
    Ok(out)
}

/// `cond_type_embedding` row `r` of the `[3, dim]` bf16 buffer as a `[dim]` view.
fn cond_row(cond: BufRef, r: u64, dim: u32, asz: u64) -> BufRef {
    BufRef::view(
        cond.id,
        cond.offset + r * dim as u64 * asz,
        dim as u64 * asz,
    )
}

/// `THINFER_AR_DIAG=1`: per-stage host stats (nonfinite counts / std) to
/// localize a divergence without a debugger. Zero cost when unset.
fn ar_diag() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("THINFER_AR_DIAG").is_ok_and(|v| v != "0"))
}

/// Decode raw bf16 act bytes to f32 (diag only).
fn bf16_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
        .collect()
}

fn diag_stats(label: &str, x: &[f32]) {
    let nonfinite = x.iter().filter(|v| !v.is_finite()).count();
    let n = x.len().max(1) as f64;
    let mean = x
        .iter()
        .filter(|v| v.is_finite())
        .map(|v| *v as f64)
        .sum::<f64>()
        / n;
    let var = x
        .iter()
        .filter(|v| v.is_finite())
        .map(|v| (*v as f64 - mean).powi(2))
        .sum::<f64>()
        / n;
    eprintln!(
        "[ar diag] {label}: n={} nonfinite={nonfinite} mean={mean:.4e} std={:.4e}",
        x.len(),
        var.sqrt()
    );
}

/// Estimated host bytes for the full KV cache of a generation (the app layer
/// gates frame counts on this against the RAM budget).
pub fn kv_cache_bytes(grid: (usize, usize, usize), txt_seq: usize) -> u64 {
    let (t, h, w) = grid;
    let tokens = (arcfg::VISION_TOKENS + txt_seq + t * h * w) as u64;
    let row = cfg::HIDDEN as u64 * 2; // bf16
    cfg::DOUBLE_BLOCKS as u64 * 2 * tokens * row
}

pub struct HunyuanArDit {
    pub pipelines: HunyuanDitPipelines,
    refiner: HunyuanRefiner,
    rope: RopeEmbedder,
    handles: DitH,
    vis: VisH,
}

impl HunyuanArDit {
    /// `i8` MUST match [`HunyuanDitPipelines::compile_with`] (same contract as
    /// the parent T2V [`HunyuanDit::new`]).
    pub fn new<S: WeightSource>(
        pipelines: HunyuanDitPipelines,
        refiner: HunyuanRefiner,
        residency: &WeightResidency<S>,
        i8: bool,
    ) -> Result<Self, LoadError> {
        let coopmat = pipelines.bp.coopmat_ffn_down.is_some();
        let vw = VisW::new();
        let vis = VisH {
            ln0: reg_ln(residency, &vw.ln0)?,
            lin1: LinH {
                weight: register_linear(residency, &vw.lin1.weight)?,
                bias: register_passthrough(residency, &vw.lin1.bias)?,
            },
            lin3: LinH {
                weight: register_linear(residency, &vw.lin3.weight)?,
                bias: register_passthrough(residency, &vw.lin3.bias)?,
            },
            ln4: reg_ln(residency, &vw.ln4)?,
        };
        Ok(Self {
            pipelines,
            refiner,
            rope: RopeEmbedder::new(cfg::ROPE_THETA as f32, cfg::ROPE_DIM, [1024, 1024, 1024]),
            handles: DitH::register(residency, i8, coopmat)?,
            vis,
        })
    }

    async fn acquire_vis<'r, S: WeightSource>(
        &self,
        res: &'r WeightResidency<S>,
        backend: &WgpuBackend,
        pins: &mut Vec<GpuView<'r>>,
    ) -> Result<VisBufs, ResidencyError<S::Error, WgpuError>> {
        Ok(VisBufs {
            ln0: acq_lin(res, backend, self.vis.ln0, pins).await?,
            lin1: acq_lin(res, backend, self.vis.lin1, pins).await?,
            lin3: acq_lin(res, backend, self.vis.lin3, pins).await?,
            ln4: acq_lin(res, backend, self.vis.ln4, pins).await?,
        })
    }

    /// RoPE ids for latent frames `[t0, t0+tn)` over the full `(h, w)` grid --
    /// the chunk's ABSOLUTE positions (`start_rope_start_idx` upstream).
    fn chunk_grid_ids(t0: usize, tn: usize, h: usize, w: usize) -> Vec<i32> {
        let mut ids = Vec::with_capacity(tn * h * w * 3);
        for ti in t0..t0 + tn {
            for hi in 0..h {
                for wi in 0..w {
                    ids.push(ti as i32);
                    ids.push(hi as i32);
                    ids.push(wi as i32);
                }
            }
        }
        ids
    }

    /// Text pass: build the txt-side stream `[vision(729) ; txt(seq)]`, run it
    /// through all 54 blocks (txt-only attention, t=0 modulation), and return
    /// the per-block post-qknorm K/V as the initial host cache. `vision = None`
    /// is the upstream `mask_type="t2v"` path: zero-masked vision tokens are
    /// dropped before the block loop, so the stream is txt-only (a PROBE mode;
    /// the dmd checkpoint was trained i2v).
    #[allow(clippy::too_many_arguments)]
    async fn build_txt_cache<S: WeightSource>(
        &self,
        backend: &WgpuBackend,
        residency: &WeightResidency<S>,
        ws: &Workspace<WgpuBackend>,
        text: &[f32],
        seq: usize,
        vision: Option<&[f32]>,
    ) -> Result<Vec<BlockKv>, HunyuanDitError<S::Error>> {
        if let Some(v) = vision {
            assert_eq!(
                v.len(),
                arcfg::VISION_TOKENS * arcfg::VISION_DIM,
                "vision tokens"
            );
        }
        let act = self.pipelines.act();
        let asz = self.pipelines.act_size();
        let dim = cfg::HIDDEN as u32;
        let heads = cfg::HEADS as u32;
        let hd = cfg::HEAD_DIM as u32;
        let mlp_h = cfg::MLP_HIDDEN as u32;
        let vtok = if vision.is_some() {
            arcfg::VISION_TOKENS as u32
        } else {
            0
        };
        let vdim = arcfg::VISION_DIM as u32;
        let s = vtok + seq as u32;

        // Refined text tokens at t=0 (host round-trip, as the parent forward).
        let txt = self
            .refiner
            .refine(backend, residency, ws, text, seq, 0.0, None)
            .await
            .map_err(|e| match e {
                crate::hunyuan::refiner::HunyuanRefinerError::Wgpu(e) => HunyuanDitError::Wgpu(e),
                crate::hunyuan::refiner::HunyuanRefinerError::Load(e) => HunyuanDitError::Load(e),
                crate::hunyuan::refiner::HunyuanRefinerError::Residency(e) => {
                    HunyuanDitError::Residency(e)
                }
            })?;

        let upload = |slice: &[f32]| -> Result<WsBuf<WgpuBackend>, WgpuError> {
            let bytes = act_upload_bytes(act, slice);
            let buf = ws.alloc(bytes.len() as u64)?;
            backend.write_buffer(buf.id(), 0, &bytes)?;
            Ok(buf)
        };
        let vis_up = match vision {
            Some(v) => Some(upload(v)?),
            None => None,
        };
        let txt_up = upload(&txt)?;
        let tsin_up = upload(&timestep_sinusoid(0.0))?;

        let mut pins: Vec<GpuView> = Vec::new();
        let top = acquire_top(&self.handles.top, residency, backend, &mut pins).await?;
        let vis_bufs = match vision {
            Some(_) => Some(self.acquire_vis(residency, backend, &mut pins).await?),
            None => None,
        };

        // Setup: silu(vec_txt at t=0); stream = [vision_in(vis)+cond2 ; txt+cond0].
        let stream_ws = ws.alloc((s * dim) as u64 * asz)?;
        let silu_vec_ws = ws.alloc(dim as u64 * asz)?;
        {
            let scope = ws.batch();
            let bp = &self.pipelines.bp;
            let ts = scope.import_copy(tsin_up.as_buf_ref());
            let h0 = linear(
                &scope,
                bp,
                ts,
                &top.time0,
                1,
                dim,
                FREQ_DIM as u32,
                Site::Module,
            )?;
            let h0a = silu(&scope, bp, h0, dim)?;
            let vec = linear(&scope, bp, h0a, &top.time2, 1, dim, dim, Site::Module)?;
            let sv = silu(&scope, bp, vec, dim)?;
            let dst = scope.import_copy(silu_vec_ws.as_buf_ref());
            scope.copy_buffer_to_buffer(sv, 0, dst, 0, dim as u64 * asz)?;

            // vision_in: LN -> Linear -> GELU -> Linear -> LN, then + cond_type[2].
            let dstream = scope.import_copy(stream_ws.as_buf_ref());
            if let (Some(vis_up), Some(vis_bufs)) = (&vis_up, &vis_bufs) {
                let vx = scope.import_copy(vis_up.as_buf_ref());
                let v0 = affine_ln(&scope, bp, vx, &vis_bufs.ln0, vtok, vdim)?;
                let v1 = linear(
                    &scope,
                    bp,
                    v0,
                    &vis_bufs.lin1,
                    vtok,
                    vdim,
                    vdim,
                    Site::Module,
                )?;
                let vg = scope.alloc(bp.act_bytes(vtok * vdim))?;
                scope.dispatch_op::<GeluF32>(&bp.gelu, &[v1], vg)?;
                let v3 = linear(
                    &scope,
                    bp,
                    vg,
                    &vis_bufs.lin3,
                    vtok,
                    dim,
                    vdim,
                    Site::Module,
                )?;
                let v4 = affine_ln(&scope, bp, v3, &vis_bufs.ln4, vtok, dim)?;
                // + cond_type rows: compute into scope allocs, then copy into
                // the stream regions (ops write whole scope buffers, not views).
                let cu = scope.u32x4_uniform(dim, 0, 0, 0)?;
                let cond2 = scope.import_copy(cond_row(top.cond_type, 2, dim, asz));
                let vis_out = scope.alloc(bp.act_bytes(vtok * dim))?;
                scope.bcast_add::<BcastAddF32>(
                    &bp.bcast_add,
                    v4,
                    cond2,
                    cu,
                    vis_out,
                    vtok * dim,
                )?;
                scope.copy_buffer_to_buffer(vis_out, 0, dstream, 0, (vtok * dim) as u64 * asz)?;
            }

            // txt + cond_type[0] into rows [vtok, vtok+seq).
            let tx = scope.import_copy(txt_up.as_buf_ref());
            let cu2 = scope.u32x4_uniform(dim, 0, 0, 0)?;
            let cond0 = scope.import_copy(cond_row(top.cond_type, 0, dim, asz));
            let txt_out = scope.alloc(bp.act_bytes(seq as u32 * dim))?;
            scope.bcast_add::<BcastAddF32>(
                &bp.bcast_add,
                tx,
                cond0,
                cu2,
                txt_out,
                seq as u32 * dim,
            )?;
            scope.copy_buffer_to_buffer(
                txt_out,
                0,
                dstream,
                (vtok * dim) as u64 * asz,
                (seq as u32 * dim) as u64 * asz,
            )?;
            scope.submit_void().await?;
        }

        // Block loop: txt-only attention; cache each block's post-qknorm K/V.
        let bp = &self.pipelines.bp;
        let nblocks = self.handles.blocks.len();
        let mut cache: Vec<BlockKv> = Vec::with_capacity(nblocks);
        let mut cur = stream_ws;
        let mut cur_pins: Vec<GpuView> = Vec::new();
        let mut cur_b =
            acquire_block(&self.handles.blocks[0], residency, backend, &mut cur_pins).await?;
        for bi in 0..nblocks {
            let nxt = ws.alloc((s * dim) as u64 * asz)?;
            let mut next_pins: Vec<GpuView> = Vec::new();
            let compute = async {
                let (k_ws, v_ws);
                #[allow(clippy::type_complexity)]
                let diag_taps: Option<(
                    WsBuf<WgpuBackend>,
                    WsBuf<WgpuBackend>,
                    WsBuf<WgpuBackend>,
                    WsBuf<WgpuBackend>,
                )>;
                {
                    let scope = ws.batch();
                    let sv = scope.import_copy(silu_vec_ws.as_buf_ref());
                    let tmodp = linear(
                        &scope,
                        bp,
                        sv,
                        &cur_b.txt_mod,
                        1,
                        6 * dim,
                        dim,
                        Site::Module,
                    )?;
                    let tsig = |k| mod_sig(&scope, bp, tmodp, k, dim);
                    let (sh1, sc1, g1) = (tsig(0)?, tsig(1)?, tsig(2)?);
                    let (sh2, sc2, g2) = (tsig(3)?, tsig(4)?, tsig(5)?);
                    let x = scope.import_copy(cur.as_buf_ref());
                    let tm = norm_modulate(&scope, bp, x, sc1, sh1, s, dim)?;
                    let tm_a = qkv_a_side(&scope, bp, tm, s, dim)?;
                    let tq = linear_a(&scope, bp, tm_a, &cur_b.txt_q, s, dim, dim, Site::QkvSelf)?;
                    let tq = qk_norm_rope(&scope, bp, tq, cur_b.txt_qn, None, s, heads, hd)?;
                    let tk = linear_a(&scope, bp, tm_a, &cur_b.txt_k, s, dim, dim, Site::QkvSelf)?;
                    let tk = qk_norm_rope(&scope, bp, tk, cur_b.txt_kn, None, s, heads, hd)?;
                    let tv = linear_a(&scope, bp, tm_a, &cur_b.txt_v, s, dim, dim, Site::QkvSelf)?;
                    k_ws = persist(&scope, ws, tk, (s * dim) as usize, asz)?;
                    v_ws = persist(&scope, ws, tv, (s * dim) as usize, asz)?;
                    let sa = attention(&scope, bp, tq, tk, tv, s, 0, 1, 0, heads, hd)?;
                    let tp = linear(&scope, bp, sa, &cur_b.txt_proj, s, dim, dim, Site::Proj)?;
                    let x1 = gate_residual(&scope, bp, x, g1, tp, s, dim)?;
                    let tm2 = norm_modulate(&scope, bp, x1, sc2, sh2, s, dim)?;
                    let tmlp = mlp(
                        &scope,
                        bp,
                        tm2,
                        &cur_b.txt_fc1,
                        &cur_b.txt_fc2,
                        s,
                        dim,
                        mlp_h,
                    )?;
                    let x2 = gate_residual(&scope, bp, x1, g2, tmlp, s, dim)?;
                    let d = scope.import_copy(nxt.as_buf_ref());
                    scope.copy_buffer_to_buffer(x2, 0, d, 0, (s * dim) as u64 * asz)?;
                    // Block-0 sublayer taps (diag): localize which op explodes.
                    diag_taps = if ar_diag() && bi == 0 {
                        Some((
                            persist(&scope, ws, tq, (s * dim) as usize, asz)?,
                            persist(&scope, ws, tv, (s * dim) as usize, asz)?,
                            persist(&scope, ws, sa, (s * dim) as usize, asz)?,
                            persist(&scope, ws, tmlp, (s * dim) as usize, asz)?,
                        ))
                    } else {
                        None
                    };
                    scope.submit_void().await?;
                }
                let kb = backend
                    .read_buffer(k_ws.id(), 0, (s * dim) as u64 * asz)
                    .await?;
                let vb = backend
                    .read_buffer(v_ws.id(), 0, (s * dim) as u64 * asz)
                    .await?;
                if let Some((tq, tv, sa, tmlp)) = diag_taps {
                    for (label, buf) in [("tq", tq), ("tv", tv), ("sa", sa), ("mlp", tmlp)] {
                        let bytes = backend
                            .read_buffer(buf.id(), 0, (s * dim) as u64 * asz)
                            .await?;
                        diag_stats(&format!("txt block0 {label}"), &bf16_to_f32(&bytes));
                    }
                }
                Ok::<BlockKv, WgpuError>(BlockKv {
                    k: kb,
                    v: vb,
                    rows: s as usize,
                })
            };
            let prefetch = async {
                if bi + 1 < nblocks {
                    acquire_block(
                        &self.handles.blocks[bi + 1],
                        residency,
                        backend,
                        &mut next_pins,
                    )
                    .await
                    .map(Some)
                } else {
                    Ok(None)
                }
            };
            let (c_res, p_res) = futures::join!(compute, prefetch);
            cache.push(c_res?);
            if ar_diag() {
                diag_stats(&format!("txt block {bi} k"), &bf16_to_f32(&cache[bi].k));
            }
            let next_b = p_res?;
            drop(cur_pins);
            cur = nxt;
            cur_pins = next_pins;
            if let Some(b) = next_b {
                cur_b = b;
            }
        }
        drop(pins);
        Ok(cache)
    }

    /// One chunk forward: img-only block passes over the cached KV prefix.
    /// `commit=false` returns the velocity `[rows, 32]`; `commit=true` appends
    /// the chunk's post-rope K/V to `cache` and returns `None`.
    #[allow(clippy::too_many_arguments)]
    async fn forward_chunk<S: WeightSource>(
        &self,
        backend: &WgpuBackend,
        residency: &WeightResidency<S>,
        ws: &Workspace<WgpuBackend>,
        img_tokens: &[f32],
        chunk_t: usize,
        start_t: usize,
        gh: usize,
        gw: usize,
        t: f32,
        cache: &mut [BlockKv],
        commit: bool,
        kv_ping: &[(WsBuf<WgpuBackend>, WsBuf<WgpuBackend>); 2],
    ) -> Result<Option<Vec<f32>>, HunyuanDitError<S::Error>> {
        let act = self.pipelines.act();
        let asz = self.pipelines.act_size();
        let dim = cfg::HIDDEN as u32;
        let heads = cfg::HEADS as u32;
        let hd = cfg::HEAD_DIM as u32;
        let mlp_h = cfg::MLP_HIDDEN as u32;
        let latent = cfg::LATENT_CHANNELS as u32;
        let conv_in = cfg::CONV_IN_CHANNELS as u32;
        let rows = (chunk_t * gh * gw) as u32;
        assert_eq!(img_tokens.len(), rows as usize * conv_in as usize);
        let dim_row = asz * dim as u64;
        // Every block shares the same prefix ROW COUNT (contents differ).
        let prefix = cache[0].rows as u32;
        let s_k = prefix + rows;

        let upload = |slice: &[f32]| -> Result<WsBuf<WgpuBackend>, WgpuError> {
            let bytes = act_upload_bytes(act, slice);
            let buf = ws.alloc(bytes.len() as u64)?;
            backend.write_buffer(buf.id(), 0, &bytes)?;
            Ok(buf)
        };
        let img_up = upload(img_tokens)?;
        let tsin_up = upload(&timestep_sinusoid(t))?;
        let freqs = self
            .rope
            .lookup(&Self::chunk_grid_ids(start_t, chunk_t, gh, gw));
        let freqs_up = {
            let bytes = freqs_upload_bytes(act, &freqs);
            let buf = ws.alloc(bytes.len() as u64)?;
            backend.write_buffer(buf.id(), 0, &bytes)?;
            buf
        };

        let mut pins: Vec<GpuView> = Vec::new();
        let top = acquire_top(&self.handles.top, residency, backend, &mut pins).await?;

        // Setup: vec/silu_vec at this timestep; img = img_in(tokens).
        let img_ws = ws.alloc((rows * dim) as u64 * asz)?;
        let silu_vec_ws = ws.alloc(dim as u64 * asz)?;
        {
            let scope = ws.batch();
            let bp = &self.pipelines.bp;
            let ts = scope.import_copy(tsin_up.as_buf_ref());
            let h0 = linear(
                &scope,
                bp,
                ts,
                &top.time0,
                1,
                dim,
                FREQ_DIM as u32,
                Site::Module,
            )?;
            let h0a = silu(&scope, bp, h0, dim)?;
            let vec = linear(&scope, bp, h0a, &top.time2, 1, dim, dim, Site::Module)?;
            let sv = silu(&scope, bp, vec, dim)?;
            let dst = scope.import_copy(silu_vec_ws.as_buf_ref());
            scope.copy_buffer_to_buffer(sv, 0, dst, 0, dim as u64 * asz)?;
            let xi = scope.import_copy(img_up.as_buf_ref());
            let img = linear(
                &scope,
                bp,
                xi,
                &top.img_in,
                rows,
                dim,
                conv_in,
                Site::Module,
            )?;
            let dimg = scope.import_copy(img_ws.as_buf_ref());
            scope.copy_buffer_to_buffer(img, 0, dimg, 0, (rows * dim) as u64 * asz)?;
            scope.submit_void().await?;
        }

        // Host-write block 0's KV prefix before its scope runs.
        let write_prefix = |slot: &(WsBuf<WgpuBackend>, WsBuf<WgpuBackend>),
                            b: &BlockKv|
         -> Result<(), WgpuError> {
            backend.write_buffer(slot.0.id(), 0, &b.k)?;
            backend.write_buffer(slot.1.id(), 0, &b.v)?;
            Ok(())
        };
        write_prefix(&kv_ping[0], &cache[0])?;

        let bp = &self.pipelines.bp;
        let nblocks = self.handles.blocks.len();
        let mut cur = img_ws;
        let mut cur_pins: Vec<GpuView> = Vec::new();
        let mut cur_b =
            acquire_block(&self.handles.blocks[0], residency, backend, &mut cur_pins).await?;
        // Committed chunk K/V per block (raw act bytes), appended after the loop
        // (avoids aliasing `cache` between compute and prefetch).
        let mut committed: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
        for bi in 0..nblocks {
            let nxt = ws.alloc((rows * dim) as u64 * asz)?;
            let mut next_pins: Vec<GpuView> = Vec::new();
            let slot = &kv_ping[bi % 2];
            let compute = async {
                let kv_dl: Option<(WsBuf<WgpuBackend>, WsBuf<WgpuBackend>)>;
                {
                    let scope = ws.batch();
                    let sv = scope.import_copy(silu_vec_ws.as_buf_ref());
                    let imodp = linear(
                        &scope,
                        bp,
                        sv,
                        &cur_b.img_mod,
                        1,
                        6 * dim,
                        dim,
                        Site::Module,
                    )?;
                    let isig = |k| mod_sig(&scope, bp, imodp, k, dim);
                    let (sh1, sc1, g1) = (isig(0)?, isig(1)?, isig(2)?);
                    let (sh2, sc2, g2) = (isig(3)?, isig(4)?, isig(5)?);
                    let x = scope.import_copy(cur.as_buf_ref());
                    let im = norm_modulate(&scope, bp, x, sc1, sh1, rows, dim)?;
                    let im_a = qkv_a_side(&scope, bp, im, rows, dim)?;
                    let fq = scope.import_copy(freqs_up.as_buf_ref());
                    let iq = linear_a(
                        &scope,
                        bp,
                        im_a,
                        &cur_b.img_q,
                        rows,
                        dim,
                        dim,
                        Site::QkvSelf,
                    )?;
                    let iq = qk_norm_rope(&scope, bp, iq, cur_b.img_qn, Some(fq), rows, heads, hd)?;
                    let fq2 = scope.import_copy(freqs_up.as_buf_ref());
                    let ik = linear_a(
                        &scope,
                        bp,
                        im_a,
                        &cur_b.img_k,
                        rows,
                        dim,
                        dim,
                        Site::QkvSelf,
                    )?;
                    let ik =
                        qk_norm_rope(&scope, bp, ik, cur_b.img_kn, Some(fq2), rows, heads, hd)?;
                    let iv = linear_a(
                        &scope,
                        bp,
                        im_a,
                        &cur_b.img_v,
                        rows,
                        dim,
                        dim,
                        Site::QkvSelf,
                    )?;

                    // kv = [prefix (host-written) ; chunk]: append chunk K/V.
                    let kb = slot.0.as_buf_ref();
                    let vb = slot.1.as_buf_ref();
                    let kall =
                        scope.import_copy(BufRef::view(kb.id, kb.offset, s_k as u64 * dim_row));
                    let vall =
                        scope.import_copy(BufRef::view(vb.id, vb.offset, s_k as u64 * dim_row));
                    scope.copy_buffer_to_buffer(
                        ik,
                        0,
                        kall,
                        prefix as u64 * dim_row,
                        rows as u64 * dim_row,
                    )?;
                    scope.copy_buffer_to_buffer(
                        iv,
                        0,
                        vall,
                        prefix as u64 * dim_row,
                        rows as u64 * dim_row,
                    )?;
                    kv_dl = if commit {
                        Some((
                            persist(&scope, ws, ik, (rows * dim) as usize, asz)?,
                            persist(&scope, ws, iv, (rows * dim) as usize, asz)?,
                        ))
                    } else {
                        None
                    };

                    let sa = scope.alloc(bp.act_bytes(rows * dim))?;
                    let mask = scope.write_uniform(&0f32.to_le_bytes())?;
                    let scale = 1.0_f32 / (hd as f32).sqrt();
                    op_sdpa_f16_win(
                        &scope,
                        bp,
                        ActBuf::dense(iq),
                        ActBuf::dense(kall),
                        ActBuf::dense(vall),
                        mask,
                        ActBuf::dense(sa),
                        1,
                        rows,
                        s_k,
                        heads,
                        heads,
                        hd,
                        scale,
                        0,
                        1, // period (unused at window=0)
                        0, // window: full attention over the causal cache
                        0,
                    )?;
                    let ip = linear(&scope, bp, sa, &cur_b.img_proj, rows, dim, dim, Site::Proj)?;
                    let x1 = gate_residual(&scope, bp, x, g1, ip, rows, dim)?;
                    let im2 = norm_modulate(&scope, bp, x1, sc2, sh2, rows, dim)?;
                    let imlp = mlp(
                        &scope,
                        bp,
                        im2,
                        &cur_b.img_fc1,
                        &cur_b.img_fc2,
                        rows,
                        dim,
                        mlp_h,
                    )?;
                    let x2 = gate_residual(&scope, bp, x1, g2, imlp, rows, dim)?;
                    let d = scope.import_copy(nxt.as_buf_ref());
                    scope.copy_buffer_to_buffer(x2, 0, d, 0, (rows * dim) as u64 * asz)?;
                    scope.submit_void().await?;
                }
                let out = if let Some((k_ws, v_ws)) = kv_dl {
                    let kb = backend
                        .read_buffer(k_ws.id(), 0, (rows * dim) as u64 * asz)
                        .await?;
                    let vb = backend
                        .read_buffer(v_ws.id(), 0, (rows * dim) as u64 * asz)
                        .await?;
                    Some((kb, vb))
                } else {
                    None
                };
                Ok::<Option<(Vec<u8>, Vec<u8>)>, WgpuError>(out)
            };
            let prefetch = async {
                let b = if bi + 1 < nblocks {
                    Some(
                        acquire_block(
                            &self.handles.blocks[bi + 1],
                            residency,
                            backend,
                            &mut next_pins,
                        )
                        .await?,
                    )
                } else {
                    None
                };
                // Stage the NEXT block's KV prefix into the other ping-pong slot
                // while this block computes (the slot's last reader finished a
                // full iteration ago).
                if bi + 1 < nblocks {
                    write_prefix(&kv_ping[(bi + 1) % 2], &cache[bi + 1])
                        .map_err(ResidencyError::Backend)?;
                }
                Ok::<_, ResidencyError<S::Error, WgpuError>>(b)
            };
            let (c_res, p_res) = futures::join!(compute, prefetch);
            if let Some(kv) = c_res? {
                committed.push(kv);
            }
            let next_b = p_res?;
            drop(cur_pins);
            cur = nxt;
            cur_pins = next_pins;
            if let Some(b) = next_b {
                cur_b = b;
            }
        }

        if commit {
            for (bi, (kb, vb)) in committed.into_iter().enumerate() {
                cache[bi].k.extend_from_slice(&kb);
                cache[bi].v.extend_from_slice(&vb);
                cache[bi].rows += rows as usize;
            }
            drop(pins);
            return Ok(None);
        }

        // final_layer over the chunk rows.
        let vel_ws = ws.alloc((rows * latent) as u64 * asz)?;
        {
            let scope = ws.batch();
            let sv = scope.import_copy(silu_vec_ws.as_buf_ref());
            let emb = linear(
                &scope,
                bp,
                sv,
                &top.final_adaln,
                1,
                2 * dim,
                dim,
                Site::Module,
            )?;
            let shift = mod_sig(&scope, bp, emb, 0, dim)?;
            let scale = mod_sig(&scope, bp, emb, 1, dim)?;
            let img = scope.import_copy(cur.as_buf_ref());
            let modded = norm_modulate(&scope, bp, img, scale, shift, rows, dim)?;
            let vel = linear(
                &scope,
                bp,
                modded,
                &top.final_lin,
                rows,
                latent,
                dim,
                Site::Module,
            )?;
            let dst = scope.import_copy(vel_ws.as_buf_ref());
            scope.copy_buffer_to_buffer(vel, 0, dst, 0, (rows * latent) as u64 * asz)?;
            scope.submit_void().await?;
        }
        drop(pins);
        let v = read_acts(
            backend,
            &vel_ws.as_buf_ref(),
            rows as usize * cfg::LATENT_CHANNELS,
            act,
        )
        .await?;
        Ok(Some(v))
    }

    /// Pack one chunk's `[rows, 65]` img tokens: `[noisy32 | cond32 | mask1]`.
    /// `x` is the full latent `[32, T, H, W]`; the cond block carries the
    /// VAE-encoded first frame on GLOBAL frame 0 only (mask likewise).
    /// `cond0 = None` (the t2v probe) leaves cond + mask all-zero, matching the
    /// parent T2V's `[noise | 0 | 0]` packing.
    fn pack_chunk_tokens(
        x: &[f32],
        cond0: Option<&[f32]>,
        start_t: usize,
        chunk_t: usize,
        gt: usize,
        gh: usize,
        gw: usize,
    ) -> Vec<f32> {
        let lat = cfg::LATENT_CHANNELS;
        let conv_in = cfg::CONV_IN_CHANNELS;
        let hw = gh * gw;
        let thw = gt * hw;
        if let Some(c0) = cond0 {
            assert_eq!(c0.len(), lat * hw, "cond latent [32, H, W]");
        }
        let mut out = vec![0.0f32; chunk_t * hw * conv_in];
        for ti in 0..chunk_t {
            let gt_i = start_t + ti;
            for p in 0..hw {
                let row = (ti * hw + p) * conv_in;
                let n = gt_i * hw + p;
                for c in 0..lat {
                    out[row + c] = x[c * thw + n];
                }
                if gt_i == 0
                    && let Some(c0) = cond0
                {
                    for c in 0..lat {
                        out[row + lat + c] = c0[c * hw + p];
                    }
                    out[row + 2 * lat] = 1.0;
                }
            }
        }
        out
    }

    /// Full causal AR generation. `text [seq, 3584]` (Qwen2.5-VL embeds),
    /// `vision [729, 1152]` (SigLIP) + `cond0 [32, H, W]` (VAE-encoded first
    /// frame) for I2V, BOTH `None` for the text-only t2v probe (out of the dmd
    /// checkpoint's training distribution), `init_noise [32, T, H, W]`.
    /// `grid.0` (T) must be a multiple of [`arcfg::CHUNK_LATENT_FRAMES`].
    /// Returns the denoised latent `[32,T,H,W]`.
    /// `progress(chunk, n_chunks, step, n_steps)` is 1-based; `n_steps` is
    /// `schedule.steps() + 1` with the final "step" being the recache pass.
    #[allow(clippy::too_many_arguments)]
    pub async fn generate<S: WeightSource>(
        &self,
        backend: &WgpuBackend,
        residency: &WeightResidency<S>,
        ws: &Workspace<WgpuBackend>,
        text: &[f32],
        seq: usize,
        vision: Option<&[f32]>,
        cond0: Option<&[f32]>,
        init_noise: &[f32],
        grid: (usize, usize, usize),
        schedule: &crate::hunyuan::scheduler::FlowMatchSchedule,
        progress: Option<&dyn Fn(u32, u32, u32, u32)>,
        cancel: Option<&dyn Fn() -> bool>,
    ) -> Result<Vec<f32>, HunyuanDitError<S::Error>> {
        let (gt, gh, gw) = grid;
        let chunk_t = arcfg::CHUNK_LATENT_FRAMES;
        assert_eq!(gt % chunk_t, 0, "latent frames must chunk evenly");
        let n_chunks = gt / chunk_t;
        let lat = cfg::LATENT_CHANNELS;
        let thw = gt * gh * gw;
        assert_eq!(init_noise.len(), lat * thw, "init noise size");
        let steps = schedule.steps();
        let n_steps = (steps + 1) as u32; // + the recache pass

        let mut cache = self
            .build_txt_cache(backend, residency, ws, text, seq, vision)
            .await?;
        if ar_diag() {
            diag_stats("txt cache k[0]", &bf16_to_f32(&cache[0].k));
            diag_stats("txt cache v[0]", &bf16_to_f32(&cache[0].v));
            let last = cache.len() - 1;
            diag_stats("txt cache k[last]", &bf16_to_f32(&cache[last].k));
        }

        let asz = self.pipelines.act_size();
        let dim = cfg::HIDDEN as u64;
        let rows = (chunk_t * gh * gw) as u32;
        let mut x = init_noise.to_vec();

        for chunk in 0..n_chunks {
            let start_t = chunk * chunk_t;
            // Ping-pong KV buffers sized for THIS chunk's prefix + chunk rows
            // (the prefix is fixed across the chunk's forwards; the commit only
            // lands after them).
            let s_k = (cache[0].rows + rows as usize) as u64;
            let kv_ping = [
                (ws.alloc(s_k * dim * asz)?, ws.alloc(s_k * dim * asz)?),
                (ws.alloc(s_k * dim * asz)?, ws.alloc(s_k * dim * asz)?),
            ];

            for i in 0..steps {
                if cancel.is_some_and(|c| c()) {
                    return Err(HunyuanDitError::Cancelled);
                }
                if let Some(p) = progress {
                    p(chunk as u32 + 1, n_chunks as u32, i as u32 + 1, n_steps);
                }
                let tokens = Self::pack_chunk_tokens(&x, cond0, start_t, chunk_t, gt, gh, gw);
                let v = self
                    .forward_chunk(
                        backend,
                        residency,
                        ws,
                        &tokens,
                        chunk_t,
                        start_t,
                        gh,
                        gw,
                        schedule.timesteps[i],
                        &mut cache,
                        false,
                        &kv_ping,
                    )
                    .await?
                    .expect("denoise forward returns velocity");
                if ar_diag() {
                    diag_stats(&format!("chunk {chunk} step {i} velocity"), &v);
                }
                // Euler over the chunk region: x += dt * v (v is token-major).
                let dt = schedule.dt(i);
                let hw = gh * gw;
                for r in 0..rows as usize {
                    let n = start_t * hw + r;
                    for c in 0..lat {
                        x[c * thw + n] += dt * v[r * lat + c];
                    }
                }
            }

            // Recache: commit the denoised chunk's K/V at t=0.
            if cancel.is_some_and(|c| c()) {
                return Err(HunyuanDitError::Cancelled);
            }
            if let Some(p) = progress {
                p(chunk as u32 + 1, n_chunks as u32, n_steps, n_steps);
            }
            let tokens = Self::pack_chunk_tokens(&x, cond0, start_t, chunk_t, gt, gh, gw);
            self.forward_chunk(
                backend,
                residency,
                ws,
                &tokens,
                chunk_t,
                start_t,
                gh,
                gw,
                arcfg::RECACHE_TIMESTEP,
                &mut cache,
                true,
                &kv_ping,
            )
            .await?;
        }
        Ok(x)
    }
}
