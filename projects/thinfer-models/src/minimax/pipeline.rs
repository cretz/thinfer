//! MiniMax-H3 t2va denoise: one packed sequence, one forward per step.
//!
//! The checkpoint is CFG-distilled, so there is no unconditional pass and no
//! guidance scale. Video and audio run INDEPENDENT sigma schedules (shift 12.0
//! and 3.0) over that one shared forward, which is why each modality steps with
//! its own timestep.
//!
//! **Sequence-length budget.** Activations scale with the packed row count, and
//! the widest intermediate is the fused SwiGLU at `rows x 2 * ffn_hidden`. At
//! the native 1344x768 canvas that is ~37.8k rows, so the fused buffer alone is
//! 4.3 GiB in f32 and the block does not fit an 8 GiB card however the weights
//! are streamed. Making the native canvas fit needs the row-independent sites
//! (both patch projections and the whole feed-forward) chunked over rows; only
//! attention genuinely needs the full sequence resident. Until that lands,
//! [`check_sequence_budget`] refuses a canvas it cannot run instead of dying
//! mid-denoise with a device loss.

use thinfer_core::backend::{Backend, BufRef, WgpuBackend, WgpuError};
use thinfer_core::residency::{ResidencyError, WeightResidency};
use thinfer_core::trace::{self, PHASE};
use thinfer_core::weight::WeightSource;
use thinfer_core::workspace::Workspace;
use tracing::Instrument;

use crate::common::block::{ActBuf, Block, alloc_matmul_out_buf};
use crate::common::seq;
use crate::ltx::sampler::gaussian_noise;

use super::adaln::{ModSegment, ModulationPlan};
use super::dit::{
    BlockBufs, DitHandles, DitPipelines, DitShape, ModulationTable, forward_block,
    forward_final_norm, forward_refiner_block,
};
use super::loader::ModulationTables;
use super::packing::{self, PackedSequence};
use super::scheduler::JointSchedule;
use super::{DitConfig, TAG_AUDIO, TAG_TEXT, TAG_VIDEO};

/// Rows above which one block's activations no longer fit the 8 GiB target.
///
/// Derived from the fused SwiGLU, the widest intermediate: `rows x 28672 x 4`
/// bytes plus the qkv `rows x 21504 x 4`. 6144 rows is ~1.2 GiB of fused buffer,
/// which leaves headroom for the streamed weights inside a 6 GiB budget.
pub const MAX_PACKED_ROWS: usize = 6144;

#[derive(Debug)]
pub enum PipelineError<SE: core::fmt::Debug> {
    Wgpu(WgpuError),
    Residency(ResidencyError<SE, WgpuError>),
    Geometry(packing::GeometryError),
    /// The canvas resolves to more rows than one block's activations can hold.
    SequenceTooLong {
        rows: usize,
        limit: usize,
    },
    /// The packed layout is not the contiguous `[text | audio | video]` the
    /// t2va path assumes.
    NonContiguousLayout,
}

impl<SE: core::fmt::Debug> From<WgpuError> for PipelineError<SE> {
    fn from(e: WgpuError) -> Self {
        Self::Wgpu(e)
    }
}
impl<SE: core::fmt::Debug> From<ResidencyError<SE, WgpuError>> for PipelineError<SE> {
    fn from(e: ResidencyError<SE, WgpuError>) -> Self {
        Self::Residency(e)
    }
}
impl<SE: core::fmt::Debug> From<packing::GeometryError> for PipelineError<SE> {
    fn from(e: packing::GeometryError) -> Self {
        Self::Geometry(e)
    }
}

/// Refuse a canvas whose packed sequence cannot run, with the numbers that
/// decide it rather than an out-of-memory later.
pub fn check_sequence_budget<SE: core::fmt::Debug>(rows: usize) -> Result<(), PipelineError<SE>> {
    if rows > MAX_PACKED_ROWS {
        return Err(PipelineError::SequenceTooLong {
            rows,
            limit: MAX_PACKED_ROWS,
        });
    }
    Ok(())
}

/// Upload the host modulation tables into resident GPU buffers.
///
/// These stay resident for the whole run: ~48 MB at four steps, against the
/// ~7 GiB of `adaln_proj` they replace.
pub fn upload_modulation_tables(
    backend: &WgpuBackend,
    workspace: &Workspace<WgpuBackend>,
    tables: &ModulationTables,
    act: thinfer_core::ops::ActDtype,
) -> Result<
    (
        ModulationTable,
        Vec<thinfer_core::workspace::WsBuf<WgpuBackend>>,
    ),
    WgpuError,
> {
    let mut held = Vec::with_capacity(tables.blocks.len() + 1);
    let mut blocks = Vec::with_capacity(tables.blocks.len());
    for table in &tables.blocks {
        let bytes = seq::act_upload_bytes(act, table);
        let buf = workspace.alloc(bytes.len() as u64)?;
        backend.write_buffer(buf.id(), 0, &bytes)?;
        blocks.push(buf.as_buf_ref());
        held.push(buf);
    }
    let bytes = seq::act_upload_bytes(act, &tables.final_layer);
    let buf = workspace.alloc(bytes.len() as u64)?;
    backend.write_buffer(buf.id(), 0, &bytes)?;
    let final_layer = buf.as_buf_ref();
    held.push(buf);
    Ok((
        ModulationTable {
            layout: tables.layout,
            blocks,
            final_layer,
        },
        held,
    ))
}

/// Everything one t2va request needs that does not change between steps.
pub struct DenoisePlan {
    pub packed: PackedSequence,
    pub schedule: JointSchedule,
    pub modulation: ModulationPlan,
    /// `[seq, rotary_dim]` rotary frequencies for the packed grid.
    pub freqs: Vec<f32>,
    pub num_video_rows: usize,
    pub num_audio_rows: usize,
    pub num_text_rows: usize,
}

impl DenoisePlan {
    /// Build the layout, schedules and modulation plan for a t2va request.
    pub fn build<SE: core::fmt::Debug>(
        cfg: &DitConfig,
        num_text_rows: usize,
        num_latent_frames: usize,
        latent_height: usize,
        latent_width: usize,
        num_audio_latents: usize,
        steps: usize,
    ) -> Result<Self, PipelineError<SE>> {
        let text_tags = vec![TAG_TEXT; num_text_rows];
        let packed = packing::build_packed_sequence(
            &text_tags,
            num_latent_frames,
            latent_height,
            latent_width,
            num_audio_latents,
            cfg.patch,
            &[],
        );
        check_sequence_budget::<SE>(packed.seq_len)?;

        // `steps` is MODEL EVALUATIONS, but the scheduler is built from a sigma
        // GRID and `N` sigmas drive `N - 1` evaluations (the grid includes the
        // terminal 0). Passing `steps` straight through ran a 4-step turbo
        // distillation on a 3-step schedule.
        let schedule = JointSchedule::new(steps + 1).expect("schedule");
        // t2va pins no conditioning rows, so both conditioning levels are
        // unused; text rows inherit the video timestep.
        let per_step: Vec<_> = (0..schedule.num_steps())
            .map(|i| {
                packing::build_row_timesteps(
                    &packed,
                    schedule.video.timesteps()[i],
                    schedule.audio.timesteps()[i],
                    schedule.video.timesteps()[i],
                    schedule.audio.timesteps()[i],
                )
            })
            .collect();
        let modulation = ModulationPlan::build(cfg, &per_step);
        let freqs = super::rope3d::build_freqs(cfg, &packed.position_ids);

        Ok(Self {
            num_video_rows: packed.video_indices.len(),
            num_audio_rows: packed.audio_indices.len(),
            num_text_rows,
            packed,
            schedule,
            modulation,
            freqs,
        })
    }

    /// t2va emits `[text | audio | video]` with no conditioning rows, so every
    /// modality occupies one contiguous span and the scatter is a slice write
    /// rather than a gather.
    fn contiguous_spans<SE: core::fmt::Debug>(
        &self,
    ) -> Result<[(usize, usize); 3], PipelineError<SE>> {
        let spans = [
            (&self.packed.text_indices, TAG_TEXT),
            (&self.packed.audio_indices, TAG_AUDIO),
            (&self.packed.video_indices, TAG_VIDEO),
        ];
        let mut out = [(0usize, 0usize); 3];
        for (slot, (idx, _)) in spans.iter().enumerate() {
            let Some(&first) = idx.first() else {
                return Err(PipelineError::NonContiguousLayout);
            };
            if idx
                .iter()
                .enumerate()
                .any(|(k, &v)| v as usize != first as usize + k)
            {
                return Err(PipelineError::NonContiguousLayout);
            }
            out[slot] = (first as usize, idx.len());
        }
        Ok(out)
    }
}

/// Predicted velocity of the generated rows, split by modality.
pub struct Velocity {
    pub video: Vec<f32>,
    pub audio: Vec<f32>,
}

/// One DiT forward over the packed sequence.
///
/// `video_rows` is `[num_video_rows, video_patch_dim]`, `audio_rows` is
/// `[num_audio_rows, audio_latents_dim]` and `text` is the conditioner's
/// `[num_text_rows, text_dim]` hidden state. All host f32.
#[allow(clippy::too_many_arguments)]
pub async fn forward<S: WeightSource>(
    cfg: &DitConfig,
    backend: &WgpuBackend,
    pipelines: &DitPipelines,
    residency: &WeightResidency<S>,
    workspace: &Workspace<WgpuBackend>,
    handles: &DitHandles,
    table: &ModulationTable,
    plan: &DenoisePlan,
    step: usize,
    video_rows: &[f32],
    audio_rows: &[f32],
    text: &[f32],
) -> Result<Velocity, PipelineError<S::Error>> {
    let p = &pipelines.block;
    let act = p.act_dtype;
    let seq_len = plan.packed.seq_len;
    let hidden = cfg.hidden;
    let [(text_at, n_text), (audio_at, n_audio), (video_at, n_video)] = plan.contiguous_spans()?;

    let shape = DitShape {
        cfg: *cfg,
        rows: seq_len,
    };
    let segments = plan.modulation.segments(step, &plan.packed.token_tags);
    let ts_segments = timestep_segments(&plan.modulation, step);

    // Packed activation buffer, filled per modality.
    let x_bytes = p.act_bytes((seq_len * hidden) as u32);
    let x_buf = workspace.alloc(x_bytes)?;
    let freqs_bytes = seq::freqs_upload_bytes(act, &plan.freqs);
    let freqs_buf = workspace.alloc(freqs_bytes.len() as u64)?;
    backend.write_buffer(freqs_buf.id(), 0, &freqs_bytes)?;

    // --- input projections ---
    {
        let vid_in = upload_rows(backend, workspace, act, video_rows)?;
        let aud_in = upload_rows(backend, workspace, act, audio_rows)?;
        let txt_in = upload_rows(backend, workspace, act, text)?;

        let vp = residency.acquire(handles.video_patch_proj, backend).await?;
        let vpb = residency
            .acquire(handles.video_patch_proj_bias, backend)
            .await?;
        let ap = residency.acquire(handles.audio_patch_proj, backend).await?;
        let apb = residency
            .acquire(handles.audio_patch_proj_bias, backend)
            .await?;
        let cp = residency.acquire(handles.condition_proj, backend).await?;
        let cpb = residency
            .acquire(handles.condition_proj_bias, backend)
            .await?;

        let x_ref = x_buf.as_buf_ref();
        let row_bytes = hidden as u64 * p.act_bytes(1);
        let scope = workspace.batch();
        let x_all = scope.import(&x_ref);

        for (src, w, b, rows, in_w, at) in [
            (
                vid_in.as_buf_ref(),
                vp.buf(),
                vpb.buf(),
                n_video,
                cfg.video_patch_dim(),
                video_at,
            ),
            (
                aud_in.as_buf_ref(),
                ap.buf(),
                apb.buf(),
                n_audio,
                cfg.audio_latents_dim,
                audio_at,
            ),
            (
                txt_in.as_buf_ref(),
                cp.buf(),
                cpb.buf(),
                n_text,
                cfg.text_dim,
                text_at,
            ),
        ] {
            let dst = scope.subview(&x_all, at as u64 * row_bytes, rows as u64 * row_bytes);
            project_into(
                &scope,
                p,
                ActBuf::dense(scope.import_copy(src)),
                w,
                b,
                dst,
                rows as u32,
                hidden as u32,
                in_w as u32,
            )?;
        }
        scope.submit_void().await?;
    }

    // --- token refiner over the text rows, in place ---
    {
        let refiner_shape = DitShape {
            cfg: *cfg,
            rows: n_text,
        };
        let row_bytes = hidden as u64 * p.act_bytes(1);
        let text_bytes = n_text as u64 * row_bytes;
        // The refiner writes the text rows back in place, so each block runs
        // into `scratch` and is copied home. Two blocks, so a second full-size
        // buffer would cost more than the copies.
        let cur = BufRef::view(
            x_buf.as_buf_ref().id,
            x_buf.as_buf_ref().offset + text_at as u64 * row_bytes,
            text_bytes,
        );
        let scratch = workspace.alloc(text_bytes)?;
        for block in &handles.refiner {
            let views = block.acquire(residency, backend).await?;
            let bufs = views.bufs();
            let scope = workspace.batch();
            forward_refiner_block(
                &scope,
                pipelines,
                refiner_shape,
                scope.import_copy(cur),
                scope.import_copy(scratch.as_buf_ref()),
                &bufs,
                block.kinds(),
            )?;
            let a = scope.import_copy(scratch.as_buf_ref());
            let b = scope.import_copy(cur);
            scope.copy_buffer_to_buffer(a, 0, b, 0, text_bytes)?;
            scope.submit_void().await?;
            drop(views);
        }
        // `token_refiner.final_norm` closes the refiner.
        let fnorm = residency
            .acquire(handles.refiner_final_norm, backend)
            .await?;
        let scope = workspace.batch();
        let src = ActBuf::dense(scope.import_copy(cur));
        let dst = ActBuf::dense(scope.import_copy(scratch.as_buf_ref()));
        crate::common::block::op_rmsnorm(
            &scope,
            p,
            src,
            scope.import_copy(fnorm.buf()),
            dst,
            n_text as u32,
            hidden as u32,
            cfg.final_norm_eps,
        )?;
        let a = scope.import_copy(scratch.as_buf_ref());
        let b = scope.import_copy(cur);
        scope.copy_buffer_to_buffer(a, 0, b, 0, text_bytes)?;
        scope.submit_void().await?;
    }

    // --- the 50-block stack ---
    let freqs_ref = freqs_buf.as_buf_ref();
    let mut cur = x_buf;
    let mut pending = match handles.blocks.first() {
        Some(h) => Some(h.acquire(residency, backend).await?),
        None => None,
    };
    for idx in 0..handles.blocks.len() {
        let _guard = trace::scope!(format!("minimax.dit.block.{idx}")).entered();
        let views = pending.take().expect("pending acquire missing");
        let bufs: BlockBufs = views.bufs();
        let cur_ref = cur.as_buf_ref();
        let nxt = workspace.alloc(x_bytes)?;
        let nxt_ref = nxt.as_buf_ref();
        let scope = workspace.batch();
        forward_block(
            &scope,
            pipelines,
            shape,
            scope.import(&cur_ref),
            scope.import(&freqs_ref),
            scope.import(&nxt_ref),
            &bufs,
            handles.blocks[idx].kinds(),
            table,
            idx,
            &segments,
        )?;
        let next_acquire = async {
            match handles.blocks.get(idx + 1) {
                Some(h) => Ok::<_, ResidencyError<S::Error, WgpuError>>(Some(
                    h.acquire(residency, backend).await?,
                )),
                None => Ok(None),
            }
        };
        let submit = scope
            .submit_void()
            .instrument(tracing::debug_span!(target: PHASE, "minimax.dit.submit", idx));
        let (submit_res, next_res) = futures::join!(submit, next_acquire);
        submit_res?;
        pending = next_res?;
        drop(views);
        cur = nxt;
        workspace.drain_pool();
    }

    // --- final layer: shared modulated norm, then both heads over every row ---
    let normed = workspace.alloc(x_bytes)?;
    {
        let fnorm = residency.acquire(handles.final_norm, backend).await?;
        let cur_ref = cur.as_buf_ref();
        let out_ref = normed.as_buf_ref();
        let scope = workspace.batch();
        forward_final_norm(
            &scope,
            pipelines,
            shape,
            scope.import(&cur_ref),
            &fnorm.buf(),
            scope.import(&out_ref),
            table,
            &ts_segments,
        )?;
        scope.submit_void().await?;
    }

    let video = read_head(
        cfg,
        backend,
        pipelines,
        residency,
        workspace,
        &normed,
        handles.video_out,
        handles.video_out_bias,
        seq_len,
        cfg.video_patch_dim(),
        video_at,
        n_video,
    )
    .await?;
    let audio = read_head(
        cfg,
        backend,
        pipelines,
        residency,
        workspace,
        &normed,
        handles.audio_out,
        handles.audio_out_bias,
        seq_len,
        cfg.audio_latents_dim,
        audio_at,
        n_audio,
    )
    .await?;
    Ok(Velocity { video, audio })
}

/// Both heads run over EVERY row and the modality's rows are selected after,
/// matching the reference. The slice is contiguous for t2va.
#[allow(clippy::too_many_arguments)]
async fn read_head<S: WeightSource>(
    _cfg: &DitConfig,
    backend: &WgpuBackend,
    pipelines: &DitPipelines,
    residency: &WeightResidency<S>,
    workspace: &Workspace<WgpuBackend>,
    normed: &thinfer_core::workspace::WsBuf<WgpuBackend>,
    w: thinfer_core::residency::WeightHandle,
    b: thinfer_core::residency::WeightHandle,
    seq_len: usize,
    out_w: usize,
    at: usize,
    rows: usize,
) -> Result<Vec<f32>, PipelineError<S::Error>> {
    let p = &pipelines.block;
    let act = p.act_dtype;
    let wv = residency.acquire(w, backend).await?;
    let bv = residency.acquire(b, backend).await?;
    let out = workspace.alloc(p.act_bytes((seq_len * out_w) as u32))?;
    {
        let src = normed.as_buf_ref();
        let dst = out.as_buf_ref();
        let scope = workspace.batch();
        project_into(
            &scope,
            p,
            ActBuf::dense(scope.import_copy(src)),
            wv.buf(),
            bv.buf(),
            scope.import_copy(dst),
            seq_len as u32,
            out_w as u32,
            _cfg.hidden as u32,
        )?;
        scope.submit_void().await?;
    }
    let bytes = backend
        .read_buffer(
            out.id(),
            at as u64 * out_w as u64 * p.act_bytes(1),
            rows as u64 * out_w as u64 * p.act_bytes(1),
        )
        .await?;
    Ok(seq::act_readback_to_f32(act, &bytes, rows * out_w))
}

fn upload_rows(
    backend: &WgpuBackend,
    workspace: &Workspace<WgpuBackend>,
    act: thinfer_core::ops::ActDtype,
    rows: &[f32],
) -> Result<thinfer_core::workspace::WsBuf<WgpuBackend>, WgpuError> {
    let bytes = seq::act_upload_bytes(act, rows);
    let buf = workspace.alloc(bytes.len().max(4) as u64)?;
    backend.write_buffer(buf.id(), 0, &bytes)?;
    Ok(buf)
}

/// Matmul with bias straight into `dst`.
#[allow(clippy::too_many_arguments)]
fn project_into<'w>(
    scope: &thinfer_core::workspace::BatchScope<'w, WgpuBackend>,
    p: &crate::common::block::BlockPipelines,
    src: ActBuf<'w>,
    w: BufRef,
    bias: BufRef,
    dst: thinfer_core::workspace::BatchBuf<'w>,
    rows: u32,
    out_w: u32,
    in_w: u32,
) -> Result<(), WgpuError> {
    let scratch = alloc_matmul_out_buf(scope, p, rows * out_w)?;
    let dims = scope.u32x4_uniform(rows, out_w, in_w, 0)?;
    Block::dispatch_matmul_site(
        scope,
        p,
        src,
        scope.import_copy(w),
        scratch,
        dims,
        p.matmul_i8_qkv.as_ref(),
        p.dequant_i8_qkv.as_ref(),
        None,
        // Patch projections and both output heads are dense F16 `Linear2D
        // [K, N]` weights, so they take the MODULE pipeline. The block pipeline
        // expects the N-major workspace a quant site's dequant emits.
        &p.matmul_module,
        &p.matmuls.module,
        rows,
        out_w,
        in_w,
    )?;
    let bu = scope.u32x4_uniform(out_w, 0, 0, 0)?;
    scope.bcast_add::<thinfer_core::ops::BcastAddF32>(
        &p.bcast_add,
        scratch,
        scope.import_copy(bias),
        bu,
        dst,
        rows * out_w,
    )
}

/// The final layer's modulation is keyed by TIMESTEP alone, not modality, so
/// its segments come from the per-row timestep index rather than the AdaLN row.
fn timestep_segments(plan: &ModulationPlan, step: usize) -> Vec<ModSegment> {
    let rows = &plan.per_step_row_index[step];
    let mut out: Vec<ModSegment> = Vec::new();
    for (i, &row) in rows.iter().enumerate() {
        // AdaLN rows are `timestep * MODALITY_NUM + tag`; recover the timestep.
        let ts = row / super::MODALITY_NUM as u32;
        match out.last_mut() {
            Some(seg) if seg.table_row == ts => seg.len += 1,
            _ => out.push(ModSegment {
                start: i as u32,
                len: 1,
                table_row: ts,
            }),
        }
    }
    out
}

/// The generated latents at the end of a run, still in packed-row layout.
pub struct Latents {
    /// `[num_video_rows, video_patch_dim]`.
    pub video: Vec<f32>,
    /// `[num_audio_rows, audio_latents_dim]`, channel-major over rows.
    pub audio: Vec<f32>,
}

/// Run the whole t2va denoise loop.
///
/// Video and audio ride INDEPENDENT sigma schedules over one shared forward, so
/// each modality steps with its own coefficients against the same predicted
/// velocity. `progress` is called with `(step, num_steps)` after each step.
#[allow(clippy::too_many_arguments)]
pub async fn denoise<S: WeightSource>(
    cfg: &DitConfig,
    backend: &WgpuBackend,
    pipelines: &DitPipelines,
    residency: &WeightResidency<S>,
    workspace: &Workspace<WgpuBackend>,
    handles: &DitHandles,
    table: &ModulationTable,
    plan: &DenoisePlan,
    text: &[f32],
    seed: u64,
    mut progress: impl FnMut(usize, usize),
) -> Result<Latents, PipelineError<S::Error>> {
    let (mut video, mut audio) = initial_noise(cfg, plan.num_video_rows, plan.num_audio_rows, seed);
    let steps = plan.schedule.num_steps();
    for step in 0..steps {
        let _guard = trace::scope!(format!("minimax.denoise.step.{step}")).entered();
        let v = forward(
            cfg, backend, pipelines, residency, workspace, handles, table, plan, step, &video,
            &audio, text,
        )
        .await?;
        // The reference evaluates `ratio * x + (1 - ratio) * (x + sigma * v)`
        // rather than the algebraically equal `x + (1 - ratio) * sigma * v`;
        // the two are not bit-identical in f32, so keep this form.
        let vc = plan.schedule.video.step_coeffs(step).expect("video step");
        let ac = plan.schedule.audio.step_coeffs(step).expect("audio step");
        for (x, dv) in video.iter_mut().zip(&v.video) {
            let x0 = *x + vc.sigma_from_timestep * dv;
            *x = vc.ratio * *x + vc.one_minus_ratio * x0;
        }
        for (x, dv) in audio.iter_mut().zip(&v.audio) {
            let x0 = *x + ac.sigma_from_timestep * dv;
            *x = ac.ratio * *x + ac.one_minus_ratio * x0;
        }
        progress(step + 1, steps);
    }
    Ok(Latents { video, audio })
}

/// Re-lay the packed video rows as the video VAE's `[frames, h, w, channels]`,
/// denormalized by the autoencoder's per-channel statistics.
///
/// Two conversions ride here: [`packing::unpatchify_video_tokens`] emits
/// CHANNEL-FIRST `[c, f, h, w]`, and the decoder reads channels-LAST.
pub fn video_rows_to_latents(
    cfg: &DitConfig,
    rows: &[f32],
    frames: usize,
    lat_h: usize,
    lat_w: usize,
    latents_mean: &[f32],
    latents_std: &[f32],
) -> Vec<f32> {
    let c = cfg.latents_dim;
    let planar = packing::unpatchify_video_tokens(rows, c, frames, lat_h, lat_w, cfg.patch);
    let mut out = vec![0.0f32; planar.len()];
    for ci in 0..c {
        let (mu, sd) = (latents_mean[ci], latents_std[ci]);
        for f in 0..frames {
            for h in 0..lat_h {
                for w in 0..lat_w {
                    let src = ((ci * frames + f) * lat_h + h) * lat_w + w;
                    let dst = ((f * lat_h + h) * lat_w + w) * c + ci;
                    out[dst] = planar[src] * sd + mu;
                }
            }
        }
    }
    out
}

/// Re-lay the packed audio rows as the audio VAE's `[channels, latent_channels,
/// frames]`. The packed form is channel-major over rows, each row one frame's
/// latent vector; the decoder wants each channel's latents transposed.
///
/// Denormalization is NOT applied here: `AudioVae::decode` owns it.
pub fn audio_rows_to_latents(cfg: &DitConfig, rows: &[f32], num_audio_latents: usize) -> Vec<f32> {
    let k = cfg.audio_latents_dim;
    let ch = rows.len() / (k * num_audio_latents);
    let mut out = vec![0.0f32; rows.len()];
    for s in 0..ch {
        for t in 0..num_audio_latents {
            for ki in 0..k {
                out[(s * k + ki) * num_audio_latents + t] =
                    rows[(s * num_audio_latents + t) * k + ki];
            }
        }
    }
    out
}

/// Draw the initial noise for a t2va run.
///
/// The draw order matches the reference so a seed reproduces a run: video
/// first, then audio.
pub fn initial_noise(
    cfg: &DitConfig,
    num_video_rows: usize,
    num_audio_rows: usize,
    seed: u64,
) -> (Vec<f32>, Vec<f32>) {
    let v = num_video_rows * cfg.video_patch_dim();
    let a = num_audio_rows * cfg.audio_latents_dim;
    let all = gaussian_noise(v + a, seed);
    let (video, audio) = all.split_at(v);
    (video.to_vec(), audio.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The native canvas is knowingly out of budget; the guard must say so
    /// rather than let a device loss report it.
    #[test]
    fn native_canvas_is_refused_until_row_chunking_lands() {
        // 1344x768 -> latent 84x48 -> patch (1,2,2) -> 1008 rows/frame.
        let native_rows = 37 * 1008 + 414 + 64;
        assert!(check_sequence_budget::<std::convert::Infallible>(native_rows).is_err());
        assert!(check_sequence_budget::<std::convert::Infallible>(4600).is_ok());
    }

    /// Timestep segments must collapse contiguous equal timesteps, or the final
    /// layer dispatches one tiny modulate per row.
    #[test]
    fn timestep_segments_collapse_runs() {
        let plan = ModulationPlan {
            timesteps: vec![0.0, 1.0],
            // rows = timestep * 3 + tag: [t0/text, t0/audio, t1/video, t1/text]
            per_step_row_index: vec![vec![1, 2, 3, 4]],
            layout: super::super::adaln::ModulationLayout::new(2, 8),
        };
        let segs = timestep_segments(&plan, 0);
        assert_eq!(segs.len(), 2);
        assert_eq!((segs[0].start, segs[0].len, segs[0].table_row), (0, 2, 0));
        assert_eq!((segs[1].start, segs[1].len, segs[1].table_row), (2, 2, 1));
    }

    /// Channel-first to channels-last, with the per-channel denormalization
    /// applied to the right channel. Getting the transpose wrong scrambles the
    /// decode with no NaN, and a swapped mean/std pair shifts every frame.
    #[test]
    fn video_latents_transpose_and_denormalize() {
        let mut cfg = DitConfig::h3_33b();
        cfg.latents_dim = 2;
        cfg.patch = (1, 1, 1);
        let (frames, h, w) = (1, 1, 2);
        // One row per (f, h, w) position, each `latents_dim` wide.
        let rows = vec![1.0f32, 2.0, 3.0, 4.0];
        let out = video_rows_to_latents(&cfg, &rows, frames, h, w, &[10.0, 20.0], &[2.0, 3.0]);
        // Channels-last: [w0c0, w0c1, w1c0, w1c1].
        assert_eq!(
            out,
            vec![
                1.0 * 2.0 + 10.0,
                2.0 * 3.0 + 20.0,
                3.0 * 2.0 + 10.0,
                4.0 * 3.0 + 20.0
            ]
        );
    }

    /// Audio rows are one frame each, channel-major; the decoder wants each
    /// channel's latents transposed to `[latent_channels, frames]`.
    #[test]
    fn audio_latents_transpose_per_channel() {
        let mut cfg = DitConfig::h3_33b();
        cfg.audio_latents_dim = 2;
        // 2 channels x 3 frames x 2 latent channels.
        let rows: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let out = audio_rows_to_latents(&cfg, &rows, 3);
        // Channel 0 frames are rows 0..3 -> [[0,2,4],[1,3,5]].
        assert_eq!(&out[..6], &[0.0, 2.0, 4.0, 1.0, 3.0, 5.0]);
        // Channel 1 frames are rows 3..6 -> [[6,8,10],[7,9,11]].
        assert_eq!(&out[6..], &[6.0, 8.0, 10.0, 7.0, 9.0, 11.0]);
    }

    /// `steps` must mean model evaluations. The scheduler counts SIGMAS, and
    /// the off-by-one silently runs a 4-step distillation at 3 steps.
    #[test]
    fn requested_steps_are_model_evaluations() {
        let cfg = DitConfig::h3_33b();
        for steps in [2usize, 4, 8] {
            let plan = DenoisePlan::build::<std::convert::Infallible>(&cfg, 4, 2, 4, 4, 3, steps)
                .expect("plan");
            assert_eq!(
                plan.schedule.num_steps(),
                steps,
                "{steps} requested steps must drive {steps} evaluations"
            );
        }
    }

    #[test]
    fn noise_is_seed_reproducible_and_split_by_modality() {
        let c = DitConfig::h3_33b();
        let (v1, a1) = initial_noise(&c, 10, 4, 7);
        let (v2, a2) = initial_noise(&c, 10, 4, 7);
        assert_eq!(v1, v2);
        assert_eq!(a1, a2);
        assert_eq!(v1.len(), 10 * c.video_patch_dim());
        assert_eq!(a1.len(), 4 * c.audio_latents_dim);
        assert_ne!(initial_noise(&c, 10, 4, 8).0, v1);
    }
}
