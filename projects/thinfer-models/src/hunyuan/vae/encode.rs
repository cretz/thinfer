//! HunyuanVideo 1.5 VAE ENCODER, single-frame path (the causal I2V first-frame
//! conditioning). Ground truth: `minWM/HY15/trainer/models/hyvideo/vae/
//! hunyuanvideo_15_vae_w_cache.py::{Encoder, Downsample}` at `T=1`.
//!
//! Graph: conv_in(3->128) -> 4 down stages (2 channel-preserving resnets +
//! hierarchical Downsample: k3 conv to `out/factor`, spatial pixel-unshuffle,
//! temporal levels duplicate at T=1, plus a group-mean shortcut) -> stage 4
//! resnets -> mid(resnet, attn, resnet) -> `h = conv_out(silu(norm(h))) +
//! group_mean_16(h)` -> mean half of `[64]` * SCALING_FACTOR.
//!
//! Runs ONCE per generation on one 480p frame, so the pixel-unshuffle /
//! group-mean rearranges round-trip through the host between per-stage GPU
//! scopes (bounded peaks, zero new kernels). Determinism: we take the
//! latent-distribution MEAN (upstream `.sample()` draws noise; the mean is the
//! standard deterministic inference choice).

use super::*;

struct StageW {
    blocks: Vec<ResnetW>,
    /// `Some((conv, out_ch, temporal))`: the Downsample conv + its target
    /// channel count + whether the level is temporal (factor 8 vs 4).
    down: Option<(ConvW, u32, bool)>,
}

struct EncoderW {
    conv_in: ConvW,
    stages: Vec<StageW>,
    mid: MidW,
    norm_out: RmsW,
    conv_out: ConvW,
}

impl EncoderW {
    fn new() -> Self {
        let chans = super::super::config::vae::BLOCK_OUT_CHANNELS;
        let spatial_levels = 4; // log2(FFACTOR_SPATIAL 16)
        let temporal_from = 2; // log2(16 / FFACTOR_TEMPORAL 4)
        let mut stages = Vec::with_capacity(chans.len());
        for (i, _ch) in chans.iter().enumerate() {
            let blocks = (0..super::super::config::vae::LAYERS_PER_BLOCK)
                .map(|b| resnet_w(&format!("encoder.down.{i}.block.{b}")))
                .collect();
            let down = (i < spatial_levels).then(|| {
                (
                    conv_w(&format!("encoder.down.{i}.downsample.conv.conv")),
                    chans[i + 1] as u32,
                    i >= temporal_from,
                )
            });
            stages.push(StageW { blocks, down });
        }
        Self {
            conv_in: conv_w("encoder.conv_in.conv"),
            stages,
            mid: MidW {
                block_1: resnet_w("encoder.mid.block_1"),
                attn_1: attn_w("encoder.mid.attn_1"),
                block_2: resnet_w("encoder.mid.block_2"),
            },
            norm_out: rms_w("encoder.norm_out"),
            conv_out: conv_w("encoder.conv_out.conv"),
        }
    }
}

struct StageH {
    blocks: Vec<ResnetH>,
    down: Option<(ConvH, u32, bool)>,
}

pub struct HunyuanVaeEncoder {
    conv_in: ConvH,
    stages: Vec<StageH>,
    mid: MidH,
    norm_out: RmsH,
    conv_out: ConvH,
}

impl HunyuanVaeEncoder {
    pub fn new<S: WeightSource>(res: &WeightResidency<S>) -> Result<Self, LoadError> {
        let w = EncoderW::new();
        let mut stages = Vec::with_capacity(w.stages.len());
        for sw in &w.stages {
            let blocks = sw
                .blocks
                .iter()
                .map(|b| reg_resnet(res, b))
                .collect::<Result<Vec<_>, _>>()?;
            let down = match &sw.down {
                Some((c, out, temporal)) => Some((reg_conv(res, c)?, *out, *temporal)),
                None => None,
            };
            stages.push(StageH { blocks, down });
        }
        Ok(Self {
            conv_in: reg_conv(res, &w.conv_in)?,
            stages,
            mid: MidH {
                block_1: reg_resnet(res, &w.mid.block_1)?,
                attn_1: reg_attn(res, &w.mid.attn_1)?,
                block_2: reg_resnet(res, &w.mid.block_2)?,
            },
            norm_out: reg_rms(res, &w.norm_out)?,
            conv_out: reg_conv(res, &w.conv_out)?,
        })
    }

    /// Encode one normalized frame `[3, H, W]` (values in [-1, 1], H/W
    /// multiples of 16) to the conditioning latent `[32, H/16, W/16]`
    /// (distribution mean, scaled by [`SCALING_FACTOR`]).
    pub async fn encode_frame<S: WeightSource>(
        &self,
        backend: &WgpuBackend,
        residency: &WeightResidency<S>,
        workspace: &Workspace<WgpuBackend>,
        pl: &HunyuanVaePipelines,
        frame: &[f32],
        height: usize,
        width: usize,
    ) -> Result<Vec<f32>, HunyuanVaeError<S::Error>> {
        assert_eq!(frame.len(), 3 * height * width, "frame [3,H,W]");
        assert!(
            height.is_multiple_of(16) && width.is_multiple_of(16),
            "dims % 16"
        );
        let act = pl.act;
        let act_size = pl.act_size();

        // One scope per GPU segment; host rearranges in between.
        let mut host: Vec<f32> = frame.to_vec();
        let mut s = Shape {
            c: 3,
            t: 1,
            h: height as u32,
            w: width as u32,
        };

        // Segment A: conv_in + stage resnets + down conv; returns (h_conv, x_in)
        // for the host-side unshuffle/shortcut math per downsampling stage.
        for (si, stage) in self.stages.iter().enumerate() {
            let mut pins: Vec<GpuView> = Vec::new();
            let conv_in = (si == 0).then_some(&self.conv_in);
            let ci = match conv_in {
                Some(c) => Some(acquire_conv(residency, backend, *c, &mut pins).await?),
                None => None,
            };
            let mut blocks = Vec::new();
            for b in &stage.blocks {
                blocks.push(acquire_resnet(residency, backend, b, &mut pins).await?);
            }
            let down = match &stage.down {
                Some((c, out, temporal)) => Some((
                    acquire_conv(residency, backend, *c, &mut pins).await?,
                    *out,
                    *temporal,
                )),
                None => None,
            };

            let up_bytes = act_upload_bytes(act, &host);
            let x_up = workspace.alloc(up_bytes.len() as u64)?;
            backend.write_buffer(x_up.id(), 0, &up_bytes)?;

            // conv output (pre-rearrange) + the resnet output (shortcut input).
            let (conv_out_ws, conv_s, x_out_ws, x_s);
            {
                let scope = workspace.batch();
                let mut x = scope.import_copy(x_up.as_buf_ref());
                if let Some(ci) = &ci {
                    let (y, ns) = conv3d_k3(&scope, pl, x, s, ci, 128)?;
                    x = y;
                    s = ns;
                }
                for b in &blocks {
                    x = resnet(&scope, pl, x, s, b)?;
                }
                x_out_ws = persist(&scope, workspace, x, s, act_size)?;
                x_s = s;
                match &down {
                    Some((c, out, temporal)) => {
                        let factor = if *temporal { 8 } else { 4 };
                        let (y, ns) = conv3d_k3(&scope, pl, x, s, c, out / factor)?;
                        conv_out_ws = Some(persist(&scope, workspace, y, ns, act_size)?);
                        conv_s = Some(ns);
                    }
                    None => {
                        conv_out_ws = None;
                        conv_s = None;
                    }
                }
                scope.submit_void().await?;
            }

            match (&down, conv_out_ws, conv_s) {
                (Some((_, out, temporal)), Some(cw), Some(cs)) => {
                    // Host: pixel-unshuffle h + group-mean shortcut, x = h + sc.
                    let h = read_host(backend, &cw, cs, act).await?;
                    let xin = read_host(backend, &x_out_ws, x_s, act).await?;
                    let mut hr = spatial_unshuffle(&h, cs.c as usize, cs.h as usize, cs.w as usize);
                    let mut h_ch = cs.c as usize * 4;
                    if *temporal {
                        // T=1: duplicate (upstream `cat([h, h], dim=1)`).
                        let mut dup = Vec::with_capacity(hr.len() * 2);
                        dup.extend_from_slice(&hr);
                        dup.extend_from_slice(&hr);
                        hr = dup;
                        h_ch *= 2;
                    }
                    assert_eq!(h_ch, *out as usize, "downsample channel math");
                    let sc =
                        spatial_unshuffle(&xin, x_s.c as usize, x_s.h as usize, x_s.w as usize);
                    let sc_ch = x_s.c as usize * 4;
                    let group = sc_ch / h_ch;
                    // Post-unshuffle spatial dims are HALVED relative to the
                    // conv output (the 2x2 moved into channels).
                    let hw = ((cs.h / 2) * (cs.w / 2)) as usize;
                    let sc = group_mean(&sc, h_ch, group, hw);
                    for (a, b) in hr.iter_mut().zip(sc.iter()) {
                        *a += b;
                    }
                    host = hr;
                    s = Shape {
                        c: *out,
                        t: 1,
                        h: cs.h / 2,
                        w: cs.w / 2,
                    };
                }
                _ => {
                    host = read_host(backend, &x_out_ws, x_s, act).await?;
                }
            }
            drop(pins);
        }

        // Segment B: mid + end (norm/silu/conv_out + group-mean shortcut).
        let z_out = 2 * LATENT_CHANNELS as u32; // 64
        let (h_ws, sc_host);
        {
            let mut pins: Vec<GpuView> = Vec::new();
            let mid = MidBufs {
                block_1: acquire_resnet(residency, backend, &self.mid.block_1, &mut pins).await?,
                attn_1: acquire_attn(residency, backend, &self.mid.attn_1, &mut pins).await?,
                block_2: acquire_resnet(residency, backend, &self.mid.block_2, &mut pins).await?,
            };
            let norm_out = acquire_rms(residency, backend, self.norm_out, &mut pins).await?;
            let conv_out = acquire_conv(residency, backend, self.conv_out, &mut pins).await?;

            let up_bytes = act_upload_bytes(act, &host);
            let x_up = workspace.alloc(up_bytes.len() as u64)?;
            backend.write_buffer(x_up.id(), 0, &up_bytes)?;
            let mask_scratch = workspace.alloc(act_size as u64)?;
            {
                let scope = workspace.batch();
                let mut x = scope.import_copy(x_up.as_buf_ref());
                x = resnet(&scope, pl, x, s, &mid.block_1)?;
                let mask = scope.import_copy(mask_scratch.as_buf_ref());
                x = mid_attention(&scope, pl, x, s, &mid.attn_1, mask)?;
                x = resnet(&scope, pl, x, s, &mid.block_2)?;
                let pre = persist(&scope, workspace, x, s, act_size)?;
                let n = rmsnorm3d(&scope, pl, x, s, &norm_out)?;
                let a = silu(&scope, pl, n, s)?;
                let (y, ys) = conv3d_k3(&scope, pl, a, s, &conv_out, z_out)?;
                let yp = persist(&scope, workspace, y, ys, act_size)?;
                scope.submit_void().await?;
                sc_host = read_host(backend, &pre, s, act).await?;
                h_ws = (yp, ys);
            }
            drop(pins);
        }

        // Host end: h += group_mean_16(pre); z = mean half * SCALING_FACTOR.
        let (yp, ys) = h_ws;
        let mut h = read_host(backend, &yp, ys, act).await?;
        let hw = (ys.h * ys.w) as usize;
        let sc = group_mean(&sc_host, z_out as usize, s.c as usize / z_out as usize, hw);
        for (a, b) in h.iter_mut().zip(sc.iter()) {
            *a += b;
        }
        let mut z = h[..LATENT_CHANNELS * hw].to_vec();
        for v in &mut z {
            *v *= SCALING_FACTOR;
        }
        Ok(z)
    }
}

async fn read_host(
    backend: &WgpuBackend,
    buf: &WsBuf<WgpuBackend>,
    s: Shape,
    act: ActDtype,
) -> Result<Vec<f32>, WgpuError> {
    let bytes = backend
        .read_buffer(buf.id(), 0, s.elems() as u64 * act.bytes_per_elem())
        .await?;
    Ok(act_readback_to_f32(act, &bytes, s.elems() as usize))
}

/// einops `c (h 2) (w 2) -> ((r2 r3) c) h w`: new channel = (r2*2+r3)*C + c.
fn spatial_unshuffle(x: &[f32], c: usize, h: usize, w: usize) -> Vec<f32> {
    assert_eq!(x.len(), c * h * w);
    let (ho, wo) = (h / 2, w / 2);
    let mut out = vec![0.0f32; 4 * c * ho * wo];
    for ci in 0..c {
        for y in 0..ho {
            for xx in 0..wo {
                for r2 in 0..2 {
                    for r3 in 0..2 {
                        let src = ci * h * w + (2 * y + r2) * w + (2 * xx + r3);
                        let oc = (r2 * 2 + r3) * c + ci;
                        out[oc * ho * wo + y * wo + xx] = x[src];
                    }
                }
            }
        }
    }
    out
}

/// `view(out_c, group, hw).mean(dim=1)`: channel groups are CONTIGUOUS
/// (out channel `o` averages input channels `[o*group, (o+1)*group)`).
fn group_mean(x: &[f32], out_c: usize, group: usize, hw: usize) -> Vec<f32> {
    assert_eq!(x.len(), out_c * group * hw);
    let mut out = vec![0.0f32; out_c * hw];
    let inv = 1.0 / group as f32;
    for o in 0..out_c {
        for g in 0..group {
            let base = (o * group + g) * hw;
            let dst = o * hw;
            for p in 0..hw {
                out[dst + p] += x[base + p];
            }
        }
    }
    for v in &mut out {
        *v *= inv;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unshuffle_and_group_mean() {
        // 1 channel, 2x2 image -> 4 channels of 1x1 in (r2 r3) order.
        let x = [1.0, 2.0, 3.0, 4.0];
        let u = spatial_unshuffle(&x, 1, 2, 2);
        assert_eq!(u, vec![1.0, 2.0, 3.0, 4.0]);
        // group_mean over contiguous channel pairs.
        let m = group_mean(&u, 2, 2, 1);
        assert_eq!(m, vec![1.5, 3.5]);
    }
}
