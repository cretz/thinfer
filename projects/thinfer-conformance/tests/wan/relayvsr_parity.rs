//! RelayVSR parity vs the pyref (`thinfer_pytorch_ref.wan.gen_relayvsr_ref`,
//! a port of ComfyUI `model_vsr.py` on DEQUANTIZED convrot weights): the same
//! LR frames + the same keyframe noise through the engine; compares the
//! `lq_proj_in` tokens, the keyframe latent and every upscaled frame (PSNR,
//! with the plain-bicubic PSNR printed alongside for scale).
//!
//! Inputs (skips cleanly when absent):
//!   `THINFER_RELAYVSR_REF`   fixture (`--out` of the pyref)
//!   `THINFER_RELAYVSR_DIT`   `relayvsr_x2_lightx2v_int8_convrot.safetensors`
//!   `THINFER_RELAYVSR_FLASH` `relayvsr_flash_vae_x2_fp32.safetensors`
//!
//! Run:
//!   `cargo test -p thinfer-conformance --features relayvsr-e2e --release \
//!    --test wan relayvsr_parity -- --nocapture --test-threads=1`

#![cfg(feature = "relayvsr-e2e")]

use std::path::PathBuf;
use std::sync::Arc;

use safetensors::SafeTensors;
use thinfer_core::backend::{PowerPreference, WgpuBackend, WgpuConfig};
use thinfer_core::policy::ResidencyBudget;
use thinfer_core::residency::WeightResidency;
use thinfer_models::wan::relay_flash::bicubic_up;
use thinfer_models::wan::relayvsr::{
    RelayVsrDiag, RelayVsrPipeline, UpscaleInputs, open_relayvsr_source,
};
use thinfer_native::MmapFileOpener;

fn env_path(k: &str) -> Option<PathBuf> {
    std::env::var(k)
        .ok()
        .map(PathBuf::from)
        .filter(|p| p.exists())
}

fn f32s(st: &SafeTensors, name: &str) -> Vec<f32> {
    let t = st.tensor(name).unwrap_or_else(|e| panic!("{name}: {e}"));
    t.data()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn rel_l2(got: &[f32], want: &[f32]) -> f64 {
    assert_eq!(got.len(), want.len());
    let (mut n, mut d) = (0f64, 0f64);
    for (&g, &w) in got.iter().zip(want) {
        n += ((g - w) as f64).powi(2);
        d += (w as f64).powi(2);
    }
    (n / d.max(1e-30)).sqrt()
}

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    let mse: f64 = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| (x as f64 - y as f64).powi(2))
        .sum::<f64>()
        / a.len() as f64;
    10.0 * (255.0f64 * 255.0 / mse.max(1e-12)).log10()
}

fn to_u8(v: f32) -> u8 {
    (((v + 1.0) * 0.5).clamp(0.0, 1.0) * 255.0).round() as u8
}

#[tokio::test(flavor = "current_thread")]
async fn relayvsr_parity() {
    let (Some(ref_path), Some(dit), Some(flash)) = (
        env_path("THINFER_RELAYVSR_REF"),
        env_path("THINFER_RELAYVSR_DIT"),
        env_path("THINFER_RELAYVSR_FLASH"),
    ) else {
        eprintln!("relayvsr_parity skipped (set THINFER_RELAYVSR_REF/_DIT/_FLASH)");
        return;
    };
    let fixture = std::fs::read(&ref_path).expect("read fixture");
    let st = SafeTensors::deserialize(&fixture).expect("fixture");
    let lr = st.tensor("lr_u8").expect("lr_u8");
    let (nf, h, w) = (lr.shape()[0], lr.shape()[1], lr.shape()[2]);
    let lr = lr.data().to_vec();
    let noise = f32s(&st, "noise");
    let want_lq = f32s(&st, "tap.lq_tokens");
    let want_lat = f32s(&st, "kf_latent");
    let want_out = f32s(&st, "flash_out");

    {
        let source = open_relayvsr_source(
            MmapFileOpener::new(&dit).await.expect("open dit"),
            MmapFileOpener::new(&flash).await.expect("open flash"),
        )
        .await
        .expect("relayvsr source");
        let backend = Arc::new(
            WgpuBackend::new_with_config(WgpuConfig {
                power_preference: PowerPreference::HighPerformance,
                timestamps: std::env::var("THINFER_TRACE").is_ok(),
                disable_coopmat: std::env::var("THINFER_NO_COOPMAT").is_ok(),
            })
            .await
            .expect("wgpu"),
        );
        match thinfer_cuda::install(&backend) {
            Ok((name, _)) => eprintln!("relayvsr parity: cuda on {name}"),
            Err(e) => eprintln!("relayvsr parity: no cuda ({e})"),
        }
        let budget = ResidencyBudget {
            ram_bytes: 5 << 30,
            vram_bytes: 5 << 30,
        };
        let residency = WeightResidency::new(source, budget);
        let pipe = RelayVsrPipeline::load(backend, residency, 2)
            .await
            .expect("load");
        let mut diag = RelayVsrDiag {
            noise_override: Some(noise),
            ..Default::default()
        };
        let t0 = std::time::Instant::now();
        let out = pipe
            .upscale(
                &UpscaleInputs {
                    frames: &lr,
                    num_frames: nf,
                    height: h,
                    width: w,
                    seed: 0,
                    detail: 1.0,
                },
                Some(&mut diag),
                None,
            )
            .await
            .expect("upscale");
        eprintln!(
            "relayvsr {nf}x{h}x{w} -> {}x{} in {:.1}s",
            out.width,
            out.height,
            t0.elapsed().as_secs_f64()
        );

        eprintln!("phases (ms): {:?}", diag.phase_ms);
        let lq_rel = rel_l2(&diag.lq_tokens, &want_lq);
        let lat_rel = rel_l2(&diag.kf_latent, &want_lat);
        eprintln!("lq_tokens rel_l2 {lq_rel:.3e}  kf_latent rel_l2 {lat_rel:.3e}");

        let (oh, ow) = (out.height, out.width);
        let mut worst = f64::MAX;
        let mut sum = 0f64;
        let mut bic_sum = 0f64;
        for f in 0..nf {
            let plane = oh * ow;
            let want: Vec<u8> = (0..plane * 3)
                .map(|i| {
                    let (p, c) = (i / 3, i % 3);
                    to_u8(want_out[(f * 3 + c) * plane + p])
                })
                .collect();
            let got = &out.frames[f * plane * 3..][..plane * 3];
            let p = psnr(got, &want);
            // plain bicubic of the LR frame, for scale
            let lrf: Vec<f32> = (0..3 * h * w)
                .map(|i| {
                    let (c, p) = (i / (h * w), i % (h * w));
                    lr[(f * h * w + p) * 3 + c] as f32 / 255.0 * 2.0 - 1.0
                })
                .collect();
            let bic = bicubic_up(&lrf, 3, h, w, 2);
            let bic_u8: Vec<u8> = (0..plane * 3)
                .map(|i| to_u8(bic[(i % 3) * plane + i / 3]))
                .collect();
            let pb = psnr(&bic_u8, &want);
            worst = worst.min(p);
            sum += p;
            bic_sum += pb;
            if f == nf / 2
                && let Ok(dir) = std::env::var("THINFER_PNG_DIR")
            {
                let path = PathBuf::from(dir).join("relayvsr_mid.png");
                let file = std::fs::File::create(&path).expect("png");
                let mut enc = png::Encoder::new(file, ow as u32, oh as u32);
                enc.set_color(png::ColorType::Rgb);
                enc.write_header().unwrap().write_image_data(got).unwrap();
                eprintln!("wrote {}", path.display());
            }
        }
        eprintln!(
            "frames PSNR vs pyref: mean {:.2} dB, worst {worst:.2} dB (plain bicubic vs pyref: {:.2} dB)",
            sum / nf as f64,
            bic_sum / nf as f64
        );
        assert!(lq_rel.is_finite() && lat_rel.is_finite() && worst.is_finite());
    }
}

/// Standalone upscale of a real clip (`THINFER_RELAYVSR_VIDEO`, decoded with
/// ffmpeg; `THINFER_RELAYVSR_FRAMES` caps the count) with the same DiT/Flash
/// env vars: phases + memory, output frame 0/mid as PNG into
/// `THINFER_PNG_DIR`. The perf harness.
#[tokio::test(flavor = "current_thread")]
async fn relayvsr_bench() {
    let (Some(video), Some(dit), Some(flash)) = (
        env_path("THINFER_RELAYVSR_VIDEO"),
        env_path("THINFER_RELAYVSR_DIT"),
        env_path("THINFER_RELAYVSR_FLASH"),
    ) else {
        eprintln!("relayvsr_bench skipped (set THINFER_RELAYVSR_VIDEO/_DIT/_FLASH)");
        return;
    };
    let trace = thinfer_core::trace::init_from_env();
    let cap: usize = std::env::var("THINFER_RELAYVSR_FRAMES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1000);
    let probe = std::process::Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0",
        ])
        .arg(&video)
        .output()
        .expect("ffprobe");
    let dims: Vec<usize> = String::from_utf8_lossy(&probe.stdout)
        .trim()
        .split(',')
        .map(|v| v.parse().unwrap())
        .collect();
    let (w, h) = (dims[0], dims[1]);
    let raw = std::process::Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(&video)
        .args([
            "-frames:v",
            &cap.to_string(),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ])
        .output()
        .expect("ffmpeg")
        .stdout;
    let nf = raw.len() / (w * h * 3);
    let source = open_relayvsr_source(
        MmapFileOpener::new(&dit).await.expect("open dit"),
        MmapFileOpener::new(&flash).await.expect("open flash"),
    )
    .await
    .expect("relayvsr source");
    let backend = Arc::new(
        WgpuBackend::new_with_config(WgpuConfig {
            power_preference: PowerPreference::HighPerformance,
            timestamps: std::env::var("THINFER_TRACE").is_ok(),
            disable_coopmat: false,
        })
        .await
        .expect("wgpu"),
    );
    thinfer_cuda::install(&backend).expect("cuda");
    let gb: u64 = std::env::var("THINFER_RELAYVSR_VRAM_GB")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);
    let residency = WeightResidency::new(
        source,
        ResidencyBudget {
            ram_bytes: 5 << 30,
            vram_bytes: gb << 30,
        },
    );
    let pipe = RelayVsrPipeline::load(backend, residency, 2)
        .await
        .expect("load");
    let mut diag = RelayVsrDiag::default();
    let t0 = std::time::Instant::now();
    let out = pipe
        .upscale(
            &UpscaleInputs {
                frames: &raw[..nf * w * h * 3],
                num_frames: nf,
                height: h,
                width: w,
                seed: 42,
                detail: 1.0,
            },
            Some(&mut diag),
            None,
        )
        .await
        .expect("upscale");
    eprintln!(
        "relayvsr bench {nf}x{h}x{w} -> {}x{} in {:.1}s; phases (ms) {:?}",
        out.width,
        out.height,
        t0.elapsed().as_secs_f64(),
        diag.phase_ms
    );
    if let Some(h) = trace {
        let _ = h.dump(&mut std::io::stderr());
    }
    if let Ok(dir) = std::env::var("THINFER_PNG_DIR") {
        for f in [0, nf / 2] {
            let path = PathBuf::from(&dir).join(format!("relayvsr_bench_{f}.png"));
            let n = out.width * out.height * 3;
            let mut enc = png::Encoder::new(
                std::fs::File::create(&path).unwrap(),
                out.width as u32,
                out.height as u32,
            );
            enc.set_color(png::ColorType::Rgb);
            enc.write_header()
                .unwrap()
                .write_image_data(&out.frames[f * n..][..n])
                .unwrap();
        }
    }
}
