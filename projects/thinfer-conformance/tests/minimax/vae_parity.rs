//! MiniMax-H3 video VAE decoder parity: the engine's ViT decoder (`decode`, the
//! production chunked path) vs FastVideo's torch `AutoencoderKLMiniMaxH3` over the
//! SAME Comfy-Org fp16 file (`minimax/gen_vae_ref.py`, bf16 on CPU).
//!
//! The latents are not synthetic noise: the reference ENCODES a moving scene, so
//! the decode must also resemble that input. Both engine-vs-torch parity and
//! engine-vs-input reconstruction are reported; parity is the gate, first at a
//! ladder of ViT depths (localizes a block bug), then for the full decode.

#![cfg(feature = "minimax-e2e")]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use thinfer_core::backend::{PowerPreference, WgpuBackend, WgpuConfig};
use thinfer_core::format::safetensors::SafetensorsSource;
use thinfer_core::policy::ResidencyBudget;
use thinfer_core::residency::WeightResidency;
use thinfer_core::weight::{WeightId, WeightReader, WeightSource, decode_into};
use thinfer_core::workspace::Workspace;
use thinfer_models::minimax::manifest::{self, role};
use thinfer_models::minimax::video_vae::{
    DecoderPipelines, TilePolicy, VideoVaeConfig, decode, decode_tiles, load_consts,
    register_handles,
};
use thinfer_native::MmapFileOpener;
use thinfer_native::cache;

use crate::parity_util::{read_f32, report};

/// bf16 torch on CPU vs the engine's f32-act / bf16-weight decoder.
const SLOPE_TOL: f64 = 0.03;
const REL_TOL: f64 = 0.05;
/// Block depths the reference dumps the output head at (`head{d}.bin`).
const HEAD_DEPTHS: [usize; 6] = [0, 1, 2, 6, 18, 36];

#[tokio::test(flavor = "current_thread")]
async fn vae_parity() {
    let _trace = thinfer_core::trace::init_from_env();
    let fr = manifest::MANIFEST
        .get(role::VIDEO_VAE)
        .expect("video vae role");
    let Some(path) = cache::resolve(fr) else {
        eprintln!("skipped[minimax vae_parity]: video VAE not in HF cache");
        return;
    };
    let dir = ensure_pyref(&path);
    let meta: Vec<usize> = std::fs::read_to_string(dir.join("meta.txt"))
        .expect("meta.txt")
        .split_whitespace()
        .map(|t| t.parse().expect("meta int"))
        .collect();
    let [t, h, w, f_ref, _, _] = meta[..] else {
        panic!("meta.txt needs 6 fields")
    };

    let backend = Arc::new(
        WgpuBackend::new_with_config(WgpuConfig {
            power_preference: PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await
        .expect("wgpu backend"),
    );
    // Opt-in CUDA (`THINFER_CUDA=on`): the same gate then measures the
    // accelerator path (deferred block sequences) against the same pyref.
    if std::env::var("THINFER_CUDA").is_ok_and(|v| v == "on") {
        match thinfer_cuda::install(&backend) {
            Ok((name, sm)) => eprintln!("cuda hot ops ON: {name} sm_{}{}", sm.0, sm.1),
            Err(e) => eprintln!("cuda hot ops requested but unavailable: {e}"),
        }
    }
    let cfg = VideoVaeConfig::h3();
    let opener = MmapFileOpener::new(&path).await.expect("open video vae");
    let src = SafetensorsSource::open(opener)
        .await
        .expect("parse video vae");
    let c = cfg.latent_channels;
    let mean = read_vec(&src, "latents_mean", c).await;
    let std = read_vec(&src, "latents_std", c).await;
    let residency = WeightResidency::new(
        src,
        ResidencyBudget {
            ram_bytes: 16 << 30,
            vram_bytes: 6 << 30,
        },
    );
    let handles = register_handles(&cfg, &residency).expect("register decoder");
    let pipelines = DecoderPipelines::compile(&backend)
        .await
        .expect("compile decoder");
    let ws = Workspace::new(Arc::clone(&backend), Arc::clone(residency.arbiter()));
    let consts = load_consts(&cfg, &backend, &residency, &handles)
        .await
        .expect("decoder consts");

    // Reference latents are DiT space `[C, T, h, w]`; `decode` takes them
    // denormalized and channels-last `[T, h, w, C]`.
    let lat = read_f32(&dir.join("latents.bin"));
    let plane = t * h * w;
    let mut z = vec![0f32; lat.len()];
    for ch in 0..c {
        for i in 0..plane {
            z[i * c + ch] = lat[ch * plane + i] * std[ch] + mean[ch];
        }
    }

    // Depth ladder: the latents are exactly one clip and one tile, so a single
    // `decode_tiles` pass with the block stack truncated to `d` is comparable to
    // the reference's output head after `d` blocks. The shallowest failing depth
    // localizes a divergence before unpatchify/stitch can blur it.
    let mut failures = Vec::new();
    for d in HEAD_DEPTHS {
        let mut hd = register_handles(&cfg, &residency).expect("register decoder");
        hd.blocks.truncate(d);
        let got = decode_tiles(
            &cfg, &backend, &pipelines, &residency, &ws, &hd, &consts, &z, 1, t, h, w,
        )
        .await
        .expect("decode_tiles");
        let (slope, rel) = report(
            &format!("head{d}"),
            &read_f32(&dir.join(format!("head{d}.bin"))),
            &got,
        );
        if !slope.is_finite() || (slope - 1.0).abs() > SLOPE_TOL || rel > REL_TOL {
            failures.push(format!("head{d}: slope {slope:.4} rel {:.2}%", rel * 100.0));
        }
    }
    // Batching invariant: two tiles decoded in one pass must equal each decoded
    // alone (tiles only share a pass, never attention).
    {
        let half = w / 2;
        let gather = |x0: usize| -> Vec<f32> {
            let mut v = Vec::with_capacity(t * h * half * c);
            for ti in 0..t {
                for hi in 0..h {
                    let base = ((ti * h + hi) * w + x0) * c;
                    v.extend_from_slice(&z[base..base + half * c]);
                }
            }
            v
        };
        let (a, b) = (gather(0), gather(w - half));
        let pair: Vec<f32> = a.iter().chain(&b).copied().collect();
        let both = decode_tiles(
            &cfg, &backend, &pipelines, &residency, &ws, &handles, &consts, &pair, 2, t, h, half,
        )
        .await
        .expect("decode_tiles batch 2");
        let alone = decode_tiles(
            &cfg, &backend, &pipelines, &residency, &ws, &handles, &consts, &b, 1, t, h, half,
        )
        .await
        .expect("decode_tiles batch 1");
        let (slope, rel) = report("batch2[1] vs alone", &alone, &both[both.len() / 2..]);
        if !slope.is_finite() || (slope - 1.0).abs() > 1e-3 || rel > 1e-3 {
            failures.push(format!(
                "batched tile differs from its solo decode: slope {slope:.4} rel {:.3}%",
                rel * 100.0
            ));
        }
    }

    let out = decode(
        &cfg,
        &backend,
        &pipelines,
        &residency,
        &ws,
        &handles,
        &consts,
        TilePolicy::default(),
        &z,
        t,
        h,
        w,
    )
    .await
    .expect("decode");

    // Engine pixels -> [3, F, H, W] in [0, 1], trimmed to the reference's frames.
    let planar = out.to_signed_planar();
    let full = out.frames * out.height * out.width;
    let keep = f_ref.min(out.frames) * out.height * out.width;
    let got: Vec<f32> = (0..3)
        .flat_map(|ch| {
            planar[ch * full..ch * full + keep]
                .iter()
                .map(|v| (v + 1.0) * 0.5)
        })
        .collect();
    let trim = |v: Vec<f32>| -> Vec<f32> {
        let full = f_ref * out.height * out.width;
        (0..3)
            .flat_map(|ch| v[ch * full..ch * full + keep].to_vec())
            .collect()
    };
    // Kept for offline diagnosis (frame alignment, tile seams).
    let got_bytes: Vec<u8> = got.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(dir.join("got.bin"), got_bytes).expect("write got.bin");
    let exp = trim(read_f32(&dir.join("decoded.bin")));
    let input = trim(read_f32(&dir.join("input.bin")));
    eprintln!(
        "engine {}x{}x{} frames (reference {f_ref})",
        out.width, out.height, out.frames
    );
    report("vs input (info)", &input, &got);
    report("torch vs input (info)", &input, &exp);
    let (slope, rel) = report("vs torch", &exp, &got);
    if !slope.is_finite() || (slope - 1.0).abs() > SLOPE_TOL || rel > REL_TOL {
        failures.push(format!("decode: slope {slope:.4} rel {:.2}%", rel * 100.0));
    }
    assert!(
        failures.is_empty(),
        "minimax vae parity failed:
{}",
        failures.join(
            "
"
        )
    );
}

async fn read_vec<S: WeightSource>(src: &S, name: &str, n: usize) -> Vec<f32> {
    let id = WeightId(name.into());
    let enc = src
        .catalog()
        .get(&id)
        .and_then(|e| e.encoding)
        .expect("encoding");
    let mut r = src.open(&id).await.expect("open");
    let mut bytes = vec![0u8; r.len() as usize];
    r.read_at(0, &mut bytes).await.expect("read");
    let mut out = vec![0u8; n * 4];
    decode_into(enc, &bytes, &mut out).expect("decode");
    out.chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

fn ensure_pyref(vae: &Path) -> PathBuf {
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("minimax_vae_parity");
    std::fs::create_dir_all(&tmp).expect("tmpdir");
    let marker = tmp.join("done.txt");
    if marker.exists() {
        eprintln!(
            "minimax vae-parity: reusing cached pyref dumps ({})",
            tmp.display()
        );
        return tmp;
    }
    let py_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python");
    let status = Command::new("uv")
        .args([
            "run",
            "--directory",
            py_dir.to_str().unwrap(),
            "python",
            "-m",
        ])
        .arg("thinfer_pytorch_ref.minimax.gen_vae_ref")
        .arg("--vae")
        .arg(vae)
        .arg("--out")
        .arg(&tmp)
        .env("OMP_NUM_THREADS", "4")
        .status()
        .expect("failed to spawn `uv run` (is uv installed?)");
    assert!(status.success(), "minimax vae pyref failed");
    std::fs::write(&marker, "ok").expect("write marker");
    tmp
}
