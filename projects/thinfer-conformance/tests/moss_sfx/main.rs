//! MOSS-SoundEffect v2.0 parity vs the reference modules. Gated on `sfx-e2e`;
//! needs the repo in the HF cache and the reference tensors from
//! `parity_ref.py` in `THINFER_SFX_PARITY` (skips otherwise): prompt `ids`,
//! text `hidden` (bf16 Qwen3), DiT velocities `v_{cond,uncond}_{0,1}` of `x`
//! at `sigmas` (f32 weights, bf16 autocast), and the DAC `wav` of `lat` (f32).

#![cfg(feature = "sfx-e2e")]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use thinfer_core::backend::{PowerPreference, WgpuBackend, WgpuConfig};
use thinfer_core::format::pytorch::PytorchSource;
use thinfer_core::format::safetensors::{SafetensorsSource, ShardedSafetensorsSource};
use thinfer_core::format::union::UnionSource;
use thinfer_core::policy::ResidencyBudget;
use thinfer_core::residency::WeightResidency;
use thinfer_core::tokenizer::Tokenizer;
use thinfer_models::common::weight_norm::WeightNormSource;
use thinfer_models::moss_sfx::manifest::{MANIFEST, role};
use thinfer_models::moss_sfx::pipeline::SfxModel;
use thinfer_models::moss_sfx::{DECODE_MARGIN, HOP, LATENT_CH, decode_window, prompt_text};
use thinfer_native::MmapFileOpener;

const PROMPT: &str = "A dog barking loudly in a park, birds chirping in the background.";

fn parity_dir() -> Option<PathBuf> {
    let d = PathBuf::from(std::env::var("THINFER_SFX_PARITY").ok()?);
    d.join("wav.npy").exists().then_some(d)
}

/// A little-endian 4-byte-dtype `.npy` payload.
fn npy(p: &Path) -> Vec<u8> {
    let b = std::fs::read(p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
    let hl = u16::from_le_bytes([b[8], b[9]]) as usize;
    let hdr = std::str::from_utf8(&b[10..10 + hl]).unwrap();
    assert!(hdr.contains("'fortran_order': False"), "{}", p.display());
    b[10 + hl..].to_vec()
}

fn f32s(d: &Path, name: &str) -> Vec<f32> {
    bytemuck::pod_collect_to_vec(&npy(&d.join(name)))
}

fn compare(label: &str, exp: &[f32], got: &[f32]) -> (f64, f64) {
    assert_eq!(exp.len(), got.len(), "{label}: length");
    let (mut se, mut ee, mut gg, mut eg) = (0f64, 0f64, 0f64, 0f64);
    for (&e, &g) in exp.iter().zip(got) {
        let (e, g) = (e as f64, g as f64);
        se += (g - e) * (g - e);
        ee += e * e;
        gg += g * g;
        eg += e * g;
    }
    let rel = (se / ee.max(1e-30)).sqrt();
    let cos = eg / (ee.sqrt() * gg.sqrt()).max(1e-30);
    eprintln!(
        "[{label}] n={} rel={:.4}% cos={cos:.6}",
        exp.len(),
        rel * 100.0
    );
    (rel, cos)
}

async fn backend() -> Arc<WgpuBackend> {
    let b = Arc::new(
        WgpuBackend::new_with_config(WgpuConfig {
            power_preference: PowerPreference::HighPerformance,
            timestamps: false,
            disable_coopmat: false,
        })
        .await
        .expect("wgpu adapter"),
    );
    if std::env::var("THINFER_CUDA").as_deref() != Ok("off") {
        match thinfer_cuda::install(&b) {
            Ok((name, sm)) => eprintln!("cuda: {name} sm {sm:?}"),
            Err(e) => eprintln!("cuda unavailable: {e}"),
        }
    }
    b
}

type Src = UnionSource<
    UnionSource<SafetensorsSource<MmapFileOpener>, ShardedSafetensorsSource<MmapFileOpener>>,
    WeightNormSource<PytorchSource<MmapFileOpener>>,
>;

fn path(r: &str) -> Option<PathBuf> {
    thinfer_native::cache::resolve(MANIFEST.get(r)?)
}

async fn open(p: PathBuf) -> MmapFileOpener {
    MmapFileOpener::new(&p).await.expect("mmap")
}

async fn residency() -> Option<WeightResidency<Src>> {
    let dit = SafetensorsSource::open(open(path(role::DIT)?).await)
        .await
        .expect("parse dit");
    let text = ShardedSafetensorsSource::open(vec![
        open(path(role::TEXT_SHARD_1)?).await,
        open(path(role::TEXT_SHARD_2)?).await,
    ])
    .await
    .expect("parse text encoder");
    let vae = PytorchSource::open(open(path(role::VAE)?).await)
        .await
        .expect("parse vae");
    Some(WeightResidency::new(
        UnionSource::new(UnionSource::new(dit, text), WeightNormSource::new(vae)),
        ResidencyBudget {
            ram_bytes: 8 << 30,
            vram_bytes: 5 << 30,
        },
    ))
}

/// Tokenizer ids, text states, DiT velocities and the DAC decode against
/// the reference; one model load for all four.
#[tokio::test(flavor = "current_thread")]
async fn moss_sfx_parity() {
    let Some(d) = parity_dir() else {
        eprintln!("THINFER_SFX_PARITY not set / no tensors: skipped");
        return;
    };
    let (Some(v), Some(mg), Some(c)) = (
        path(role::VOCAB),
        path(role::MERGES),
        path(role::TOKENIZER_CONFIG),
    ) else {
        eprintln!("tokenizer not cached: skipped");
        return;
    };
    let tok = thinfer_native::tokenizer::HfTokenizer::from_bpe_files(v, mg, c)
        .await
        .unwrap();
    let ids = tok.encode(&prompt_text(PROMPT, 10.0), false).unwrap();
    let ref_ids: Vec<u32> = bytemuck::pod_collect_to_vec(&npy(&d.join("ids.npy")));
    assert_eq!(ids, ref_ids, "prompt token ids");

    let Some(res) = residency().await else {
        eprintln!("MOSS-SoundEffect not cached: skipped");
        return;
    };
    let model = SfxModel::load(backend().await, res).await.expect("load");

    // Text states (zero tail included).
    let hidden = model.encode_text(&ids).await.expect("text");
    let (rel, cos) = compare("text", &f32s(&d, "hidden.npy"), &hidden);
    assert!(cos > 0.999 && rel < 0.05, "text states");

    // DiT velocity on the reference's own text states (isolates the DiT).
    let ctx = f32s(&d, "hidden.npy");
    let zero = vec![0f32; ctx.len()];
    let x = f32s(&d, "x.npy");
    let sigmas = f32s(&d, "sigmas.npy");
    for (k, &sigma) in sigmas.iter().enumerate() {
        let vs = model
            .velocities(&x, sigma, &[&ctx, &zero])
            .await
            .expect("dit");
        for (name, v) in ["cond", "uncond"].iter().zip(&vs) {
            let (rel, cos) = compare(
                &format!("dit {name} sigma={sigma:.4}"),
                &f32s(&d, &format!("v_{name}_{k}.npy")),
                v,
            );
            assert!(cos > 0.998 && rel < 0.06, "dit {name} {k}");
        }
    }

    // DAC: the whole latent, then a window of its first half (exact up to
    // the window's end: the margin covers the receptive field).
    let lat = f32s(&d, "lat.npy");
    let wav_ref = f32s(&d, "wav.npy");
    let wav = model.decode(&lat).await.expect("dac");
    let (rel, _) = compare("dac", &wav_ref, &wav);
    assert!(rel < 0.01, "dac decode");
    let cols = lat.len() / LATENT_CH;
    let half = cols / 2;
    let (window, lead) = decode_window(&lat, cols, half, false, DECODE_MARGIN);
    assert_eq!(lead, 0);
    let part = model.decode(&window).await.expect("dac window");
    let n = half * HOP;
    let (rel_w, _) = compare("dac window", &wav[..n], &part[..n]);
    assert!(rel_w < 1e-5, "windowed decode must match the full decode");
}
