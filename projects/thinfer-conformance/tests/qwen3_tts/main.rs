//! Qwen3-TTS parity vs the reference package (bf16 GPU pyref). Gated on
//! `tts-e2e`; needs the official repos in the HF cache and the reference
//! tensors from `parity_ref.py` in `THINFER_TTS_PARITY` (skips otherwise):
//! `text_ids` / `instruct_ids` (VoiceDesign prompt), greedy `codes` and the
//! first frames' codebook-0 `logits0`, the codec's `wav` of those codes, and
//! `ref_wav` / `mel` / `xvec` (Base speaker encoder); `wav_f32` from
//! `codec_f32.py`.

#![cfg(feature = "tts-e2e")]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use thinfer_core::backend::{PowerPreference, WgpuBackend, WgpuConfig};
use thinfer_core::format::safetensors::SafetensorsSource;
use thinfer_core::manifest::ModelManifest;
use thinfer_core::policy::ResidencyBudget;
use thinfer_core::residency::WeightResidency;
use thinfer_core::workspace::Workspace;
use thinfer_models::llm::kernels::LlmPipelines;
use thinfer_models::qwen3_tts::codec::{CodecDecoder, CodecEncoder};
use thinfer_models::qwen3_tts::manifest::{self, role};
use thinfer_models::qwen3_tts::prompt::{self, PromptSpec};
use thinfer_models::qwen3_tts::speaker::SpeakerEncoder;
use thinfer_models::qwen3_tts::talker::{FrameSink, GenSpec, Talker, TalkerWeights};
use thinfer_models::qwen3_tts::{CODE_GROUPS, TtsSampling, language_id};
use thinfer_native::MmapFileOpener;

type Src = SafetensorsSource<MmapFileOpener>;

fn parity_dir() -> Option<PathBuf> {
    let d = PathBuf::from(std::env::var("THINFER_TTS_PARITY").ok()?);
    d.join("codes.npy").exists().then_some(d)
}

/// A little-endian `.npy` payload (any dtype of 4 bytes) and its shape.
fn npy(p: &Path) -> (Vec<u8>, Vec<usize>) {
    let b = std::fs::read(p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
    let hl = u16::from_le_bytes([b[8], b[9]]) as usize;
    let hdr = std::str::from_utf8(&b[10..10 + hl]).unwrap();
    assert!(
        hdr.contains("'fortran_order': False"),
        "{}: Fortran-order .npy (save np.ascontiguousarray)",
        p.display()
    );
    let shape = hdr
        .split("'shape': (")
        .nth(1)
        .unwrap()
        .split(')')
        .next()
        .unwrap();
    let shape = shape
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    (b[10 + hl..].to_vec(), shape)
}

fn npy_f32(p: &Path) -> Vec<f32> {
    bytemuck::cast_slice(&npy(p).0).to_vec()
}

fn npy_u32(p: &Path) -> Vec<u32> {
    bytemuck::cast_slice(&npy(p).0).to_vec()
}

fn compare(label: &str, exp: &[f32], got: &[f32]) -> (f64, f64) {
    let n = exp.len().min(got.len());
    let (mut se, mut ee, mut gg, mut eg) = (0f64, 0f64, 0f64, 0f64);
    for (&e, &g) in exp[..n].iter().zip(&got[..n]) {
        let (e, g) = (e as f64, g as f64);
        se += (g - e) * (g - e);
        ee += e * e;
        gg += g * g;
        eg += e * g;
    }
    let rel = (se / ee.max(1e-30)).sqrt();
    let cos = eg / (ee.sqrt() * gg.sqrt()).max(1e-30);
    eprintln!(
        "[{label}] n={n} (exp {} got {}) rel={:.4}% cos={cos:.6}",
        exp.len(),
        got.len(),
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

async fn source(m: &ModelManifest, r: &str) -> Option<WeightResidency<Src>> {
    let path = thinfer_native::cache::resolve(m.get(r)?)?;
    let src = SafetensorsSource::open(MmapFileOpener::new(&path).await.ok()?)
        .await
        .expect("parse safetensors");
    Some(WeightResidency::new(
        src,
        ResidencyBudget {
            ram_bytes: 8 << 30,
            vram_bytes: 5 << 30,
        },
    ))
}

struct Count(u32);
impl FrameSink for Count {
    fn frame(&mut self, n: u32) -> bool {
        self.0 = n;
        true
    }
}

/// The BPE built from vocab + merges + config gives the reference ids
/// (from `transformers` `AutoTokenizer` on the same files).
#[tokio::test(flavor = "current_thread")]
async fn tokenizer_ids() {
    use thinfer_core::tokenizer::Tokenizer;
    let m = &manifest::BASE_MANIFEST;
    let get = |r| m.get(r).and_then(thinfer_native::cache::resolve);
    let (Some(v), Some(mg), Some(c)) = (
        get(role::VOCAB),
        get(role::MERGES),
        get(role::TOKENIZER_CONFIG),
    ) else {
        eprintln!("tokenizer files not cached: skipped");
        return;
    };
    let tok = thinfer_native::tokenizer::HfTokenizer::from_bpe_files(v, mg, c)
        .await
        .unwrap();
    let cases: [(&str, &[u32]); 2] = [
        (
            "<|im_start|>assistant\nHello there, I'm Aria!<|im_end|>\n<|im_start|>assistant\n",
            &[
                151644, 77091, 198, 9707, 1052, 11, 358, 2776, 362, 4204, 0, 151645, 198, 151644,
                77091, 198,
            ],
        ),
        (
            "<|im_start|>user\nWhisper softly.<|im_end|>\n",
            &[151644, 872, 198, 1639, 27370, 56050, 13, 151645, 198],
        ),
    ];
    for (s, exp) in cases {
        assert_eq!(tok.encode(s, false).unwrap(), exp, "{s:?}");
    }
}

#[tokio::test(flavor = "current_thread")]
async fn codec_parity() {
    let Some(dir) = parity_dir() else {
        eprintln!("THINFER_TTS_PARITY unset or empty: skipped");
        return;
    };
    let Some(res) = source(&manifest::DESIGN_MANIFEST, role::CODEC).await else {
        eprintln!("codec not cached: skipped");
        return;
    };
    let b = backend().await;
    let p = LlmPipelines::compile(&b).await.unwrap();
    let ws = Workspace::new(b.clone(), Arc::clone(res.arbiter()));
    let dec = CodecDecoder::load(&b, &res).await.unwrap();
    let flat = npy_u32(&dir.join("codes.npy"));
    let codes: Vec<[u32; CODE_GROUPS]> = flat
        .chunks(CODE_GROUPS)
        .map(|c| c.try_into().unwrap())
        .collect();
    let t0 = std::time::Instant::now();
    let wav = dec.decode(&b, &res, &p, &ws, &codes).await.unwrap();
    eprintln!(
        "decode {} frames: {:.3}s",
        codes.len(),
        t0.elapsed().as_secs_f64()
    );
    // The f32 reference decode (`codec_f32.py`): the codec ships and runs
    // in f32; the bf16 one differs from it by ~1.7% on its own.
    let exp = npy_f32(&dir.join("wav_f32.npy"));
    let (rel, _) = compare("codec wav vs f32 ref", &exp, &wav);
    compare(
        "codec wav vs bf16 ref",
        &npy_f32(&dir.join("wav.npy")),
        &wav,
    );
    assert_eq!(exp.len(), wav.len(), "sample count");
    assert!(rel < 1e-3, "codec rel {rel}");
}

/// Encoder codes of `ref_wav` vs the f32 reference (`enc_ref.py`; the
/// bf16 reference agrees with f32 on only ~55% of codes, the residual
/// codebooks being that sensitive).
#[tokio::test(flavor = "current_thread")]
async fn encoder_parity() {
    let Some(dir) = parity_dir() else {
        eprintln!("THINFER_TTS_PARITY unset or empty: skipped");
        return;
    };
    let Some(res) = source(&manifest::BASE_MANIFEST, role::CODEC).await else {
        eprintln!("codec not cached: skipped");
        return;
    };
    let b = backend().await;
    let p = LlmPipelines::compile(&b).await.unwrap();
    let ws = Workspace::new(b.clone(), Arc::clone(res.arbiter()));
    let enc = CodecEncoder::load(&b, &res).await.unwrap();
    let wav = npy_f32(&dir.join("ref_wav.npy"));
    let t0 = std::time::Instant::now();
    let mut trace = Vec::new();
    let codes = enc
        .encode(&b, &res, &p, &ws, &wav, Some(&mut trace))
        .await
        .unwrap();
    for (name, got) in &trace {
        let f = dir.join(format!("enc_{name}.npy"));
        if f.exists() {
            compare(&format!("encoder {name}"), &npy_f32(&f), got);
        }
    }
    eprintln!(
        "encode {:.2}s of audio: {:.3}s",
        wav.len() as f64 / 24e3,
        t0.elapsed().as_secs_f64()
    );
    let exp = npy_u32(&dir.join("enc_codes.npy"));
    let zref = dir.join("enc_down.npy");
    if zref.exists() {
        let z = npy_f32(&zref);
        let q = enc.quantize(&z, z.len() / 512);
        let qf: Vec<u32> = q.iter().flatten().copied().collect();
        let same = qf.iter().zip(&exp).filter(|(a, b)| a == b).count();
        eprintln!(
            "quantize(reference z): {:.1}% equal",
            100.0 * same as f64 / exp.len() as f64
        );
    }
    let got: Vec<u32> = codes.iter().flatten().copied().collect();
    assert_eq!(got.len(), exp.len(), "code count");
    let rate = |g: Option<usize>| {
        let pairs = got
            .iter()
            .zip(&exp)
            .enumerate()
            .filter(|(i, _)| g.is_none_or(|g| i % CODE_GROUPS == g));
        let (n, same) = pairs.fold((0, 0), |(n, s), (_, (a, b))| (n + 1, s + (a == b) as usize));
        same as f64 / n as f64
    };
    let (all, first) = (rate(None), rate(Some(0)));
    eprintln!(
        "encoder codes equal: all {:.1}%, codebook 0 {:.1}%",
        all * 100.0,
        first * 100.0
    );
    assert!(
        first > 0.99 && all > 0.9,
        "encoder codes all {all} first {first}"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn speaker_parity() {
    let Some(dir) = parity_dir() else {
        eprintln!("THINFER_TTS_PARITY unset or empty: skipped");
        return;
    };
    let Some(res) = source(&manifest::BASE_MANIFEST, role::MODEL).await else {
        eprintln!("base not cached: skipped");
        return;
    };
    let enc = SpeakerEncoder::load(res.source()).await.unwrap();
    let wav = npy_f32(&dir.join("ref_wav.npy"));
    let (mel, frames) = enc.mel(&wav);
    let exp_mel = npy_f32(&dir.join("mel.npy"));
    eprintln!("mel frames {frames}");
    let (mrel, _) = compare("mel", &exp_mel, &mel);
    let t0 = std::time::Instant::now();
    let xv = enc.embed(&wav);
    eprintln!("x-vector: {:.3}s", t0.elapsed().as_secs_f64());
    let (_, cos) = compare("xvec", &npy_f32(&dir.join("xvec.npy")), &xv);
    assert!(mrel < 1e-3, "mel rel {mrel}");
    assert!(cos > 0.999, "xvec cos {cos}");
}

#[tokio::test(flavor = "current_thread")]
async fn talker_parity() {
    let Some(dir) = parity_dir() else {
        eprintln!("THINFER_TTS_PARITY unset or empty: skipped");
        return;
    };
    let Some(res) = source(&manifest::DESIGN_MANIFEST, role::MODEL).await else {
        eprintln!("voice design not cached: skipped");
        return;
    };
    let b = backend().await;
    let p = LlmPipelines::compile(&b).await.unwrap();
    let ws = Workspace::new(b.clone(), Arc::clone(res.arbiter()));
    let w = TalkerWeights::load(&b, &res).await.unwrap();
    let text = npy_u32(&dir.join("text_ids.npy"));
    let ins = npy_u32(&dir.join("instruct_ids.npy"));
    let pr = prompt::build(&PromptSpec {
        text_ids: &text,
        instruct_ids: Some(&ins),
        language: language_id("english").unwrap(),
        speaker: false,
        icl: None,
        non_streaming: true,
    })
    .unwrap();
    let exp_codes = npy_u32(&dir.join("codes.npy"));
    let exp_logits = npy_f32(&dir.join("logits0.npy"));
    let t = Talker {
        w: &w,
        residency: &res,
        backend: &b,
        p: &p,
        ws: &ws,
    };
    let spec = GenSpec {
        prompt: &pr,
        speaker: None,
        ref_codes: &[],
        sampling: TtsSampling::greedy(),
        seed: 0,
        max_frames: 400,
        tap_logits: 4,
        force: None,
    };
    let t0 = std::time::Instant::now();
    let mut sink = Count(0);
    let out = t.generate(&spec, &mut sink).await.unwrap();
    let secs = t0.elapsed().as_secs_f64();
    let frames = out.codes.len();
    eprintln!(
        "talker: {frames} frames in {secs:.2}s ({:.1} ms/frame, rtf {:.3})",
        secs * 1e3 / frames.max(1) as f64,
        secs / (frames as f64 / 12.5)
    );
    let vocab = exp_logits.len() / 4;
    let (_, cos) = compare("logits frame0", &exp_logits[..vocab], &out.logits[0]);
    for i in 1..out.logits.len().min(4) {
        compare(
            &format!("logits frame{i}"),
            &exp_logits[i * vocab..(i + 1) * vocab],
            &out.logits[i],
        );
    }
    let got: Vec<u32> = out.codes.iter().flatten().copied().collect();
    let n = got.len().min(exp_codes.len());
    let same = got[..n]
        .iter()
        .zip(&exp_codes[..n])
        .filter(|(a, b)| a == b)
        .count();
    let first_diff = got
        .chunks(CODE_GROUPS)
        .zip(exp_codes.chunks(CODE_GROUPS))
        .position(|(a, b)| a != b);
    eprintln!(
        "codes: {} vs {} frames, {:.1}% equal, first differing frame {first_diff:?}",
        frames,
        exp_codes.len() / CODE_GROUPS,
        100.0 * same as f64 / n.max(1) as f64
    );
    // Teacher forced on the reference codes: every tapped frame sees the
    // reference's inputs, so its logits must match, not just frame 0's.
    let forced: Vec<[u32; CODE_GROUPS]> = exp_codes
        .chunks(CODE_GROUPS)
        .take(4)
        .map(|c| c.try_into().unwrap())
        .collect();
    let tf = t
        .generate(
            &GenSpec {
                force: Some(&forced),
                ..spec
            },
            &mut sink,
        )
        .await
        .unwrap();
    let mut worst = 1f64;
    for (i, l) in tf.logits.iter().enumerate() {
        let (_, c) = compare(
            &format!("forced logits frame{i}"),
            &exp_logits[i * vocab..(i + 1) * vocab],
            l,
        );
        worst = worst.min(c);
    }
    assert!(cos > 0.999, "frame-0 logits cos {cos}");
    assert!(worst > 0.999, "teacher-forced logits cos {worst}");
}
