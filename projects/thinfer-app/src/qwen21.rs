//! Qwen-Image-2.1-Turbo: text-to-image, or an edit / composition from input
//! images. Mirrors `diffusers` `QwenImage21Pipeline`:
//!
//! - The prompt goes in a raw chat template (NOT `apply_chat_template`); each
//!   condition image adds `<imageN><|vision_start|><|image_pad|><|vision_end|>`
//!   with the pad expanded to its merged vision tokens. The system message's
//!   rows are dropped from the conditioner output.
//! - A condition image is resized ONCE to the 1 MP area at its own aspect
//!   (sides multiples of 32, Lanczos), RGBA. The vision tower sees it
//!   composited over white; the VAE reads all four channels.
//! - With input images, the output keeps the last image's aspect at the
//!   requested pixel area (the reference derives its size from the input).

use std::sync::Arc;

use image::imageops::FilterType;
use thinfer_core::backend::WgpuBackend;
use thinfer_core::format::gguf::GgufSource;
use thinfer_core::format::safetensors::SafetensorsSource;
use thinfer_core::format::union::{RenamedSource, UnionSource};
use thinfer_core::residency::WeightResidency;
use thinfer_core::tokenizer::Tokenizer;
use thinfer_core::workspace::Workspace;
use thinfer_models::common::block::BlockPipelines;
#[cfg(feature = "vault")]
use thinfer_models::common::lora::{LoraFoldSource, discover_specs};
use thinfer_models::minimax::vision::{self, VisionOutput};
use thinfer_models::qwen_image21::config;
#[cfg(feature = "vault")]
use thinfer_models::qwen_image21::lora::{self as qwen21_lora, AdaptedLora};
use thinfer_models::qwen_image21::manifest::role;
use thinfer_models::qwen_image21::pipeline::{
    ConditionImage, ProgressEvent, Qwen21Pipeline, Request,
};
use thinfer_models::z_image::qwen3_gguf_renames;
use thinfer_native::tokenizer::HfTokenizer;

use crate::config::{ResidencyBudget, random_seed};
use crate::download::resolve_role;
use crate::minimax::open_mmap;
use crate::progress::{ProgressSink, Stage};
use crate::request::{ImageRequest, JobSummary};

const SYSTEM: &str = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n";
const USER: &str = "<|im_start|>user\n";
const END: &str = "<|im_end|>\n<|im_start|>assistant\n";
const IMAGE_PAD: (&str, u32) = ("<|image_pad|>", 151_655);
const VISION_START: &str = "<|vision_start|>";
const VISION_END: &str = "<|vision_end|>";

/// Which DiT file a generate loads. Q4_K_M fits an 8 GB card whole (no
/// per-step streaming); Q8_0 is the parity canary.
pub const DIT_ROLE: &str = role::DIT_GGUF_Q4_K_M;

/// A decoded, resized condition image: RGBA8 `[h, w, 4]`. User media, so
/// page-locked on a server and zeroed on drop.
struct Prepared {
    rgba: thinfer_core::secmem::Locked<u8>,
    height: usize,
    width: usize,
}

/// Python's `round` (half to even).
fn round_half_even(x: f64) -> f64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 && r % 2.0 != 0.0 {
        r - x.signum()
    } else {
        r
    }
}

/// `calculate_dimensions(area, ratio)`: `(width, height)`, multiples of 32.
pub fn dims_for(area: f64, ratio: f64) -> (u32, u32) {
    let w = (area * ratio).sqrt();
    let h = w / ratio;
    let m = config::SIDE_MULTIPLE as f64;
    let side = |v: f64| ((round_half_even(v / m) * m) as u32).max(config::SIDE_MULTIPLE);
    (side(w), side(h))
}

fn prepare(bytes: &[u8]) -> Result<Prepared, String> {
    let img = image::load_from_memory(bytes).map_err(|e| format!("decode input image: {e}"))?;
    let (w, h) = (img.width() as f64, img.height() as f64);
    let (tw, th) = dims_for(config::CONDITION_AREA as f64, w / h);
    let img = if (img.width(), img.height()) == (tw, th) {
        img
    } else {
        img.resize_exact(tw, th, FilterType::Lanczos3)
    };
    Ok(Prepared {
        rgba: img.to_rgba8().into_raw().into(),
        height: th as usize,
        width: tw as usize,
    })
}

/// RGBA over white -> RGB8, what the vision tower sees.
fn over_white(p: &Prepared) -> Vec<u8> {
    p.rgba
        .chunks_exact(4)
        .flat_map(|px| {
            let a = px[3] as f32 / 255.0;
            [0, 1, 2].map(|c| (px[c] as f32 * a + 255.0 * (1.0 - a)).round() as u8)
        })
        .collect()
}

/// RGBA8 -> planar `[4, h, w]` in `[-1, 1]`, what the VAE reads.
fn vae_planar(p: &Prepared) -> Vec<f32> {
    let plane = p.height * p.width;
    let mut out = vec![0.0f32; 4 * plane];
    for (i, px) in p.rgba.chunks_exact(4).enumerate() {
        for c in 0..4 {
            out[c * plane + i] = px[c] as f32 / 127.5 - 1.0;
        }
    }
    out
}

/// The tokenized template and where each image's pads begin.
struct Prompt {
    ids: Vec<u32>,
    drop: usize,
    pad_starts: Vec<usize>,
}

fn tokenize(tok: &impl Tokenizer, prompt: &str, pads: &[usize]) -> Result<Prompt, String> {
    let enc = |s: &str| tok.encode(s, false).map_err(|e| format!("tokenize: {e:?}"));
    if enc(IMAGE_PAD.0)? != [IMAGE_PAD.1] {
        return Err("tokenizer does not map <|image_pad|> to its id".into());
    }
    // Qwen has no BOS: an empty prompt would leave the encoder nothing to read.
    let prompt = if prompt.is_empty() { " " } else { prompt };
    let images: Vec<String> = pads
        .iter()
        .enumerate()
        .map(|(i, &n)| {
            let sep = if i == 0 { "" } else { " " };
            format!(
                "{sep}<image{}>{VISION_START}{}{VISION_END}",
                i + 1,
                IMAGE_PAD.0.repeat(n)
            )
        })
        .collect();
    let text = format!("{SYSTEM}{USER}{}{prompt}{END}", images.concat());
    let ids = enc(&text)?;
    let drop = enc(SYSTEM)?.len();
    // Each image's pads are one contiguous run, in order.
    let mut pad_starts = Vec::with_capacity(pads.len());
    let mut i = 0;
    while i < ids.len() {
        if ids[i] == IMAGE_PAD.1 {
            pad_starts.push(i);
            while i < ids.len() && ids[i] == IMAGE_PAD.1 {
                i += 1;
            }
        } else {
            i += 1;
        }
    }
    let runs_ok = pad_starts.len() == pads.len()
        && ids.iter().filter(|&&t| t == IMAGE_PAD.1).count() == pads.iter().sum::<usize>();
    if !runs_ok {
        return Err(format!(
            "image pads tokenized as {} runs, expected {}",
            pad_starts.len(),
            pads.len()
        ));
    }
    Ok(Prompt {
        ids,
        drop,
        pad_starts,
    })
}

async fn vision_features(
    backend: &Arc<WgpuBackend>,
    budget: ResidencyBudget,
    images: &[Prepared],
    manifest: &thinfer_core::manifest::ModelManifest,
) -> Result<Vec<VisionOutput>, String> {
    let path = resolve_role(manifest, role::VISION)?;
    let src = GgufSource::open(open_mmap(&path).await?)
        .await
        .map_err(|e| format!("parse vision mmproj: {e:?}"))?;
    let residency = WeightResidency::new(src, budget);
    let handles =
        vision::register_handles(&residency).map_err(|e| format!("register vision: {e:?}"))?;
    let bp = BlockPipelines::compile(backend, &vision::wgsl_configs())
        .await
        .map_err(|e| format!("compile vision pipelines: {e:?}"))?;
    let ws = Workspace::new(Arc::clone(backend), Arc::clone(residency.arbiter()));
    let mut out = Vec::with_capacity(images.len());
    for p in images {
        let rgb = over_white(p);
        out.push(
            vision::forward(
                backend, &bp, &residency, &ws, &handles, &rgb, p.height, p.width,
            )
            .await
            .map_err(|e| format!("vision forward: {e:?}"))?,
        );
    }
    residency.evict_all_and_free(&**backend);
    Ok(out)
}

/// PNG of planar `[4, h, w]` in `[-1, 1]`: RGBA when any pixel is not opaque.
fn encode_png(rgba: &[f32], width: u32, height: u32) -> Result<Vec<u8>, String> {
    let plane = (width * height) as usize;
    let byte = |v: f32| ((v.clamp(-1.0, 1.0) + 1.0) * 127.5).round() as u8;
    let alpha: Vec<u8> = rgba[3 * plane..].iter().map(|&v| byte(v)).collect();
    let opaque = alpha.iter().all(|&a| a == 255);
    let mut buf = Vec::with_capacity(plane * 4);
    for i in 0..plane {
        for c in 0..3 {
            buf.push(byte(rgba[c * plane + i]));
        }
        if !opaque {
            buf.push(alpha[i]);
        }
    }
    let img = if opaque {
        image::DynamicImage::ImageRgb8(
            image::RgbImage::from_raw(width, height, buf).ok_or("png buffer size")?,
        )
    } else {
        image::DynamicImage::ImageRgba8(
            image::RgbaImage::from_raw(width, height, buf).ok_or("png buffer size")?,
        )
    };
    let mut png = Vec::new();
    img.write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
        .map_err(|e| format!("encode png: {e}"))?;
    Ok(png)
}

fn map_progress(ev: ProgressEvent) -> Option<Stage> {
    match ev {
        ProgressEvent::TextEncode => Some(Stage::TextEncode),
        ProgressEvent::Step { i, n } => Some(Stage::Step { i, n }),
        ProgressEvent::VaeDecode => Some(Stage::VaeDecode),
        ProgressEvent::ImageEncode => None,
    }
}

pub async fn run(
    backend: &Arc<WgpuBackend>,
    req: &ImageRequest,
    budget: ResidencyBudget,
    sink: &dyn ProgressSink,
) -> Result<JobSummary, String> {
    let manifest = req.model.manifest();
    let images: Vec<Prepared> = req
        .input_images
        .iter()
        .map(|b| prepare(&b.0))
        .collect::<Result<_, _>>()?;
    // An edit keeps the input's aspect at the requested pixel area.
    let (width, height) = match images.last() {
        Some(p) => dims_for(
            req.width as f64 * req.height as f64,
            p.width as f64 / p.height as f64,
        ),
        None => (req.width, req.height),
    };
    let tokenizer = HfTokenizer::from_path(&resolve_role(manifest, role::TOKENIZER)?)
        .await
        .map_err(|e| format!("tokenizer: {e:?}"))?;
    let pads: Vec<usize> = images
        .iter()
        .map(|p| (p.height / 32) * (p.width / 32))
        .collect();
    let prompt = tokenize(&tokenizer, &req.prompt, &pads)?;
    let seed = req.seed.unwrap_or_else(random_seed);
    tracing::info!(
        target: thinfer_core::trace::DIAG,
        model = %req.model, width, height, seed, tokens = prompt.ids.len(),
        images = images.len(), "qwen-image-2.1 generate start",
    );
    sink.note(&format!(
        "Generating {width}x{height} image{}, 8 steps, seed {seed} ({})",
        if images.is_empty() {
            String::new()
        } else {
            format!(" from {} input image(s)", images.len())
        },
        req.model,
    ));

    sink.stage(Stage::TextEncode);
    let vision = if images.is_empty() {
        Vec::new()
    } else {
        vision_features(backend, budget, &images, manifest).await?
    };
    let conditions: Vec<ConditionImage> = images
        .iter()
        .zip(vision)
        .zip(&prompt.pad_starts)
        .map(|((p, vision), &pad_start)| ConditionImage {
            rgba: vae_planar(p),
            height: p.height,
            width: p.width,
            vision,
            pad_start,
        })
        .collect();
    drop(images);

    let dit = GgufSource::open(open_mmap(&resolve_role(manifest, DIT_ROLE)?).await?)
        .await
        .map_err(|e| format!("parse dit gguf: {e:?}"))?;
    let enc = RenamedSource::with_passthrough(
        GgufSource::open(open_mmap(&resolve_role(manifest, role::ENCODER_GGUF)?).await?)
            .await
            .map_err(|e| format!("parse encoder gguf: {e:?}"))?,
        qwen3_gguf_renames(),
    );
    let vae = SafetensorsSource::open(open_mmap(&resolve_role(manifest, role::VAE)?).await?)
        .await
        .map_err(|e| format!("parse vae safetensors: {e:?}"))?;
    let job = Job {
        backend,
        req,
        budget,
        sink,
        prompt: &prompt,
        conditions: &conditions,
        size: (width, height),
        seed,
    };
    // Vault adapters fold into the DiT (a distinct source type, hence the two
    // calls into the shared generic finish).
    #[cfg(feature = "vault")]
    if !req.lora.is_empty() {
        let folded = fold_adapters(dit, req, sink).await?;
        return job
            .finish(UnionSource::new(UnionSource::new(folded, enc), vae))
            .await;
    }
    #[cfg(not(feature = "vault"))]
    if !req.lora.is_empty() {
        return Err("this build has no adapter (vault) support".into());
    }
    job.finish(UnionSource::new(UnionSource::new(dit, enc), vae))
        .await
}

/// Everything [`Job::finish`] needs besides the weight source.
struct Job<'a> {
    backend: &'a Arc<WgpuBackend>,
    req: &'a ImageRequest,
    budget: ResidencyBudget,
    sink: &'a dyn ProgressSink,
    prompt: &'a Prompt,
    conditions: &'a [ConditionImage],
    size: (u32, u32),
    seed: u64,
}

impl Job<'_> {
    async fn finish<S: thinfer_core::weight::WeightSource>(
        self,
        source: S,
    ) -> Result<JobSummary, String> {
        let Job {
            backend,
            req,
            budget,
            sink,
            prompt,
            conditions,
            size: (width, height),
            seed,
        } = self;
        let residency = WeightResidency::new(source, budget);
        let pipeline = Qwen21Pipeline::load(Arc::clone(backend), residency)
            .await
            .map_err(|e| format!("model load: {e:?}"))?;
        let progress = |ev: ProgressEvent| {
            if let Some(s) = map_progress(ev) {
                sink.stage(s);
            } else {
                sink.note("Encoding input image(s)");
            }
        };
        let rgba = pipeline
            .generate(
                &Request {
                    token_ids: &prompt.ids,
                    drop: prompt.drop,
                    images: conditions,
                    height,
                    width,
                    seed,
                },
                Some(&progress),
            )
            .await
            .map_err(|e| format!("generate: {e:?}"))?;
        let png = encode_png(&rgba, width, height)?;
        let bytes = png.len();
        crate::progress::write_artifact(sink, &req.output, png)?;
        tracing::info!(target: thinfer_core::trace::DIAG, path = %req.output.display(), bytes, "wrote output");
        Ok(JobSummary {
            output: req.output.clone(),
            width,
            height,
            frames: 1,
            fps: None,
            seed: Some(seed),
        })
    }
}

/// Decrypt this request's vault adapters, normalize each onto the GGUF's keys
/// (any of the community layouts, split MLP halves merged into the fused
/// `gate_up`), and wrap `base` in the generic fold. The password is re-derived
/// here and never logged; plaintext adapter bytes live only for the fold.
#[cfg(feature = "vault")]
async fn fold_adapters<B: thinfer_core::weight::WeightSource>(
    base: B,
    req: &ImageRequest,
    sink: &dyn ProgressSink,
) -> Result<LoraFoldSource<B, AdaptedLora>, String> {
    use thinfer_core::format::safetensors::ShardedSafetensorsSource;
    let password = req
        .vault_password
        .as_ref()
        .ok_or("a vault password is required to use adapters")?;
    let vault = crate::vault::Vault::new(&req.vault_dir);
    let model = req.model.to_string();
    let mut stacks = Vec::with_capacity(req.lora.len());
    for l in &req.lora {
        let bytes = vault
            .open(password.expose(), &model, &l.id)
            .map_err(|e| e.to_string())?;
        let raw = ShardedSafetensorsSource::open(vec![crate::executor::BytesOpener::new(bytes)])
            .await
            .map_err(|e| format!("parse adapter {}: {e:?}", l.id))?;
        let adapted = qwen21_lora::adapt(&raw)
            .await
            .map_err(|e| format!("adapter {}: {e}", l.id))?;
        let specs = discover_specs(&base, &adapted)
            .await
            .map_err(|e| format!("discover adapter {} sites: {e:?}", l.id))?;
        tracing::info!(
            target: thinfer_core::trace::DIAG,
            sites = specs.len(), strength = l.weight, "adapter sites discovered",
        );
        stacks.push((adapted, l.weight, specs));
    }
    let sites: usize = stacks.iter().map(|(_, _, s)| s.len()).sum();
    if sites == 0 {
        return Err(format!(
            "adapter(s) folded 0 sites into {model} -- the LoRA keys match no Qwen-Image-2.1 \
             DiT tensor (is this a LoRA for a different model?)"
        ));
    }
    sink.note(&format!(
        "Folding {} adapter(s) into {model} ({sites} sites); first-touch quantizes each site",
        stacks.len(),
    ));
    LoraFoldSource::new(base, stacks).map_err(|e| format!("build adapter fold: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dims_follow_the_reference_rounding() {
        // 1 MP square stays 1024; a 3:2 input lands on 32-multiples.
        assert_eq!(dims_for(1024.0 * 1024.0, 1.0), (1024, 1024));
        let (w, h) = dims_for(1024.0 * 1024.0, 1.5);
        assert_eq!((w % 32, h % 32), (0, 0));
        assert_eq!((w, h), (1248, 832));
    }
}
