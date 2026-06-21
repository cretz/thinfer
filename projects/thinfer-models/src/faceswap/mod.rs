//! Face-swap pipeline: SCRFD detect -> ArcFace source embedding -> HyperSwap
//! per-face swap -> feathered paste-back. Mirrors intabai
//! `web/src/video-face-swap/pipeline.ts` (HyperSwap path, SCRFD detector, no
//! enhancer/XSeg). The three ONNX models run on the thinfer GPU executor
//! (`thinfer_core::onnx`); geometry is host-side (`image`).

pub mod detect;
pub mod image;

use std::collections::HashMap;
use std::sync::Arc;

use thinfer_core::backend::WgpuBackend;
use thinfer_core::onnx::{OnnxError, OnnxModel};

pub use detect::Face;
use image::{Affine, Image};

/// FaceFusion 5-point alignment templates (normalized to `[0, 1]`; multiply by
/// the crop size). From `pipeline.ts`. Kept at the upstream digit count
/// (faithful to FaceFusion) even though it exceeds f32 precision.
#[allow(clippy::excessive_precision)]
const TEMPLATE_ARCFACE_112_V2: [[f32; 2]; 5] = [
    [0.34191607, 0.46157411],
    [0.65653393, 0.45983393],
    [0.500225, 0.64050536],
    [0.37097589, 0.82469196],
    [0.63151696, 0.82325089],
];
#[allow(clippy::excessive_precision)]
const TEMPLATE_ARCFACE_128: [[f32; 2]; 5] = [
    [0.36167656, 0.40387734],
    [0.63696719, 0.40235469],
    [0.50019687, 0.56044219],
    [0.38710391, 0.72160547],
    [0.61507734, 0.72034453],
];

const ARCFACE_SIZE: usize = 112;
const SWAP_SIZE: usize = 256;
const EMBED_DIM: usize = 512;

pub struct FaceSwapper {
    scrfd: OnnxModel,
    arcface: OnnxModel,
    hyperswap: OnnxModel,
    hyperswap_source_name: String,
    hyperswap_target_name: String,
}

impl FaceSwapper {
    /// Build the swapper from the three ONNX model byte blobs. Input shapes are
    /// fixed (SCRFD 640, ArcFace 112, HyperSwap 256 + 512-d source).
    pub async fn load(
        backend: Arc<WgpuBackend>,
        scrfd_onnx: &[u8],
        arcface_onnx: &[u8],
        hyperswap_onnx: &[u8],
    ) -> Result<Self, OnnxError> {
        let scrfd = OnnxModel::load(
            Arc::clone(&backend),
            scrfd_onnx,
            &input_shape_map(scrfd_onnx, &[1, 3, 640, 640])?,
        )
        .await?;
        let arcface = OnnxModel::load(
            Arc::clone(&backend),
            arcface_onnx,
            &input_shape_map(
                arcface_onnx,
                &[1, 3, ARCFACE_SIZE as i64, ARCFACE_SIZE as i64],
            )?,
        )
        .await?;
        let hyperswap = OnnxModel::load(
            Arc::clone(&backend),
            hyperswap_onnx,
            &hyperswap_shapes(hyperswap_onnx)?,
        )
        .await?;

        // HyperSwap has two inputs: the 512-d source embedding and the 256x256
        // target crop. Identify by name (FaceFusion uses "source"/"target").
        let source_name = hyperswap
            .input_names
            .iter()
            .find(|n| n.to_lowercase().contains("source") || n.to_lowercase().contains("emb"))
            .cloned()
            .unwrap_or_else(|| hyperswap.input_names[0].clone());
        let target_name = hyperswap
            .input_names
            .iter()
            .find(|n| **n != source_name)
            .cloned()
            .unwrap_or_else(|| hyperswap.input_names[1].clone());

        Ok(Self {
            scrfd,
            arcface,
            hyperswap,
            hyperswap_source_name: source_name,
            hyperswap_target_name: target_name,
        })
    }

    /// Extract the 512-d (L2-normalized) ArcFace embedding of the first face in
    /// `img`. Errors if no face is detected.
    pub async fn source_embedding(&self, img: &Image) -> Result<Vec<f32>, OnnxError> {
        let faces = detect::detect(&self.scrfd, img).await?;
        let face = faces
            .first()
            .ok_or_else(|| OnnxError::Unsupported("no face in source image".into()))?;
        let crop = warp_to_template(img, &face.landmark5, &TEMPLATE_ARCFACE_112_V2, ARCFACE_SIZE).0;
        // ArcFace: RGB, p/127.5 - 1.
        let input = to_nchw(&crop, false, 1.0 / 127.5, -1.0);
        let mut feeds = HashMap::new();
        feeds.insert(self.arcface.input_names[0].clone(), input);
        let out = self.arcface.run(&feeds).await?;
        let (_shape, emb) = out
            .get(&self.arcface.output_names[0])
            .ok_or_else(|| OnnxError::Unsupported("arcface output missing".into()))?;
        Ok(l2_normalize(emb))
    }

    /// Detect every face in `frame` and swap each one using `source_embedding`.
    /// Returns a new frame; if no face is found, returns a clone unchanged.
    pub async fn swap_frame(
        &self,
        frame: &Image,
        source_embedding: &[f32],
    ) -> Result<Image, OnnxError> {
        let prof = std::env::var_os("THINFER_FS_PROFILE").is_some();
        let t = std::time::Instant::now();
        let faces = detect::detect(&self.scrfd, frame).await?;
        let detect_ms = t.elapsed().as_secs_f64() * 1e3;
        let mut result = frame.clone();
        for face in &faces {
            result = self.swap_one(&result, face, source_embedding).await?;
        }
        if prof {
            eprintln!("[fs-profile] detect={detect_ms:.1}ms faces={}", faces.len());
        }
        Ok(result)
    }

    /// Detect faces in `img` (NMS-filtered, descending score). Exposed for
    /// diagnostics / tests.
    pub async fn detect(&self, img: &Image) -> Result<Vec<Face>, OnnxError> {
        detect::detect(&self.scrfd, img).await
    }

    async fn swap_one(
        &self,
        frame: &Image,
        face: &Face,
        source_embedding: &[f32],
    ) -> Result<Image, OnnxError> {
        let prof = std::env::var_os("THINFER_FS_PROFILE").is_some();
        let t = std::time::Instant::now();
        let (crop, matrix) =
            warp_to_template(frame, &face.landmark5, &TEMPLATE_ARCFACE_128, SWAP_SIZE);
        // HyperSwap target: RGB, (p/255 - 0.5)/0.5 == p/127.5 - 1.
        let target = to_nchw(&crop, false, 1.0 / 127.5, -1.0);
        let warp_ms = t.elapsed().as_secs_f64() * 1e3;
        let mut feeds = HashMap::new();
        feeds.insert(self.hyperswap_target_name.clone(), target);
        feeds.insert(
            self.hyperswap_source_name.clone(),
            source_embedding.to_vec(),
        );
        let t = std::time::Instant::now();
        let out = self.hyperswap.run(&feeds).await?;
        let run_ms = t.elapsed().as_secs_f64() * 1e3;
        // Output: prefer the "output" tensor; fall back to the first non-mask.
        let (_shape, data) = out
            .iter()
            .find(|(k, _)| k.to_lowercase().contains("output"))
            .or_else(|| out.iter().find(|(k, _)| !k.to_lowercase().contains("mask")))
            .map(|(_, v)| v)
            .ok_or_else(|| OnnxError::Unsupported("hyperswap output missing".into()))?
            .clone();
        // De-normalize CHW [-1, 1] -> HWC RGB [0, 255].
        let swapped = nchw_to_image(&data, SWAP_SIZE, SWAP_SIZE);
        let t = std::time::Instant::now();
        let out_img = image::paste_back(frame, &swapped, &matrix);
        if prof {
            eprintln!(
                "[fs-profile] warp={warp_ms:.1}ms hyperswap_run={run_ms:.1}ms paste={:.1}ms",
                t.elapsed().as_secs_f64() * 1e3
            );
        }
        Ok(out_img)
    }
}

/// Affine (landmarks -> template*size) and the warped crop.
fn warp_to_template(
    img: &Image,
    landmarks5: &[[f32; 2]; 5],
    template: &[[f32; 2]; 5],
    size: usize,
) -> (Image, Affine) {
    let dst: Vec<[f32; 2]> = template
        .iter()
        .map(|&[x, y]| [x * size as f32, y * size as f32])
        .collect();
    let matrix = image::estimate_similarity_transform(landmarks5, &dst);
    let crop = image::warp_affine(img, &matrix, size, size);
    (crop, matrix)
}

/// HWC RGB `[0,255]` -> NCHW f32 with per-channel `a*p + b`; `bgr` swaps R/B.
fn to_nchw(img: &Image, bgr: bool, a: f32, b: f32) -> Vec<f32> {
    let hw = img.w * img.h;
    let mut out = vec![0.0f32; 3 * hw];
    for i in 0..hw {
        let r = img.data[i * 3];
        let g = img.data[i * 3 + 1];
        let bl = img.data[i * 3 + 2];
        let (c0, c1, c2) = if bgr { (bl, g, r) } else { (r, g, bl) };
        out[i] = a * c0 + b;
        out[hw + i] = a * c1 + b;
        out[2 * hw + i] = a * c2 + b;
    }
    out
}

/// NCHW `[1,3,h,w]` in `[-1,1]` -> HWC RGB `[0,255]` Image (`*0.5+0.5`, clamp).
fn nchw_to_image(data: &[f32], w: usize, h: usize) -> Image {
    let hw = w * h;
    let mut img = Image::new(w, h);
    for i in 0..hw {
        for c in 0..3 {
            let v = (data[c * hw + i] * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0;
            img.data[i * 3 + c] = v;
        }
    }
    img
}

fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    let inv = if norm > 0.0 { 1.0 / norm } else { 0.0 };
    v.iter().map(|x| x * inv).collect()
}

/// Build the `{first_input_name: shape}` map for a single-input model.
fn input_shape_map(onnx: &[u8], shape: &[i64]) -> Result<HashMap<String, Vec<i64>>, OnnxError> {
    let graph =
        thinfer_core::onnx::proto::parse_model(onnx).map_err(|e| OnnxError::Io(e.to_string()))?;
    let name = graph
        .inputs
        .iter()
        .find(|vi| !graph.initializers.iter().any(|t| t.name == vi.name))
        .map(|vi| vi.name.clone())
        .ok_or_else(|| OnnxError::Unsupported("model has no input".into()))?;
    let mut m = HashMap::new();
    m.insert(name, shape.to_vec());
    Ok(m)
}

/// HyperSwap input-shape map: 512-d source + 256x256x3 target, bound by name.
fn hyperswap_shapes(onnx: &[u8]) -> Result<HashMap<String, Vec<i64>>, OnnxError> {
    let graph =
        thinfer_core::onnx::proto::parse_model(onnx).map_err(|e| OnnxError::Io(e.to_string()))?;
    let mut m = HashMap::new();
    for vi in &graph.inputs {
        if graph.initializers.iter().any(|t| t.name == vi.name) {
            continue;
        }
        // Distinguish by declared rank: the 4D input is the target image.
        let is_target = vi.dims.len() == 4;
        let shape = if is_target {
            vec![1, 3, SWAP_SIZE as i64, SWAP_SIZE as i64]
        } else {
            vec![1, EMBED_DIM as i64]
        };
        m.insert(vi.name.clone(), shape);
    }
    Ok(m)
}
