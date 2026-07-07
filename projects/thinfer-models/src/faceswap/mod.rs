//! Face-swap pipeline: SCRFD detect -> ArcFace source embedding -> HyperSwap
//! per-face swap -> feathered paste-back. Mirrors intabai
//! `web/src/video-face-swap/pipeline.ts` (HyperSwap path, SCRFD detector, no
//! enhancer/XSeg). The three ONNX models run on the thinfer GPU executor
//! (`thinfer_core::onnx`); geometry is host-side (`image`).

pub mod detect;
pub mod dwpose;
pub mod enhancer;
pub mod image;
pub mod occluder;

use std::collections::HashMap;
use std::sync::Arc;

use thinfer_core::backend::WgpuBackend;
use thinfer_core::onnx::{OnnxError, OnnxModel};

pub use detect::Face;
use enhancer::Enhancer;
use image::{Affine, Image};
use occluder::Occluder;

/// Compositing/quality options for the swap. Every field defaults off so the
/// baseline (feather-only paste) is unchanged; each one is user-opt-in because
/// it changes the visual output (or, for detection stride, trades temporal
/// accuracy for speed).
#[derive(Clone, Copy, Debug, Default)]
pub struct SwapOptions {
    /// Intersect the paste mask with HyperSwap's own confidence mask output
    /// (its 2nd output, otherwise ignored). Tightens the blend to the region the
    /// GAN actually synthesized. Free (already computed).
    pub hyperswap_mask: bool,
    /// Intersect the paste mask with the XSeg occlusion mask (requires an XSeg
    /// model to be loaded). Lets occluders (hands/hair/glasses) crossing the face
    /// show the original frame through instead of swapped pixels. Adds one ONNX
    /// forward per face.
    pub occlusion: bool,
    /// Run the GFPGAN face enhancer on the swapped face (requires a GFPGAN model
    /// to be loaded). Restores GAN detail (skin/eyes/teeth). Adds one ONNX
    /// forward per face.
    pub enhance: bool,
}

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

/// FFHQ 5-point template (normalized) for the face enhancer, from FaceFusion
/// `ffhq_512`. Multiplied by the enhancer crop size (512).
#[allow(clippy::excessive_precision)]
const TEMPLATE_FFHQ_512: [[f32; 2]; 5] = [
    [0.37691676, 0.46864664],
    [0.62285697, 0.46912813],
    [0.50123859, 0.61331904],
    [0.39308822, 0.72541100],
    [0.61150205, 0.72490465],
];

const ARCFACE_SIZE: usize = 112;
const SWAP_SIZE: usize = 256;
const EMBED_DIM: usize = 512;
/// HyperSwap forward batch: how many aligned face crops are stacked into one
/// GPU dispatch. The 256x256 swapper badly underutilizes the GPU at batch 1
/// (measured ~13x over compute-ideal); stacking crops fills idle SM capacity
/// for ~1.75x per-crop throughput at B=4 (the knee before the 8 GiB card OOMs
/// on activations). Batching is bit-identical to serial (each row is
/// independent; the single source embedding broadcasts across the batch), so
/// it is always on. Video batches crops across consecutive frames; a still
/// image with fewer faces pads the batch (its extra rows are discarded).
///
/// Public so a video pipeline can size its frame window to the batch.
pub const SWAP_BATCH: usize = 4;

pub struct FaceSwapper {
    scrfd: OnnxModel,
    arcface: OnnxModel,
    hyperswap: OnnxModel,
    hyperswap_source_name: String,
    hyperswap_target_name: String,
    occluder: Option<Occluder>,
    enhancer: Option<Enhancer>,
    opts: SwapOptions,
}

impl FaceSwapper {
    /// Build the swapper from the three ONNX model byte blobs (feather-only
    /// paste, no occlusion). Input shapes are fixed (SCRFD 640, ArcFace 112,
    /// HyperSwap 256 + 512-d source).
    pub async fn load(
        backend: Arc<WgpuBackend>,
        scrfd_onnx: &[u8],
        arcface_onnx: &[u8],
        hyperswap_onnx: &[u8],
    ) -> Result<Self, OnnxError> {
        Self::load_with(
            backend,
            scrfd_onnx,
            arcface_onnx,
            hyperswap_onnx,
            SwapOptions::default(),
            None,
            None,
        )
        .await
    }

    /// As [`load`], with compositing options and optional XSeg occlusion + GFPGAN
    /// enhancer models. `xseg_onnx` must be `Some` when `opts.occlusion` is set;
    /// `gfpgan_onnx` must be `Some` when `opts.enhance` is set.
    pub async fn load_with(
        backend: Arc<WgpuBackend>,
        scrfd_onnx: &[u8],
        arcface_onnx: &[u8],
        hyperswap_onnx: &[u8],
        opts: SwapOptions,
        xseg_onnx: Option<&[u8]>,
        gfpgan_onnx: Option<&[u8]>,
    ) -> Result<Self, OnnxError> {
        let occluder = match (opts.occlusion, xseg_onnx) {
            (true, Some(bytes)) => Some(Occluder::load(Arc::clone(&backend), bytes).await?),
            (true, None) => {
                return Err(OnnxError::Unsupported(
                    "occlusion mask requested but no XSeg model provided".into(),
                ));
            }
            _ => None,
        };
        let enhancer = match (opts.enhance, gfpgan_onnx) {
            (true, Some(bytes)) => Some(Enhancer::load(Arc::clone(&backend), bytes).await?),
            (true, None) => {
                return Err(OnnxError::Unsupported(
                    "enhancer requested but no GFPGAN model provided".into(),
                ));
            }
            _ => None,
        };
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
        // Opt-in i8 DP4A conv for HyperSwap ONLY (the detector/embedder keep f32
        // -- their i8 error is unvalidated and detection quality is load-bearing).
        // Gated OFF by default; i8 rounds the GAN and needs a visual eyeball.
        let i8_conv = std::env::var_os("THINFER_ONNX_I8_CONV").is_some();
        let hyperswap = OnnxModel::load_i8(
            Arc::clone(&backend),
            hyperswap_onnx,
            &hyperswap_shapes(hyperswap_onnx, SWAP_BATCH)?,
            i8_conv,
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
            occluder,
            enhancer,
            opts,
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
        if prof {
            eprintln!("[fs-profile] detect={detect_ms:.1}ms faces={}", faces.len());
        }
        self.swap_predetected(frame, &faces, source_embedding).await
    }

    /// Swap `faces` (already detected) into `frame`. Lets a video pipeline run
    /// detection on a stride and reuse the last frame's faces for the frames in
    /// between (the strided-detection speed option): the per-frame HyperSwap
    /// still runs, only the SCRFD pass is skipped.
    pub async fn swap_predetected(
        &self,
        frame: &Image,
        faces: &[Face],
        source_embedding: &[f32],
    ) -> Result<Image, OnnxError> {
        let mut result = [frame.clone()];
        let faces_win = [faces.to_vec()];
        self.swap_predetected_multi(&mut result, &faces_win, source_embedding)
            .await?;
        let [out] = result;
        Ok(out)
    }

    /// Swap every face across a window of frames in place, batching the
    /// HyperSwap forward across all crops (up to [`SWAP_BATCH`] per dispatch) so
    /// the GPU runs full instead of one small crop at a time. `faces_per_frame`
    /// is aligned to `frames` (detection is the caller's job, so a video can run
    /// SCRFD on a stride). This is the throughput entry point; a single frame is
    /// the len-1 case. Result is identical to swapping each face serially (the
    /// swapper is per-crop independent), only faster.
    pub async fn swap_predetected_multi(
        &self,
        frames: &mut [Image],
        faces_per_frame: &[Vec<Face>],
        source_embedding: &[f32],
    ) -> Result<(), OnnxError> {
        assert_eq!(frames.len(), faces_per_frame.len());
        let prof = std::env::var_os("THINFER_FS_PROFILE").is_some();

        // Warp every face to the swap template up front; a job records where its
        // result pastes back. The target crop reads the frame's ORIGINAL pixels
        // (before any paste), matching the serial path for non-overlapping faces.
        struct CropJob {
            frame_idx: usize,
            face: Face,
            matrix: Affine,
            crop: Image,
            target: Vec<f32>,
        }
        let mut jobs: Vec<CropJob> = Vec::new();
        for (frame_idx, faces) in faces_per_frame.iter().enumerate() {
            for face in faces {
                let (crop, matrix) = warp_to_template(
                    &frames[frame_idx],
                    &face.landmark5,
                    &TEMPLATE_ARCFACE_128,
                    SWAP_SIZE,
                );
                // HyperSwap target: RGB, (p/255 - 0.5)/0.5 == p/127.5 - 1.
                let target = to_nchw(&crop, false, 1.0 / 127.5, -1.0);
                jobs.push(CropJob {
                    frame_idx,
                    face: face.clone(),
                    matrix,
                    crop,
                    target,
                });
            }
        }
        if jobs.is_empty() {
            return Ok(());
        }

        // Batched HyperSwap: stack up to SWAP_BATCH crops per dispatch.
        let t = std::time::Instant::now();
        let mut swaps: Vec<(Image, Option<Vec<f32>>)> = Vec::with_capacity(jobs.len());
        for chunk in jobs.chunks(SWAP_BATCH) {
            let targets: Vec<&[f32]> = chunk.iter().map(|j| j.target.as_slice()).collect();
            swaps.extend(self.hyperswap_batch(&targets, source_embedding).await?);
        }
        if prof {
            eprintln!(
                "[fs-profile] hyperswap_batch crops={} run={:.1}ms",
                jobs.len(),
                t.elapsed().as_secs_f64() * 1e3
            );
        }

        // Composite each crop back into its frame, in job order (faces on the
        // same frame paste in detection order, matching the serial chain).
        for (job, (swapped, mask)) in jobs.iter().zip(swaps) {
            let extra = self.compose_extra_mask(&job.crop, mask).await?;
            image::paste_back_into(
                &mut frames[job.frame_idx],
                &swapped,
                &job.matrix,
                extra.as_deref(),
            );
            if let Some(enh) = &self.enhancer {
                let (crop, ffhq) = warp_to_template(
                    &frames[job.frame_idx],
                    &job.face.landmark5,
                    &TEMPLATE_FFHQ_512,
                    enhancer::GFPGAN_SIZE,
                );
                let restored = enh.enhance(&crop).await?;
                image::paste_back_into(&mut frames[job.frame_idx], &restored, &ffhq, None);
            }
        }
        Ok(())
    }

    /// Run HyperSwap over `targets` (<= [`SWAP_BATCH`] NCHW crops) in one
    /// dispatch. The batch is padded to SWAP_BATCH (extra rows discarded); the
    /// single `source_embedding` broadcasts across every row. Returns, per input
    /// crop, the de-normalized swapped image and its raw confidence mask (if the
    /// model emits one).
    async fn hyperswap_batch(
        &self,
        targets: &[&[f32]],
        source_embedding: &[f32],
    ) -> Result<Vec<(Image, Option<Vec<f32>>)>, OnnxError> {
        let n = targets.len();
        assert!((1..=SWAP_BATCH).contains(&n));
        let per = 3 * SWAP_SIZE * SWAP_SIZE;
        let mut stacked = vec![0.0f32; SWAP_BATCH * per];
        for (i, t) in targets.iter().enumerate() {
            stacked[i * per..(i + 1) * per].copy_from_slice(t);
        }
        let mut feeds = HashMap::new();
        feeds.insert(self.hyperswap_target_name.clone(), stacked);
        feeds.insert(
            self.hyperswap_source_name.clone(),
            source_embedding.to_vec(),
        );
        let out = self.hyperswap.run(&feeds).await?;

        // Locate the 3-channel image output and the optional 1-channel mask by
        // shape (batch-robust; the name heuristic only picks the tensor).
        let img = out
            .iter()
            .find(|(k, (sh, _))| sh.len() == 4 && sh[1] == 3 && k.to_lowercase().contains("output"))
            .or_else(|| out.iter().find(|(_, (sh, _))| sh.len() == 4 && sh[1] == 3))
            .map(|(_, v)| v)
            .ok_or_else(|| OnnxError::Unsupported("hyperswap output missing".into()))?;
        let mask = out
            .iter()
            .find(|(k, (sh, _))| sh.len() == 4 && sh[1] == 1 && k.to_lowercase().contains("mask"))
            .or_else(|| out.iter().find(|(_, (sh, _))| sh.len() == 4 && sh[1] == 1))
            .map(|(_, (_, d))| d);

        let hw = SWAP_SIZE * SWAP_SIZE;
        let mut res = Vec::with_capacity(n);
        for i in 0..n {
            let swapped = nchw_to_image(&img.1[i * per..(i + 1) * per], SWAP_SIZE, SWAP_SIZE);
            let m = mask.map(|md| md[i * hw..(i + 1) * hw].to_vec());
            res.push((swapped, m));
        }
        Ok(res)
    }

    /// Build the optional crop-space paste mask = min of the enabled masks
    /// (HyperSwap's own confidence output, XSeg occlusion). `None` -> feather
    /// only.
    async fn compose_extra_mask(
        &self,
        crop: &Image,
        hyperswap_mask: Option<Vec<f32>>,
    ) -> Result<Option<Vec<f32>>, OnnxError> {
        let mut extra: Option<Vec<f32>> = None;
        if self.opts.hyperswap_mask
            && let Some(m) = hyperswap_mask
            && m.len() == SWAP_SIZE * SWAP_SIZE
        {
            extra = Some(m.iter().map(|v| v.clamp(0.0, 1.0)).collect());
        }
        if let Some(occ) = &self.occluder {
            let occ_mask = occ.mask(crop).await?;
            extra = Some(match extra {
                Some(mut e) => {
                    for (a, &b) in e.iter_mut().zip(&occ_mask) {
                        *a = a.min(b);
                    }
                    e
                }
                None => occ_mask,
            });
        }
        Ok(extra)
    }

    /// Detect faces in `img` (NMS-filtered, descending score). Exposed for
    /// diagnostics / tests.
    pub async fn detect(&self, img: &Image) -> Result<Vec<Face>, OnnxError> {
        detect::detect(&self.scrfd, img).await
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

/// HyperSwap input-shape map: 512-d source + `batch`x256x256x3 target, bound by
/// name. The source stays `[1, 512]` and broadcasts across the target batch (one
/// source face swapped into every crop).
fn hyperswap_shapes(onnx: &[u8], batch: usize) -> Result<HashMap<String, Vec<i64>>, OnnxError> {
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
            vec![batch as i64, 3, SWAP_SIZE as i64, SWAP_SIZE as i64]
        } else {
            vec![1, EMBED_DIM as i64]
        };
        m.insert(vi.name.clone(), shape);
    }
    Ok(m)
}
