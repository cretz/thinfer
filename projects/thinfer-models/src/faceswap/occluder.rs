//! XSeg occlusion mask (FaceFusion `xseg_1.onnx`). Given an aligned 256 face
//! crop, predicts a per-pixel "is-face-skin" mask in `[0,1]`: it drops to 0 over
//! occluders (hands, hair, glasses, mic) crossing the face. `min`-combined with
//! the paste feather so occluders show the original frame through instead of the
//! swapped pixels. Mirrors FaceFusion `face_masker.create_occlusion_mask`
//! (NHWC BGR input, `/255`; output clipped + lightly blurred).

use std::collections::HashMap;

use std::sync::Arc;
use thinfer_core::backend::WgpuBackend;
use thinfer_core::onnx::{OnnxError, OnnxModel};

use super::image::Image;

const XSEG_SIZE: usize = 256;

pub struct Occluder {
    model: OnnxModel,
    input_name: String,
}

impl Occluder {
    /// Load XSeg. Input is NHWC `[1,256,256,3]` (the model transposes to NCHW
    /// internally), so the crop is fed in HWC order.
    pub async fn load(backend: Arc<WgpuBackend>, xseg_onnx: &[u8]) -> Result<Self, OnnxError> {
        let graph = thinfer_core::onnx::proto::parse_model(xseg_onnx)
            .map_err(|e| OnnxError::Io(e.to_string()))?;
        let input_name = graph
            .inputs
            .iter()
            .find(|vi| !graph.initializers.iter().any(|t| t.name == vi.name))
            .map(|vi| vi.name.clone())
            .ok_or_else(|| OnnxError::Unsupported("xseg has no input".into()))?;
        let mut shapes = HashMap::new();
        shapes.insert(
            input_name.clone(),
            vec![1, XSEG_SIZE as i64, XSEG_SIZE as i64, 3],
        );
        let model = OnnxModel::load(backend, xseg_onnx, &shapes).await?;
        Ok(Self { model, input_name })
    }

    /// Occlusion mask (`256*256`, row-major, `[0,1]`, 1 = keep-swap) for `crop`
    /// (an aligned 256 face crop, HWC RGB `[0,255]`).
    pub async fn mask(&self, crop: &Image) -> Result<Vec<f32>, OnnxError> {
        debug_assert_eq!((crop.w, crop.h), (XSEG_SIZE, XSEG_SIZE));
        // NHWC, BGR, /255 (FaceFusion feeds the opencv BGR crop directly).
        let hw = XSEG_SIZE * XSEG_SIZE;
        let mut input = vec![0.0f32; hw * 3];
        for i in 0..hw {
            input[i * 3] = crop.data[i * 3 + 2] / 255.0;
            input[i * 3 + 1] = crop.data[i * 3 + 1] / 255.0;
            input[i * 3 + 2] = crop.data[i * 3] / 255.0;
        }
        let mut feeds = HashMap::new();
        feeds.insert(self.input_name.clone(), input);
        let out = self.model.run(&feeds).await?;
        let (_shape, data) = out
            .get(&self.model.output_names[0])
            .ok_or_else(|| OnnxError::Unsupported("xseg output missing".into()))?;
        // Output is NHWC [1,256,256,1] -> already HxW. Clip, then soften edges.
        let mut mask: Vec<f32> = data.iter().map(|&v| v.clamp(0.0, 1.0)).collect();
        gaussian_blur_inplace(&mut mask, XSEG_SIZE, XSEG_SIZE, 5.0);
        Ok(mask)
    }
}

/// Separable Gaussian blur in place (border-replicate), matching the light
/// softening FaceFusion/intabai apply to the occlusion mask before compositing.
fn gaussian_blur_inplace(buf: &mut [f32], w: usize, h: usize, sigma: f32) {
    let radius = (sigma * 3.0).ceil().max(1.0) as isize;
    let mut kernel = vec![0.0f32; (2 * radius + 1) as usize];
    let mut sum = 0.0f32;
    for (k, slot) in kernel.iter_mut().enumerate() {
        let d = k as isize - radius;
        let v = (-(d * d) as f32 / (2.0 * sigma * sigma)).exp();
        *slot = v;
        sum += v;
    }
    for v in &mut kernel {
        *v /= sum;
    }
    let mut tmp = vec![0.0f32; w * h];
    // Horizontal.
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f32;
            for (k, &kv) in kernel.iter().enumerate() {
                let sx = (x as isize + k as isize - radius).clamp(0, w as isize - 1) as usize;
                acc += kv * buf[y * w + sx];
            }
            tmp[y * w + x] = acc;
        }
    }
    // Vertical.
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f32;
            for (k, &kv) in kernel.iter().enumerate() {
                let sy = (y as isize + k as isize - radius).clamp(0, h as isize - 1) as usize;
                acc += kv * tmp[sy * w + x];
            }
            buf[y * w + x] = acc;
        }
    }
}
