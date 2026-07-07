//! GFPGAN face restoration (FaceFusion `gfpgan_1.4.onnx`). Runs the enhancer on
//! an FFHQ-512-aligned face crop and returns the restored crop. Sharpens/denoises
//! the swapped face (GAN detail, teeth/eyes/skin). Mirrors FaceFusion
//! `face_enhancer` (RGB, `(p/255-0.5)/0.5` in, `*0.5+0.5` out). The warp to the
//! FFHQ template and the paste-back live in the caller (`mod.rs`).

use std::collections::HashMap;
use std::sync::Arc;

use thinfer_core::backend::WgpuBackend;
use thinfer_core::onnx::{OnnxError, OnnxModel};

use super::image::Image;

pub const GFPGAN_SIZE: usize = 512;

pub struct Enhancer {
    model: OnnxModel,
    input_name: String,
}

impl Enhancer {
    /// Load GFPGAN. Input is NCHW `[1,3,512,512]`.
    pub async fn load(backend: Arc<WgpuBackend>, gfpgan_onnx: &[u8]) -> Result<Self, OnnxError> {
        let graph = thinfer_core::onnx::proto::parse_model(gfpgan_onnx)
            .map_err(|e| OnnxError::Io(e.to_string()))?;
        let input_name = graph
            .inputs
            .iter()
            .find(|vi| !graph.initializers.iter().any(|t| t.name == vi.name))
            .map(|vi| vi.name.clone())
            .ok_or_else(|| OnnxError::Unsupported("gfpgan has no input".into()))?;
        let mut shapes = HashMap::new();
        shapes.insert(
            input_name.clone(),
            vec![1, 3, GFPGAN_SIZE as i64, GFPGAN_SIZE as i64],
        );
        let model = OnnxModel::load(backend, gfpgan_onnx, &shapes).await?;
        Ok(Self { model, input_name })
    }

    /// Restore an FFHQ-512-aligned face crop (HWC RGB `[0,255]`). Returns the
    /// enhanced crop in the same layout.
    pub async fn enhance(&self, crop: &Image) -> Result<Image, OnnxError> {
        debug_assert_eq!((crop.w, crop.h), (GFPGAN_SIZE, GFPGAN_SIZE));
        // NCHW, RGB, (p/255 - 0.5)/0.5 == p/127.5 - 1.
        let hw = GFPGAN_SIZE * GFPGAN_SIZE;
        let mut input = vec![0.0f32; hw * 3];
        for i in 0..hw {
            for c in 0..3 {
                input[c * hw + i] = crop.data[i * 3 + c] / 127.5 - 1.0;
            }
        }
        let mut feeds = HashMap::new();
        feeds.insert(self.input_name.clone(), input);
        let out = self.model.run(&feeds).await?;
        let (_shape, data) = out
            .get(&self.model.output_names[0])
            .ok_or_else(|| OnnxError::Unsupported("gfpgan output missing".into()))?;
        // NCHW [-1,1] -> HWC RGB [0,255] (*0.5+0.5, clamp).
        let mut img = Image::new(GFPGAN_SIZE, GFPGAN_SIZE);
        for i in 0..hw {
            for c in 0..3 {
                img.data[i * 3 + c] = ((data[c * hw + i] * 0.5 + 0.5).clamp(0.0, 1.0)) * 255.0;
            }
        }
        Ok(img)
    }
}
