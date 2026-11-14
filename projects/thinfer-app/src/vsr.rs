//! RelayVSR super-resolution of a finished video (`--super-res`): the whole
//! clip upscaled by 2 or 4 before encoding (`thinfer_models::wan::relayvsr`).

use std::sync::Arc;

use thinfer_core::backend::WgpuBackend;
use thinfer_core::residency::WeightResidency;
use thinfer_models::wan::relayvsr::{
    RelayVsrPipeline, UpscaleInputs, UpscaledVideo, manifest as vsr, open_relayvsr_source,
};
use thinfer_native::MmapFileOpener;

use crate::config::ResidencyBudget;
use crate::download::resolve_role;
use crate::model::SuperRes;
use crate::progress::ProgressSink;

/// Interleaved RGB8 frames `[num_frames, height, width, 3]`.
pub struct Rgb8Video {
    /// Job plaintext: page-locked on a server, zeroized on drop.
    pub frames: thinfer_core::secmem::Locked<u8>,
    pub num_frames: usize,
    pub height: usize,
    pub width: usize,
}

/// VRAM residency cap for the upscale (weights + activations).
const VRAM_BUDGET_CAP: u64 = 3 << 30;

pub async fn upscale(
    backend: &Arc<WgpuBackend>,
    video: &Rgb8Video,
    sr: SuperRes,
    seed: u64,
    budget: ResidencyBudget,
    sink: &dyn ProgressSink,
) -> Result<UpscaledVideo, String> {
    let up = sr.factor();
    let (dit_role, flash_role) = vsr::roles(up);
    let dit = resolve_role(&vsr::MANIFEST, dit_role)?;
    let flash = resolve_role(&vsr::MANIFEST, flash_role)?;
    let open = |p: std::path::PathBuf| async move {
        MmapFileOpener::new(&p)
            .await
            .map_err(|e| format!("open {}: {e}", p.display()))
    };
    let source = open_relayvsr_source(open(dit).await?, open(flash).await?)
        .await
        .map_err(|e| format!("open RelayVSR: {e:?}"))?;
    // Every weight is read once (one keyframe forward, one Flash pass), so a
    // cache beyond a few blocks of prefetch buys nothing; the cap keeps the
    // pass well inside the card next to whatever the generation still holds.
    let budget = ResidencyBudget {
        vram_bytes: budget.vram_bytes.min(VRAM_BUDGET_CAP),
        ..budget
    };
    let residency = WeightResidency::new(source, budget);
    let pipe = RelayVsrPipeline::load(Arc::clone(backend), residency, up)
        .await
        .map_err(|e| format!("load RelayVSR: {e:?}"))?;
    sink.note(&format!(
        "Upscaling {} frames {}x{} -> {}x{} (RelayVSR {up}x)",
        video.num_frames,
        video.width,
        video.height,
        video.width * up,
        video.height * up
    ));
    let cancelled = || sink.cancelled();
    pipe.upscale(
        &UpscaleInputs {
            frames: &video.frames,
            num_frames: video.num_frames,
            height: video.height,
            width: video.width,
            seed,
            detail: 1.0,
        },
        None,
        Some(&cancelled),
    )
    .await
    .map_err(|e| format!("RelayVSR upscale: {e:?}"))
}
