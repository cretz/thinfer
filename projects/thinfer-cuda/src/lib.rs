//! Optional native CUDA fast path for hot ops on NVIDIA GPUs (Windows and
//! Linux). An optimization, never a requirement: every op here has a wgpu
//! implementation that runs when CUDA is unavailable (non-NVIDIA adapter, no
//! driver or libraries, unsupported shape/dtype, or macOS). CUDA ops work in
//! place on wgpu's VRAM through external-memory interop
//! ([`interop::SharedBuffer`]), so the residency/streaming engine is unchanged.
//! On macOS only [`install`] exists (wgpu runs on Metal there, no Vulkan interop).

#[cfg(not(target_os = "macos"))]
pub mod accel;
#[cfg(not(target_os = "macos"))]
pub mod device;
#[cfg(not(target_os = "macos"))]
pub mod elementwise;
#[cfg(not(target_os = "macos"))]
pub mod gemm;
#[cfg(not(target_os = "macos"))]
pub mod gemm_fp4;
#[cfg(not(target_os = "macos"))]
pub mod interop;
#[cfg(not(target_os = "macos"))]
pub mod libs;
#[cfg(not(target_os = "macos"))]
mod nvrtc;
#[cfg(not(target_os = "macos"))]
pub mod sdpa;
#[cfg(not(target_os = "macos"))]
pub mod vsa;

#[cfg(not(target_os = "macos"))]
pub use accel::CudaAccel;
#[cfg(not(target_os = "macos"))]
pub use device::{CudaDevice, probe};
#[cfg(not(target_os = "macos"))]
pub use interop::{CudaAllocator, CudaMapping, SharedBuffer, alloc_shared};

/// Why the CUDA path is unavailable. Logged once by the caller; never fatal.
#[derive(Debug)]
pub enum Unavailable {
    /// The wgpu device is not on the Vulkan backend (no shared-memory interop).
    NotVulkan,
    /// The Vulkan driver reports no valid device UUID.
    NoVulkanUuid,
    /// The Vulkan device cannot export memory to another API on this OS.
    NoExternalMemory,
    /// The CUDA driver library is not installed.
    NoDriver,
    /// The driver loaded but a call failed.
    Driver(String),
    /// No CUDA device has the adapter's UUID (e.g. a non-NVIDIA adapter).
    NoMatch,
}

impl std::fmt::Display for Unavailable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotVulkan => f.write_str("wgpu device is not on the Vulkan backend"),
            Self::NoVulkanUuid => f.write_str("Vulkan driver reports no device UUID"),
            Self::NoExternalMemory => f.write_str("Vulkan device cannot export memory to CUDA"),
            Self::NoDriver => f.write_str("CUDA driver library not found"),
            Self::Driver(e) => write!(f, "CUDA driver: {e}"),
            Self::NoMatch => f.write_str("no CUDA device matches the wgpu adapter"),
        }
    }
}

/// macOS: wgpu is on Metal, so there is nothing to share VRAM with.
#[cfg(target_os = "macos")]
pub fn install(
    _backend: &thinfer_core::backend::WgpuBackend,
) -> Result<(String, (i32, i32)), Unavailable> {
    Err(Unavailable::NotVulkan)
}

/// Make `backend` CUDA-accelerated when its adapter is an NVIDIA GPU with a
/// working driver: every later buffer becomes CUDA-addressable and hot ops are
/// offered to [`CudaAccel`]. Returns the device name and compute capability, or
/// why CUDA is unavailable (the backend is then left untouched: wgpu only).
#[cfg(not(target_os = "macos"))]
pub fn install(
    backend: &thinfer_core::backend::WgpuBackend,
) -> Result<(String, (i32, i32)), Unavailable> {
    let dev = std::sync::Arc::new(probe(backend.wgpu_device())?);
    let info = (dev.name.clone(), dev.sm);
    backend.set_external_allocator(std::sync::Arc::new(CudaAllocator::new(dev.clone())));
    backend.set_accelerator(std::sync::Arc::new(CudaAccel::new(dev)));
    Ok(info)
}
