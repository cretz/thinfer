//! Optional native CUDA fast path for hot ops on NVIDIA GPUs (Windows and
//! Linux). An optimization, never a requirement: every op here has a wgpu
//! implementation that runs when CUDA is unavailable (non-NVIDIA adapter, no
//! driver or libraries, unsupported shape/dtype, or macOS). CUDA ops work in
//! place on wgpu's VRAM through external-memory interop
//! ([`interop::SharedBuffer`]), so the residency/streaming engine is unchanged.

pub mod accel;
pub mod device;
pub mod elementwise;
pub mod gemm;
pub mod gemm_fp4;
pub mod interop;
pub mod libs;
mod nvrtc;
pub mod sdpa;
pub mod vsa;

pub use accel::CudaAccel;
pub use device::{CudaDevice, Unavailable, probe};
pub use interop::{CudaAllocator, CudaMapping, SharedBuffer, alloc_shared};

/// Make `backend` CUDA-accelerated when its adapter is an NVIDIA GPU with a
/// working driver: every later buffer becomes CUDA-addressable and hot ops are
/// offered to [`CudaAccel`]. Returns the device name and compute capability, or
/// why CUDA is unavailable (the backend is then left untouched: wgpu only).
pub fn install(
    backend: &thinfer_core::backend::WgpuBackend,
) -> Result<(String, (i32, i32)), Unavailable> {
    let dev = std::sync::Arc::new(probe(backend.wgpu_device())?);
    let info = (dev.name.clone(), dev.sm);
    backend.set_external_allocator(std::sync::Arc::new(CudaAllocator::new(dev.clone())));
    backend.set_accelerator(std::sync::Arc::new(CudaAccel::new(dev)));
    Ok(info)
}
