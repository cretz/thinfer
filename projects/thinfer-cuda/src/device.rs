//! Finding the CUDA device that IS the wgpu adapter. CUDA ops share VRAM with
//! the wgpu engine (external memory), so they must run on the same physical
//! GPU: the match is by device UUID (Vulkan `deviceUUID` == CUDA `cuDeviceGetUuid`
//! on NVIDIA). Anything else (non-Vulkan backend, no driver, no match) yields
//! `None` and the caller keeps the wgpu path.

use std::sync::Arc;

use cudarc::driver::{CudaContext, result, sys};

use crate::Unavailable;

/// The CUDA side of the wgpu adapter.
pub struct CudaDevice {
    pub ctx: Arc<CudaContext>,
    pub name: String,
    /// Compute capability `(major, minor)`, e.g. `(12, 0)` for consumer Blackwell.
    pub sm: (i32, i32),
}

/// Whether the CUDA driver library loads. cudarc PANICS on the first driver
/// call without it, so check this before any `CudaContext` or `result` call.
pub fn driver_present() -> bool {
    // SAFETY: only probes whether the shared library can be loaded.
    unsafe { sys::is_culib_present() }
}

fn driver_err(e: result::DriverError) -> Unavailable {
    Unavailable::Driver(format!("{e:?}"))
}

/// The wgpu device's Vulkan `deviceUUID`.
fn vulkan_uuid(device: &wgpu::Device) -> Result<[u8; 16], Unavailable> {
    // SAFETY: the guard is only used to read immutable physical-device
    // properties; no handle outlives it.
    let hal = unsafe { device.as_hal::<wgpu::hal::api::Vulkan>() }.ok_or(Unavailable::NotVulkan)?;
    let instance = hal.shared_instance().raw_instance();
    let mut id = ash::vk::PhysicalDeviceIDProperties::default();
    let mut props = ash::vk::PhysicalDeviceProperties2::default().push_next(&mut id);
    // SAFETY: valid instance + physical device from the live wgpu device.
    unsafe { instance.get_physical_device_properties2(hal.raw_physical_device(), &mut props) };
    let uuid = id.device_uuid;
    if uuid.iter().all(|b| *b == 0) {
        return Err(Unavailable::NoVulkanUuid);
    }
    Ok(uuid)
}

/// The CUDA device backing `device`, with a retained primary context.
pub fn probe(device: &wgpu::Device) -> Result<CudaDevice, Unavailable> {
    let want = vulkan_uuid(device)?;
    if !crate::interop::export_supported(device) {
        return Err(Unavailable::NoExternalMemory);
    }
    if !driver_present() {
        return Err(Unavailable::NoDriver);
    }
    result::init().map_err(driver_err)?;
    let count = result::device::get_count().map_err(driver_err)?;
    for ordinal in 0..count {
        let dev = result::device::get(ordinal).map_err(driver_err)?;
        let uuid = result::device::get_uuid(dev).map_err(driver_err)?;
        if uuid.bytes.map(|b| b.to_ne_bytes()[0]) != want {
            continue;
        }
        let ctx = CudaContext::new(ordinal as usize).map_err(driver_err)?;
        let name = ctx.name().map_err(driver_err)?;
        let sm = ctx.compute_capability().map_err(driver_err)?;
        return Ok(CudaDevice { ctx, name, sm });
    }
    Err(Unavailable::NoMatch)
}
