//! What the OS lets this process keep resident in device memory.
//!
//! A VRAM budget is a ceiling the deployer sets, but the OS can grant less:
//! on Windows, WDDM gives each process a device-local budget below the card's
//! size (other processes, the desktop), and allocations past it are silently
//! demoted to system memory, where every kernel read crosses PCIe (measured:
//! a 784 MB LM head at 14 GB/s instead of 240). Sizing pools inside the
//! granted budget keeps everything resident.

/// Bytes of device-local memory this process can still allocate before the
/// OS demotes allocations (`VK_EXT_memory_budget`: the largest device-local
/// heap's budget minus its current usage). `None` off Vulkan or without the
/// extension.
#[cfg(not(any(windows, target_os = "linux")))]
pub fn device_local_headroom(_device: &wgpu::Device) -> Option<u64> {
    None
}

/// See the other-platform stub's doc.
#[cfg(any(windows, target_os = "linux"))]
pub fn device_local_headroom(device: &wgpu::Device) -> Option<u64> {
    use ash::vk;
    use wgpu::hal::api::Vulkan;
    // SAFETY: plain physical-device queries through the live hal device's
    // instance; the budget struct outlives the call it is chained into.
    unsafe {
        let hal = device.as_hal::<Vulkan>()?;
        let instance = hal.shared_instance().raw_instance();
        let pd = hal.raw_physical_device();
        let exts = instance.enumerate_device_extension_properties(pd).ok()?;
        let supported = exts
            .iter()
            .any(|e| e.extension_name_as_c_str() == Ok(ash::ext::memory_budget::NAME));
        if !supported {
            return None;
        }
        let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
        let mut props = vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
        instance.get_physical_device_memory_properties2(pd, &mut props);
        let heaps = props.memory_properties;
        (0..heaps.memory_heap_count as usize)
            .filter(|&i| {
                heaps.memory_heaps[i]
                    .flags
                    .contains(vk::MemoryHeapFlags::DEVICE_LOCAL)
            })
            .map(|i| budget.heap_budget[i].saturating_sub(budget.heap_usage[i]))
            .max()
    }
}

/// Allocations outside a model's VRAM budget (driver modules, staging,
/// accelerator context) kept free below the OS headroom.
const UNBUDGETED_MARGIN: u64 = 384 << 20;

/// `requested`, lowered to what the OS will keep resident (see the module
/// doc). Never raises it.
pub fn resident_vram_budget(device: &wgpu::Device, requested: u64) -> u64 {
    match device_local_headroom(device) {
        Some(h) => requested.min(h.saturating_sub(UNBUDGETED_MARGIN)),
        None => requested,
    }
}
