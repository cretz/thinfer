//! One VRAM allocation visible to both engines: a Vulkan buffer the wgpu engine
//! binds like any other, whose memory is exported (an OS handle: Win32 NT
//! handle on Windows, opaque fd on Linux) and imported into CUDA as a device
//! pointer. No copies cross the boundary; the caller orders access (wgpu
//! submit + wait, then the CUDA stream, then sync).
//!
//! Ownership: wgpu owns the Vulkan buffer and its memory (wgpu-hal frees both
//! when the `wgpu::Buffer` is destroyed). The CUDA mapping is released first, in
//! `Drop`, before the wgpu buffer field drops.

use std::sync::Arc;

use ash::vk;
use cudarc::driver::{CudaContext, sys};

use crate::device::CudaDevice;

/// A storage buffer shared by wgpu and CUDA.
pub struct SharedBuffer {
    /// CUDA's view. Declared first: released before the wgpu buffer drops.
    pub mapping: CudaMapping,
    /// The wgpu view of the same memory.
    pub buffer: wgpu::Buffer,
}

/// CUDA's import of a shared allocation; released on drop.
pub struct CudaMapping {
    ctx: Arc<CudaContext>,
    ext_mem: sys::CUexternalMemory,
    /// CUDA device pointer to the start of the buffer.
    pub ptr: sys::CUdeviceptr,
    pub size: u64,
}

impl Drop for CudaMapping {
    fn drop(&mut self) {
        // Errors here mean the context is already gone; nothing to recover.
        if self.ctx.bind_to_thread().is_ok() {
            // SAFETY: `ptr` / `ext_mem` came from this import and are released once.
            unsafe {
                let _ = sys::cuMemFree_v2(self.ptr).result();
                let _ = sys::cuDestroyExternalMemory(self.ext_mem).result();
            }
        }
    }
}

// SAFETY: the raw CUDA handles are plain driver handles, valid from any thread
// once the context is bound (every use binds first).
unsafe impl Send for CudaMapping {}
unsafe impl Sync for CudaMapping {}

/// The per-OS export/import handle plumbing.
#[cfg(windows)]
mod os {
    use super::*;

    pub const HANDLE: vk::ExternalMemoryHandleTypeFlags =
        vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32;
    pub type Handle = vk::HANDLE;

    /// Whether `device` can export memory this way.
    pub fn supported(device: &wgpu::Device) -> bool {
        device
            .features()
            .contains(wgpu::Features::VULKAN_EXTERNAL_MEMORY_WIN32)
    }

    /// # Safety
    /// `memory` is a live exportable allocation on `raw`.
    pub unsafe fn export(
        instance: &ash::Instance,
        raw: &ash::Device,
        memory: vk::DeviceMemory,
    ) -> Result<Handle, vk::Result> {
        let get = vk::MemoryGetWin32HandleInfoKHR::default()
            .memory(memory)
            .handle_type(HANDLE);
        unsafe {
            ash::khr::external_memory_win32::Device::new(instance, raw)
                .get_memory_win32_handle(&get)
        }
    }

    pub fn import_handle(h: Handle) -> sys::CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st__bindgen_ty_1 {
        sys::CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st__bindgen_ty_1 {
            win32: sys::CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st__bindgen_ty_1__bindgen_ty_1 {
                handle: h as *mut _,
                name: std::ptr::null(),
            },
        }
    }

    pub const CU_TYPE: sys::CUexternalMemoryHandleType =
        sys::CUexternalMemoryHandleType::CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32;

    /// The NT handle stays ours after import (CUDA holds its own reference).
    pub fn after_import(h: Handle, _imported: bool) {
        // SAFETY: `h` is the handle Vulkan exported to us; closed once.
        unsafe { windows_sys::Win32::Foundation::CloseHandle(h as _) };
    }
}

#[cfg(unix)]
mod os {
    use super::*;

    pub const HANDLE: vk::ExternalMemoryHandleTypeFlags =
        vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD;
    pub type Handle = std::ffi::c_int;

    /// wgpu-hal enables `VK_KHR_external_memory_fd` whenever the driver has it.
    pub fn supported(device: &wgpu::Device) -> bool {
        // SAFETY: only reads the enabled-extension list.
        unsafe { device.as_hal::<wgpu::hal::api::Vulkan>() }.is_some_and(|h| {
            h.enabled_device_extensions()
                .contains(&ash::khr::external_memory_fd::NAME)
        })
    }

    /// # Safety
    /// `memory` is a live exportable allocation on `raw`.
    pub unsafe fn export(
        instance: &ash::Instance,
        raw: &ash::Device,
        memory: vk::DeviceMemory,
    ) -> Result<Handle, vk::Result> {
        let get = vk::MemoryGetFdInfoKHR::default()
            .memory(memory)
            .handle_type(HANDLE);
        unsafe { ash::khr::external_memory_fd::Device::new(instance, raw).get_memory_fd(&get) }
    }

    pub fn import_handle(fd: Handle) -> sys::CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st__bindgen_ty_1 {
        sys::CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st__bindgen_ty_1 { fd }
    }

    pub const CU_TYPE: sys::CUexternalMemoryHandleType =
        sys::CUexternalMemoryHandleType::CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD;

    /// A successful import transfers fd ownership to CUDA; close it only if
    /// the import failed.
    pub fn after_import(fd: Handle, imported: bool) {
        if !imported {
            // SAFETY: `fd` is the descriptor Vulkan exported to us; closed once.
            unsafe { libc_close(fd) };
        }
    }

    unsafe extern "C" {
        #[link_name = "close"]
        fn libc_close(fd: std::ffi::c_int) -> std::ffi::c_int;
    }
}

/// Whether `device` can share memory with CUDA on this OS.
pub fn export_supported(device: &wgpu::Device) -> bool {
    os::supported(device)
}

fn vk_err(what: &str, e: vk::Result) -> String {
    format!("{what}: {e:?}")
}

/// Allocate a `size`-byte storage buffer shared between `device` (wgpu, Vulkan
/// backend, with exportable memory, see [`export_supported`]) and `cuda`.
pub fn alloc_shared(
    device: &wgpu::Device,
    cuda: &CudaDevice,
    size: u64,
    label: &str,
) -> Result<SharedBuffer, String> {
    use wgpu::hal::api::Vulkan;

    // Same usages as the engine's plain buffers (`WgpuBackend::allocate_in`).
    let usage = wgpu::BufferUsages::STORAGE
        | wgpu::BufferUsages::UNIFORM
        | wgpu::BufferUsages::COPY_SRC
        | wgpu::BufferUsages::COPY_DST;

    // SAFETY: raw handles are used only within this block, while the guard
    // keeps the hal device alive; the resulting buffer is handed to wgpu.
    let (hal_buffer, handle, alloc_size) = unsafe {
        let hal = device
            .as_hal::<Vulkan>()
            .ok_or("wgpu device is not on the Vulkan backend")?;
        let raw = hal.raw_device();
        let instance = hal.shared_instance().raw_instance();

        let mut ext_buf = vk::ExternalMemoryBufferCreateInfo::default().handle_types(os::HANDLE);
        let buf_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::UNIFORM_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .push_next(&mut ext_buf);
        let vk_buffer = raw
            .create_buffer(&buf_info, None)
            .map_err(|e| vk_err("vkCreateBuffer", e))?;

        let req = raw.get_buffer_memory_requirements(vk_buffer);
        let mem_props = instance.get_physical_device_memory_properties(hal.raw_physical_device());
        let Some(type_index) = (0..mem_props.memory_type_count).find(|&i| {
            req.memory_type_bits & (1 << i) != 0
                && mem_props.memory_types[i as usize]
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
        }) else {
            raw.destroy_buffer(vk_buffer, None);
            return Err("no DEVICE_LOCAL memory type for an exportable buffer".into());
        };

        let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().buffer(vk_buffer);
        let mut export = vk::ExportMemoryAllocateInfo::default().handle_types(os::HANDLE);
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(req.size)
            .memory_type_index(type_index)
            .push_next(&mut export)
            .push_next(&mut dedicated);
        let memory = match raw.allocate_memory(&alloc_info, None) {
            Ok(m) => m,
            Err(e) => {
                raw.destroy_buffer(vk_buffer, None);
                return Err(vk_err("vkAllocateMemory (exportable)", e));
            }
        };
        if let Err(e) = raw.bind_buffer_memory(vk_buffer, memory, 0) {
            raw.destroy_buffer(vk_buffer, None);
            raw.free_memory(memory, None);
            return Err(vk_err("vkBindBufferMemory", e));
        }
        let handle = match os::export(instance, raw, memory) {
            Ok(h) => h,
            Err(e) => {
                raw.destroy_buffer(vk_buffer, None);
                raw.free_memory(memory, None);
                return Err(vk_err("export memory handle", e));
            }
        };
        (
            wgpu::hal::vulkan::Buffer::from_raw_managed(vk_buffer, memory, 0, req.size),
            handle,
            req.size,
        )
    };

    // SAFETY: `hal_buffer` was created on this device with `usage`-compatible
    // flags and non-zero size; wgpu takes ownership of buffer + memory.
    let buffer = unsafe {
        device.create_buffer_from_hal::<Vulkan>(
            hal_buffer,
            &wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            },
        )
    };

    let import = import(&cuda.ctx, handle, alloc_size, size);
    os::after_import(handle, import.is_ok());
    let (ext_mem, ptr) = import?;
    Ok(SharedBuffer {
        mapping: CudaMapping {
            ctx: cuda.ctx.clone(),
            ext_mem,
            ptr,
            size,
        },
        buffer,
    })
}

fn import(
    ctx: &Arc<CudaContext>,
    handle: os::Handle,
    alloc_size: u64,
    size: u64,
) -> Result<(sys::CUexternalMemory, sys::CUdeviceptr), String> {
    ctx.bind_to_thread()
        .map_err(|e| format!("bind CUDA context: {e:?}"))?;
    let desc = sys::CUDA_EXTERNAL_MEMORY_HANDLE_DESC {
        type_: os::CU_TYPE,
        handle: os::import_handle(handle),
        size: alloc_size,
        flags: sys::CUDA_EXTERNAL_MEMORY_DEDICATED,
        reserved: [0; 16],
    };
    let mut ext_mem = std::ptr::null_mut();
    // SAFETY: descriptor fully initialized above; out-pointer is valid.
    unsafe { sys::cuImportExternalMemory(&mut ext_mem, &desc) }
        .result()
        .map_err(|e| format!("cuImportExternalMemory: {e:?}"))?;

    let buf_desc = sys::CUDA_EXTERNAL_MEMORY_BUFFER_DESC {
        offset: 0,
        size,
        flags: 0,
        reserved: [0; 16],
    };
    let mut ptr = 0;
    // SAFETY: `ext_mem` was just imported; on failure it is destroyed here.
    if let Err(e) =
        unsafe { sys::cuExternalMemoryGetMappedBuffer(&mut ptr, ext_mem, &buf_desc) }.result()
    {
        // SAFETY: as above.
        unsafe {
            let _ = sys::cuDestroyExternalMemory(ext_mem).result();
        }
        return Err(format!("cuExternalMemoryGetMappedBuffer: {e:?}"));
    }
    Ok((ext_mem, ptr))
}

/// [`thinfer_core::backend::ExternalAllocator`] over [`alloc_shared`]: install
/// on a `WgpuBackend` so every engine buffer is CUDA-addressable.
pub struct CudaAllocator(Arc<CudaDevice>);

impl CudaAllocator {
    pub fn new(dev: Arc<CudaDevice>) -> Self {
        Self(dev)
    }
}

impl thinfer_core::backend::ExternalAllocator for CudaAllocator {
    fn allocate(
        &self,
        device: &wgpu::Device,
        bytes: u64,
    ) -> Result<thinfer_core::backend::ExternalBuffer, String> {
        let shared = alloc_shared(device, &self.0, bytes, "cuda-shared")?;
        Ok(thinfer_core::backend::ExternalBuffer {
            address: shared.mapping.ptr,
            buffer: shared.buffer,
            mapping: Box::new(shared.mapping),
        })
    }
}
