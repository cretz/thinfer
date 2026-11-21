//! Page-locked, device-mapped host memory ([`MappedHost`]): kernels read it in
//! place over the bus (zero-copy), the host fills it like a plain slice.

use cudarc::driver::sys;
use thinfer_core::backend::MappedHost;

use crate::device::CudaDevice;
use crate::nvrtc::drv;

pub struct MappedHostMem {
    ptr: *mut u8,
    len: usize,
    dev_ptr: u64,
    ctx: std::sync::Arc<cudarc::driver::CudaContext>,
}

// SAFETY: the allocation is owned exclusively by this value; the raw pointer is
// only dereferenced through `&mut self`.
unsafe impl Send for MappedHostMem {}
unsafe impl Sync for MappedHostMem {}

impl MappedHostMem {
    pub fn alloc(dev: &CudaDevice, bytes: u64) -> Result<Self, String> {
        dev.ctx
            .bind_to_thread()
            .map_err(|e| format!("bind: {e:?}"))?;
        let len = bytes as usize;
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        // SAFETY: plain driver allocation on the bound context.
        drv(
            unsafe {
                sys::cuMemHostAlloc(
                    &mut ptr,
                    len,
                    sys::CU_MEMHOSTALLOC_DEVICEMAP | sys::CU_MEMHOSTALLOC_PORTABLE,
                )
            },
            "cuMemHostAlloc",
        )?;
        let mut dev_ptr = 0;
        // SAFETY: `ptr` is a live DEVICEMAP allocation.
        let r = unsafe { sys::cuMemHostGetDevicePointer_v2(&mut dev_ptr, ptr, 0) };
        if let Err(e) = drv(r, "cuMemHostGetDevicePointer") {
            // SAFETY: freeing the allocation made above.
            unsafe { sys::cuMemFreeHost(ptr) };
            return Err(e);
        }
        Ok(Self {
            ptr: ptr.cast(),
            len,
            dev_ptr,
            ctx: dev.ctx.clone(),
        })
    }
}

impl MappedHost for MappedHostMem {
    fn bytes(&mut self) -> &mut [u8] {
        // SAFETY: `ptr` is a live allocation of `len` bytes owned by `self`.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    fn device_addr(&self) -> u64 {
        self.dev_ptr
    }
}

impl Drop for MappedHostMem {
    fn drop(&mut self) {
        let _ = self.ctx.bind_to_thread();
        // SAFETY: frees the allocation this value owns; no kernel may still
        // read it (owners sync before dropping the expert tier).
        unsafe { sys::cuMemFreeHost(self.ptr.cast()) };
    }
}
