//! Runtime kernel compilation: CUDA C source -> cubin for the device's compute
//! capability via NVRTC (no driver PTX JIT, so a newer toolkit than the driver
//! still loads under CUDA minor-version compatibility), then a loaded module.

use std::ffi::{CStr, CString};
use std::sync::Arc;

use cudarc::driver::{CudaContext, sys};
use cudarc::nvrtc::sys as nvrtc;

/// Stack for the NVRTC compile thread (see [`Module::compile`]).
const NVRTC_STACK: usize = 64 << 20;

fn nvrtc_check(r: nvrtc::nvrtcResult, what: &str) -> Result<(), String> {
    if r == nvrtc::nvrtcResult::NVRTC_SUCCESS {
        Ok(())
    } else {
        Err(format!("{what}: {r:?}"))
    }
}

pub(crate) fn drv(r: sys::CUresult, what: &str) -> Result<(), String> {
    r.result().map_err(|e| format!("{what}: {e:?}"))
}

fn compile(source: &str, name: &str, arch: &str) -> Result<Vec<u8>, String> {
    crate::libs::libs()
        .nvrtc
        .as_ref()
        .ok_or("NVRTC not found (install the CUDA 13 toolkit or bundle it in <exe>/cuda)")?;
    let include = crate::libs::include_dir().ok_or("CUDA include dir not found")?;
    let src = CString::new(source).unwrap();
    let name = CString::new(name).unwrap();
    let opts: Vec<CString> = [
        format!("--gpu-architecture={arch}"),
        "--std=c++17".to_string(),
        format!("--include-path={}", include.display()),
    ]
    .into_iter()
    .map(|o| CString::new(o).unwrap())
    .collect();
    let opt_ptrs: Vec<*const std::ffi::c_char> = opts.iter().map(|o| o.as_ptr()).collect();
    // SAFETY: standard NVRTC create/compile/get/destroy sequence; every buffer
    // is sized from the matching *Size call.
    unsafe {
        let mut prog = std::ptr::null_mut();
        nvrtc_check(
            nvrtc::nvrtcCreateProgram(
                &mut prog,
                src.as_ptr(),
                name.as_ptr(),
                0,
                std::ptr::null(),
                std::ptr::null(),
            ),
            "nvrtcCreateProgram",
        )?;
        let compiled = nvrtc::nvrtcCompileProgram(prog, opt_ptrs.len() as i32, opt_ptrs.as_ptr());
        let result = if compiled == nvrtc::nvrtcResult::NVRTC_SUCCESS {
            let mut size = 0usize;
            nvrtc_check(
                nvrtc::nvrtcGetCUBINSize(prog, &mut size),
                "nvrtcGetCUBINSize",
            )
            .and_then(|()| {
                let mut cubin = vec![0u8; size];
                nvrtc_check(
                    nvrtc::nvrtcGetCUBIN(prog, cubin.as_mut_ptr().cast()),
                    "nvrtcGetCUBIN",
                )
                .map(|()| cubin)
            })
        } else {
            let mut size = 0usize;
            let _ = nvrtc::nvrtcGetProgramLogSize(prog, &mut size);
            let mut log = vec![0u8; size.max(1)];
            let _ = nvrtc::nvrtcGetProgramLog(prog, log.as_mut_ptr().cast());
            let log = CStr::from_bytes_until_nul(&log)
                .map(|c| c.to_string_lossy().into_owned())
                .unwrap_or_default();
            Err(format!("nvrtc compile failed ({compiled:?}):\n{log}"))
        };
        let _ = nvrtc::nvrtcDestroyProgram(&mut prog);
        result
    }
}

/// A loaded module on one context. Kept for the process lifetime.
pub(crate) struct Module {
    pub ctx: Arc<CudaContext>,
    raw: sys::CUmodule,
}

// SAFETY: module/function handles are plain driver handles; every use binds
// the owning context first.
unsafe impl Send for Module {}
unsafe impl Sync for Module {}

impl Module {
    /// Compile `source` for `sm` and load it on `ctx`.
    pub fn compile(
        ctx: Arc<CudaContext>,
        source: &'static str,
        name: &'static str,
        sm: (i32, i32),
    ) -> Result<Self, String> {
        Self::compile_arch(ctx, source, name, format!("sm_{}{}", sm.0, sm.1))
    }

    /// [`Self::compile`] for an explicit `--gpu-architecture` (e.g. the
    /// arch-specific `sm_120a` that block-scaled MMA needs).
    pub fn compile_arch(
        ctx: Arc<CudaContext>,
        source: &'static str,
        name: &'static str,
        arch: String,
    ) -> Result<Self, String> {
        // NVRTC (LLVM-based) needs far more stack than a caller deep in an
        // async render may have left (the Windows main thread is 1 MiB), so
        // compile on a dedicated thread with a generous stack.
        let cubin = std::thread::Builder::new()
            .name(format!("nvrtc-{name}"))
            .stack_size(NVRTC_STACK)
            .spawn(move || compile(source, name, &arch))
            .map_err(|e| format!("spawn nvrtc thread: {e}"))?
            .join()
            .map_err(|_| "nvrtc thread panicked".to_string())??;
        ctx.bind_to_thread().map_err(|e| format!("bind: {e:?}"))?;
        let mut raw = std::ptr::null_mut();
        // SAFETY: `cubin` is a complete image for this device.
        drv(
            unsafe { sys::cuModuleLoadData(&mut raw, cubin.as_ptr().cast()) },
            "cuModuleLoadData",
        )?;
        Ok(Self { ctx, raw })
    }

    /// Look up a kernel and allow it `smem` bytes of dynamic shared memory.
    pub fn function(&self, name: &CStr, smem: u32) -> Result<sys::CUfunction, String> {
        let mut f = std::ptr::null_mut();
        // SAFETY: `self.raw` is a loaded module on the bound context.
        unsafe {
            drv(
                sys::cuModuleGetFunction(&mut f, self.raw, name.as_ptr()),
                "cuModuleGetFunction",
            )?;
            drv(
                sys::cuFuncSetAttribute(
                    f,
                    sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    smem as i32,
                ),
                "cuFuncSetAttribute(smem)",
            )?;
        }
        Ok(f)
    }

    /// Device address of a `__device__` global.
    pub fn global(&self, name: &CStr) -> Result<sys::CUdeviceptr, String> {
        let (mut ptr, mut size) = (0, 0);
        // SAFETY: `self.raw` is a loaded module on the bound context.
        drv(
            unsafe { sys::cuModuleGetGlobal_v2(&mut ptr, &mut size, self.raw, name.as_ptr()) },
            "cuModuleGetGlobal",
        )?;
        Ok(ptr)
    }

    pub fn bind(&self) -> Result<(), String> {
        self.ctx
            .bind_to_thread()
            .map_err(|e| format!("bind: {e:?}"))
    }

    /// Launch `f` on the default stream and wait for it.
    pub fn launch_sync(
        &self,
        f: sys::CUfunction,
        grid: (u32, u32, u32),
        threads: u32,
        smem: u32,
        args: &mut [*mut std::ffi::c_void],
        what: &str,
    ) -> Result<(), String> {
        self.launch(f, grid, threads, smem, args, what)?;
        // SAFETY: plain context-wide wait on the bound context.
        drv(unsafe { sys::cuCtxSynchronize() }, what)
    }

    /// Launch `f` on the default stream without waiting (default-stream
    /// launches run in order, so a following [`Self::launch_sync`] orders it).
    pub fn launch(
        &self,
        f: sys::CUfunction,
        grid: (u32, u32, u32),
        threads: u32,
        smem: u32,
        args: &mut [*mut std::ffi::c_void],
        what: &str,
    ) -> Result<(), String> {
        // SAFETY: the caller passes arguments matching `f`'s signature and a
        // grid/smem matching its assumptions.
        unsafe {
            drv(
                sys::cuLaunchKernel(
                    f,
                    grid.0,
                    grid.1,
                    grid.2,
                    threads,
                    1,
                    1,
                    smem,
                    std::ptr::null_mut(),
                    args.as_mut_ptr(),
                    std::ptr::null_mut(),
                ),
                what,
            )
        }
    }

    /// Copy `size_of::<T>()` bytes from device `src` into a `T`.
    pub fn read<T: Default + Copy>(&self, src: sys::CUdeviceptr) -> Result<T, String> {
        self.bind()?;
        let mut v = T::default();
        // SAFETY: reads exactly `size_of::<T>()` bytes into `v` (plain data).
        drv(
            unsafe { sys::cuMemcpyDtoH_v2((&raw mut v).cast(), src, std::mem::size_of::<T>()) },
            "cuMemcpyDtoH",
        )?;
        Ok(v)
    }

    /// Copy a `T` to device `dst`.
    pub fn write<T: Copy>(&self, dst: sys::CUdeviceptr, v: T) -> Result<(), String> {
        self.bind()?;
        // SAFETY: writes exactly `size_of::<T>()` bytes from `v` (plain data).
        drv(
            unsafe { sys::cuMemcpyHtoD_v2(dst, (&raw const v).cast(), std::mem::size_of::<T>()) },
            "cuMemcpyHtoD",
        )
    }
}
