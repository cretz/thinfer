//! The [`thinfer_core::backend::Accelerator`] for NVIDIA: hot ops as CUDA
//! kernels on the engine's shared VRAM. Kernels compile lazily on first use;
//! a compile failure (no NVRTC, unsupported toolkit) is logged once and the op
//! declines from then on, leaving the wgpu path.

use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

use cudarc::driver::sys;
use thinfer_core::backend::{
    AccelFence, Accelerator, Conv2dAddrs, Conv3dAddrs, ElementwiseOp, GemmAddrs, GemmNvfp4Addrs,
    SdpaAddrs, TileAttnAddrs, TileAttnShape, VsaAddrs, VsaPart, VsaScratch, VsaShape,
};
use thinfer_core::ops::GemmDesc;

use crate::device::CudaDevice;
use crate::elementwise::ElementwiseKernels;
use crate::gemm::{self, GemmKernels};
use crate::gemm_fp4::GemmFp4Kernels;
use crate::llm::LlmKernels;
use crate::sdpa::{SdpaKernels, SdpaUniform, UNIFORM_LEN};
use crate::vsa::VsaKernels;
use crate::window::WindowKernels;

pub struct CudaAccel {
    dev: Arc<CudaDevice>,
    sdpa: OnceLock<Option<SdpaKernels>>,
    gemm: OnceLock<Option<GemmKernels>>,
    gemm_fp4: OnceLock<Option<GemmFp4Kernels>>,
    vsa: OnceLock<Option<VsaKernels>>,
    window: OnceLock<Option<WindowKernels>>,
    elementwise: OnceLock<Option<ElementwiseKernels>>,
    llm: OnceLock<Option<LlmKernels>>,
    /// Side copy stream of [`Accelerator::copy_ahead`] (created on first use).
    ahead: OnceLock<Result<AheadStream, String>>,
    /// Logged once per op kind, on the first call that actually runs on CUDA.
    engaged: AtomicBool,
    gemm_engaged: AtomicBool,
    fp4_engaged: AtomicBool,
    vsa_engaged: AtomicBool,
}

/// A non-blocking stream (it does not serialize with the legacy default
/// stream the engine enqueues on) and the event that joins it back.
struct AheadStream {
    stream: sys::CUstream,
    event: sys::CUevent,
}

// SAFETY: plain driver handles, valid for the context's lifetime; every use
// binds the context first.
unsafe impl Send for AheadStream {}
unsafe impl Sync for AheadStream {}

impl CudaAccel {
    /// The side copy stream, created on first use (context bound).
    fn ahead(&self) -> Result<&AheadStream, String> {
        self.dev
            .ctx
            .bind_to_thread()
            .map_err(|e| format!("bind: {e:?}"))?;
        self.ahead
            .get_or_init(|| {
                let mut stream = std::ptr::null_mut();
                let mut event = std::ptr::null_mut();
                // SAFETY: plain creation calls on the bound context.
                unsafe {
                    crate::nvrtc::drv(
                        sys::cuStreamCreate(
                            &mut stream,
                            sys::CUstream_flags::CU_STREAM_NON_BLOCKING as u32,
                        ),
                        "cuStreamCreate",
                    )?;
                    crate::nvrtc::drv(
                        sys::cuEventCreate(
                            &mut event,
                            sys::CUevent_flags::CU_EVENT_DISABLE_TIMING as u32,
                        ),
                        "cuEventCreate",
                    )?;
                }
                Ok(AheadStream { stream, event })
            })
            .as_ref()
            .map_err(Clone::clone)
    }

    pub fn new(dev: Arc<CudaDevice>) -> Self {
        Self {
            dev,
            sdpa: OnceLock::new(),
            gemm: OnceLock::new(),
            gemm_fp4: OnceLock::new(),
            vsa: OnceLock::new(),
            window: OnceLock::new(),
            elementwise: OnceLock::new(),
            llm: OnceLock::new(),
            ahead: OnceLock::new(),
            engaged: AtomicBool::new(false),
            gemm_engaged: AtomicBool::new(false),
            fp4_engaged: AtomicBool::new(false),
            vsa_engaged: AtomicBool::new(false),
        }
    }

    fn gemm_kernels(&self) -> Option<&GemmKernels> {
        self.gemm
            .get_or_init(|| match GemmKernels::new(self.dev.ctx.clone(), self.dev.sm) {
                Ok(k) => Some(k),
                Err(e) => {
                    tracing::warn!(target: thinfer_core::trace::ADAPTER, error = %e, "cuda gemm unavailable; wgpu path");
                    None
                }
            })
            .as_ref()
    }

    fn gemm_fp4_kernels(&self) -> Option<&GemmFp4Kernels> {
        self.gemm_fp4
            .get_or_init(|| match GemmFp4Kernels::new(self.dev.ctx.clone(), self.dev.sm) {
                Ok(k) => Some(k),
                Err(e) => {
                    tracing::warn!(target: thinfer_core::trace::ADAPTER, error = %e, "cuda nvfp4 gemm unavailable; dequant path");
                    None
                }
            })
            .as_ref()
    }

    fn vsa_kernels(&self) -> Option<&VsaKernels> {
        self.vsa
            .get_or_init(|| match VsaKernels::new(self.dev.ctx.clone(), self.dev.sm) {
                Ok(k) => Some(k),
                Err(e) => {
                    tracing::warn!(target: thinfer_core::trace::ADAPTER, error = %e, "cuda vsa unavailable; dense attention");
                    None
                }
            })
            .as_ref()
    }

    /// Log the first VSA launch (once per accelerator).
    fn note_vsa(&self, ran: bool, s: &VsaShape, a: &VsaAddrs) {
        if ran && !self.vsa_engaged.swap(true, Ordering::Relaxed) {
            tracing::info!(
                target: thinfer_core::trace::ADAPTER,
                rows = s.rows, tiles = s.n_tiles, prefix = s.n_prefix, topk = s.topk, gate = a.gate != 0,
                "cuda vsa engaged",
            );
        }
    }

    fn elementwise_kernels(&self) -> Option<&ElementwiseKernels> {
        self.elementwise
            .get_or_init(|| match ElementwiseKernels::new(self.dev.ctx.clone(), self.dev.sm) {
                Ok(k) => Some(k),
                Err(e) => {
                    tracing::warn!(target: thinfer_core::trace::ADAPTER, error = %e, "cuda elementwise unavailable; wgpu path");
                    None
                }
            })
            .as_ref()
    }

    fn llm_kernels(&self) -> Option<&LlmKernels> {
        self.llm
            .get_or_init(|| match LlmKernels::new(self.dev.ctx.clone(), self.dev.sm) {
                Ok(k) => {
                    tracing::info!(target: thinfer_core::trace::ADAPTER, "cuda llm kernels ready");
                    Some(k)
                }
                Err(e) => {
                    tracing::warn!(target: thinfer_core::trace::ADAPTER, error = %e, "cuda llm unavailable; wgpu path");
                    None
                }
            })
            .as_ref()
    }

    fn window(&self) -> Option<&WindowKernels> {
        self.window
            .get_or_init(|| match WindowKernels::new(self.dev.ctx.clone(), self.dev.sm) {
                Ok(k) => Some(k),
                Err(e) => {
                    tracing::warn!(target: thinfer_core::trace::ADAPTER, error = %e, "cuda window attention unavailable; wgpu path");
                    None
                }
            })
            .as_ref()
    }

    fn sdpa(&self) -> Option<&SdpaKernels> {
        self.sdpa
            .get_or_init(|| match SdpaKernels::new(self.dev.ctx.clone(), self.dev.sm) {
                Ok(k) => Some(k),
                Err(e) => {
                    tracing::warn!(target: thinfer_core::trace::ADAPTER, error = %e, "cuda sdpa unavailable; wgpu path");
                    None
                }
            })
            .as_ref()
    }
}

impl Accelerator for CudaAccel {
    fn accepts_sdpa_f16(&self, uniform_len: u64, head_dim: u32) -> bool {
        // The plain `sdpa_sg` uniform only (the windowed one is longer), at the
        // head dims the kernel is built for.
        uniform_len == UNIFORM_LEN && matches!(head_dim, 64 | 128) && self.sdpa().is_some()
    }

    fn sdpa_f16(&self, a: &SdpaAddrs) -> Result<bool, String> {
        let Some(k) = self.sdpa() else {
            return Ok(false);
        };
        let u = k.read_uniform(a.uniform)?;
        let ran = k.run(&u, a.q, a.k, a.v, a.mask, a.out, a.uniform)?;
        if ran && !self.engaged.swap(true, Ordering::Relaxed) {
            tracing::info!(
                target: thinfer_core::trace::ADAPTER,
                s_q = u.s_q, s_k = u.s_k, h_q = u.h_q, d = u.d,
                "cuda sdpa engaged",
            );
        }
        Ok(ran)
    }

    fn accepts_dense_attention(&self, head_dim: u32) -> bool {
        matches!(head_dim, 64 | 128) && self.sdpa().is_some()
    }

    fn dense_attention_f16(
        &self,
        a: &thinfer_core::backend::DenseAttention,
    ) -> Result<bool, String> {
        let Some(k) = self.sdpa() else {
            return Ok(false);
        };
        let u = SdpaUniform {
            b: 1,
            h_q: a.heads,
            h_kv: a.kv_heads,
            s_q: a.s_q,
            s_k: a.s_k,
            d: a.head_dim,
            scale: a.scale,
            has_mask: 0,
        };
        k.enqueue(&u, a.q, a.k, a.v, a.out)
    }

    fn window_attn_f16(
        &self,
        u: &thinfer_core::ops::window_attn::WindowAttnUniform,
        a: &thinfer_core::backend::WindowAttnAddrs,
    ) -> Result<bool, String> {
        if !WindowKernels::accepts(u) {
            return Ok(false);
        }
        let Some(k) = self.window() else {
            return Ok(false);
        };
        k.run(u, a.q, a.k, a.v, a.out)
    }

    fn gemm_q8(
        &self,
        a: &thinfer_core::backend::GemmQ8Addrs,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<bool, String> {
        match self.gemm_kernels() {
            Some(g) => g.run_q8(a, m, n, k),
            None => Ok(false),
        }
    }

    fn quant_rows_q8(
        &self,
        format: thinfer_core::backend::Q8,
        x: u64,
        q: u64,
        scale: u64,
        rows: u32,
        k: u32,
    ) -> Result<bool, String> {
        match self.gemm_kernels() {
            Some(g) => g.quant_rows_q8(format, x, q, scale, rows, k),
            None => Ok(false),
        }
    }

    fn accepts_gemm(&self, desc: &GemmDesc, m: u32, n: u32) -> bool {
        gemm::accepts(desc, m, n) && self.gemm_kernels().is_some()
    }

    fn gemm(&self, desc: &GemmDesc, a: &GemmAddrs) -> Result<bool, String> {
        let Some(k) = self.gemm_kernels() else {
            return Ok(false);
        };
        let ran = k.run(desc, a)?;
        if ran && !self.gemm_engaged.swap(true, Ordering::Relaxed) {
            tracing::info!(
                target: thinfer_core::trace::ADAPTER,
                m = a.m, n = a.n, ?desc,
                "cuda gemm engaged",
            );
        }
        Ok(ran)
    }

    fn conv2d(&self, a: &Conv2dAddrs) -> Result<bool, String> {
        match self.gemm_kernels() {
            Some(k) => k.run_conv2d(a),
            None => Ok(false),
        }
    }

    fn accepts_conv2d(&self) -> bool {
        self.gemm_kernels().is_some()
    }

    fn conv3d_f16(&self, a: &Conv3dAddrs) -> Result<bool, String> {
        match self.gemm_kernels() {
            Some(k) => k.run_conv3d_f16(a),
            None => Ok(false),
        }
    }

    fn accepts_gemm_nvfp4(&self, m: u32, n: u32, k: u32) -> Option<u64> {
        (crate::gemm_fp4::accepts(m, n, k) && self.gemm_fp4_kernels().is_some())
            .then(|| crate::gemm_fp4::scratch_bytes(m, k))
    }

    fn gemm_nvfp4(&self, a: &GemmNvfp4Addrs) -> Result<bool, String> {
        let Some(k) = self.gemm_fp4_kernels() else {
            return Ok(false);
        };
        let ran = k.run(a)?;
        if ran && !self.fp4_engaged.swap(true, Ordering::Relaxed) {
            tracing::info!(
                target: thinfer_core::trace::ADAPTER,
                m = a.m, n = a.n, k = a.k,
                "cuda nvfp4 gemm engaged",
            );
        }
        Ok(ran)
    }

    fn accepts_vsa(&self, s: &VsaShape) -> Option<VsaScratch> {
        (crate::vsa::accepts(s) && self.vsa_kernels().is_some()).then(|| VsaScratch {
            scratch: crate::vsa::scratch_bytes(s),
            qk_i8: crate::vsa::qk_i8_bytes(s),
        })
    }

    fn vsa(&self, s: &VsaShape, a: &VsaAddrs) -> Result<bool, String> {
        let Some(k) = self.vsa_kernels() else {
            return Ok(false);
        };
        let ran = k.run(s, a)?;
        self.note_vsa(ran, s, a);
        Ok(ran)
    }

    fn vsa_part(&self, s: &VsaShape, part: VsaPart, a: &VsaAddrs) -> Result<bool, String> {
        let Some(k) = self.vsa_kernels() else {
            return Ok(false);
        };
        let ran = k.part(s, part, a)?;
        self.note_vsa(ran, s, a);
        Ok(ran)
    }

    fn accepts_elementwise(&self) -> bool {
        self.elementwise_kernels().is_some()
    }

    fn elementwise(&self, op: &ElementwiseOp) -> Result<bool, String> {
        match self.elementwise_kernels() {
            Some(k) => k.launch(op),
            None => Ok(false),
        }
    }

    fn accepts_llm(&self) -> bool {
        self.llm_kernels().is_some()
    }

    fn llm(&self, op: &thinfer_core::ops::llm::LlmOp) -> Result<bool, String> {
        match self.llm_kernels() {
            Some(k) => k.run(op),
            None => Ok(false),
        }
    }

    fn host_alloc(
        &self,
        bytes: u64,
    ) -> Result<Option<Box<dyn thinfer_core::backend::MappedHost>>, String> {
        crate::host::MappedHostMem::alloc(&self.dev, bytes)
            .map(|m| Some(Box::new(m) as Box<dyn thinfer_core::backend::MappedHost>))
    }

    fn copy(&self, dst: u64, src: u64, len: u64) -> Result<bool, String> {
        self.dev
            .ctx
            .bind_to_thread()
            .map_err(|e| format!("bind: {e:?}"))?;
        // SAFETY: both addresses are live device / mapped-host allocations of
        // at least `len` bytes (caller contract); default-stream ordered.
        crate::nvrtc::drv(
            unsafe { sys::cuMemcpyAsync(dst, src, len as usize, std::ptr::null_mut()) },
            "cuMemcpyAsync",
        )?;
        Ok(true)
    }

    fn copy_ahead(&self, dst: u64, src: u64, len: u64) -> Result<bool, String> {
        let a = self.ahead()?;
        // SAFETY: both addresses are live device / mapped-host allocations of
        // at least `len` bytes (caller contract); ordered on the side stream.
        crate::nvrtc::drv(
            unsafe { sys::cuMemcpyAsync(dst, src, len as usize, a.stream) },
            "cuMemcpyAsync (ahead)",
        )?;
        Ok(true)
    }

    fn join_ahead(&self) -> Result<(), String> {
        let Some(Ok(a)) = self.ahead.get() else {
            return Ok(());
        };
        self.dev
            .ctx
            .bind_to_thread()
            .map_err(|e| format!("bind: {e:?}"))?;
        // SAFETY: the event and stream live as long as `self`; the legacy
        // default stream waits for the side stream's work so far.
        unsafe {
            crate::nvrtc::drv(sys::cuEventRecord(a.event, a.stream), "cuEventRecord")?;
            crate::nvrtc::drv(
                sys::cuStreamWaitEvent(std::ptr::null_mut(), a.event, 0),
                "cuStreamWaitEvent",
            )?;
        }
        Ok(())
    }

    fn write(&self, dst: u64, bytes: &[u8]) -> Result<bool, String> {
        self.dev
            .ctx
            .bind_to_thread()
            .map_err(|e| format!("bind: {e:?}"))?;
        // SAFETY: `dst` holds at least `bytes.len()` bytes; a pageable source is
        // staged before the call returns, so `bytes` may be freed after.
        crate::nvrtc::drv(
            unsafe {
                sys::cuMemcpyHtoDAsync_v2(
                    dst,
                    bytes.as_ptr().cast(),
                    bytes.len(),
                    std::ptr::null_mut(),
                )
            },
            "cuMemcpyHtoDAsync",
        )?;
        Ok(true)
    }

    fn read(&self, src: u64, dst: &mut [u8]) -> Result<bool, String> {
        self.sync()?;
        // SAFETY: `src` holds at least `dst.len()` bytes; the stream is idle.
        crate::nvrtc::drv(
            unsafe { sys::cuMemcpyDtoH_v2(dst.as_mut_ptr().cast(), src, dst.len()) },
            "cuMemcpyDtoH",
        )?;
        Ok(true)
    }

    fn sync(&self) -> Result<(), String> {
        self.dev
            .ctx
            .bind_to_thread()
            .map_err(|e| format!("bind: {e:?}"))?;
        // SAFETY: plain context-wide wait on the bound context.
        crate::nvrtc::drv(unsafe { cudarc::driver::sys::cuCtxSynchronize() }, "sync")?;
        if let Some(Some(k)) = self.gemm.get()
            && k.take_unchecked_overflow()?
        {
            return Err("an unchecked f16 GEMM overflowed narrowing its bf16 weight".into());
        }
        Ok(())
    }

    fn fence(&self) -> Result<Option<AccelFence>, String> {
        self.dev
            .ctx
            .bind_to_thread()
            .map_err(|e| format!("bind: {e:?}"))?;
        let probe = match self.gemm.get() {
            Some(Some(k)) => k.probe_unchecked_overflow()?,
            _ => None,
        };
        let (tx, rx) = futures_channel::oneshot::channel::<sys::CUresult>();
        let user = Box::into_raw(Box::new(tx));
        // SAFETY: the callback runs once, after everything enqueued on the
        // default stream so far, and takes back the box (freed here instead
        // when the enqueue fails).
        let r = unsafe {
            sys::cuStreamAddCallback(std::ptr::null_mut(), Some(fence_done), user.cast(), 0)
        };
        if let Err(e) = crate::nvrtc::drv(r, "cuStreamAddCallback") {
            // SAFETY: the driver did not take the callback.
            drop(unsafe { Box::from_raw(user) });
            return Err(e);
        }
        Ok(Some(Box::pin(async move {
            let status = rx
                .await
                .map_err(|_| "accelerator fence dropped".to_string())?;
            crate::nvrtc::drv(status, "accelerator fence")?;
            if probe.is_some_and(|p| p.take()) {
                return Err("an unchecked f16 GEMM overflowed narrowing its bf16 weight".into());
            }
            Ok(())
        })))
    }

    fn accepts_tile_attention(&self, head_dim: u32) -> bool {
        crate::vsa::attention_head_dim_supported(head_dim) && self.vsa_kernels().is_some()
    }

    fn tile_attention(&self, s: &TileAttnShape, a: &TileAttnAddrs) -> Result<bool, String> {
        match self.vsa_kernels() {
            Some(k) => k.run_lists(s, a),
            None => Ok(false),
        }
    }
}

/// [`CudaAccel::fence`]'s stream callback: report the stream status.
unsafe extern "C" fn fence_done(_: sys::CUstream, status: sys::CUresult, user: *mut c_void) {
    // SAFETY: `user` is the box `fence` leaked for exactly this one call.
    let tx =
        unsafe { Box::from_raw(user.cast::<futures_channel::oneshot::Sender<sys::CUresult>>()) };
    let _ = tx.send(status);
}
