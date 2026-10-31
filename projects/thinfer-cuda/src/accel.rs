//! The [`thinfer_core::backend::Accelerator`] for NVIDIA: hot ops as CUDA
//! kernels on the engine's shared VRAM. Kernels compile lazily on first use;
//! a compile failure (no NVRTC, unsupported toolkit) is logged once and the op
//! declines from then on, leaving the wgpu path.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

use thinfer_core::backend::{
    Accelerator, ElementwiseOp, GemmAddrs, GemmNvfp4Addrs, SdpaAddrs, TileAttnAddrs, TileAttnShape,
    VsaAddrs, VsaPart, VsaScratch, VsaShape,
};
use thinfer_core::ops::GemmDesc;

use crate::device::CudaDevice;
use crate::elementwise::ElementwiseKernels;
use crate::gemm::{GemmKernels, MIN_M, index};
use crate::gemm_fp4::GemmFp4Kernels;
use crate::sdpa::{SdpaKernels, UNIFORM_LEN};
use crate::vsa::VsaKernels;

pub struct CudaAccel {
    dev: Arc<CudaDevice>,
    sdpa: OnceLock<Option<SdpaKernels>>,
    gemm: OnceLock<Option<GemmKernels>>,
    gemm_fp4: OnceLock<Option<GemmFp4Kernels>>,
    vsa: OnceLock<Option<VsaKernels>>,
    elementwise: OnceLock<Option<ElementwiseKernels>>,
    /// Logged once per op kind, on the first call that actually runs on CUDA.
    engaged: AtomicBool,
    gemm_engaged: AtomicBool,
    fp4_engaged: AtomicBool,
    vsa_engaged: AtomicBool,
}

impl CudaAccel {
    pub fn new(dev: Arc<CudaDevice>) -> Self {
        Self {
            dev,
            sdpa: OnceLock::new(),
            gemm: OnceLock::new(),
            gemm_fp4: OnceLock::new(),
            vsa: OnceLock::new(),
            elementwise: OnceLock::new(),
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
    fn accepts_sdpa_f16(&self, uniform_len: u64) -> bool {
        // The plain `sdpa_sg` uniform only (the windowed one is longer).
        uniform_len == UNIFORM_LEN && self.sdpa().is_some()
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

    fn accepts_gemm(&self, desc: &GemmDesc, m: u32, n: u32) -> bool {
        m >= MIN_M && n.is_multiple_of(8) && index(desc).is_some() && self.gemm_kernels().is_some()
    }

    fn gemm(&self, desc: &GemmDesc, a: &GemmAddrs) -> Result<bool, String> {
        let Some(k) = self.gemm_kernels() else {
            return Ok(false);
        };
        let ran = k.run(desc, a.a, a.b, a.out, a.dims, a.m, a.n, a.k, a.epilogue)?;
        if ran && !self.gemm_engaged.swap(true, Ordering::Relaxed) {
            tracing::info!(
                target: thinfer_core::trace::ADAPTER,
                m = a.m, n = a.n, ?desc,
                "cuda gemm engaged",
            );
        }
        Ok(ran)
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

    fn sync(&self) -> Result<(), String> {
        self.dev
            .ctx
            .bind_to_thread()
            .map_err(|e| format!("bind: {e:?}"))?;
        // SAFETY: plain context-wide wait on the bound context.
        crate::nvrtc::drv(unsafe { cudarc::driver::sys::cuCtxSynchronize() }, "sync")
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
