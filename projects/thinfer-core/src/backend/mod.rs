use crate::mem::{MemAccount, VramCategory};
use crate::tensor::GpuBufferId;
use core::future::Future;
use std::sync::Arc;

mod poll;
pub mod wgpu;
pub use wgpu::{
    CommandEncoderState, PowerPreference, WgpuBackend, WgpuConfig, WgpuError, WgpuPipeline,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BindingKind {
    StorageRead,
    StorageReadWrite,
    Uniform,
}

#[derive(Clone, Copy, Debug)]
pub struct BindingLayout {
    pub slot: u32,
    pub kind: BindingKind,
}

#[derive(Clone, Copy, Debug)]
pub struct Binding {
    pub slot: u32,
    pub buffer: GpuBufferId,
    pub offset: u64,
    pub size: u64,
}

/// `(buffer, offset, byte_len)` triple — caller-built once at allocation or
/// sub-allocation, threaded through dispatch helpers so they build `Binding`s
/// without re-querying lengths. `offset != 0` is used by `Workspace` (and
/// future buffer-pool tiers) to slice multiple activations out of one backing
/// buffer; whole-buffer callers use `BufRef::new` and get `offset == 0`.
#[derive(Clone, Copy, Debug)]
pub struct BufRef {
    pub id: GpuBufferId,
    pub offset: u64,
    pub len: u64,
}

impl BufRef {
    pub fn new(id: GpuBufferId, len: u64) -> Self {
        Self { id, offset: 0, len }
    }
    pub fn view(id: GpuBufferId, offset: u64, len: u64) -> Self {
        Self { id, offset, len }
    }
    pub fn binding(&self, slot: u32) -> Binding {
        Binding {
            slot,
            buffer: self.id,
            offset: self.offset,
            size: self.len,
        }
    }
}

/// GPU-side weight preparation performed by `Backend::weight_prep` at upload
/// time (residency miss path). Shapes describe the raw bf16 source as
/// `[n, k]` row-major. See `ops::weight_prep` for the kernels.
#[derive(Clone, Copy, Debug)]
pub enum WeightPrep {
    /// bf16 -> GGUF Q8_0 block stream, same element order (`k % 32 == 0`,
    /// even total block count).
    Q8_0FromBf16 { n: u32, k: u32 },
    /// bf16 `[n, k]` -> `[k, n]` (nn.Linear upload transpose).
    TransposeBf16 { n: u32, k: u32 },
    /// f32 `[n, k]` row-major -> bf16 `[k, n]`: fused RNE narrow + nn.Linear
    /// upload transpose. Replaces the CPU `narrow_f32_to_bf16` +
    /// `transpose_bf16_cpu` pair on the f32 (safetensors) weight path, which
    /// dominates cold text-encode / denoise wall time (~1.4s per umT5 layer,
    /// single-threaded, while the GPU idles). Bit-exact vs `half::bf16::from_f32`.
    ///
    /// Banded: `n` is the global row count (the output `[k, n]` column stride);
    /// the source is just rows `[n0, n0 + band_n)` of `[n, k]`, so the f32
    /// staging buffer is bounded to one band instead of the whole tensor (a
    /// whole-tensor f32 spike busts the VRAM budget - 2x the bf16 footprint).
    /// `n` even and `n0`, `band_n` even keep output u32 writes (two rows each)
    /// from straddling adjacent threads / bands.
    NarrowTransposeF32 {
        n: u32,
        k: u32,
        n0: u32,
        band_n: u32,
    },
}

/// Compute backend abstraction. v1 carries a single bind group (group 0); the
/// concept generalizes when we hit a kernel that wants more.
pub trait Backend: 'static {
    type Error: core::fmt::Debug + Send + 'static;
    type CommandEncoder;
    type Pipeline;

    fn allocate(&self, bytes: u64) -> Result<GpuBufferId, Self::Error>;

    /// Construct the backend's "allocation refused by the budget" error for a
    /// `bytes`-sized request, WITHOUT touching the device. Used by strict-budget
    /// workspace allocs (the arbiter's reclaim chain ran dry) so a hard-ceiling
    /// caller fails at the budget boundary instead of overshooting into a real
    /// device OOM. Callers that recover (e.g. the VAE tiler) match the same
    /// variant they match for a device OOM.
    fn budget_oom_error(&self, bytes: u64) -> Self::Error;
    /// Categorized allocation. Default implementation ignores the category
    /// (test mocks). Real backends override to attribute the bytes to the
    /// right `MemAccount` counter so eviction policy and budget assertions
    /// see the correct picture.
    fn allocate_in(&self, bytes: u64, _cat: VramCategory) -> Result<GpuBufferId, Self::Error> {
        self.allocate(bytes)
    }
    fn free(&self, id: GpuBufferId);

    /// Shared memory accountant. Backends own one `MemAccount` for their
    /// lifetime; this hands out a clone of the `Arc` so residency, workspace,
    /// and budget assertions see the same counters. Test mocks hold a
    /// throwaway `MemAccount` in a field.
    fn mem_account(&self) -> &Arc<MemAccount>;

    /// Host bytes → GPU buffer. Test inputs and weight uploads land here; on
    /// web, weight bytes arrive in bounded chunks (residency streams through
    /// a fixed scratch; tensor-sized wasm allocations are banned). `dst_offset`
    /// and `src.len()` must be 4-byte aligned (wgpu `COPY_BUFFER_ALIGNMENT`).
    fn write_buffer(
        &self,
        dst: GpuBufferId,
        dst_offset: u64,
        src: &[u8],
    ) -> Result<(), Self::Error>;

    fn create_command_encoder(&self) -> Self::CommandEncoder;

    fn dispatch(
        &self,
        encoder: &mut Self::CommandEncoder,
        pipeline: &Self::Pipeline,
        bindings: &[Binding],
        workgroups: [u32; 3],
    ) -> Result<(), Self::Error>;

    fn copy_buffer_to_buffer(
        &self,
        encoder: &mut Self::CommandEncoder,
        src: GpuBufferId,
        src_offset: u64,
        dst: GpuBufferId,
        dst_offset: u64,
        len: u64,
    ) -> Result<(), Self::Error>;

    fn submit(
        &self,
        encoder: Self::CommandEncoder,
    ) -> impl Future<Output = Result<(), Self::Error>>;

    /// `label` names the pipeline for telemetry (compile/dispatch events,
    /// rollup tables) and backend debug labels; `entry` stays the WGSL entry
    /// point (almost always "main").
    fn create_pipeline(
        &self,
        label: &str,
        wgsl: &str,
        entry: &str,
        layout: &[BindingLayout],
    ) -> impl Future<Output = Result<Self::Pipeline, Self::Error>>;

    fn read_buffer(
        &self,
        src: GpuBufferId,
        offset: u64,
        len: u64,
    ) -> impl Future<Output = Result<Vec<u8>, Self::Error>>;

    /// Whether `weight_prep` implements `op` on the GPU. Residency checks
    /// this before reading any bytes: a supported op streams the raw source
    /// straight into a staging buffer (bounded host scratch), an unsupported
    /// one (test mocks) takes the CPU transform path.
    fn supports_weight_prep(&self, _op: WeightPrep) -> bool {
        false
    }

    /// Run the `op` prep kernel over `staging` (raw on-disk bytes, uploaded
    /// by the caller, who owns its alloc/free) into `dst`. Only called when
    /// `supports_weight_prep(op)`.
    fn weight_prep(
        &self,
        _op: WeightPrep,
        _staging: &BufRef,
        _dst: &BufRef,
    ) -> impl Future<Output = Result<(), Self::Error>> {
        async { unreachable!("weight_prep called without supports_weight_prep") }
    }
}
