use crate::mem::{MemAccount, VramCategory};
use crate::tensor::GpuBufferId;
use core::future::Future;
use std::sync::Arc;

mod poll;
pub mod wgpu;
pub use wgpu::{
    CommandEncoderState, ExternalAllocator, ExternalBuffer, PowerPreference, WgpuBackend,
    WgpuConfig, WgpuError, WgpuPipeline,
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
    /// f16 `[n, k]` -> bf16 `[k, n]`: exact f16 decode + RNE narrow fused with
    /// the nn.Linear transpose. Replaces the CPU decode + transpose pair on f16
    /// sources (the MiniMax-H3 video VAE: ~0.8s of host time per decoder block,
    /// ~27s a decode, while the GPU idled). Same geometry as `TransposeBf16`.
    TransposeF16 { n: u32, k: u32 },
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
    /// Staging memory handed out by [`Self::stage_write`].
    type StagedWrite: StagedWrite;

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

    /// [`Self::write_buffer`] filled in place: stage `len` bytes for
    /// `dst[dst_offset..]` and hand back the staging memory. The caller writes
    /// it through [`StagedWrite::bytes`] (one copy, instead of filling a host
    /// buffer that `write_buffer` copies again); dropping the guard schedules
    /// the transfer exactly as `write_buffer` would. Same alignment rules.
    fn stage_write(
        &self,
        dst: GpuBufferId,
        dst_offset: u64,
        len: u64,
    ) -> Result<Self::StagedWrite, Self::Error>;

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

    /// Submit `encoder`'s accumulated commands as a standalone command buffer
    /// WITHOUT awaiting GPU completion, error-scope capture, or timestamp
    /// resolution. Used to break one logical scope into several command buffers
    /// so no single submit's GPU time exceeds the OS GPU watchdog (~2s Windows
    /// TDR) — e.g. a video DiT's multi-second self-attention. The caller
    /// installs a fresh encoder and keeps the scope's buffer guards alive, so
    /// the flushed work reads valid resources; any device-loss surfaces at the
    /// next awaited [`Self::submit`].
    fn flush_encoder(&self, encoder: Self::CommandEncoder) -> Result<(), Self::Error>;

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

    /// Optional out-of-band executor for hot ops (see [`Accelerator`]). `None`
    /// (the default, and always on web) means every op runs on this backend.
    fn accelerator(&self) -> Option<&dyn Accelerator> {
        None
    }

    /// The accelerator's address of `buf` (external-memory buffers only).
    fn device_address(&self, _buf: &BufRef) -> Option<u64> {
        None
    }

    /// Block until all submitted GPU work has completed. Orders backend work
    /// before an [`Accelerator`] op that reads its results.
    fn wait_idle(&self) {}

    /// The dense-matmul contract `p` was compiled with (from its WGSL), if it
    /// is one an [`Accelerator`] may run in its place.
    fn pipeline_gemm(&self, _p: &Self::Pipeline) -> Option<crate::ops::GemmDesc> {
        None
    }
}

/// Writable staging memory for one [`Backend::stage_write`]; dropping it
/// schedules the transfer.
pub trait StagedWrite {
    fn bytes(&mut self) -> &mut [u8];
}

/// A host buffer stands in where the backend has no mapped staging path.
impl StagedWrite for Vec<u8> {
    fn bytes(&mut self) -> &mut [u8] {
        self
    }
}

/// An optional out-of-band executor (native CUDA) for hot ops on the same VRAM
/// the backend owns, addressed by [`Backend::device_address`]. Purely an
/// optimization: every method may decline (`Ok(false)`), and the caller then
/// runs the backend's own op unchanged. Implementations run synchronously: the
/// result is complete in VRAM when the call returns.
pub trait Accelerator: Send + Sync {
    /// Cheap host-side precheck for [`Self::sdpa_f16`], before the caller pays
    /// to flush and drain the GPU. `uniform_len` distinguishes the plain `sdpa_sg`
    /// uniform from the windowed one.
    fn accepts_sdpa_f16(&self, uniform_len: u64) -> bool;

    /// Fused f16 attention with exactly the `sdpa_sg` contract (BSHD f16 Q/K/V/
    /// out, GQA, mask modes 0/1/2, f32 softmax; shapes read from the uniform).
    /// Inputs must be complete in VRAM (the caller drained the GPU).
    fn sdpa_f16(&self, a: &SdpaAddrs) -> Result<bool, String>;

    /// Cheap host-side precheck for [`Self::gemm`] (before the flush + drain).
    fn accepts_gemm(&self, desc: &crate::ops::GemmDesc, m: u32, n: u32) -> bool;

    /// Dense 16-bit matmul with exactly the `desc` contract (see
    /// [`crate::ops::GemmDesc`]); `k` is read from the dims uniform. Inputs
    /// must be complete in VRAM (the caller drained the GPU).
    fn gemm(&self, desc: &crate::ops::GemmDesc, a: &GemmAddrs) -> Result<bool, String>;

    /// Cheap host-side precheck for [`Self::gemm_nvfp4`]; also sizes the
    /// caller's scratch (`Some(bytes)` = accepted).
    fn accepts_gemm_nvfp4(&self, _m: u32, _n: u32, _k: u32) -> Option<u64> {
        None
    }

    /// W4A4 matmul against an NVFP4 composite weight (`crate::ops::nvfp4`):
    /// bf16 A `[M,K]`, bf16 out `[M,N]`, activations quantized with the
    /// composite's static input scale (the checkpoint's own numerics). Inputs
    /// must be complete in VRAM (the caller drained the GPU).
    fn gemm_nvfp4(&self, _a: &GemmNvfp4Addrs) -> Result<bool, String> {
        Ok(false)
    }

    /// Cheap host-side precheck for [`Self::vsa`]; `Some(scratch sizes)` =
    /// accepted.
    fn accepts_vsa(&self, _s: &VsaShape) -> Option<VsaScratch> {
        None
    }

    /// Tile-sparse attention with the gated compression branch (MiniMax-H3
    /// VSA) over bf16 `[rows, H, D]` Q/K/V/gate/out; see [`VsaShape`]. Inputs
    /// must be complete in VRAM (the caller drained the GPU).
    fn vsa(&self, _s: &VsaShape, _a: &VsaAddrs) -> Result<bool, String> {
        Ok(false)
    }

    /// One part of [`Self::vsa`] (see [`VsaPart`]). Enqueued only.
    fn vsa_part(&self, _s: &VsaShape, _part: VsaPart, _a: &VsaAddrs) -> Result<bool, String> {
        Ok(false)
    }

    /// Cheap host-side precheck for [`Self::tile_attention`].
    fn accepts_tile_attention(&self, _head_dim: u32) -> bool {
        false
    }

    /// Block-sparse attention over 64-row tiles with caller-built key-tile
    /// lists (e.g. block-diagonal: batched attention within each item), bf16
    /// `[rows, H, D]` q/k/v/out. Enqueued only, like [`Self::vsa`].
    fn tile_attention(&self, _s: &TileAttnShape, _a: &TileAttnAddrs) -> Result<bool, String> {
        Ok(false)
    }

    /// Whether [`Self::elementwise`] is available at all (kernels compiled).
    fn accepts_elementwise(&self) -> bool {
        false
    }

    /// Enqueue one [`ElementwiseOp`]; complete after [`Self::sync`].
    fn elementwise(&self, _op: &ElementwiseOp) -> Result<bool, String> {
        Ok(false)
    }

    /// Wait for every op this accelerator has enqueued. [`Self::gemm_nvfp4`],
    /// [`Self::vsa`] and [`Self::elementwise`] only ENQUEUE (in order, on one
    /// stream); the caller syncs before anything else reads their outputs or
    /// recycles their buffers.
    fn sync(&self) -> Result<(), String> {
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TileAttnShape {
    pub rows: u32,
    pub heads: u32,
    pub head_dim: u32,
    pub n_tiles: u32,
    pub scale: f32,
}

/// `geometry` as for [`VsaAddrs`]; `lists` = per (head, query tile) `n_tiles`
/// u32 key-tile ids, then `heads * n_tiles` u32 counts.
#[derive(Clone, Copy, Debug)]
pub struct TileAttnAddrs {
    pub q: u64,
    pub k: u64,
    pub v: u64,
    pub out: u64,
    pub geometry: u64,
    pub lists: u64,
}

/// Row segments for the per-segment ops in [`ElementwiseOp`]: rows
/// `[end[i-1], end[i])` read the i-th `a` / `b` vectors. A reference-image
/// request splits its text span by tag (labels vs vision blocks) and adds a
/// pinned-timestep segment per block, so this leaves room for several.
/// (The CUDA kernels' `MAX_SEG` must match.)
pub const MAX_SEGMENTS: usize = 16;

#[derive(Clone, Copy, Debug, Default)]
pub struct RowSegments {
    pub n: u32,
    pub end: [u32; MAX_SEGMENTS],
    pub a: [u64; MAX_SEGMENTS],
    pub b: [u64; MAX_SEGMENTS],
}

/// The non-matmul ops of a MiniMax-H3 block, on bf16 activations (device
/// addresses), each matching the wgpu op it replaces including its bf16
/// rounding points.
#[derive(Clone, Copy, Debug)]
pub enum ElementwiseOp {
    /// `out = bf16(bf16(rmsnorm(x) * w) * (1 + a[seg]) + b[seg])` over
    /// `[rows, dim]`: RMSNorm then AdaLN modulation (`a` = scale, `b` = shift).
    NormModulate {
        x: u64,
        w: u64,
        out: u64,
        rows: u32,
        dim: u32,
        eps: f32,
        seg: RowSegments,
    },
    /// Fused `[rows, 3 * heads * head_dim]` qkv -> per-head RMSNorm of q and k
    /// (gains `q_norm` / `k_norm`; address 0 = unaffine), partial half-rot
    /// RoPE over the leading `2 * rot_pairs` channels (bf16 `(cos, sin)` pairs,
    /// `rot_pairs` per row), and v copied out. Row layout `[q | k | v]` slabs,
    /// or `headwise`: `[q_h | k_h | v_h]` per head.
    QkNormRope {
        headwise: bool,
        /// `qkv` holds only the q projection (`[rows, heads * head_dim]`,
        /// slab layout); `k` / `v` / `k_norm` are ignored.
        q_only: bool,
        qkv: u64,
        q_norm: u64,
        k_norm: u64,
        freqs: u64,
        q: u64,
        k: u64,
        v: u64,
        rows: u32,
        heads: u32,
        head_dim: u32,
        rot_pairs: u32,
        eps: f32,
    },
    /// `out = bf16(x + a[seg] * y)` over `[rows, dim]`.
    GateAdd {
        x: u64,
        y: u64,
        out: u64,
        rows: u32,
        dim: u32,
        seg: RowSegments,
    },
    /// `out = bf16(x + bias[c])` over `[rows, dim]`. `out` may alias `x`.
    AddBias {
        x: u64,
        bias: u64,
        out: u64,
        rows: u32,
        dim: u32,
    },
    /// `out[r, c] = silu(x[r, c]) * x[r, h + c]` for `x` `[rows, 2h]`.
    SiluMulFused { x: u64, out: u64, rows: u32, h: u32 },
}

/// Shapes for [`Accelerator::vsa`]. Tiles are 64 rows: the first `n_prefix`
/// are dense (their queries see every key), the rest are video tiles whose
/// queries see every prefix tile plus their `topk` best video tiles.
#[derive(Clone, Copy, Debug)]
pub struct VsaShape {
    pub rows: u32,
    pub heads: u32,
    pub head_dim: u32,
    /// Rows per tile (64 or 128).
    pub tile: u32,
    /// QK^T in INT8 (K smoothed by its per-head mean, per-tile scales; P.V
    /// stays bf16): SageAttention-style, not bit-identical to the bf16 path.
    pub int8_qk: bool,
    pub n_tiles: u32,
    pub n_prefix: u32,
    pub topk: u32,
    pub scale: f32,
    /// Most tiles one [`VsaPart::Pool`] / [`VsaPart::Attend`] slab covers
    /// (sizes the INT8 Q); `n_tiles` for a whole-sequence call.
    pub slab_tiles: u32,
}

/// A run of whole tiles whose rows are the contiguous packed rows starting
/// at `row0`: the query side's unit when the forward runs in parts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VsaSlab {
    pub tile0: u32,
    pub tiles: u32,
    pub row0: u32,
}

impl VsaSlab {
    /// Every tile, rows from 0.
    pub fn whole(s: &VsaShape) -> Self {
        Self {
            tile0: 0,
            tiles: s.n_tiles,
            row0: 0,
        }
    }
}

/// [`Accelerator::vsa`] in parts, for a caller that never holds the whole
/// sequence's queries: `Pool` each slab as its rows exist (`q` = the slab's
/// rows, `k` / `v` every row), `Select` once all are pooled (tile scores,
/// key-tile lists, INT8 K), then `Attend` each slab (`q` / `gate` / `out` =
/// the slab's rows). The tables persist in `scratch` between parts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VsaPart {
    Pool(VsaSlab),
    Select,
    Attend(VsaSlab),
}

/// Device addresses for [`Accelerator::vsa`]. `gate == 0` = no compression
/// branch. `geometry` = `n_tiles * tile` u32 row ids then `n_tiles` u32 sizes.
#[derive(Clone, Copy, Debug)]
pub struct VsaAddrs {
    pub q: u64,
    pub k: u64,
    pub v: u64,
    pub gate: u64,
    pub out: u64,
    pub geometry: u64,
    pub scratch: u64,
    /// INT8 path only ([`VsaShape::int8_qk`]): the quantized Q/K region,
    /// [`VsaScratch::qk_i8`] bytes. Unused (0) on the bf16 path.
    pub qk_i8: u64,
}

/// Scratch [`Accelerator::vsa`] needs, in bytes: `scratch` for its tables and
/// `qk_i8` for the INT8 path's quantized Q/K (0 on the bf16 path). Separate
/// so the caller can place the large `qk_i8` in memory dead during the call.
#[derive(Clone, Copy, Debug)]
pub struct VsaScratch {
    pub scratch: u64,
    pub qk_i8: u64,
}

/// Device addresses + shape for [`Accelerator::gemm_nvfp4`].
#[derive(Clone, Copy, Debug)]
pub struct GemmNvfp4Addrs {
    pub a: u64,
    pub w: u64,
    pub out: u64,
    pub scratch: u64,
    pub m: u32,
    pub n: u32,
    pub k: u32,
    /// Optional low-rank term added in the epilogue (`lora.rank == 0` =
    /// none): `out += mid @ bt`, `mid` bf16 `[m, rank]`, `bt` bf16 `[rank,
    /// n]` (a runtime adapter's second factor, pre-transposed).
    pub lora: Nvfp4Lora,
    pub format: Nvfp4Format,
    /// Multiply only weight rows `[base, base + n)` of a composite with
    /// `total` rows (`base` a multiple of 128); `None` = the whole composite
    /// (`n` rows). The low-rank `bt` is then `[rank, total]`, read at `base`.
    pub rows: Option<Nvfp4Rows>,
}

/// A 128-row-aligned weight-row range of an NVFP4 composite.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Nvfp4Rows {
    pub base: u32,
    pub total: u32,
}

/// How a checkpoint's NVFP4 weights expect their activations quantized.
/// Default = FastVideo / FlashInfer: low nibble = even `k`, the composite's
/// static `input_sf`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Nvfp4Format {
    /// Activation global scale from this call's `amax(|a|)` (Comfy PTQ:
    /// `scale = bf16(amax / (448 * 6))`) instead of the static `input_sf`.
    pub dynamic_act_scale: bool,
    /// Weight codes packed HIGH nibble = even `k` (Comfy). The activations
    /// are packed the same way, which permutes `k` within each byte on both
    /// operands and leaves the product (and every 16-wide scale block) as is.
    pub hi_first: bool,
}

/// The low-rank epilogue term of [`GemmNvfp4Addrs`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Nvfp4Lora {
    pub mid: u64,
    pub bt: u64,
    pub rank: u32,
}

/// Largest adapter rank [`Accelerator::gemm_nvfp4`] fuses (the term's tiles
/// reuse the main loop's shared memory); see [`nvfp4_fuses_lora_rank`].
pub const NVFP4_FUSED_LORA_MAX_RANK: u32 = 64;

/// Whether [`Accelerator::gemm_nvfp4`] takes a low-rank term of this rank:
/// whole k16 MMA steps up to [`NVFP4_FUSED_LORA_MAX_RANK`]. Other adapters run
/// as separate GEMMs.
pub fn nvfp4_fuses_lora_rank(rank: u32) -> bool {
    rank > 0 && rank.is_multiple_of(16) && rank <= NVFP4_FUSED_LORA_MAX_RANK
}

/// Device addresses + launch shape for [`Accelerator::gemm`] (same roles as
/// `MatmulBufs`).
#[derive(Clone, Copy, Debug)]
pub struct GemmAddrs {
    pub a: u64,
    pub b: u64,
    pub out: u64,
    /// The `(m, n, k, _)` uniform, read for `k` when `k == 0`. A host `k`
    /// avoids that device read, which would wait for every enqueued op.
    pub dims: u64,
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub epilogue: GemmEpilogue,
}

/// Fused GEMM epilogue on device addresses, each part optional (0 = absent):
/// `out = residual + gate[n] * (acc + bias[n])`, one rounding. `bias` and
/// `gate` are per-output-column vectors, `residual` is `[m, n]` like `out` and
/// may alias it (an in-place accumulate). bf16-output GEMMs only; any other
/// output type declines a non-empty epilogue.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GemmEpilogue {
    pub bias: u64,
    pub residual: u64,
    pub gate: u64,
}

impl GemmEpilogue {
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }
}

/// Device addresses for [`Accelerator::sdpa_f16`] (same roles as `SdpaBufs`).
#[derive(Clone, Copy, Debug)]
pub struct SdpaAddrs {
    pub q: u64,
    pub k: u64,
    pub v: u64,
    pub mask: u64,
    pub out: u64,
    pub uniform: u64,
}
