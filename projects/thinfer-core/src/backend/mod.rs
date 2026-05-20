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

/// Compute backend abstraction. v1 carries a single bind group (group 0); the
/// concept generalizes when we hit a kernel that wants more.
pub trait Backend: 'static {
    type Error: core::fmt::Debug + Send + 'static;
    type CommandEncoder;
    type Pipeline;

    fn allocate(&self, bytes: u64) -> Result<GpuBufferId, Self::Error>;
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

    /// Host bytes → GPU buffer. Test inputs and host-resident weights land here.
    /// JS-heap weights on web go through a backend-specific path that bypasses
    /// `&[u8]` (per plan-details: no weight bytes in WASM linear memory).
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

    fn create_pipeline(
        &self,
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
}
