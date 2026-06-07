use crate::backend::{Backend, BufRef};
use crate::cache::PipelineLookup;
use crate::tensor::{ComputeDtype, TensorDesc};
use crate::weight::WeightId;
use crate::workspace::{WeightTable, Workspace};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ModuleId(pub u32);

#[derive(Clone, Debug)]
pub struct ModuleSignature {
    pub inputs: Vec<TensorDesc>,
    pub outputs: Vec<TensorDesc>,
    pub weights: Vec<WeightId>,
    /// Bytes of `Workspace` scratch the module needs at peak. Sized at
    /// `prepare()` time from concrete activation shapes; runtime sizes the
    /// scratch buffer once for the whole pipeline (max over modules).
    pub workspace_bytes: u64,
}

/// Everything `forward` needs to issue dispatches into a caller-owned
/// `CommandEncoder`. Per plan-details: no cross-module readback, no
/// mid-forward awaits, single submit per module.
///
/// Borrows are split so `encoder` and `scratch` can be `&mut` independently
/// (separate accesses; no aliasing).
pub struct ForwardCtx<'a, B: Backend> {
    pub backend: &'a B,
    pub encoder: &'a mut B::CommandEncoder,
    pub pipelines: &'a dyn PipelineLookup<B>,
    pub weights: &'a dyn WeightTable<B>,
    pub scratch: &'a Workspace<B>,
}

pub trait Module<B: Backend> {
    type Dtype: ComputeDtype;

    fn signature(&self) -> &ModuleSignature;

    /// Append all dispatches for one forward pass to `ctx.encoder`. No awaits
    /// inside; runtime owns submit. Caller passes activation `BufRef`s for
    /// inputs/outputs (whole buffers or workspace views — module-agnostic).
    fn forward(
        &self,
        ctx: &mut ForwardCtx<'_, B>,
        inputs: &[BufRef],
        outputs: &[BufRef],
    ) -> Result<(), B::Error>;
}
