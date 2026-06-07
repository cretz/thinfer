use crate::backend::Backend;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct KernelKey {
    pub kernel_id: &'static str,
    pub hint: String,
}

pub trait PipelineCache<B: Backend> {
    fn get(&self, key: &KernelKey) -> Option<&B::Pipeline>;
    fn insert(&mut self, key: KernelKey, pipeline: B::Pipeline);
    fn contains(&self, key: &KernelKey) -> bool;
}

/// Read-only, object-safe pipeline accessor. `ForwardCtx` carries this
/// instead of the full `PipelineCache` to keep `forward()` immune to
/// cache-mutation surface area: cold-path inserts happen at module *prepare*
/// time so the first dispatch is never blocked on compile (per
/// plan-details "Concurrency on the hot path").
pub trait PipelineLookup<B: Backend> {
    fn get(&self, key: &KernelKey) -> Option<&B::Pipeline>;
}

impl<B: Backend, T: PipelineCache<B> + ?Sized> PipelineLookup<B> for T {
    fn get(&self, key: &KernelKey) -> Option<&B::Pipeline> {
        PipelineCache::get(self, key)
    }
}
