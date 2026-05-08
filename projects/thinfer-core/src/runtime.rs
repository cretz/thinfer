use crate::module::ModuleId;
use crate::pipeline::WorkTag;
use std::collections::HashSet;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tier {
    GpuDevice,
    GpuHostVisible,
    Host,
    Cold,
}

#[derive(Clone, Debug, Default)]
pub struct TierBudget {
    pub bytes_total: u64,
    pub bytes_in_use: u64,
}

impl TierBudget {
    pub fn remaining(&self) -> u64 {
        self.bytes_total.saturating_sub(self.bytes_in_use)
    }
}

#[derive(Default)]
pub struct RuntimeState {
    pub completed: HashSet<WorkTag>,
    pub step: u64,
    pub gpu_device: TierBudget,
    pub gpu_host_visible: TierBudget,
    pub host: TierBudget,
    pub cold: TierBudget,
}

impl RuntimeState {
    pub fn budget(&self, tier: Tier) -> &TierBudget {
        match tier {
            Tier::GpuDevice => &self.gpu_device,
            Tier::GpuHostVisible => &self.gpu_host_visible,
            Tier::Host => &self.host,
            Tier::Cold => &self.cold,
        }
    }

    pub fn is_completed(&self, tag: WorkTag) -> bool {
        self.completed.contains(&tag)
    }
}

pub trait Runtime {
    type Error;

    fn load_module(
        &mut self,
        id: ModuleId,
    ) -> impl core::future::Future<Output = Result<(), Self::Error>>;

    fn evict_module(&mut self, id: ModuleId);

    fn state(&self) -> &RuntimeState;
}
