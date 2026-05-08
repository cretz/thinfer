use crate::module::ModuleId;
use crate::runtime::RuntimeState;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WorkTag(pub u64);

pub struct WorkItem {
    pub module: ModuleId,
    pub inputs: Vec<TensorRef>,
    pub outputs: Vec<TensorRef>,
    pub tag: WorkTag,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TensorRef(pub u64);

pub enum NextWork {
    Ready(WorkItem),
    Pending,
    Done,
}

pub trait Pipeline {
    fn next_work(&mut self, state: &RuntimeState) -> NextWork;
}
