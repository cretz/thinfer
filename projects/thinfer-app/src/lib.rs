//! thinfer-app: the neutral orchestration layer between `thinfer-native` /
//! `thinfer-models` and the binaries (CLI today; serve / desktop / python
//! later). It owns the `Job` request types, the unified progress vocabulary,
//! the shared download/load/generate/encode dance, and the [`LocalExecutor`].
//!
//! Front ends are thin adapters: they translate their input (clap flags, HTTP
//! JSON) into a [`JobRequest`], provide a [`ProgressSink`], ensure the request's
//! files are cached, and call [`LocalExecutor::run`].

pub mod codec;
pub mod config;
pub mod download;
pub mod executor;
pub mod hunyuan;
pub mod ltx;
pub mod model;
pub mod preprocess;
pub mod progress;
#[cfg(feature = "remote")]
pub mod remote;
pub mod request;
#[cfg(feature = "rewrite")]
pub mod rewrite;
#[cfg(feature = "vault")]
pub mod vault;
#[cfg(feature = "serde")]
pub mod wire;

pub use config::{BackendConfig, DEFAULT_BUDGET_BYTES, ResidencyBudget, parse_budget, report_mem};
pub use executor::LocalExecutor;
pub use progress::{NullSink, ProgressSink, Stage};
pub use request::{
    FaceSwapRequest, ImageRequest, JobRequest, JobSummary, VideoRequest, resolve_output_format,
};
