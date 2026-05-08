//! Per-device pump that drains wgpu's callback queue (map_async, queue
//! completion) without blocking the async executor.
//!
//! Native: a single OS thread parked at rest. While any `PollGuard` is alive
//! it loops on `device.poll(Maintain::Wait)`; when all guards drop it re-parks.
//! `Arc::strong_count` on a sentinel is the wake/sleep signal — baseline is 2
//! (one in `WgpuPoll`, one in the thread); a guard pushes it to 3+.
//!
//! Web: no-op. The browser event loop drives wgpu callbacks natively, so a
//! Rust-side poller would be both useless and unspawnable on wasm.
//!
//! Pattern adapted from cubecl (`crates/cubecl-wgpu/src/compute/poll.rs`).

use std::sync::Arc;

#[cfg(not(target_arch = "wasm32"))]
pub struct WgpuPoll {
    sentinel: Arc<()>,
    thread: std::thread::Thread,
}

#[cfg(target_arch = "wasm32")]
pub struct WgpuPoll;

#[cfg(not(target_arch = "wasm32"))]
impl WgpuPoll {
    pub fn new(device: Arc<wgpu::Device>) -> Self {
        let sentinel = Arc::new(());
        let sentinel_thread = sentinel.clone();
        let handle = std::thread::Builder::new()
            .name("thinfer-wgpu-poll".into())
            .spawn(move || {
                // Baseline strong_count = 2 (this clone + the WgpuPoll's clone).
                // A live PollGuard makes it 3+.
                loop {
                    if Arc::strong_count(&sentinel_thread) > 2 {
                        if let Err(err) = device.poll(wgpu::PollType::wait_indefinitely()) {
                            tracing::error!(
                                target: crate::trace::WGPU_ERR,
                                kind = "poll",
                                error = ?err,
                            );
                        }
                    } else {
                        std::thread::park();
                    }
                }
            })
            .expect("spawn wgpu poll thread");
        Self {
            sentinel,
            thread: handle.thread().clone(),
        }
    }

    pub fn poll_guard(&self) -> PollGuard {
        let g = self.sentinel.clone();
        self.thread.unpark();
        PollGuard { _sentinel: g }
    }
}

#[cfg(target_arch = "wasm32")]
impl WgpuPoll {
    pub fn new(_device: Arc<wgpu::Device>) -> Self {
        Self
    }

    pub fn poll_guard(&self) -> PollGuard {
        PollGuard
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub struct PollGuard {
    _sentinel: Arc<()>,
}

#[cfg(target_arch = "wasm32")]
pub struct PollGuard;
