//! Per-device pump that drains wgpu's callback queue (map_async, queue
//! completion) without blocking the async executor.
//!
//! Native: a single OS thread parked at rest. While any `PollGuard` is alive
//! it loops on `device.poll(Maintain::Wait)`; when all guards drop it re-parks.
//! `Arc::strong_count` on a sentinel is the wake/sleep signal — baseline is 2
//! (one in `WgpuPoll`, one in the thread); a guard pushes it to 3+.
//!
//! The thread EXITS when [`WgpuPoll`] drops (a `shutdown` flag the thread checks
//! at the top of its loop), and `Drop` joins it. This is load-bearing for the
//! serve's fresh-device-per-job model: the thread `move`s a clone of
//! `Arc<wgpu::Device>`, so a thread that never exits pins the device (and a
//! parked OS thread) for the life of the process — leaking one device wrapper +
//! one thread per job even though the backend is dropped. Joining on drop
//! releases the thread's device clone so the device Arc can reach zero.
//!
//! Web: no-op. The browser event loop drives wgpu callbacks natively, so a
//! Rust-side poller would be both useless and unspawnable on wasm.
//!
//! Pattern adapted from cubecl (`crates/cubecl-wgpu/src/compute/poll.rs`).

use std::sync::Arc;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(not(target_arch = "wasm32"))]
pub struct WgpuPoll {
    sentinel: Arc<()>,
    shutdown: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

#[cfg(target_arch = "wasm32")]
pub struct WgpuPoll;

#[cfg(not(target_arch = "wasm32"))]
impl WgpuPoll {
    pub fn new(device: Arc<wgpu::Device>) -> Self {
        let sentinel = Arc::new(());
        let sentinel_thread = sentinel.clone();
        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_thread = shutdown.clone();
        let handle = std::thread::Builder::new()
            .name("thinfer-wgpu-poll".into())
            .spawn(move || {
                // Baseline strong_count = 2 (this clone + the WgpuPoll's clone).
                // A live PollGuard makes it 3+.
                loop {
                    // Checked first so a shutdown always wins, even mid-guard.
                    if shutdown_thread.load(Ordering::Acquire) {
                        break;
                    }
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
                // `device` + the sentinel/shutdown clones drop here, releasing
                // this thread's hold on the wgpu device.
            })
            .expect("spawn wgpu poll thread");
        Self {
            sentinel,
            shutdown,
            handle: Some(handle),
        }
    }

    pub fn poll_guard(&self) -> PollGuard {
        let g = self.sentinel.clone();
        if let Some(h) = &self.handle {
            h.thread().unpark();
        }
        PollGuard { _sentinel: g }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Drop for WgpuPoll {
    fn drop(&mut self) {
        // Signal exit, then wake the (possibly parked) thread so it observes the
        // flag and returns, dropping its device clone. Join so the device is
        // fully released before this returns (per-job device teardown on serve).
        self.shutdown.store(true, Ordering::Release);
        if let Some(handle) = self.handle.take() {
            handle.thread().unpark();
            let _ = handle.join();
        }
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
