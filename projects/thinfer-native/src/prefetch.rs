//! Warm the OS page cache with files a later phase will read, while an earlier
//! phase leaves the disk idle. Bytes go nowhere but the page cache (no process
//! memory, no disk cache of our own); a machine without the RAM to hold them
//! skips it rather than evict what the current phase needs.

use std::io::Read;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;

/// RAM left to everything else after a prefetch: the running phase's working
/// set and the rest of the system.
const RESERVE: u64 = 8 << 30;
const CHUNK: usize = 16 << 20;

/// Physical memory the OS reports available right now (free plus reclaimable
/// standby/cache). `None` where unknown: callers treat that as "no room".
pub fn available_ram() -> Option<u64> {
    #[cfg(windows)]
    {
        use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
        let mut s = MEMORYSTATUSEX {
            dwLength: size_of::<MEMORYSTATUSEX>() as u32,
            ..Default::default()
        };
        // SAFETY: `s` is a valid MEMORYSTATUSEX with dwLength set.
        (unsafe { GlobalMemoryStatusEx(&mut s) } != 0).then_some(s.ullAvailPhys)
    }
    #[cfg(target_os = "linux")]
    {
        let info = std::fs::read_to_string("/proc/meminfo").ok()?;
        let kb: u64 = info
            .lines()
            .find_map(|l| l.strip_prefix("MemAvailable:"))?
            .trim()
            .trim_end_matches("kB")
            .trim()
            .parse()
            .ok()?;
        Some(kb * 1024)
    }
    #[cfg(not(any(windows, target_os = "linux")))]
    {
        None
    }
}

/// A background read of `paths`, front to back. Stops at the next chunk once
/// dropped.
pub struct Prefetch {
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl Prefetch {
    /// Start reading `paths` unless their total plus [`RESERVE`] exceeds the
    /// available RAM (then the bytes would only evict each other).
    pub fn start(paths: Vec<PathBuf>) -> Option<Self> {
        let total: u64 = paths
            .iter()
            .filter_map(|p| std::fs::metadata(p).ok())
            .map(|m| m.len())
            .sum();
        let avail = available_ram()?;
        if total + RESERVE > avail {
            tracing::info!(
                target: thinfer_core::trace::DIAG,
                total, avail, "prefetch skipped: not enough free RAM",
            );
            return None;
        }
        let stop = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&stop);
        let thread = std::thread::Builder::new()
            .name("prefetch".into())
            .spawn(move || {
                let mut buf = vec![0u8; CHUNK];
                for path in &paths {
                    let Ok(mut file) = std::fs::File::open(path) else {
                        continue;
                    };
                    while !flag.load(Ordering::Relaxed) {
                        match file.read(&mut buf) {
                            Ok(0) | Err(_) => break,
                            Ok(_) => {}
                        }
                    }
                }
                tracing::info!(target: thinfer_core::trace::DIAG, "prefetch done");
            })
            .ok()?;
        Some(Self {
            stop,
            thread: Some(thread),
        })
    }
}

impl Drop for Prefetch {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}
