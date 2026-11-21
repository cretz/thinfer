//! Batched direct reads: many small ranges (MoE expert slices, ~0.6 MiB)
//! coalesced into few large unbuffered reads.
//!
//! Measured on a Gen4 NVMe (8 readers, random offsets, unbuffered): 640 KiB
//! requests 2.35 GB/s at any queue depth, 2 MiB 3.7, 8 MiB 5.3; the same
//! 640 KiB / 4 MiB reads through the page cache 1.5 / 2.5 GB/s. A request
//! costs ~0.16 ms beyond its bytes and a byte ~0.18 ms/MB, so reading through
//! a gap is cheaper than a new request up to [`MERGE_GAP`]. Unbuffered reads
//! also keep the bytes out of the OS page cache: the caller's budgeted
//! buffers are the only copy.

use std::cell::RefCell;
use std::io;
use std::path::Path;
use std::sync::Arc;

use super::{RawChunk, Workers, read_pool};

/// Sector alignment for unbuffered reads (covers 512 B and 4 KiB sectors).
const ALIGN: u64 = 4096;
/// Ranges closer than this are read as one request, gap included.
const MERGE_GAP: u64 = 768 << 10;
/// A run stops growing past this (larger requests gain little, see above).
const MAX_RUN: u64 = 8 << 20;

/// A handle that bypasses the OS page cache, or `None` when the platform or
/// filesystem refuses one (reads then go through the buffered handle).
pub(crate) fn open(path: &Path) -> Option<std::fs::File> {
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        const FILE_FLAG_NO_BUFFERING: u32 = 0x2000_0000;
        std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(FILE_FLAG_NO_BUFFERING)
            .open(path)
            .ok()
    }
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::fs::OpenOptionsExt;
        std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECT)
            .open(path)
            .ok()
    }
    #[cfg(target_os = "macos")]
    {
        use std::os::fd::AsRawFd;
        let f = std::fs::File::open(path).ok()?;
        // SAFETY: plain fcntl on a descriptor we own.
        (unsafe { libc::fcntl(f.as_raw_fd(), libc::F_NOCACHE, 1) } != -1).then_some(f)
    }
    #[cfg(not(any(windows, target_os = "linux", target_os = "macos")))]
    {
        let _ = path;
        None
    }
}

/// One coalesced read: `[start, end)` of the file and the requests inside it.
struct Run {
    start: u64,
    end: u64,
    parts: Vec<(u64, RawChunk)>,
}

/// Fill every `(offset, dst)` from `file` (`len` bytes long) with coalesced,
/// sector-aligned reads on the read pool. `dst`s must be disjoint.
pub(crate) async fn read_batch(
    file: &Arc<std::fs::File>,
    len: u64,
    reqs: &mut [(u64, &mut [u8])],
) -> io::Result<()> {
    let mut order: Vec<usize> = (0..reqs.len()).filter(|&i| !reqs[i].1.is_empty()).collect();
    order.sort_unstable_by_key(|&i| reqs[i].0);
    let mut runs: Vec<Run> = Vec::new();
    for i in order {
        let (off, dst) = (reqs[i].0, &mut *reqs[i].1);
        let end = off + dst.len() as u64;
        if end > len {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "read past file end",
            ));
        }
        let part = (off, RawChunk(dst.as_mut_ptr(), dst.len()));
        match runs.last_mut() {
            Some(r) if off <= r.end + MERGE_GAP && end.max(r.end) - r.start <= MAX_RUN => {
                r.end = r.end.max(end);
                r.parts.push(part);
            }
            _ => runs.push(Run {
                start: off,
                end,
                parts: vec![part],
            }),
        }
    }
    let mut workers = Workers(Vec::with_capacity(runs.len()));
    let mut done = Vec::with_capacity(runs.len());
    for run in runs {
        let (tx, rx) = futures_channel::oneshot::channel();
        let file = Arc::clone(file);
        workers.0.push(read_pool().run(move || {
            let _ = tx.send(read_run(&file, len, run));
        }));
        done.push(rx);
    }
    for rx in done {
        rx.await
            .map_err(|_| io::Error::other("read worker panicked"))??;
    }
    Ok(())
}

/// Read one run into this worker's aligned staging buffer and scatter it.
fn read_run(file: &std::fs::File, len: u64, run: Run) -> io::Result<()> {
    let a0 = run.start / ALIGN * ALIGN;
    let a1 = run.end.next_multiple_of(ALIGN);
    let want = (a1 - a0) as usize;
    STAGE.with_borrow_mut(|stage| {
        let buf = stage.get(want);
        // Whole sectors are requested; only the file's tail may come back short.
        let need = (run.end - a0) as usize;
        let mut got = 0;
        while got < need {
            #[cfg(windows)]
            let n =
                std::os::windows::fs::FileExt::seek_read(file, &mut buf[got..], a0 + got as u64)?;
            #[cfg(unix)]
            let n = std::os::unix::fs::FileExt::read_at(file, &mut buf[got..], a0 + got as u64)?;
            if n == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    format!("short read at {} of a {len} B file", a0 + got as u64),
                ));
            }
            got += n;
        }
        for (off, chunk) in run.parts {
            let at = (off - a0) as usize;
            // SAFETY: the batch's `dst`s are disjoint and outlive this job
            // (`Workers` waits for it even if the batch future is dropped).
            let dst = unsafe { chunk.slice() };
            let n = dst.len();
            dst.copy_from_slice(&buf[at..at + n]);
        }
        Ok(())
    })
}

thread_local! {
    /// Per-reader-thread staging, kept at the largest run seen (<= MAX_RUN
    /// plus alignment, or one oversized request).
    static STAGE: RefCell<Aligned> = const { RefCell::new(Aligned { ptr: std::ptr::null_mut(), cap: 0 }) };
}

/// A sector-aligned byte buffer (unbuffered reads need an aligned target).
struct Aligned {
    ptr: *mut u8,
    cap: usize,
}

impl Aligned {
    fn layout(cap: usize) -> std::alloc::Layout {
        std::alloc::Layout::from_size_align(cap, ALIGN as usize).expect("staging layout")
    }

    fn get(&mut self, len: usize) -> &mut [u8] {
        if len > self.cap {
            self.release();
            let cap = len.next_multiple_of(1 << 20);
            // SAFETY: non-zero size, valid alignment.
            let ptr = unsafe { std::alloc::alloc_zeroed(Self::layout(cap)) };
            if ptr.is_null() {
                std::alloc::handle_alloc_error(Self::layout(cap));
            }
            (self.ptr, self.cap) = (ptr, cap);
        }
        // SAFETY: `ptr` owns `cap >= len` initialized (zeroed) bytes.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, len) }
    }

    fn release(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: allocated in `get` with this layout.
            unsafe { std::alloc::dealloc(self.ptr, Self::layout(self.cap)) };
            (self.ptr, self.cap) = (std::ptr::null_mut(), 0);
        }
    }
}

impl Drop for Aligned {
    fn drop(&mut self) {
        self.release();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_matches_file_bytes() {
        // Not sector-sized, so the last run ends in a short read.
        let len = (21 << 20) + 1234;
        let bytes: Vec<u8> = (0..len as u32)
            .map(|i| (i.wrapping_mul(2_654_435_761) >> 13) as u8)
            .collect();
        let path = std::env::temp_dir().join(format!("thinfer-direct-{}.bin", std::process::id()));
        std::fs::write(&path, &bytes).unwrap();
        // Unsorted; adjacent, gapped (merged and not), page-straddling, one
        // past MAX_RUN, one ending at the file's last byte.
        let spans: [(u64, usize); 7] = [
            (5 << 20, 655_361),
            (17, 4000),
            (4100, 600_000),
            (604_100, 3),
            (2 << 20, 9 << 20),
            (len as u64 - 70_001, 70_001),
            (3_000_001, 12_345),
        ];
        let mut dsts: Vec<Vec<u8>> = spans.iter().map(|&(_, n)| vec![0; n]).collect();
        let mut reqs: Vec<(u64, &mut [u8])> = spans
            .iter()
            .zip(dsts.iter_mut())
            .map(|(&(o, _), d)| (o, d.as_mut_slice()))
            .collect();
        for file in [open(&path), Some(std::fs::File::open(&path).unwrap())] {
            let file = Arc::new(file.expect("direct handle"));
            reqs.iter_mut().for_each(|(_, d)| d.fill(0));
            pollster::block_on(read_batch(&file, len as u64, &mut reqs)).unwrap();
            for (&(o, n), (_, d)) in spans.iter().zip(reqs.iter()) {
                assert!(d[..] == bytes[o as usize..o as usize + n], "span at {o}");
            }
        }
        let mut past = [(len as u64 - 10, &mut [0u8; 11][..])];
        let file = Arc::new(std::fs::File::open(&path).unwrap());
        assert!(pollster::block_on(read_batch(&file, len as u64, &mut past)).is_err());
        std::fs::remove_file(&path).unwrap();
    }
}
