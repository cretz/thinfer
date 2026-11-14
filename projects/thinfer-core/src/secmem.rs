//! Job plaintext that never reaches swap and never outlives its owner.
//!
//! [`Locked`] owns a buffer of job data (decoded frames, decrypted inputs,
//! output bytes before sealing): its pages are locked in RAM while it lives
//! and it is zeroized, then unlocked, on drop. Locking only happens once a
//! server calls [`enforce`]; elsewhere (the local CLI) a `Locked` still
//! zeroizes but leaves paging alone.
//!
//! Under enforcement a lock that cannot be taken (no lockable memory left)
//! does not abort the allocation: it raises a process-wide flag
//! ([`failed`]) that the job runner treats as a cancellation, so the job ends
//! with an error instead of carrying on with swappable plaintext. Model
//! weights are never locked (they are not job data and would not fit).
//!
//! Locks are page-granular and not reference counted: two buffers sharing an
//! edge page unlock it when the first is dropped. Large buffers come from
//! their own mappings, so this only ever concerns a buffer's edge pages.

use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicBool, Ordering};

use zeroize::{DefaultIsZeroes, Zeroize};

static ENFORCED: AtomicBool = AtomicBool::new(false);
static FAILED: AtomicBool = AtomicBool::new(false);

/// Lock every [`Locked`] buffer from now on (a server, at startup), raising
/// the process's lock limits where the platform allows it.
pub fn enforce() {
    platform::raise_limits();
    ENFORCED.store(true, Ordering::SeqCst);
}

/// Whether a lock failed since the last [`clear_failure`]: some job data is
/// swappable, so the running job must stop.
pub fn failed() -> bool {
    FAILED.load(Ordering::SeqCst)
}

/// Reset [`failed`] (at a job's start).
pub fn clear_failure() {
    FAILED.store(false, Ordering::SeqCst);
}

/// A page-locked (under [`enforce`]), zeroize-on-drop buffer. Its length is
/// fixed: it derefs to a slice, so it can never reallocate out of the locked
/// pages.
pub struct Locked<T: DefaultIsZeroes> {
    buf: Vec<T>,
    locked: bool,
}

impl<T: DefaultIsZeroes> Locked<T> {
    /// Take ownership of `buf` and lock its pages.
    pub fn new(buf: Vec<T>) -> Self {
        let locked = ENFORCED.load(Ordering::SeqCst) && !buf.is_empty() && {
            let ok = platform::lock(buf.as_ptr().cast(), std::mem::size_of_val(&buf[..]));
            if !ok {
                FAILED.store(true, Ordering::SeqCst);
                tracing::error!(
                    target: crate::trace::ADAPTER,
                    bytes = std::mem::size_of_val(&buf[..]),
                    "could not lock job data in RAM (not enough lockable memory); stopping the job",
                );
            }
            ok
        };
        Self { buf, locked }
    }

    /// A zeroed buffer of `len` elements.
    pub fn zeroed(len: usize) -> Self {
        Self::new(vec![T::default(); len])
    }
}

impl<T: DefaultIsZeroes> Deref for Locked<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        &self.buf
    }
}

impl<T: DefaultIsZeroes> DerefMut for Locked<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        &mut self.buf
    }
}

impl<T: DefaultIsZeroes> From<Vec<T>> for Locked<T> {
    fn from(buf: Vec<T>) -> Self {
        Self::new(buf)
    }
}

impl<T: DefaultIsZeroes + std::fmt::Debug> std::fmt::Debug for Locked<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Locked({} elements)", self.buf.len())
    }
}

impl<T: DefaultIsZeroes> Drop for Locked<T> {
    fn drop(&mut self) {
        self.buf.as_mut_slice().zeroize();
        if self.locked {
            platform::unlock(
                self.buf.as_ptr().cast(),
                std::mem::size_of_val(&self.buf[..]),
            );
        }
    }
}

/// Lock `bytes` for as long as the caller keeps them (no ownership): for a
/// buffer whose owner zeroizes it already. Pair with [`release`].
pub fn lock_slice(bytes: &[u8]) {
    if ENFORCED.load(Ordering::SeqCst)
        && !bytes.is_empty()
        && !platform::lock(bytes.as_ptr().cast(), bytes.len())
    {
        FAILED.store(true, Ordering::SeqCst);
        tracing::error!(
            target: crate::trace::ADAPTER,
            bytes = bytes.len(),
            "could not lock job data in RAM (not enough lockable memory); stopping the job",
        );
    }
}

/// Zeroize `bytes` and drop a [`lock_slice`] lock.
pub fn release(bytes: &mut [u8]) {
    bytes.zeroize();
    if ENFORCED.load(Ordering::SeqCst) && !bytes.is_empty() {
        platform::unlock(bytes.as_ptr().cast(), bytes.len());
    }
}

#[cfg(windows)]
mod platform {
    //! `VirtualLock` caps locked pages at the process's minimum working set,
    //! so each lock first grows that minimum by its size (and each unlock
    //! shrinks it back): the process asks the OS for exactly what it holds.
    use std::ffi::c_void;
    use std::sync::Mutex;

    use windows_sys::Win32::System::Memory::{VirtualLock, VirtualUnlock};
    use windows_sys::Win32::System::Threading::{
        GetCurrentProcess, GetProcessWorkingSetSize, SetProcessWorkingSetSize,
    };

    /// Serializes working-set adjustments.
    static WS: Mutex<()> = Mutex::new(());
    /// Headroom past the locked bytes for the pages a lock call itself
    /// touches (page tables, rounding to whole pages).
    const SLACK: usize = 16 << 20;

    pub fn raise_limits() {}

    fn grow(delta: isize) -> bool {
        let _g = WS.lock().unwrap_or_else(|p| p.into_inner());
        let (mut min, mut max) = (0usize, 0usize);
        // SAFETY: the current-process pseudo handle and two out-params.
        unsafe {
            if GetProcessWorkingSetSize(GetCurrentProcess(), &mut min, &mut max) == 0 {
                return false;
            }
            let min = min.saturating_add_signed(delta);
            SetProcessWorkingSetSize(GetCurrentProcess(), min, max.max(min + SLACK)) != 0
        }
    }

    pub fn lock(ptr: *const c_void, len: usize) -> bool {
        if !grow((len + SLACK) as isize) {
            return false;
        }
        // SAFETY: `ptr..ptr + len` is a live allocation the caller owns.
        let ok = unsafe { VirtualLock(ptr, len) } != 0;
        if !ok {
            grow(-((len + SLACK) as isize));
        }
        ok
    }

    pub fn unlock(ptr: *const c_void, len: usize) {
        // SAFETY: as in `lock`; unlocking pages already unlocked is harmless.
        unsafe { VirtualUnlock(ptr, len) };
        grow(-((len + SLACK) as isize));
    }
}

#[cfg(unix)]
mod platform {
    use std::ffi::c_void;

    /// The soft lock limit up to the hard one (the most an unprivileged
    /// process may lock).
    pub fn raise_limits() {
        let mut lim = libc::rlimit {
            rlim_cur: 0,
            rlim_max: 0,
        };
        // SAFETY: plain get/setrlimit on the current process.
        unsafe {
            if libc::getrlimit(libc::RLIMIT_MEMLOCK, &mut lim) == 0 && lim.rlim_cur < lim.rlim_max {
                lim.rlim_cur = lim.rlim_max;
                libc::setrlimit(libc::RLIMIT_MEMLOCK, &lim);
            }
        }
    }

    pub fn lock(ptr: *const c_void, len: usize) -> bool {
        // SAFETY: `ptr..ptr + len` is a live allocation the caller owns.
        let ok = unsafe { libc::mlock(ptr, len) } == 0;
        #[cfg(target_os = "linux")]
        if ok {
            // Out of core dumps too (page-aligned range covering the buffer).
            let page = 4096usize;
            let start = (ptr as usize) & !(page - 1);
            let end = (ptr as usize + len).next_multiple_of(page);
            // SAFETY: advice only, over pages of the caller's allocation.
            unsafe { libc::madvise(start as *mut c_void, end - start, libc::MADV_DONTDUMP) };
        }
        ok
    }

    pub fn unlock(ptr: *const c_void, len: usize) {
        // SAFETY: as in `lock`.
        unsafe { libc::munlock(ptr, len) };
    }
}

#[cfg(not(any(windows, unix)))]
mod platform {
    //! wasm: no paging to guard against; zeroizing still applies.
    use std::ffi::c_void;

    pub fn raise_limits() {}

    pub fn lock(_ptr: *const c_void, _len: usize) -> bool {
        true
    }

    pub fn unlock(_ptr: *const c_void, _len: usize) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Locking really happens under enforcement, the data reads back intact,
    /// and drop releases it (no failure flagged for a small buffer).
    #[test]
    fn locked_buffer_round_trips() {
        enforce();
        clear_failure();
        let mut b = Locked::new(vec![1.5f32; 1 << 20]);
        assert!(b.locked || cfg!(not(any(windows, unix))));
        b[7] = 2.0;
        assert_eq!(b[7], 2.0);
        assert_eq!(b.len(), 1 << 20);
        drop(b);
        assert!(!failed());
    }
}
