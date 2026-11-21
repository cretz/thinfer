//! Job plaintext that never reaches swap and never outlives its owner.
//!
//! [`Locked`] owns a buffer of job data (decoded frames, decrypted inputs,
//! output bytes before sealing): its pages are locked in RAM while it lives
//! and it is zeroized, then unlocked, on drop. Locking only happens once a
//! server calls [`enforce`]; elsewhere (the local CLI) a `Locked` still
//! zeroizes but leaves paging alone.
//!
//! Under enforcement a lock that cannot be taken (no lockable memory left)
//! does not abort the allocation: it bumps a process-wide failure count
//! ([`failed_since`]) that the job runner treats as a cancellation, so the job ends
//! with an error instead of carrying on with swappable plaintext. Model
//! weights are never locked (they are not job data and would not fit).
//!
//! Locks are page-granular, and two small buffers can share a page. A
//! buffer's interior pages are its own; its two edge pages are counted
//! ([`EDGES`]), and a page is unlocked only once no live buffer holds it.

use std::collections::BTreeMap;
use std::ops::{Deref, DerefMut};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use zeroize::{DefaultIsZeroes, Zeroize};

static ENFORCED: AtomicBool = AtomicBool::new(false);
/// Lock failures so far (monotone: a job compares against its start mark, so
/// concurrent jobs never reset each other's failures).
static FAILURES: AtomicU64 = AtomicU64::new(0);

/// Lock every [`Locked`] buffer from now on (a server, at startup), raising
/// the process's lock limits where the platform allows it.
pub fn enforce() {
    platform::raise_limits();
    ENFORCED.store(true, Ordering::SeqCst);
}

/// A mark to take at a job's start, for [`failed_since`].
pub fn failure_mark() -> u64 {
    FAILURES.load(Ordering::SeqCst)
}

/// Whether a lock failed since `mark`: some job data is swappable, so the
/// running job must stop. Process-wide: a failure stops every job running at
/// the time (lockable memory is shared), but no job can clear another's.
pub fn failed_since(mark: u64) -> bool {
    FAILURES.load(Ordering::SeqCst) != mark
}

fn record_failure() {
    FAILURES.fetch_add(1, Ordering::SeqCst);
}

/// Live holders of each locked buffer's edge pages (page index -> count).
/// Held across the OS lock / unlock calls, so a page another buffer is just
/// locking is never unlocked under it.
static EDGES: Mutex<BTreeMap<usize, u32>> = Mutex::new(BTreeMap::new());

/// The first and last page of a non-empty range (equal for a one-page range).
fn edge_pages(ptr: usize, len: usize) -> (usize, usize) {
    let page = platform::page_size();
    (ptr / page, (ptr + len - 1) / page)
}

/// Lock `len > 0` bytes at `ptr`; `false` (nothing held) when the OS refused.
fn acquire(ptr: *const u8, len: usize) -> bool {
    let mut edges = EDGES.lock().unwrap_or_else(|p| p.into_inner());
    if !platform::lock(ptr.cast(), len) {
        return false;
    }
    let (first, last) = edge_pages(ptr as usize, len);
    *edges.entry(first).or_insert(0) += 1;
    if last != first {
        *edges.entry(last).or_insert(0) += 1;
    }
    true
}

/// Drop an [`acquire`] of the same range: unlock its pages except edge pages
/// another live buffer still holds.
fn give_back(ptr: *const u8, len: usize) {
    let mut edges = EDGES.lock().unwrap_or_else(|p| p.into_inner());
    let page = platform::page_size();
    let (first, last) = edge_pages(ptr as usize, len);
    // Drop one hold on page `p`; whether another buffer still holds it.
    let mut still_held = |p: usize| {
        let n = edges.get_mut(&p).expect("edge page of a held lock");
        *n -= 1;
        if *n == 0 {
            edges.remove(&p);
            false
        } else {
            true
        }
    };
    let (mut lo, mut hi) = (ptr as usize, ptr as usize + len);
    if first == last {
        if still_held(first) {
            hi = lo;
        }
    } else {
        if still_held(first) {
            lo = (first + 1) * page;
        }
        if still_held(last) {
            hi = last * page;
        }
    }
    if lo < hi {
        platform::unlock(lo as *const _, hi - lo);
    }
    platform::forget(len);
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
            let ok = acquire(buf.as_ptr().cast(), std::mem::size_of_val(&buf[..]));
            if !ok {
                record_failure();
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

    /// Enforcement is on but this buffer's pages could not be locked: do not
    /// fill it with plaintext.
    pub fn lock_failed(&self) -> bool {
        ENFORCED.load(Ordering::SeqCst) && !self.buf.is_empty() && !self.locked
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
            give_back(
                self.buf.as_ptr().cast(),
                std::mem::size_of_val(&self.buf[..]),
            );
        }
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

    /// Unlock pages of a held lock (a sub-range of it: shared edges stay).
    pub fn unlock(ptr: *const c_void, len: usize) {
        // SAFETY: `ptr..ptr + len` lies within a live, locked allocation.
        unsafe { VirtualUnlock(ptr, len) };
    }

    /// Return a `len`-byte lock's working-set growth.
    pub fn forget(len: usize) {
        grow(-((len + SLACK) as isize));
    }

    pub fn page_size() -> usize {
        use std::sync::OnceLock;
        use windows_sys::Win32::System::SystemInformation::{GetSystemInfo, SYSTEM_INFO};
        static PAGE: OnceLock<usize> = OnceLock::new();
        *PAGE.get_or_init(|| {
            // SAFETY: fills a plain out-struct.
            unsafe {
                let mut info: SYSTEM_INFO = std::mem::zeroed();
                GetSystemInfo(&mut info);
                info.dwPageSize as usize
            }
        })
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
            let page = page_size();
            let start = (ptr as usize) & !(page - 1);
            let end = (ptr as usize + len).next_multiple_of(page);
            // SAFETY: advice only, over pages of the caller's allocation.
            unsafe { libc::madvise(start as *mut c_void, end - start, libc::MADV_DONTDUMP) };
        }
        ok
    }

    /// Unlock pages of a held lock (a sub-range of it: shared edges stay).
    pub fn unlock(ptr: *const c_void, len: usize) {
        // SAFETY: `ptr..ptr + len` lies within a live, locked allocation.
        unsafe { libc::munlock(ptr, len) };
    }

    pub fn forget(_len: usize) {}

    pub fn page_size() -> usize {
        // SAFETY: a plain sysconf query.
        unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize }
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

    pub fn forget(_len: usize) {}

    pub fn page_size() -> usize {
        65536
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Locking really happens under enforcement, the data reads back intact,
    /// and drop releases it (no failure flagged for a small buffer).
    #[test]
    fn locked_buffer_round_trips() {
        enforce();
        let mark = failure_mark();
        let mut b = Locked::new(vec![1.5f32; 1 << 20]);
        assert!(b.locked || cfg!(not(any(windows, unix))));
        b[7] = 2.0;
        assert_eq!(b[7], 2.0);
        assert_eq!(b.len(), 1 << 20);
        drop(b);
        assert!(!failed_since(mark));
    }

    /// Two buffers sharing a page: dropping one leaves the shared page held
    /// for the other, and the table forgets it once both are gone.
    #[test]
    fn shared_edge_page_stays_locked_until_its_last_holder_drops() {
        enforce();
        let page = platform::page_size();
        let backing = vec![0u8; 4 * page];
        let base = backing.as_ptr() as usize;
        // A page boundary inside `backing`, and two ranges sharing the page
        // after it: `a` straddles the boundary, `b` sits within that page.
        let mid = (base / page + 2) * page;
        let a = ((mid - 100) as *const u8, 200);
        let b = ((mid + 50) as *const u8, 100);
        let holders = |addr: usize| EDGES.lock().unwrap().get(&(addr / page)).copied();
        assert!(acquire(a.0, a.1));
        assert!(acquire(b.0, b.1));
        assert_eq!(holders(mid), Some(2));
        give_back(a.0, a.1);
        assert_eq!(holders(mid), Some(1));
        assert_eq!(holders(mid - 100), None);
        give_back(b.0, b.1);
        assert_eq!(holders(mid), None);
        drop(backing);
    }
}
