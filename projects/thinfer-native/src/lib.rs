// thinfer-native: desktop/native impls of thinfer-core IO traits.

pub mod cache;
pub mod prefetch;
pub mod sources;
pub mod tokenizer;

use std::io;
use std::path::Path;
use std::sync::Arc;
use thinfer_core::weight::{FileOpener, WeightReader};

/// `FileOpener` backed by a single mmap'd view of the file. Weight bytes live
/// in the OS page cache, not our process heap: RSS scales with working set,
/// not model size. `open()` clones the `Arc`s and is effectively free.
/// `read_at` is a memcpy from the mapped region for small reads and a
/// positioned file read for large ones (see [`DIRECT_READ_MIN`]).
pub struct MmapFileOpener {
    mmap: Arc<memmap2::Mmap>,
    file: Arc<std::fs::File>,
}

impl MmapFileOpener {
    pub async fn new(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let (mmap, file) = tokio::task::spawn_blocking(move || {
            let file = std::fs::File::open(&path)?;
            // SAFETY: We hold no claim that the underlying file is immutable
            // for the lifetime of the mmap; concurrent external truncate or
            // rewrite is UB. Same assumption as safetensors's own mmap loaders
            // and pytorch's safetensors mmap path. Out of scope to defend.
            let mmap = unsafe { memmap2::Mmap::map(&file) }?;
            io::Result::Ok((mmap, file))
        })
        .await
        .expect("mmap blocking task panicked")?;
        Ok(Self {
            mmap: Arc::new(mmap),
            file: Arc::new(file),
        })
    }
}

impl FileOpener for MmapFileOpener {
    type Reader = MmapFile;
    type Error = io::Error;
    async fn open(&self) -> Result<Self::Reader, Self::Error> {
        Ok(MmapFile {
            mmap: Arc::clone(&self.mmap),
            file: Arc::clone(&self.file),
        })
    }
}

/// Reads at least this large bypass the mapping. A cold mmap copy pages in
/// fault by fault (measured ~1 GB/s on an NVMe that serves 3+ GB/s to plain
/// 16 MiB reads); a positioned read hands the OS the whole range as one
/// request. Warm (page-cached) bytes cost the same either way.
const DIRECT_READ_MIN: usize = 1 << 20;

pub struct MmapFile {
    mmap: Arc<memmap2::Mmap>,
    file: Arc<std::fs::File>,
}

/// Reads at least this large are split into [`READ_PARTS`] concurrent
/// positioned reads: one request at a time leaves an NVMe at queue depth 1
/// (measured cold: 3.1 GB/s at one reader, 4.8 GB/s at four).
const SPLIT_READ_MIN: usize = 8 << 20;
const READ_PARTS: usize = 4;

/// [`read_exact_at`] on worker threads, as [`READ_PARTS`] concurrent reads
/// when `dst` is at least [`SPLIT_READ_MIN`], awaited without blocking the
/// calling thread: the engine runs its legs on one thread, and a read held
/// there stalls every `join!`ed sibling (the GPU submits of the block whose
/// next weights are streaming idle the device for the read's duration).
async fn read_exact_at_split(
    file: &Arc<std::fs::File>,
    offset: u64,
    dst: &mut [u8],
) -> io::Result<()> {
    let part = if dst.len() < SPLIT_READ_MIN {
        dst.len()
    } else {
        dst.len().div_ceil(READ_PARTS)
    };
    let mut workers = Workers(Vec::new());
    let mut done = Vec::new();
    for (i, chunk) in dst.chunks_mut(part).enumerate() {
        let (tx, rx) = futures_channel::oneshot::channel();
        let (file, at) = (Arc::clone(file), offset + (i * part) as u64);
        let chunk = RawChunk(chunk.as_mut_ptr(), chunk.len());
        workers.0.push(std::thread::spawn(move || {
            // SAFETY: `dst` outlives this thread: `Workers` joins it before the
            // borrow ends, even when the future is dropped mid-read.
            let _ = tx.send(read_exact_at(&file, at, unsafe { chunk.slice() }));
        }));
        done.push(rx);
    }
    for rx in done {
        rx.await
            .map_err(|_| io::Error::other("read worker panicked"))??;
    }
    Ok(())
}

/// One worker's slice of the destination buffer.
struct RawChunk(*mut u8, usize);
// SAFETY: each worker gets a disjoint slice; `Workers` bounds its lifetime.
unsafe impl Send for RawChunk {}

impl RawChunk {
    /// # Safety
    /// The slice must still be live and written by no one else.
    unsafe fn slice<'a>(self) -> &'a mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.0, self.1) }
    }
}

/// Joins the read workers on drop, so none outlives the buffer it writes.
struct Workers(Vec<std::thread::JoinHandle<()>>);

impl Drop for Workers {
    fn drop(&mut self) {
        for w in self.0.drain(..) {
            let _ = w.join();
        }
    }
}

/// Fill `dst` from `file` at `offset` without touching a shared file cursor.
fn read_exact_at(file: &std::fs::File, mut offset: u64, mut dst: &mut [u8]) -> io::Result<()> {
    while !dst.is_empty() {
        #[cfg(windows)]
        let n = std::os::windows::fs::FileExt::seek_read(file, dst, offset)?;
        #[cfg(unix)]
        let n = std::os::unix::fs::FileExt::read_at(file, dst, offset)?;
        if n == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "read past file end",
            ));
        }
        offset += n as u64;
        dst = &mut dst[n..];
    }
    Ok(())
}

impl WeightReader for MmapFile {
    type Error = io::Error;
    fn len(&self) -> u64 {
        self.mmap.len() as u64
    }
    async fn read_at(&mut self, offset: u64, dst: &mut [u8]) -> Result<(), Self::Error> {
        let off = offset as usize;
        let end = off
            .checked_add(dst.len())
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "offset+len overflow"))?;
        if end > self.mmap.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "read past mmap end",
            ));
        }
        if dst.len() >= DIRECT_READ_MIN {
            return read_exact_at_split(&self.file, offset, dst).await;
        }
        dst.copy_from_slice(&self.mmap[off..end]);
        Ok(())
    }

    /// Real OS readahead. The streaming chunk loop hints `READ_PREFETCH_CHUNKS`
    /// ahead; starting the page-in now (batched large IOs) means `read_at`'s
    /// memcpy hits resident pages instead of fault-per-page at disk latency
    /// (measured: cold Gemma stream read_at ~1GB/s = 12.5s of a 15s encode).
    /// Best-effort: a failed or ignored hint only costs speed, never bytes.
    /// The OS call blocks while it issues the IO (~60 ms per 276 MB encoder
    /// layer), so a range of at least [`BACKGROUND_HINT_MIN`] is hinted from
    /// a detached thread that holds the mapping.
    fn will_read(&mut self, offset: u64, len: u64) {
        let off = offset as usize;
        let Some(end) = off.checked_add(len as usize) else {
            return;
        };
        if len == 0 || end > self.mmap.len() {
            return;
        }
        if (len as usize) < BACKGROUND_HINT_MIN {
            prefetch(&self.mmap, off, len as usize);
            return;
        }
        let mmap = Arc::clone(&self.mmap);
        // A hint that cannot get a thread is dropped, as any failed hint is.
        let _ = std::thread::Builder::new()
            .name("thinfer-prefetch".into())
            .spawn(move || prefetch(&mmap, off, len as usize));
    }
}

/// Hints at least this large page in from their own thread (see
/// [`MmapFile::will_read`]).
const BACKGROUND_HINT_MIN: usize = 16 << 20;

/// Start paging in `mmap[off..off + len]` (in bounds, checked by the caller).
fn prefetch(mmap: &memmap2::Mmap, off: usize, len: usize) {
    #[cfg(unix)]
    {
        let _ = mmap.advise_range(memmap2::Advice::WillNeed, off, len);
    }
    #[cfg(windows)]
    // SAFETY: the range is inside the live mapping (bounds-checked by the
    // caller); PrefetchVirtualMemory only initiates paging, it does not mutate.
    unsafe {
        use windows_sys::Win32::System::Memory::{PrefetchVirtualMemory, WIN32_MEMORY_RANGE_ENTRY};
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        let entry = WIN32_MEMORY_RANGE_ENTRY {
            VirtualAddress: mmap.as_ptr().add(off) as *mut core::ffi::c_void,
            NumberOfBytes: len,
        };
        let _ = PrefetchVirtualMemory(GetCurrentProcess(), 1, &entry, 0);
    }
}

/// Sequential read-bandwidth bench over the real weight-read paths. Gated on
/// `THINFER_READ_BENCH_PATH` (skips when unset) so it never runs in CI.
/// Reports mmap-path cold/warm and buffered `File::read` warm, separating
/// "disk is slow" from "the mmap fault/copy path is slow" from "page cache
/// does not retain between passes".
#[cfg(test)]
mod read_bench {
    use super::*;
    use std::time::Instant;
    use thinfer_core::weight::{FileOpener, WeightReader};

    const CHUNK: usize = 32 << 20;

    fn report(label: &str, bytes: u64, secs: f64) {
        eprintln!(
            "read_bench: {label}: {:.0} MB/s ({:.2} GiB in {:.2}s)",
            bytes as f64 / 1e6 / secs,
            bytes as f64 / (1u64 << 30) as f64,
            secs
        );
    }

    #[test]
    fn read_bench() {
        let Some(path) = std::env::var_os("THINFER_READ_BENCH_PATH") else {
            eprintln!("read_bench: THINFER_READ_BENCH_PATH unset; skipping");
            return;
        };
        let mut buf = vec![0u8; CHUNK];
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        rt.block_on(async {
            let opener = MmapFileOpener::new(&path).await.unwrap();
            let mut reader = opener.open().await.unwrap();
            let len = reader.len();
            for pass in ["mmap pass 1 (cold-ish)", "mmap pass 2 (warm)"] {
                let t = Instant::now();
                let mut off = 0u64;
                while off < len {
                    let n = CHUNK.min((len - off) as usize);
                    reader.read_at(off, &mut buf[..n]).await.unwrap();
                    std::hint::black_box(&buf);
                    off += n as u64;
                }
                report(pass, len, t.elapsed().as_secs_f64());
            }
        });
        // Buffered ReadFile path (no mmap) on now-warm pages, for comparison.
        use std::io::Read as _;
        let mut f = std::fs::File::open(&path).unwrap();
        let len = f.metadata().unwrap().len();
        let t = Instant::now();
        let mut off = 0u64;
        while off < len {
            let n = CHUNK.min((len - off) as usize);
            f.read_exact(&mut buf[..n]).unwrap();
            std::hint::black_box(&buf);
            off += n as u64;
        }
        report("File::read (warm)", len, t.elapsed().as_secs_f64());
    }
}

/// Smoke-test the `.pt` reader against a real `torch.save` checkpoint (e.g.
/// LongLive `model_bf16.pt`). Gated on `THINFER_PT_PATH` (skips when unset).
/// Parses the ZIP64 directory + pickle index, then prints every tensor name,
/// shape and dtype - both proof the parser handles a 10GB real file and the
/// source of truth for the DiT rename map.
#[cfg(test)]
mod pt_smoke {
    use super::*;
    use thinfer_core::format::pytorch::PytorchSource;
    use thinfer_core::weight::WeightSource;

    #[test]
    fn dump_pt_catalog() {
        let Some(path) = std::env::var_os("THINFER_PT_PATH") else {
            eprintln!("pt_smoke: THINFER_PT_PATH unset; skipping");
            return;
        };
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        rt.block_on(async {
            let opener = MmapFileOpener::new(&path).await.unwrap();
            let src = PytorchSource::open(opener).await.expect("parse .pt");
            let mut names: Vec<_> = src
                .catalog()
                .entries
                .iter()
                .map(|(id, e)| {
                    (
                        id.0.clone(),
                        format!("{:?}", e.shape.0),
                        e.encoding_label.clone(),
                        e.size,
                    )
                })
                .collect();
            names.sort();
            let total: u64 = names.iter().map(|(_, _, _, sz)| *sz).sum();
            eprintln!("pt_smoke: {} tensors, {} bytes total", names.len(), total);
            for (n, shape, dt, sz) in &names {
                eprintln!("pt_smoke: {n}\t{dt}\t{shape}\t{sz}");
            }
        });
    }
}
