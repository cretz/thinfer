// thinfer-native: desktop/native impls of thinfer-core IO traits.

pub mod cache;
pub mod tokenizer;

use std::io;
use std::path::Path;
use std::sync::Arc;
use thinfer_core::weight::{FileOpener, WeightReader};

/// `FileOpener` backed by a single mmap'd view of the file. Weight bytes live
/// in the OS page cache, not our process heap: RSS scales with working set,
/// not model size. `open()` clones the `Arc<Mmap>` and is effectively free.
/// `read_at` is a memcpy from the mapped region.
pub struct MmapFileOpener {
    mmap: Arc<memmap2::Mmap>,
}

impl MmapFileOpener {
    pub async fn new(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let mmap = tokio::task::spawn_blocking(move || {
            let file = std::fs::File::open(&path)?;
            // SAFETY: We hold no claim that the underlying file is immutable
            // for the lifetime of the mmap; concurrent external truncate or
            // rewrite is UB. Same assumption as safetensors's own mmap loaders
            // and pytorch's safetensors mmap path. Out of scope to defend.
            unsafe { memmap2::Mmap::map(&file) }
        })
        .await
        .expect("mmap blocking task panicked")?;
        Ok(Self {
            mmap: Arc::new(mmap),
        })
    }
}

impl FileOpener for MmapFileOpener {
    type Reader = MmapFile;
    type Error = io::Error;
    async fn open(&self) -> Result<Self::Reader, Self::Error> {
        Ok(MmapFile {
            mmap: Arc::clone(&self.mmap),
        })
    }
}

pub struct MmapFile {
    mmap: Arc<memmap2::Mmap>,
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
        dst.copy_from_slice(&self.mmap[off..end]);
        Ok(())
    }
}
