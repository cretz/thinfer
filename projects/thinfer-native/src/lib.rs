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
