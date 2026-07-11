use crate::tensor::{Shape, StorageEncoding};
use core::future::Future;
use half::{bf16, f16};
use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct WeightId(pub String);

#[derive(Clone, Debug)]
pub struct WeightEntry {
    pub offset: u64,
    pub size: u64,
    /// `None` when the on-wire dtype isn't one we map to a `StorageEncoding`
    /// (sanity-check surfaces these as failures using `encoding_label`).
    pub encoding: Option<StorageEncoding>,
    /// On-wire dtype name from the source format (e.g. "F32", "BF16", "I64").
    /// Carried so reports and error messages can reference what the file
    /// actually said, regardless of whether we mapped it.
    pub encoding_label: String,
    pub shape: Shape,
}

pub struct WeightCatalog {
    pub entries: HashMap<WeightId, WeightEntry>,
}

impl WeightCatalog {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }
    pub fn get(&self, id: &WeightId) -> Option<&WeightEntry> {
        self.entries.get(id)
    }
}

impl Default for WeightCatalog {
    fn default() -> Self {
        Self::new()
    }
}

/// Async random-access reader over a single tensor's storage. Caller drives
/// offsets and sizes; every backing tier we care about (mmap, pread on
/// Windows/Unix, OPFS sync handle) is random-access, and an async future
/// covers the cases where I/O genuinely is async (real disk on tokio,
/// non-OPFS web sources later).
///
/// `&mut self` on `read_at` enforces atomic-read-per-reader at the type
/// level — no lock needed; seek+read pairs serialize via the borrow checker.
///
/// Native-flavored: `&mut [u8]` lands bytes in caller-owned memory. The web
/// weight-upload path bypasses this trait per the no-weight-bytes-in-WASM-
/// linear-memory rule; native and web converge at the GPU buffer destination,
/// not at the CPU-bytes layer.
pub trait WeightReader {
    type Error: core::fmt::Debug;
    fn len(&self) -> u64;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn read_at(
        &mut self,
        offset: u64,
        dst: &mut [u8],
    ) -> impl Future<Output = Result<(), Self::Error>>;

    /// Prefetch hint: the next `read_at` will be exactly `(offset, len)`.
    /// Chunked consumers (`stream_source_to_gpu`) call this before doing
    /// CPU/GPU work on the current chunk so a reader with async IO (web
    /// OPFS) can overlap the next read with that work. Default no-op;
    /// readers that don't benefit (mmap memcpy) ignore it. Wrong hints are
    /// a perf bug, never a correctness one: `read_at` must validate any
    /// prefetched bytes against its actual arguments.
    fn will_read(&mut self, _offset: u64, _len: u64) {}
}

pub trait WeightSource {
    type Reader: WeightReader;
    type Error: core::fmt::Debug;

    fn catalog(&self) -> &WeightCatalog;

    fn open(&self, id: &WeightId) -> impl Future<Output = Result<Self::Reader, Self::Error>>;
}

/// Opens a fresh `WeightReader` over a whole file. Each `open()` returns an
/// independent reader; per-reader atomic seek+read is enforced by the
/// `&mut self` on `WeightReader::read_at`. Format-aware sources
/// (`SafetensorsSource`, future GGUF, etc) compose with an opener to get
/// per-tensor readers without holding a shared file handle.
pub trait FileOpener {
    type Reader: WeightReader;
    type Error: core::fmt::Debug;
    fn open(&self) -> impl Future<Output = Result<Self::Reader, Self::Error>>;
}

/// Restricts a whole-file reader to the byte range `[base, base + len)` and
/// rebases offsets so callers see `[0, len)`. Used by format sources to hand
/// out per-tensor readers backed by a whole-file reader.
pub struct OffsetView<R: WeightReader> {
    inner: R,
    base: u64,
    len: u64,
}

impl<R: WeightReader> OffsetView<R> {
    pub fn new(inner: R, base: u64, len: u64) -> Self {
        Self { inner, base, len }
    }
}

impl<R: WeightReader> WeightReader for OffsetView<R> {
    type Error = R::Error;
    fn len(&self) -> u64 {
        self.len
    }
    async fn read_at(&mut self, offset: u64, dst: &mut [u8]) -> Result<(), Self::Error> {
        debug_assert!(offset + dst.len() as u64 <= self.len);
        self.inner.read_at(self.base + offset, dst).await
    }
    fn will_read(&mut self, offset: u64, len: u64) {
        self.inner.will_read(self.base + offset, len);
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum DecodeError {
    /// fp16 storage is intentionally unsupported for Z-Image M1; reachable only
    /// if a future model registers an fp16-stored tensor before the kernels
    /// gain fp16 compute support.
    UnsupportedEncoding(StorageEncoding),
    /// Source ended mid-element (odd trailing byte for bf16).
    TrailingByte,
    /// `dst` not large enough for the decoded bytes from `src`.
    DstTooSmall,
}

/// Streaming storage->fp32 decoder. Mirrors `sanity::ValidateState`'s shape
/// but writes decoded bytes into caller-owned memory instead of xor'ing into
/// a sink. Supports the F32 / Bf16 / F16 on-wire dtypes (F16 added for the
/// GGUF VAEs, whose conv weights ship fp16); both 2-byte forms expand to fp32
/// host-side, then the upload path converts to the device weight dtype.
///
/// Use via successive `feed(src, &mut dst[written..])` calls; carry across
/// chunks is internal. Call `finish` at end-of-tensor to assert no half-pair
/// remains.
pub struct Decoder {
    encoding: StorageEncoding,
    /// Leftover odd byte of a 2-byte element (Bf16 or F16) across chunks.
    carry: Option<u8>,
}

impl Decoder {
    pub fn new(encoding: StorageEncoding) -> Result<Self, DecodeError> {
        match encoding {
            StorageEncoding::F32 | StorageEncoding::Bf16 | StorageEncoding::F16 => Ok(Self {
                encoding,
                carry: None,
            }),
            enc => Err(DecodeError::UnsupportedEncoding(enc)),
        }
    }

    /// Consume `src` and write decoded bytes into `dst` starting at index 0.
    /// Returns the number of `dst` bytes written; caller advances its own
    /// cursor. May write 0 if `src` is a single carried bf16 byte.
    pub fn feed(&mut self, src: &[u8], dst: &mut [u8]) -> Result<usize, DecodeError> {
        match self.encoding {
            StorageEncoding::F32 => {
                if dst.len() < src.len() {
                    return Err(DecodeError::DstTooSmall);
                }
                dst[..src.len()].copy_from_slice(src);
                Ok(src.len())
            }
            StorageEncoding::Bf16 => self.feed_pairs(src, dst, write_bf16_pair),
            StorageEncoding::F16 => self.feed_pairs(src, dst, write_f16_pair),
            _ => unreachable!("rejected in new()"),
        }
    }

    /// Expand little-endian 2-byte elements to fp32 via `write`, carrying any
    /// odd trailing byte to the next chunk. Shared by Bf16 and F16.
    fn feed_pairs(
        &mut self,
        src: &[u8],
        dst: &mut [u8],
        write: fn(u8, u8, &mut [u8]),
    ) -> Result<usize, DecodeError> {
        let mut written = 0;
        let mut start = 0;
        if let Some(c) = self.carry.take() {
            if let Some(&b) = src.first() {
                if dst.len() < 4 {
                    return Err(DecodeError::DstTooSmall);
                }
                write(c, b, &mut dst[..4]);
                written += 4;
                start = 1;
            } else {
                self.carry = Some(c);
                return Ok(0);
            }
        }
        let body = &src[start..];
        let pairs = body.chunks_exact(2);
        let remainder = pairs.remainder();
        let needed = pairs.len() * 4;
        if dst.len() < written + needed {
            return Err(DecodeError::DstTooSmall);
        }
        for pair in pairs {
            write(pair[0], pair[1], &mut dst[written..written + 4]);
            written += 4;
        }
        if let Some(&b) = remainder.first() {
            self.carry = Some(b);
        }
        Ok(written)
    }

    pub fn finish(self) -> Result<(), DecodeError> {
        if self.carry.is_some() {
            return Err(DecodeError::TrailingByte);
        }
        Ok(())
    }
}

fn write_bf16_pair(lo: u8, hi: u8, dst: &mut [u8]) {
    let f = bf16::from_bits(u16::from_le_bytes([lo, hi])).to_f32();
    dst[..4].copy_from_slice(&f.to_ne_bytes());
}

fn write_f16_pair(lo: u8, hi: u8, dst: &mut [u8]) {
    let f = f16::from_bits(u16::from_le_bytes([lo, hi])).to_f32();
    dst[..4].copy_from_slice(&f.to_ne_bytes());
}

/// One-shot decode of a complete `src` buffer into `dst`. Convenience for
/// callers that already have all bytes in memory (tests, non-streaming
/// loaders). Streaming callers should drive `Decoder` directly.
pub fn decode_into(
    encoding: StorageEncoding,
    src: &[u8],
    dst: &mut [u8],
) -> Result<usize, DecodeError> {
    let mut d = Decoder::new(encoding)?;
    let n = d.feed(src, dst)?;
    d.finish()?;
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_passthrough() {
        let src: Vec<u8> = (0..32u8).collect();
        let mut dst = vec![0u8; 32];
        let n = decode_into(StorageEncoding::F32, &src, &mut dst).unwrap();
        assert_eq!(n, 32);
        assert_eq!(dst, src);
    }

    #[test]
    fn bf16_to_fp32_matches_half_crate() {
        // Three bf16 values: 1.0, -2.5, 0.0.
        let vals = [1.0f32, -2.5, 0.0];
        let mut src = Vec::with_capacity(6);
        for &v in &vals {
            src.extend_from_slice(&bf16::from_f32(v).to_bits().to_le_bytes());
        }
        let mut dst = vec![0u8; 12];
        let n = decode_into(StorageEncoding::Bf16, &src, &mut dst).unwrap();
        assert_eq!(n, 12);
        let got: Vec<f32> = dst
            .chunks_exact(4)
            .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(got, vec![1.0, -2.5, 0.0]);
    }

    #[test]
    fn bf16_streaming_carry_split() {
        let vals = [1.0f32, -2.5, 0.5, 0.25];
        let mut src = Vec::with_capacity(8);
        for &v in &vals {
            src.extend_from_slice(&bf16::from_f32(v).to_bits().to_le_bytes());
        }
        // Split src at every odd offset to force the carry path.
        for split in [1usize, 3, 5, 7] {
            let mut d = Decoder::new(StorageEncoding::Bf16).unwrap();
            let mut dst = [0u8; 16];
            let mut written = 0;
            written += d.feed(&src[..split], &mut dst[written..]).unwrap();
            written += d.feed(&src[split..], &mut dst[written..]).unwrap();
            d.finish().unwrap();
            assert_eq!(written, 16, "split={split}");
            let got: Vec<f32> = dst
                .chunks_exact(4)
                .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            assert_eq!(got, vec![1.0, -2.5, 0.5, 0.25], "split={split}");
        }
    }

    #[test]
    fn bf16_trailing_byte_errors() {
        let mut d = Decoder::new(StorageEncoding::Bf16).unwrap();
        let mut dst = [0u8; 4];
        let _ = d.feed(&[0x12], &mut dst).unwrap();
        assert_eq!(d.finish(), Err(DecodeError::TrailingByte));
    }

    #[test]
    fn f16_to_fp32_matches_half_crate() {
        // f16 values incl. one (65504) that overflows bf16's range to prove the
        // decoder reads the value as true fp16, not bf16.
        let vals = [1.0f32, -2.5, 0.0, 65504.0];
        let mut src = Vec::with_capacity(8);
        for &v in &vals {
            src.extend_from_slice(&f16::from_f32(v).to_bits().to_le_bytes());
        }
        let mut dst = vec![0u8; 16];
        let n = decode_into(StorageEncoding::F16, &src, &mut dst).unwrap();
        assert_eq!(n, 16);
        let got: Vec<f32> = dst
            .chunks_exact(4)
            .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(got, vec![1.0, -2.5, 0.0, 65504.0]);
    }

    #[test]
    fn f16_streaming_carry_split() {
        let vals = [1.0f32, -2.5, 0.5, 0.25];
        let mut src = Vec::with_capacity(8);
        for &v in &vals {
            src.extend_from_slice(&f16::from_f32(v).to_bits().to_le_bytes());
        }
        for split in [1usize, 3, 5, 7] {
            let mut d = Decoder::new(StorageEncoding::F16).unwrap();
            let mut dst = [0u8; 16];
            let mut written = 0;
            written += d.feed(&src[..split], &mut dst[written..]).unwrap();
            written += d.feed(&src[split..], &mut dst[written..]).unwrap();
            d.finish().unwrap();
            assert_eq!(written, 16, "split={split}");
            let got: Vec<f32> = dst
                .chunks_exact(4)
                .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            assert_eq!(got, vec![1.0, -2.5, 0.5, 0.25], "split={split}");
        }
    }

    #[test]
    fn dst_too_small() {
        let mut d = Decoder::new(StorageEncoding::Bf16).unwrap();
        let src = [0x00, 0x3f, 0x00, 0x40];
        let mut dst = [0u8; 4];
        assert_eq!(d.feed(&src, &mut dst), Err(DecodeError::DstTooSmall));
    }
}
