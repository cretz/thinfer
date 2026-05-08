use crate::tensor::{Shape, StorageEncoding};
use core::future::Future;
use half::bf16;
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
/// a sink. fp16 is rejected at construction time: Z-Image's bf16 training
/// makes the fp16 magnitude clamp unsafe, and we have no other fp16 model
/// in M1.
///
/// Use via successive `feed(src, &mut dst[written..])` calls; carry across
/// chunks is internal. Call `finish` at end-of-tensor to assert no half-pair
/// remains.
pub struct Decoder {
    encoding: StorageEncoding,
    bf16_carry: Option<u8>,
}

impl Decoder {
    pub fn new(encoding: StorageEncoding) -> Result<Self, DecodeError> {
        match encoding {
            StorageEncoding::F32 | StorageEncoding::Bf16 => Ok(Self {
                encoding,
                bf16_carry: None,
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
            StorageEncoding::Bf16 => self.feed_bf16(src, dst),
            _ => unreachable!("rejected in new()"),
        }
    }

    fn feed_bf16(&mut self, src: &[u8], dst: &mut [u8]) -> Result<usize, DecodeError> {
        let mut written = 0;
        let mut start = 0;
        if let Some(c) = self.bf16_carry.take() {
            if let Some(&b) = src.first() {
                if dst.len() < 4 {
                    return Err(DecodeError::DstTooSmall);
                }
                write_bf16_pair(c, b, &mut dst[..4]);
                written += 4;
                start = 1;
            } else {
                self.bf16_carry = Some(c);
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
            write_bf16_pair(pair[0], pair[1], &mut dst[written..written + 4]);
            written += 4;
        }
        if let Some(&b) = remainder.first() {
            self.bf16_carry = Some(b);
        }
        Ok(written)
    }

    pub fn finish(self) -> Result<(), DecodeError> {
        if self.bf16_carry.is_some() {
            return Err(DecodeError::TrailingByte);
        }
        Ok(())
    }
}

fn write_bf16_pair(lo: u8, hi: u8, dst: &mut [u8]) {
    let f = bf16::from_bits(u16::from_le_bytes([lo, hi])).to_f32();
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
    fn rejects_fp16() {
        assert!(matches!(
            Decoder::new(StorageEncoding::F16),
            Err(DecodeError::UnsupportedEncoding(StorageEncoding::F16))
        ));
    }

    #[test]
    fn dst_too_small() {
        let mut d = Decoder::new(StorageEncoding::Bf16).unwrap();
        let src = [0x00, 0x3f, 0x00, 0x40];
        let mut dst = [0u8; 4];
        assert_eq!(d.feed(&src, &mut dst), Err(DecodeError::DstTooSmall));
    }
}
