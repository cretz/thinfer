//! GGUF v3 header parser. Pure: bytes in, catalog out. Dependency-free.
//!
//! Layout (little-endian throughout):
//!
//! ```text
//! magic: u32 = "GGUF" (0x46554747)
//! version: u32 = 3
//! n_tensors: u64
//! n_kv: u64
//! metadata: KV[n_kv]              -- key (string), type (u32), value (typed)
//! tensor_infos: TensorInfo[n_tensors]
//!   name (string), n_dims (u32), dims (u64[n_dims]), ggml_type (u32),
//!   offset (u64, from start of tensor_data)
//! padding: pad to `alignment` (default 32; metadata key `general.alignment`)
//! tensor_data: bytes
//! ```
//!
//! Strings: `u64 len` + `len bytes` (UTF-8, not null-terminated).
//!
//! KV value types (subset relevant for catalog construction):
//!
//! | code | meaning  |
//! |------|----------|
//! | 0    | u8       |
//! | 1    | i8       |
//! | 2    | u16      |
//! | 3    | i16      |
//! | 4    | u32      |
//! | 5    | i32      |
//! | 6    | f32      |
//! | 7    | bool (u8)|
//! | 8    | string   |
//! | 9    | array    |
//! | 10   | u64      |
//! | 11   | i64      |
//! | 12   | f64      |
//!
//! Streaming pattern: the parser tracks a cursor and returns
//! [`ParseError::NeedMore`] with the absolute byte offset it needed to make
//! progress. The source loop reads more bytes and retries until parse
//! succeeds. No whole-file load required.

use crate::quant::QuantKind;
use crate::tensor::{Shape, StorageEncoding};
use crate::weight::{
    FileOpener, OffsetView, WeightCatalog, WeightEntry, WeightId, WeightReader, WeightSource,
};
use std::collections::HashMap;

pub const MAGIC: u32 = 0x4655_4747; // "GGUF" little-endian
pub const VERSION: u32 = 3;
pub const DEFAULT_ALIGNMENT: u64 = 32;

#[derive(Debug)]
pub enum ParseError {
    /// Buffer too short to satisfy a field read. `at` is the byte offset
    /// the parser needed to read up to. Caller reads more bytes (up to
    /// at least `at`) and retries.
    NeedMore {
        at: u64,
    },
    BadMagic(u32),
    UnsupportedVersion(u32),
    /// String length / array length / metadata count / tensor count beyond
    /// the configured guardrail. Cheap corruption check.
    Oversized {
        field: &'static str,
        got: u64,
        max: u64,
    },
    BadValueType(u32),
    /// Bool byte was neither 0 nor 1.
    BadBool(u8),
    /// String not valid UTF-8.
    BadUtf8,
    /// `alignment` metadata value is zero or not a power of two.
    BadAlignment(u64),
    /// Tensor dimension count outside [1, 8].
    BadDimCount(u32),
    /// `general.alignment` exceeded `MAX_ALIGNMENT`.
    AlignmentTooLarge(u64),
}

const MAX_STRING_BYTES: u64 = 16 * 1024 * 1024; // 16 MiB
const MAX_KV_COUNT: u64 = 1_000_000;
const MAX_TENSORS: u64 = 1_000_000;
const MAX_ARRAY_LEN: u64 = 64 * 1024 * 1024;
const MAX_ALIGNMENT: u64 = 64 * 1024;

/// Header parse output. Per-tensor info is keyed by name.
#[derive(Debug, Clone)]
pub struct GgufHeader {
    pub version: u32,
    pub alignment: u64,
    /// Byte offset (from start of file) where tensor data begins, AFTER
    /// the post-info alignment pad.
    pub tensor_data_offset: u64,
    /// Number of header bytes consumed (== `tensor_data_offset`).
    pub header_bytes: u64,
    pub tensors: Vec<GgufTensorInfo>,
}

#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    pub name: String,
    pub shape: Shape,
    pub ggml_type: u32,
    /// Offset from start of tensor_data (i.e., absolute file offset =
    /// `header.tensor_data_offset + offset_in_tensor_data`).
    pub offset_in_tensor_data: u64,
    /// Mapped storage encoding. `None` if ggml_type is unrecognized.
    pub encoding: Option<StorageEncoding>,
    /// On-disk byte count derived from `shape` + `encoding`. Quant
    /// tensors use [`QuantKind::bytes_for_elements`]; floats use
    /// `bytes_per_elem * elements`. `None` for unmapped encodings.
    pub on_disk_bytes: Option<u64>,
}

/// One-shot header parse over a buffer that starts at the beginning of
/// the GGUF file. Returns `NeedMore { at }` if `buf.len() < at`. Caller
/// reads up to at least `at` bytes and retries.
pub fn parse_header(buf: &[u8]) -> Result<GgufHeader, ParseError> {
    let mut c = Cursor::new(buf);
    let magic = c.read_u32()?;
    if magic != MAGIC {
        return Err(ParseError::BadMagic(magic));
    }
    let version = c.read_u32()?;
    if version != VERSION {
        return Err(ParseError::UnsupportedVersion(version));
    }
    let n_tensors = c.read_u64()?;
    if n_tensors > MAX_TENSORS {
        return Err(ParseError::Oversized {
            field: "n_tensors",
            got: n_tensors,
            max: MAX_TENSORS,
        });
    }
    let n_kv = c.read_u64()?;
    if n_kv > MAX_KV_COUNT {
        return Err(ParseError::Oversized {
            field: "n_kv",
            got: n_kv,
            max: MAX_KV_COUNT,
        });
    }

    // Parse metadata KV pairs. The only key we read by name is
    // `general.alignment` (u32 per spec); everything else is skipped
    // without retaining the value, but we still have to walk the bytes.
    let mut alignment: u64 = DEFAULT_ALIGNMENT;
    for _ in 0..n_kv {
        let key = c.read_string()?;
        let vtype = c.read_u32()?;
        let v = parse_value(&mut c, vtype)?;
        if let Some(parsed) = v
            && key == "general.alignment"
        {
            if parsed == 0 || !parsed.is_power_of_two() {
                return Err(ParseError::BadAlignment(parsed));
            }
            if parsed > MAX_ALIGNMENT {
                return Err(ParseError::AlignmentTooLarge(parsed));
            }
            alignment = parsed;
        }
    }

    let mut tensors = Vec::with_capacity(n_tensors as usize);
    for _ in 0..n_tensors {
        let name = c.read_string()?;
        let n_dims = c.read_u32()?;
        if !(1..=8).contains(&n_dims) {
            return Err(ParseError::BadDimCount(n_dims));
        }
        let mut dims = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            dims.push(c.read_u64()? as usize);
        }
        let ggml_type = c.read_u32()?;
        let offset_in_tensor_data = c.read_u64()?;
        let encoding = encoding_from_ggml_type(ggml_type);
        let shape = Shape(dims);
        let on_disk_bytes = encoding.and_then(|e| bytes_for(&shape, e));
        tensors.push(GgufTensorInfo {
            name,
            shape,
            ggml_type,
            offset_in_tensor_data,
            encoding,
            on_disk_bytes,
        });
    }

    let post_info_pos = c.pos();
    let pad = (alignment - (post_info_pos % alignment)) % alignment;
    let tensor_data_offset = post_info_pos + pad;
    Ok(GgufHeader {
        version,
        alignment,
        tensor_data_offset,
        header_bytes: tensor_data_offset,
        tensors,
    })
}

/// Translate a `GgufHeader` into a `WeightCatalog`. Final tensor's `size`
/// uses `file_size` to bound it; other tensors derive size from the next
/// tensor's offset. Pass `None` for `file_size` if the caller doesn't have
/// it; the final tensor's `size` falls back to `on_disk_bytes`.
///
/// Shape convention: GGUF stores dims innermost-first (`ne[0]` = fastest
/// axis), the reverse of the PyTorch/safetensors outer-first order the rest
/// of the engine uses (`nn.Linear` weight `[N, K]`). The byte stream is the
/// same row-major data either way, so the catalog REVERSES dims to
/// outer-first: one shape convention engine-wide.
pub fn catalog_from(h: &GgufHeader, file_size: Option<u64>) -> WeightCatalog {
    // Tensors in GGUF are listed in arbitrary order in the header, but
    // their tensor_data offsets are monotonically increasing per the
    // spec (writer appends sequentially). We sort defensively rather
    // than rely on header order.
    let mut idx: Vec<usize> = (0..h.tensors.len()).collect();
    idx.sort_by_key(|&i| h.tensors[i].offset_in_tensor_data);
    let mut entries = HashMap::with_capacity(h.tensors.len());
    for k in 0..idx.len() {
        let i = idx[k];
        let t = &h.tensors[i];
        let abs_offset = h.tensor_data_offset + t.offset_in_tensor_data;
        let next_abs = if k + 1 < idx.len() {
            h.tensor_data_offset + h.tensors[idx[k + 1]].offset_in_tensor_data
        } else {
            file_size.unwrap_or(abs_offset + t.on_disk_bytes.unwrap_or(0))
        };
        let size = next_abs.saturating_sub(abs_offset);
        // Innermost-first GGUF dims -> outer-first engine convention.
        let shape = Shape(t.shape.0.iter().rev().copied().collect());
        entries.insert(
            WeightId(t.name.clone()),
            WeightEntry {
                offset: abs_offset,
                size,
                encoding: t.encoding,
                encoding_label: ggml_type_label(t.ggml_type).to_string(),
                shape,
            },
        );
    }
    WeightCatalog { entries }
}

fn bytes_for(shape: &Shape, enc: StorageEncoding) -> Option<u64> {
    let elements: u64 = shape.0.iter().map(|&d| d as u64).product();
    match enc {
        StorageEncoding::F32 => Some(elements * 4),
        StorageEncoding::F16 | StorageEncoding::Bf16 => Some(elements * 2),
        StorageEncoding::Quant(k) => Some(k.bytes_for_elements(elements)),
        StorageEncoding::I8 => Some(elements),
        StorageEncoding::I4 => Some(elements.div_ceil(2)),
    }
}

/// ggml_type -> StorageEncoding. Returns `None` for types we don't yet
/// surface (e.g. Q4_1, Q5_0, Q5_1, Q5_K, Q6_K, Q8_1, Q8_K, IQ*). The
/// catalog still records the raw type number + label so audits can
/// report it.
pub fn encoding_from_ggml_type(t: u32) -> Option<StorageEncoding> {
    match t {
        0 => Some(StorageEncoding::F32),
        1 => Some(StorageEncoding::F16),
        2 => Some(StorageEncoding::Quant(QuantKind::Q4_0)),
        8 => Some(StorageEncoding::Quant(QuantKind::Q8_0)),
        12 => Some(StorageEncoding::Quant(QuantKind::Q4_K)),
        13 => Some(StorageEncoding::Quant(QuantKind::Q5_K)),
        14 => Some(StorageEncoding::Quant(QuantKind::Q6_K)),
        30 => Some(StorageEncoding::Bf16),
        _ => None,
    }
}

/// Human-readable label for a ggml_type code. Mirrors llama.cpp's table
/// so audit logs are greppable against upstream.
pub fn ggml_type_label(t: u32) -> &'static str {
    match t {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        6 => "Q5_0",
        7 => "Q5_1",
        8 => "Q8_0",
        9 => "Q8_1",
        10 => "Q2_K",
        11 => "Q3_K",
        12 => "Q4_K",
        13 => "Q5_K",
        14 => "Q6_K",
        15 => "Q8_K",
        16 => "IQ2_XXS",
        17 => "IQ2_XS",
        18 => "IQ3_XXS",
        19 => "IQ1_S",
        20 => "IQ4_NL",
        21 => "IQ3_S",
        22 => "IQ2_S",
        23 => "IQ4_XS",
        24 => "I8",
        25 => "I16",
        26 => "I32",
        27 => "I64",
        28 => "F64",
        29 => "IQ1_M",
        30 => "BF16",
        _ => "UNKNOWN",
    }
}

// --- Internal cursor + value parsing ---

const VT_U8: u32 = 0;
const VT_I8: u32 = 1;
const VT_U16: u32 = 2;
const VT_I16: u32 = 3;
const VT_U32: u32 = 4;
const VT_I32: u32 = 5;
const VT_F32: u32 = 6;
const VT_BOOL: u32 = 7;
const VT_STRING: u32 = 8;
const VT_ARRAY: u32 = 9;
const VT_U64: u32 = 10;
const VT_I64: u32 = 11;
const VT_F64: u32 = 12;

/// Walk one typed KV value, returning its scalar interpretation as `u64`
/// when the type is integral — sufficient for `general.alignment` and
/// any future integer-typed key we care about. Returns `None` for
/// strings/floats/bools/arrays (still consumed from the cursor, just
/// not surfaced). Recurses through arrays without retaining elements.
fn parse_value(c: &mut Cursor<'_>, vt: u32) -> Result<Option<u64>, ParseError> {
    Ok(match vt {
        VT_U8 => Some(u64::from(c.read_u8()?)),
        VT_I8 => {
            let _ = c.read_u8()?;
            None
        }
        VT_U16 => Some(u64::from(c.read_u16()?)),
        VT_I16 => {
            let _ = c.read_u16()?;
            None
        }
        VT_U32 => Some(u64::from(c.read_u32()?)),
        VT_I32 => {
            let _ = c.read_u32()?;
            None
        }
        VT_F32 => {
            let _ = c.read_u32()?;
            None
        }
        VT_BOOL => {
            let b = c.read_u8()?;
            if b > 1 {
                return Err(ParseError::BadBool(b));
            }
            None
        }
        VT_STRING => {
            let _ = c.read_string()?;
            None
        }
        VT_ARRAY => {
            let inner = c.read_u32()?;
            let len = c.read_u64()?;
            if len > MAX_ARRAY_LEN {
                return Err(ParseError::Oversized {
                    field: "array_len",
                    got: len,
                    max: MAX_ARRAY_LEN,
                });
            }
            for _ in 0..len {
                let _ = parse_value(c, inner)?;
            }
            None
        }
        VT_U64 => Some(c.read_u64()?),
        VT_I64 => {
            let _ = c.read_u64()?;
            None
        }
        VT_F64 => {
            let _ = c.read_u64()?;
            None
        }
        other => return Err(ParseError::BadValueType(other)),
    })
}

struct Cursor<'a> {
    buf: &'a [u8],
    pos: u64,
}

impl<'a> Cursor<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }
    fn pos(&self) -> u64 {
        self.pos
    }
    fn ensure(&self, n: u64) -> Result<(), ParseError> {
        if (self.buf.len() as u64) < self.pos + n {
            return Err(ParseError::NeedMore { at: self.pos + n });
        }
        Ok(())
    }
    fn read_u8(&mut self) -> Result<u8, ParseError> {
        self.ensure(1)?;
        let v = self.buf[self.pos as usize];
        self.pos += 1;
        Ok(v)
    }
    fn read_u16(&mut self) -> Result<u16, ParseError> {
        self.ensure(2)?;
        let s = self.pos as usize;
        let v = u16::from_le_bytes([self.buf[s], self.buf[s + 1]]);
        self.pos += 2;
        Ok(v)
    }
    fn read_u32(&mut self) -> Result<u32, ParseError> {
        self.ensure(4)?;
        let s = self.pos as usize;
        let v = u32::from_le_bytes(self.buf[s..s + 4].try_into().unwrap());
        self.pos += 4;
        Ok(v)
    }
    fn read_u64(&mut self) -> Result<u64, ParseError> {
        self.ensure(8)?;
        let s = self.pos as usize;
        let v = u64::from_le_bytes(self.buf[s..s + 8].try_into().unwrap());
        self.pos += 8;
        Ok(v)
    }
    fn read_string(&mut self) -> Result<String, ParseError> {
        let len = self.read_u64()?;
        if len > MAX_STRING_BYTES {
            return Err(ParseError::Oversized {
                field: "string_len",
                got: len,
                max: MAX_STRING_BYTES,
            });
        }
        self.ensure(len)?;
        let s = self.pos as usize;
        let e = s + len as usize;
        let txt = core::str::from_utf8(&self.buf[s..e])
            .map_err(|_| ParseError::BadUtf8)?
            .to_string();
        self.pos += len;
        Ok(txt)
    }
}

// --- GgufSource ---

#[derive(Debug)]
pub enum SourceError<E: core::fmt::Debug> {
    Opener(E),
    Reader(String),
    Parse(ParseError),
    UnknownTensor(WeightId),
    /// Header didn't finish parsing within `MAX_HEADER_BYTES`. A real GGUF
    /// header is at most a few MiB; anything beyond suggests corruption.
    HeaderTooLarge(u64),
}

const MAX_HEADER_BYTES: u64 = 256 * 1024 * 1024;
const HEADER_READ_INITIAL: u64 = 1 << 20; // 1 MiB first read
const HEADER_READ_GROW: u64 = 1 << 22; // 4 MiB per retry

/// Iteratively reads from `opener` until the header parses. Yields a
/// `WeightSource` over per-tensor `OffsetView`s. Does not load tensor
/// data into memory.
pub struct GgufSource<F: FileOpener> {
    catalog: WeightCatalog,
    opener: F,
}

impl<F: FileOpener> GgufSource<F> {
    pub async fn open(opener: F) -> Result<Self, SourceError<F::Error>> {
        let mut reader = opener.open().await.map_err(SourceError::Opener)?;
        let total = reader.len();
        let mut want = HEADER_READ_INITIAL.min(total);
        let mut buf: Vec<u8> = vec![0u8; want as usize];
        reader
            .read_at(0, &mut buf)
            .await
            .map_err(|e| SourceError::Reader(format!("{e:?}")))?;
        let header = loop {
            match parse_header(&buf) {
                Ok(h) => break h,
                Err(ParseError::NeedMore { at }) => {
                    if at > total {
                        return Err(SourceError::Parse(ParseError::NeedMore { at }));
                    }
                    if at > MAX_HEADER_BYTES {
                        return Err(SourceError::HeaderTooLarge(at));
                    }
                    let next_want = (at + HEADER_READ_GROW).min(total);
                    if next_want <= want {
                        return Err(SourceError::Parse(ParseError::NeedMore { at }));
                    }
                    buf.resize(next_want as usize, 0);
                    reader
                        .read_at(want, &mut buf[want as usize..])
                        .await
                        .map_err(|e| SourceError::Reader(format!("{e:?}")))?;
                    want = next_want;
                }
                Err(e) => return Err(SourceError::Parse(e)),
            }
        };
        let catalog = catalog_from(&header, Some(total));
        Ok(Self { catalog, opener })
    }

    pub fn catalog(&self) -> &WeightCatalog {
        &self.catalog
    }
}

impl<F: FileOpener> WeightSource for GgufSource<F> {
    type Reader = OffsetView<F::Reader>;
    type Error = SourceError<F::Error>;

    fn catalog(&self) -> &WeightCatalog {
        &self.catalog
    }

    async fn open(&self, id: &WeightId) -> Result<Self::Reader, Self::Error> {
        let entry = self
            .catalog
            .get(id)
            .ok_or_else(|| SourceError::UnknownTensor(id.clone()))?;
        let inner = self.opener.open().await.map_err(SourceError::Opener)?;
        Ok(OffsetView::new(inner, entry.offset, entry.size))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a synthetic GGUF file: magic + version + n_tensors=2 + n_kv=1,
    /// one alignment KV, two Q8_0 tensors of [32] and [64] elements.
    fn build_minimal_gguf() -> (Vec<u8>, u64, u64) {
        let mut b = Vec::<u8>::new();
        b.extend_from_slice(&MAGIC.to_le_bytes());
        b.extend_from_slice(&VERSION.to_le_bytes());
        b.extend_from_slice(&2u64.to_le_bytes()); // n_tensors
        b.extend_from_slice(&1u64.to_le_bytes()); // n_kv

        // KV: general.alignment = u32(32)
        let key = "general.alignment";
        b.extend_from_slice(&(key.len() as u64).to_le_bytes());
        b.extend_from_slice(key.as_bytes());
        b.extend_from_slice(&VT_U32.to_le_bytes());
        b.extend_from_slice(&32u32.to_le_bytes());

        // Tensor 0: name="a", dims=[32], ggml_type=8 (Q8_0), offset=0
        let name = "a";
        b.extend_from_slice(&(name.len() as u64).to_le_bytes());
        b.extend_from_slice(name.as_bytes());
        b.extend_from_slice(&1u32.to_le_bytes()); // n_dims
        b.extend_from_slice(&32u64.to_le_bytes()); // dim 0
        b.extend_from_slice(&8u32.to_le_bytes()); // ggml_type = Q8_0
        b.extend_from_slice(&0u64.to_le_bytes()); // offset_in_tensor_data

        // Tensor 1: name="bb", GGUF dims=[32, 2] (innermost-first; engine
        // sees [2, 32]), ggml_type=8 (Q8_0), offset=34
        let name = "bb";
        b.extend_from_slice(&(name.len() as u64).to_le_bytes());
        b.extend_from_slice(name.as_bytes());
        b.extend_from_slice(&2u32.to_le_bytes());
        b.extend_from_slice(&32u64.to_le_bytes());
        b.extend_from_slice(&2u64.to_le_bytes());
        b.extend_from_slice(&8u32.to_le_bytes());
        b.extend_from_slice(&34u64.to_le_bytes());

        let pre_pad_len = b.len() as u64;
        let alignment = 32u64;
        let pad = (alignment - (pre_pad_len % alignment)) % alignment;
        b.resize((pre_pad_len + pad) as usize, 0);
        let data_offset = b.len() as u64;
        // Tensor 0 (34 bytes Q8_0) + Tensor 1 (68 bytes Q8_0)
        b.resize((data_offset + 34 + 68) as usize, 0);
        let total = b.len() as u64;
        (b, data_offset, total)
    }

    #[test]
    fn parse_minimal_header() {
        let (buf, expected_data_offset, _file_size) = build_minimal_gguf();
        let h = parse_header(&buf).unwrap();
        assert_eq!(h.version, VERSION);
        assert_eq!(h.alignment, 32);
        assert_eq!(h.tensor_data_offset, expected_data_offset);
        assert_eq!(h.tensors.len(), 2);
        let a = h.tensors.iter().find(|t| t.name == "a").unwrap();
        assert_eq!(a.shape.0, vec![32]);
        assert_eq!(a.encoding, Some(StorageEncoding::Quant(QuantKind::Q8_0)));
        assert_eq!(a.on_disk_bytes, Some(34));
        let bb = h.tensors.iter().find(|t| t.name == "bb").unwrap();
        // Header keeps raw GGUF (innermost-first) order; the catalog reverses.
        assert_eq!(bb.shape.0, vec![32, 2]);
        assert_eq!(bb.on_disk_bytes, Some(68));
    }

    #[test]
    fn truncated_buffer_yields_need_more() {
        let (buf, _, _) = build_minimal_gguf();
        // Truncate mid-header. Must return NeedMore with a usable target.
        for cut in [4, 16, 32, 64, 100] {
            if cut >= buf.len() {
                break;
            }
            match parse_header(&buf[..cut]) {
                Err(ParseError::NeedMore { at }) => {
                    assert!(at as usize > cut, "at={at} must exceed cut={cut}");
                }
                Ok(_) => panic!("unexpected success at cut={cut}"),
                Err(e) => panic!("unexpected error {e:?} at cut={cut}"),
            }
        }
    }

    #[test]
    fn bad_magic_rejected() {
        let mut buf = vec![0u8; 8];
        buf[..4].copy_from_slice(&0xDEAD_BEEFu32.to_le_bytes());
        buf[4..8].copy_from_slice(&VERSION.to_le_bytes());
        assert!(matches!(parse_header(&buf), Err(ParseError::BadMagic(_))));
    }

    #[test]
    fn unsupported_version_rejected() {
        let mut buf = vec![0u8; 8];
        buf[..4].copy_from_slice(&MAGIC.to_le_bytes());
        buf[4..8].copy_from_slice(&2u32.to_le_bytes());
        assert!(matches!(
            parse_header(&buf),
            Err(ParseError::UnsupportedVersion(2))
        ));
    }

    #[test]
    fn catalog_uses_next_offset_for_size() {
        let (buf, _, total) = build_minimal_gguf();
        let h = parse_header(&buf).unwrap();
        let cat = catalog_from(&h, Some(total));
        let a = cat.get(&WeightId("a".to_string())).unwrap();
        assert_eq!(a.size, 34);
        let bb = cat.get(&WeightId("bb".to_string())).unwrap();
        assert_eq!(bb.size, 68);
        // Catalog shape is outer-first (engine convention), reversed from
        // the GGUF header's innermost-first dims.
        assert_eq!(bb.shape.0, vec![2, 32]);
    }
}
