//! PyTorch `.pt` / `.pth` checkpoint reader (the `torch.save` zip format).
//!
//! A `torch.save` file with the default (new) zipfile serialization is a ZIP
//! archive holding `<prefix>/data.pkl` (a pickle of the state dict) plus one
//! `<prefix>/data/<key>` entry per tensor storage (raw little-endian bytes,
//! STORED/uncompressed, 64-byte aligned). We parse the central directory
//! (ZIP64-aware: a 10GB checkpoint exceeds the 4GB classic-EOCD limits), then
//! walk a restricted subset of the pickle opcodes to recover, for every
//! `_rebuild_tensor_v2` tensor, its storage key + dtype + shape. Each tensor
//! then maps to a contiguous byte range in the file, exposed as a
//! `WeightCatalog` so the residency/upload path consumes a `.pt` exactly like a
//! safetensors shard. Pure parsing over a `FileOpener`; no torch, no IO policy.
//!
//! Scope: inference checkpoints only. Tensors must be contiguous with storage
//! offset 0 (true for every saved weight); anything else is surfaced loudly
//! rather than silently mis-read.

use crate::tensor::{Shape, StorageEncoding};
use crate::weight::{
    FileOpener, OffsetView, WeightCatalog, WeightEntry, WeightId, WeightReader, WeightSource,
};
use std::collections::HashMap;

#[derive(Debug)]
pub enum PtError {
    Opener(String),
    Reader(String),
    /// File too small / signature missing / truncated record.
    BadZip(&'static str),
    /// A pickle opcode we don't implement, or a malformed operand.
    Pickle(String),
    /// A tensor whose storage layout we can't map to a flat byte range
    /// (non-contiguous, non-zero storage offset, or unknown dtype).
    UnsupportedTensor {
        name: String,
        reason: String,
    },
    /// The pickle referenced a storage key with no matching zip entry.
    MissingStorage {
        name: String,
        key: String,
    },
}

/// `WeightSource` over a single PyTorch `.pt` file. Mirrors
/// [`crate::format::safetensors::SafetensorsSource`]: a catalog plus an opener
/// that yields a fresh whole-file reader per tensor, scoped via `OffsetView`.
pub struct PytorchSource<F: FileOpener> {
    catalog: WeightCatalog,
    opener: F,
}

impl<F: FileOpener> PytorchSource<F> {
    pub async fn open(opener: F) -> Result<Self, PtError> {
        let mut reader = opener
            .open()
            .await
            .map_err(|e| PtError::Opener(format!("{e:?}")))?;
        let catalog = build_catalog(&mut reader).await?;
        Ok(Self { catalog, opener })
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.catalog.entries.keys().map(|k| k.0.as_str())
    }
}

impl<F: FileOpener> WeightSource for PytorchSource<F> {
    type Reader = OffsetView<F::Reader>;
    type Error = PtError;

    fn catalog(&self) -> &WeightCatalog {
        &self.catalog
    }

    async fn open(&self, id: &WeightId) -> Result<Self::Reader, Self::Error> {
        let entry = self
            .catalog
            .get(id)
            .ok_or_else(|| PtError::Pickle(format!("unknown tensor {}", id.0)))?;
        let inner = self
            .opener
            .open()
            .await
            .map_err(|e| PtError::Opener(format!("{e:?}")))?;
        Ok(OffsetView::new(inner, entry.offset, entry.size))
    }
}

// ---------------------------------------------------------------------------
// Catalog assembly: zip directory -> pickle index -> byte ranges.
// ---------------------------------------------------------------------------

async fn read_exact<R: WeightReader>(
    reader: &mut R,
    offset: u64,
    len: usize,
) -> Result<Vec<u8>, PtError> {
    let mut buf = vec![0u8; len];
    reader
        .read_at(offset, &mut buf)
        .await
        .map_err(|e| PtError::Reader(format!("{e:?}")))?;
    Ok(buf)
}

async fn build_catalog<R: WeightReader>(reader: &mut R) -> Result<WeightCatalog, PtError> {
    let file_len = reader.len();
    let dir = ZipDir::read(reader, file_len).await?;

    // Locate `<prefix>/data.pkl`; derive the archive prefix from its name.
    let (pkl_name, pkl_entry) = dir
        .entries
        .iter()
        .find(|(name, _)| name.ends_with("/data.pkl") || *name == "data.pkl")
        .ok_or(PtError::BadZip("no data.pkl entry"))?;
    let prefix = pkl_name
        .strip_suffix("data.pkl")
        .unwrap_or("")
        .trim_end_matches('/')
        .to_string();
    let storage_dir = if prefix.is_empty() {
        "data/".to_string()
    } else {
        format!("{prefix}/data/")
    };

    let pkl_offset = data_offset(reader, pkl_entry.local_header_offset).await?;
    let pkl_bytes = read_exact(reader, pkl_offset, pkl_entry.size as usize).await?;
    let tensors = parse_pickle(&pkl_bytes).map_err(PtError::Pickle)?;

    let mut entries = HashMap::with_capacity(tensors.len());
    for (name, spec) in tensors {
        let enc = match dtype_of(&spec.storage_dtype) {
            Some(e) => e,
            None => {
                // Non-weight leaf (e.g. an int64 buffer) we don't compute on:
                // surface the label but skip rather than fail the whole load.
                continue;
            }
        };
        let elem = elem_size(enc);
        let numel: usize = spec.shape.iter().product();
        if spec.storage_offset != 0 || !is_contiguous(&spec.shape, &spec.stride) {
            return Err(PtError::UnsupportedTensor {
                name,
                reason: format!(
                    "non-contiguous or offset!=0 (offset={}, shape={:?}, stride={:?})",
                    spec.storage_offset, spec.shape, spec.stride
                ),
            });
        }
        let key_name = format!("{storage_dir}{}", spec.storage_key);
        let storage = dir.entries.get(&key_name).ok_or_else(|| {
            // Some archives use just the bare key under data/; retry without
            // re-derivation already covered by storage_dir. Report loudly.
            PtError::MissingStorage {
                name: name.clone(),
                key: key_name.clone(),
            }
        })?;
        let blob_offset = data_offset(reader, storage.local_header_offset).await?;
        let want = (numel * elem) as u64;
        if storage.size < want {
            return Err(PtError::UnsupportedTensor {
                name,
                reason: format!("storage {key_name} is {} bytes, need {want}", storage.size),
            });
        }
        entries.insert(
            WeightId(name),
            WeightEntry {
                offset: blob_offset,
                size: want,
                encoding: Some(enc),
                encoding_label: spec.storage_dtype,
                shape: Shape(spec.shape),
            },
        );
    }
    Ok(WeightCatalog { entries })
}

fn dtype_of(storage_class: &str) -> Option<StorageEncoding> {
    // `torch.save` persistent ids name a typed storage class. Map the ones a
    // weight checkpoint actually uses; integer/double storages (positions,
    // counters) return None and are skipped by the caller.
    match storage_class {
        "FloatStorage" => Some(StorageEncoding::F32),
        "HalfStorage" => Some(StorageEncoding::F16),
        "BFloat16Storage" => Some(StorageEncoding::Bf16),
        _ => None,
    }
}

fn elem_size(enc: StorageEncoding) -> usize {
    match enc {
        StorageEncoding::F32 => 4,
        StorageEncoding::F16 | StorageEncoding::Bf16 => 2,
        StorageEncoding::I8 => 1,
        _ => 1,
    }
}

fn is_contiguous(shape: &[usize], stride: &[usize]) -> bool {
    if shape.len() != stride.len() {
        return false;
    }
    let mut expected = 1usize;
    for i in (0..shape.len()).rev() {
        // A length-1 axis may carry any stride; ignore it (torch does).
        if shape[i] != 1 && stride[i] != expected {
            return false;
        }
        expected *= shape[i];
    }
    true
}

/// Resolve a zip entry's local-header relative offset to the absolute file
/// offset of its data (skip the 30-byte fixed header + name + extra fields).
async fn data_offset<R: WeightReader>(
    reader: &mut R,
    local_header_offset: u64,
) -> Result<u64, PtError> {
    let hdr = read_exact(reader, local_header_offset, 30).await?;
    if rd_u32(&hdr, 0) != 0x0403_4b50 {
        return Err(PtError::BadZip("bad local file header signature"));
    }
    let name_len = rd_u16(&hdr, 26) as u64;
    let extra_len = rd_u16(&hdr, 28) as u64;
    Ok(local_header_offset + 30 + name_len + extra_len)
}

// ---------------------------------------------------------------------------
// Minimal ZIP central-directory reader (ZIP64-aware, STORED entries only).
// ---------------------------------------------------------------------------

struct ZipEntry {
    local_header_offset: u64,
    /// Uncompressed == compressed size (STORED). The tensor blob length.
    size: u64,
}

struct ZipDir {
    entries: HashMap<String, ZipEntry>,
}

impl ZipDir {
    async fn read<R: WeightReader>(reader: &mut R, file_len: u64) -> Result<Self, PtError> {
        // Scan the tail for the End-Of-Central-Directory signature. Torch zips
        // carry no archive comment, so EOCD is the final 22 bytes, but a tail
        // scan is robust to either.
        const EOCD_SIG: u32 = 0x0605_4b50;
        let tail_len = file_len.min(64 * 1024 + 22);
        let tail_start = file_len - tail_len;
        let tail = read_exact(reader, tail_start, tail_len as usize).await?;
        let eocd_rel = (0..=tail.len().saturating_sub(22))
            .rev()
            .find(|&i| rd_u32(&tail, i) == EOCD_SIG)
            .ok_or(PtError::BadZip("no EOCD record"))?;
        let eocd = &tail[eocd_rel..];

        let mut total_entries = rd_u16(eocd, 10) as u64;
        let mut cd_size = rd_u32(eocd, 12) as u64;
        let mut cd_offset = rd_u32(eocd, 16) as u64;

        // ZIP64: any field saturated to 0xFFFF.. means the real value lives in
        // the ZIP64 EOCD record, found via the locator just before the EOCD.
        let needs_zip64 =
            total_entries == 0xFFFF || cd_size == 0xFFFF_FFFF || cd_offset == 0xFFFF_FFFF;
        if needs_zip64 {
            let loc_rel = eocd_rel
                .checked_sub(20)
                .ok_or(PtError::BadZip("missing ZIP64 EOCD locator"))?;
            let loc = &tail[loc_rel..];
            if rd_u32(loc, 0) != 0x0706_4b50 {
                return Err(PtError::BadZip("bad ZIP64 EOCD locator signature"));
            }
            let z64_eocd_offset = rd_u64(loc, 8);
            let z64 = read_exact(reader, z64_eocd_offset, 56).await?;
            if rd_u32(&z64, 0) != 0x0606_4b50 {
                return Err(PtError::BadZip("bad ZIP64 EOCD record signature"));
            }
            total_entries = rd_u64(&z64, 32);
            cd_size = rd_u64(&z64, 40);
            cd_offset = rd_u64(&z64, 48);
        }

        let cd = read_exact(reader, cd_offset, cd_size as usize).await?;
        let mut entries = HashMap::with_capacity(total_entries as usize);
        let mut pos = 0usize;
        for _ in 0..total_entries {
            if pos + 46 > cd.len() || rd_u32(&cd, pos) != 0x0201_4b50 {
                return Err(PtError::BadZip("bad central directory header"));
            }
            let comp_size = rd_u32(&cd, pos + 20) as u64;
            let uncomp_size = rd_u32(&cd, pos + 24) as u64;
            let name_len = rd_u16(&cd, pos + 28) as usize;
            let extra_len = rd_u16(&cd, pos + 30) as usize;
            let comment_len = rd_u16(&cd, pos + 32) as usize;
            let local_off32 = rd_u32(&cd, pos + 42) as u64;
            let name_start = pos + 46;
            let name = String::from_utf8_lossy(&cd[name_start..name_start + name_len]).into_owned();
            let extra = &cd[name_start + name_len..name_start + name_len + extra_len];

            let mut size = uncomp_size.max(comp_size);
            let mut local_header_offset = local_off32;
            // Patch saturated fields from the ZIP64 extra block (id 0x0001).
            if uncomp_size == 0xFFFF_FFFF || comp_size == 0xFFFF_FFFF || local_off32 == 0xFFFF_FFFF
            {
                let z = find_zip64_extra(extra).ok_or(PtError::BadZip("missing ZIP64 extra"))?;
                let mut zp = 0usize;
                if uncomp_size == 0xFFFF_FFFF {
                    size = rd_u64(z, zp);
                    zp += 8;
                }
                if comp_size == 0xFFFF_FFFF {
                    size = size.max(rd_u64(z, zp));
                    zp += 8;
                }
                if local_off32 == 0xFFFF_FFFF {
                    local_header_offset = rd_u64(z, zp);
                }
            }
            entries.insert(
                name,
                ZipEntry {
                    local_header_offset,
                    size,
                },
            );
            pos = name_start + name_len + extra_len + comment_len;
        }
        Ok(Self { entries })
    }
}

/// Find the ZIP64 extended-information extra field (header id 0x0001) within an
/// extra-field blob, returning its payload slice.
fn find_zip64_extra(extra: &[u8]) -> Option<&[u8]> {
    let mut p = 0usize;
    while p + 4 <= extra.len() {
        let id = rd_u16(extra, p);
        let len = rd_u16(extra, p + 2) as usize;
        if id == 0x0001 {
            return extra.get(p + 4..p + 4 + len);
        }
        p += 4 + len;
    }
    None
}

#[inline]
fn rd_u16(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}
#[inline]
fn rd_u32(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}
#[inline]
fn rd_u64(b: &[u8], o: usize) -> u64 {
    u64::from_le_bytes([
        b[o],
        b[o + 1],
        b[o + 2],
        b[o + 3],
        b[o + 4],
        b[o + 5],
        b[o + 6],
        b[o + 7],
    ])
}

// ---------------------------------------------------------------------------
// Restricted pickle interpreter: just enough to walk a torch state dict.
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct TensorSpec {
    storage_key: String,
    storage_dtype: String,
    storage_offset: usize,
    shape: Vec<usize>,
    stride: Vec<usize>,
}

#[derive(Clone, Debug)]
enum Val {
    None,
    Int(i64),
    Str(String),
    Tuple(Vec<Val>),
    List(Vec<Val>),
    Dict(Vec<(Val, Val)>),
    /// A `module.name` global; only the trailing `name` distinguishes the
    /// callables and storage classes we care about, so the module is dropped.
    Global(String),
    /// Result of `persistent_load`: a typed storage handle (class, key).
    Storage {
        dtype: String,
        key: String,
    },
    Tensor(TensorSpec),
    Mark,
}

/// Walk the pickle and return every tensor reachable from the top-level mapping,
/// keyed by its dotted state-dict path. Non-tensor leaves are ignored.
fn parse_pickle(buf: &[u8]) -> Result<Vec<(String, TensorSpec)>, String> {
    let root = Unpickler::new(buf).run()?;
    let mut out = Vec::new();
    flatten(&root, String::new(), &mut out);
    if out.is_empty() {
        return Err("pickle produced no tensors".to_string());
    }
    Ok(out)
}

fn flatten(val: &Val, prefix: String, out: &mut Vec<(String, TensorSpec)>) {
    match val {
        Val::Dict(items) => {
            for (k, v) in items {
                if let Val::Str(key) = k {
                    let path = if prefix.is_empty() {
                        key.clone()
                    } else {
                        format!("{prefix}.{key}")
                    };
                    flatten(v, path, out);
                }
            }
        }
        Val::Tensor(spec) => out.push((prefix, spec.clone())),
        _ => {}
    }
}

struct Unpickler<'a> {
    buf: &'a [u8],
    pos: usize,
    stack: Vec<Val>,
    memo: Vec<Val>,
}

impl<'a> Unpickler<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self {
            buf,
            pos: 0,
            stack: Vec::new(),
            memo: Vec::new(),
        }
    }

    fn run(mut self) -> Result<Val, String> {
        loop {
            let op = self.u8()?;
            match op {
                0x80 => {
                    self.u8()?; // PROTO version
                }
                0x95 => {
                    self.skip(8)?; // FRAME length
                }
                b'(' => self.stack.push(Val::Mark), // MARK
                b'.' => break,                      // STOP
                b'0' => {
                    self.stack.pop(); // POP
                }
                b'1' => self.pop_to_mark().map(|_| ())?, // POP_MARK
                b'N' => self.stack.push(Val::None),      // NONE
                0x88 => self.stack.push(Val::None),      // NEWTRUE
                0x89 => self.stack.push(Val::None),      // NEWFALSE
                b'K' => {
                    let v = self.u8()? as i64;
                    self.stack.push(Val::Int(v)); // BININT1
                }
                b'M' => {
                    let v = self.u16()? as i64;
                    self.stack.push(Val::Int(v)); // BININT2
                }
                b'J' => {
                    let v = self.u32()? as i32 as i64;
                    self.stack.push(Val::Int(v)); // BININT
                }
                0x8a => {
                    let n = self.u8()? as usize; // LONG1
                    let v = self.read_long(n)?;
                    self.stack.push(Val::Int(v));
                }
                0x8b => {
                    let n = self.u32()? as usize; // LONG4
                    let v = self.read_long(n)?;
                    self.stack.push(Val::Int(v));
                }
                b'G' => {
                    self.skip(8)?; // BINFLOAT (unused operand)
                    self.stack.push(Val::None);
                }
                0x8c => {
                    let n = self.u8()? as usize; // SHORT_BINUNICODE
                    let s = self.read_str(n)?;
                    self.stack.push(Val::Str(s));
                }
                b'X' => {
                    let n = self.u32()? as usize; // BINUNICODE
                    let s = self.read_str(n)?;
                    self.stack.push(Val::Str(s));
                }
                0x8d => {
                    let n = self.u64()? as usize; // BINUNICODE8
                    let s = self.read_str(n)?;
                    self.stack.push(Val::Str(s));
                }
                b']' => self.stack.push(Val::List(Vec::new())), // EMPTY_LIST
                b'}' => self.stack.push(Val::Dict(Vec::new())), // EMPTY_DICT
                b')' => self.stack.push(Val::Tuple(Vec::new())), // EMPTY_TUPLE
                0x85 => self.make_tuple(1)?,                    // TUPLE1
                0x86 => self.make_tuple(2)?,                    // TUPLE2
                0x87 => self.make_tuple(3)?,                    // TUPLE3
                b't' => {
                    let items = self.pop_to_mark()?; // TUPLE
                    self.stack.push(Val::Tuple(items));
                }
                b'a' => self.append()?,   // APPEND
                b'e' => self.appends()?,  // APPENDS
                b's' => self.setitem()?,  // SETITEM
                b'u' => self.setitems()?, // SETITEMS
                b'q' => {
                    let i = self.u8()? as usize; // BINPUT
                    self.memo_put(i);
                }
                b'r' => {
                    let i = self.u32()? as usize; // LONG_BINPUT
                    self.memo_put(i);
                }
                0x94 => {
                    // MEMOIZE: append top to memo.
                    let idx = self.memo.len();
                    self.memo_put(idx);
                }
                b'h' => {
                    let i = self.u8()? as usize; // BINGET
                    self.memo_get(i)?;
                }
                b'j' => {
                    let i = self.u32()? as usize; // LONG_BINGET
                    self.memo_get(i)?;
                }
                b'c' => self.global_text()?,  // GLOBAL
                0x93 => self.stack_global()?, // STACK_GLOBAL
                0x81 => self.newobj()?,       // NEWOBJ
                b'R' => self.reduce()?,       // REDUCE
                b'b' => self.build()?,        // BUILD
                b'Q' => self.binpersid()?,    // BINPERSID
                other => {
                    return Err(format!(
                        "unsupported pickle opcode 0x{other:02x} at {}",
                        self.pos - 1
                    ));
                }
            }
        }
        self.stack
            .pop()
            .ok_or_else(|| "empty stack at STOP".to_string())
    }

    // --- operand readers ---
    fn u8(&mut self) -> Result<u8, String> {
        let b = *self.buf.get(self.pos).ok_or("unexpected EOF")?;
        self.pos += 1;
        Ok(b)
    }
    fn u16(&mut self) -> Result<u16, String> {
        let s = self.slice(2)?;
        Ok(u16::from_le_bytes([s[0], s[1]]))
    }
    fn u32(&mut self) -> Result<u32, String> {
        let s = self.slice(4)?;
        Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }
    fn u64(&mut self) -> Result<u64, String> {
        let s = self.slice(8)?;
        Ok(u64::from_le_bytes([
            s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7],
        ]))
    }
    fn slice(&mut self, n: usize) -> Result<&'a [u8], String> {
        let s = self
            .buf
            .get(self.pos..self.pos + n)
            .ok_or("unexpected EOF")?;
        self.pos += n;
        Ok(s)
    }
    fn skip(&mut self, n: usize) -> Result<(), String> {
        self.slice(n).map(|_| ())
    }
    fn read_str(&mut self, n: usize) -> Result<String, String> {
        let s = self.slice(n)?;
        Ok(String::from_utf8_lossy(s).into_owned())
    }
    fn read_long(&mut self, n: usize) -> Result<i64, String> {
        if n == 0 {
            return Ok(0);
        }
        let s = self.slice(n)?;
        let mut v: i64 = 0;
        for (i, &b) in s.iter().enumerate() {
            v |= (b as i64) << (8 * i);
        }
        // Sign-extend from the top byte (little-endian two's complement).
        if s[n - 1] & 0x80 != 0 && n < 8 {
            v |= -1i64 << (8 * n);
        }
        Ok(v)
    }

    // --- read a `line\n` text token (GLOBAL operands) ---
    fn line(&mut self) -> Result<String, String> {
        let start = self.pos;
        while *self.buf.get(self.pos).ok_or("unexpected EOF")? != b'\n' {
            self.pos += 1;
        }
        let s = String::from_utf8_lossy(&self.buf[start..self.pos]).into_owned();
        self.pos += 1; // consume newline
        Ok(s)
    }

    // --- memo ---
    fn memo_put(&mut self, idx: usize) {
        let top = self.stack.last().cloned().unwrap_or(Val::None);
        if idx >= self.memo.len() {
            self.memo.resize(idx + 1, Val::None);
        }
        self.memo[idx] = top;
    }
    fn memo_get(&mut self, idx: usize) -> Result<(), String> {
        let v = self.memo.get(idx).cloned().ok_or("memo miss")?;
        self.stack.push(v);
        Ok(())
    }

    // --- structural ops ---
    fn pop_to_mark(&mut self) -> Result<Vec<Val>, String> {
        let mark = self
            .stack
            .iter()
            .rposition(|v| matches!(v, Val::Mark))
            .ok_or("no MARK on stack")?;
        let items = self.stack.split_off(mark + 1);
        self.stack.pop(); // remove the mark itself
        Ok(items)
    }
    fn make_tuple(&mut self, n: usize) -> Result<(), String> {
        let at = self.stack.len().checked_sub(n).ok_or("stack underflow")?;
        let items = self.stack.split_off(at);
        self.stack.push(Val::Tuple(items));
        Ok(())
    }
    fn append(&mut self) -> Result<(), String> {
        let v = self.stack.pop().ok_or("stack underflow")?;
        if let Some(Val::List(l)) = self.stack.last_mut() {
            l.push(v);
        }
        Ok(())
    }
    fn appends(&mut self) -> Result<(), String> {
        let items = self.pop_to_mark()?;
        if let Some(Val::List(l)) = self.stack.last_mut() {
            l.extend(items);
        }
        Ok(())
    }
    fn setitem(&mut self) -> Result<(), String> {
        let v = self.stack.pop().ok_or("stack underflow")?;
        let k = self.stack.pop().ok_or("stack underflow")?;
        if let Some(Val::Dict(d)) = self.stack.last_mut() {
            d.push((k, v));
        }
        Ok(())
    }
    fn setitems(&mut self) -> Result<(), String> {
        let items = self.pop_to_mark()?;
        if let Some(Val::Dict(d)) = self.stack.last_mut() {
            let mut it = items.into_iter();
            while let (Some(k), Some(v)) = (it.next(), it.next()) {
                d.push((k, v));
            }
        }
        Ok(())
    }

    // --- globals / object construction ---
    fn global_text(&mut self) -> Result<(), String> {
        let _module = self.line()?;
        let name = self.line()?;
        self.stack.push(Val::Global(name));
        Ok(())
    }
    fn stack_global(&mut self) -> Result<(), String> {
        let name = self.pop_str()?;
        let _module = self.pop_str()?;
        self.stack.push(Val::Global(name));
        Ok(())
    }
    fn pop_str(&mut self) -> Result<String, String> {
        match self.stack.pop() {
            Some(Val::Str(s)) => Ok(s),
            _ => Err("expected string operand".to_string()),
        }
    }
    fn newobj(&mut self) -> Result<(), String> {
        // NEWOBJ(cls, args): we don't instantiate arbitrary classes; the only
        // objects we model come through REDUCE. Treat as the args tuple so any
        // following BUILD can still attach state harmlessly.
        let _args = self.stack.pop().ok_or("stack underflow")?;
        let cls = self.stack.pop().ok_or("stack underflow")?;
        self.stack.push(cls);
        Ok(())
    }
    fn reduce(&mut self) -> Result<(), String> {
        let args = self.stack.pop().ok_or("stack underflow")?;
        let callable = self.stack.pop().ok_or("stack underflow")?;
        let args = match args {
            Val::Tuple(t) => t,
            other => vec![other],
        };
        let result = match &callable {
            Val::Global(name) if name == "_rebuild_tensor_v2" || name == "_rebuild_tensor" => {
                rebuild_tensor(&args)?
            }
            Val::Global(name) if name == "_rebuild_parameter" => {
                // (data, requires_grad, backward_hooks) -> the tensor itself.
                args.into_iter().next().ok_or("empty _rebuild_parameter")?
            }
            Val::Global(name) if name == "OrderedDict" => Val::Dict(Vec::new()),
            // Unknown reduce: keep something on the stack so the program runs.
            _ => Val::None,
        };
        self.stack.push(result);
        Ok(())
    }
    fn build(&mut self) -> Result<(), String> {
        // BUILD(obj, state): for OrderedDict the items arrive via SETITEMS, not
        // here; for our purposes state is discarded.
        let _state = self.stack.pop().ok_or("stack underflow")?;
        Ok(())
    }
    fn binpersid(&mut self) -> Result<(), String> {
        // The persistent id is a tuple: ('storage', <StorageClass global>,
        // key, location, numel).
        let pid = self.stack.pop().ok_or("stack underflow")?;
        let t = match pid {
            Val::Tuple(t) => t,
            _ => return Err("persistent id is not a tuple".to_string()),
        };
        let dtype = match t.get(1) {
            Some(Val::Global(cls)) => cls.clone(),
            _ => return Err("persistent id missing storage class".to_string()),
        };
        let key = match t.get(2) {
            Some(Val::Str(s)) => s.clone(),
            Some(Val::Int(i)) => i.to_string(),
            _ => return Err("persistent id missing storage key".to_string()),
        };
        self.stack.push(Val::Storage { dtype, key });
        Ok(())
    }
}

fn rebuild_tensor(args: &[Val]) -> Result<Val, String> {
    // _rebuild_tensor_v2(storage, storage_offset, size, stride, ...)
    let (dtype, key) = match args.first() {
        Some(Val::Storage { dtype, key }) => (dtype.clone(), key.clone()),
        _ => return Err("_rebuild_tensor_v2 missing storage".to_string()),
    };
    let storage_offset = match args.get(1) {
        Some(Val::Int(i)) => *i as usize,
        _ => 0,
    };
    let shape = int_tuple(args.get(2))?;
    let stride = int_tuple(args.get(3))?;
    Ok(Val::Tensor(TensorSpec {
        storage_key: key,
        storage_dtype: dtype,
        storage_offset,
        shape,
        stride,
    }))
}

fn int_tuple(v: Option<&Val>) -> Result<Vec<usize>, String> {
    match v {
        Some(Val::Tuple(items)) | Some(Val::List(items)) => items
            .iter()
            .map(|x| match x {
                Val::Int(i) => Ok(*i as usize),
                _ => Err("expected int in size/stride tuple".to_string()),
            })
            .collect(),
        _ => Err("expected size/stride tuple".to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contiguity_check() {
        assert!(is_contiguous(&[2, 3, 4], &[12, 4, 1]));
        assert!(!is_contiguous(&[2, 3, 4], &[1, 2, 6]));
        // length-1 axis may carry any stride
        assert!(is_contiguous(&[1, 4], &[4, 1]));
        assert!(is_contiguous(&[3], &[1]));
    }

    #[test]
    fn long1_sign_extend() {
        // LONG1 with one byte 0x80 == -128.
        let mut u = Unpickler::new(&[]);
        u.buf = &[0x80];
        u.pos = 0;
        assert_eq!(u.read_long(1).unwrap(), -128);
    }

    /// Hand-built protocol-2 pickle of {"w": tensor(BFloat16, key="0",
    /// shape=(2,3), stride=(3,1))}. Exercises GLOBAL, BINPERSID, REDUCE,
    /// EMPTY_DICT, SETITEMS, tuples and ints end to end.
    #[test]
    fn parses_minimal_state_dict() {
        let mut p: Vec<u8> = Vec::new();
        p.push(0x80);
        p.push(2); // PROTO 2
        p.push(b'}'); // EMPTY_DICT
        p.push(b'('); // MARK
        // key "w"
        p.push(0x8c);
        p.push(1);
        p.extend_from_slice(b"w");
        // value: _rebuild_tensor_v2((storage, 0, (2,3), (3,1), False, dict))
        p.push(b'c');
        p.extend_from_slice(b"torch._utils\n_rebuild_tensor_v2\n"); // GLOBAL
        p.push(b'('); // MARK for args tuple
        // arg0: persistent storage tuple via BINPERSID
        p.push(b'('); // MARK
        p.push(0x8c);
        p.push(7);
        p.extend_from_slice(b"storage");
        p.push(b'c');
        p.extend_from_slice(b"torch\nBFloat16Storage\n"); // storage class
        p.push(0x8c);
        p.push(1);
        p.extend_from_slice(b"0"); // key
        p.push(0x8c);
        p.push(3);
        p.extend_from_slice(b"cpu"); // location
        p.push(b'K');
        p.push(6); // numel
        p.push(b't'); // TUPLE (pop to mark)
        p.push(b'Q'); // BINPERSID
        // arg1: storage_offset 0
        p.push(b'K');
        p.push(0);
        // arg2: size (2,3)
        p.push(b'K');
        p.push(2);
        p.push(b'K');
        p.push(3);
        p.push(0x86); // TUPLE2
        // arg3: stride (3,1)
        p.push(b'K');
        p.push(3);
        p.push(b'K');
        p.push(1);
        p.push(0x86); // TUPLE2
        // arg4: requires_grad False
        p.push(0x89);
        // arg5: backward hooks (empty dict)
        p.push(b'}');
        p.push(b't'); // TUPLE (args)
        p.push(b'R'); // REDUCE
        p.push(b'u'); // SETITEMS
        p.push(b'.'); // STOP

        let tensors = parse_pickle(&p).unwrap();
        assert_eq!(tensors.len(), 1);
        let (name, spec) = &tensors[0];
        assert_eq!(name, "w");
        assert_eq!(spec.storage_key, "0");
        assert_eq!(spec.storage_dtype, "BFloat16Storage");
        assert_eq!(spec.shape, vec![2, 3]);
        assert_eq!(spec.stride, vec![3, 1]);
        assert_eq!(spec.storage_offset, 0);
    }
}
