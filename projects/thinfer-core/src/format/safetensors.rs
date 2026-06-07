use crate::tensor::{Shape, StorageEncoding};
use crate::weight::{
    FileOpener, OffsetView, WeightCatalog, WeightEntry, WeightId, WeightReader, WeightSource,
};
use serde::Deserialize;
use std::collections::HashMap;

/// Bytes preceding the JSON header in a .safetensors file: a little-endian
/// u64 holding the JSON header's length.
pub const HEADER_SIZE_PREFIX: usize = 8;

#[derive(Debug)]
pub enum ParseError {
    /// Buffer didn't even contain the 8-byte size prefix.
    HeaderTooSmall,
    /// Size prefix exceeds `MAX_HEADER_BYTES`.
    HeaderTooLarge(u64),
    /// Buffer is shorter than `8 + header_size`.
    HeaderTruncated { expected: u64, got: u64 },
    /// JSON deserialization failed.
    Json(String),
}

impl From<serde_json::Error> for ParseError {
    fn from(e: serde_json::Error) -> Self {
        Self::Json(e.to_string())
    }
}

/// Parse the prefix-size value out of the leading 8 bytes. Caller uses this
/// to decide how many more bytes to read before calling `parse`.
pub fn header_size(prefix: &[u8; HEADER_SIZE_PREFIX]) -> u64 {
    u64::from_le_bytes(*prefix)
}

/// Parse a prefix+header buffer (`HEADER_SIZE_PREFIX + header_size` bytes;
/// tensor data must NOT be required) into a `WeightCatalog`. Tensor `offset`
/// fields point at absolute file positions. Header-only by design: the
/// safetensors crate's `read_metadata` requires the full file in the buffer
/// (validates `8 + header + sum(tensor_sizes) == buf.len()`), which defeats
/// the streaming-first premise of this engine. We deserialize the JSON header
/// directly with serde_json and skip the buffer-length check.
pub fn parse(buf: &[u8]) -> Result<WeightCatalog, ParseError> {
    if buf.len() < HEADER_SIZE_PREFIX {
        return Err(ParseError::HeaderTooSmall);
    }
    let prefix: [u8; HEADER_SIZE_PREFIX] = buf[..HEADER_SIZE_PREFIX].try_into().unwrap();
    let n = header_size(&prefix);
    if n > MAX_HEADER_BYTES {
        return Err(ParseError::HeaderTooLarge(n));
    }
    let stop = HEADER_SIZE_PREFIX as u64 + n;
    if (buf.len() as u64) < stop {
        return Err(ParseError::HeaderTruncated {
            expected: stop,
            got: buf.len() as u64,
        });
    }
    let json = &buf[HEADER_SIZE_PREFIX..stop as usize];
    let raw: HashMap<String, serde_json::Value> = serde_json::from_slice(json)?;
    let data_offset = stop;
    let mut entries = HashMap::with_capacity(raw.len());
    for (name, value) in raw {
        if name == "__metadata__" {
            continue;
        }
        let info: TensorInfoJson = serde_json::from_value(value)?;
        let (start, end) = (info.data_offsets[0], info.data_offsets[1]);
        entries.insert(
            WeightId(name),
            WeightEntry {
                offset: data_offset + start,
                size: end.saturating_sub(start),
                encoding: encoding_from(&info.dtype),
                encoding_label: info.dtype,
                shape: Shape(info.shape),
            },
        );
    }
    Ok(WeightCatalog { entries })
}

#[derive(Deserialize)]
struct TensorInfoJson {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}

fn encoding_from(dt: &str) -> Option<StorageEncoding> {
    match dt {
        "F32" => Some(StorageEncoding::F32),
        "F16" => Some(StorageEncoding::F16),
        "BF16" => Some(StorageEncoding::Bf16),
        "I8" => Some(StorageEncoding::I8),
        _ => None,
    }
}

/// Cap on header read so a corrupt size prefix can't make us allocate
/// gigabytes. 256 MiB is far past any real safetensors header.
pub const MAX_HEADER_BYTES: u64 = 256 * 1024 * 1024;

#[derive(Debug)]
pub enum SourceError<E: core::fmt::Debug> {
    Opener(E),
    Reader(String),
    Parse(ParseError),
    UnknownTensor(WeightId),
    /// Same tensor name present in two shards of a sharded safetensors set.
    /// HF's `model.safetensors.index.json` guarantees this won't happen for a
    /// well-formed checkpoint; surface it loudly if it does.
    DuplicateTensor {
        id: WeightId,
        shard_a: usize,
        shard_b: usize,
    },
}

/// `WeightSource` over a single .safetensors file, generic over the file
/// opener. Composition point for IO impls (native `TokioFileOpener`, future
/// web equivalent): each tensor read opens a fresh whole-file reader via the
/// opener and wraps it in an `OffsetView` scoped to that tensor's bytes.
pub struct SafetensorsSource<F: FileOpener> {
    catalog: WeightCatalog,
    opener: F,
}

impl<F: FileOpener> SafetensorsSource<F> {
    pub async fn open(opener: F) -> Result<Self, SourceError<F::Error>> {
        let mut reader = opener.open().await.map_err(SourceError::Opener)?;
        let mut prefix = [0u8; HEADER_SIZE_PREFIX];
        reader
            .read_at(0, &mut prefix)
            .await
            .map_err(|e| SourceError::Reader(format!("{e:?}")))?;
        let header_size = header_size(&prefix);
        if header_size > MAX_HEADER_BYTES {
            return Err(SourceError::Parse(ParseError::HeaderTooLarge(header_size)));
        }
        let mut buf = vec![0u8; HEADER_SIZE_PREFIX + header_size as usize];
        buf[..HEADER_SIZE_PREFIX].copy_from_slice(&prefix);
        reader
            .read_at(HEADER_SIZE_PREFIX as u64, &mut buf[HEADER_SIZE_PREFIX..])
            .await
            .map_err(|e| SourceError::Reader(format!("{e:?}")))?;
        let catalog = parse(&buf).map_err(SourceError::Parse)?;
        Ok(Self { catalog, opener })
    }
}

impl<F: FileOpener> WeightSource for SafetensorsSource<F> {
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

/// `WeightSource` over a multi-shard safetensors set (HF's
/// `model-XXXXX-of-YYYYY.safetensors` layout). Each shard is parsed
/// independently; tensor names are merged into a unified catalog plus a
/// per-tensor shard index for `open()` routing. No reliance on the upstream
/// `model.safetensors.index.json` - the per-shard headers are the source of
/// truth (avoids a divergent-index footgun).
pub struct ShardedSafetensorsSource<F: FileOpener> {
    shards: Vec<SafetensorsSource<F>>,
    catalog: WeightCatalog,
    tensor_shard: HashMap<WeightId, usize>,
}

impl<F: FileOpener> ShardedSafetensorsSource<F> {
    pub async fn open(openers: Vec<F>) -> Result<Self, SourceError<F::Error>> {
        let mut shards = Vec::with_capacity(openers.len());
        for o in openers {
            shards.push(SafetensorsSource::open(o).await?);
        }
        let mut catalog = WeightCatalog::new();
        let mut tensor_shard: HashMap<WeightId, usize> = HashMap::new();
        for (i, s) in shards.iter().enumerate() {
            for (id, entry) in &s.catalog().entries {
                if let Some(&prev) = tensor_shard.get(id) {
                    return Err(SourceError::DuplicateTensor {
                        id: id.clone(),
                        shard_a: prev,
                        shard_b: i,
                    });
                }
                tensor_shard.insert(id.clone(), i);
                catalog.entries.insert(id.clone(), entry.clone());
            }
        }
        Ok(Self {
            shards,
            catalog,
            tensor_shard,
        })
    }

    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }
}

impl<F: FileOpener> WeightSource for ShardedSafetensorsSource<F> {
    type Reader = OffsetView<F::Reader>;
    type Error = SourceError<F::Error>;

    fn catalog(&self) -> &WeightCatalog {
        &self.catalog
    }

    async fn open(&self, id: &WeightId) -> Result<Self::Reader, Self::Error> {
        let &idx = self
            .tensor_shard
            .get(id)
            .ok_or_else(|| SourceError::UnknownTensor(id.clone()))?;
        <SafetensorsSource<F> as WeightSource>::open(&self.shards[idx], id).await
    }
}
