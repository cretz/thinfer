use crate::tensor::{Shape, StorageEncoding};
use crate::weight::{Decoder, WeightEntry, WeightId, WeightReader, WeightSource};

#[derive(Debug)]
pub enum Outcome {
    Pass {
        /// Bytes that would land in the compute-dtype buffer (passthrough = src
        /// bytes; bf16 -> fp32 = 2x src bytes).
        decoded_bytes: u64,
    },
    Fail(FailReason),
}

#[derive(Debug)]
pub enum FailReason {
    /// On-disk dtype isn't one we can map to a `StorageEncoding`.
    UnknownStorageDtype(String),
    /// Encoding maps but no decoder exists yet (e.g. quantized).
    NoDecoder(StorageEncoding),
    /// Byte count from the source doesn't match `elements * elem_size`.
    SizeMismatch { expected: u64, got: u64 },
    /// Underlying reader returned an error.
    ReadError(String),
}

#[derive(Debug)]
pub struct TensorReport {
    pub name: String,
    pub encoding_label: String,
    pub shape: Shape,
    pub bytes_on_disk: u64,
    pub outcome: Outcome,
}

#[derive(Debug, Default)]
pub struct Report {
    pub tensors: Vec<TensorReport>,
}

impl Report {
    pub fn pass_count(&self) -> usize {
        self.tensors
            .iter()
            .filter(|t| matches!(t.outcome, Outcome::Pass { .. }))
            .count()
    }
    pub fn fail_count(&self) -> usize {
        self.tensors.len() - self.pass_count()
    }
    pub fn ok(&self) -> bool {
        self.fail_count() == 0
    }
}

const SCRATCH_CHUNK: usize = 64 * 1024;

/// Walk every tensor in `source`'s catalog, opening a reader per entry and
/// validating that the canonical storage->compute decode would succeed.
/// Format-agnostic: works against any `WeightSource`. Iteration order is
/// alphabetical so reports are deterministic.
pub async fn check_source<S: WeightSource>(source: &S) -> Report {
    let mut report = Report::default();
    let mut ids: Vec<&WeightId> = source.catalog().entries.keys().collect();
    ids.sort_by(|a, b| a.0.cmp(&b.0));
    for id in ids {
        let entry = source.catalog().get(id).expect("id from catalog iter");
        let outcome = check_one(source, id, entry).await;
        report.tensors.push(TensorReport {
            name: id.0.clone(),
            encoding_label: entry.encoding_label.to_string(),
            shape: entry.shape.clone(),
            bytes_on_disk: entry.size,
            outcome,
        });
    }
    report
}

async fn check_one<S: WeightSource>(source: &S, id: &WeightId, entry: &WeightEntry) -> Outcome {
    let Some(encoding) = entry.encoding else {
        return Outcome::Fail(FailReason::UnknownStorageDtype(
            entry.encoding_label.to_string(),
        ));
    };
    let elements: u64 = entry.shape.0.iter().map(|&d| d as u64).product();
    let mut reader = match source.open(id).await {
        Ok(r) => r,
        Err(e) => return Outcome::Fail(FailReason::ReadError(format!("{e:?}"))),
    };
    match check_tensor(encoding, elements, &mut reader).await {
        Ok(decoded_bytes) => Outcome::Pass { decoded_bytes },
        Err(reason) => Outcome::Fail(reason),
    }
}

/// Walk one tensor's bytes via `reader`, validating that the canonical
/// storage->compute decode would succeed. 64KB scratch buffer internally.
/// Public so callers that already have an open reader (tests, future runtime
/// debug paths) can validate a single tensor without going through a Source.
pub async fn check_tensor<R: WeightReader>(
    encoding: StorageEncoding,
    elements: u64,
    reader: &mut R,
) -> Result<u64, FailReason> {
    let mut state = ValidateState::new(encoding, elements)?;
    let total = reader.len();
    let mut buf = vec![0u8; SCRATCH_CHUNK];
    let mut offset: u64 = 0;
    while offset < total {
        let n = ((total - offset) as usize).min(buf.len());
        reader
            .read_at(offset, &mut buf[..n])
            .await
            .map_err(|e| FailReason::ReadError(format!("{e:?}")))?;
        state.feed(&buf[..n])?;
        offset += n as u64;
    }
    state.finish()
}

/// Per-tensor canonical-decode validator. Drives the runtime `Decoder` into a
/// 64KB scratch buffer and xors decoded bytes into a `black_box` sink so the
/// hot conversion loop can't be elided. fp16 storage is rejected here too:
/// it's flagged as `NoDecoder` (the engine has no fp16 compute path in M1),
/// so sanity output matches what real loading would do.
struct ValidateState {
    encoding: StorageEncoding,
    decoder: Decoder,
    expected_src_bytes: u64,
    bytes_seen: u64,
    scratch: Vec<u8>,
    /// Anti-optimization sink so the decode loop can't be elided.
    sink: u32,
}

const DECODE_SCRATCH: usize = 256 * 1024;

impl ValidateState {
    fn new(encoding: StorageEncoding, elements: u64) -> Result<Self, FailReason> {
        // Quant encodings deliberately fall through to NoDecoder: sanity
        // is an xor-of-decoded-fp32 integrity check, and GGUF quant
        // tensors don't have a stream-decode-to-fp32 path. Their
        // integrity is validated by the GGUF tensor-data offset + size
        // check at parse time.
        let expected = match encoding {
            StorageEncoding::F32 => elements * 4,
            StorageEncoding::F16 | StorageEncoding::Bf16 => elements * 2,
            enc @ (StorageEncoding::I8 | StorageEncoding::I4 | StorageEncoding::Quant(_)) => {
                return Err(FailReason::NoDecoder(enc));
            }
        };
        let decoder = Decoder::new(encoding).map_err(|_| FailReason::NoDecoder(encoding))?;
        Ok(Self {
            encoding,
            decoder,
            expected_src_bytes: expected,
            bytes_seen: 0,
            scratch: vec![0u8; DECODE_SCRATCH],
            sink: 0,
        })
    }

    fn feed(&mut self, chunk: &[u8]) -> Result<(), FailReason> {
        self.bytes_seen += chunk.len() as u64;
        if self.bytes_seen > self.expected_src_bytes {
            return Err(FailReason::SizeMismatch {
                expected: self.expected_src_bytes,
                got: self.bytes_seen,
            });
        }
        // bf16 doubles the byte count; chunk src against scratch capacity.
        let mut remaining = chunk;
        while !remaining.is_empty() {
            let take = remaining.len().min(self.scratch.len() / 2);
            let (head, tail) = remaining.split_at(take);
            let n = self
                .decoder
                .feed(head, &mut self.scratch)
                .map_err(|e| FailReason::ReadError(format!("decode: {e:?}")))?;
            for word in self.scratch[..n].chunks_exact(4) {
                self.sink ^= u32::from_ne_bytes([word[0], word[1], word[2], word[3]]);
            }
            remaining = tail;
        }
        Ok(())
    }

    fn finish(self) -> Result<u64, FailReason> {
        if self.bytes_seen != self.expected_src_bytes {
            return Err(FailReason::SizeMismatch {
                expected: self.expected_src_bytes,
                got: self.bytes_seen,
            });
        }
        self.decoder
            .finish()
            .map_err(|_| FailReason::SizeMismatch {
                expected: self.expected_src_bytes,
                got: self.bytes_seen,
            })?;
        core::hint::black_box(self.sink);
        let decoded = match self.encoding {
            StorageEncoding::F32 => self.expected_src_bytes,
            StorageEncoding::Bf16 => self.expected_src_bytes * 2,
            StorageEncoding::F16
            | StorageEncoding::I8
            | StorageEncoding::I4
            | StorageEncoding::Quant(_) => unreachable!(),
        };
        Ok(decoded)
    }
}
