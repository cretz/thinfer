//! `FileOpener`/`WeightReader` over the TS `WeightFile` duck type
//! (`sizeBytes` + `readAt(offset, length) -> Promise<Uint8Array>`).
//!
//! Bytes live in the JS heap until the residency layer's bounded scratch
//! pulls one chunk at a time (`Uint8Array::copy_to` into the caller's
//! `dst`), so wasm linear memory never holds more than a chunk per the
//! no-weight-bytes-in-wasm rule. `will_read` starts the next chunk's OPFS
//! read eagerly (JS promises run on creation), overlapping it with the
//! current chunk's copy + GPU upload.
//!
//! The reader holds a *queue* of in-flight reads (not a single slot): the
//! streaming caller primes several chunk reads ahead, so the OPFS IO worker
//! processes them back-to-back and never idles waiting for the engine thread
//! to issue the next request between chunks (the engine thread can be busy
//! encoding GPU work). Only the resolved chunks live in the JS heap (bounded
//! by `READ_QUEUE_DEPTH` chunks); wasm linear memory still holds at most one.

use std::collections::VecDeque;

use thinfer_core::weight::{FileOpener, WeightReader};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

/// Max reads the streaming caller may keep in flight at once. Matches the
/// residency prefetch depth so the IO worker's request queue stays full
/// across a weight's chunks. Caps JS-heap in-flight bytes at
/// `READ_QUEUE_DEPTH * UPLOAD_CHUNK_BYTES`.
pub const READ_QUEUE_DEPTH: usize = 4;

#[wasm_bindgen]
extern "C" {
    /// Duck type of the TS `WeightFile` (see `src/types.ts`).
    pub type JsWeightFile;
    #[wasm_bindgen(method, getter, js_name = sizeBytes)]
    pub fn size_bytes(this: &JsWeightFile) -> f64;
    #[wasm_bindgen(method, js_name = readAt)]
    pub fn read_at(this: &JsWeightFile, offset: f64, length: f64) -> js_sys::Promise;
}

// Handle copy (the macro doesn't derive it for imported types).
impl Clone for JsWeightFile {
    fn clone(&self) -> Self {
        use wasm_bindgen::JsCast;
        Self::unchecked_from_js(JsValue::clone(self.as_ref()))
    }
}

/// `Debug`-able error carrying the stringified `JsValue` rejection.
pub struct JsIoError(String);

impl core::fmt::Debug for JsIoError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&self.0)
    }
}

impl JsIoError {
    fn from_js(context: &str, v: JsValue) -> Self {
        Self(format!("{context}: {v:?}"))
    }
}

pub struct WebFileOpener {
    file: JsWeightFile,
    len: u64,
}

impl WebFileOpener {
    pub fn new(file: JsWeightFile) -> Self {
        let len = file.size_bytes() as u64;
        Self { file, len }
    }
}

impl FileOpener for WebFileOpener {
    type Reader = WebFileReader;
    type Error = JsIoError;

    async fn open(&self) -> Result<WebFileReader, JsIoError> {
        Ok(WebFileReader {
            file: self.file.clone(),
            len: self.len,
            pending: VecDeque::new(),
        })
    }
}

pub struct WebFileReader {
    file: JsWeightFile,
    len: u64,
    /// In-flight `will_read` prefetches in issue order: `(offset, len, read)`.
    /// `read_at` consumes the front when its arguments match (the streaming
    /// caller reads chunks in the same order it hints them); a mismatched
    /// front means the queued hints are stale (new read pattern), so they're
    /// dropped (wasted reads, never wrong bytes). Bounded to
    /// `READ_QUEUE_DEPTH` so the IO worker is kept fed without unbounded
    /// JS-heap in-flight bytes.
    pending: VecDeque<(u64, u64, JsFuture)>,
}

impl WebFileReader {
    fn start_read(&self, offset: u64, len: u64) -> JsFuture {
        JsFuture::from(self.file.read_at(offset as f64, len as f64))
    }
}

impl WeightReader for WebFileReader {
    type Error = JsIoError;

    fn len(&self) -> u64 {
        self.len
    }

    async fn read_at(&mut self, offset: u64, dst: &mut [u8]) -> Result<(), JsIoError> {
        let len = dst.len() as u64;
        let fut = match self.pending.front() {
            Some((o, l, _)) if *o == offset && *l == len => self.pending.pop_front().unwrap().2,
            // Front doesn't match: any queued hints are stale (the caller's
            // read pattern changed). Drop them and read directly.
            _ => {
                self.pending.clear();
                self.start_read(offset, len)
            }
        };
        let val = fut
            .await
            .map_err(|e| JsIoError::from_js("WeightFile.readAt rejected", e))?;
        let arr: js_sys::Uint8Array = val
            .dyn_into()
            .map_err(|v| JsIoError::from_js("WeightFile.readAt resolved to non-Uint8Array", v))?;
        if u64::from(arr.length()) != len {
            return Err(JsIoError(format!(
                "WeightFile.readAt returned {} bytes, wanted {len} at offset {offset}",
                arr.length()
            )));
        }
        arr.copy_to(dst);
        Ok(())
    }

    fn will_read(&mut self, offset: u64, len: u64) {
        // Keep the in-flight queue bounded; drop the hint once full (the
        // caller refills as it consumes from the front).
        if self.pending.len() >= READ_QUEUE_DEPTH {
            return;
        }
        // Already queued: don't double-issue the same read.
        if self
            .pending
            .iter()
            .any(|(o, l, _)| *o == offset && *l == len)
        {
            return;
        }
        let fut = self.start_read(offset, len);
        self.pending.push_back((offset, len, fut));
    }
}
