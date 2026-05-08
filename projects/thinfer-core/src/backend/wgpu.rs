use crate::backend::poll::WgpuPoll;
use crate::backend::{Backend, Binding, BindingKind, BindingLayout};
use crate::tensor::GpuBufferId;
use crate::trace;
use core::future::Future;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

pub struct WgpuBackend {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    next_id: Mutex<u64>,
    buffers: Mutex<HashMap<GpuBufferId, Arc<wgpu::Buffer>>>,
    poll: WgpuPoll,
    /// Monotonic submit ordinal. Used to tag validation errors and buffer
    /// lifecycle traces so failures can be attributed to a specific submit.
    submit_ordinal: AtomicU64,
    /// Monotonic ordinal for `read_buffer_via_encoder` calls. Gray-PNG diag.
    rbe_ordinal: AtomicU64,
    /// Per-dispatch GPU timing. `Some` only when `WgpuConfig.timestamps` was
    /// requested AND the adapter exposes `Features::TIMESTAMP_QUERY`.
    /// `period_ns` is the queue's timestamp tick length (multiply tick deltas
    /// by this to get ns of GPU time).
    timestamps: Option<TimestampCfg>,
    /// Sink for uncaptured errors. The wgpu uncaptured handler stores the
    /// FIRST error received here (later ones are eprintln'd but not stored,
    /// so the root cause isn't shadowed by its own cascade). Drained at sync
    /// API boundaries (currently `allocate`). On native, wgpu reports
    /// buffer-creation failures synchronously so this catches them at the
    /// allocation point; on wasm the handler may fire asynchronously and the
    /// per-submit scopes catch what slips past.
    uncaptured: Arc<Mutex<Option<wgpu::Error>>>,
}

#[derive(Debug)]
pub enum WgpuError {
    AdapterUnavailable(wgpu::RequestAdapterError),
    DeviceRequest(wgpu::RequestDeviceError),
    UnknownBuffer(GpuBufferId),
    BufferMap(wgpu::BufferAsyncError),
    /// `create_buffer` was rejected by the driver. Drained from the
    /// uncaptured-error sink at the allocation point so the first failure
    /// stops the cascade of "Buffer invalid" follow-ons. Match the inner
    /// variant to distinguish OOM (actionable: evict + retry) from
    /// Validation/Internal (real bug).
    Allocate {
        bytes: u64,
        source: wgpu::Error,
    },
    SubmitFailed {
        ordinal: u64,
        message: String,
    },
    ReadbackRejected {
        ordinal: u64,
        message: String,
    },
}

#[derive(Clone)]
struct TimestampCfg {
    period_ns: f32,
    /// Slot capacity allocated per encoder's QuerySet. 2 slots per dispatch
    /// (begin, end); cap chosen to comfortably cover one DiT block worth of
    /// dispatches per submit without per-encoder reallocation.
    slots_per_encoder: u32,
}

/// Per-encoder timestamp-query state. Carries the wgpu QuerySet plus the
/// (pipeline name, span) records correlated with slot pairs, so submit-time
/// readback can emit per-dispatch GPU time back into the right scope.
struct EncoderTimestamps {
    query_set: wgpu::QuerySet,
    capacity: u32,
    cursor: u32,
    records: Vec<TsRecord>,
}

struct TsRecord {
    pipeline: String,
    span: tracing::Span,
    slot_begin: u32,
    slot_end: u32,
}

/// Wgpu's `CommandEncoder` plus optional timestamp-query state. When the
/// backend was constructed without `WgpuConfig.timestamps` (or the adapter
/// lacks `Features::TIMESTAMP_QUERY`), `ts` stays `None` and there is zero
/// per-dispatch overhead beyond a branch.
pub struct CommandEncoderState {
    enc: wgpu::CommandEncoder,
    ts: Option<EncoderTimestamps>,
}

/// Moved into the submit completion future so the staging buffer and query
/// set outlive the GPU work. The leading underscore on fields we never read
/// after the move is intentional - they exist solely to keep wgpu resources
/// alive across the await.
struct PendingTimestamps {
    staging_buf: wgpu::Buffer,
    records: Vec<TsRecord>,
    period_ns: f32,
    _query_set: wgpu::QuerySet,
    _resolve_buf: wgpu::Buffer,
}

/// Map the timestamp staging buffer, decode u64 tick pairs, and emit one
/// `DISPATCH_GPU` event per record back into the span where the original
/// `DISPATCH` was issued. Failures here are non-fatal: a missed readback
/// would only mute timings, not corrupt the submit result.
async fn emit_dispatch_gpu(pt: PendingTimestamps) {
    let (tx, rx) = futures_channel::oneshot::channel();
    pt.staging_buf
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
    if rx.await.is_err() {
        return;
    }
    let data = pt.staging_buf.slice(..).get_mapped_range().to_vec();
    pt.staging_buf.unmap();
    if data.len() < 16 {
        return;
    }
    let ticks: Vec<u64> = data
        .chunks_exact(8)
        .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    for r in pt.records {
        let begin = ticks.get(r.slot_begin as usize).copied().unwrap_or(0);
        let end = ticks.get(r.slot_end as usize).copied().unwrap_or(0);
        if end < begin {
            continue;
        }
        let gpu_ms = ((end - begin) as f64) * (pt.period_ns as f64) / 1_000_000.0;
        let _g = r.span.enter();
        tracing::info!(
            target: trace::DISPATCH_GPU,
            pipeline = %r.pipeline,
            gpu_ms = gpu_ms,
        );
    }
}

/// Adapter selection preference. Mirrors `wgpu::PowerPreference` so callers
/// in thinfer-core don't pull in wgpu types at their own API surface.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PowerPreference {
    HighPerformance,
    LowPower,
    #[default]
    None,
}

/// Construction-time options for `WgpuBackend`. Read at the binary edge
/// (CLI, tests) and passed in - thinfer-core stays env-var-free for wasm.
#[derive(Clone, Debug, Default)]
pub struct WgpuConfig {
    /// Adapter selection. `HighPerformance` steers wgpu toward the discrete
    /// GPU on hybrid systems; `LowPower` forces the integrated adapter (e.g.
    /// for larger shared-memory budget). CLI maps `THINFER_POWER_PREF` here.
    pub power_preference: PowerPreference,
    /// Request wgpu's `TIMESTAMP_QUERY` feature and emit per-dispatch
    /// `DISPATCH_GPU` tracing events. Only fed by callers that already
    /// installed the rollup subscriber. Silently degrades to off when the
    /// adapter doesn't expose the feature (logged once via `trace::ADAPTER`).
    pub timestamps: bool,
}

pub struct WgpuPipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    /// Entry-point name, captured at `create_pipeline`. Fed into every
    /// `trace::DISPATCH` event so the rollup can bucket time by op kind
    /// (matmul / sdpa / rmsnorm / etc) without callers passing extra labels.
    name: String,
}

impl WgpuBackend {
    pub async fn new() -> Result<Self, WgpuError> {
        Self::new_with_config(WgpuConfig::default()).await
    }

    pub async fn new_with_config(cfg: WgpuConfig) -> Result<Self, WgpuError> {
        let instance = wgpu::Instance::default();
        let power_preference = match cfg.power_preference {
            PowerPreference::HighPerformance => wgpu::PowerPreference::HighPerformance,
            PowerPreference::LowPower => wgpu::PowerPreference::LowPower,
            PowerPreference::None => wgpu::PowerPreference::None,
        };
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .map_err(WgpuError::AdapterUnavailable)?;
        // Request the adapter's max binding size. The downlevel default
        // (128 MiB) is below the largest weight tensors we bind in one go
        // (DiT FFN at 3840*10240*4 = 150 MiB; some VAE convs even larger),
        // so without this the runtime trips wgpu validation. On the web,
        // baseline WebGPU caps at 128 MiB - we'll need a matmul chunking
        // path (or split-storage weights) for those targets. Tracked as a
        // M2 follow-up; native is unblocked by simply requesting more.
        let adapter_limits = adapter.limits();
        let adapter_has_ts = adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        let request_ts = cfg.timestamps && adapter_has_ts;
        if cfg.timestamps && !adapter_has_ts {
            tracing::warn!(
                target: trace::ADAPTER,
                "WgpuConfig.timestamps requested but adapter lacks TIMESTAMP_QUERY; per-dispatch gpu_ms unavailable",
            );
        }
        let required_features = if request_ts {
            wgpu::Features::TIMESTAMP_QUERY
        } else {
            wgpu::Features::empty()
        };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("thinfer"),
                required_features,
                required_limits: wgpu::Limits {
                    max_storage_buffers_per_shader_stage: 8,
                    max_storage_buffer_binding_size: adapter_limits.max_storage_buffer_binding_size,
                    max_buffer_size: adapter_limits.max_buffer_size,
                    ..wgpu::Limits::downlevel_defaults()
                },
                memory_hints: wgpu::MemoryHints::default(),
                experimental_features: wgpu::ExperimentalFeatures::default(),
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(WgpuError::DeviceRequest)?;
        let info = adapter.get_info();
        tracing::info!(
            target: trace::ADAPTER,
            name = %info.name,
            vendor = info.vendor,
            device_id = info.device,
            backend = ?info.backend,
            driver = %info.driver,
            driver_info = %info.driver_info,
            device_type = ?info.device_type,
            "wgpu adapter",
        );
        let uncaptured: Arc<Mutex<Option<wgpu::Error>>> = Arc::new(Mutex::new(None));
        let uncaptured_handler = Arc::clone(&uncaptured);
        device.on_uncaptured_error(Arc::new(move |err| {
            tracing::error!(target: trace::WGPU_ERR, kind = "uncaptured", error = %err);
            // First-wins: the root cause arrives first; later entries are
            // typically its cascade (write_buffer / create_bind_group on the
            // now-invalid buffer). Store only the first so callers see the
            // actionable error, not the noise downstream of it.
            uncaptured_handler.lock().unwrap().get_or_insert(err);
        }));
        let device = Arc::new(device);
        let poll = WgpuPoll::new(device.clone());
        let timestamps = request_ts.then(|| TimestampCfg {
            period_ns: queue.get_timestamp_period(),
            // 4096 slots = 2048 dispatches per submit. Empirically covers one
            // DiT block (~20 dispatches) with two orders of magnitude headroom;
            // dispatches beyond the cap silently skip the timestamp_writes.
            slots_per_encoder: 4096,
        });
        Ok(Self {
            device,
            queue: Arc::new(queue),
            next_id: Mutex::new(1),
            buffers: Mutex::new(HashMap::new()),
            poll,
            submit_ordinal: AtomicU64::new(0),
            rbe_ordinal: AtomicU64::new(0),
            uncaptured,
            timestamps,
        })
    }

    fn get_buffer(&self, id: GpuBufferId) -> Result<Arc<wgpu::Buffer>, WgpuError> {
        self.buffers
            .lock()
            .unwrap()
            .get(&id)
            .cloned()
            .ok_or(WgpuError::UnknownBuffer(id))
    }

    /// Stage a readback inside the caller's encoder. The caller submits that
    /// encoder (via `submit`) and then awaits the returned future. Avoids the
    /// "two queue.submit in a row" pattern that wedges some drivers after a
    /// heavy compute submit (see `read_buffer`'s diagnostic notes).
    pub fn read_buffer_via_encoder(
        &self,
        encoder: &mut CommandEncoderState,
        src: GpuBufferId,
        offset: u64,
        len: u64,
    ) -> Result<impl Future<Output = Result<Vec<u8>, WgpuError>> + use<>, WgpuError> {
        let encoder = &mut encoder.enc;
        let src_buf = self.get_buffer(src)?;
        let src_size = src_buf.size();
        let ord = self.rbe_ordinal.fetch_add(1, Ordering::Relaxed);
        // Validation scope around just this copy, so any record-time rejection
        // is attributable to THIS readback and not an earlier dispatch.
        let scope_guard = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        tracing::debug!(target: crate::trace::PHASE, len, "rbe.staging");
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: len,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&src_buf, offset, &staging, 0, len);
        let scope = scope_guard.pop();
        let guard = self.poll.poll_guard();
        tracing::info!(
            target: trace::RBE,
            op = "record",
            ordinal = ord,
            src_id = src.0,
            src_total = src_size,
            offset = offset,
            len = len,
        );
        Ok(async move {
            let _guard = guard;
            if let Some(err) = scope.await {
                tracing::error!(
                    target: trace::RBE,
                    op = "rejected",
                    ordinal = ord,
                    error = %err,
                );
                return Err(WgpuError::ReadbackRejected {
                    ordinal: ord,
                    message: format!("{err}"),
                });
            }
            tracing::debug!(target: crate::trace::PHASE, "rbe.map_async");
            let (tx, rx) = futures_channel::oneshot::channel();
            staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            tracing::debug!(target: crate::trace::PHASE, "rbe.await");
            rx.await
                .expect("map_async sender dropped")
                .map_err(WgpuError::BufferMap)?;
            tracing::debug!(target: crate::trace::PHASE, "rbe.mapped");
            let data = staging.slice(..).get_mapped_range().to_vec();
            staging.unmap();
            tracing::info!(
                target: trace::RBE,
                op = "complete",
                ordinal = ord,
                len = data.len() as u64,
            );
            Ok(data)
        })
    }
}

impl Backend for WgpuBackend {
    type Error = WgpuError;
    type CommandEncoder = CommandEncoderState;
    type Pipeline = WgpuPipeline;

    fn allocate(&self, bytes: u64) -> Result<GpuBufferId, Self::Error> {
        // Drain any stale uncaptured error before the call so a failure here
        // is attributable to THIS create_buffer, not an earlier sync op.
        let _ = self.uncaptured.lock().unwrap().take();
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::UNIFORM
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        if let Some(err) = self.uncaptured.lock().unwrap().take() {
            tracing::error!(
                target: trace::WGPU_ERR,
                kind = "alloc",
                bytes = bytes,
                error = %err,
            );
            return Err(WgpuError::Allocate { bytes, source: err });
        }
        let id = {
            let mut n = self.next_id.lock().unwrap();
            let id = GpuBufferId(*n);
            *n += 1;
            id
        };
        self.buffers.lock().unwrap().insert(id, Arc::new(buf));
        tracing::info!(
            target: trace::BUF,
            op = "alloc",
            id = id.0,
            bytes = bytes,
        );
        Ok(id)
    }

    fn free(&self, id: GpuBufferId) {
        let bytes = self
            .buffers
            .lock()
            .unwrap()
            .remove(&id)
            .map(|b| b.size())
            .unwrap_or(0);
        tracing::info!(
            target: trace::BUF,
            op = "free",
            id = id.0,
            bytes = bytes,
        );
    }

    fn write_buffer(
        &self,
        dst: GpuBufferId,
        dst_offset: u64,
        src: &[u8],
    ) -> Result<(), Self::Error> {
        let buf = self.get_buffer(dst)?;
        self.queue.write_buffer(&buf, dst_offset, src);
        Ok(())
    }

    fn create_command_encoder(&self) -> Self::CommandEncoder {
        let enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let ts = self.timestamps.as_ref().map(|cfg| {
            let query_set = self.device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("thinfer-ts"),
                ty: wgpu::QueryType::Timestamp,
                count: cfg.slots_per_encoder,
            });
            EncoderTimestamps {
                query_set,
                capacity: cfg.slots_per_encoder,
                cursor: 0,
                records: Vec::new(),
            }
        });
        CommandEncoderState { enc, ts }
    }

    fn dispatch(
        &self,
        encoder: &mut Self::CommandEncoder,
        pipeline: &Self::Pipeline,
        bindings: &[Binding],
        workgroups: [u32; 3],
    ) -> Result<(), Self::Error> {
        let bufs: Vec<Arc<wgpu::Buffer>> = bindings
            .iter()
            .map(|b| self.get_buffer(b.buffer))
            .collect::<Result<_, _>>()?;
        let entries: Vec<wgpu::BindGroupEntry> = bindings
            .iter()
            .zip(&bufs)
            .map(|(b, buf)| wgpu::BindGroupEntry {
                binding: b.slot,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: buf,
                    offset: b.offset,
                    size: core::num::NonZeroU64::new(b.size),
                }),
            })
            .collect();
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.bind_group_layout,
            entries: &entries,
        });
        tracing::info!(
            target: trace::DISPATCH,
            pipeline = %pipeline.name,
            wg_x = workgroups[0],
            wg_y = workgroups[1],
            wg_z = workgroups[2],
            n_bindings = bindings.len() as u32,
        );
        // Reserve two timestamp slots iff the encoder has a query set and
        // there is room. Capacity overflow silently falls back to no
        // timestamp_writes for this dispatch.
        let ts_slots: Option<(&wgpu::QuerySet, u32, u32)> = match encoder.ts.as_mut() {
            Some(ts) if ts.cursor + 2 <= ts.capacity => {
                let slot_begin = ts.cursor;
                let slot_end = ts.cursor + 1;
                ts.cursor += 2;
                ts.records.push(TsRecord {
                    pipeline: pipeline.name.clone(),
                    span: tracing::Span::current(),
                    slot_begin,
                    slot_end,
                });
                Some((&ts.query_set, slot_begin, slot_end))
            }
            _ => None,
        };
        let timestamp_writes = ts_slots.map(|(query_set, b, e)| wgpu::ComputePassTimestampWrites {
            query_set,
            beginning_of_pass_write_index: Some(b),
            end_of_pass_write_index: Some(e),
        });
        let mut pass = encoder
            .enc
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes,
            });
        pass.set_pipeline(&pipeline.pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(workgroups[0], workgroups[1], workgroups[2]);
        Ok(())
    }

    fn copy_buffer_to_buffer(
        &self,
        encoder: &mut Self::CommandEncoder,
        src: GpuBufferId,
        src_offset: u64,
        dst: GpuBufferId,
        dst_offset: u64,
        len: u64,
    ) -> Result<(), Self::Error> {
        let src_buf = self.get_buffer(src)?;
        let dst_buf = self.get_buffer(dst)?;
        encoder
            .enc
            .copy_buffer_to_buffer(&src_buf, src_offset, &dst_buf, dst_offset, len);
        Ok(())
    }

    fn submit(
        &self,
        mut encoder: Self::CommandEncoder,
    ) -> impl Future<Output = Result<(), Self::Error>> {
        let device = self.device.clone();
        let queue = self.queue.clone();
        let uncaptured = Arc::clone(&self.uncaptured);
        let ordinal = self.submit_ordinal.fetch_add(1, Ordering::Relaxed);
        // Drain any pre-existing uncaptured error so anything we catch here is
        // attributable to this submit, not a prior dispatch's cascade.
        if let Some(pre) = uncaptured.lock().unwrap().take() {
            tracing::warn!(
                target: trace::WGPU_ERR,
                kind = "prior_uncaptured_cleared",
                ordinal = ordinal,
                error = %pre,
            );
        }
        // Resolve timestamp queries before finishing the encoder. The resolve
        // copies u64 ticks from the QuerySet into `resolve_buf`, then a second
        // copy lands in a mappable `staging_buf` we read back after the submit
        // completes. Both buffers (and the QuerySet via `EncoderTimestamps`)
        // live in the `pending_ts` payload moved into the completion future.
        let pending_ts = encoder.ts.take().and_then(|mut ts| {
            if ts.cursor == 0 || ts.records.is_empty() {
                return None;
            }
            let used_slots = ts.cursor;
            let bytes = (used_slots as u64) * 8;
            let resolve_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("thinfer-ts-resolve"),
                size: bytes,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("thinfer-ts-staging"),
                size: bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            encoder
                .enc
                .resolve_query_set(&ts.query_set, 0..used_slots, &resolve_buf, 0);
            encoder
                .enc
                .copy_buffer_to_buffer(&resolve_buf, 0, &staging_buf, 0, bytes);
            // Drop the query set after the records vec is moved out; keeping
            // it alive until then is what wgpu requires for the resolve above.
            let records = std::mem::take(&mut ts.records);
            Some(PendingTimestamps {
                staging_buf,
                records,
                period_ns: self.timestamps.as_ref().map(|c| c.period_ns).unwrap_or(1.0),
                _query_set: ts.query_set,
                _resolve_buf: resolve_buf,
            })
        });
        let validation_guard = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let oom_guard = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let internal_guard = device.push_error_scope(wgpu::ErrorFilter::Internal);
        let t_finish = std::time::Instant::now();
        let cmdbuf = encoder.enc.finish();
        let finish_ms = t_finish.elapsed().as_secs_f64() * 1000.0;
        let t_submit = std::time::Instant::now();
        queue.submit([cmdbuf]);
        let submit_call_ms = t_submit.elapsed().as_secs_f64() * 1000.0;
        let internal_scope = internal_guard.pop();
        let oom_scope = oom_guard.pop();
        let validation_scope = validation_guard.pop();
        let (tx, rx) = futures_channel::oneshot::channel::<()>();
        queue.on_submitted_work_done(move || {
            let _ = tx.send(());
        });
        let guard = self.poll.poll_guard();
        async move {
            let _guard = guard;
            let t_wait = std::time::Instant::now();
            rx.await.expect("on_submitted_work_done sender dropped");
            let gpu_ms = t_wait.elapsed().as_secs_f64() * 1000.0;
            let mut errs: Vec<String> = Vec::new();
            if let Some(err) = validation_scope.await {
                tracing::error!(target: trace::WGPU_ERR, kind = "validation", ordinal = ordinal, error = %err);
                errs.push(format!("validation: {err}"));
            }
            if let Some(err) = oom_scope.await {
                tracing::error!(target: trace::WGPU_ERR, kind = "oom", ordinal = ordinal, error = %err);
                errs.push(format!("OOM: {err}"));
            }
            if let Some(err) = internal_scope.await {
                tracing::error!(target: trace::WGPU_ERR, kind = "internal", ordinal = ordinal, error = %err);
                errs.push(format!("internal: {err}"));
            }
            if let Some(post) = uncaptured.lock().unwrap().take() {
                tracing::error!(target: trace::WGPU_ERR, kind = "post_submit_uncaptured", ordinal = ordinal, error = %post);
                errs.push(format!("post-submit uncaptured: {post}"));
            }
            tracing::info!(
                target: trace::SUBMIT,
                ordinal = ordinal,
                finish_ms = finish_ms,
                submit_call_ms = submit_call_ms,
                gpu_ms = gpu_ms,
                n_errs = errs.len() as u32,
            );
            if let Some(pt) = pending_ts {
                emit_dispatch_gpu(pt).await;
            }
            if !errs.is_empty() {
                return Err(WgpuError::SubmitFailed {
                    ordinal,
                    message: errs.join("; "),
                });
            }
            Ok(())
        }
    }

    fn create_pipeline(
        &self,
        wgsl: &str,
        entry: &str,
        layout: &[BindingLayout],
    ) -> impl Future<Output = Result<Self::Pipeline, Self::Error>> {
        let device = self.device.clone();
        let wgsl = wgsl.to_owned();
        let entry = entry.to_owned();
        let layout: Vec<BindingLayout> = layout.to_vec();
        async move {
            tracing::debug!(target: crate::trace::COMPILE, %entry, "wgsl compile");
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(wgsl.into()),
            });
            let entries: Vec<wgpu::BindGroupLayoutEntry> = layout
                .iter()
                .map(|l| wgpu::BindGroupLayoutEntry {
                    binding: l.slot,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: match l.kind {
                            BindingKind::StorageRead => {
                                wgpu::BufferBindingType::Storage { read_only: true }
                            }
                            BindingKind::StorageReadWrite => {
                                wgpu::BufferBindingType::Storage { read_only: false }
                            }
                            BindingKind::Uniform => wgpu::BufferBindingType::Uniform,
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                })
                .collect();
            let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &entries,
            });
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&bgl],
                immediate_size: 0,
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: Some(&pl),
                module: &module,
                entry_point: Some(&entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
            Ok(WgpuPipeline {
                pipeline,
                bind_group_layout: bgl,
                name: entry,
            })
        }
    }

    fn read_buffer(
        &self,
        src: GpuBufferId,
        offset: u64,
        len: u64,
    ) -> impl Future<Output = Result<Vec<u8>, Self::Error>> {
        let device = self.device.clone();
        let queue = self.queue.clone();
        // Submit + register `map_async` BEFORE arming the poll guard. Holding
        // a poll guard across `queue.submit` deadlocks against wgpu's internal
        // Maintain::Poll pass (see `submit` for details).
        let result = self.get_buffer(src).map(|src_buf| {
            tracing::debug!(target: crate::trace::PHASE, len, "rb.staging");
            let staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: len,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut enc =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            enc.copy_buffer_to_buffer(&src_buf, offset, &staging, 0, len);
            tracing::debug!(target: crate::trace::PHASE, "rb.submit");
            queue.submit([enc.finish()]);
            tracing::debug!(target: crate::trace::PHASE, "rb.map_async");
            let (tx, rx) = futures_channel::oneshot::channel();
            staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            tracing::debug!(target: crate::trace::PHASE, "rb.armed");
            (staging, rx)
        });
        let guard = self.poll.poll_guard();
        async move {
            let _guard = guard;
            let (staging, rx) = result?;
            tracing::debug!(target: crate::trace::PHASE, "rb.await");
            rx.await
                .expect("map_async sender dropped")
                .map_err(WgpuError::BufferMap)?;
            tracing::debug!(target: crate::trace::PHASE, "rb.mapped");
            let data = staging.slice(..).get_mapped_range().to_vec();
            staging.unmap();
            Ok(data)
        }
    }
}
