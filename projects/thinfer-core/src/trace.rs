//! Tracing targets used by `thinfer-core` (and downstream crates) when emitting
//! `tracing::*` events and spans. Centralising them keeps `RUST_LOG=...` filters
//! and the rollup subscriber's pattern matching honest.
//!
//! No `eprintln!` in library code. Every diagnostic, error, and perf event goes
//! through `tracing` so the binary (CLI / test harness) chooses how to render.
//!
//! Two flavours of consumer:
//! * Live stream: `tracing_subscriber::fmt::Layer` with an `EnvFilter` driven
//!   by `RUST_LOG` (e.g. `RUST_LOG=thinfer::dispatch=info`).
//! * Aggregated rollup: `RollupLayer` (defined at the bottom of this file,
//!   gated on the `trace-subscriber` feature) buckets dispatch/submit/buf
//!   events per active `SCOPE` span path and dumps a sorted table on demand.

/// The clock for every tracing-gated timing read in the engine. `std::time`
/// on native, `performance.now()` on wasm (where `std::time::Instant::now`
/// aborts). Always behind a subscriber-interest gate so disabled tracing
/// reads no clock at all; on wasm an enabled read is one JS roundtrip.
pub use web_time::Instant;

// --- Existing targets retained (used by `debug!` calls already in the tree) ---

pub const WEIGHT_UPLOAD: &str = "thinfer::weight.upload";
pub const WEIGHT_EVICT: &str = "thinfer::weight.evict";
pub const COMPILE: &str = "thinfer::compile";
pub const RESIDENCY_MOVE: &str = "thinfer::residency.move";
pub const FENCE_AWAIT: &str = "thinfer::fence.await";
pub const PIPELINE_NEXT_WORK: &str = "thinfer::pipeline.next_work";
/// Per-layer / per-block phase timing inside model forwards. Debug-only;
/// enable with `RUST_LOG=thinfer::phase=debug` to see acquire vs submit
/// elapsed per Qwen3 layer or DiT block.
pub const PHASE: &str = "thinfer::phase";

// --- Perf-trace targets emitted by backend + workspace + model code ---

/// Nested named phase span. Use via `let _g = trace::scope("dit.block.3").entered();`
/// The `RollupLayer` walks the active SCOPE-span chain to attribute downstream
/// events. Path strings get joined with `/`.
pub const SCOPE: &str = "thinfer::scope";

/// Per-compute-dispatch event. Fields: `pipeline` (entry-point name),
/// `wg_x`, `wg_y`, `wg_z` (workgroup counts), `n_bindings`.
pub const DISPATCH: &str = "thinfer::dispatch";

/// Per-dispatch GPU-side time recovered from wgpu timestamp queries. Emitted
/// after submit completion, re-entering the dispatching scope so the rollup
/// attributes it to the correct sub-region. Fields: `pipeline`, `gpu_ms`.
/// Only present when `THINFER_TRACE` is set AND the adapter exposes
/// `Features::TIMESTAMP_QUERY`.
pub const DISPATCH_GPU: &str = "thinfer::dispatch.gpu";

/// Per-queue-submit event. Fields: `ordinal`, `finish_ms` (encoder.finish()
/// host cost), `submit_call_ms` (queue.submit host cost), `gpu_ms` (host wall
/// time waiting for `on_submitted_work_done`).
pub const SUBMIT: &str = "thinfer::submit";

/// Backend buffer allocate / free. Fields: `op` ("alloc" | "free"), `id`,
/// `bytes` (alloc only).
pub const BUF: &str = "thinfer::buf";

/// Workspace pool alloc / release / drop. Fields: `op`
/// ("alloc" | "release" | "drop"), `id`, `class`, `bytes`, `reused` (alloc).
pub const WS: &str = "thinfer::ws";

/// `read_buffer_via_encoder` lifecycle. Fields: `op`
/// ("record" | "complete" | "rejected"), `ordinal`, `len` / `err`.
pub const RBE: &str = "thinfer::rbe";

/// Memory-budget arbiter reclaim chain. Fields: `op` ("reclaim" | "overshoot"),
/// `source` (reclaimer label), `want`, `freed` / `bytes`, `over`.
pub const ARBITER: &str = "thinfer::arbiter";

/// wgpu adapter info emitted once at backend construction. Fields: `name`,
/// `vendor`, `device`, `backend`, `driver`, `driver_info`, `device_type`.
pub const ADAPTER: &str = "thinfer::adapter";

/// wgpu uncaptured / submit-scope errors. Fields: `kind`
/// ("uncaptured" | "validation" | "oom" | "internal"), `message`.
pub const WGPU_ERR: &str = "thinfer::wgpu.err";

/// High-level user-facing diagnostics: parity dumps, milestone markers, CLI
/// progress. Treated as `INFO`-level by default; the live fmt layer should
/// render them with minimal decoration so they read like the prior `eprintln`s.
pub const DIAG: &str = "thinfer::diag";

/// Build a `SCOPE`-target span. RAII enter: `let _g = trace::scope!("...").entered();`
///
/// The `name` lands on a `name` field (visited by `RollupLayer`). Span itself
/// uses a fixed name "scope" so static metadata is predictable. Implemented
/// as a macro (not fn) so the `name` expression is evaluated lazily by the
/// underlying `tracing::span!` machinery — when the subscriber says the
/// callsite is disabled (default state: no `THINFER_TRACE`, CLI filter at
/// `warn`), the `name` value is never touched. A `pub fn scope(impl Into<String>)`
/// would allocate unconditionally; the macro form is zero-cost when off.
///
/// Accepts any `Display` (`&'static str`, `&str`, `String`, `format_args!`).
#[macro_export]
macro_rules! trace_scope {
    ($name:expr $(,)?) => {
        ::tracing::span!(
            target: $crate::trace::SCOPE,
            ::tracing::Level::INFO,
            "scope",
            name = %$name,
        )
    };
    ($name:expr, $($field:tt)+) => {
        ::tracing::span!(
            target: $crate::trace::SCOPE,
            ::tracing::Level::INFO,
            "scope",
            name = %$name,
            $($field)+
        )
    };
}
pub use crate::trace_scope as scope;

// ============================================================================
// Aggregating subscriber (native-only). Gated on `trace-subscriber` feature so
// wasm builds neither pull `tracing-subscriber` nor compile any of this code.
// Raw `tracing::*` emission above is wasm-clean and stays unconditional.
// ============================================================================

#[cfg(feature = "trace-subscriber")]
mod sub_impl {
    use super::Instant;
    use std::collections::HashMap;
    use std::io::Write;
    use std::sync::{Arc, Mutex};
    use tracing::Subscriber;
    use tracing::field::{Field, Visit};
    use tracing::span::{Attributes, Id};
    use tracing_subscriber::EnvFilter;
    use tracing_subscriber::Registry;
    use tracing_subscriber::filter::FilterFn;
    use tracing_subscriber::fmt;
    use tracing_subscriber::layer::{Context, Layer, SubscriberExt};
    use tracing_subscriber::registry::{LookupSpan, SpanRef};
    use tracing_subscriber::util::SubscriberInitExt;

    use super::{BUF, DISPATCH, DISPATCH_GPU, SCOPE, SUBMIT, WS};

    /// Per-layer filter admitting exactly the rollup's inputs (SCOPE spans
    /// plus dispatch/submit/buf/ws events) at every level. The rollup must
    /// see them regardless of `RUST_LOG`: the live fmt stream filters those
    /// targets out at its default level (they sit at `trace`), but the
    /// aggregate tables are built from them. Always compose `RollupLayer`
    /// with this via `with_filter`, and put the `EnvFilter` on the fmt layer
    /// (a bare `EnvFilter` layer filters the whole stack, rollup included).
    pub fn rollup_filter() -> FilterFn {
        FilterFn::new(|meta| {
            let t = meta.target();
            t == SCOPE || t == DISPATCH || t == DISPATCH_GPU || t == SUBMIT || t == BUF || t == WS
        })
    }

    #[derive(Default)]
    struct ScopeStats {
        busy_ns: u128,
        n_enters: u64,
        n_dispatches: u64,
        n_submits: u64,
        gpu_ms_total: f64,
        /// Sum of per-dispatch GPU time recovered from wgpu timestamp queries
        /// (`DISPATCH_GPU` events), attributed to this scope. Distinct from
        /// `gpu_ms_total`, which counts host wall-time spent in
        /// `on_submitted_work_done` and lands on the submit site only.
        dispatch_gpu_ms_total: f64,
        n_dispatch_gpu: u64,
        submit_finish_ms_total: f64,
        submit_call_ms_total: f64,
        bytes_alloc: u64,
        n_alloc: u64,
        n_free: u64,
        n_ws_alloc: u64,
        n_ws_reuse: u64,
        pipeline_calls: HashMap<String, u64>,
    }

    #[derive(Default)]
    struct State {
        scopes: HashMap<String, ScopeStats>,
        total_alloc_bytes: u64,
        peak_live_bytes: u64,
        cur_live_bytes: u64,
        /// Per-pipeline GPU time from `DISPATCH_GPU` timestamp queries, summed
        /// across all scopes: (gpu_ms_total, n_dispatches). Answers "which op
        /// kind owns the wall time" (e.g. weight-feed `narrow_transpose_f32`
        /// vs `matmul_*`/`sdpa_sg`), which the per-scope table cannot split.
        pipeline_gpu_ms: HashMap<String, (f64, u64)>,
    }

    /// Handle to the rollup's accumulated state. Cheap to clone (Arc inside).
    #[derive(Clone)]
    pub struct RollupHandle {
        inner: Arc<Mutex<State>>,
    }

    /// Aggregating `Layer`. Compose with `Registry` + optional `fmt::Layer`.
    pub struct RollupLayer {
        inner: Arc<Mutex<State>>,
    }

    impl Default for RollupLayer {
        fn default() -> Self {
            Self::new()
        }
    }

    impl RollupLayer {
        pub fn new() -> Self {
            Self {
                inner: Arc::new(Mutex::new(State::default())),
            }
        }
        pub fn handle(&self) -> RollupHandle {
            RollupHandle {
                inner: Arc::clone(&self.inner),
            }
        }
    }

    /// Cached per-span data the layer stashes in `span.extensions`. Built once
    /// on `on_new_span` so `on_enter` / `on_exit` / `on_event` never re-walk
    /// parents.
    struct SpanData {
        path: String,
        enter_at: Option<Instant>,
    }

    impl<S> Layer<S> for RollupLayer
    where
        S: Subscriber + for<'a> LookupSpan<'a>,
    {
        fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
            if attrs.metadata().target() != SCOPE {
                return;
            }
            let mut v = NameVisitor::default();
            attrs.record(&mut v);
            let name = v.name.unwrap_or_else(|| "<unnamed>".to_owned());
            let parent_path = ctx.lookup_current().and_then(|s| nearest_scope_path(&s));
            let path = match parent_path {
                Some(p) => format!("{p}/{name}"),
                None => name,
            };
            if let Some(span) = ctx.span(id) {
                span.extensions_mut().insert(SpanData {
                    path,
                    enter_at: None,
                });
            }
        }

        fn on_enter(&self, id: &Id, ctx: Context<'_, S>) {
            if let Some(span) = ctx.span(id) {
                let mut ext = span.extensions_mut();
                if let Some(d) = ext.get_mut::<SpanData>() {
                    d.enter_at = Some(Instant::now());
                }
            }
        }

        fn on_exit(&self, id: &Id, ctx: Context<'_, S>) {
            let path_busy = ctx.span(id).and_then(|span| {
                let mut ext = span.extensions_mut();
                let d = ext.get_mut::<SpanData>()?;
                let start = d.enter_at.take()?;
                Some((d.path.clone(), start.elapsed().as_nanos()))
            });
            if let Some((path, busy_ns)) = path_busy {
                let mut st = self.inner.lock().unwrap();
                let s = st.scopes.entry(path).or_default();
                s.busy_ns += busy_ns;
                s.n_enters += 1;
            }
        }

        fn on_event(&self, event: &tracing::Event<'_>, ctx: Context<'_, S>) {
            let target = event.metadata().target();
            let interested = target == DISPATCH
                || target == DISPATCH_GPU
                || target == SUBMIT
                || target == BUF
                || target == WS;
            if !interested {
                return;
            }
            let path = current_scope_path(&ctx).unwrap_or_else(|| "<root>".to_owned());
            let mut st = self.inner.lock().unwrap();
            let s = st.scopes.entry(path).or_default();
            if target == DISPATCH {
                s.n_dispatches += 1;
                let mut v = DispatchVisitor::default();
                event.record(&mut v);
                if let Some(pn) = v.pipeline {
                    *s.pipeline_calls.entry(pn).or_default() += 1;
                }
            } else if target == DISPATCH_GPU {
                let mut v = SubmitVisitor::default();
                event.record(&mut v);
                if let Some(ms) = v.gpu_ms {
                    s.dispatch_gpu_ms_total += ms;
                    s.n_dispatch_gpu += 1;
                    // `s` is unused past this point in the branch, so the
                    // disjoint `st.pipeline_gpu_ms` reborrow is sound (NLL).
                    let name = v.pipeline.unwrap_or_else(|| "<unknown>".to_owned());
                    let e = st.pipeline_gpu_ms.entry(name).or_default();
                    e.0 += ms;
                    e.1 += 1;
                }
            } else if target == SUBMIT {
                s.n_submits += 1;
                let mut v = SubmitVisitor::default();
                event.record(&mut v);
                s.gpu_ms_total += v.gpu_ms.unwrap_or(0.0);
                s.submit_finish_ms_total += v.finish_ms.unwrap_or(0.0);
                s.submit_call_ms_total += v.submit_call_ms.unwrap_or(0.0);
            } else if target == BUF {
                let mut v = OpBytesVisitor::default();
                event.record(&mut v);
                match v.op.as_deref() {
                    Some("alloc") => {
                        s.n_alloc += 1;
                        let b = v.bytes.unwrap_or(0);
                        s.bytes_alloc += b;
                        st.total_alloc_bytes += b;
                        st.cur_live_bytes += b;
                        if st.cur_live_bytes > st.peak_live_bytes {
                            st.peak_live_bytes = st.cur_live_bytes;
                        }
                    }
                    Some("free") => {
                        s.n_free += 1;
                        let b = v.bytes.unwrap_or(0);
                        st.cur_live_bytes = st.cur_live_bytes.saturating_sub(b);
                    }
                    _ => {}
                }
            } else if target == WS && v_is_alloc(event) {
                s.n_ws_alloc += 1;
                let mut v = OpBytesVisitor::default();
                event.record(&mut v);
                if v.reused.unwrap_or(false) {
                    s.n_ws_reuse += 1;
                }
            }
        }
    }

    fn v_is_alloc(event: &tracing::Event<'_>) -> bool {
        let mut v = OpBytesVisitor::default();
        event.record(&mut v);
        v.op.as_deref() == Some("alloc")
    }

    fn nearest_scope_path<'a, S>(span: &SpanRef<'a, S>) -> Option<String>
    where
        S: Subscriber + for<'b> LookupSpan<'b>,
    {
        if let Some(d) = span.extensions().get::<SpanData>() {
            return Some(d.path.clone());
        }
        let mut cur = span.parent();
        while let Some(s) = cur {
            if let Some(d) = s.extensions().get::<SpanData>() {
                return Some(d.path.clone());
            }
            cur = s.parent();
        }
        None
    }

    fn current_scope_path<S>(ctx: &Context<'_, S>) -> Option<String>
    where
        S: Subscriber + for<'b> LookupSpan<'b>,
    {
        ctx.lookup_current().and_then(|s| nearest_scope_path(&s))
    }

    // ---- Field visitors ----

    #[derive(Default)]
    struct NameVisitor {
        name: Option<String>,
    }
    impl Visit for NameVisitor {
        fn record_str(&mut self, f: &Field, v: &str) {
            if f.name() == "name" {
                self.name = Some(v.to_owned());
            }
        }
        fn record_debug(&mut self, f: &Field, v: &dyn core::fmt::Debug) {
            if f.name() == "name" {
                let s = format!("{v:?}");
                self.name = Some(s.trim_matches('"').to_owned());
            }
        }
    }

    #[derive(Default)]
    struct DispatchVisitor {
        pipeline: Option<String>,
    }
    impl Visit for DispatchVisitor {
        fn record_str(&mut self, f: &Field, v: &str) {
            if f.name() == "pipeline" {
                self.pipeline = Some(v.to_owned());
            }
        }
        fn record_debug(&mut self, f: &Field, v: &dyn core::fmt::Debug) {
            if f.name() == "pipeline" {
                let s = format!("{v:?}");
                self.pipeline = Some(s.trim_matches('"').to_owned());
            }
        }
    }

    #[derive(Default)]
    struct SubmitVisitor {
        gpu_ms: Option<f64>,
        finish_ms: Option<f64>,
        submit_call_ms: Option<f64>,
        /// Only populated for `DISPATCH_GPU` events (carries `pipeline`); SUBMIT
        /// events leave it `None`.
        pipeline: Option<String>,
    }
    impl Visit for SubmitVisitor {
        fn record_f64(&mut self, f: &Field, v: f64) {
            match f.name() {
                "gpu_ms" => self.gpu_ms = Some(v),
                "finish_ms" => self.finish_ms = Some(v),
                "submit_call_ms" => self.submit_call_ms = Some(v),
                _ => {}
            }
        }
        fn record_str(&mut self, f: &Field, v: &str) {
            if f.name() == "pipeline" {
                self.pipeline = Some(v.to_owned());
            }
        }
        fn record_debug(&mut self, f: &Field, v: &dyn core::fmt::Debug) {
            if f.name() == "pipeline" {
                let s = format!("{v:?}");
                self.pipeline = Some(s.trim_matches('"').to_owned());
            }
        }
    }

    #[derive(Default)]
    struct OpBytesVisitor {
        op: Option<String>,
        bytes: Option<u64>,
        reused: Option<bool>,
    }
    impl Visit for OpBytesVisitor {
        fn record_str(&mut self, f: &Field, v: &str) {
            if f.name() == "op" {
                self.op = Some(v.to_owned());
            }
        }
        fn record_u64(&mut self, f: &Field, v: u64) {
            if f.name() == "bytes" {
                self.bytes = Some(v);
            }
        }
        fn record_bool(&mut self, f: &Field, v: bool) {
            if f.name() == "reused" {
                self.reused = Some(v);
            }
        }
        fn record_debug(&mut self, _f: &Field, _v: &dyn core::fmt::Debug) {}
    }

    // ---- Rollup dump ----

    impl RollupHandle {
        /// Render a sorted-by-busy-ms scope table. Caller picks the writer
        /// (stderr, file, in-memory). Intentionally plain text: easy to tee,
        /// grep, diff.
        pub fn dump<W: Write>(&self, w: &mut W) -> std::io::Result<()> {
            let st = self.inner.lock().unwrap();
            writeln!(w, "==== thinfer trace rollup ====")?;
            writeln!(
                w,
                "buffers: total_alloc={} MiB peak_live={} MiB cur_live={} MiB",
                st.total_alloc_bytes / (1024 * 1024),
                st.peak_live_bytes / (1024 * 1024),
                st.cur_live_bytes / (1024 * 1024),
            )?;
            writeln!(w)?;
            writeln!(
                w,
                "{:<60} {:>5} {:>10} {:>10} {:>10} {:>9} {:>9} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
                "scope",
                "iter",
                "busy_ms",
                "gpu_ms",
                "gpu_disp_ms",
                "finish_ms",
                "submit_ms",
                "n_disp",
                "n_subm",
                "n_alloc",
                "alloc_MB",
                "ws_alloc",
                "ws_reuse",
            )?;
            let mut entries: Vec<(&String, &ScopeStats)> = st.scopes.iter().collect();
            entries.sort_by_key(|e| std::cmp::Reverse(e.1.busy_ns));
            for (path, s) in &entries {
                writeln!(
                    w,
                    "{:<60} {:>5} {:>10.1} {:>10.1} {:>10.1} {:>9.1} {:>9.1} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
                    path,
                    s.n_enters,
                    (s.busy_ns as f64) / 1_000_000.0,
                    s.gpu_ms_total,
                    s.dispatch_gpu_ms_total,
                    s.submit_finish_ms_total,
                    s.submit_call_ms_total,
                    s.n_dispatches,
                    s.n_submits,
                    s.n_alloc,
                    s.bytes_alloc / (1024 * 1024),
                    s.n_ws_alloc,
                    s.n_ws_reuse,
                )?;
            }
            writeln!(w)?;
            writeln!(w, "---- pipelines by scope ----")?;
            for (path, s) in &entries {
                if s.pipeline_calls.is_empty() {
                    continue;
                }
                let mut pls: Vec<(&String, &u64)> = s.pipeline_calls.iter().collect();
                pls.sort_by(|a, b| b.1.cmp(a.1));
                write!(w, "{path}: ")?;
                for (i, (name, n)) in pls.iter().enumerate() {
                    if i > 0 {
                        write!(w, ", ")?;
                    }
                    write!(w, "{name}={n}")?;
                }
                writeln!(w)?;
            }
            writeln!(w)?;
            writeln!(
                w,
                "---- gpu_ms by pipeline (timestamp totals, all scopes) ----"
            )?;
            let mut by_op: Vec<(&String, &(f64, u64))> = st.pipeline_gpu_ms.iter().collect();
            by_op.sort_by(|a, b| {
                b.1.0
                    .partial_cmp(&a.1.0)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            for (name, (ms, n)) in &by_op {
                writeln!(
                    w,
                    "{:<28} {:>12.1} ms {:>8} disp {:>9.3} ms/disp",
                    name,
                    ms,
                    n,
                    ms / (*n as f64).max(1.0),
                )?;
            }
            Ok(())
        }
    }

    // ---- Subscriber installation ----

    /// Install a global subscriber composed of `RollupLayer` + optional
    /// `fmt::Layer` based on `THINFER_TRACE`. Returns `Some(handle)` when
    /// installed (call `handle.dump(&mut io::stderr())` at end-of-run), `None`
    /// when env var is unset.
    ///
    /// Reads `RUST_LOG` for the live filter (defaults to `info`); it scopes
    /// the fmt stream only, while the rollup always sees its own targets via
    /// [`rollup_filter`]. Span-close rendering is only added when
    /// `THINFER_TRACE=verbose` (or `v` / `2`) to keep silent runs quiet.
    ///
    /// Returns `Some(handle)` for the first caller in a process; subsequent
    /// callers (e.g. a second `#[test]` in the same binary) get `None` because
    /// the global subscriber is already installed. Production binaries call
    /// this exactly once; tests that need the handle should fall back to
    /// running their assertions without the rollup dump.
    pub fn init_from_env() -> Option<RollupHandle> {
        let val = std::env::var("THINFER_TRACE").ok()?;
        let rollup = RollupLayer::new();
        let handle = rollup.handle();
        let env_filter =
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
        let verbose = matches!(val.as_str(), "verbose" | "v" | "2");
        // At any non-empty THINFER_TRACE we install the fmt layer so adapter
        // / pipeline / dispatch events print to stderr. Span-close noise
        // (per-dispatch CLOSE events) is gated to verbose because it floods
        // the stream; at THINFER_TRACE=1 only event!() calls render.
        let fmt_layer = if verbose {
            fmt::layer()
                .with_writer(std::io::stderr)
                .with_target(true)
                .with_span_events(fmt::format::FmtSpan::CLOSE)
                .boxed()
        } else {
            fmt::layer()
                .with_writer(std::io::stderr)
                .with_target(true)
                .boxed()
        };
        let installed = Registry::default()
            .with(rollup.with_filter(rollup_filter()))
            .with(fmt_layer.with_filter(env_filter))
            .try_init()
            .is_ok();
        if installed { Some(handle) } else { None }
    }

    /// Like `init_from_env` but yields the layer for the caller to compose
    /// with their own subscriber (CLI already installs `fmt::Subscriber`).
    /// Compose it as `layer.with_filter(rollup_filter())` and keep the
    /// `EnvFilter` on the fmt layer, not as a stack-wide layer. Returns
    /// `None` when `THINFER_TRACE` is unset.
    pub fn rollup_layer_from_env() -> Option<(RollupLayer, RollupHandle)> {
        std::env::var("THINFER_TRACE").ok()?;
        let layer = RollupLayer::new();
        let handle = layer.handle();
        Some((layer, handle))
    }
}

#[cfg(feature = "trace-subscriber")]
pub use sub_impl::{
    RollupHandle, RollupLayer, init_from_env, rollup_filter, rollup_layer_from_env,
};
