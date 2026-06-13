//! Rust `tracing` -> JS sink bridge. The npm package routes this into its
//! `setLogger` sink so engine telemetry lands wherever library logs do
//! (console by default, a page textarea in the example's debug mode).
//!
//! tracing supports exactly one process-global subscriber, so the bridge is
//! installed once (first `setTraceLevel`/`setTraceSink` call) and both the
//! sink and the level are swappable afterwards. Level default is `off`:
//! zero per-event cost until a caller opts in (`enabled()` gates every
//! callsite and the interest cache is rebuilt on level changes).
//!
//! Spans are accepted (ids handed out) but only events are forwarded;
//! span-timing telemetry stays the rollup subscriber's job on native.

use std::cell::RefCell;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};

use tracing::level_filters::LevelFilter;
use tracing::{Level, Metadata};
use wasm_bindgen::prelude::*;

// u8-encoded `LevelFilter` (0 = off .. 5 = trace). Atomics over Cell only
// because `Subscriber` requires `Sync`; wasm is single-threaded.
static MAX_LEVEL: AtomicU8 = AtomicU8::new(0);
static NEXT_SPAN_ID: AtomicU64 = AtomicU64::new(1);

thread_local! {
    static SINK: RefCell<Option<js_sys::Function>> = const { RefCell::new(None) };
}

fn level_to_u8(f: LevelFilter) -> u8 {
    match f {
        LevelFilter::OFF => 0,
        LevelFilter::ERROR => 1,
        LevelFilter::WARN => 2,
        LevelFilter::INFO => 3,
        LevelFilter::DEBUG => 4,
        LevelFilter::TRACE => 5,
    }
}

fn current_filter() -> LevelFilter {
    match MAX_LEVEL.load(Ordering::Relaxed) {
        0 => LevelFilter::OFF,
        1 => LevelFilter::ERROR,
        2 => LevelFilter::WARN,
        3 => LevelFilter::INFO,
        4 => LevelFilter::DEBUG,
        _ => LevelFilter::TRACE,
    }
}

/// Set the global trace level: `off`, `error`, `warn`, `info`, `debug`,
/// `trace`. Installs the bridge subscriber on first call.
#[wasm_bindgen(js_name = setTraceLevel)]
pub fn set_trace_level(level: &str) -> Result<(), JsError> {
    let filter = match level {
        "off" => LevelFilter::OFF,
        "error" => LevelFilter::ERROR,
        "warn" => LevelFilter::WARN,
        "info" => LevelFilter::INFO,
        "debug" => LevelFilter::DEBUG,
        "trace" => LevelFilter::TRACE,
        other => return Err(JsError::new(&format!("unknown trace level: {other}"))),
    };
    ensure_installed();
    MAX_LEVEL.store(level_to_u8(filter), Ordering::Relaxed);
    // Callsites cache their enabled-ness; invalidate so the new level
    // takes effect on already-hit callsites.
    tracing::callsite::rebuild_interest_cache();
    Ok(())
}

/// Install the JS function `(level: "debug"|"info"|"warn"|"error",
/// message: string) => void` that receives every forwarded event. `None`
/// drops events on the floor (level still gates callsites).
#[wasm_bindgen(js_name = setTraceSink)]
pub fn set_trace_sink(sink: Option<js_sys::Function>) {
    ensure_installed();
    SINK.with(|s| *s.borrow_mut() = sink);
}

fn ensure_installed() {
    use std::sync::Once;
    static INSTALL: Once = Once::new();
    INSTALL.call_once(|| {
        // Failure means another subscriber got there first (impossible in
        // the package's own wasm instance); surface it loudly.
        tracing::subscriber::set_global_default(BridgeSubscriber)
            .expect("tracing subscriber already installed");
    });
}

struct BridgeSubscriber;

impl tracing::Subscriber for BridgeSubscriber {
    fn enabled(&self, metadata: &Metadata<'_>) -> bool {
        let filter = current_filter();
        // `thinfer::diag` INFO callsites are behavior gates for probe
        // readbacks (DiT per-block taps: GPU readback + queue drain), not
        // plain log lines. Level semantics: info = milestones + rollups,
        // debug adds residency per-weight traffic (diag DEBUG, plain check
        // below); the probe firehose rides only `trace`. Mirrors the native
        // perf convention of scoping `thinfer::diag` out of RUST_LOG.
        if metadata.target() == thinfer_core::trace::DIAG && *metadata.level() == Level::INFO {
            return filter >= LevelFilter::TRACE;
        }
        metadata.level() <= &filter
    }

    fn max_level_hint(&self) -> Option<LevelFilter> {
        Some(current_filter())
    }

    fn new_span(&self, _attrs: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(NEXT_SPAN_ID.fetch_add(1, Ordering::Relaxed))
    }

    fn record(&self, _span: &tracing::span::Id, _values: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _span: &tracing::span::Id, _follows: &tracing::span::Id) {}

    fn event(&self, event: &tracing::Event<'_>) {
        let mut msg = String::new();
        let meta = event.metadata();
        msg.push_str(meta.target());
        let mut visitor = FieldFormatter {
            out: &mut msg,
            seen_message: false,
        };
        event.record(&mut visitor);
        let level = match *meta.level() {
            Level::ERROR => "error",
            Level::WARN => "warn",
            Level::INFO => "info",
            // TS `LogLevel` has no "trace"; fold into debug.
            Level::DEBUG | Level::TRACE => "debug",
        };
        SINK.with(|s| {
            if let Some(f) = s.borrow().as_ref() {
                let _ = f.call2(
                    &JsValue::NULL,
                    &JsValue::from_str(level),
                    &msg.as_str().into(),
                );
            }
        });
    }

    fn enter(&self, _span: &tracing::span::Id) {}
    fn exit(&self, _span: &tracing::span::Id) {}
}

/// `target: message k=v k=v`. Fields append in record order; the `message`
/// field renders as the `: message` head (tracing records it first).
struct FieldFormatter<'a> {
    out: &'a mut String,
    seen_message: bool,
}

impl tracing::field::Visit for FieldFormatter<'_> {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn core::fmt::Debug) {
        use core::fmt::Write;
        if field.name() == "message" && !self.seen_message {
            self.seen_message = true;
            let _ = write!(self.out, ": {value:?}");
        } else {
            let _ = write!(self.out, " {}={value:?}", field.name());
        }
    }

    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        use core::fmt::Write;
        if field.name() == "message" && !self.seen_message {
            self.seen_message = true;
            let _ = write!(self.out, ": {value}");
        } else {
            let _ = write!(self.out, " {}={value}", field.name());
        }
    }
}
