import wasmInit, { setTraceLevel as wasmSetTraceLevel, setTraceSink } from "#wasm";

import { log } from "./types.js";
import type { TraceLevel, WasmSource } from "./types.js";

let initPromise: Promise<void> | undefined;

/** Idempotent wasm init; every public entry point awaits this first. Only
 * the first call's `wasmSource` matters: later calls join the in-flight (or
 * completed) init. */
export function ensureWasm(wasmSource?: WasmSource): Promise<void> {
  initPromise ??= wasmInit(
    wasmSource === undefined ? undefined : { module_or_path: wasmSource },
  ).then(() => {
    // Engine telemetry rides the same sink as library logs; `setLogger`
    // re-routes both. Events only flow once `setTraceLevel` opts in.
    setTraceSink(log);
  });
  return initPromise;
}

/** Max level for Rust-side `tracing` telemetry (per-stage timings, residency
 * traffic, ...). Events land on the `setLogger` sink. Default "off"; "info"
 * gives stage timings, "debug"/"trace" get very chatty. Each enabled trace
 * point costs a wasm->JS call, so leave this off in production. */
export async function setTraceLevel(level: TraceLevel): Promise<void> {
  await ensureWasm();
  wasmSetTraceLevel(level);
}

/** Optional explicit init: call early to override where the wasm blob comes
 * from, or just to front-load the fetch/compile before the first API call.
 * Never required; all entry points init lazily with defaults. */
export function init(opts: { wasmSource?: WasmSource } = {}): Promise<void> {
  return ensureWasm(opts.wasmSource);
}
