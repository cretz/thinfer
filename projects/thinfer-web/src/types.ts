// Public API types. No DOM types anywhere in this file (or the public API):
// the library must stay usable from workers and from Node/Deno hosts that
// polyfill `navigator.gpu` (Dawn's `webgpu` package, Deno's built-in).

/** Model identifiers. Mirrors the Rust `ModelId` registry. */
export type ModelId = "zimage-turbo-q8" | "zimage-turbo-q4" | "zimage-turbo-bf16";

/** One file a model needs. `url` is CORS-enabled on the Hugging Face hub, so
 * implementers can fetch it directly (or via `downloadModelFiles`). */
export interface ModelFile {
  /** Stable cache key: `${repo}/${path}@${revision ?? "main"}`. */
  key: string;
  /** Role within the model (e.g. "dit/shard-1", "tokenizer"). */
  role: string;
  repo: string;
  path: string;
  revision?: string;
  url: string;
}

/** Random-access handle to one cached weight file. `readAt` returns bytes in
 * the JS heap; the engine uploads them to the GPU without staging them in
 * wasm linear memory. */
export interface WeightFile {
  sizeBytes: number;
  readAt(offset: number, length: number): Promise<Uint8Array>;
}

/** Pluggable weight storage. The browser default is OPFS
 * (`OpfsWeightCache`); Node hosts can supply an fs-backed impl. */
export interface WeightCache {
  /** `null` when the key is absent. */
  open(key: string): Promise<WeightFile | null>;
  /** Store a file under `key`, replacing any existing content. Must be
   * atomic: a partial write (abort, crash) must not leave `open` returning
   * truncated data. */
  put(key: string, data: ReadableStream<Uint8Array>): Promise<void>;
  delete(key: string): Promise<void>;
  /** Release any file locks held for reading (OPFS sync access handles are
   * exclusive per file). The engine calls this after `loadModel` and after
   * each generate, so locks are held only while reads are actually running
   * and other contexts (tabs) can read the same cache in between. Reads
   * re-acquire on demand. Optional: lock-free caches omit it. */
  releaseLocks?(): Promise<void>;
}

export type LogLevel = "debug" | "info" | "warn" | "error";

/** Max level for Rust-side `tracing` telemetry (see `setTraceLevel`). */
export type TraceLevel = "off" | LogLevel | "trace";

/** Library-wide log sink, installed via `setLogger`. The library does not
 * filter by level; sinks decide what to keep. Default sink is the console. */
export type Logger = (level: LogLevel, message: string) => void;

const consoleLogger: Logger = (level, message) => console[level](message);

let currentLogger: Logger = consoleLogger;

/** Replace the library-wide log sink (`null` restores the console default).
 * Global rather than per-engine: Rust-side `tracing` (bridged into this sink
 * at debug level) only supports a process-global subscriber. */
export function setLogger(logger: Logger | null): void {
  currentLogger = logger ?? consoleLogger;
}

/** Internal: emit to the installed sink. */
export function log(level: LogLevel, message: string): void {
  currentLogger(level, message);
}

/** Generation progress. Mirrors the Rust `ProgressEvent`. */
export type ProgressEvent =
  | { type: "textEncode" }
  | { type: "step"; i: number; n: number }
  | { type: "vaeDecode" };

export interface GenerateImageParams {
  prompt: string;
  /** Pixels, multiple of 16. Default 768. */
  width?: number;
  /** Pixels, multiple of 16. Default 768. */
  height?: number;
  /** Denoising steps. Default 8 (Z-Image-Turbo). */
  steps?: number;
  /** Omit for a random seed; the seed used is always reported back. */
  seed?: number | bigint;
  onProgress?: (ev: ProgressEvent) => void;
}

export interface GenerateImageResult {
  /** Encoded PNG bytes. */
  png: Uint8Array;
  seed: bigint;
}

export interface DownloadProgressEvent {
  file: ModelFile;
  loadedBytes: number;
  /** Absent when the server omits Content-Length. */
  totalBytes?: number;
}

export interface DownloadOptions {
  cache?: WeightCache;
  onProgress?: (ev: DownloadProgressEvent) => void;
  signal?: AbortSignal;
  /** Override the Hugging Face hub origin (self-hosted mirrors). */
  urlBase?: string;
}

/** Acceptable locations/forms of the wasm blob. */
export type WasmSource = string | URL | Request | Response | WebAssembly.Module;

export interface CreateEngineOptions {
  /** Where to fetch the wasm blob from. Default resolves next to the JS glue
   * via `new URL(..., import.meta.url)`, which every major bundler rewrites
   * correctly. Escape hatch for CDNs / strict CSP. */
  wasmSource?: WasmSource;
  /** WebGPU adapter preference. Default "high-performance", matching the
   * CLI: drivers treat an unset preference as a background-priority hint
   * (clamped clocks on some iGPUs). */
  powerPreference?: "low-power" | "high-performance";
  cache?: WeightCache;
  /** Residency budgets in bytes (ceilings, not reservations: a smaller
   * budget pages weights more, it does not fail). Independent ram/vram.
   * Default 4 GiB each: web IO is ~20x slower than native, so paging is
   * costly and a larger default keeps the q4 DiT mostly resident. */
  budgets?: { ramBytes?: number; vramBytes?: number };
}

/** Thrown by `loadModel` when required files are not in the cache. There is
 * deliberately no lazy download: callers own the download UX and either run
 * their own fetch over `missing[].url` or call `downloadModelFiles`. */
export class MissingModelFilesError extends Error {
  readonly missing: ModelFile[];
  constructor(model: ModelId, missing: ModelFile[]) {
    super(
      `model ${model} is missing ${missing.length} cached file(s): ` +
        missing.map((f) => f.key).join(", "),
    );
    this.name = "MissingModelFilesError";
    this.missing = missing;
  }
}
