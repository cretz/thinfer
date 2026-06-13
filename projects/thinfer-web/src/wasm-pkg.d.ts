// Ambient contract for the wasm-bindgen output (built into pkg/ by
// `pnpm build:wasm`; resolved via the package.json "#wasm" subpath import).
// Package-internal: `#`-prefixed subpath imports are private by node spec
// and nothing in the public `exports` map reaches pkg/, so this surface
// (including its positional-args style, chosen to keep the wasm boundary
// free of JsValue object plumbing) is not consumer-visible.
// Hand-maintained mirror of the #[wasm_bindgen] surface in src/lib.rs so
// typecheck never requires the wasm to be built first. Keep in sync with the
// Rust exports; drift fails at runtime, not compile time.
declare module "#wasm" {
  /** wasm-bindgen-generated init: fetches and instantiates the wasm blob.
   * Defaults to `new URL("thinfer_web_bg.wasm", import.meta.url)`. */
  export default function init(opts?: {
    module_or_path?: string | URL | Request | Response | WebAssembly.Module;
  }): Promise<unknown>;

  /** JSON `[{ role, repo, path, revision? }]` for a model id. Throws on an
   * unknown id. */
  export function modelFilesJson(modelId: string): string;

  /** Rust `tracing` max level. Throws on an unknown level string. */
  export function setTraceLevel(level: "off" | "error" | "warn" | "info" | "debug" | "trace"): void;

  /** Sink for forwarded `tracing` events ("trace" folds into "debug").
   * `undefined` drops events. */
  export function setTraceSink(
    sink: ((level: "debug" | "info" | "warn" | "error", message: string) => void) | undefined,
  ): void;

  export class WasmEngine {
    static create(
      powerPreference: string | undefined,
      ramBudgetBytes: bigint,
      vramBudgetBytes: bigint,
    ): Promise<WasmEngine>;
    /** `roles[i]` names the role of `files[i]`; each file is a JS object
     * satisfying the `WeightFile` duck type (sizeBytes + readAt). */
    loadModel(modelId: string, roles: string[], files: unknown[]): Promise<WasmModel>;
    /** wasm-bindgen-generated: drops the Rust object. */
    free(): void;
  }

  export class WasmModel {
    /** Resolves to encoded PNG bytes. */
    generate(
      prompt: string,
      width: number,
      height: number,
      steps: number,
      seed: bigint,
      onProgress?: (ev: { type: string; i?: number; n?: number }) => void,
    ): Promise<Uint8Array>;
    /** wasm-bindgen-generated: drops the Rust object. */
    free(): void;
  }
}
