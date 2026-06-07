import { WasmEngine } from "#wasm";
import type { WasmModel } from "#wasm";

import { modelFiles } from "./download.js";
import { OpfsWeightCache } from "./opfs.js";
import { MissingModelFilesError, log } from "./types.js";
import type {
  CreateEngineOptions,
  GenerateImageParams,
  GenerateImageResult,
  ModelId,
  ProgressEvent,
  WeightCache,
} from "./types.js";
import { ensureWasm } from "./wasm.js";

/** Default residency ceiling, both tiers. Native's CLI floor is 2 GiB, but
 * web IO is ~20x slower so paging is expensive; 4 GiB keeps the q4 DiT
 * mostly resident on a mid GPU. Caller overrides each tier independently. */
const DEFAULT_BUDGET_BYTES = 4 * 1024 ** 3;

export async function createEngine(opts: CreateEngineOptions = {}): Promise<Engine> {
  await ensureWasm(opts.wasmSource);
  const inner = await WasmEngine.create(
    opts.powerPreference,
    BigInt(opts.budgets?.ramBytes ?? DEFAULT_BUDGET_BYTES),
    BigInt(opts.budgets?.vramBytes ?? DEFAULT_BUDGET_BYTES),
  );
  log("debug", "engine created");
  return new Engine(inner, opts.cache ?? new OpfsWeightCache());
}

export class Engine {
  readonly #inner: WasmEngine;
  readonly #cache: WeightCache;

  /** Internal; use `createEngine`. */
  constructor(inner: WasmEngine, cache: WeightCache) {
    this.#inner = inner;
    this.#cache = cache;
  }

  /** Loads a model whose files are already in the cache. Heavy (catalog
   * parse, weight registration, pipeline compiles); hold the returned
   * `Model` and reuse it across generates. Never downloads: throws
   * `MissingModelFilesError` when files are absent. */
  async loadModel(model: ModelId): Promise<Model> {
    const files = await modelFiles(model);
    const opened = await Promise.all(
      files.map(async (file) => ({ file, weight: await this.#cache.open(file.key) })),
    );
    const missing = opened.filter((o) => o.weight === null).map((o) => o.file);
    if (missing.length > 0) {
      throw new MissingModelFilesError(model, missing);
    }
    const t0 = performance.now();
    let inner: WasmModel;
    try {
      inner = await this.#inner.loadModel(
        model,
        opened.map((o) => o.file.role),
        opened.map((o) => o.weight),
      );
    } finally {
      // Read locks are held only while reads run: release between
      // operations so other contexts (tabs) can read the same cache.
      await this.#cache.releaseLocks?.();
    }
    log("debug", `model ${model} loaded in ${((performance.now() - t0) / 1000).toFixed(1)}s`);
    return new Model(inner, this.#cache);
  }

  /** Destroys the engine: the instance is unusable afterwards. The GPU
   * device is released once every model loaded from it is destroyed too.
   * Deterministic teardown needs this explicit call (or `using`); GC alone
   * frees wasm objects at an arbitrary later time. */
  destroy(): void {
    this.#inner.free();
  }

  [Symbol.dispose](): void {
    this.destroy();
  }
}

export class Model {
  readonly #inner: WasmModel;
  readonly #cache: WeightCache;

  /** Internal; use `Engine.loadModel`. */
  constructor(inner: WasmModel, cache: WeightCache) {
    this.#inner = inner;
    this.#cache = cache;
  }

  async generateImage(params: GenerateImageParams): Promise<GenerateImageResult> {
    const width = params.width ?? 768;
    const height = params.height ?? 768;
    const steps = params.steps ?? 8;
    validateDim("width", width);
    validateDim("height", height);
    if (!Number.isInteger(steps) || steps < 1) {
      throw new Error(`steps must be a positive integer (got ${steps})`);
    }
    const seed = params.seed === undefined ? randomSeed() : BigInt(params.seed);
    const onProgress = params.onProgress;
    let png: Uint8Array;
    try {
      png = await this.#inner.generate(
        params.prompt,
        width,
        height,
        steps,
        seed,
        onProgress && ((ev) => onProgress(ev as ProgressEvent)),
      );
    } finally {
      // See Engine.loadModel: locks live only as long as the operation.
      await this.#cache.releaseLocks?.();
    }
    return { png, seed };
  }

  /** Destroys the model (compiled pipelines, resident weights): the
   * instance is unusable afterwards. See `Engine.destroy`. */
  destroy(): void {
    this.#inner.free();
  }

  [Symbol.dispose](): void {
    this.destroy();
  }
}

/** Uniform random u64. `crypto` is global in every supported host (browser,
 * worker, Node, Deno); no DOM dependency. */
function randomSeed(): bigint {
  return crypto.getRandomValues(new BigUint64Array(1))[0];
}

function validateDim(name: string, v: number): void {
  // Same rule as native: the VAE needs dims divisible by 16.
  if (!Number.isInteger(v) || v <= 0 || v % 16 !== 0) {
    throw new Error(`${name} must be a positive multiple of 16 (got ${v})`);
  }
}
