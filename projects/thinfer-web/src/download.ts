import { modelFilesJson } from "#wasm";

import { OpfsWeightCache } from "./opfs.js";
import { log } from "./types.js";
import type { DownloadOptions, ModelFile, ModelId, WeightCache } from "./types.js";
import { ensureWasm } from "./wasm.js";

const HF_BASE = "https://huggingface.co";

/** Wire shape of `modelFilesJson` entries (see wasm-pkg.d.ts). */
interface RawFile {
  role: string;
  repo: string;
  path: string;
  revision?: string;
}

/** All files `model` needs, with CORS-enabled Hugging Face hub URLs. The
 * manifest is single-sourced from the Rust model registry inside the wasm. */
export async function modelFiles(
  model: ModelId,
  opts?: { urlBase?: string },
): Promise<ModelFile[]> {
  await ensureWasm();
  const raw = JSON.parse(modelFilesJson(model)) as RawFile[];
  const base = opts?.urlBase ?? HF_BASE;
  return raw.map((f) => ({
    ...f,
    key: `${f.repo}/${f.path}@${f.revision ?? "main"}`,
    url: `${base}/${f.repo}/resolve/${f.revision ?? "main"}/${f.path}`,
  }));
}

/** Subset of `modelFiles(model)` not present in the cache. Empty result
 * means `loadModel` will not throw `MissingModelFilesError`. */
export async function missingModelFiles(
  model: ModelId,
  opts?: { cache?: WeightCache; urlBase?: string },
): Promise<ModelFile[]> {
  const files = await modelFiles(model, opts);
  const cache = opts?.cache ?? new OpfsWeightCache();
  const present = await Promise.all(files.map(async (f) => (await cache.open(f.key)) !== null));
  return files.filter((_, i) => !present[i]);
}

/** Explicit download helper: fetches `missingModelFiles(model)` into the
 * cache, sequentially (these are multi-GB files; parallelism buys nothing on
 * a bandwidth-bound path). The library never downloads implicitly; call this
 * (or run your own fetch over `modelFiles(...)[].url`) before `loadModel`. */
export async function downloadModelFiles(
  model: ModelId,
  opts: DownloadOptions = {},
): Promise<void> {
  const cache = opts.cache ?? new OpfsWeightCache();
  for (const file of await missingModelFiles(model, { cache, urlBase: opts.urlBase })) {
    const resp = await fetch(file.url, { signal: opts.signal });
    if (!resp.ok || !resp.body) {
      throw new Error(`GET ${file.url}: ${resp.status} ${resp.statusText}`);
    }
    const contentLength = resp.headers.get("content-length");
    const totalBytes = contentLength ? Number(contentLength) : undefined;
    log(
      "info",
      `downloading ${file.key}${totalBytes ? ` (${(totalBytes / 1024 ** 3).toFixed(2)} GiB)` : ""}`,
    );
    let loadedBytes = 0;
    // Log each 10% boundary; a single chunk can cross several, so emit every
    // decile between the previous one and the current one.
    let lastDecile = 0;
    const counted = resp.body.pipeThrough(
      new TransformStream<Uint8Array, Uint8Array>({
        transform(chunk, controller) {
          loadedBytes += chunk.byteLength;
          if (totalBytes) {
            const decile = Math.min(10, Math.floor((loadedBytes * 10) / totalBytes));
            for (; lastDecile < decile; lastDecile++) {
              log("info", `${file.key}: ${(lastDecile + 1) * 10}%`);
            }
          }
          opts.onProgress?.({ file, loadedBytes, totalBytes });
          controller.enqueue(chunk);
        },
      }),
    );
    await cache.put(file.key, counted);
    log("info", `${file.key}: download done`);
  }
}
