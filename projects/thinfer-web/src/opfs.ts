import { log } from "./types.js";
import type { WeightCache, WeightFile } from "./types.js";

export interface OpfsWeightCacheOptions {
  /** OPFS directory holding the cached files. Default "thinfer-weights". */
  dirName?: string;
  /** How file IO runs:
   * - "worker": a dedicated IO worker owning sync access handles. Fast path
   *   (main-thread `file.slice()` reads measure ~20x slower); the consumer's
   *   CSP needs `worker-src 'self'`.
   * - "inline": no extra worker. Sync access handles when already inside a
   *   worker (caller hosts the engine there), main-thread `File` reads
   *   otherwise.
   * - "auto" (default): "worker", falling back to "inline" when spawning
   *   fails (CSP, no `Worker` global). */
  io?: "auto" | "worker" | "inline";
}

/** OPFS-backed `WeightCache`, the browser default. Files live in one flat
 * OPFS directory; keys become file names via `encodeURIComponent`. A file is
 * visible to `open` only once its completion marker (`<file>#ok`) exists; the
 * marker is created after a fully successful `put`, so an interrupted put
 * reads as absent. Worker contexts write through sync access handles
 * (coalesced large writes); the main-thread fallback keeps `createWritable`. */
export class OpfsWeightCache implements WeightCache {
  readonly #dirName: string;
  readonly #io: "auto" | "worker" | "inline";
  #store?: Promise<OpfsStore>;

  constructor(opts: OpfsWeightCacheOptions = {}) {
    this.#dirName = opts.dirName ?? "thinfer-weights";
    this.#io = opts.io ?? "auto";
  }

  #resolve(): Promise<OpfsStore> {
    this.#store ??= openStore(this.#io, this.#dirName);
    return this.#store;
  }

  async open(key: string): Promise<WeightFile | null> {
    const store = await this.#resolve();
    const size = await store.openSize(key);
    if (size === null) {
      return null;
    }
    return {
      sizeBytes: size,
      readAt: (offset, length) => store.read(key, offset, length),
    };
  }

  async put(key: string, data: ReadableStream<Uint8Array>): Promise<void> {
    const store = await this.#resolve();
    const s = await store.put(key, data);
    const mibs = (ms: number) => (ms > 0 ? (s.bytes / 1024 ** 2 / (ms / 1000)).toFixed(0) : "-");
    log(
      "info",
      `opfs put ${key} [${store.kind}]: ${(s.bytes / 1024 ** 3).toFixed(2)} GiB in ` +
        `${s.chunks} chunks (avg ${s.chunks ? (s.bytes / s.chunks / 1024).toFixed(0) : 0} KiB, ` +
        `max ${(s.maxChunkBytes / 1024).toFixed(0)} KiB); ` +
        `net ${(s.netMs / 1000).toFixed(1)}s (${mibs(s.netMs)} MiB/s), ` +
        `write ${(s.writeMs / 1000).toFixed(1)}s (${mibs(s.writeMs)} MiB/s, ` +
        `max ${s.maxWriteMs.toFixed(0)}ms/chunk), ` +
        `open ${s.openMs.toFixed(0)}ms, close ${(s.closeMs / 1000).toFixed(1)}s`,
    );
  }

  async delete(key: string): Promise<void> {
    await (await this.#resolve()).delete(key);
  }

  /** Remove every cached file (the whole cache directory). */
  async clear(): Promise<void> {
    await (await this.#resolve()).clear();
  }

  /** Close all read handles, releasing their exclusive file locks so other
   * contexts (tabs) can read the cache. The engine calls this after
   * `loadModel` and after each generate; reads re-acquire on demand. */
  async releaseLocks(): Promise<void> {
    // Never opened a store: nothing can be locked, and resolving one here
    // would spawn the IO worker just to tell it to do nothing.
    if (!this.#store) {
      return;
    }
    await (await this.#store).releaseLocks();
  }
}

/** `put` timing breakdown: time awaiting the source stream (network for a
 * download) vs time awaiting disk writes, plus chunk-shape stats and the
 * fixed open (`createWritable`) / close (atomic swap) costs. For the worker
 * backend `netMs` includes cross-thread stream-transfer pulls. */
export interface PutStats {
  bytes: number;
  chunks: number;
  maxChunkBytes: number;
  netMs: number;
  writeMs: number;
  maxWriteMs: number;
  openMs: number;
  closeMs: number;
}

/** Backend-agnostic store ops. Implemented three ways: sync access handles
 * in this context (worker-hosted engine), RPC to a dedicated IO worker, or
 * main-thread `File` reads (slow fallback). */
interface OpfsStore {
  /** Which IO backend this is, for log attribution. */
  readonly kind: "worker" | "sync" | "main";
  openSize(key: string): Promise<number | null>;
  read(key: string, offset: number, length: number): Promise<Uint8Array>;
  put(key: string, data: ReadableStream<Uint8Array>): Promise<PutStats>;
  delete(key: string): Promise<void>;
  clear(): Promise<void>;
  /** Close every read handle (releasing the per-file exclusive locks).
   * Reads re-open on demand. */
  releaseLocks(): Promise<void>;
}

async function openStore(io: "auto" | "worker" | "inline", dirName: string): Promise<OpfsStore> {
  if (io !== "inline") {
    try {
      return await WorkerStore.spawn(dirName);
    } catch (e) {
      if (io === "worker") {
        throw e;
      }
      log("debug", `opfs io worker unavailable (${e}); using inline io`);
    }
  }
  // `createSyncAccessHandle` is only exposed in dedicated workers, so its
  // presence on the prototype IS the "are we in a worker" check.
  const sync =
    typeof FileSystemFileHandle !== "undefined" &&
    "createSyncAccessHandle" in FileSystemFileHandle.prototype;
  return sync ? new OpfsSyncStore(dirName) : new MainThreadStore(dirName);
}

const fileName = (key: string) => encodeURIComponent(key);

// "#" never appears in encodeURIComponent output, so a marker name cannot
// collide with any key's data file.
const markerName = (key: string) => `${fileName(key)}#ok`;

async function putMarker(dir: FileSystemDirectoryHandle, key: string): Promise<void> {
  await dir.getFileHandle(markerName(key), { create: true });
}

async function removeMarker(dir: FileSystemDirectoryHandle, key: string): Promise<void> {
  try {
    await dir.removeEntry(markerName(key));
  } catch {
    // Absent already.
  }
}

async function hasMarker(dirName: string, key: string): Promise<boolean> {
  try {
    const dir = await cacheDir(dirName, false);
    await dir.getFileHandle(markerName(key));
    return true;
  } catch {
    return false;
  }
}

async function cacheDir(dirName: string, create: boolean): Promise<FileSystemDirectoryHandle> {
  const root = await navigator.storage.getDirectory();
  return root.getDirectoryHandle(dirName, { create });
}

/** Streams `data` into `name` under `dir` via `createWritable`. Main-thread
 * fallback only (sync access handles are worker-only); known-slow per write
 * call. Times stream pulls vs disk writes separately so a slow download can
 * be attributed to network or disk. */
async function writeAtomic(
  dir: FileSystemDirectoryHandle,
  name: string,
  data: ReadableStream<Uint8Array>,
): Promise<PutStats> {
  const stats: PutStats = {
    bytes: 0,
    chunks: 0,
    maxChunkBytes: 0,
    netMs: 0,
    writeMs: 0,
    maxWriteMs: 0,
    openMs: 0,
    closeMs: 0,
  };
  let t = performance.now();
  const handle = await dir.getFileHandle(name, { create: true });
  const writable = await handle.createWritable({ keepExistingData: false });
  stats.openMs = performance.now() - t;
  const reader = data.getReader();
  try {
    for (;;) {
      t = performance.now();
      const { done, value } = await reader.read();
      stats.netMs += performance.now() - t;
      if (done) {
        break;
      }
      stats.bytes += value.byteLength;
      stats.chunks++;
      stats.maxChunkBytes = Math.max(stats.maxChunkBytes, value.byteLength);
      t = performance.now();
      // The cast strips a `SharedArrayBuffer` possibility the lib types
      // allow but fetch/transfer streams never produce.
      await writable.write(value as Uint8Array<ArrayBuffer>);
      const w = performance.now() - t;
      stats.writeMs += w;
      stats.maxWriteMs = Math.max(stats.maxWriteMs, w);
    }
    t = performance.now();
    await writable.close();
    stats.closeMs = performance.now() - t;
  } catch (e) {
    await writable.abort();
    throw e;
  }
  return stats;
}

// `createSyncAccessHandle` and its return type are worker-only APIs, absent
// from the "DOM" lib this package compiles against. Minimal local shape.
interface SyncAccessHandle {
  getSize(): number;
  read(buffer: Uint8Array, options?: { at?: number }): number;
  write(buffer: Uint8Array, options?: { at?: number }): number;
  truncate(newSize: number): void;
  flush(): void;
  close(): void;
}

/** Coalesce size for sync-access-handle writes: network chunks accumulate
 * into one buffer this big before hitting the handle, so the disk sees few
 * large writes. Per-call overhead is what made the `createWritable` path
 * slow (~108 MiB/s measured on desktop NVMe at ~192 KiB chunks). */
const COALESCE_BYTES = 16 * 1024 * 1024;

/** Streams `data` into an open sync access handle with coalesced writes.
 * NOT atomic on its own: callers gate visibility with the completion marker.
 * `closeMs` reports the final `flush()`. */
async function writeViaSyncHandle(
  handle: SyncAccessHandle,
  data: ReadableStream<Uint8Array>,
): Promise<PutStats> {
  const stats: PutStats = {
    bytes: 0,
    chunks: 0,
    maxChunkBytes: 0,
    netMs: 0,
    writeMs: 0,
    maxWriteMs: 0,
    openMs: 0,
    closeMs: 0,
  };
  const buf = new Uint8Array(COALESCE_BYTES);
  let fill = 0;
  let pos = 0;
  const drain = () => {
    const t = performance.now();
    const n = handle.write(buf.subarray(0, fill), { at: pos });
    if (n !== fill) {
      throw new Error(`opfs write: wrote ${n} of ${fill} bytes at ${pos}`);
    }
    const w = performance.now() - t;
    stats.writeMs += w;
    stats.maxWriteMs = Math.max(stats.maxWriteMs, w);
    pos += fill;
    fill = 0;
  };
  const reader = data.getReader();
  for (;;) {
    const t = performance.now();
    const { done, value } = await reader.read();
    stats.netMs += performance.now() - t;
    if (done) {
      break;
    }
    stats.bytes += value.byteLength;
    stats.chunks++;
    stats.maxChunkBytes = Math.max(stats.maxChunkBytes, value.byteLength);
    for (let off = 0; off < value.byteLength; ) {
      const n = Math.min(value.byteLength - off, COALESCE_BYTES - fill);
      buf.set(value.subarray(off, off + n), fill);
      fill += n;
      off += n;
      if (fill === COALESCE_BYTES) {
        drain();
      }
    }
  }
  if (fill > 0) {
    drain();
  }
  const t = performance.now();
  handle.flush();
  stats.closeMs = performance.now() - t;
  return stats;
}

interface SyncFileHandle extends FileSystemFileHandle {
  createSyncAccessHandle(): Promise<SyncAccessHandle>;
}

/** Sync-access-handle store. Worker contexts only. Handles stay open across
 * reads (open/close per read costs more than the read); `put`/`delete` close
 * first because both the handle and `createWritable` take the file lock. */
export class OpfsSyncStore implements OpfsStore {
  readonly kind = "sync";
  readonly #dirName: string;
  readonly #handles = new Map<string, SyncAccessHandle>();

  constructor(dirName: string) {
    this.#dirName = dirName;
  }

  async #handle(key: string): Promise<SyncAccessHandle | null> {
    const cached = this.#handles.get(key);
    if (cached) {
      return cached;
    }
    let handle: SyncAccessHandle;
    try {
      const dir = await cacheDir(this.#dirName, false);
      const file = (await dir.getFileHandle(fileName(key))) as SyncFileHandle;
      handle = await file.createSyncAccessHandle();
    } catch (e) {
      // Lock contention is not absence: the sync handle is an exclusive
      // per-file lock, so another context (tab) reading the same cache
      // surfaces here. Report it as what it is.
      if (e instanceof DOMException && e.name === "NoModificationAllowedError") {
        throw new Error(`opfs read ${key}: file locked by another context (another tab?)`);
      }
      return null;
    }
    this.#handles.set(key, handle);
    return handle;
  }

  #close(key: string): void {
    this.#handles.get(key)?.close();
    this.#handles.delete(key);
  }

  async openSize(key: string): Promise<number | null> {
    if (!(await hasMarker(this.#dirName, key))) {
      return null;
    }
    // Lock-free stat: a sync access handle takes the file's exclusive lock,
    // which would block every other context (e.g. an app checking presence
    // on the main thread while a worker-hosted engine holds read handles).
    // Only `read` opens and caches handles.
    try {
      const dir = await cacheDir(this.#dirName, false);
      return (await (await dir.getFileHandle(fileName(key))).getFile()).size;
    } catch {
      return null;
    }
  }

  async read(key: string, offset: number, length: number): Promise<Uint8Array> {
    const handle = await this.#handle(key);
    if (!handle) {
      throw new Error(`opfs read ${key}: file absent`);
    }
    const buf = new Uint8Array(length);
    const n = handle.read(buf, { at: offset });
    if (n !== length) {
      throw new Error(`opfs read ${key}: got ${n} bytes, wanted ${length} at ${offset}`);
    }
    return buf;
  }

  async put(key: string, data: ReadableStream<Uint8Array>): Promise<PutStats> {
    this.#close(key);
    const dir = await cacheDir(this.#dirName, true);
    // A failed or interrupted put must read as absent: drop the marker
    // before touching the data file, recreate it only on full success.
    await removeMarker(dir, key);
    const t = performance.now();
    const file = (await dir.getFileHandle(fileName(key), { create: true })) as SyncFileHandle;
    const handle = await file.createSyncAccessHandle();
    const openMs = performance.now() - t;
    let stats: PutStats;
    try {
      handle.truncate(0);
      stats = await writeViaSyncHandle(handle, data);
    } catch (e) {
      handle.close();
      await this.delete(key);
      throw e;
    }
    handle.close();
    await putMarker(dir, key);
    stats.openMs = openMs;
    return stats;
  }

  async delete(key: string): Promise<void> {
    this.#close(key);
    try {
      const dir = await cacheDir(this.#dirName, false);
      await removeMarker(dir, key);
      await dir.removeEntry(fileName(key));
    } catch {
      // Absent already: deletion is idempotent.
    }
  }

  async clear(): Promise<void> {
    await this.releaseLocks();
    try {
      const root = await navigator.storage.getDirectory();
      await root.removeEntry(this.#dirName, { recursive: true });
    } catch {
      // Absent already: clearing is idempotent.
    }
  }

  async releaseLocks(): Promise<void> {
    for (const handle of this.#handles.values()) {
      handle.close();
    }
    this.#handles.clear();
  }
}

/** Main-thread fallback: `File.slice().arrayBuffer()` reads. Known-slow
 * (browser blob IPC); only used when both worker spawn and sync access
 * handles are unavailable. */
class MainThreadStore implements OpfsStore {
  readonly kind = "main";
  readonly #dirName: string;

  constructor(dirName: string) {
    this.#dirName = dirName;
  }

  async #file(key: string): Promise<File | null> {
    try {
      const dir = await cacheDir(this.#dirName, false);
      return await (await dir.getFileHandle(fileName(key))).getFile();
    } catch {
      return null;
    }
  }

  async openSize(key: string): Promise<number | null> {
    if (!(await hasMarker(this.#dirName, key))) {
      return null;
    }
    return (await this.#file(key))?.size ?? null;
  }

  async read(key: string, offset: number, length: number): Promise<Uint8Array> {
    const file = await this.#file(key);
    if (!file) {
      throw new Error(`opfs read ${key}: file absent`);
    }
    return new Uint8Array(await file.slice(offset, offset + length).arrayBuffer());
  }

  async releaseLocks(): Promise<void> {
    // `getFile()` reads take no lock; nothing to release.
  }

  async put(key: string, data: ReadableStream<Uint8Array>): Promise<PutStats> {
    const dir = await cacheDir(this.#dirName, true);
    await removeMarker(dir, key);
    const stats = await writeAtomic(dir, fileName(key), data);
    await putMarker(dir, key);
    return stats;
  }

  async delete(key: string): Promise<void> {
    try {
      const dir = await cacheDir(this.#dirName, false);
      await removeMarker(dir, key);
      await dir.removeEntry(fileName(key));
    } catch {
      // Absent already: deletion is idempotent.
    }
  }

  async clear(): Promise<void> {
    try {
      const root = await navigator.storage.getDirectory();
      await root.removeEntry(this.#dirName, { recursive: true });
    } catch {
      // Absent already: clearing is idempotent.
    }
  }
}

// ---- IO worker RPC (client side; the worker entry is opfs-worker.ts) ----

/** Requests the client sends; `init` is the spawn handshake. */
type WorkerRequestBody =
  | { op: "init"; dirName: string }
  | { op: "openSize"; key: string }
  | { op: "read"; key: string; offset: number; length: number }
  | { op: "put"; key: string; data: ReadableStream<Uint8Array> }
  | { op: "delete"; key: string }
  | { op: "clear" }
  | { op: "releaseLocks" };

export type WorkerRequest = WorkerRequestBody & { id: number };

export type WorkerResponse =
  | { id: number; ok: true; value: unknown }
  | { id: number; ok: false; error: string };

class WorkerStore implements OpfsStore {
  readonly kind = "worker";
  readonly #worker: Worker;
  readonly #pending = new Map<
    number,
    { resolve: (v: unknown) => void; reject: (e: Error) => void }
  >();
  #nextId = 1;

  /** Spawns the worker and completes the `init` handshake; rejects (instead
   * of hanging) when CSP blocks the spawn, because a blocked worker fires
   * `error` and the handshake rejects every pending request. */
  static async spawn(dirName: string): Promise<WorkerStore> {
    if (typeof Worker === "undefined") {
      throw new Error("no Worker global");
    }
    const store = new WorkerStore(
      new Worker(new URL("./opfs-worker.js", import.meta.url), { type: "module" }),
    );
    await store.#request({ op: "init", dirName });
    return store;
  }

  constructor(worker: Worker) {
    this.#worker = worker;
    worker.onmessage = (ev: MessageEvent<WorkerResponse>) => {
      const resp = ev.data;
      const pending = this.#pending.get(resp.id);
      if (!pending) {
        return;
      }
      this.#pending.delete(resp.id);
      if (resp.ok) {
        pending.resolve(resp.value);
      } else {
        pending.reject(new Error(resp.error));
      }
    };
    worker.onerror = (ev) => {
      const err = new Error(`opfs io worker error: ${ev.message || "spawn blocked"}`);
      for (const p of this.#pending.values()) {
        p.reject(err);
      }
      this.#pending.clear();
    };
  }

  #request(req: WorkerRequestBody, transfer: Transferable[] = []): Promise<unknown> {
    const id = this.#nextId++;
    return new Promise((resolve, reject) => {
      this.#pending.set(id, { resolve, reject });
      this.#worker.postMessage({ id, ...req }, transfer);
    });
  }

  async openSize(key: string): Promise<number | null> {
    return (await this.#request({ op: "openSize", key })) as number | null;
  }

  async read(key: string, offset: number, length: number): Promise<Uint8Array> {
    return (await this.#request({ op: "read", key, offset, length })) as Uint8Array;
  }

  async put(key: string, data: ReadableStream<Uint8Array>): Promise<PutStats> {
    // ReadableStream transfer moves the network pull into the worker too.
    return (await this.#request({ op: "put", key, data }, [data])) as PutStats;
  }

  async delete(key: string): Promise<void> {
    await this.#request({ op: "delete", key });
  }

  async clear(): Promise<void> {
    await this.#request({ op: "clear" });
  }

  async releaseLocks(): Promise<void> {
    // The worker handles requests in arrival order, so this lands after
    // every read already enqueued.
    await this.#request({ op: "releaseLocks" });
  }
}
