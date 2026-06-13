// Dedicated IO worker entry (spawned by `WorkerStore` in opfs.ts). Owns the
// sync access handles: reads block this thread, not the engine's, and
// `will_read` prefetches overlap disk IO with GPU work on the engine thread.

import { OpfsSyncStore } from "./opfs.js";
import type { WorkerRequest, WorkerResponse } from "./opfs.js";

// Dedicated-worker globals, absent from the "DOM" lib this package compiles
// against. Minimal local shape.
const ctx = globalThis as unknown as {
  onmessage: ((ev: MessageEvent<WorkerRequest>) => void) | null;
  postMessage(msg: WorkerResponse, transfer?: Transferable[]): void;
};

let store: OpfsSyncStore;

async function handle(req: WorkerRequest): Promise<{ value: unknown; transfer: Transferable[] }> {
  switch (req.op) {
    case "init":
      store = new OpfsSyncStore(req.dirName);
      return { value: null, transfer: [] };
    case "openSize":
      return { value: await store.openSize(req.key), transfer: [] };
    case "read": {
      const bytes = await store.read(req.key, req.offset, req.length);
      return { value: bytes, transfer: [bytes.buffer] };
    }
    case "put":
      return { value: await store.put(req.key, req.data), transfer: [] };
    case "delete":
      await store.delete(req.key);
      return { value: null, transfer: [] };
    case "clear":
      await store.clear();
      return { value: null, transfer: [] };
    case "releaseLocks":
      await store.releaseLocks();
      return { value: null, transfer: [] };
  }
}

ctx.onmessage = (ev) => {
  const req = ev.data;
  void handle(req).then(
    ({ value, transfer }) => ctx.postMessage({ id: req.id, ok: true, value }, transfer),
    (e: unknown) => ctx.postMessage({ id: req.id, ok: false, error: String(e) }),
  );
};
