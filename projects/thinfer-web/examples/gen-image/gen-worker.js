// Engine host worker: generation runs here so main-thread jank/throttling
// (acute on mobile) never stalls the engine; only control messages and the
// finished PNG (transferred, not copied) cross to the page. The library is
// context-agnostic - hosting it in a worker is this app's choice. Import
// maps don't apply in workers, so the library is imported by path (its
// emitted modules use relative specifiers only).
import { OpfsWeightCache, createEngine, setLogger, setTraceLevel } from "/dist/index.js";

const post = (msg, transfer = []) => self.postMessage(msg, transfer);

// Engine logs and Rust tracing stream to the page; the worker has no UI.
setLogger((level, message) => post({ type: "log", level, message }));

const cache = new OpfsWeightCache();

const ops = {
  setTrace: ({ level }) => setTraceLevel(level),

  // The engine's OPFS read handles live in this context; only their owner
  // can close them before deleting the files, so cache clearing runs here.
  clearCache: () => cache.clear(),

  // Nothing GPU-side outlives a generate: engine and model are created on
  // entry and destroyed on exit (success or failure), so an idle page holds
  // no device, pipelines, or buffers, and a crashed device never leaves
  // zombie handles behind.
  generate: async ({ model, budgetGib, params }) => {
    const bytes = budgetGib * 1024 ** 3;
    const engine = await createEngine({ cache, budgets: { ramBytes: bytes, vramBytes: bytes } });
    try {
      const loaded = await engine.loadModel(model);
      try {
        return await loaded.generateImage({
          ...params,
          seed: params.seed ? BigInt(params.seed) : undefined,
          onProgress: (ev) => post({ type: "progress", ev }),
        });
      } finally {
        loaded.destroy();
      }
    } finally {
      engine.destroy();
    }
  },
};

self.onmessage = async ({ data: req }) => {
  try {
    const value = await ops[req.op](req);
    post({ id: req.id, ok: true, value }, value?.png ? [value.png.buffer] : []);
  } catch (e) {
    post({ id: req.id, ok: false, error: String(e) });
  }
};
