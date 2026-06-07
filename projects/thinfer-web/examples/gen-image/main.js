// Example consumer of the thinfer library. Bundler-free: "thinfer" resolves
// via the import map in index.html. Downloads and cache checks run here
// (network-bound, import-map friendly); generation RPCs to gen-worker.js so
// main-thread jank/throttling never stalls the engine, and nothing GPU-side
// outlives a generate.
import { OpfsWeightCache, downloadModelFiles, missingModelFiles, setLogger } from "thinfer";

const $ = (id) => document.getElementById(id);
const status = (text) => {
  $("status").textContent = text;
};

// Mirror library logs into the page so they are visible on mobile, where
// there is no dev console. Worker-side logs arrive as "log" messages below.
const appendLog = (level, message) => {
  console[level](message);
  const log = $("log");
  log.value += `[${level}] ${message}\n`;
  log.scrollTop = log.scrollHeight;
};
setLogger(appendLog);

// Generation worker RPC: requests carry an id, responses echo it back;
// unsolicited {type: "log" | "progress"} events stream alongside.
const worker = new Worker("./gen-worker.js", { type: "module" });
const pending = new Map();
let nextId = 1;
const call = (op, body) =>
  new Promise((resolve, reject) => {
    pending.set(nextId, { resolve, reject });
    worker.postMessage({ id: nextId++, op, ...body });
  });
worker.onmessage = ({ data: msg }) => {
  if (msg.type === "log") {
    appendLog(msg.level, msg.message);
    return;
  }
  if (msg.type === "progress") {
    status(progressText(msg.ev));
    return;
  }
  const p = pending.get(msg.id);
  pending.delete(msg.id);
  if (msg.ok) {
    p.resolve(msg.value);
  } else {
    p.reject(new Error(msg.error));
  }
};

const cache = new OpfsWeightCache();

async function refreshModelState() {
  $("download").disabled = true;
  $("gen-fields").disabled = true;
  $("model-status").textContent = "Checking cache…";
  const missing = await missingModelFiles($("model").value, { cache });
  if (missing.length === 0) {
    $("model-status").textContent = "Model cached.";
    $("gen-fields").disabled = false;
  } else {
    $("model-status").textContent = `${missing.length} file(s) to download.`;
    $("download").disabled = false;
  }
}

$("model").addEventListener("change", () => void refreshModelState());

// Engine telemetry (Rust tracing) lives in the worker's wasm instance; its
// events ride the worker "log" messages into the textarea above.
$("trace").addEventListener("change", () => {
  void call("setTrace", { level: $("trace").value });
});

$("download").addEventListener("click", async () => {
  $("download").disabled = true;
  try {
    await downloadModelFiles($("model").value, {
      cache,
      onProgress: ({ file, loadedBytes, totalBytes }) => {
        const pct = totalBytes ? ` ${Math.floor((loadedBytes / totalBytes) * 100)}%` : "";
        status(`Downloading ${file.path}${pct}`);
      },
    });
    status("Download complete.");
  } catch (e) {
    status(`Download failed: ${e}`);
  }
  await refreshModelState();
});

// The worker's engine holds the OPFS read handles; clearing the cache must
// run where the handles live so they close before the files are removed.
$("clear-cache").addEventListener("click", async () => {
  $("clear-cache").disabled = true;
  try {
    await call("clearCache");
    status("Cache cleared.");
  } catch (e) {
    status(`Clear failed: ${e}`);
  }
  $("clear-cache").disabled = false;
  await refreshModelState();
});

$("form").addEventListener("submit", (ev) => {
  ev.preventDefault();
  void generate();
});

const progressText = (ev) => {
  switch (ev.type) {
    case "textEncode":
      return "Encoding prompt";
    case "step":
      return `Denoising step ${ev.i}/${ev.n}`;
    case "vaeDecode":
      return "Decoding latents (VAE)";
  }
};

async function generate() {
  $("gen-fields").disabled = true;
  const t0 = performance.now();
  status("Loading model…");
  try {
    const { png, seed } = await call("generate", {
      model: $("model").value,
      budgetGib: Number($("budget").value),
      params: {
        prompt: $("prompt").value,
        width: Number($("width").value),
        height: Number($("height").value),
        steps: Number($("steps").value),
        seed: $("seed").value || null,
      },
    });
    const img = $("output");
    if (img.src) URL.revokeObjectURL(img.src);
    img.src = URL.createObjectURL(new Blob([png], { type: "image/png" }));
    status(`Done: seed ${seed}, ${((performance.now() - t0) / 1000).toFixed(1)}s.`);
  } catch (e) {
    status(`Generation failed: ${e}`);
  } finally {
    $("gen-fields").disabled = false;
  }
}

void refreshModelState();
