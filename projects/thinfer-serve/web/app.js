// Server-backed thinfer UI. Generation runs on the thinfer-serve box this page
// is served from: POST a job, tail its SSE event stream, then download the
// finished artifact. All three calls go through fetch (not EventSource) so the
// optional bearer token rides an Authorization header the whole way.
//
// Privacy: the browser generates an RSA keypair and sends only the PUBLIC key
// with each job. The server encrypts the result with it and cannot decrypt --
// only this browser's private key (kept in memory, never sent) can. The result
// is fetched as ciphertext, decrypted here, and shown from an in-memory blob;
// nothing decrypted ever touches the server. WebCrypto needs a secure context
// (https or localhost); over plain http we warn and fall back to plaintext.

const $ = (id) => document.getElementById(id);
const setStatus = (text) => {
  $("status").textContent = text;
};
// Set when a generation starts; log lines are then prefixed with elapsed-from-
// start, mirroring the CLI's stamped stderr sink (`[  12.3s] ...`, width-6 like
// the CLI's `{:6.1}`). Null before/after a run, so pre-gen warnings are unstamped.
let genStart = null;
const stamp = () => (genStart === null ? "" : `[${((performance.now() - genStart) / 1000).toFixed(1).padStart(6)}s] `);
const log = (line) => {
  const el = $("log");
  el.value += `${stamp()}${line}\n`;
  el.scrollTop = el.scrollHeight;
};

const MODELS = {
  image: ["zimage-turbo-q4", "zimage-turbo-q8", "zimage-turbo-bf16", "ideogram4-q8", "qwen-image-rapid", "qwen-image-edit-rapid"],
  video: ["fastwan-ti2v-5b", "wan2.2-t2v-a14b", "hunyuan-video-1.5-t2v", "hunyuan-video-1.5-ti2v", "longlive-2.0-5b", "ltx-2.3-distilled", "ltx-2.3-distilled-q4", "sulphur-2", "sulphur-2-q4"],
};
// LTX-2.3 is a joint audio-video model with its own grid (/64 dims, 8k+1 frames).
// Its video VAE decode tiles to the VRAM budget, so larger dims are allowed (more
// tiles = slower; the temporal dim is whole, so keep clips short). Dims default
// to upstream's distilled 768x512 (3:2 landscape, what the model was distilled
// for); 256x256 is the explicit "fastest" floor -- below that the model is out of
// distribution and output is incoherent. LTX also exposes an Audio toggle.
// Sulphur-2 is an LTX-2.3 DiT finetune: byte-identical architecture, same grid +
// audio path, so it shares the whole LTX UI surface.
const isLtxModel = (model) => model.startsWith("ltx-2.3-distilled") || model.startsWith("sulphur-2");
// Wan2.2-T2V-A14B (MoE 14B): a heavier Wan-family tier with its own surface. It
// runs a fixed 4-step LightX2V distill schedule (the steps/sampler knobs do not
// apply) and only the full Wan2.1 VAE (no tiny-VAE path), so the Steps + Quality
// rows are hidden for it. Defaults to 832x480 (the industry-norm 480p distill
// regime). Longer clips up to the model's ~5s (81f) envelope are allowed via the
// duration field, but on the 8GB card the 14B self-attention is O(rows^2), so
// longer = slower (the default 33f ~2.1s is the longest length validated e2e).
const isWan22Model = (model) => model === "wan2.2-t2v-a14b";
// HunyuanVideo 1.5 T2V: its own pipeline (Qwen2.5-VL encoder, dual-stream MMDiT,
// causal-conv VAE). Fixed lightx2v 4-step flow-match schedule (the steps/sampler
// knobs do not apply) and its own VAE (no tiny path), so those rows are hidden.
// /16 grid, industry-norm 832x480 / 81f @ 16fps default.
const isHunyuanModel = (model) => model.startsWith("hunyuan-video-1.5");
const isHunyuanI2vModel = (model) => model === "hunyuan-video-1.5-ti2v";
// Multi-shot video: only LongLive treats each prompt line as a separate shot.
// Every other video model is single-prompt (the whole box is one prompt).
const isMultiShotModel = (model) => model === "longlive-2.0-5b";
// Image models that edit a reference image instead of pure text-to-image: they
// REQUIRE an uploaded input image (mirrors the server's QwenImageEdit kind).
const EDIT_MODELS = new Set(["qwen-image-edit-rapid"]);
const isEditModel = (model) => EDIT_MODELS.has(model);
// Size presets per kind. First entry is the default. Hand-typed dims are still
// allowed and flip the dropdown to "Custom". Dim rules differ by kind: image
// (Z-Image VAE) needs /16, video (Wan2.2 16x16x4 VAE + patch 2) needs /32 --
// `DIM_STEP` enforces it on both the presets and the number inputs.
// Video presets follow Wan2.2-TI2V-5B's trained aspects (720p is natively
// 1280x704 / 704x1280); the ladder steps down same-aspect for speed on the 8GB
// card. Avoid square for video -- it is off the trained aspect and degrades
// composition. Smaller = fewer latent tokens = faster (cost scales with w*h).
const DIM_STEP = { image: 16, video: 32 };
const PRESETS = {
  image: [
    { label: "768x768 (default)", width: 768, height: 768 },
    { label: "1024x1024 (max detail, slower)", width: 1024, height: 1024 },
    { label: "512x512 (fast)", width: 512, height: 512 },
    { label: "768x1024 portrait", width: 768, height: 1024 },
    { label: "1024x768 landscape", width: 1024, height: 768 },
  ],
  video: [
    { label: "540p landscape (960x544, default)", width: 960, height: 544 },
    { label: "720p landscape (1280x704, native, slowest)", width: 1280, height: 704 },
    { label: "480p landscape (832x480)", width: 832, height: 480 },
    { label: "small landscape (640x352, fast)", width: 640, height: 352 },
    { label: "tiny landscape (512x288, fastest)", width: 512, height: 288 },
    { label: "720p portrait (704x1280, native, slowest)", width: 704, height: 1280 },
    { label: "540p portrait (544x960)", width: 544, height: 960 },
    { label: "480p portrait (480x832)", width: 480, height: 832 },
  ],
};
// LTX-2.3 has its own grid: dims must be a multiple of 64 and at least 256 (the
// server rejects below 256). The decode tiles to the VRAM budget so larger dims
// are allowed. Default first = 1280x704 (16:9 widescreen), the regime the
// distilled model is in-distribution for: lower-res LTX stays coherent but
// renders the wrong subject/action. The widescreen default is reached via the
// two-stage upscale path (auto-enabled for LTX). The low-res entries are kept as
// fast/preview options but are out of distribution -- expect off-prompt content.
// All /64.
const LTX_MIN_DIM = 256;
const LTX_PRESETS = [
  { label: "720p landscape (1280x704, default)", width: 1280, height: 704 },
  { label: "720p portrait (704x1280)", width: 704, height: 1280 },
  { label: "960x576 landscape (faster)", width: 960, height: 576 },
  { label: "768x512 landscape (fast, less faithful)", width: 768, height: 512 },
  { label: "512x320 landscape (fastest, off-prompt / OOD)", width: 512, height: 320 },
];
// Wan2.2-T2V-A14B size ladder (/16 grid, the Wan2.1 VAE 8x + patch 2). The
// default is 832x480, the industry-norm 480p distill regime. (The earlier
// low-res default existed only to dodge a device-loss that turned out to be the
// 2s Windows GPU watchdog tripping on one long self-attention dispatch, now fixed
// by per-dispatch query chunking; it was never a resolution/VRAM ceiling.) Clip
// length is a wall-time choice now: the server defaults to 33f and allows up to
// the 81f envelope via the duration field. First = default.
const WAN22_PRESETS = [
  { label: "480p landscape (832x480, default)", width: 832, height: 480 },
  { label: "480p portrait (480x832)", width: 480, height: 832 },
  { label: "small landscape (640x384, faster)", width: 640, height: 384 },
  { label: "288p landscape (512x288, fastest)", width: 512, height: 288 },
  { label: "288p portrait (288x512)", width: 288, height: 512 },
];
// HunyuanVideo 1.5 size ladder (/16 grid: 16x spatial VAE, patch 1). Default =
// 832x480, the model's native 480p T2V regime. Smaller = fewer latent tokens =
// faster on the 8GB card (cost scales with w*h and frames).
const HUNYUAN_PRESETS = [
  { label: "480p landscape (832x480, default)", width: 832, height: 480 },
  { label: "480p portrait (480x832)", width: 480, height: 832 },
  { label: "small landscape (640x368, faster)", width: 640, height: 368 },
  { label: "288p landscape (512x288, fastest)", width: 512, height: 288 },
  { label: "288p portrait (288x512)", width: 288, height: 512 },
];
/// Size preset + grid step + min dim for a (kind, model): LTX overrides the video
/// defaults with its /64 grid and 256 floor; Wan2.2-A14B uses its /16 ladder led
/// by 832x480; everything else uses the per-kind presets (min = step).
function sizeSpec(kind, model) {
  if (kind === "video" && isLtxModel(model)) return { presets: LTX_PRESETS, step: 64, min: LTX_MIN_DIM };
  if (kind === "video" && isWan22Model(model)) return { presets: WAN22_PRESETS, step: 16, min: 16 };
  if (kind === "video" && isHunyuanModel(model)) return { presets: HUNYUAN_PRESETS, step: 16, min: 16 };
  const step = DIM_STEP[kind];
  return { presets: PRESETS[kind], step, min: step };
}

// Steps input config per kind. Image (Z-Image Turbo) defaults to 8, unbounded
// above. Video (FastWan UniPC, the served sampler) defaults to 4, capped at 8 --
// matching the public FastWan Spaces' 1..=8 slider. DMD (the parity sampler)
// ignores steps and is reachable from the CLI, not exposed here.
const STEPS = {
  image: { value: 8, min: 1, max: "" },
  video: { value: 4, min: 1, max: 8 },
};
// Per-model steps default, mirroring the server's `ModelId::defaults()`. Models
// not listed fall back to the kind default above. The 4-step distilled image
// models (ideogram4, qwen-edit) differ from Z-Image Turbo's 8.
const MODEL_STEPS = {
  "ideogram4-q8": 4,
  "qwen-image-rapid": 4,
  "qwen-image-edit-rapid": 4,
};
// Default clip length (seconds) shown in the duration PLACEHOLDER per video
// model, mirroring the server's default_frames so a blank field advertises the
// REAL default. Blank -> the server uses its default; typing overrides. LTX is
// distilled for ~5s (121 frames), but on the 8GB card the per-resolution frame
// cap (see ltx_max_frames) sizes the real default by dims: at the 1280x704
// widescreen default that is 49 frames (~2s). Pick a lower-res preset for a
// longer clip (e.g. 1024x576 -> ~3s). This 2s is the default-resolution figure.
// Wan2.2-A14B: defaults to 33 frames (~2.1s @16fps) at 832x480, the longest
// length validated e2e. Longer (up to ~5s / 81f) is allowed by typing a duration;
// it runs progressively slower (the 14B self-attention is O(rows^2)).
const VIDEO_DURATION = {
  "fastwan-ti2v-5b": 5,
  "wan2.2-t2v-a14b": 2.1,
  "longlive-2.0-5b": 5,
  "ltx-2.3-distilled": 2,
  "ltx-2.3-distilled-q4": 2,
  "sulphur-2": 2,
  "sulphur-2-q4": 2,
  "hunyuan-video-1.5-t2v": 5,
  "hunyuan-video-1.5-ti2v": 5,
};

const subtle = globalThis.crypto?.subtle;
const SECURE = Boolean(subtle);
if (!SECURE) {
  log("WARNING: insecure context (http) - WebCrypto unavailable, results will NOT be encrypted. Use https or localhost to enable encryption.");
}

// --- base64 <-> bytes ---------------------------------------------------------
const bytesToBase64 = (bytes) => btoa(String.fromCharCode(...bytes));
const base64ToBytes = (b64) => Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));

// Read a picked File as raw base64 (FileReader yields a `data:...;base64,XXXX`
// data URL; strip the prefix to leave just the payload the server decodes).
const fileToBase64 = (file) =>
  new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onerror = () => reject(new Error("could not read the selected image"));
    r.onload = () => resolve(String(r.result).replace(/^data:[^,]*,/, ""));
    r.readAsDataURL(file);
  });

// --- token persistence (this browser only) ------------------------------------
const TOKEN_KEY = "thinfer_token";
$("token").value = localStorage.getItem(TOKEN_KEY) ?? "";
$("token").addEventListener("change", () => {
  const t = $("token").value.trim();
  if (t) localStorage.setItem(TOKEN_KEY, t);
  else localStorage.removeItem(TOKEN_KEY);
});
const authHeaders = (extra = {}) => {
  const t = $("token").value.trim();
  return t ? { ...extra, authorization: `Bearer ${t}` } : extra;
};

// --- result encryption keypair (in-browser, private key never leaves) ---------
let keypairPromise = null;
function ensureKeypair() {
  if (!SECURE) return Promise.resolve(null);
  if (!keypairPromise) {
    keypairPromise = (async () => {
      const kp = await subtle.generateKey(
        {
          name: "RSA-OAEP",
          modulusLength: 2048,
          publicExponent: new Uint8Array([1, 0, 1]),
          hash: "SHA-256",
        },
        true,
        ["encrypt", "decrypt"],
      );
      const spki = new Uint8Array(await subtle.exportKey("spki", kp.publicKey));
      return { privateKey: kp.privateKey, publicKeyB64: bytesToBase64(spki) };
    })();
  }
  return keypairPromise;
}

// Decrypt the server blob: [u16 wrappedLen][RSA-wrapped AES key][12B nonce][AES-GCM body].
async function decryptResult(privateKey, blob) {
  const view = new DataView(blob.buffer, blob.byteOffset, blob.byteLength);
  const wlen = view.getUint16(0, false);
  let off = 2;
  const wrapped = blob.subarray(off, off + wlen);
  off += wlen;
  const nonce = blob.subarray(off, off + 12);
  off += 12;
  const body = blob.subarray(off);
  const aesRaw = await subtle.decrypt({ name: "RSA-OAEP" }, privateKey, wrapped);
  const aesKey = await subtle.importKey("raw", aesRaw, "AES-GCM", false, ["decrypt"]);
  const plain = await subtle.decrypt({ name: "AES-GCM", iv: nonce }, aesKey, body);
  return new Uint8Array(plain);
}

// --- per-type form wiring -----------------------------------------------------
// Populate the size dropdown + dim grid for the active (kind, model). Kept
// separate from applyKind so a model switch (e.g. to LTX, /64 grid) re-grids the
// size controls without re-touching the rest of the form.
function populateSize(kind, model) {
  const { presets, step, min } = sizeSpec(kind, model);
  // Number inputs enforce this grid; presets all satisfy it.
  for (const id of ["width", "height"]) {
    $(id).step = step;
    $(id).min = min;
  }
  // Size dropdown: presets plus a trailing "Custom". The first preset is default.
  $("preset").replaceChildren(
    ...presets.map((p, i) => {
      const o = document.createElement("option");
      o.value = `${p.width}x${p.height}`;
      o.textContent = p.label;
      o.selected = i === 0;
      return o;
    }),
    new Option("Custom", "custom"),
  );
  $("width").value = presets[0].width;
  $("height").value = presets[0].height;
}

function applyKind() {
  const kind = $("kind").value;
  document.body.className = `kind-${kind}`;
  const model = $("model");
  model.replaceChildren(
    ...MODELS[kind].map((m, i) => {
      const o = document.createElement("option");
      o.value = m;
      o.textContent = m;
      o.selected = i === 0;
      return o;
    }),
  );
  // Steps range is kind-specific (see STEPS); applyModel sets the per-model value.
  const st = STEPS[kind];
  $("steps").min = st.min;
  $("steps").max = st.max;
  applyModel();
}
$("kind").addEventListener("change", applyKind);

// Re-grid the size controls + reference-image picker + steps for the selected
// model. LTX switches to its /64 preset grid; edit models reveal the image
// picker.
function applyModel() {
  const model = $("model").value;
  const kind = $("kind").value;
  const edit = kind === "image" && isEditModel(model);
  // The causal Hunyuan TI2V optionally animates a first-frame image: same
  // picker as the image-edit models, but NOT required (no image = text-only).
  const i2v = kind === "video" && isHunyuanI2vModel(model);
  $("input-image-row").hidden = !edit && !i2v;
  $("input-image").required = edit;
  // Audio toggle + upscale toggle are LTX-only (Wan models are silent and have
  // no two-stage upscale path).
  const isLtxVideo = kind === "video" && isLtxModel(model);
  $("audio-row").hidden = !isLtxVideo;
  $("upscale-row").hidden = !isLtxVideo;
  // Text-encoder quant (q8/q4) is an LTX-only knob (shared Gemma encoder).
  $("encoder-row").hidden = !isLtxVideo;
  // Two-stage is the in-distribution path for LTX (the widescreen default OOMs
  // single-stage and low-res single-stage is OOD), so default it on; the user can
  // still uncheck it for a fast low-res single-stage preview.
  $("upscale").checked = isLtxVideo;
  // The tiny/full VAE choice does not apply to LTX (own full VAE) or to
  // Wan2.2-A14B (full Wan2.1 VAE only, no tiny path), so hide it rather than show
  // a misleading "Tiny VAE" default those models ignore. Hunyuan 1.5 HAS both (a
  // TAEHV tiny default + the conv3d full VAE), so it keeps the dropdown.
  const isWan22 = kind === "video" && isWan22Model(model);
  const isHunyuan = kind === "video" && isHunyuanModel(model);
  $("vae-row").hidden = isLtxVideo || isWan22;
  // The Hunyuan-tuned tiny VAE is a Hunyuan-only checkpoint; only offer it there.
  // If a non-Hunyuan model is selected while it was chosen, fall back to tiny.
  $("vae-opt-tiny-ft").hidden = !isHunyuan;
  if (!isHunyuan && $("vae").value === "tiny-ft") $("vae").value = "tiny";
  // LTX runs a fixed distilled schedule (8 steps single-stage, or 8 + 3 = 11 when
  // upscaling), Wan2.2-A14B a fixed 4-step LightX2V distill, and Hunyuan 1.5 a
  // fixed lightx2v 4-step flow-match schedule; all ignore the steps/sampler knobs,
  // so hide the field for them.
  $("steps-row").hidden = isLtxVideo || isWan22 || isHunyuan;
  // Temporal attention window is a video-DiT perf knob: Wan2.2-14B long clips and
  // Hunyuan 1.5 (joint windowed attention -- image queries see ±W latent frames +
  // all text). 0 = full attention. Wan ships an eyeballed W=3 default; Hunyuan
  // defaults to FULL attention (blank field): W=3 broke multi-subject coherence
  // (second-cat spawn at latent frame ~14, 2026-07-01 eyeball), so it is opt-in.
  $("attn-window-row").hidden = !isWan22 && !(isHunyuan && !i2v);
  $("attn-window").value = isWan22 ? "3" : "";
  // Prompt rewrite (Hunyuan only): the model needs detailed captions, so the
  // "Enhance prompt" toggle is on by default and shown only for Hunyuan. The
  // rewriter-model picker (Fast 4B / Full 8B) rides alongside it.
  // On TI2V the rewriter applies to text-only runs (skipped server-side when
  // an image is attached), so the toggle stays visible.
  $("rewrite-row").hidden = !isHunyuan;
  $("rewrite-quality-row").hidden = !isHunyuan;
  populateSize(kind, model);
  // Steps default is per-model (mirrors the server): the 4-step distilled image
  // models start at 4, everything else at the kind default. LTX ignores steps
  // (fixed two-stage schedule); the field is hidden there.
  $("steps").value = MODEL_STEPS[model] ?? STEPS[kind].value;
  // Show the real per-model default in the duration PLACEHOLDER (blank field ->
  // the server uses this default; typing overrides). Clear any value carried
  // over from a previous model so blank truly means "use the shown default".
  const dur = VIDEO_DURATION[model];
  $("duration").value = "";
  $("duration").placeholder = kind === "video" && dur != null ? `${dur}s (default)` : "(seconds)";
}
$("model").addEventListener("change", applyModel);

// Picking a preset fills the dims; typing dims that match no preset flips the
// dropdown to "Custom" (hand-typed dims are always honored).
$("preset").addEventListener("change", () => {
  const v = $("preset").value;
  if (v === "custom") return;
  const [w, h] = v.split("x");
  $("width").value = w;
  $("height").value = h;
});
const syncPresetToDims = () => {
  const v = `${$("width").value}x${$("height").value}`;
  const sel = $("preset");
  sel.value = [...sel.options].some((o) => o.value === v) ? v : "custom";
};
$("width").addEventListener("input", syncPresetToDims);
$("height").addEventListener("input", syncPresetToDims);

applyKind();

// Grey out the coopmat toggle on a server whose GPU can't accelerate it. Best
// effort: on any error (older server without /capabilities, auth not yet set)
// leave the toggle as-is (the server falls back gracefully regardless).
async function refreshCapabilities() {
  try {
    const resp = await fetch("capabilities", { headers: authHeaders() });
    if (!resp.ok) return;
    const caps = await resp.json();
    if (!caps.coopmat) {
      const cb = $("coopmat");
      cb.checked = false;
      cb.disabled = true;
      const row = $("coopmat-row");
      if (row) row.title = "This server's GPU has no tensor-core (coopmat) support.";
    }
  } catch {
    /* leave the toggle interactive; the backend degrades gracefully */
  }
}
refreshCapabilities();

// --- spec building ------------------------------------------------------------
const intOrNull = (v) => {
  const s = String(v).trim();
  if (!s) return null;
  const n = Number(s);
  return Number.isFinite(n) ? Math.trunc(n) : null;
};
const floatOrNull = (v) => {
  const s = String(v).trim();
  if (!s) return null;
  const n = Number(s);
  return Number.isFinite(n) ? n : null;
};

async function buildSpec() {
  const kind = $("kind").value;
  const seed = intOrNull($("seed").value);
  const width = intOrNull($("width").value);
  const height = intOrNull($("height").value);
  const model = $("model").value;
  if (kind === "image") {
    const spec = {
      kind: "image",
      model,
      prompt: $("prompt").value,
      width,
      height,
      steps: intOrNull($("steps").value),
      seed,
    };
    // Edit models (qwen-image-edit) consume a reference image; encode the
    // picked file as base64 (camelCase `inputImage` matches the wire schema).
    if (isEditModel(model)) {
      const file = $("input-image").files[0];
      if (!file) throw new Error(`${model} requires a reference image; choose one first`);
      spec.inputImage = await fileToBase64(file);
    }
    return spec;
  }
  // Multi-shot (LongLive only): each non-empty line of the prompt box is one
  // shot. Every other video model is single-prompt, so send the whole box
  // verbatim -- splitting on newlines would turn one multi-line prompt into
  // bogus extra "shots" and the server would 400.
  const prompts = isMultiShotModel(model)
    ? $("prompt")
        .value.split("\n")
        .map((s) => s.trim())
        .filter(Boolean)
    : [$("prompt").value.trim()];
  // One duration (seconds) covers the whole clip; for multi-shot LongLive the
  // server splits it evenly across shots. Blank = the model default (5s). fps
  // is the model's native rate, so it is not exposed.
  const duration = floatOrNull($("duration").value);
  // LTX honors dims (its /64 grid, from the LTX presets), duration (server snaps
  // to 8k+1 frames then caps to the per-resolution 8GB frame budget), the Audio
  // toggle, and Upscale (two-stage, default-on). It ignores the Wan sampler/vae/
  // steps knobs, so omit them. Blank dims/duration -> the server's LTX defaults
  // (1280x704 two-stage, ~2s with audio).
  if (isLtxModel(model)) {
    return {
      kind: "video",
      model,
      prompts,
      width,
      height,
      durations: duration === null ? null : [duration],
      audio: $("audio").checked,
      upscale: $("upscale").checked,
      encoder: $("encoder").value,
      seed,
    };
  }
  // Wan2.2-A14B runs a fixed 4-step LightX2V distill (steps/sampler ignored) and
  // only the full Wan2.1 VAE, so omit steps and pin vae=full (sending vae=tiny
  // would trigger a pointless tiny-VAE fetch the server then ignores). The server
  // snaps duration to 4n+1 frames then caps to the per-resolution 8GB budget.
  if (isWan22Model(model)) {
    return {
      kind: "video",
      model,
      prompts,
      width,
      height,
      durations: duration === null ? null : [duration],
      vae: "full",
      // Temporal self-attention window radius in latent frames; blank = full
      // attention. Only the long-clip activation-tiled path honors it.
      attnWindow: intOrNull($("attn-window").value),
      seed,
    };
  }
  // Causal Hunyuan TI2V: optional first-frame image (with = I2V, without =
  // text-only) + prompt; fixed 4-step chunked AR schedule (steps/sampler/
  // attn-window do not apply); the tiny/full VAE choice governs the decode;
  // the rewriter runs on text-only requests.
  if (isHunyuanI2vModel(model)) {
    const file = $("input-image").files[0];
    const spec = {
      kind: "video",
      model,
      prompts,
      width,
      height,
      durations: duration === null ? null : [duration],
      vae: $("vae").value,
      rewrite: $("rewrite").checked,
      rewriteQuality: $("rewrite-quality").value,
      seed,
    };
    if (file) spec.inputImage = await fileToBase64(file);
    return spec;
  }
  // HunyuanVideo 1.5 runs a fixed lightx2v 4-step flow-match schedule (steps/
  // sampler ignored) but DOES honor the tiny/full VAE choice (Tiny TAEHV is the
  // fast default; Full is the conv3d parity VAE).
  if (isHunyuanModel(model)) {
    return {
      kind: "video",
      model,
      prompts,
      width,
      height,
      durations: duration === null ? null : [duration],
      vae: $("vae").value,
      // Temporal joint-windowed attention radius in latent frames; blank = full
      // attention (the O(frames²)→O(frames·W) DiT lever). Was MISSING here, so
      // the web field was silently dropped and Hunyuan always ran full attention.
      attnWindow: intOrNull($("attn-window").value),
      // Expand a short prompt into a detailed, structured caption before
      // encoding (the model is trained on long captions; raw short prompts are
      // out-of-distribution). Needs the rewrite endpoint running; serve falls
      // back to the raw prompt if it is unreachable.
      rewrite: $("rewrite").checked,
      // Which rewriter model: fast (4B, default) or full (8B, slower).
      rewriteQuality: $("rewrite-quality").value,
      seed,
    };
  }
  return {
    kind: "video",
    model,
    prompts,
    width,
    height,
    durations: duration === null ? null : [duration],
    // FastWan UniPC step count (1..=8). Server default sampler is UniPC.
    steps: intOrNull($("steps").value),
    vae: $("vae").value,
    seed,
  };
}

// --- progress rendering (mirrors the CLI wording) -----------------------------
function progressText(stage) {
  switch (stage.stage) {
    case "textEncode":
      return "Encoding prompt";
    case "step":
      return `Denoising step ${stage.i}/${stage.n}`;
    case "chunkStep":
      return `Denoising chunk ${stage.chunk}/${stage.numChunks} step ${stage.step}/${stage.numSteps}`;
    case "vaeDecode":
      return "Decoding latents (VAE)";
    case "frameSwapped":
      return `Swapped frame ${stage.done}/${stage.total}`;
    default:
      return JSON.stringify(stage);
  }
}

// --- SSE over fetch -----------------------------------------------------------
// Reads the event body as a stream, splitting on blank-line frame boundaries and
// JSON-parsing each frame's data: payload (one JobEvent). Resolves with the
// `done` result, rejects on error/cancel or a dropped stream.
async function streamEvents(id) {
  const resp = await fetch(`jobs/${id}/events`, { headers: authHeaders() });
  if (!resp.ok) throw new Error(await errorText(resp));
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let sep;
    while ((sep = firstSep(buf)) !== -1) {
      const frame = buf.slice(0, sep.at);
      buf = buf.slice(sep.at + sep.len);
      const ev = parseFrame(frame);
      if (!ev) continue;
      const result = handleEvent(ev);
      if (result !== undefined) return result;
    }
  }
  throw new Error("event stream ended before the job finished");
}

function firstSep(buf) {
  const lf = buf.indexOf("\n\n");
  const crlf = buf.indexOf("\r\n\r\n");
  if (crlf !== -1 && (lf === -1 || crlf < lf)) return { at: crlf, len: 4 };
  if (lf !== -1) return { at: lf, len: 2 };
  return -1;
}

function parseFrame(frame) {
  const data = frame
    .split(/\r?\n/)
    .filter((l) => l.startsWith("data:"))
    .map((l) => l.slice(5).replace(/^ /, ""))
    .join("\n");
  if (!data) return null;
  return JSON.parse(data);
}

// Returns the `done` result to stop streaming, throws on terminal failure, or
// returns undefined to keep going.
function handleEvent(ev) {
  switch (ev.type) {
    case "queued":
      setStatus(`Queued at position ${ev.position}`);
      log(`Queued at position ${ev.position}`);
      return undefined;
    case "started":
      setStatus("Started");
      return undefined;
    case "progress":
      setStatus(progressText(ev.stage));
      log(progressText(ev.stage));
      return undefined;
    case "log":
      log(ev.message);
      return undefined;
    case "done":
      return ev.result;
    case "error":
      throw new Error(ev.message);
    case "cancelled":
      throw new Error("job was cancelled");
    default:
      return undefined;
  }
}

async function errorText(resp) {
  const body = await resp.text().catch(() => "");
  try {
    const j = JSON.parse(body);
    if (j.error) return `server returned ${resp.status}: ${j.error}`;
  } catch {
    /* not JSON */
  }
  return `server returned ${resp.status}: ${body || resp.statusText}`;
}

// --- result rendering (fetch ciphertext -> decrypt -> in-memory blob) ---------
async function showResult(kind, result, privateKey) {
  const resp = await fetch(result.resultUrl.replace(/^\//, ""), { headers: authHeaders() });
  if (!resp.ok) throw new Error(await errorText(resp));
  let bytes = new Uint8Array(await resp.arrayBuffer());
  if (privateKey) bytes = await decryptResult(privateKey, bytes);
  const mime = kind === "image" ? "image/png" : "video/mp4";
  const url = URL.createObjectURL(new Blob([bytes], { type: mime }));
  const img = $("image-out");
  const video = $("video-out");
  if (kind === "image") {
    if (img.src) URL.revokeObjectURL(img.src);
    img.src = url;
    img.hidden = false;
    video.hidden = true;
  } else {
    if (video.src) URL.revokeObjectURL(video.src);
    video.src = url;
    video.hidden = false;
    img.hidden = true;
  }
  // A real download link with an explicit filename: saving the blob: URL
  // directly would drop the extension and the OS would not know it is an mp4.
  const filename = kind === "image" ? "thinfer.png" : "thinfer.mp4";
  const dl = $("download");
  dl.href = url;
  dl.download = filename;
  dl.textContent = `Download ${filename}`;
  dl.hidden = false;
}

// --- submit -------------------------------------------------------------------
$("form").addEventListener("submit", (ev) => {
  ev.preventDefault();
  void generate();
});

// Job id of the in-flight run, for the Cancel button. Null when idle.
let currentJobId = null;
$("cancel").addEventListener("click", async () => {
  if (!currentJobId) return;
  $("cancel").disabled = true;
  setStatus("Cancelling…");
  try {
    // A queued job is dropped immediately; a running one is flagged and stops
    // at the next denoise step boundary. The events stream then ends in
    // `cancelled` and `generate()`'s catch reports it.
    await fetch(`jobs/${currentJobId}/cancel`, {
      method: "POST",
      headers: authHeaders(),
    });
  } catch (e) {
    log(`Cancel request failed: ${e.message ?? e}`);
  }
});

async function generate() {
  const kind = $("kind").value;
  $("fields").disabled = true;
  const t0 = performance.now();
  genStart = t0; // stamp every log line with elapsed-from-submit (see `stamp`)
  setStatus("Submitting…");
  try {
    const keypair = await ensureKeypair();
    const spec = await buildSpec();
    if (keypair) spec.publicKey = keypair.publicKeyB64;
    // Coopmat is opt-OUT: only send the flag when the user unticked it (the
    // server treats absent as "use default"). A no-op on GPUs without support.
    if (!$("coopmat").checked) spec.disableCoopmat = true;
    const resp = await fetch("jobs", {
      method: "POST",
      headers: authHeaders({ "content-type": "application/json" }),
      body: JSON.stringify(spec),
    });
    if (!resp.ok) throw new Error(await errorText(resp));
    const { id } = await resp.json();
    currentJobId = id;
    const cancelBtn = $("cancel");
    cancelBtn.disabled = false;
    cancelBtn.hidden = false;
    log(`Submitted job ${id}${keypair ? " (encrypted)" : " (PLAINTEXT - insecure context)"}`);
    const result = await streamEvents(id);
    await showResult(kind, result, keypair?.privateKey ?? null);
    const secs = ((performance.now() - t0) / 1000).toFixed(1);
    const seed = result.seed != null ? `seed ${result.seed}, ` : "";
    setStatus(`Done: ${seed}${secs}s.${keypair ? "" : " (not encrypted)"}`);
  } catch (e) {
    setStatus(`Failed: ${e.message ?? e}`);
    log(`Failed: ${e.message ?? e}`);
  } finally {
    $("fields").disabled = false;
    genStart = null; // stop stamping; the next run re-arms it
    currentJobId = null;
    $("cancel").hidden = true;
  }
}
