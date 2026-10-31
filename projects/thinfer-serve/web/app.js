// Server-backed thinfer UI. Generation runs on the thinfer-serve box this page
// is served from: POST a job, tail its SSE event stream, then download the
// finished artifact. All three calls go through fetch (not EventSource) so the
// optional bearer token rides an Authorization header the whole way.
//
// Privacy: every job gets a fresh AES-256-GCM key generated here. The job spec
// (prompts, images) and any uploaded video are sealed with it BEFORE they leave
// the browser, so the server queues only ciphertext. The key itself is sent
// only when the server's worker is ready to run the job (`keyRequired`); the
// server decrypts right before the model uses the data, seals the result with
// the same key, and forgets the key. The result is fetched as ciphertext (with
// a SHA-256-of-key proof), opened here, and shown from an in-memory blob; the
// raw key is wiped after the job. WebCrypto needs a secure context (https or
// localhost); without one, generation is refused rather than sent in the clear.

const $ = (id) => document.getElementById(id);
const setStatus = (text) => {
  $("status").textContent = text;
};
// Toggle the `hidden` attribute on a control AND its `<label for=id>` together
// (for the bare label|input grid rows that are not wrapped in a row div).
const setLabeledHidden = (id, hidden) => {
  const el = $(id);
  if (el) el.hidden = hidden;
  const lab = document.querySelector(`label[for="${id}"]`);
  if (lab) lab.hidden = hidden;
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
  image: ["zimage-turbo-q4", "zimage-turbo-q8", "zimage-turbo-bf16", "ideogram4-q8", "qwen-image-rapid", "qwen-image-edit-rapid", "krea-2-turbo"],
  // ltx-2.3-distilled-q4 is the fast tier: a pre-distilled Q4_K_M DiT skips the
  // ~60s distill fold the dev ids pay and streams 14GB instead of 23GB per step.
  // ltx-2.3-dev-q4 (Q4 dev + fold) stays CLI-only: it pays Q4 quality AND the
  // fold. Which checkpoints back these ids is the user's weight-sources choice.
  video: ["fastwan-ti2v-5b", "fastwan2.1-t2v-1.3b", "wan2.1-t2v-1.3b", "mobilewan", "wan2.2-t2v-a14b", "anyflow-t2v-14b", "hunyuan-video-1.5-t2v", "hunyuan-video-1.5-ti2v", "longlive-2.0-5b", "ltx-2.3-distilled", "ltx-2.3-distilled-q4", "ltx-2.3-dev", "ltx2-rapid", "minimax-h3", "fasth3-trim", "fasth3-omni", "dreamid-v"],
  // HyperSwap ONNX face-swap: a source face pasted into every frame of a video.
  // Its own JobSpec (FaceSwapSpec); no prompt / size / steps.
  "face-swap": ["hyperswap-1a", "hyperswap-1b", "hyperswap-1c"],
};
// LTX-2.3 is a joint audio-video model with its own grid (/64 dims, 8k+1 frames).
// Its video VAE decode tiles to the VRAM budget, so larger dims are allowed (more
// tiles = slower; the temporal dim is whole, so keep clips short). Dims default
// to upstream's distilled 768x512 (3:2 landscape, what the model was distilled
// for); 256x256 is the explicit "fastest" floor -- below that the model is out of
// distribution and output is incoherent. LTX also exposes an Audio toggle.
// ltx-2.3-dev (dev DiT + distill fold) is byte-identical architecture, same grid
// + audio path, so it shares the whole LTX UI surface.
// ltx2-rapid (LTX-2 19B merge) shares the LTX grid + audio/VAE surface but is
// single-stage (no upscaler) and video-only for now; it matches neither existing
// prefix, so list it explicitly.
const isLtxModel = (model) => model.startsWith("ltx-2.3-") || model === "ltx2-rapid";
// ltx2-rapid is single-stage + video-only, and (post conditioning + VAE fixes) is
// in-distribution at its own low res, so it takes its own preset list + toggle
// defaults instead of the 22B two-stage LTX ones.
const isLtxRapid = (model) => model === "ltx2-rapid";
// MiniMax-H3: a 33B single-stream joint-AV model on its own /32 grid with a
// fixed 24 fps and a 17n+5 frame grid. It shares the LTX audio surface (an Audio
// toggle) but none of the LTX encoder/upscale knobs. Its packed sequence puts
// text, audio and video in ONE self-attention stream, so resolution and clip
// length trade against each other against a hard row cap -- hence its own
// preset ladder rather than a generic one. FastH3-Trim (its 8-step DMD
// distillation) rides the same pipeline and surface with a fixed schedule.
// FastH3-Omni (the reference-image distill) rides it too, plus reference images.
const isFastH3Model = (model) => model === "fasth3-trim" || model === "fasth3-omni";
const isMinimaxModel = (model) => model === "minimax-h3" || isFastH3Model(model);
// Wan2.2-T2V-A14B (MoE 14B): a heavier Wan-family tier with its own surface. It
// runs a fixed 4-step LightX2V distill schedule (the steps/sampler knobs do not
// apply) and only the full Wan2.1 VAE (no tiny-VAE path), so the Steps + Quality
// rows are hidden for it. Defaults to 832x480 (the industry-norm 480p distill
// regime). Longer clips up to the model's ~5s (81f) envelope are allowed via the
// duration field, but on the 8GB card the 14B self-attention is O(rows^2), so
// longer = slower (the default 33f ~2.1s is the longest length validated e2e).
const isWan22Model = (model) => model === "wan2.2-t2v-a14b";
// AnyFlow-Wan2.1-T2V-14B: any-step flow-map distill on the Wan2.1-14B backbone.
// Same 14B/Wan2.1-VAE surface as Wan2.2-A14B (832x480 default, full VAE only,
// /16 grid) EXCEPT the step count is the model's headline feature: the user
// picks it (2 = the fast play, more steps = steady quality gains), so the Steps
// row stays visible with no upper cap.
const isAnyflowModel = (model) => model === "anyflow-t2v-14b";
// The Wan2.1-T2V-1.3B DMD 3-step family (Wan2.1 VAE, /16 grid, 16fps, 832x448):
// FastWan2.1 (its own distill) and wan2.1-t2v-1.3b (stock or finetuned DiT + the
// CausVid distill LoRA, same schedule). Both fold base-Wan2.1-1.3B vault LoRAs.
const isFastwan21Model = (model) => model === "fastwan2.1-t2v-1.3b" || model === "wan2.1-t2v-1.3b";
// HunyuanVideo 1.5 T2V: its own pipeline (Qwen2.5-VL encoder, dual-stream MMDiT,
// causal-conv VAE). Fixed lightx2v 4-step flow-match schedule (the steps/sampler
// knobs do not apply) and its own VAE (no tiny path), so those rows are hidden.
// /16 grid, industry-norm 832x480 / 81f @ 16fps default.
const isHunyuanModel = (model) => model.startsWith("hunyuan-video-1.5");
const isHunyuanI2vModel = (model) => model === "hunyuan-video-1.5-ti2v";
// DreamID-V: a diffusion VIDEO face-swap. Not a text-prompt model -- it consumes
// a target video + a source face image (the context is baked). Runs on the Wan
// backbone but through its own pipeline (no sampler/vae/steps-cap knobs); its own
// image-CFG guidance-scale field applies. 832x480 default.
const isDreamidModel = (model) => model === "dreamid-v";
// Multi-shot video: only LongLive treats each prompt line as a separate shot.
// Every other video model is single-prompt (the whole box is one prompt).
const isMultiShotModel = (model) => model === "longlive-2.0-5b";
// Image models that edit a reference image instead of pure text-to-image: they
// REQUIRE an uploaded input image (mirrors the server's QwenImageEdit kind).
const EDIT_MODELS = new Set(["qwen-image-edit-rapid"]);
const isEditModel = (model) => EDIT_MODELS.has(model);

// Z-Image variants whose DiT source is bf16 safetensors: they take the
// "Full precision" toggle (default off = load-time Q8 transcode, fast; on = bf16,
// slow, max precision). The GGUF variants are already quantized and ignore it.
const BF16_MODELS = new Set(["zimage-turbo-bf16"]);
const isBf16Model = (model) => BF16_MODELS.has(model);
// Image models that accept request-time user adapters (the encrypted LoRA
// vault), mirroring the server's `ImageModelId::supports_adapters`. Adapters are
// per-model, so the UI only lists/uses adapters stored for the selected model.
const ADAPTER_MODELS = new Set(["krea-2-turbo", "qwen-image-rapid", "qwen-image-edit-rapid"]);
// Video models that fold vault LoRAs (diffusers-named Wan2.1 DiTs; the base-Wan2.1
// LoRA ecosystem transfers). Mirrors the server's
// `ImageModelId::supports_adapters` + `VideoModelId::supports_adapters`.
const VIDEO_ADAPTER_MODELS = new Set([
  "anyflow-t2v-14b",
  "fastwan2.1-t2v-1.3b",
  "wan2.1-t2v-1.3b",
  "minimax-h3",
  "fasth3-trim",
  "fasth3-omni",
]);
const supportsAdapters = (kind, model) =>
  (kind === "image" && ADAPTER_MODELS.has(model)) || (kind === "video" && VIDEO_ADAPTER_MODELS.has(model));
// Fold any checked vault adapters into `spec` (mutated in place) plus the vault
// password. Shared by the image and adapter-video spec builders; a no-op when the
// model takes no adapters or none are checked.
function attachAdapters(spec, kind, model) {
  if (!supportsAdapters(kind, model)) return;
  const picks = [...document.querySelectorAll(".adapter-pick")].filter((cb) => cb.checked);
  if (!picks.length) return;
  const password = vaultPassword();
  if (!password) throw new Error("enter the adapter password to use adapters");
  spec.lora = picks.map((cb) => {
    const w = document.querySelector(`.adapter-weight[data-id="${cb.dataset.id}"]`);
    const weight = floatOrNull(w?.value);
    return weight === null ? { id: cb.dataset.id } : { id: cb.dataset.id, weight };
  });
  spec.password = password;
}
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
// Image models authored at 1024 (resolution-aware schedules): Ideogram-4, both
// Qwen-Image-Rapid variants, and Krea 2 Turbo default to 1024x1024 (mirrors the
// server `ImageModelId::defaults`). Z-Image stays on the 768-led PRESETS.image.
const IMAGE_1024_MODELS = new Set(["ideogram4-q8", "qwen-image-rapid", "qwen-image-edit-rapid", "krea-2-turbo"]);
const IMAGE_1024_PRESETS = [
  { label: "1024x1024 (default)", width: 1024, height: 1024 },
  { label: "768x768 (faster)", width: 768, height: 768 },
  { label: "512x512 (fastest)", width: 512, height: 512 },
  { label: "768x1024 portrait", width: 768, height: 1024 },
  { label: "1024x768 landscape", width: 1024, height: 768 },
];
// LTX-2.3 has its own grid: dims must be a multiple of 64 and at least 256 (the
// server rejects below 256). The decode tiles to the VRAM budget so larger dims
// are allowed. All /64.
//
// MEASURED 2026-09-06 (same prompt + seed, a distilled LTX-2.3 Q4_K_M DiT). Resolution here
// trades QUALITY against CLIP LENGTH, *not* against time -- every rung costs about
// the same ~400s because `ltx_max_frames` shrinks the clip as dims grow, so the
// token budget is roughly constant on an 8GB card:
//
//   1280x704  max 2.0s  399s  PHOTOREAL, best prompt adherence
//   1024x576  max 3.0s  392s  stylized 3D render
//   768x512   max 5.0s  418s  cartoon, weaker adherence
//   768x448+  shorter still in quality: anatomy degrades, then prompt is ignored
//
// RE-MEASURED 2026-09-06 at a FIXED 49 frames, so resolution is the only variable
// (the run above let the frame cap grow as dims shrank, which confounded
// resolution with clip length). What degrades below the floor is the SCENE, not
// the subject -- a bearded man with an umbrella renders at every rung, while the
// pier/lighthouse/storm behind him thins out, dissolves, and is finally replaced
// by an unrelated setting:
//
//   1280x704 901k px  513s  subject + pier
//   1024x576 590k px  362s  subject + pier
//    896x512 459k px  286s  photoreal, pier lost
//    768x512 393k px  254s  subject + pier + correct overcast sky   <- FLOOR
//    896x448 401k px  260s  CARTOON, scene gone (MORE px, short side 448)
//    640x512 328k px  227s  claymation, subject cropped (512 short side, too few px)
//    640x384 245k px  192s  scene dissolves to a blank void
//    512x320 164k px  153s  scene REPLACED (indoor shop, not a pier)
//
// So the floor is TWO conditions, area AND short side: 896x448 out-pixels the
// floor and still fails, 640x512 has the short side and still fails. 768x512 is
// the smallest /64 point meeting both, and the largest where a full 5s fits.
// NB adherence is not monotonic rung-to-rung at one seed (896x512 scored worse
// than the smaller 768x512), so these labels separate GROUPS, not neighbours.
// Labels state the LENGTH CAP, because that is the real cost of quality.
const LTX_MIN_DIM = 256;
// Adherence floor (mirrors `VideoModelId::LTX_ADHERENCE_FLOOR_PX` /
// `_MIN_SIDE`). BOTH conditions bind: 896x448 has more pixels than 768x512 and
// still goes cartoon (short side 448), while 640x512 has the 512 short side and
// still collapses (only 328k px).
const LTX_ADHERENCE_FLOOR_PX = 768 * 512;
const LTX_ADHERENCE_MIN_SIDE = 512;
const ltxDimsHoldAdherence = (w, h) =>
  w * h >= LTX_ADHERENCE_FLOOR_PX && Math.min(w, h) >= LTX_ADHERENCE_MIN_SIDE;
const LTX_PRESETS = [
  { label: "720p landscape (1280x704, default, best adherence - max ~2s clip)", width: 1280, height: 704 },
  { label: "1024x576 landscape (holds the scene - max ~3s clip)", width: 1024, height: 576 },
  { label: "896x512 landscape (holds the scene - max ~4s clip)", width: 896, height: 512 },
  { label: "768x512 landscape (adherence floor - full 5s clip)", width: 768, height: 512 },
  { label: "720p portrait (704x1280, max ~2s clip)", width: 704, height: 1280 },
  { label: "512x768 portrait (adherence floor, transposed)", width: 512, height: 768 },
  { label: "896x448 landscape (BELOW FLOOR: goes cartoon, scene lost)", width: 896, height: 448 },
  { label: "640x512 landscape (BELOW FLOOR: subject cropped, scene lost)", width: 640, height: 512 },
  { label: "640x384 landscape (BELOW FLOOR: background dissolves)", width: 640, height: 384 },
  { label: "512x320 landscape (BELOW FLOOR: wrong scene entirely)", width: 512, height: 320 },
];
// ltx2-rapid single-stage presets. Default is 768x512 (the v62 workflow's native
// res); 512x320 is the clean fast floor (in-distribution post the conditioning +
// VAE fixes). Higher res is single-stage here (no upscaler yet), so it is slower
// and labeled by cost, not by faithfulness. All /64, min 256.
const RAPID_PRESETS = [
  { label: "768x512 landscape (default)", width: 768, height: 512 },
  { label: "512x320 landscape (fastest, clean)", width: 512, height: 320 },
  { label: "960x544 landscape (slower)", width: 960, height: 544 },
  { label: "1280x704 landscape (slowest, near VRAM cap)", width: 1280, height: 704 },
];
// MiniMax-H3 / FastH3 size ladder (/32 grid: 16x VAE spatial compression x the
// (1,2,2) DiT patch). FastVideo runs FastH3 at 832x480 (480p) and H3's native
// 1344x768; below 480p is off-distribution (faces and fine detail fall apart).
// Every modality shares one packed self-attention sequence, capped per model
// (`max_packed_rows`: FastH3's row-chunked CUDA block 73728 rows, base H3
// 16384), so larger canvases fit shorter clips: a blank duration takes the
// longest default clip that fits (see `h3Seconds`). First = default.
const H3_PRESETS = [
  { label: "480p landscape (832x480, default)", width: 832, height: 480 },
  { label: "480p portrait (480x832)", width: 480, height: 832 },
  { label: "1024x576 landscape", width: 1024, height: 576 },
  { label: "1280x704 landscape", width: 1280, height: 704 },
  { label: "768p landscape (1344x768, native)", width: 1344, height: 768 },
  { label: "768p portrait (768x1344, native)", width: 768, height: 1344 },
  { label: "640x352 landscape (below 480p: weaker faces/detail)", width: 640, height: 352 },
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
// FastWan2.1-1.3B size ladder (/16 Wan2.1 grid) led by its 832x448 training res.
const FASTWAN21_PRESETS = [
  { label: "448p landscape (832x448, default)", width: 832, height: 448 },
  { label: "448p portrait (448x832)", width: 448, height: 832 },
  { label: "480p landscape (848x480)", width: 848, height: 480 },
  { label: "small landscape (640x352, faster)", width: 640, height: 352 },
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
  if (kind === "video" && isLtxRapid(model)) return { presets: RAPID_PRESETS, step: 64, min: LTX_MIN_DIM };
  if (kind === "video" && isLtxModel(model)) return { presets: LTX_PRESETS, step: 64, min: LTX_MIN_DIM };
  if (kind === "video" && isWan22Model(model)) return { presets: WAN22_PRESETS, step: 16, min: 16 };
  if (kind === "video" && isMinimaxModel(model)) return { presets: H3_PRESETS, step: 32, min: 32 };
  // AnyFlow (Wan2.1 14B) and mobilewan (Wan2.2 5B) both run the /16 Wan grid led
  // by the industry-norm 832x480; the generic 960x544 /32 video ladder is wrong
  // for them (wrong default + a /32 step that rejects legal /16 dims).
  if (kind === "video" && (isAnyflowModel(model) || model === "mobilewan"))
    return { presets: WAN22_PRESETS, step: 16, min: 16 };
  if (kind === "video" && isFastwan21Model(model)) return { presets: FASTWAN21_PRESETS, step: 16, min: 16 };
  if (kind === "image" && IMAGE_1024_MODELS.has(model)) return { presets: IMAGE_1024_PRESETS, step: 16, min: 16 };
  if (kind === "video" && isHunyuanModel(model)) return { presets: HUNYUAN_PRESETS, step: 16, min: 16 };
  // DreamID-V: the /16 Wan2.1-VAE grid led by its 832x480 default (the target
  // clip is downsampled toward this area).
  if (kind === "video" && isDreamidModel(model)) return { presets: HUNYUAN_PRESETS, step: 16, min: 16 };
  // Face-swap has no size (output matches the input video); fall back to the
  // video presets so populateSize has data, but the row is hidden in applyModel.
  const step = DIM_STEP[kind] ?? DIM_STEP.video;
  return { presets: PRESETS[kind] ?? PRESETS.video, step, min: step };
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
  "krea-2-turbo": 8,
  "dreamid-v": 16,
  // AnyFlow any-step flow map: 2 holds 4-step quality on our sweep and each step
  // re-streams the 15GB DiT, so 2 is the quality-neutral fast default.
  "anyflow-t2v-14b": 2,
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
  "mobilewan": 5,
  "wan2.2-t2v-a14b": 2.1,
  // AnyFlow is NOT special-cased in the server's default_frames, so a blank field
  // yields snap_frames(5s*16fps)=81f ~= 5s at its 832x480 default (not 2.1s).
  "anyflow-t2v-14b": 5,
  // snap_frames(5s*16fps) = 81f at its 832x448 default.
  "fastwan2.1-t2v-1.3b": 5,
  "wan2.1-t2v-1.3b": 5,
  "longlive-2.0-5b": 5,
  "ltx-2.3-distilled": 2,
  "ltx-2.3-distilled-q4": 2,
  "ltx-2.3-dev": 2,
  "ltx-2.3-dev-q4": 2,
  // rapid's blank default is 121f (LTX_DEFAULT_FRAMES, uncapped at its 768x512
  // default res) ~= 5s @ 24fps.
  "ltx2-rapid": 5,
  "hunyuan-video-1.5-t2v": 5,
  "hunyuan-video-1.5-ti2v": 5,
  // Default frames on the 17n+5 grid at a fixed 24 fps (56f base, 124f FastH3),
  // capped by the canvas: see `h3Seconds`.
  "minimax-h3": 2.3,
  "fasth3-trim": 5.2,
  "fasth3-omni": 5.2,
};

const subtle = globalThis.crypto?.subtle;
const SECURE = Boolean(subtle);
if (!SECURE) {
  log("ERROR: insecure context (http) - WebCrypto unavailable, so jobs cannot be encrypted and will not be sent. Use https or localhost.");
}

// --- base64 <-> bytes ---------------------------------------------------------
// Chunked: spreading a multi-MB array into fromCharCode overflows the stack.
const bytesToBase64 = (bytes) => {
  let bin = "";
  for (let i = 0; i < bytes.length; i += 0x8000) {
    bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  }
  return btoa(bin);
};
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

// Seal a video file under the job key and POST it to /uploads as a raw body
// (no base64, no JSON); returns its opaque id. The server stores only this
// ciphertext and cannot read it until the job's key arrives.
async function uploadVideo(file, jobKey) {
  setStatus("Encrypting + uploading video…");
  const sealed = await seal(jobKey, new Uint8Array(await file.arrayBuffer()));
  const resp = await fetch("uploads", {
    method: "POST",
    headers: authHeaders({ "content-type": "application/octet-stream" }),
    body: sealed,
  });
  if (!resp.ok) throw new Error(await errorText(resp));
  const { id } = await resp.json();
  return id;
}

// --- token persistence (this tab's session only) ------------------------------
const TOKEN_KEY = "thinfer_token";
$("token").value = sessionStorage.getItem(TOKEN_KEY) ?? "";
$("token").addEventListener("change", () => {
  const t = $("token").value.trim();
  if (t) sessionStorage.setItem(TOKEN_KEY, t);
  else sessionStorage.removeItem(TOKEN_KEY);
});
const authHeaders = (extra = {}) => {
  const t = $("token").value.trim();
  return t ? { ...extra, authorization: `Bearer ${t}` } : extra;
};

// --- per-job key (in-browser; sent only when the worker asks) -----------------
// `raw` is kept only to deliver the key and prove ownership of the result; it is
// zero-filled when the job ends. `key` is a non-extractable WebCrypto handle.
async function newJobKey() {
  const raw = crypto.getRandomValues(new Uint8Array(32));
  const key = await subtle.importKey("raw", raw, "AES-GCM", false, ["encrypt", "decrypt"]);
  return { raw, key };
}

// `[12-byte nonce][AES-GCM ciphertext+tag]` (thinfer_app::jobkey layout).
async function seal(jobKey, bytes) {
  const nonce = crypto.getRandomValues(new Uint8Array(12));
  const body = new Uint8Array(await subtle.encrypt({ name: "AES-GCM", iv: nonce }, jobKey.key, bytes));
  const out = new Uint8Array(12 + body.length);
  out.set(nonce, 0);
  out.set(body, 12);
  return out;
}

async function openSealed(jobKey, blob) {
  const plain = await subtle.decrypt({ name: "AES-GCM", iv: blob.subarray(0, 12) }, jobKey.key, blob.subarray(12));
  return new Uint8Array(plain);
}

// Base64 SHA-256 of the raw key: the result-fetch proof (reveals nothing of it).
async function keyProof(jobKey) {
  return bytesToBase64(new Uint8Array(await subtle.digest("SHA-256", jobKey.raw)));
}

async function deliverKey(id, jobKey) {
  const resp = await fetch(`jobs/${id}/key`, {
    method: "POST",
    headers: authHeaders({ "content-type": "application/json" }),
    body: JSON.stringify({ key: bytesToBase64(jobKey.raw) }),
  });
  if (!resp.ok) throw new Error(await errorText(resp));
  log("Job key delivered; the server can now decrypt and run it");
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
  // Face-swap has no steps, so fall back to the video range (the row is hidden).
  const st = STEPS[kind] ?? STEPS.video;
  $("steps").min = st.min;
  $("steps").max = st.max;
  applyModel();
}
$("kind").addEventListener("change", applyKind);
// Reference images accumulate across picks (a file input replaces its
// selection on every pick, and phone pickers often allow only one), in pick
// order. The prompt names them by position, so the list shows which is which.
const MAX_REF_IMAGES = 9;
const refImages = [];
function renderRefImages() {
  $("ref-images-order").replaceChildren(
    ...refImages.map((f, i) => {
      const row = document.createElement("div");
      const remove = document.createElement("button");
      remove.type = "button";
      remove.textContent = "Remove";
      remove.addEventListener("click", () => {
        refImages.splice(i, 1);
        renderRefImages();
      });
      row.append(`<Picture ${i + 1}> = ${f.name} `, remove);
      return row;
    }),
  );
}
$("ref-images").addEventListener("change", () => {
  const input = $("ref-images");
  refImages.push(...[...input.files].slice(0, MAX_REF_IMAGES - refImages.length));
  input.value = ""; // so re-picking the same file fires `change` again
  renderRefImages();
});

// Re-grid the size controls + reference-image picker + steps for the selected
// model. LTX switches to its /64 preset grid; edit models reveal the image
// picker.
function applyModel() {
  const model = $("model").value;
  const kind = $("kind").value;
  const edit = kind === "image" && isEditModel(model);
  // The causal Hunyuan TI2V optionally animates a first-frame image: same
  // picker as the image-edit models, but NOT required (no image = text-only).
  const i2v = kind === "video" && (isHunyuanI2vModel(model) || isLtxRapid(model));
  $("input-image-row").hidden = !edit && !i2v;
  $("input-image").required = edit;
  // Frame-0 strength: ltx2-rapid I2V only (optional image, so always visible for
  // rapid; server ignores it when no image is sent).
  $("strength-row").hidden = !(kind === "video" && isLtxRapid(model));
  // FastH3-Omni reference images (optional; the prompt names them).
  $("ref-images-row").hidden = !(kind === "video" && model === "fasth3-omni");
  // DreamID-V (video kind) and the face-swap kind both consume a source FACE
  // image + a target VIDEO instead of a text prompt; reveal those uploaders and
  // drop the prompt / size / steps / seed rows that do not apply.
  const dreamid = kind === "video" && isDreamidModel(model);
  const faceSwap = kind === "face-swap";
  const usesFaceInputs = dreamid || faceSwap;
  $("source-image-row").hidden = !usesFaceInputs;
  $("source-image").required = usesFaceInputs;
  $("input-video-row").hidden = !usesFaceInputs;
  $("input-video").required = usesFaceInputs;
  // Mask compositing options are HyperSwap-path-only (DreamID-V has its own mask).
  $("occlusion-mask-row").hidden = !faceSwap;
  $("hyperswap-mask-row").hidden = !faceSwap;
  $("enhance-row").hidden = !faceSwap;
  // Full-precision (bf16 vs default Q8) toggle: only the bf16-source Z-Image
  // variant. GGUF variants are already quantized.
  $("full-precision-row").hidden = !(kind === "image" && isBf16Model(model));
  $("detect-stride-row").hidden = !faceSwap;
  $("bitrate-scale-row").hidden = !faceSwap;
  $("trim-start-row").hidden = !faceSwap;
  $("trim-end-row").hidden = !faceSwap;
  setLabeledHidden("prompt", usesFaceInputs);
  $("prompt").required = !usesFaceInputs;
  $("guide-scale-row").hidden = !dreamid; // DreamID-V image-CFG guidance scale
  // Face-swap output tracks the input clip: no size / seed. DreamID-V keeps size
  // (it drives the target resolution) but has no free-form duration.
  setLabeledHidden("preset", faceSwap);
  setLabeledHidden("width", faceSwap);
  setLabeledHidden("height", faceSwap);
  setLabeledHidden("seed", faceSwap);
  setLabeledHidden("duration", faceSwap || dreamid);
  // Audio toggle + upscale toggle are LTX-only (Wan models are silent and have
  // no two-stage upscale path).
  const isLtxVideo = kind === "video" && isLtxModel(model);
  const rapid = isLtxRapid(model);
  // Audio is joint-AV for ALL LTX lines now, incl. rapid: its 19B DiT denoises the
  // audio stream too, decoded by the shared audio VAE + a plain-HiFiGAN vocoder
  // (16kHz). Upscale (two-stage) is 22B-only -- rapid ships no spatial upscaler.
  // MiniMax-H3 is joint-AV too, so it takes the Audio toggle -- but none of the
  // LTX-specific upscale / HQ-audio / encoder-quant knobs apply to it.
  const isMinimax = kind === "video" && isMinimaxModel(model);
  $("audio-row").hidden = !isLtxVideo && !isMinimax;
  $("upscale-row").hidden = !isLtxVideo || rapid;
  // HQ audio (finer denoise) applies to EVERY LTX line: the coarse distilled
  // sigma jumps under-resolve the joint audio residual on all of them. It refines
  // the single-stage schedule for rapid and BOTH stage schedules for the 22B
  // two-stage path. (It was rapid-only here, which made it dead for the 22B line
  // -- the one place speech gibberish is worst.)
  $("audio-hq-row").hidden = !isLtxVideo;
  // Text-encoder quant (q8/q4) is an LTX knob (shared Gemma encoder); rapid uses
  // the same encoder, so keep it for rapid too.
  $("encoder-row").hidden = !isLtxVideo;
  // Two-stage is the in-distribution path for the 22B LTX (the widescreen default
  // OOMs single-stage and low-res single-stage is OOD), so default it on there;
  // the user can still uncheck it for a fast low-res single-stage preview. rapid
  // is force-single-stage in the executor, so keep it off.
  $("upscale").checked = isLtxVideo && !rapid;
  // The tiny/full VAE choice does not apply to LTX (own full VAE) or to
  // Wan2.2-A14B (full Wan2.1 VAE only, no tiny path), so hide it rather than show
  // a misleading "Tiny VAE" default those models ignore. Hunyuan 1.5 HAS both (a
  // TAEHV tiny default + the conv3d full VAE), so it keeps the dropdown.
  const isWan22 = kind === "video" && isWan22Model(model);
  const isAnyflow = kind === "video" && isAnyflowModel(model);
  const isHunyuan = kind === "video" && isHunyuanModel(model);
  // AnyFlow keeps the dropdown: tiny = taew2_1 (Wan2.1 z16), full = the real
  // Wan2.1 VAE (parity path).
  $("vae-row").hidden = isLtxVideo || isWan22 || dreamid || faceSwap;
  // The Hunyuan-tuned tiny VAE is a Hunyuan-only checkpoint; only offer it there.
  // If a non-Hunyuan model is selected while it was chosen, fall back to tiny.
  $("vae-opt-tiny-ft").hidden = !isHunyuan;
  if (!isHunyuan && $("vae").value === "tiny-ft") $("vae").value = "tiny";
  // MiniMax-H3 / FastH3 decode with FastVideo's NVFP4 decoder by default (the
  // server falls back to their fp16 VAE off Blackwell); "full" = that fp16 VAE,
  // "nvfp4-fast" = the 8-layer one. The tiny VAEs do not apply to them.
  $("vae-opt-tiny").hidden = isMinimax;
  $("vae-opt-nvfp4").hidden = !isMinimax;
  $("vae-opt-nvfp4-fast").hidden = !isMinimax;
  const vae = $("vae").value;
  if (isMinimax && (vae === "tiny" || vae === "tiny-ft")) $("vae").value = "nvfp4";
  if (!isMinimax && vae.startsWith("nvfp4")) $("vae").value = "tiny";
  // INT8 attention applies to the FastH3 models' sparse attention only.
  $("attn-row").hidden = !(kind === "video" && isFastH3Model(model));
  // LTX runs a fixed distilled schedule (8 steps single-stage, or 8 + 3 = 11 when
  // upscaling), Wan2.2-A14B a fixed 4-step LightX2V distill, and Hunyuan 1.5 a
  // fixed lightx2v 4-step flow-match schedule; all ignore the steps/sampler knobs,
  // so hide the field for them.
  // MiniMax-H3 runs the lightx2v 4-step turbo distillation, which is a REQUIRED
  // part of its run path (the base checkpoint needs ~50 steps), so its step
  // count is fixed too.
  // FastWan2.1-1.3B runs its fixed DMD 3-step schedule (the server default
  // sampler for it), so steps do not apply either.
  const isFastwan21 = kind === "video" && isFastwan21Model(model);
  $("steps-row").hidden = isLtxVideo || isWan22 || isHunyuan || faceSwap || isMinimax || isFastwan21;
  // AnyFlow is ANY-step; DreamID-V runs a 16-step distill. Both exceed the FastWan
  // 1..=8 cap, so lift the max for them (else the pre-filled default is invalid and
  // the browser blocks Generate).
  if (isAnyflow || dreamid) {
    $("steps").max = "";
  }
  // Temporal attention window is a video-DiT perf knob: Wan2.2-14B long clips,
  // AnyFlow-14B (same Wan backbone), and Hunyuan 1.5 (joint windowed attention --
  // image queries see ±W latent frames + all text). 0 = full attention. Wan ships
  // an eyeballed W=3 default; Hunyuan and AnyFlow default to FULL attention
  // (blank field): W=3 broke multi-subject coherence on Hunyuan (second-cat spawn
  // at latent frame ~14, 2026-07-01 eyeball), so it is opt-in pending eyeballs.
  $("attn-window-row").hidden = !isWan22 && !isAnyflow && !(isHunyuan && !i2v);
  $("attn-window").value = isWan22 ? "3" : "";
  // Prompt rewrite (Hunyuan + ltx2-rapid): both are trained on detailed
  // captions, so the "Enhance prompt" toggle is on by default and shown for
  // them. The rewriter-model picker (Fast 4B / Full 8B) rides alongside it.
  // On Hunyuan TI2V the rewriter applies to text-only runs (skipped server-side
  // when an image is attached), so the toggle stays visible.
  const canRewrite = isHunyuan || rapid;
  $("rewrite-row").hidden = !canRewrite;
  $("rewrite-quality-row").hidden = !canRewrite;
  // Adapter (LoRA) vault: only image models that support adapters. Clear any
  // rendered list on a model switch so a stale other-model list can't linger
  // (adapters are per-model, so a fresh List is required after switching).
  const adapters = supportsAdapters(kind, model);
  $("adapters-section").hidden = !adapters;
  if (adapters) {
    $("adapters-list").replaceChildren();
    $("adapters-status").textContent = "";
  }
  // Coopmat (tensor-core) only accelerates the Wan-family video DiT hot path; it
  // is inert for image, face-swap, LTX (own pipeline), and Hunyuan. Hide the
  // toggle where it does nothing rather than show a dead control. (The
  // /capabilities probe separately disables it on GPUs without support.)
  const usesCoopmat = kind === "video" && !isLtxModel(model) && !isHunyuanModel(model);
  const coopRow = $("coopmat-row");
  if (coopRow) coopRow.hidden = !usesCoopmat;
  populateSize(kind, model);
  // Steps default is per-model (mirrors the server): the 4-step distilled image
  // models start at 4, everything else at the kind default. LTX ignores steps
  // (fixed two-stage schedule); the field is hidden there.
  $("steps").value = MODEL_STEPS[model] ?? STEPS[kind].value;
  // Show the real per-model default in the duration PLACEHOLDER (blank field ->
  // the server uses this default; typing overrides). Clear any value carried
  // over from a previous model so blank truly means "use the shown default".
  $("duration").value = "";
  refreshDurationPlaceholder();
  refreshDimsWarning();
}
$("model").addEventListener("change", applyModel);

// The LTX default clip length is resolution-derived on the server
// (min(121, ltx_max_frames(w,h))): the 8GB activation envelope caps frames at
// high res, but low res (e.g. 512x320) fits the full 121f/~5s. Mirror that here
// so the blank-field placeholder tracks the SELECTED dims, not a static figure
// (a static "2s" was only right at the 1280x704 widescreen default). fps 24,
// VAE temporal factor 8, /32 latent grid, LTX_MAX_LATENT_CELLS 6300.
const ltxDefaultSeconds = (w, h) => {
  const cellsPerFrame = Math.max(1, Math.floor(h / 32)) * Math.max(1, Math.floor(w / 32));
  const maxFLat = Math.max(1, Math.floor(6300 / cellsPerFrame));
  const framesCap = 8 * (maxFLat - 1) + 1;
  const frames = Math.min(121, framesCap);
  return Math.round((frames / 24) * 10) / 10;
};
// MiniMax-H3 mirror of the server's row budget (`max_fitting_frames`): the
// longest 17n+5 clip whose packed rows (a 256-row prompt reserve + video
// (5n+2) x (h/32)(w/32) + 2 audio rows per 40 Hz latent) fit the model's row
// limit. `def` caps at the model default (a blank field), `max` at H3's ~15s
// design length as the server renders it (a 15s request rounds up the 17n+5
// grid to 362 frames, 15.1s). Seconds, or null.
const H3_MAX_FRAMES = 362;
const h3Seconds = (model, w, h) => {
  const fasth3 = isFastH3Model(model);
  const rowLimit = fasth3 ? 73728 : 16384;
  const perFrame = Math.floor(h / 32) * Math.floor(w / 32);
  const fit = (cap) => {
    let best = null;
    for (let n = 0; 17 * n + 5 <= cap; n++) {
      const frames = 17 * n + 5;
      const audio = 2 * Math.round((frames / 24) * 40);
      if (256 + (5 * n + 2) * perFrame + audio > rowLimit) break;
      best = frames;
    }
    return best == null ? null : Math.round((best / 24) * 10) / 10;
  };
  return { def: fit(fasth3 ? 124 : 56), max: fit(H3_MAX_FRAMES) };
};
function refreshDurationPlaceholder() {
  const model = $("model").value;
  if ($("kind").value !== "video") {
    $("duration").placeholder = "(seconds)";
    return;
  }
  const [w, h] = [Number($("width").value), Number($("height").value)];
  if (isMinimaxModel(model)) {
    const { def, max } = h3Seconds(model, w, h);
    $("duration").placeholder = def != null ? `${def}s (default, max ${max}s)` : "too large";
    return;
  }
  const dur = isLtxModel(model) ? ltxDefaultSeconds(w, h) : VIDEO_DURATION[model];
  $("duration").placeholder = dur != null ? `${dur}s (default)` : "(seconds)";
}

// Show the adherence caveat whenever the chosen LTX dims fall under the floor.
// Mirrors the server-side warning in `request.rs::resolve_ltx` -- the server
// still warns on the job itself, but by then the user has already committed to a
// multi-minute run, so say it while the choice is still being made.
function refreshDimsWarning() {
  const el = $("dims-warning");
  const model = $("model").value;
  const w = Number($("width").value);
  const h = Number($("height").value);
  const under =
    $("kind").value === "video" &&
    isLtxModel(model) &&
    !isLtxRapid(model) &&
    w > 0 &&
    h > 0 &&
    !ltxDimsHoldAdherence(w, h);
  el.hidden = !under;
  if (under) {
    el.textContent =
      "Below the 768x512 prompt-adherence floor (needs both 393216+ pixels and " +
      "a short side of 512+). The clip will still look clean, but the SCENE is " +
      "what degrades: backgrounds thin out, then dissolve, then get replaced by " +
      "an unrelated setting, while the subject survives. It also drifts from " +
      "photoreal toward cartoon. Going lower gets you a LONGER clip, not a " +
      "materially faster one - the frame cap grows as the dims shrink, so the " +
      "clip lengthens and eats most of the saving.";
  }
}

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
// LTX's default clip length depends on the selected dims -> refresh the
// duration placeholder when either the preset or a typed dim changes.
$("preset").addEventListener("change", refreshDurationPlaceholder);
$("width").addEventListener("input", refreshDurationPlaceholder);
$("height").addEventListener("input", refreshDurationPlaceholder);
$("preset").addEventListener("change", refreshDimsWarning);
$("width").addEventListener("input", refreshDimsWarning);
$("height").addEventListener("input", refreshDimsWarning);

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

// --- adapter (LoRA) vault -----------------------------------------------------
// All three ops POST JSON to same-origin serve endpoints (through authHeaders,
// like the job calls) carrying the vault password. The server holds no key; the
// password unlocks the model's adapters for this call only.
function renderAdapters(adapters) {
  const box = $("adapters-list");
  box.replaceChildren();
  if (!adapters.length) {
    box.textContent = "No adapters stored for this model yet.";
    return;
  }
  for (const a of adapters) {
    const row = document.createElement("div");
    row.className = "adapter-item";
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.className = "adapter-pick";
    cb.dataset.id = a.id;
    cb.id = `adapter-${a.id}`;
    const label = document.createElement("label");
    label.htmlFor = cb.id;
    label.textContent = `${a.name} (${(a.size / (1024 * 1024)).toFixed(1)} MB)`;
    const weight = document.createElement("input");
    weight.type = "text";
    weight.inputMode = "decimal";
    weight.className = "adapter-weight";
    weight.dataset.id = a.id;
    weight.value = a.extra?.weight ?? "1.0";
    weight.size = 4;
    weight.title = "Blend weight for this adapter";
    const rm = document.createElement("button");
    rm.type = "button";
    rm.textContent = "✕";
    rm.title = "Remove from the vault";
    rm.addEventListener("click", () => void removeAdapter(a.id, a.name));
    row.append(cb, label, weight, rm);
    box.append(row);
  }
}

// The vault password from the adapters section (kept in memory only).
const vaultPassword = () => $("vault-password").value;

async function vaultFetch(path, body) {
  const resp = await fetch(path, {
    method: "POST",
    headers: authHeaders({ "content-type": "application/json" }),
    body: JSON.stringify(body),
  });
  if (!resp.ok) throw new Error(await errorText(resp));
  return resp;
}

async function listAdapters() {
  const password = vaultPassword();
  if (!password) {
    $("adapters-status").textContent = "Enter the adapter password first.";
    return;
  }
  $("adapters-status").textContent = "Loading…";
  try {
    const resp = await vaultFetch("vault/adapters/list", { model: $("model").value, password });
    const { adapters } = await resp.json();
    renderAdapters(adapters);
    $("adapters-status").textContent = `${adapters.length} adapter(s).`;
  } catch (e) {
    $("adapters-status").textContent = `Failed: ${e.message ?? e}`;
  }
}

async function addAdapter() {
  const password = vaultPassword();
  const url = $("adapter-url").value.trim();
  if (!password) {
    $("adapters-status").textContent = "Enter the adapter password first.";
    return;
  }
  if (!url) {
    $("adapters-status").textContent = "Enter a download URL.";
    return;
  }
  const body = { model: $("model").value, url, password };
  const token = $("adapter-token").value.trim();
  if (token) body.token = token;
  const name = $("adapter-name").value.trim();
  if (name) body.name = name;
  const weight = floatOrNull($("adapter-weight").value);
  if (weight !== null) body.weight = weight;

  const btn = $("adapter-add-btn");
  btn.disabled = true;
  $("adapters-status").textContent = "Downloading + encrypting…";
  try {
    const resp = await vaultFetch("vault/adapters/add", body);
    const info = await resp.json();
    $("adapter-url").value = "";
    $("adapter-token").value = "";
    $("adapter-name").value = "";
    $("adapter-weight").value = "";
    $("adapters-status").textContent = `Added "${info.name}".`;
    await listAdapters();
  } catch (e) {
    $("adapters-status").textContent = `Add failed: ${e.message ?? e}`;
  } finally {
    btn.disabled = false;
  }
}

async function removeAdapter(id, name) {
  const password = vaultPassword();
  if (!password) {
    $("adapters-status").textContent = "Enter the adapter password first.";
    return;
  }
  if (!globalThis.confirm(`Remove "${name}" from the vault? This cannot be undone.`)) return;
  try {
    await vaultFetch("vault/adapters/remove", { model: $("model").value, id, password });
    $("adapters-status").textContent = `Removed "${name}".`;
    await listAdapters();
  } catch (e) {
    $("adapters-status").textContent = `Remove failed: ${e.message ?? e}`;
  }
}

// The vault fields sit inside the generate form visually but belong to their
// own forms (`form=` attributes; HTML forbids nesting), so Enter in them submits
// the vault action, never a generation.
$("vault-form").addEventListener("submit", (ev) => {
  ev.preventDefault();
  void listAdapters();
});
$("adapter-add-form").addEventListener("submit", (ev) => {
  ev.preventDefault();
  void addAdapter();
});

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

async function buildSpec(jobKey) {
  const kind = $("kind").value;
  const seed = intOrNull($("seed").value);
  const width = intOrNull($("width").value);
  const height = intOrNull($("height").value);
  const model = $("model").value;
  // HyperSwap face-swap: source FACE image (small, base64 in the job body) +
  // target VIDEO (streamed to /uploads, referenced by id). No prompt/size.
  if (kind === "face-swap") {
    const face = $("source-image").files[0];
    const vid = $("input-video").files[0];
    if (!face) throw new Error("choose a source face image");
    if (!vid) throw new Error("choose a target video (mp4)");
    return {
      kind: "faceSwap",
      model,
      sourceImageB64: await fileToBase64(face),
      inputVideoUpload: await uploadVideo(vid, jobKey),
      occlusionMask: $("occlusion-mask").checked,
      hyperswapMask: $("hyperswap-mask").checked,
      enhance: $("enhance").checked,
      detectStride: intOrNull($("detect-stride").value) ?? 1,
      bitrateScale: floatOrNull($("bitrate-scale").value) ?? 1.15,
      startSecs: floatOrNull($("trim-start").value),
      endSecs: floatOrNull($("trim-end").value),
    };
  }
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
    // bf16 Z-Image variants: opt into full bf16 precision (default off = Q8).
    if (isBf16Model(model) && $("full-precision").checked) spec.fullPrecision = true;
    // Fold any checked vault adapters into the DiT (per-adapter weight).
    attachAdapters(spec, "image", model);
    return spec;
  }
  // DreamID-V diffusion video face-swap: source FACE image + target VIDEO (no
  // prompt; the context is baked). Width/height drive the target resolution; the
  // image-CFG guidance scale tunes identity transfer. The video streams to
  // /uploads (referenced by id); the server handles it RAM-first /
  // encrypted-spill.
  if (isDreamidModel(model)) {
    const face = $("source-image").files[0];
    const vid = $("input-video").files[0];
    if (!face) throw new Error("dreamid-v needs a source face image");
    if (!vid) throw new Error("dreamid-v needs a target video (mp4)");
    return {
      kind: "video",
      model,
      prompts: [],
      width,
      height,
      steps: intOrNull($("steps").value),
      guideScale: floatOrNull($("guide-scale").value),
      sourceImage: await fileToBase64(face),
      inputVideoUpload: await uploadVideo(vid, jobKey),
      seed,
    };
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
  // MiniMax-H3 honors dims (its /32 grid), duration (the server snaps to the
  // 17n+5 frame grid), the Audio toggle and the decoder choice (its own fp16
  // VAE = "full", or the opt-in NVFP4 decoders). Its step count is fixed by the
  // required 4-step turbo distillation, its fps is fixed at 24, and it has no
  // upscale / encoder-quant / sampler knobs.
  if (isMinimaxModel(model)) {
    const spec = {
      kind: "video",
      model,
      prompts,
      width,
      height,
      durations: duration === null ? null : [duration],
      audio: $("audio").checked,
      vae: $("vae").value,
      seed,
    };
    if (isFastH3Model(model)) spec.attn = $("attn").value;
    if (model === "fasth3-omni") {
      if (refImages.length) spec.refImages = await Promise.all(refImages.map(fileToBase64));
    }
    // Vault LoRAs, applied at runtime (stacked low-rank pairs, no refold).
    attachAdapters(spec, "video", model);
    return spec;
  }
  // LTX honors dims (its /64 grid, from the LTX presets), duration (server snaps
  // to 8k+1 frames then caps to the per-resolution 8GB frame budget), the Audio
  // toggle, and Upscale (two-stage, default-on). It ignores the Wan sampler/vae/
  // steps knobs, so omit them. Blank dims/duration -> the server's LTX defaults
  // (1280x704 two-stage, ~2s with audio).
  if (isLtxModel(model)) {
    const spec = {
      kind: "video",
      model,
      prompts,
      width,
      height,
      durations: duration === null ? null : [duration],
      // Audio is available for every LTX line (incl. rapid). Upscale is 22B-only
      // (rapid is single-stage); force it off for rapid regardless of stale state.
      audio: $("audio").checked,
      upscale: isLtxRapid(model) ? false : $("upscale").checked,
      // HQ audio applies to every LTX line: single-stage for rapid, both stage
      // schedules for the 22B two-stage path.
      audioHq: isLtxModel(model) ? $("audio-hq").checked : false,
      encoder: $("encoder").value,
      // Prompt rewrite: rapid only (the 22B path ignores it server-side and its
      // rewriter files are not in the LTX-2.3 fetch set).
      rewrite: isLtxRapid(model) ? $("rewrite").checked : false,
      rewriteQuality: $("rewrite-quality").value,
      seed,
    };
    // rapid native I2V: an optional first-frame image (with = I2V frame-0
    // anchor, without = t2v) + its conditioning strength.
    if (isLtxRapid(model)) {
      const file = $("input-image").files[0];
      if (file) {
        spec.inputImage = await fileToBase64(file);
        spec.strength = floatOrNull($("strength").value) ?? 1.0;
      }
    }
    return spec;
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
  // FastWan2.1-1.3B: fixed DMD 3-step (the server's default sampler for it, so
  // no steps/sampler sent); VAE is the user's choice; folds vault LoRAs.
  if (isFastwan21Model(model)) {
    const spec = {
      kind: "video",
      model,
      prompts,
      width,
      height,
      durations: duration === null ? null : [duration],
      vae: $("vae").value,
      seed,
    };
    attachAdapters(spec, "video", model);
    return spec;
  }
  // AnyFlow: any-step flow-map schedule -- steps is the primary knob. VAE is
  // the user's choice (tiny = taew2_1 fast decode, full = parity Wan2.1 VAE);
  // attn-window is the opt-in perf lever (blank = full attention).
  if (isAnyflowModel(model)) {
    const spec = {
      kind: "video",
      model,
      prompts,
      width,
      height,
      durations: duration === null ? null : [duration],
      vae: $("vae").value,
      steps: intOrNull($("steps").value),
      attnWindow: intOrNull($("attn-window").value),
      seed,
    };
    // AnyFlow folds base-Wan2.1 vault LoRAs (per-adapter weight + password).
    attachAdapters(spec, "video", model);
    return spec;
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
// `done` result, rejects on error/cancel. Phones drop the stream whenever the
// tab backgrounds or the screen locks while the job keeps running, so a dropped
// stream reconnects with `Last-Event-ID` (the server replays what was missed)
// instead of failing; only a run of failed reconnects gives up.
const RECONNECT_TRIES = 30;
async function streamEvents(id, jobKey) {
  const cursor = { lastId: null, fails: 0 };
  for (;;) {
    try {
      return await streamOnce(id, jobKey, cursor);
    } catch (e) {
      if (!(e instanceof StreamDropped) || ++cursor.fails > RECONNECT_TRIES) throw e;
      setStatus("Connection lost, reconnecting…");
      await new Promise((r) => setTimeout(r, Math.min(500 * 2 ** cursor.fails, 8000)));
    }
  }
}

// A network-level drop (fetch/read failed or the body ended early): retryable,
// unlike a server error status or a terminal job event.
class StreamDropped extends Error {}

async function streamOnce(id, jobKey, cursor) {
  const headers = cursor.lastId == null ? authHeaders() : authHeaders({ "last-event-id": cursor.lastId });
  let resp;
  try {
    resp = await fetch(`jobs/${id}/events`, { headers });
  } catch (e) {
    throw new StreamDropped(e.message ?? String(e));
  }
  if (!resp.ok) throw new Error(await errorText(resp));
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    let chunk;
    try {
      chunk = await reader.read();
    } catch (e) {
      throw new StreamDropped(e.message ?? String(e));
    }
    if (chunk.done) break;
    buf += decoder.decode(chunk.value, { stream: true });
    let sep;
    while ((sep = firstSep(buf)) !== -1) {
      const frame = buf.slice(0, sep.at);
      buf = buf.slice(sep.at + sep.len);
      const parsed = parseFrame(frame);
      if (!parsed) continue;
      cursor.lastId = parsed.id ?? cursor.lastId;
      cursor.fails = 0;
      const ev = parsed.event;
      if (ev.type === "keyRequired") {
        await deliverKey(id, jobKey);
        continue;
      }
      const result = handleEvent(ev);
      if (result !== undefined) return result;
    }
  }
  throw new StreamDropped("event stream ended before the job finished");
}

function firstSep(buf) {
  const lf = buf.indexOf("\n\n");
  const crlf = buf.indexOf("\r\n\r\n");
  if (crlf !== -1 && (lf === -1 || crlf < lf)) return { at: crlf, len: 4 };
  if (lf !== -1) return { at: lf, len: 2 };
  return -1;
}

// One SSE frame -> `{ id, event }` (id = the frame's `id:` sequence, the
// reconnect cursor), or null for keep-alive/comment frames.
function parseFrame(frame) {
  const lines = frame.split(/\r?\n/);
  const field = (name) =>
    lines.filter((l) => l.startsWith(`${name}:`)).map((l) => l.slice(name.length + 1).replace(/^ /, ""));
  const data = field("data").join("\n");
  if (!data) return null;
  return { id: field("id").at(-1) ?? null, event: JSON.parse(data) };
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

// --- result rendering (fetch ciphertext -> open -> in-memory blob) ------------
async function showResult(kind, result, jobKey) {
  const resp = await fetch(result.resultUrl.replace(/^\//, ""), {
    headers: authHeaders({ "x-job-key-sha256": await keyProof(jobKey) }),
    cache: "no-store",
  });
  if (!resp.ok) throw new Error(await errorText(resp));
  const bytes = await openSealed(jobKey, new Uint8Array(await resp.arrayBuffer()));
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
    // Diagnostics: an inline-playback failure often surfaces NOTHING (0:00 white
    // frame) rather than a MediaError -- e.g. Safari/WebKit refusing a blob: video
    // URL. Log the container sniff + the load lifecycle so the failure mode is
    // visible in the on-page log (the file may still download + play fine).
    const isMp4 = bytes.length > 12 && String.fromCharCode(...bytes.subarray(4, 8)) === "ftyp";
    log(
      `result: ${(bytes.length / 1e6).toFixed(1)}MB, ${mime}, ftyp=${isMp4}; ua=${navigator.userAgent}`,
    );
    const CODES = { 1: "ABORTED", 2: "NETWORK", 3: "DECODE", 4: "SRC_NOT_SUPPORTED" };
    video.onerror = () => {
      const e = video.error;
      log(`video error: code ${e?.code ?? "?"} (${CODES[e?.code] ?? "?"})${e?.message ? ` - ${e.message}` : ""}`);
    };
    video.onloadedmetadata = () =>
      log(`video metadata: duration=${video.duration.toFixed(2)}s ${video.videoWidth}x${video.videoHeight}`);
    video.onstalled = () => log("video stalled (source not delivering data)");
    video.src = url;
    video.load();
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

// Job id of the in-flight run, for the Cancel button. Null when idle. Also
// remembered in localStorage so a reloaded page (phones discard backgrounded
// tabs) can still cancel it; the result itself is lost with the old page's key.
let currentJobId = null;
// True while `generate()` owns a job: leaving the page then loses the result
// (its key lives only here), so ask first. Browsers show their own wording,
// and phones often skip the prompt; the orphan watcher below covers those.
let ownJobRunning = false;
window.addEventListener("beforeunload", (ev) => {
  if (ownJobRunning) ev.preventDefault();
});
const ACTIVE_JOB_KEY = "thinfer.activeJob";
function rememberJob(id) {
  try {
    if (id) localStorage.setItem(ACTIVE_JOB_KEY, id);
    else localStorage.removeItem(ACTIVE_JOB_KEY);
  } catch {
    /* storage unavailable: cancel still works for this page's own job */
  }
}
function showCancel(show) {
  const btn = $("cancel");
  btn.disabled = false;
  btn.hidden = !show;
}

$("cancel").addEventListener("click", async () => {
  if (!currentJobId) return;
  $("cancel").disabled = true;
  setStatus("Cancelling…");
  try {
    // A queued job is dropped immediately; a running one is flagged and stops
    // at the next block or phase boundary. The events stream then ends in
    // `cancelled` and `generate()`'s catch reports it.
    const resp = await fetch(`jobs/${currentJobId}/cancel`, {
      method: "POST",
      headers: authHeaders(),
    });
    if (!resp.ok) throw new Error(await errorText(resp));
  } catch (e) {
    log(`Cancel request failed: ${e.message ?? e}`);
    setStatus(`Cancel failed: ${e.message ?? e}`);
    $("cancel").disabled = false;
  }
});

// After a reload: offer Cancel for a job the previous page left running, and
// watch it until it ends. Skipped while this page has its own job.
async function resumeOrphanJob() {
  let id = null;
  try {
    id = localStorage.getItem(ACTIVE_JOB_KEY);
  } catch {
    return;
  }
  if (!id) return;
  for (;;) {
    if (currentJobId && currentJobId !== id) return;
    let state;
    try {
      const resp = await fetch(`jobs/${id}`, { headers: authHeaders() });
      state = resp.ok ? (await resp.json()).state : "gone";
    } catch {
      state = "unreachable";
    }
    if (state !== "queued" && state !== "running") {
      if (currentJobId === id) {
        currentJobId = null;
        showCancel(false);
        setStatus(`Earlier job ended (${state}).`);
      }
      rememberJob(null);
      return;
    }
    if (currentJobId !== id) {
      currentJobId = id;
      showCancel(true);
      setStatus("A job from before this page loaded is still running (its result can't be shown here).");
    }
    await new Promise((r) => setTimeout(r, 3000));
  }
}
resumeOrphanJob();

async function generate() {
  const kind = $("kind").value;
  $("fields").disabled = true;
  const t0 = performance.now();
  genStart = t0; // stamp every log line with elapsed-from-submit (see `stamp`)
  setStatus("Submitting…");
  let jobKey = null;
  let streamGaveUp = false;
  try {
    if (!SECURE) throw new Error("insecure context: open this page over https (or localhost) so jobs can be encrypted");
    jobKey = await newJobKey();
    const spec = await buildSpec(jobKey);
    // Coopmat is opt-OUT: only send the flag when the user unticked it (the
    // server treats absent as "use default"). A no-op on GPUs without support.
    if (!$("coopmat").checked) spec.disableCoopmat = true;
    const sealed = await seal(jobKey, new TextEncoder().encode(JSON.stringify(spec)));
    const resp = await fetch("jobs", {
      method: "POST",
      headers: authHeaders({ "content-type": "application/json" }),
      body: JSON.stringify({ sealed: bytesToBase64(sealed), largeInput: kind === "face-swap" }),
    });
    if (!resp.ok) throw new Error(await errorText(resp));
    const { id } = await resp.json();
    currentJobId = id;
    ownJobRunning = true;
    rememberJob(id);
    showCancel(true);
    log(`Submitted job ${id} (sealed)`);
    const result = await streamEvents(id, jobKey);
    await showResult(kind, result, jobKey);
    const secs = ((performance.now() - t0) / 1000).toFixed(1);
    const seed = result.seed != null ? `seed ${result.seed}, ` : "";
    setStatus(`Done: ${seed}${secs}s.`);
  } catch (e) {
    streamGaveUp = e instanceof StreamDropped;
    setStatus(`Failed: ${e.message ?? e}`);
    log(`Failed: ${e.message ?? e}`);
  } finally {
    jobKey?.raw.fill(0); // wipe the raw key; the CryptoKey handle is dropped with it
    $("fields").disabled = false;
    genStart = null; // stop stamping; the next run re-arms it
    ownJobRunning = false;
    // Forget the job only once it reached a terminal state (or never got an
    // id); a stream that gave up leaves it remembered so a reload can cancel.
    if (currentJobId && !streamGaveUp) rememberJob(null);
    currentJobId = null;
    showCancel(false);
    if (streamGaveUp) resumeOrphanJob();
  }
}
