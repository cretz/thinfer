//! LTX-2.3 distilled-1.1 file manifest. A 22B joint audio-video diffusion model
//! (Lightricks LTX-2). The DiT is GGUF (diffusers-native tensor keys, no comfy
//! rename); the connector/embeds, VAEs, and spatial upscaler are safetensors;
//! the text encoder is the Gemma-3-12B QAT GGUF.
//!
//! Parity discipline: every pyref dequantizes the SAME GGUF/safetensors the
//! engine loads. Q8_0 DiT is the quality+perf canary AND baseline (Qwen lesson:
//! a per-request whole-DiT Q4 fold re-quantizes every request -> slower + worse).
//! Q4_K_M / Q2_K are comparison/footprint experiments only.
//!
//! Roles are open-set strings to keep `ModelManifest` model-agnostic; the
//! `role::*` consts are the typed accessors for callers.

use thinfer_core::manifest::{FileRef, ModelManifest};

/// DiT GGUF + connector/embeds + both VAEs all live in the unsloth GGUF repo.
const REPO_GGUF: &str = "unsloth/LTX-2.3-GGUF";
/// Gemma-3-12B-it GGUF (text encoder). The pure Q8_0 (NOT the QAT UD-dynamic
/// file): every matmul site is uniformly Q8_0, which the per-site cfg path needs
/// (the QAT UD-Q4_K_XL varies quant per-layer and uses IQ4_XS, which the engine
/// has no kernel for). Q8_0 is near-lossless vs the bf16 reference LTX validated
/// against, so it is the highest-fidelity conditioning AND unlocks i8 DP4A. The
/// pyref streams it layer-by-layer (one resident layer, host RAM well under 10GB).
const REPO_GEMMA: &str = "unsloth/gemma-3-12b-it-GGUF";
/// Spatial upscaler x2 (latent-space CNN) lives in the upstream fp model repo.
const REPO_UPSCALER: &str = "Lightricks/LTX-2.3";
/// Gemma-3-12B-it tokenizer (ungated unsloth mirror; the GGUF repo ships no
/// `tokenizer.json`). This is the PRODUCT tokenizer and matches upstream
/// `LTXVGemmaTokenizer` (a plain HF `AutoTokenizer`, `text.strip()` +
/// `add_special_tokens=True` -> a leading `<bos>`=2 + real BPE merges). NB it
/// deliberately does NOT match the conformance pyref (`gen_tokens_ref.py`),
/// which builds a tokenizer FROM the GGUF -- that reconstruction is degenerate
/// (no BOS, no merges) and is only self-consistent for the encoder-parity gate;
/// it is the wrong tokenization for real generation. Verified: this json,
/// loaded by the Rust `tokenizers` crate, matches upstream's HF tokenizer.
const REPO_TOKENIZER: &str = "unsloth/gemma-3-12b-it";

/// LTX-2.3 22B `dev` (base, non-distilled) DiT GGUFs, same unsloth repo as the
/// distilled ones. Same `ltxv` arch / tensors / `config` KV as the distilled
/// DiT, so it loads through the identical path; the distill LoRA is folded in at
/// load so the 8-step CFG-free distilled sampler converges (see `ltx::lora`).
/// Any byte-layout-identical LTX-2.3 finetune can fill the dev slots.
const DEV_DIT_Q8_0: &str = "ltx-2.3-22b-dev-Q8_0.gguf";
const DEV_DIT_Q4_K_M: &str = "ltx-2.3-22b-dev-Q4_K_M.gguf";

/// ltx2-rapid: an LTX-2 19B distilled + native-I2V merge (user weight slot, no
/// default; see [`slot`]). Same `ltxv` DiT topology as LTX-2.3 (48 layers) but
/// the 19B conditioning line: FeatureExtractor V1 + in-transformer caption
/// projection, 2-layer/30-head/3840 ungated connector, 6-way block modulation, no
/// gated attention, no prompt-AdaLN. Merge, so no
/// upstream pyref: validated component-wise + against ComfyUI on the same GGUF.
/// 19B embeddings connector (+ FE V1 aggregate embed) and both VAEs (bf16
/// safetensors, comfy-converted). The connector file holds the 2-layer video +
/// audio connectors, their learnable registers, and `text_embedding_projection.
/// aggregate_embed` (the single FE V1 embed). The DiT GGUF holds the caption
/// projections. VAEs are the LTX-2 (non-.3) video + audio autoencoders.
const REPO_RAPID_CONN_VAE: &str = "Kijai/LTXV2_comfy";
/// Prompt-rewriter LM (shared with HunyuanVideo 1.5, keyed by the same rewriter
/// roles): the 19B merge is trained on long, structured captions, so a terse
/// prompt is out-of-distribution and collapses to the model's portrait prior. The
/// optional on-device Qwen3-VL rewriter expands it. Files are model-agnostic
/// (same public repos Hunyuan uses); opt-in, fetched only when rewriting is on.
const REPO_REWRITER_GGUF_4B: &str = "unsloth/Qwen3-VL-4B-Instruct-GGUF";
const REPO_REWRITER_GGUF_8B: &str = "unsloth/Qwen3-VL-8B-Instruct-GGUF";
const REPO_REWRITER_TOKENIZER: &str = "Qwen/Qwen3-VL-8B-Instruct";

pub mod role {
    /// DiT GGUF, Q8_0 (quality+perf baseline + parity canary). 22.8G.
    pub const DIT_GGUF_Q8_0: &str = "dit/gguf-q8_0";
    /// DiT GGUF, Q4_K_M (comparison only). 14.2G.
    pub const DIT_GGUF_Q4_K_M: &str = "dit/gguf-q4_k_m";
    /// DiT GGUF, Q2_K (footprint-floor experiment only). 7.94G.
    pub const DIT_GGUF_Q2_K: &str = "dit/gguf-q2_k";
    /// DiT GGUF, Q5_K_M (ltx2-rapid default; mixed Q5_K/Q6_K per tensor). 14.2G.
    pub const DIT_GGUF_Q5_K_M: &str = "dit/gguf-q5_k_m";
    /// DiT GGUF, F16 (ltx2-rapid quality canary). 37.8G.
    pub const DIT_GGUF_F16: &str = "dit/gguf-f16";
    /// Gemma-3-12B-it text encoder, pure Q8_0 (uniform per-site). ~12.5G.
    pub const ENCODER_GGUF: &str = "encoder/gemma-gguf";
    /// Gemma-3-12B-it text encoder, Q4_K_M (~7.3G; Q4_K + Q6_K mix per tensor).
    /// Opt-in (`THINFER_LTX_ENCODER=q4_k_m`): ~42% smaller disk/residency than the
    /// Q8_0 baseline -> faster cold load (the encoder is stream-bound), at the cost
    /// of lower-precision conditioning. The encoder's per-site dequant dispatch
    /// reads the mixed kinds directly (see `text_encoder::register_handles`). Q8_0
    /// stays the conditioning-quality default.
    pub const ENCODER_GGUF_Q4: &str = "encoder/gemma-gguf-q4";
    /// Embeddings connector + feature-extractor + aggregate embeds (safetensors).
    pub const CONNECTOR: &str = "encoder/connector";
    /// Causal video VAE. NB the decoder is PLAIN (config: `timestep_conditioning
    /// =false`, `causal_decoder=false`, `norm_layer=pixel_norm`) -- not
    /// timestep-conditioned, not noise-injecting (ltx-plan was wrong; on-disk
    /// config corrects it).
    pub const VIDEO_VAE: &str = "vae/video";
    /// Audio VAE (2D mel) + vocoder (BigVGAN v2 + BWE). 102 + 1227 tensors.
    pub const AUDIO_VAE: &str = "vae/audio";
    /// Spatial upscaler x2 (latent-space CNN, distilled path).
    pub const UPSCALER: &str = "upscaler/x2";
    /// Gemma-3 tokenizer.json (HF fast tokenizer) for the text-encoder prompt.
    pub const TOKENIZER: &str = "encoder/tokenizer";
    /// Distill LoRA (safetensors) folded into a `dev` DiT at load so the 8-step
    /// CFG-free distilled sampler converges (the dev GGUF is the BASE
    /// checkpoint). Dev-only role. Default = the official Lightricks rank-384
    /// distill 1.1; a user slot, so a finetune's own distill LoRA can fill it.
    pub const DISTILL_LORA: &str = "dit/distill-lora";
    /// Official Lightricks rank-384 distill LoRA (`ltx-2.3-22b-distilled-lora-384
    /// .safetensors`, 7.6G), the second half of the ComfyUI stack recipe
    /// (`THINFER_LTX_DISTILL=stack`: [`DISTILL_LORA`] at 0.7 + this at 0.5).
    /// From the LTX-2.3 fp repo. Dev-only role.
    pub const DISTILL_LORA_384: &str = "dit/distill-lora-384";
}

/// Roles a silent-video (t2v) generate needs: DiT + encoder + connector + video
/// VAE. Q8_0 DiT baseline. (Audio adds AUDIO_VAE; see `RUNTIME_ROLES_AV_Q8`.)
/// User weight slots (see `thinfer_core::manifest::FileRef::slot`).
pub mod slot {
    pub const DISTILLED_DIT_Q8_0: &str = "ltx-2.3-distilled/dit-q8_0";
    pub const DISTILLED_DIT_Q4_K_M: &str = "ltx-2.3-distilled/dit-q4_k_m";
    pub const DEV_DIT_Q8_0: &str = "ltx-2.3-dev/dit-q8_0";
    pub const DEV_DIT_Q4_K_M: &str = "ltx-2.3-dev/dit-q4_k_m";
    pub const DEV_DISTILL_LORA: &str = "ltx-2.3-dev/distill-lora";
    /// ltx2-rapid (LTX-2 19B merge) DiT GGUFs. No default.
    pub const RAPID_DIT_Q5_K_M: &str = "ltx2-rapid/dit-q5_k_m";
    pub const RAPID_DIT_Q4_K_M: &str = "ltx2-rapid/dit-q4_k_m";
    pub const RAPID_DIT_F16: &str = "ltx2-rapid/dit-f16";
}

pub const RUNTIME_ROLES_T2V_Q8: &[&str] = &[
    role::ENCODER_GGUF,
    role::CONNECTOR,
    role::DIT_GGUF_Q8_0,
    role::VIDEO_VAE,
    role::UPSCALER,
];

/// Joint audio-video generate: t2v roles + the audio decode tail. Includes the
/// tokenizer (the only role the t2v list omits, since the silent-video parity
/// driver feeds dumped ids; the product path tokenizes in Rust).
pub const RUNTIME_ROLES_AV_Q8: &[&str] = &[
    role::ENCODER_GGUF,
    role::TOKENIZER,
    role::CONNECTOR,
    role::DIT_GGUF_Q8_0,
    role::VIDEO_VAE,
    role::AUDIO_VAE,
    role::UPSCALER,
];

/// Joint audio-video with the Q4_K_M DiT (footprint variant: 14.2GB vs 22.8GB
/// on disk + GPU residency). Same chain as [`RUNTIME_ROLES_AV_Q8`] with the
/// Q4_K_M DiT GGUF; the encoder/connector/VAEs/upscaler are unchanged (Q8_0 DiT
/// stays the quality baseline). The DiT runs the per-quant-kind dense dequant
/// path (the file mixes Q4_K + Q6_K per block); no per-step speed change at
/// product scale (the DiT is compute-bound, streaming hidden by prefetch), so
/// this is a VRAM/footprint option, not a perf one.
pub const RUNTIME_ROLES_AV_Q4: &[&str] = &[
    role::ENCODER_GGUF,
    role::TOKENIZER,
    role::CONNECTOR,
    role::DIT_GGUF_Q4_K_M,
    role::VIDEO_VAE,
    role::AUDIO_VAE,
    role::UPSCALER,
];

pub static MANIFEST: ModelManifest = ModelManifest {
    id: "ltx-2.3-distilled",
    files: &[
        (
            role::DIT_GGUF_Q8_0,
            FileRef::new(
                REPO_GGUF,
                "distilled-1.1/ltx-2.3-22b-distilled-1.1-Q8_0.gguf",
            )
            .in_slot(slot::DISTILLED_DIT_Q8_0),
        ),
        (
            role::DIT_GGUF_Q4_K_M,
            FileRef::new(
                REPO_GGUF,
                "distilled-1.1/ltx-2.3-22b-distilled-1.1-Q4_K_M.gguf",
            )
            .in_slot(slot::DISTILLED_DIT_Q4_K_M),
        ),
        (
            role::DIT_GGUF_Q2_K,
            FileRef::new(
                REPO_GGUF,
                "distilled-1.1/ltx-2.3-22b-distilled-1.1-Q2_K.gguf",
            ),
        ),
        (
            role::ENCODER_GGUF,
            FileRef::new(REPO_GEMMA, "gemma-3-12b-it-Q8_0.gguf"),
        ),
        (
            role::ENCODER_GGUF_Q4,
            FileRef::new(REPO_GEMMA, "gemma-3-12b-it-Q4_K_M.gguf"),
        ),
        (
            role::CONNECTOR,
            FileRef::new(
                REPO_GGUF,
                "text_encoders/ltx-2.3-22b-distilled_embeddings_connectors.safetensors",
            ),
        ),
        (
            role::VIDEO_VAE,
            FileRef::new(REPO_GGUF, "vae/ltx-2.3-22b-distilled_video_vae.safetensors"),
        ),
        (
            role::AUDIO_VAE,
            FileRef::new(REPO_GGUF, "vae/ltx-2.3-22b-distilled_audio_vae.safetensors"),
        ),
        (
            role::UPSCALER,
            FileRef::new(REPO_UPSCALER, "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"),
        ),
        (
            role::TOKENIZER,
            FileRef::new(REPO_TOKENIZER, "tokenizer.json"),
        ),
    ],
};

/// LTX-2.3 dev: same component chain as [`MANIFEST`] but the BASE (`dev`) DiT
/// plus the distill LoRA folded over it at load (Q8_0 baseline + Q4_K_M
/// footprint). Every other role reuses the LTX-2.3 file. The runtime-role lists
/// and `role::*` keys are shared, so the same `RUNTIME_ROLES_AV_{Q8,Q4}` resolve
/// against this manifest unchanged.
pub static DEV_MANIFEST: ModelManifest = ModelManifest {
    id: "ltx-2.3-dev",
    files: &[
        (
            role::DIT_GGUF_Q8_0,
            FileRef::new(REPO_GGUF, DEV_DIT_Q8_0).in_slot(slot::DEV_DIT_Q8_0),
        ),
        (
            role::DIT_GGUF_Q4_K_M,
            FileRef::new(REPO_GGUF, DEV_DIT_Q4_K_M).in_slot(slot::DEV_DIT_Q4_K_M),
        ),
        (
            role::DISTILL_LORA,
            FileRef::new(
                REPO_UPSCALER,
                "ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
            )
            .in_slot(slot::DEV_DISTILL_LORA),
        ),
        (
            role::DISTILL_LORA_384,
            FileRef::new(REPO_UPSCALER, "ltx-2.3-22b-distilled-lora-384.safetensors"),
        ),
        (
            role::ENCODER_GGUF,
            FileRef::new(REPO_GEMMA, "gemma-3-12b-it-Q8_0.gguf"),
        ),
        (
            role::ENCODER_GGUF_Q4,
            FileRef::new(REPO_GEMMA, "gemma-3-12b-it-Q4_K_M.gguf"),
        ),
        (
            role::CONNECTOR,
            FileRef::new(
                REPO_GGUF,
                "text_encoders/ltx-2.3-22b-distilled_embeddings_connectors.safetensors",
            ),
        ),
        (
            role::VIDEO_VAE,
            FileRef::new(REPO_GGUF, "vae/ltx-2.3-22b-distilled_video_vae.safetensors"),
        ),
        (
            role::AUDIO_VAE,
            FileRef::new(REPO_GGUF, "vae/ltx-2.3-22b-distilled_audio_vae.safetensors"),
        ),
        (
            role::UPSCALER,
            FileRef::new(REPO_UPSCALER, "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"),
        ),
        (
            role::TOKENIZER,
            FileRef::new(REPO_TOKENIZER, "tokenizer.json"),
        ),
    ],
};

/// ltx2-rapid joint audio-video roles (single-stage; no spatial upscaler). Q5_K_M
/// DiT default. Connector + VAEs come from the LTX-2 19B (Kijai) files; encoder +
/// tokenizer reuse the shared Gemma-3-12B.
pub const RUNTIME_ROLES_RAPID: &[&str] = &[
    role::ENCODER_GGUF,
    role::TOKENIZER,
    role::CONNECTOR,
    role::DIT_GGUF_Q5_K_M,
    role::VIDEO_VAE,
    role::AUDIO_VAE,
];

/// ltx2-rapid manifest. DiT GGUF (Q5_K_M default / Q4_K_M compare / F16 canary)
/// from the merge repo; connector + both VAEs from the LTX-2 19B comfy files;
/// encoder + tokenizer shared with LTX-2.3. No upscaler (single-stage path).
pub static LTX2_RAPID_MANIFEST: ModelManifest = ModelManifest {
    id: "ltx2-rapid",
    files: &[
        (role::DIT_GGUF_Q5_K_M, FileRef::user(slot::RAPID_DIT_Q5_K_M)),
        (role::DIT_GGUF_Q4_K_M, FileRef::user(slot::RAPID_DIT_Q4_K_M)),
        (role::DIT_GGUF_F16, FileRef::user(slot::RAPID_DIT_F16)),
        (
            role::ENCODER_GGUF,
            FileRef::new(REPO_GEMMA, "gemma-3-12b-it-Q8_0.gguf"),
        ),
        (
            role::ENCODER_GGUF_Q4,
            FileRef::new(REPO_GEMMA, "gemma-3-12b-it-Q4_K_M.gguf"),
        ),
        (
            role::CONNECTOR,
            FileRef::new(
                REPO_RAPID_CONN_VAE,
                "text_encoders/ltx-2-19b-embeddings_connector_distill_bf16.safetensors",
            ),
        ),
        (
            role::VIDEO_VAE,
            FileRef::new(REPO_RAPID_CONN_VAE, "VAE/LTX2_video_vae_bf16.safetensors"),
        ),
        (
            role::AUDIO_VAE,
            FileRef::new(REPO_RAPID_CONN_VAE, "VAE/LTX2_audio_vae_bf16.safetensors"),
        ),
        (
            role::TOKENIZER,
            FileRef::new(REPO_TOKENIZER, "tokenizer.json"),
        ),
        // Opt-in prompt-rewriter files (keyed by the shared Hunyuan rewriter
        // roles). Fetched only when rewriting is enabled (see request.rs).
        (
            crate::hunyuan::manifest::role::REWRITER_GGUF_4B_Q5_K_M,
            FileRef::new(REPO_REWRITER_GGUF_4B, "Qwen3-VL-4B-Instruct-Q5_K_M.gguf"),
        ),
        (
            crate::hunyuan::manifest::role::REWRITER_GGUF_8B_Q5_K_M,
            FileRef::new(REPO_REWRITER_GGUF_8B, "Qwen3-VL-8B-Instruct-Q5_K_M.gguf"),
        ),
        (
            crate::hunyuan::manifest::role::REWRITER_TOKENIZER,
            FileRef::new(REPO_REWRITER_TOKENIZER, "tokenizer.json"),
        ),
    ],
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rapid_roles_resolve() {
        for r in RUNTIME_ROLES_RAPID {
            assert!(LTX2_RAPID_MANIFEST.get(r).is_some(), "missing role {r}");
        }
        // DiT is a default-less user slot; connector/VAE from the 19B comfy repo.
        assert!(
            LTX2_RAPID_MANIFEST
                .get(role::DIT_GGUF_Q5_K_M)
                .unwrap()
                .is_unfilled()
        );
        assert_eq!(
            LTX2_RAPID_MANIFEST.get(role::CONNECTOR).unwrap().repo,
            REPO_RAPID_CONN_VAE
        );
        // Single-stage: no upscaler role.
        assert!(!RUNTIME_ROLES_RAPID.contains(&role::UPSCALER));
    }

    #[test]
    fn roles_resolve() {
        for r in RUNTIME_ROLES_T2V_Q8.iter().chain(RUNTIME_ROLES_AV_Q8) {
            assert!(MANIFEST.get(r).is_some(), "missing role {r}");
        }
        assert!(MANIFEST.get("no-such-role").is_none());
    }

    #[test]
    fn dev_roles_resolve() {
        // Dev reuses the AV runtime-role lists; all must resolve, the DiT is the
        // dev checkpoint, and shared roles stay on the LTX-2.3 files.
        for r in RUNTIME_ROLES_AV_Q8.iter().chain(RUNTIME_ROLES_AV_Q4) {
            assert!(DEV_MANIFEST.get(r).is_some(), "missing role {r}");
        }
        assert_eq!(
            DEV_MANIFEST.get(role::DIT_GGUF_Q8_0).unwrap().path,
            DEV_DIT_Q8_0
        );
        assert_eq!(
            DEV_MANIFEST.get(role::VIDEO_VAE).unwrap(),
            MANIFEST.get(role::VIDEO_VAE).unwrap()
        );
        assert!(DEV_MANIFEST.get(role::DISTILL_LORA).unwrap().slot.is_some());
    }

    #[test]
    fn dit_baseline_is_q8() {
        assert!(
            MANIFEST
                .get(role::DIT_GGUF_Q8_0)
                .unwrap()
                .path
                .ends_with("Q8_0.gguf")
        );
    }
}
