//! LLM file manifests. The GGUF roles default to the upstream unsloth quants
//! and sit in user slots, so a sources entry can point the same architecture
//! at a fine-tuned or abliterated GGUF without code changes. Config, tokenizer
//! and chat template always come from the official repo (the architecture's
//! ground truth, shared by every variant).

use thinfer_core::manifest::{FileRef, ModelManifest};

pub mod role {
    /// Runtime default quant.
    pub const GGUF: &str = "llm/gguf";
    /// Q8_0 parity canary.
    pub const GGUF_Q8_0: &str = "llm/gguf-q8_0";
    pub const CONFIG: &str = "llm/config";
    pub const TOKENIZER: &str = "llm/tokenizer";
    pub const CHAT_TEMPLATE: &str = "llm/chat-template";
    pub const GENERATION_CONFIG: &str = "llm/generation-config";
}

pub const RUNTIME_ROLES: &[&str] = &[
    role::GGUF,
    role::CONFIG,
    role::TOKENIZER,
    role::CHAT_TEMPLATE,
    role::GENERATION_CONFIG,
];

/// User slots (see `thinfer_core::manifest::FileRef::slot`).
pub mod slot {
    pub const QWEN36_GGUF: &str = "qwen3.6-35b-a3b/gguf";
    pub const QWEN36_GGUF_Q8_0: &str = "qwen3.6-35b-a3b/gguf-q8_0";
    pub const GEMMA4_GGUF: &str = "gemma-4-26b-a4b/gguf";
    pub const GEMMA4_GGUF_Q8_0: &str = "gemma-4-26b-a4b/gguf-q8_0";
}

const QWEN36: &str = "Qwen/Qwen3.6-35B-A3B";
const QWEN36_GGUF: &str = "unsloth/Qwen3.6-35B-A3B-GGUF";

pub static QWEN36_MANIFEST: ModelManifest = ModelManifest {
    id: "qwen3.6-35b-a3b",
    files: &[
        (
            role::GGUF,
            FileRef::new(QWEN36_GGUF, "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf").in_slot(slot::QWEN36_GGUF),
        ),
        (
            role::GGUF_Q8_0,
            FileRef::new(QWEN36_GGUF, "Qwen3.6-35B-A3B-Q8_0.gguf").in_slot(slot::QWEN36_GGUF_Q8_0),
        ),
        (role::CONFIG, FileRef::new(QWEN36, "config.json")),
        (role::TOKENIZER, FileRef::new(QWEN36, "tokenizer.json")),
        (
            role::CHAT_TEMPLATE,
            FileRef::new(QWEN36, "chat_template.jinja"),
        ),
        (
            role::GENERATION_CONFIG,
            FileRef::new(QWEN36, "generation_config.json"),
        ),
    ],
};

const GEMMA4: &str = "google/gemma-4-26B-A4B-it";
const GEMMA4_GGUF: &str = "unsloth/gemma-4-26B-A4B-it-GGUF";

pub static GEMMA4_MANIFEST: ModelManifest = ModelManifest {
    id: "gemma-4-26b-a4b",
    files: &[
        (
            role::GGUF,
            FileRef::new(GEMMA4_GGUF, "gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf")
                .in_slot(slot::GEMMA4_GGUF),
        ),
        (
            role::GGUF_Q8_0,
            FileRef::new(GEMMA4_GGUF, "gemma-4-26B-A4B-it-Q8_0.gguf")
                .in_slot(slot::GEMMA4_GGUF_Q8_0),
        ),
        (role::CONFIG, FileRef::new(GEMMA4, "config.json")),
        (role::TOKENIZER, FileRef::new(GEMMA4, "tokenizer.json")),
        (
            role::CHAT_TEMPLATE,
            FileRef::new(GEMMA4, "chat_template.jinja"),
        ),
        (
            role::GENERATION_CONFIG,
            FileRef::new(GEMMA4, "generation_config.json"),
        ),
    ],
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roles_resolve() {
        for r in RUNTIME_ROLES.iter().chain([&role::GGUF_Q8_0]) {
            for m in [&QWEN36_MANIFEST, &GEMMA4_MANIFEST] {
                assert!(m.get(r).is_some(), "{}: missing role {r}", m.id);
            }
        }
    }
}
