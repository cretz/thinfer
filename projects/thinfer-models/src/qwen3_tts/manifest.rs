//! Qwen3-TTS file manifests: the official 12 Hz 1.7B repos, used as shipped
//! (bf16 talker + speaker encoder, f32 codec). Every variant carries its own
//! copy of the codec and the Qwen2 BPE files.

use thinfer_core::manifest::{FileRef, ModelManifest};

pub mod role {
    pub const MODEL: &str = "tts/model";
    pub const CODEC: &str = "tts/codec";
    pub const VOCAB: &str = "tts/vocab";
    pub const MERGES: &str = "tts/merges";
    pub const TOKENIZER_CONFIG: &str = "tts/tokenizer-config";
}

pub const RUNTIME_ROLES: &[&str] = &[
    role::MODEL,
    role::CODEC,
    role::VOCAB,
    role::MERGES,
    role::TOKENIZER_CONFIG,
];

macro_rules! tts_manifest {
    ($name:ident, $id:literal, $repo:literal) => {
        pub static $name: ModelManifest = ModelManifest {
            id: $id,
            files: &[
                (role::MODEL, FileRef::new($repo, "model.safetensors")),
                (
                    role::CODEC,
                    FileRef::new($repo, "speech_tokenizer/model.safetensors"),
                ),
                (role::VOCAB, FileRef::new($repo, "vocab.json")),
                (role::MERGES, FileRef::new($repo, "merges.txt")),
                (
                    role::TOKENIZER_CONFIG,
                    FileRef::new($repo, "tokenizer_config.json"),
                ),
            ],
        };
    };
}

tts_manifest!(BASE_MANIFEST, "qwen3-tts", "Qwen/Qwen3-TTS-12Hz-1.7B-Base");
tts_manifest!(
    DESIGN_MANIFEST,
    "qwen3-tts-design",
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
);
