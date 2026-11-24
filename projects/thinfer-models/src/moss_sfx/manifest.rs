//! MOSS-SoundEffect v2.0 files, used as shipped: f32 DiT safetensors, bf16
//! Qwen3-1.7B text encoder (two shards), the DAC VAE `torch.save` checkpoint
//! (weight-normed convs, folded at load) and the Qwen2 BPE tokenizer.

use thinfer_core::manifest::{FileRef, ModelManifest};

pub const REPO: &str = "OpenMOSS-Team/MOSS-SoundEffect-v2.0";

pub mod role {
    pub const DIT: &str = "sfx/dit";
    pub const TEXT_SHARD_1: &str = "sfx/text-encoder-1";
    pub const TEXT_SHARD_2: &str = "sfx/text-encoder-2";
    pub const VAE: &str = "sfx/vae";
    pub const VOCAB: &str = "sfx/vocab";
    pub const MERGES: &str = "sfx/merges";
    pub const TOKENIZER_CONFIG: &str = "sfx/tokenizer-config";
}

pub const RUNTIME_ROLES: &[&str] = &[
    role::DIT,
    role::TEXT_SHARD_1,
    role::TEXT_SHARD_2,
    role::VAE,
    role::VOCAB,
    role::MERGES,
    role::TOKENIZER_CONFIG,
];

pub static MANIFEST: ModelManifest = ModelManifest {
    id: "moss-sfx",
    files: &[
        (
            role::DIT,
            FileRef::new(REPO, "transformer/diffusion_pytorch_model.safetensors"),
        ),
        (
            role::TEXT_SHARD_1,
            FileRef::new(REPO, "text_encoder/model-00001-of-00002.safetensors"),
        ),
        (
            role::TEXT_SHARD_2,
            FileRef::new(REPO, "text_encoder/model-00002-of-00002.safetensors"),
        ),
        (role::VAE, FileRef::new(REPO, "vae/vae_128d_48k.pth")),
        (role::VOCAB, FileRef::new(REPO, "tokenizer/vocab.json")),
        (role::MERGES, FileRef::new(REPO, "tokenizer/merges.txt")),
        (
            role::TOKENIZER_CONFIG,
            FileRef::new(REPO, "tokenizer/tokenizer_config.json"),
        ),
    ],
};
