//! Qwen3-TTS (12 Hz tokenizer): text (+ style instruction, + voice) to 24 kHz
//! speech. A Qwen3 "talker" predicts each 80 ms frame's first codebook from a
//! summed-embedding prompt; a 5-layer code predictor fills the other 15
//! codebooks of the frame; the codec decoder turns the 16 x T codes into
//! samples. Base (voice clone), CustomVoice (preset speakers) and VoiceDesign
//! (voice from a description) share every shape; they differ in weights and
//! in which prompt parts they were trained with.

pub mod codec;
pub mod manifest;
pub mod prompt;
pub mod speaker;
pub mod talker;

use crate::llm::dense::StackDims;

/// Talker shape: exactly Qwen3-1.7B (the MOSS-SoundEffect text encoder too).
pub const TALKER: StackDims = StackDims {
    hidden: 2048,
    layers: 28,
    heads: 16,
    kv_heads: 8,
    head_dim: 128,
    ffn: 6144,
    eps: 1e-6,
    rope_theta: 1_000_000.0,
};

pub const PREDICTOR: StackDims = StackDims {
    hidden: 1024,
    layers: 5,
    heads: 16,
    kv_heads: 8,
    head_dim: 128,
    ffn: 3072,
    eps: 1e-6,
    rope_theta: 1_000_000.0,
};

/// Codebooks per frame (the talker's own + the predictor's 15).
pub const CODE_GROUPS: usize = 16;
/// Entries per codebook.
pub const CODEBOOK: u32 = 2048;
/// Talker vocabulary: the 2048 codes + 1024 control ids.
pub const TALKER_VOCAB: u32 = 3072;
/// Codec frames per second and samples per frame (24 kHz).
pub const FRAME_RATE: f32 = 12.5;
pub const SAMPLE_RATE: u32 = 24_000;
pub const SAMPLES_PER_FRAME: u32 = 1920;

/// Text-side special tokens (Qwen2 vocabulary).
pub mod text_ids {
    pub const IM_START: u32 = 151_644;
    pub const IM_END: u32 = 151_645;
    pub const ASSISTANT: u32 = 77_091;
    pub const TTS_PAD: u32 = 151_671;
    pub const TTS_BOS: u32 = 151_672;
    pub const TTS_EOS: u32 = 151_673;
}

/// Talker (codec-side) control ids.
pub mod codec_ids {
    pub const PAD: u32 = 2148;
    pub const BOS: u32 = 2149;
    pub const EOS: u32 = 2150;
    pub const THINK: u32 = 2154;
    pub const NOTHINK: u32 = 2155;
    pub const THINK_BOS: u32 = 2156;
    pub const THINK_EOS: u32 = 2157;
}

/// Languages the talker has a tag for (`None` = auto, no tag).
pub fn language_id(lang: &str) -> Option<Option<u32>> {
    Some(Some(match lang.to_ascii_lowercase().as_str() {
        "auto" => return Some(None),
        "chinese" => 2055,
        "english" => 2050,
        "german" => 2053,
        "italian" => 2070,
        "portuguese" => 2071,
        "spanish" => 2054,
        "japanese" => 2058,
        "korean" => 2064,
        "french" => 2061,
        "russian" => 2069,
        _ => return None,
    }))
}

/// Every language [`language_id`] accepts, for the model listing.
pub const LANGUAGES: [&str; 11] = [
    "auto",
    "english",
    "chinese",
    "japanese",
    "korean",
    "german",
    "french",
    "russian",
    "portuguese",
    "spanish",
    "italian",
];

/// Sampling of the reference implementation's `generation_config.json`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TtsSampling {
    pub temperature: f32,
    pub top_k: u32,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub sub_temperature: f32,
    pub sub_top_k: u32,
    pub sub_top_p: f32,
}

impl Default for TtsSampling {
    fn default() -> Self {
        Self {
            temperature: 0.9,
            top_k: 50,
            top_p: 1.0,
            repetition_penalty: 1.05,
            sub_temperature: 0.9,
            sub_top_k: 50,
            sub_top_p: 1.0,
        }
    }
}

impl TtsSampling {
    /// Greedy everywhere (parity).
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            sub_temperature: 0.0,
            sub_top_k: 0,
            sub_top_p: 1.0,
        }
    }
}
