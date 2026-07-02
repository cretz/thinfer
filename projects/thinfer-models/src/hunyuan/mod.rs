//! HunyuanVideo 1.5 video models. First target: 480p T2V (lightx2v 4-step
//! distill, loaded direct from fp16 safetensors). Dual-stream MMDiT (joint
//! img/txt attention, adaLN-Zero modulation) like Qwen-Image; 3D causal-conv VAE
//! + interleaved-pair RoPE3D like Wan; flow-match Euler scheduler.
//!
//! Reuse map: Qwen2.5-VL text encoder = `crate::qwen_image`; RoPE3D / causal
//! VAE primitives / flow-match stepping = `crate::wan`. T2V = the I2V port minus
//! the image-conditioning channels, so the DiT/VAE/encoder/scheduler are shared
//! and I2V (phase 2) adds onto this core. See
//! `projects/thinfer-working-area/hunyuan-plan.md`.

pub mod config;
pub mod dit;
pub mod encoder;
pub mod manifest;
pub mod refiner;
pub mod scheduler;
pub mod siglip;
pub mod vae;
pub mod vae_tiny;
