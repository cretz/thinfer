//! Qwen-Image-2.1 DiT weight registration over unsloth's GGUF (ComfyUI keys
//! under `model.diffusion_model.`). The `_K_M` file mixes quant kinds PER
//! TENSOR (Q4_K / Q5_K / Q6_K / Q8_0), so every matmul weight carries its own
//! catalog kind and the forward routes its dequant by it. `img_in` and `txt_in`
//! ship bf16 (dense, transposed at upload); `norm_out.linear` ships F32.
//! `img_mlp.gate_up` is `[gate_layer; proj]` fused along the output axis.

use thinfer_core::quant::QuantKind;
use thinfer_core::residency::{WeightHandle, WeightResidency};
use thinfer_core::tensor::StorageEncoding;
use thinfer_core::weight::{WeightId, WeightSource};

use crate::common::loader::{LoadError, register_linear, register_passthrough};

/// Key prefix of every DiT tensor in the GGUF.
pub const PREFIX: &str = "model.diffusion_model.";

/// A matmul weight and how it is stored: `Some(kind)` = GGUF blocks `[N, K]`
/// (dequantized per use), `None` = dense bf16 uploaded as `[K, N]`.
#[derive(Clone, Copy, Debug)]
pub struct Linear {
    pub handle: WeightHandle,
    pub kind: Option<QuantKind>,
}

#[derive(Clone, Debug)]
pub struct BlockHandles {
    pub norm_q: WeightHandle,
    pub norm_k: WeightHandle,
    pub to_q: Linear,
    pub to_k: Linear,
    pub to_v: Linear,
    pub to_out: Linear,
    /// `[gate_layer; proj]`, `[2 * MLP_INNER, DIM]`.
    pub gate_up: Linear,
    pub down: Linear,
}

#[derive(Clone, Debug)]
pub struct TopHandles {
    pub img_in: Linear,
    /// Zero-centred RMSNorm weight (`scale - 1`).
    pub txt_norm: WeightHandle,
    pub txt_in: Linear,
    pub txt_out: Linear,
    pub time_1: Linear,
    pub time_2: Linear,
    /// Shared modulation `[4 * DIM, DIM]`: `[mod1.scale, mod1.gate, mod2.scale,
    /// mod2.gate]`.
    pub modulation: Linear,
    /// Final adaptive norm's scale projection `[DIM, DIM]` (no shift).
    pub norm_out: Linear,
    pub proj_out: Linear,
}

#[derive(Clone, Debug)]
pub struct Qwen21DitHandles {
    pub top: TopHandles,
    pub blocks: Vec<BlockHandles>,
}

fn id(name: &str) -> WeightId {
    WeightId(format!("{PREFIX}{name}"))
}

fn linear<S: WeightSource>(res: &WeightResidency<S>, name: &str) -> Result<Linear, LoadError> {
    let wid = id(name);
    let kind = match res.source().catalog().get(&wid).and_then(|e| e.encoding) {
        Some(StorageEncoding::Quant(k)) => Some(k),
        _ => None,
    };
    Ok(Linear {
        handle: register_linear(res, &wid)?,
        kind,
    })
}

fn param<S: WeightSource>(res: &WeightResidency<S>, name: &str) -> Result<WeightHandle, LoadError> {
    register_passthrough(res, &id(name))
}

/// Register the top-level weights and the first `n_layers` blocks.
pub fn register_handles<S: WeightSource>(
    res: &WeightResidency<S>,
    n_layers: usize,
) -> Result<Qwen21DitHandles, LoadError> {
    let top = TopHandles {
        img_in: linear(res, "img_in.weight")?,
        txt_norm: param(res, "txt_in.text_norm.weight")?,
        txt_in: linear(res, "txt_in.in_layer.weight")?,
        txt_out: linear(res, "txt_in.out_layer.weight")?,
        time_1: linear(res, "time_text_embed.timestep_embedder.linear_1.weight")?,
        time_2: linear(res, "time_text_embed.timestep_embedder.linear_2.weight")?,
        modulation: linear(res, "modulation.1.weight")?,
        norm_out: linear(res, "norm_out.linear.weight")?,
        proj_out: linear(res, "proj_out.weight")?,
    };
    let blocks = (0..n_layers)
        .map(|i| {
            let p = format!("transformer_blocks.{i}");
            Ok(BlockHandles {
                norm_q: param(res, &format!("{p}.attn.norm_q.weight"))?,
                norm_k: param(res, &format!("{p}.attn.norm_k.weight"))?,
                to_q: linear(res, &format!("{p}.attn.to_q.weight"))?,
                to_k: linear(res, &format!("{p}.attn.to_k.weight"))?,
                to_v: linear(res, &format!("{p}.attn.to_v.weight"))?,
                to_out: linear(res, &format!("{p}.attn.to_out.0.weight"))?,
                gate_up: linear(res, &format!("{p}.img_mlp.gate_up.weight"))?,
                down: linear(res, &format!("{p}.img_mlp.out.weight"))?,
            })
        })
        .collect::<Result<Vec<_>, LoadError>>()?;
    Ok(Qwen21DitHandles { top, blocks })
}
