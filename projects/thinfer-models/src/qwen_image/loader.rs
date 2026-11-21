//! Qwen-Image DiT weight registration. Tensor keys are 1:1 with the diffusers
//! `QwenImageTransformer2DModel` module paths (no rename map). Block matmul
//! weights are GGUF quant, and a `_K_M` file mixes kinds PER TENSOR (Q4_K /
//! Q5_K / Q6_K / Q8_0), so each block linear records its own on-disk kind and
//! the DiT routes its dequant by it. The top-level embedders
//! (img_in/txt_in/time_embed/norm_out/proj_out) ride a bf16 dense site whatever
//! the file stores (F16/BF16 narrow, quant dequants at upload). The number of
//! registered blocks is a parameter so the single-block dit_parity registers
//! just block 0.

use thinfer_core::quant::QuantKind;
use thinfer_core::residency::{TransposePolicy, WeightHandle, WeightResidency};
use thinfer_core::tensor::StorageEncoding;
use thinfer_core::weight::{WeightId, WeightSource};

use crate::common::embedders::LinearBiasHandles;
use crate::common::loader::{
    LoadError, register_linear, register_linear_dense_opt_ring, register_passthrough,
};

fn id(s: &str) -> WeightId {
    WeightId(s.to_string())
}

/// Linear weight + bias (weight Q8_0/F16 via `register_linear`, bias F16 dense).
fn lin_bias<S: WeightSource>(
    residency: &WeightResidency<S>,
    weight: &str,
    bias: &str,
) -> Result<LinearBiasHandles, LoadError> {
    Ok(LinearBiasHandles {
        weight: register_linear(residency, &id(weight))?,
        bias: register_passthrough(residency, &id(bias))?,
    })
}

/// Dense-site linear (the top-level embedders): bf16 `[K, N]` on the GPU
/// whatever the file encoding.
fn lin_bias_dense<S: WeightSource>(
    residency: &WeightResidency<S>,
    weight: &str,
    bias: &str,
) -> Result<LinearBiasHandles, LoadError> {
    Ok(LinearBiasHandles {
        weight: register_linear_dense_opt_ring(residency, &id(weight), None)?,
        bias: register_passthrough(residency, &id(bias))?,
    })
}

/// A block linear + its weight's on-disk quant kind (`None` = a float weight,
/// which runs the dense bf16 site).
#[derive(Clone, Debug)]
pub struct QLinearHandles {
    pub lin: LinearBiasHandles,
    pub kind: Option<QuantKind>,
}

fn qlin_bias<S: WeightSource>(
    residency: &WeightResidency<S>,
    weight: &str,
    bias: &str,
) -> Result<QLinearHandles, LoadError> {
    let kind = match residency
        .source()
        .catalog()
        .get(&id(weight))
        .and_then(|e| e.encoding)
    {
        Some(StorageEncoding::Quant(k)) => Some(k),
        _ => None,
    };
    Ok(QLinearHandles {
        lin: lin_bias(residency, weight, bias)?,
        kind,
    })
}

fn norm<S: WeightSource>(
    residency: &WeightResidency<S>,
    name: &str,
) -> Result<WeightHandle, LoadError> {
    crate::common::loader::register_one(residency, &id(name), TransposePolicy::None, None)
}

#[derive(Clone, Debug)]
pub struct DitBlockHandles {
    pub img_mod: QLinearHandles,
    pub txt_mod: QLinearHandles,
    pub to_q: QLinearHandles,
    pub to_k: QLinearHandles,
    pub to_v: QLinearHandles,
    pub to_out: QLinearHandles,
    pub add_q: QLinearHandles,
    pub add_k: QLinearHandles,
    pub add_v: QLinearHandles,
    pub to_add_out: QLinearHandles,
    pub norm_q: WeightHandle,
    pub norm_k: WeightHandle,
    pub norm_added_q: WeightHandle,
    pub norm_added_k: WeightHandle,
    pub img_mlp_0: QLinearHandles,
    pub img_mlp_2: QLinearHandles,
    pub txt_mlp_0: QLinearHandles,
    pub txt_mlp_2: QLinearHandles,
}

#[derive(Clone, Debug)]
pub struct DitTopHandles {
    pub img_in: LinearBiasHandles,
    pub txt_in: LinearBiasHandles,
    pub txt_norm: WeightHandle,
    pub time_linear_1: LinearBiasHandles,
    pub time_linear_2: LinearBiasHandles,
    pub norm_out: LinearBiasHandles,
    pub proj_out: LinearBiasHandles,
}

#[derive(Clone, Debug)]
pub struct DitHandles {
    pub top: DitTopHandles,
    pub blocks: Vec<DitBlockHandles>,
}

fn register_block<S: WeightSource>(
    residency: &WeightResidency<S>,
    i: usize,
) -> Result<DitBlockHandles, LoadError> {
    let p = format!("transformer_blocks.{i}");
    let lb = |w: &str, b: &str| qlin_bias(residency, &format!("{p}.{w}"), &format!("{p}.{b}"));
    Ok(DitBlockHandles {
        img_mod: lb("img_mod.1.weight", "img_mod.1.bias")?,
        txt_mod: lb("txt_mod.1.weight", "txt_mod.1.bias")?,
        to_q: lb("attn.to_q.weight", "attn.to_q.bias")?,
        to_k: lb("attn.to_k.weight", "attn.to_k.bias")?,
        to_v: lb("attn.to_v.weight", "attn.to_v.bias")?,
        to_out: lb("attn.to_out.0.weight", "attn.to_out.0.bias")?,
        add_q: lb("attn.add_q_proj.weight", "attn.add_q_proj.bias")?,
        add_k: lb("attn.add_k_proj.weight", "attn.add_k_proj.bias")?,
        add_v: lb("attn.add_v_proj.weight", "attn.add_v_proj.bias")?,
        to_add_out: lb("attn.to_add_out.weight", "attn.to_add_out.bias")?,
        norm_q: norm(residency, &format!("{p}.attn.norm_q.weight"))?,
        norm_k: norm(residency, &format!("{p}.attn.norm_k.weight"))?,
        norm_added_q: norm(residency, &format!("{p}.attn.norm_added_q.weight"))?,
        norm_added_k: norm(residency, &format!("{p}.attn.norm_added_k.weight"))?,
        img_mlp_0: lb("img_mlp.net.0.proj.weight", "img_mlp.net.0.proj.bias")?,
        img_mlp_2: lb("img_mlp.net.2.weight", "img_mlp.net.2.bias")?,
        txt_mlp_0: lb("txt_mlp.net.0.proj.weight", "txt_mlp.net.0.proj.bias")?,
        txt_mlp_2: lb("txt_mlp.net.2.weight", "txt_mlp.net.2.bias")?,
    })
}

/// Register the top-level embedders + the first `n_layers` transformer blocks.
/// `n_layers` = 60 for a full run, 1 for the single-block parity.
pub fn register_handles<S: WeightSource>(
    residency: &WeightResidency<S>,
    n_layers: usize,
) -> Result<DitHandles, LoadError> {
    let top = DitTopHandles {
        img_in: lin_bias_dense(residency, "img_in.weight", "img_in.bias")?,
        txt_in: lin_bias_dense(residency, "txt_in.weight", "txt_in.bias")?,
        txt_norm: norm(residency, "txt_norm.weight")?,
        time_linear_1: lin_bias_dense(
            residency,
            "time_text_embed.timestep_embedder.linear_1.weight",
            "time_text_embed.timestep_embedder.linear_1.bias",
        )?,
        time_linear_2: lin_bias_dense(
            residency,
            "time_text_embed.timestep_embedder.linear_2.weight",
            "time_text_embed.timestep_embedder.linear_2.bias",
        )?,
        norm_out: lin_bias_dense(residency, "norm_out.linear.weight", "norm_out.linear.bias")?,
        proj_out: lin_bias_dense(residency, "proj_out.weight", "proj_out.bias")?,
    };
    let mut blocks = Vec::with_capacity(n_layers);
    for i in 0..n_layers {
        blocks.push(register_block(residency, i)?);
    }
    Ok(DitHandles { top, blocks })
}
