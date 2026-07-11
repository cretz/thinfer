//! Qwen-Image DiT weight registration. Tensor keys are 1:1 with the diffusers
//! `QwenImageTransformer2DModel` module paths (no rename map). Big matmul
//! weights ship Q8_0; biases, norms, and the sensitive embedders
//! (img_in/txt_in/txt_norm/time_embed/norm_out/proj_out) ship F16 (narrowed to
//! bf16 on upload). The number of registered blocks is a parameter so the
//! single-block dit_parity registers just block 0.

use thinfer_core::residency::{TransposePolicy, WeightHandle, WeightResidency};
use thinfer_core::weight::{WeightId, WeightSource};

use crate::common::embedders::LinearBiasHandles;
use crate::common::loader::{LoadError, register_linear, register_passthrough};

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

fn norm<S: WeightSource>(
    residency: &WeightResidency<S>,
    name: &str,
) -> Result<WeightHandle, LoadError> {
    crate::common::loader::register_one(residency, &id(name), TransposePolicy::None, None)
}

#[derive(Clone, Debug)]
pub struct DitBlockHandles {
    pub img_mod: LinearBiasHandles,
    pub txt_mod: LinearBiasHandles,
    pub to_q: LinearBiasHandles,
    pub to_k: LinearBiasHandles,
    pub to_v: LinearBiasHandles,
    pub to_out: LinearBiasHandles,
    pub add_q: LinearBiasHandles,
    pub add_k: LinearBiasHandles,
    pub add_v: LinearBiasHandles,
    pub to_add_out: LinearBiasHandles,
    pub norm_q: WeightHandle,
    pub norm_k: WeightHandle,
    pub norm_added_q: WeightHandle,
    pub norm_added_k: WeightHandle,
    pub img_mlp_0: LinearBiasHandles,
    pub img_mlp_2: LinearBiasHandles,
    pub txt_mlp_0: LinearBiasHandles,
    pub txt_mlp_2: LinearBiasHandles,
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
    let lb = |w: &str, b: &str| lin_bias(residency, &format!("{p}.{w}"), &format!("{p}.{b}"));
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
        img_in: lin_bias(residency, "img_in.weight", "img_in.bias")?,
        txt_in: lin_bias(residency, "txt_in.weight", "txt_in.bias")?,
        txt_norm: norm(residency, "txt_norm.weight")?,
        time_linear_1: lin_bias(
            residency,
            "time_text_embed.timestep_embedder.linear_1.weight",
            "time_text_embed.timestep_embedder.linear_1.bias",
        )?,
        time_linear_2: lin_bias(
            residency,
            "time_text_embed.timestep_embedder.linear_2.weight",
            "time_text_embed.timestep_embedder.linear_2.bias",
        )?,
        norm_out: lin_bias(residency, "norm_out.linear.weight", "norm_out.linear.bias")?,
        proj_out: lin_bias(residency, "proj_out.weight", "proj_out.bias")?,
    };
    let mut blocks = Vec::with_capacity(n_layers);
    for i in 0..n_layers {
        blocks.push(register_block(residency, i)?);
    }
    Ok(DitHandles { top, blocks })
}
