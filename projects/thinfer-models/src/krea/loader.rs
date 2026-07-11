//! Krea 2 Turbo DiT weight registration. Tensor keys are 1:1 with the GGUF
//! (`realrebelai/KREA-2_GGUFs`, sd.cpp-native `krea2` naming: `blocks.N.*`,
//! `txtfusion.*`, `first/tmlp/tproj/txtmlp/last`). Big matmul weights ship Q8_0
//! (canary) / Q4_K_M (default); norms, biases, and the per-block modulation
//! offsets are F32; the sensitive `first`/`last.modulation` embedders are F16
//! (narrowed to bf16 on upload). `n_layers` parameterizes the block count so the
//! single-block DiT parity registers just `blocks.0`.
//!
//! All Krea linears are BIAS-FREE except `{first, tmlp.0, tmlp.2, tproj.1,
//! txtmlp.1, txtmlp.3, last.linear}`. `KreaRMSNorm` bakes `(scale + 1)`
//! (gemma-style); the DiT forward applies the `+1`, so the raw `*.scale`
//! tensors are registered verbatim here.

use thinfer_core::residency::{WeightHandle, WeightResidency};
use thinfer_core::weight::{WeightId, WeightSource};

use crate::common::embedders::LinearBiasHandles;
use crate::common::loader::{LoadError, register_linear, register_passthrough, register_raw_param};
use crate::krea::config;

fn id(s: &str) -> WeightId {
    WeightId(s.to_string())
}

/// Bias-free linear (weight only; Q8_0/Q4_K/F16 via `register_linear`).
fn lin<S: WeightSource>(
    residency: &WeightResidency<S>,
    weight: &str,
) -> Result<WeightHandle, LoadError> {
    register_linear(residency, &id(weight))
}

/// Linear weight + F32 bias.
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

/// A 1-D F32 parameter (RMSNorm scale or modulation offset), no transpose.
fn param<S: WeightSource>(
    residency: &WeightResidency<S>,
    name: &str,
) -> Result<WeightHandle, LoadError> {
    register_passthrough(residency, &id(name))
}

/// Attention weights (bias-free): GQA q/k/v projections, output gate, output
/// projection, and the per-head `(1+w)` Q/K RMSNorm scales (`[head_dim]`).
#[derive(Clone, Debug)]
pub struct KreaAttnHandles {
    pub wq: WeightHandle,
    pub wk: WeightHandle,
    pub wv: WeightHandle,
    pub gate: WeightHandle,
    pub wo: WeightHandle,
    pub qnorm: WeightHandle,
    pub knorm: WeightHandle,
}

/// SwiGLU MLP weights (bias-free): `silu(gate(x)) * up(x)` then `down`.
#[derive(Clone, Debug)]
pub struct KreaMlpHandles {
    pub gate: WeightHandle,
    pub up: WeightHandle,
    pub down: WeightHandle,
}

fn attn<S: WeightSource>(
    residency: &WeightResidency<S>,
    p: &str,
) -> Result<KreaAttnHandles, LoadError> {
    Ok(KreaAttnHandles {
        wq: lin(residency, &format!("{p}.wq.weight"))?,
        wk: lin(residency, &format!("{p}.wk.weight"))?,
        wv: lin(residency, &format!("{p}.wv.weight"))?,
        gate: lin(residency, &format!("{p}.gate.weight"))?,
        wo: lin(residency, &format!("{p}.wo.weight"))?,
        qnorm: param(residency, &format!("{p}.qknorm.qnorm.scale"))?,
        knorm: param(residency, &format!("{p}.qknorm.knorm.scale"))?,
    })
}

fn mlp<S: WeightSource>(
    residency: &WeightResidency<S>,
    p: &str,
) -> Result<KreaMlpHandles, LoadError> {
    Ok(KreaMlpHandles {
        gate: lin(residency, &format!("{p}.gate.weight"))?,
        up: lin(residency, &format!("{p}.up.weight"))?,
        down: lin(residency, &format!("{p}.down.weight"))?,
    })
}

/// One image-stream block (`blocks.N`): shared-adaLN offset + gated GQA attn +
/// SwiGLU, with pre/post RMSNorm.
#[derive(Clone, Debug)]
pub struct KreaBlockHandles {
    pub prenorm: WeightHandle,
    pub postnorm: WeightHandle,
    /// `[6*DIM]` per-block additive modulation offset (added to the shared
    /// timestep modulation before chunking into 6).
    pub mod_lin: WeightHandle,
    pub attn: KreaAttnHandles,
    pub mlp: KreaMlpHandles,
}

fn block<S: WeightSource>(
    residency: &WeightResidency<S>,
    i: usize,
) -> Result<KreaBlockHandles, LoadError> {
    let p = format!("blocks.{i}");
    Ok(KreaBlockHandles {
        prenorm: param(residency, &format!("{p}.prenorm.scale"))?,
        postnorm: param(residency, &format!("{p}.postnorm.scale"))?,
        mod_lin: param(residency, &format!("{p}.mod.lin"))?,
        attn: attn(residency, &format!("{p}.attn"))?,
        mlp: mlp(residency, &format!("{p}.mlp"))?,
    })
}

/// One text-fusion block (`txtfusion.{layerwise,refiner}_blocks.N`): full-MHA
/// attn + SwiGLU with pre/post RMSNorm, at the text width (2560). No modulation.
#[derive(Clone, Debug)]
pub struct KreaTextBlockHandles {
    pub prenorm: WeightHandle,
    pub postnorm: WeightHandle,
    pub attn: KreaAttnHandles,
    pub mlp: KreaMlpHandles,
}

fn text_block<S: WeightSource>(
    residency: &WeightResidency<S>,
    p: &str,
) -> Result<KreaTextBlockHandles, LoadError> {
    Ok(KreaTextBlockHandles {
        prenorm: param(residency, &format!("{p}.prenorm.scale"))?,
        postnorm: param(residency, &format!("{p}.postnorm.scale"))?,
        attn: attn(residency, &format!("{p}.attn"))?,
        mlp: mlp(residency, &format!("{p}.mlp"))?,
    })
}

/// Text-fusion transformer: layerwise-attn stack (over the 12-layer axis) ->
/// `projector` (`Linear(TEXT_LAYERS -> 1)`, stored as a `[TEXT_LAYERS]` vector)
/// -> refiner stack (over tokens).
#[derive(Clone, Debug)]
pub struct KreaTxtFusionHandles {
    pub layerwise: Vec<KreaTextBlockHandles>,
    pub refiner: Vec<KreaTextBlockHandles>,
    /// `[TEXT_LAYERS]` projector weights (bias-free `Linear(12->1)`).
    pub projector: WeightHandle,
}

/// Top-level (non-block) weights.
#[derive(Clone, Debug)]
pub struct KreaTopHandles {
    /// `Linear(PACKED_CH -> DIM)` image patch embed (F16).
    pub first: LinearBiasHandles,
    /// Timestep MLP: `Linear(TIMESTEP_DIM -> DIM)` -> gelu -> `Linear(DIM -> DIM)`.
    pub tmlp_0: LinearBiasHandles,
    pub tmlp_2: LinearBiasHandles,
    /// Time-modulation projection: gelu -> `Linear(DIM -> 6*DIM)`.
    pub tproj: LinearBiasHandles,
    /// Text projection MLP: RMSNorm -> `Linear(TEXT_DIM -> DIM)` -> gelu(tanh)
    /// -> `Linear(DIM -> DIM)`.
    pub txtmlp_norm: WeightHandle,
    pub txtmlp_1: LinearBiasHandles,
    pub txtmlp_3: LinearBiasHandles,
    pub txtfusion: KreaTxtFusionHandles,
    /// Final layer: RMSNorm + 2-way adaLN offset (`[DIM,2]` F16) +
    /// `Linear(DIM -> PACKED_CH)`.
    pub last_norm: WeightHandle,
    pub last_mod: WeightHandle,
    pub last_linear: LinearBiasHandles,
}

#[derive(Clone, Debug)]
pub struct KreaDitHandles {
    pub top: KreaTopHandles,
    pub blocks: Vec<KreaBlockHandles>,
}

fn txtfusion<S: WeightSource>(
    residency: &WeightResidency<S>,
) -> Result<KreaTxtFusionHandles, LoadError> {
    let mut layerwise = Vec::with_capacity(config::N_LAYERWISE_BLOCKS);
    for i in 0..config::N_LAYERWISE_BLOCKS {
        layerwise.push(text_block(
            residency,
            &format!("txtfusion.layerwise_blocks.{i}"),
        )?);
    }
    let mut refiner = Vec::with_capacity(config::N_REFINER_BLOCKS);
    for i in 0..config::N_REFINER_BLOCKS {
        refiner.push(text_block(
            residency,
            &format!("txtfusion.refiner_blocks.{i}"),
        )?);
    }
    Ok(KreaTxtFusionHandles {
        layerwise,
        refiner,
        projector: param(residency, "txtfusion.projector.weight")?,
    })
}

/// Register the top-level embedders + text-fusion stack + the first `n_layers`
/// image blocks. `n_layers` = 28 for a full run, 1 for the single-block parity.
pub fn register_handles<S: WeightSource>(
    residency: &WeightResidency<S>,
    n_layers: usize,
) -> Result<KreaDitHandles, LoadError> {
    let top = KreaTopHandles {
        first: lin_bias(residency, "first.weight", "first.bias")?,
        tmlp_0: lin_bias(residency, "tmlp.0.weight", "tmlp.0.bias")?,
        tmlp_2: lin_bias(residency, "tmlp.2.weight", "tmlp.2.bias")?,
        tproj: lin_bias(residency, "tproj.1.weight", "tproj.1.bias")?,
        txtmlp_norm: param(residency, "txtmlp.0.scale")?,
        txtmlp_1: lin_bias(residency, "txtmlp.1.weight", "txtmlp.1.bias")?,
        txtmlp_3: lin_bias(residency, "txtmlp.3.weight", "txtmlp.3.bias")?,
        txtfusion: txtfusion(residency)?,
        last_norm: param(residency, "last.norm.scale")?,
        // Raw [2, DIM] F16 scale/shift pair, read directly as rows (NOT matmul'd):
        // no transpose. `register_linear` would apply Linear2D and scramble it.
        last_mod: register_raw_param(residency, &id("last.modulation.lin"))?,
        last_linear: lin_bias(residency, "last.linear.weight", "last.linear.bias")?,
    };
    let mut blocks = Vec::with_capacity(n_layers);
    for i in 0..n_layers {
        blocks.push(block(residency, i)?);
    }
    Ok(KreaDitHandles { top, blocks })
}
