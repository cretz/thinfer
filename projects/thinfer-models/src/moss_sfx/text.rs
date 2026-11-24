//! The prompt encoder: Qwen3-1.7B (`Qwen3TextEncoder`), last hidden states
//! after the final norm (`hidden_states[-1]`). The tokenizer pads on the
//! right and attention is causal, so the valid rows are exactly a forward of
//! the prompt tokens alone; the padded rows are zeroed by the reference
//! (`prompt_emb[v:] = 0`), so they are never computed.

use std::sync::Arc;

use thinfer_core::backend::WgpuBackend;
use thinfer_core::residency::WeightResidency;
use thinfer_core::weight::WeightSource;
use thinfer_core::workspace::Workspace;
use zeroize::Zeroizing;

use super::TEXT_LEN;
use crate::llm::decoder::LlmError;
use crate::llm::dense::{DenseStack, Io, Rows, StackDims, bf16_row, rope_rows};
use crate::llm::kernels::{LlmPipelines, act_bytes};

/// `text_encoder/config.json`.
pub const ENCODER: StackDims = StackDims {
    hidden: 2048,
    layers: 28,
    heads: 16,
    kv_heads: 8,
    head_dim: 128,
    ffn: 6144,
    eps: 1e-6,
    rope_theta: 1_000_000.0,
};

const EMBED: &str = "model.embed_tokens.weight";

pub struct TextEncoder {
    stack: DenseStack,
}

impl TextEncoder {
    pub async fn load<S: WeightSource>(
        backend: &Arc<WgpuBackend>,
        residency: &WeightResidency<S>,
    ) -> Result<Self, LlmError> {
        Ok(Self {
            stack: DenseStack::load(backend, residency, "model", ENCODER).await?,
        })
    }

    /// `[TEXT_LEN, hidden]` states of `ids` (truncated to [`TEXT_LEN`]),
    /// zero past the prompt. No ids (the empty negative prompt): all zeros.
    pub async fn encode<S: WeightSource>(
        &self,
        backend: &Arc<WgpuBackend>,
        residency: &WeightResidency<S>,
        p: &LlmPipelines,
        ws: &Workspace<WgpuBackend>,
        ids: &[u32],
    ) -> Result<Zeroizing<Vec<f32>>, LlmError> {
        let h = ENCODER.hidden as usize;
        let mut out = Zeroizing::new(vec![0f32; TEXT_LEN * h]);
        let ids = &ids[..ids.len().min(TEXT_LEN)];
        if ids.is_empty() {
            return Ok(out);
        }
        let t = ids.len() as u32;
        let mut x = Zeroizing::new(vec![0f32; ids.len() * h]);
        for (i, &id) in ids.iter().enumerate() {
            bf16_row(residency.source(), EMBED, id, &mut x[i * h..(i + 1) * h]).await?;
        }
        let views = self.stack.views(residency, backend).await?;
        let kv = self.stack.kv(ws, t)?;
        let xb = ws.alloc(act_bytes(t, ENCODER.hidden))?;
        let rb = ws.alloc(act_bytes(t, ENCODER.head_dim))?;
        let ob = ws.alloc(act_bytes(t, ENCODER.hidden))?;
        let io = Io { backend };
        let s = ws.batch();
        s.accel_begin()?;
        io.put(&s, &xb, bytemuck::cast_slice(&x[..]))?;
        io.put(&s, &rb, bytemuck::cast_slice(&rope_rows(&ENCODER, 0, t)))?;
        let (xi, ri, oi) = (
            s.import_copy(xb.as_buf_ref()),
            s.import_copy(rb.as_buf_ref()),
            s.import_copy(ob.as_buf_ref()),
        );
        self.stack
            .encode(&s, p, &views, &kv, ri, xi, t, 0, oi, Rows::All)?;
        let rows = Zeroizing::new(io.submit_read(s, &ob, ids.len() * h).await?);
        out[..rows.len()].copy_from_slice(&rows);
        Ok(out)
    }
}
