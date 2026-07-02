//! Optional prompt rewriting (HunyuanVideo 1.5), run NATIVELY on-device. The
//! model is trained on long, structured captions, so short raw prompts are
//! out-of-distribution and produce incoherent video. Upstream expands them with
//! Qwen3-VL-8B-Instruct behind the official t2v rewrite system prompt; this
//! module does the same with the cached Qwen3-VL-8B GGUF and the engine's
//! KV-cached greedy generator (no Python, no HTTP sidecar).
//!
//! Phase-scoped like the other weight sets: load the rewriter GGUF into its own
//! `WeightResidency`, greedily decode the expanded caption, then
//! `evict_all_and_free` before the encoder phase so there is no VRAM-peak bump.
//! Rewriting is a quality lift, never load-bearing: any failure (uncached model,
//! GPU error) is returned so the caller keeps the original prompt. Prompt text is
//! never logged.

use std::sync::Arc;

use thinfer_core::backend::WgpuBackend;
use thinfer_core::format::gguf::GgufSource;
use thinfer_core::format::union::RenamedSource;
use thinfer_core::manifest::ModelManifest;
use thinfer_core::policy::ResidencyBudget;
use thinfer_core::residency::WeightResidency;
use thinfer_core::tokenizer::Tokenizer;
use thinfer_core::workspace::Workspace;
use thinfer_models::hunyuan::manifest::role;
use thinfer_models::qwen3_lm::forward::{Qwen3LmGenerator, compile_pipelines};
use thinfer_models::qwen3_lm::generate::{qwen3vl_gguf_renames, register_qwen3_lm};
use thinfer_native::MmapFileOpener;
use thinfer_native::tokenizer::HfTokenizer;

use crate::download::resolve_role;
use crate::model::RewriteQuality;

/// The official HunyuanVideo 1.5 t2v rewrite system prompt (verbatim from
/// `hyvideo/utils/rewrite/t2v_prompt.py`). ~5.8k tokens; prefilled once per
/// rewrite into the KV cache.
pub const T2V_REWRITE_SYSTEM_PROMPT: &str = include_str!("../assets/t2v_rewrite_system_prompt.txt");

/// Qwen3 ChatML end token (`<|im_end|>`), the rewriter EOS.
const EOS_ID: u32 = 151645;

/// Max generated caption tokens. Upstream caps rewrites at ~400; captions run
/// ~150-250 tokens, so 320 leaves headroom while bounding decode time.
const MAX_NEW: usize = 320;

/// Expand `user_prompt` into a detailed, structured caption via the on-device
/// Qwen3-VL rewriter selected by `quality` (`Fast` = the ~2.5GB 4B, `Full` = the
/// ~5.85GB 8B). `vram_budget_bytes` is the run's device VRAM ceiling and is
/// honored as a HARD ceiling: whatever fits under it stays resident and the rest
/// streams (the smaller `Fast` model is the fix for a tight budget, never a
/// floor/override of it). Errors are returned, not panicked, so the caller can
/// fall back to the original prompt.
pub async fn rewrite_prompt(
    backend: &Arc<WgpuBackend>,
    manifest: &ModelManifest,
    quality: RewriteQuality,
    vram_budget_bytes: u64,
    user_prompt: &str,
) -> Result<String, String> {
    let cfg = quality.lm_config();
    // --- resolve the cached GGUF + tokenizer (the caller ensured download) ---
    let gguf_path = resolve_role(manifest, quality.gguf_role())?;
    let tok_path = resolve_role(manifest, role::REWRITER_TOKENIZER)?;
    let tokenizer = HfTokenizer::from_path(&tok_path)
        .await
        .map_err(|e| format!("load rewriter tokenizer: {e:?}"))?;

    // --- Qwen3 ChatML: system = the t2v rewrite prompt, user = the raw prompt,
    //     assistant open (generation prompt). Never log the text. ---
    let chat = format!(
        "<|im_start|>system\n{T2V_REWRITE_SYSTEM_PROMPT}<|im_end|>\n\
         <|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
    );
    let prompt_ids = tokenizer
        .encode(&chat, false)
        .map_err(|e| format!("tokenize rewrite prompt: {e:?}"))?;
    if prompt_ids.is_empty() {
        return Err("rewrite prompt tokenized to nothing".into());
    }
    let max_ctx = prompt_ids.len() + MAX_NEW + 4;

    // --- phase-scoped residency: RESPECT the run's VRAM budget as a hard ceiling
    //     (do NOT floor/override it to force residency -- that would defeat a
    //     budget the deployer set deliberately). Whatever fits under the budget
    //     stays resident; the rest streams. The consequence is honest: an 8B
    //     rewriter (~5.85GB Q5_K_M) cannot stay resident under a 5GB ceiling, so
    //     decode streams and is slow. The fix is to fit the budget (the smaller
    //     rewriter variant), not to cheat the ceiling. ---
    let weight_budget = vram_budget_bytes;

    let opener = MmapFileOpener::new(&gguf_path)
        .await
        .map_err(|e| format!("open rewriter gguf: {e:?}"))?;
    let gguf = GgufSource::open(opener)
        .await
        .map_err(|e| format!("parse rewriter gguf: {e:?}"))?;
    let renamed = RenamedSource::with_passthrough(gguf, qwen3vl_gguf_renames(&cfg));
    let residency = WeightResidency::new(
        renamed,
        ResidencyBudget {
            ram_bytes: 4 << 30,
            vram_bytes: weight_budget,
        },
    );
    let handles =
        register_qwen3_lm(&residency, &cfg).map_err(|e| format!("register rewriter: {e:?}"))?;
    let pipelines = compile_pipelines(backend)
        .await
        .map_err(|e| format!("compile rewriter pipelines: {e:?}"))?;
    let workspace = Workspace::new(Arc::clone(backend), Arc::clone(residency.arbiter()));
    let generator = Qwen3LmGenerator::new(cfg, max_ctx);

    let ids = generator
        .generate(
            backend,
            &pipelines,
            &residency,
            &workspace,
            &handles,
            residency.source(),
            &prompt_ids,
            MAX_NEW,
            EOS_ID,
        )
        .await
        .map_err(|e| format!("rewriter generate: {e:?}"));

    // Return all rewriter VRAM before the encoder phase regardless of outcome.
    residency.evict_all_and_free(&**backend);
    let ids = ids?;

    let text = tokenizer
        .decode(&ids, true)
        .map_err(|e| format!("detokenize rewrite: {e:?}"))?
        .trim()
        .to_string();
    if text.is_empty() {
        return Err("rewrite produced empty text".into());
    }
    Ok(text)
}
