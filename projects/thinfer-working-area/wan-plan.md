# Wan 2.2 14B A14B (MoE) + LightX2V distill -- ACTIVE port

Shipped Wan models (FastWan2.2-TI2V-5B, LongLive-2.0-5B) are the backbone. This
file = the 14B-specific deltas. Engine-wide design: `plan-details.md`. All numbers
below VERIFIED against upstream (config.json / GGUF KV / diffusers / lightx2v),
not inferred from our 5B.

## Verified facts

**DiT (both experts identical, upstream `high/low_noise_model/config.json`):**
dim 5120 (40 heads x 128), num_layers 40, ffn_dim 13824, in_dim 16, out_dim 16,
freq_dim 256, eps 1e-6, text_len 512, model_type t2v. patch (1,2,2),
qk rms_norm_across_heads, cross_attn_norm, RoPE theta 10000 (interleaved-pair,
the Wan family default -- NOT half-rot). Family-invariant consts (HEAD_DIM 128,
TEXT_DIM 4096, FREQ_DIM 256, patch, rope axis split) already in
`dit_block::config`.

**GGUF (QuantStack/Wan2.2-T2V-A14B-GGUF, Q5_K_M, 1095 tensors, arch "wan"):**
ORIGINAL-Wan single-file names (blocks.{i}.self_attn.{q,k,v,o}, cross_attn.*,
norm3, ffn.0/ffn.2, head.head, head.modulation, time_embedding.0/2,
time_projection.1, text_embedding.0/2, patch_embedding) -> `dit_gguf_renames(40)`
maps these to diffusers canonical EXACTLY (already in source.rs; verified shapes).
Block matmuls MIXED quant: q/k/o = Q5_K, v + ffn.2(down) = Q6_K. Module-level =
F16 (embedders, head.head, modulations) / F32 (patch_embedding). 40 blocks.

**LoRA (lightx2v/Wan2.2-Distill-Loras, t2v high+low rank64 1217, 800 keys each):**
keys `diffusion_model.blocks.{i}.{self_attn,cross_attn}.{q,k,v,o}.lora_{down,up}.
weight` + `ffn.{0,2}.lora_{down,up}.weight`. **lora_down = A [rank,K], lora_up =
B [N,rank]** (NOT lora_A/lora_B -- discover_specs must accept down/up). rank 64,
strength 1.0, no alpha -> plain B@A. **Covers EVERY block matmul site** (self +
cross attn qkvo + ffn up/down). Names are ORIGINAL-Wan -> fold on the GGUF's
original names, BEFORE the canonical rename.

**VAE = Wan2.1 (QuantStack VAE/Wan2.1_VAE.safetensors, diffusers AutoencoderKLWan
defaults):** base_dim 96 (decoder_base_dim None -> 96), z_dim 16, in/out 3,
dim_mult [1,2,4,4], num_res_blocks 2, temperal_downsample [F,T,T], is_residual
False, patch_size None->1, 8x spatial / 4x temporal, norm_eps 1e-12. latents_mean/
std = the 16-vecs in autoencoder_kl_wan.py:986-1021. Engine `WanVaeConfig` already
parameterized + non-residual/patch1 branches coded but UNEXERCISED (5B only hit
residual) -> add `wan2_1()` ctor + verify decode path at e2e.

**MoE expert switch:** high-noise expert (index 0) at high noise, low-noise
(index 1) at low. DISTILLED (our path) switches by STEP INDEX:
`step_index < boundary_step_index(=2)` -> high, else low. 4 steps = 2 high / 2 low.
(Full-model timestep boundary 0.875 T2V is the non-distill path; we ship distill.)

**Distill sampler (lightx2v Wan22StepDistillScheduler):** 4 steps, CFG off.
denoising_step_list [1000,750,500,250]; sigmas = linspace(1,0,1001)[:-1] then
shift s'=shift*s/(1+(shift-1)*s) with shift 5.0; timesteps = s'*1000. Euler flow:
x_next = x_t - (sigma_i - sigma_{i+1})*v (v = predicted flow). Two separate LoRAs
(one per expert), each folded into its own expert.

**Defaults (upstream):** 1280x720 (832x480 for 480p/distill); 81 frames (4n+1);
fps 16; flow shift 5.0 (distill). 8GB: DiT streams per-block.

## Architecture decision (key)

**Q8_0 block matmuls (LoRA-folded) + DENSE BF16 module-level matmuls.** Because the
LoRA covers every block matmul, folding re-encodes ALL block matmuls to Q8_0
(fold_out_enc default). The module-level weights (patch embed, condition embedder,
proj_out) are NOT folded and are tiny + run once/forward, so they stay dense bf16
(F16/F32 narrow to bf16 at upload) on a dedicated `matmul_module` pipeline -- they
do NOT go through the per-site quant block pipeline. (Transcoding them to Q8 to
"reuse one pipeline" was the original plan but it FAILED: the module sites call
`scope.matmul` with a raw weight and no dequant pre-pass, so a Q8 -- or any non-
workspace-dtype -- weight there reads as garbage -> inf/NaN. bf16 module weights +
the bf16 `matmul_module` pipeline is the fix.) i8 DP4A is OFF for the 14B (it needs
F16 acts; the residual overflows f16 so acts are bf16). Parity is BAND-based (fp32
pyref vs Q8_0 engine), standard for quantized models (q8 = canary).

**Two experts, single residency, name-prefixed.** Each expert source =
`fold(gguf_original, lora) -> rename canonical -> prefix "high."/"low."`; union
both with umt5 + vae into ONE residency. Loader registers two DiT handle sets
(prefix param). Denoise picks the set per step (steps 0-1 high, 2-3 low); evict
the high blocks at the boundary (existing phase-evict pattern). Fold cache cost:
~13GB/expert Q8 in host RAM, both coexist per-generate (~26GB) -- fine on 63GB,
flagged to bound (worklog WATCH).

## Engine touch-points (build order)

1. `dit_block::WanDitConfig::wan22_14b()` {40,13824,40,16,16}. `vae::WanVaeConfig::
   wan2_1()` (values above).
2. LoRA fold generalized: `discover_specs` accepts `lora_down`/`lora_up` (and keep
   lora_A/lora_B). Lift `ltx::lora` -> shared `common::lora` (LTX + Wan both use).
3. `loader::register_wan_dit_handles`: add `prefix: &str` + `module_transcode:
   Option<QuantKind>` (-> Q8_0 for module-level embedders/proj_out/patch on the
   GGUF path). Thread prefix into the name builders.
4. `source.rs`: `open_wan22_source` -> per-expert fold+rename+prefix, union with
   umt5(reused) + vae. PrefixSource helper (or full rename map).
5. `pipeline.rs`: de-hardcode `WanModel::load` cfg/vae_cfg (pass model spec); make
   `VAE_SCALE`/`TEMPORAL_SCALE` per-`WanVaeConfig` (Wan2.1 = 8/4). Hold two DiT
   handle sets. New `VideoSampler::Wan22Distill`/denoise path with per-step expert
   pick + the step-distill scheduler (`wan/scheduler.rs` or new module).
6. `model.rs`: `VideoModelId::Wan22T2vA14b` ("wan2.2-t2v-a14b"), manifest arm,
   video_defaults (832x480 or 1280x720), dim_multiple (Wan2.1 -> 8*2=16, same),
   frames 4n+1, fps 16.
7. `manifest.rs`: REPO_WAN22 + roles DIT_HIGH/DIT_LOW/LORA_HIGH/LORA_LOW/VAE
   (+ reuse umt5 from REPO_DIFFUSERS); VariantFiles two-DiT; new ModelManifest.
8. `executor.rs`: wan22 arm -> open_wan22_source -> WanModel::load.
9. CLI auto (ValueEnum). Web: app.js dropdown string + VIDEO_DURATION.
10. e2e parity test (`wan-e2e`) + RAM-light pyref (component-wise, tiny dims).
11. Bench + perf (trace; i8 DP4A on; f16 SDPA if applicable).

## RAM-light pyref (parity)

pyref loads the GGUF (dequant to fp32) + folds the LoRA (fp32) per expert, tiny
dims (64x64, F=5 -> f_lat=2; or 4n+1 min), 2-4 steps, CFG off. 14B fp32 won't fit
-> COMPONENT-WISE only (one block, VAE encode+decode, condition embedder), never
full-DiT (OOM). Mirror the existing FastWan `video_e2e` harness + the LTX
component-gate discipline. q8 canary band; do NOT attempt full 14B pyref.

## Do-not-retry / gotchas

- Wan RoPE3D = interleaved-pair (NOT half-rot). Already correct in rope3d.rs.
- Fold on ORIGINAL gguf names then rename (LoRA keys are original-Wan).
- The full-repo `hf download QuantStack/...` pulls EVERY quant (150GB+) -- always
  name the specific file.
- Don't commit until user browser-verifies.
