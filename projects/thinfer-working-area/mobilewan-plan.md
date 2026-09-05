# mobilewan (Qualcomm-AI-Research/mobilewan) - T2V

STATUS: DONE 2026-07-21 (ReHyAt hybrid attention, eyeball PASS 512x320/33f). See
"FAITHFUL PATH" + "CONCRETE SPEC" below for the impl. Kernels in
thinfer-core/src/ops/rehyat.rs (F32, GPU unit-tested); forward is an F32 island in
`WanDitBlock::self_attn_mobilewan` with an inner temporal-chunk loop. Reference is
now public (cloned to third-party/mobilewan). Two GOTCHAS that cost time:
- `tpf` (tokens per patched-frame) = pph*ppw. `grid.ppf` is the FRAME COUNT, not
  tokens-per-frame -- passing it scrambled the chunk grouping into rainbow chaos.
- Chunk loop is INNER (per block); == the reference's outer-chunk/inner-block order,
  so the DiT stays block-major STREAMING. Local windowed attn is memory-frugal.
OPEN: 480p eyeball; DMD-3 vs UniPc-4 default; perf (bf16-packed kernels drop the
per-block bf16->f32 casts); latent parity pyref (needs Wan2.2-TI2V-5B-Diffusers base).

Efficient T2V. Wan2.2-TI2V-5B DiT backbone with two novelties. DiT-only repo
(5 safetensors shards ~10GB + index.json); umT5 + VAE + tokenizer reuse the
FastWan bundle. Weights downloaded to HF cache. arXiv:2607.06173 (MobileWan) +
2601.04342 (ReHyAt). Qualcomm GitHub code NOT public (404) -> NO parity ref;
eyeball-only validation.

## Backbone (== FastWan2.2-TI2V-5B, reuse wan/*)

dim 3072, head_dim 128, ffn 14336, 30 layers, patch (1,2,2), 48ch latent,
umT5-XXL (dim 4096), Wan VAE z48, RoPE3D interleaved-pair, AdaLN 6-way,
cross-attn attn2 (full 24 heads), gelu-tanh FFN. 3-step DMD distilled, 81f
480x832 16fps. Token layout T-major: row = t*(pph*ppw) + h*ppw + w.
81f 480x832 -> f_lat 21, pph 15, ppw 26, n_tok 8190, period 390.

## Novelties

1. **Per-block PRUNED head counts** on attn1 (self-attn) only. h_i in 11..=24,
   read from `blocks.{i}.attn1.base_attention.to_q.weight.shape[0]/128`.
   VERIFIED table (blocks 20-26 = [17,16,11,15,16,18,15]). Total 557/720 heads
   (~23% pruned). Cross-attn (attn2) keeps full 24 heads. inner_i = h_i*128;
   q/k/v [inner_i,3072], o [3072,inner_i], norm_q/norm_k [inner_i].
2. **ReHyAt hybrid attn** (softmax-local + linear-global poly feature maps
   q_embed/k_embed grouped-conv 128->384->384). NO public code.

## DECISION (SUPERSEDED -- was wrong): full softmax, skip ReHyAt

Original bet (made when Qualcomm code was 404): "ReHyAt approximates full
softmax; skip q_embed/k_embed, run full softmax." REFUTED by the now-public
reference (third-party/mobilewan, cloned 2026-07-21). Result of that bet: DiT
output collapses to a flat gray blob (denoise -> near-zero-variance velocity).
FastWan sanity run at identical settings renders cleanly, so the fault is
isolated to the self-attn novelty.

## ROOT CAUSE (from the reference)

`rehyat_utils.surgery()` overrides `transformer.forward` with
`chunk_major_transformer_forward`, so stock `pipe(...)` sampling runs CHUNK-MAJOR,
not block-major full-attention:
- Video is processed in temporal chunks of `REHYAT_CHUNK_SIZE=4` patched-frames
  (`REHYAT_CHUNK_OVERLAP=2`); each chunk threads through ALL 30 blocks before the
  next chunk. Per-block recurrent state persists across chunks (`reset_chunk_state`
  once at the top).
- `rnn_overlap_step` (attention.py): shared-denominator hybrid = LOCAL windowed
  softmax (this chunk + 2-frame overlap cache, ~6 frames) PLUS a linear-attention
  recurrent memory term `phi_q·S / phi_q·z`. `S += phi_k (x) v`, `z += phi_k` fold
  the non-overlap prefix into global linear memory.
- `phi = q_embed/k_embed` (embedding.py `FeatureEmbeddingPolynomial`): per-head
  grouped 1x1 conv `head_dim -> head_dim*degree` (128->384), GELU, grouped 1x1
  384->384, ReLU, then polynomial expansion (split into `degree_max=3` chunks,
  raise chunk i to power i, concat) -> phi_dim = 3*head_dim = 384.

Full global softmax with phi SKIPPED is a different attention the trained weights
never saw (OOD) -> collapse. The q_embed/k_embed maps + chunked recurrence are
LOAD-BEARING, not an optional perf tweak.

## FAITHFUL PATH (block-major inner-chunk; fits budget, no re-stream)

KEY INSIGHT: the chunk loop is the INNER loop, per block. The only cross-chunk
dependency is each block's own recurrent self-attn state (s,z + overlap cache).
So `for block { reset state; for chunk in order { self_attn(chunk, state) } }`
is IDENTICAL to the reference `for chunk { for block }`: block N's input for chunk
c is block N-1's output for chunk c in both orders, and each block sees chunks in
temporal order so its state is built identically. => DiT stays block-major
STREAMING (streamed once per step, no re-stream). Local windowed attention (~6
frames) is also far smaller than global softmax -> fixes the 480p OOM too. This IS
the "mobile" efficiency; it fits our budget by design.

Per-token, chunk-independent (compute ONCE on the full sequence per block): q/k/v
proj, qk-norm, RoPE, phi_q=q_embed(q_scaled_roped), phi_k=k_embed(k_roped). Only
the recurrent attention is chunked.

Build:
1. Loader (wan/loader.rs): load q_embed/k_embed per block (embed.0 grouped 1x1
   conv 128->384 groups=h_i +bias; embed.2 grouped 384->384 +bias; GELU/ReLU no
   params). Keys `blocks.{i}.attn1.{q,k}_embed.embed.{0,2}.{weight,bias}`, pruned
   to h_i groups. STOP skipping these.
2. Kernels: grouped per-head matmul (128->384, 384->384) + GELU + ReLU; poly
   expansion (split embedded [.,h_i,384] into 3 x128 chunks, chunk_i ** i for
   i=1,2,3, concat -> phi_dim 384; note ReLU makes chunks >=0 so phi>=0); the
   ReHyAt step (local windowed softmax with overlap cache + linear recurrent
   numer/denom, shared denominator per attention.py rnn_overlap_step).
3. Forward: block-major, inner temporal-chunk loop (chunk=4 patched-frames,
   overlap=2) with per-block recurrent state s[h_i,384,128], z[h_i,384], overlap
   K/V cache. Cross-attn+FFN stay full-sequence (per-token, chunk-independent).

Parity is now POSSIBLE (reference is public): reference needs
`Wan-AI/Wan2.2-TI2V-5B-Diffusers` base + the mobilewan checkpoint + diffusers/
accelerate. Consider a latent-level pyref gate.

## CONCRETE SPEC (ground-truthed from checkpoint header 2026-07-21)

Keys: `blocks.{i}.attn1.{q,k}_embed.embed.{0,2}.{weight,bias}` (bf16). Per block,
h_i = `attn1.base_attention.to_q` rows / 128 (block0 h_i=20; the worklog table
lists all 30). head_dim=128, degree_max=3, phi_dim = degree_max*head_dim = 384.
- embed.0 (conv1): weight `[384*h_i, 128, 1, 1]`, bias `[384*h_i]`; grouped
  groups=h_i => per-head linear 128->384 (in_pg=128, out_pg=384). Then GELU.
- embed.2 (conv2): weight `[384*h_i, 384, 1, 1]`, bias `[384*h_i]`; groups=h_i =>
  per-head 384->384. Then ReLU (so embedded >= 0).
- poly: split embedded [rows, h_i, 384] into 3 chunks of 128 (head_dim); raise
  chunk c to power c (c=1,2,3); concat -> phi [rows, h_i, 384].
kernel=1 => NO spatial mixing; it is a per-token per-head (block-diagonal) linear.
Layout note: reference does `b h l d -> b (h d) l 1` (channel = h*d, spatial=l).
thinfer acts are row-major [rows, inner_i]. Implement phi as a per-head/grouped
matmul in [rows, h_i, d] to avoid the NCL transpose (or transpose into Conv1dOp
which is NCL, groups-aware). New `pow` elementwise kernel for the poly step.

Attention scale: query pre-scaled by 1/sqrt(head_dim); local logits = q@k_win^T
with max-subtraction over the LOCAL window only; exp clamp min -60; recurrent numer
`phi_q·S` and denom `phi_q·z` are added UNSCALED (not max-shifted). S[h,384,128] +=
phi_k_prefix(x)v_prefix; z[h,384] += sum phi_k_prefix. eps=1e-6. Reset S,z,overlap
per block. overlap cache = last `overlap_tokens` of prev chunk's K,V, where
overlap_tokens = min(2, chunk_frames)*pph*ppw. chunk = 4 patched-frames.

## Integration (wan/*, least-invasive; from Explore agent map)

- `WanDitConfig` (dit_block.rs:101): add `rehyat: bool` (or variable-head marker).
  `WanVariant::mobilewan()` (pipeline.rs ~403) clones fastwan_ti2v_5b, rehyat=true,
  block_transcode None (bf16 acts, parity-first like longlive).
- Per-block heads: loader reads h_i from to_q shape; add `n_heads` to
  `WanDitBlockHandles`. Block SHAPE stays backbone (dim=inner=3072, 24 heads) so
  cross_ffn + FFN + residual are UNCHANGED and correct.
- NEW `WanDitBlock::self_attn_mobilewan(n1m, freqs, w, rows, h_i)`: bespoke
  self-attn with inner_i=h_i*128. Reuse `biased_proj` (k=shape.dim=3072, n=inner_i),
  `op_rmsnorm(inner_i)`, inline rope over h_i heads
  (`scope.rope(bp.rope, .., u32x4(rows,h_i,pairs), .., rows, h_i, pairs)`),
  `op_sdpa(.., h_i, h_i, 128, ..)`, then out-proj `lin(sa, o_w, .., rows, n=3072,
  k=inner_i, .., bp.matmul_proj ..)` + bias. NO i8/quant_for_sdpa (dense).
- `WanDitBlock::forward_mobilewan`: reuse norm1+modulate + gated residual from
  `forward`; call self_attn_mobilewan; then reuse `cross_ffn` verbatim.
- `WanDit::forward_with_taps` (dit.rs ~492): force n_tiles=1 when cfg.rehyat
  (untiled path); per-idx block uses handles[idx].n_heads. Bypasses
  self_qkv_tile/self_sdpa/PreparedTileWeights/i8 machinery entirely.
- Loader (wan/loader.rs): self-attn ids under `blocks.{i}.attn1.base_attention.*`
  (vs FastWan `attn1.*`); cross `attn2.*` unchanged. SKIP q_embed/k_embed.
- Source: mobilewan uses diffusers-style names; only attn1 adds `base_attention`
  nesting. FastWan Plain arm needs no renames; add the base_attention handling in
  the mobilewan loader (build the ids), or a small rename map.

## Wiring (VideoModelId::Mobilewan)

manifest.rs (wan): REPO_MOBILEWAN + DIT_MOBILEWAN_1..5 + index roles +
MOBILEWAN_WEIGHT_ROLES [dit1..5, te1..3, VAE] + VariantFiles + MANIFEST entries.
model.rs: VideoModelId::Mobilewan + Display + video_defaults (480x832) + fps 16 +
default_steps 3 + manifest + is-predicate arms (mostly fall to Wan default).
executor.rs run_video: source branch selecting WanVariant::mobilewan() (reuse
run_wan). request.rs required_files. app.js MODELS.video + steps 3 + duration.

## RAM / running

TI2V-5B class (FastWan footprint, ~6GB host), NOT wan22/anyflow-heavy. Still
CONFIRM with user before any video run (RAM sensitivity). e2e: video_e2e-style;
`THINFER_TRACE=1 THINFER_POWER_PREF=high` + PNG/mp4 dir, serial. Card 8GB, budget
<8G. No pyref (no reference impl).
