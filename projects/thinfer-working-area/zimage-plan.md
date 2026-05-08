# Z-Image plan

Per-model details. Engine-wide rules in `plan-details.md`.

## Sources

- Code: github.com/Tongyi-MAI/Z-Image (paths below are relative to that repo).
- Diffusers ZImage pipeline: github.com/huggingface/diffusers (two PRs merged).
- Weights: hf.co/tsqn/Z-Image-Turbo_fp32-fp16-bf16_full_and_ema-only (fp16 ema-only variant for M1).

## M1

bf16 ema-only storage, fp32 compute, end-to-end CLI then browser. 8 NFE turbo. Provably correct baseline that diffs against PyTorch within fp32 tolerance. fp16 storage is broken for Z-Image (bf16-trained, magnitude clamp at 65504); fp16 kernels are off the table since per-input safety isn't statically verifiable. Memory wins come from M2's quant kernels, not from compute precision.

## DiT block (`src/zimage/transformer.py:143` `ZImageTransformerBlock`)

Self-attn over concatenated `[text || image]` (NOT MMDiT). Two flavors: `modulation=True` (main) and `modulation=False` (context refiner, plain pre+post norm).

Per block (`modulation=True`):
- adaLN: `Linear(t) -> 4 chunks (scale_msa, gate_msa, scale_mlp, gate_mlp)`. `gate.tanh()`, `scale = 1+scale`. No shift.
- Double norm: pre-norm on sub-layer input AND post-norm on sub-layer output. 4 RMSNorm per block.
- Attn: separate Q/K/V Linear no-bias, GQA-capable (default `n_kv_heads = n_heads = 30`). Per-head RMSNorm on Q/K, then 3-axis RoPE (temporal+H+W) via complex-pair multiply. SDPA bidirectional, padding mask only.
- FFN: SwiGLU `w2(silu(w1) * w3)`, no bias, `hidden = dim/3*8` (3840 -> 10240).

## Op gap (over add/mul/silu)

matmul, rmsnorm (weight, eps), softmax, rope, sdpa (compose M1).

`Op` trait extensions: weight binding, small-uniform binding.

## Next slices

text encoder, VAE decoder, scheduler, weight-name mapping, adaLN time-embed source.
