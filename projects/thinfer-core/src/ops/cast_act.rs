use super::{Op, WgslConfig};
use crate::backend::{BindingKind, BindingLayout};
use crate::tensor::F32;

// Packed-word bf16<->f16 activation casts. Both directions read/write
// `array<u32>`, one thread per word (two packed activations), so the buffers
// are byte-for-byte the same size (2 bytes/elem either way). Self-contained
// WGSL: neither kernel needs `enable f16` because the conversions ride the
// built-in `pack2x16float`/`unpack2x16float` (f16 <-> two-packed-f32-word)
// intrinsics, so we never declare an `f16` value and avoid mixing the
// act_bf16/act_f16 preludes.
//
// Used by the Qwen-Image DiT fast-attention path: the residual stream stays
// bf16 (large-outlier channels exceed f16's +-65504), but Q/K/V are O(1)
// post-rmsnorm/rope and f16-safe, so the block casts only Q/K/V to f16 for the
// subgroup SDPA kernel and casts its output back to bf16. See
// `qwen_image::dit::block_attn`.

const LAYOUT: &[BindingLayout] = &[
    BindingLayout {
        slot: 0,
        kind: BindingKind::StorageRead,
    },
    BindingLayout {
        slot: 1,
        kind: BindingKind::StorageReadWrite,
    },
];

// bf16 pair -> f16 pair. Unpacks each bf16 lane to f32 (shift into the high
// 16 bits of a float), clamps to the f16 finite range so an out-of-range
// magnitude rounds to +-65504 instead of inf (Q/K/V should never reach this,
// but V can be larger than Q/K), then packs two f16 via `pack2x16float`.
const WGSL_BF16_TO_F16: &str = r#"
@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let i = gid.y * (ng.x * 64u) + gid.x;
    if (i >= arrayLength(&dst)) { return; }
    let w = src[i];
    let lo = bitcast<f32>((w & 0xFFFFu) << 16u);
    let hi = bitcast<f32>(w & 0xFFFF0000u);
    let c = clamp(vec2<f32>(lo, hi), vec2<f32>(-65504.0), vec2<f32>(65504.0));
    dst[i] = pack2x16float(c);
}
"#;

// f16 pair -> bf16 pair. Unpacks two f16 to f32 via `unpack2x16float`, then
// RNE-rounds each to bf16 (NaN/inf pass through), matching the act_store bf16
// rounding used everywhere else.
const WGSL_F16_TO_BF16: &str = r#"
fn round_bf16(x: f32) -> u32 {
    let b = bitcast<u32>(x);
    if ((b & 0x7F800000u) == 0x7F800000u) { return (b >> 16u) & 0xFFFFu; }
    let l = (b >> 16u) & 1u;
    return ((b + 0x7FFFu + l) >> 16u) & 0xFFFFu;
}

@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let i = gid.y * (ng.x * 64u) + gid.x;
    if (i >= arrayLength(&dst)) { return; }
    let v = unpack2x16float(src[i]);
    dst[i] = round_bf16(v.x) | (round_bf16(v.y) << 16u);
}
"#;

/// bf16-packed acts -> f16-packed acts (clamped to the f16 finite range). The
/// `F32` dtype is the dispatch granularity only: it makes `dispatch_op` count
/// `u32` words (one thread per packed pair), NOT a value type.
pub struct Bf16ToF16;

impl Op for Bf16ToF16 {
    const KERNEL_ID: &'static str = "cast_act.bf16_to_f16";
    type Dtype = F32;
    const INPUTS: &'static [&'static str] = &["cast_act/src"];
    const OUTPUT: &'static str = "cast_act/dst";
    fn wgsl(_cfg: &WgslConfig) -> &'static str {
        WGSL_BF16_TO_F16
    }
    fn layout() -> &'static [BindingLayout] {
        LAYOUT
    }
}

/// f16-packed acts -> bf16-packed acts (RNE rounding). See [`Bf16ToF16`] for
/// the `F32`-dtype note.
pub struct F16ToBf16;

impl Op for F16ToBf16 {
    const KERNEL_ID: &'static str = "cast_act.f16_to_bf16";
    type Dtype = F32;
    const INPUTS: &'static [&'static str] = &["cast_act/src"];
    const OUTPUT: &'static str = "cast_act/dst";
    fn wgsl(_cfg: &WgslConfig) -> &'static str {
        WGSL_F16_TO_BF16
    }
    fn layout() -> &'static [BindingLayout] {
        LAYOUT
    }
}
