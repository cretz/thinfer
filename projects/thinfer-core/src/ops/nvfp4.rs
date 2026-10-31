//! NVFP4 (NVIDIA block-scaled FP4) weight -> dense `[N, K]` dequant pass.
//!
//! One `[N, K]` linear weight travels as a single COMPOSITE tensor, the
//! checkpoint's parts concatenated unchanged (see [`Nvfp4Layout`]):
//!
//! 1. `packed`: E2M1 codes, `[N, K/2]` bytes, LOW nibble = even `k`;
//! 2. `scales`: one E4M3 scale per 16-element block, in FlashInfer's
//!    `layout_128x4` swizzle (128-row x 4-block tiles, zero-padded to whole
//!    tiles), see [`scale_offset`];
//! 3. `alpha`: one f32 per 128-row group, `1 / global_scale` of the source
//!    tensor that group came from;
//! 4. `input_sf`: one f32 per 128-row group, the source's static activation
//!    global scale (`_nvfp4_input_global_sf`), read only by the native W4A4
//!    GEMM (the dequant pass ignores it). Every group of one composite shares
//!    the same input, so they carry the same value.
//!
//! `w[r, k] = E2M1(code) * E4M3(scale[r, k/16]) * alpha[r / 128]`. Keeping the
//! source layout means streaming a weight is a plain concatenation of byte
//! ranges (no host repack per step); the swizzle is undone here, on the GPU.
//! Because a 128-row group is exactly one scale tile, a composite may also be
//! ASSEMBLED from 128-row groups of several source tensors (e.g. per-head
//! interleaved q/k/v, each with its own global scale) with no requantization.
//!
//! The output matches the block-quant dequant ([`super::dequant`]): N-major
//! dense bf16-packed or f16 pairs, what `MatMulConfig.b_nmajor = true` reads.

use crate::backend::Backend;
use crate::ops::dequant::{DequantBufs, DequantTarget};

/// E2M1 code -> value (sign in bit 3).
pub const E2M1: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// Elements per scale block.
pub const BLOCK: usize = 16;

/// Byte layout of one composite `[N, K]` NVFP4 tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Nvfp4Layout {
    pub n: usize,
    pub k: usize,
}

impl Nvfp4Layout {
    pub fn new(n: usize, k: usize) -> Self {
        assert!(
            k.is_multiple_of(BLOCK),
            "nvfp4: K={k} must be a multiple of {BLOCK}"
        );
        Self { n, k }
    }
    pub fn packed_len(&self) -> usize {
        self.n * self.k / 2
    }
    /// Scale blocks per row, padded to the 4-block tile width.
    pub fn blocks_padded(&self) -> usize {
        (self.k / BLOCK).div_ceil(4) * 4
    }
    /// The swizzled scale buffer, rows padded to 128.
    pub fn scales_len(&self) -> usize {
        self.n.div_ceil(128) * 128 * self.blocks_padded()
    }
    pub fn scales_offset(&self) -> usize {
        self.packed_len()
    }
    pub fn alpha_offset(&self) -> usize {
        self.packed_len() + self.scales_len()
    }
    /// 128-row groups, one alpha (and one input scale) each.
    pub fn groups(&self) -> usize {
        self.n.div_ceil(128)
    }
    pub fn input_sf_offset(&self) -> usize {
        self.alpha_offset() + 4 * self.groups()
    }
    pub fn total_len(&self) -> usize {
        self.input_sf_offset() + 4 * self.groups()
    }
}

/// Byte offset of row `r`, block `b` in a `layout_128x4` scale buffer whose
/// rows hold `blocks_padded` blocks.
pub fn scale_offset(r: usize, b: usize, blocks_padded: usize) -> usize {
    ((r / 128) * (blocks_padded / 4) + b / 4) * 512 + (r % 32) * 16 + ((r % 128) / 32) * 4 + b % 4
}

/// E4M3 (fn variant, bias 7) byte -> value.
pub fn e4m3(v: u8) -> f32 {
    let sign = if v & 0x80 != 0 { -1.0 } else { 1.0 };
    let e = (v >> 3) & 0xF;
    let f = (v & 7) as f32;
    sign * if e == 0 {
        f * 2f32.powi(-9)
    } else {
        (1.0 + f / 8.0) * 2f32.powi(e as i32 - 7)
    }
}

/// CPU reference: a composite tensor -> row-major `[N, K]` f32.
pub fn dequantize(bytes: &[u8], layout: Nvfp4Layout) -> Vec<f32> {
    assert_eq!(bytes.len(), layout.total_len(), "nvfp4 composite length");
    let a = layout.alpha_offset();
    let alpha = |r: usize| {
        let o = a + 4 * (r / 128);
        f32::from_le_bytes(bytes[o..o + 4].try_into().unwrap())
    };
    let scales = &bytes[layout.scales_offset()..a];
    let bp = layout.blocks_padded();
    let mut out = vec![0f32; layout.n * layout.k];
    for r in 0..layout.n {
        for c in 0..layout.k {
            let byte = bytes[(r * layout.k + c) / 2];
            let code = if c % 2 == 0 { byte & 0xF } else { byte >> 4 };
            let s = e4m3(scales[scale_offset(r, c / BLOCK, bp)]);
            out[r * layout.k + c] = E2M1[code as usize] * s * alpha(r);
        }
    }
    out
}

/// Dequant WGSL: one thread per packed word (8 elements). Dispatch with
/// [`dispatch`]; dims uniform is `(n, k, 0, 0)` like the block-quant pass.
pub fn build_wgsl(target: DequantTarget) -> String {
    let (enable, prelude, dense_decl, store) = match target {
        DequantTarget::Bf16 => (
            "",
            "fn round_bf16(x: f32) -> u32 {
    let b32 = bitcast<u32>(x);
    if ((b32 & 0x7F800000u) == 0x7F800000u) { return (b32 >> 16u) & 0xFFFFu; }
    let l = (b32 >> 16u) & 1u;
    return ((b32 + 0x7FFFu + l) >> 16u) & 0xFFFFu;
}
",
            "@group(0) @binding(1) var<storage, read_write> b_dense: array<u32>;",
            "b_dense[base + i] = round_bf16(lo) | (round_bf16(hi) << 16u);",
        ),
        DequantTarget::F16 => (
            "enable f16;
",
            "",
            "@group(0) @binding(1) var<storage, read_write> b_dense: array<vec2<f16>>;",
            "b_dense[base + i] = vec2<f16>(clamp(vec2<f32>(lo, hi), vec2<f32>(-65504.0), vec2<f32>(65504.0)));",
        ),
    };
    format!(
        r#"{enable}struct Dims {{ n: u32, k: u32, _pad0: u32, _pad1: u32 }};

@group(0) @binding(0) var<storage, read> b: array<u32>;
{dense_decl}
@group(0) @binding(2) var<uniform> d: Dims;

{prelude}
const E2M1 = array<f32, 16>(0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0);

fn byte_at(off: u32) -> u32 {{
    return (b[off >> 2u] >> ((off & 3u) * 8u)) & 0xFFu;
}}

fn e4m3(v: u32) -> f32 {{
    let e = (v >> 3u) & 15u;
    let f = f32(v & 7u);
    var m: f32;
    if (e == 0u) {{ m = f * exp2(-9.0); }} else {{ m = (1.0 + f / 8.0) * exp2(f32(e) - 7.0); }}
    return select(m, -m, (v & 128u) != 0u);
}}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {{
    let words_per_row = d.k / 8u;
    let w = gid.y * (ng.x * 64u) + gid.x;
    if (w >= d.n * words_per_row) {{ return; }}
    let r = w / words_per_row;
    let c8 = w % words_per_row;
    let bp = ((d.k / 16u + 3u) / 4u) * 4u;
    let scales_off = d.n * d.k / 2u;
    let alpha_off = scales_off + ((d.n + 127u) / 128u) * 128u * bp;
    let blk = c8 / 2u;
    let sw = ((r / 128u) * (bp / 4u) + blk / 4u) * 512u + (r % 32u) * 16u + ((r % 128u) / 32u) * 4u + (blk % 4u);
    let s = e4m3(byte_at(scales_off + sw)) * bitcast<f32>(b[(alpha_off >> 2u) + r / 128u]);
    let word = b[w];
    let base = (r * d.k + c8 * 8u) >> 1u;
    for (var i = 0u; i < 4u; i = i + 1u) {{
        let lo = E2M1[(word >> (8u * i)) & 15u] * s;
        let hi = E2M1[(word >> (8u * i + 4u)) & 15u] * s;
        {store}
    }}
}}
"#
    )
}

/// Dispatch one NVFP4 dequant pass over an `[n, k]` composite weight.
pub fn dispatch<B: Backend>(
    backend: &B,
    encoder: &mut B::CommandEncoder,
    pipeline: &B::Pipeline,
    bufs: &DequantBufs<'_>,
    n: u32,
    k: u32,
) -> Result<(), B::Error> {
    let bindings = [
        bufs.b_quant.binding(0),
        bufs.b_dense.binding(1),
        bufs.dims.binding(2),
    ];
    let words = n * (k / 8);
    backend.dispatch(
        encoder,
        pipeline,
        &bindings,
        super::linear_workgroups(words, 64),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The independent swizzle from FastVideo's MLX test
    /// (`test_minimax_h3_resident_nvfp4._swizzle`).
    fn swizzle_ref(r: usize, c: usize, padded_cols: usize) -> usize {
        ((((r / 128) * (padded_cols / 4) + c / 4) * 32 + r % 32) * 4 + (r % 128) / 32) * 4 + c % 4
    }

    #[test]
    fn scale_offset_matches_reference_swizzle() {
        for (n, kb) in [(128usize, 8usize), (140, 7), (256, 4)] {
            let bp = kb.div_ceil(4) * 4;
            let mut seen = std::collections::HashSet::new();
            for r in 0..n {
                for c in 0..kb {
                    let o = scale_offset(r, c, bp);
                    assert_eq!(o, swizzle_ref(r, c, bp));
                    assert!(seen.insert(o), "offset collision at ({r},{c})");
                    assert!(o < n.div_ceil(128) * 128 * bp);
                }
            }
        }
    }

    #[test]
    fn e4m3_decodes_known_values() {
        assert_eq!(e4m3(0x38), 1.0); // e=7, f=0
        assert_eq!(e4m3(0x40), 2.0);
        assert_eq!(e4m3(0xB8), -1.0);
        assert_eq!(e4m3(0x01), 2f32.powi(-9));
        assert_eq!(e4m3(0x7E), 448.0); // max finite
    }

    #[test]
    fn dequantize_reads_low_nibble_first_and_scales() {
        let layout = Nvfp4Layout::new(1, 16);
        let mut bytes = vec![0u8; layout.total_len()];
        bytes[0] = 0x21; // k=0 -> code 1 (0.5), k=1 -> code 2 (1.0)
        bytes[layout.scales_offset() + scale_offset(0, 0, layout.blocks_padded())] = 0x40; // 2.0
        let a = layout.alpha_offset();
        bytes[a..a + 4].copy_from_slice(&0.25f32.to_le_bytes());
        let w = dequantize(&bytes, layout);
        assert_eq!(&w[..2], &[0.5 * 2.0 * 0.25, 1.0 * 2.0 * 0.25]);
    }
}
