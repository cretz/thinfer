//! In-place block-diagonal Hadamard rotation of activation rows (ComfyUI
//! `convrot`): every 256-wide group of every row becomes `x @ H`, `H` the
//! normalized REGULAR Hadamard `kron(H4, H4, H4, H4) / 16` with
//! `H4 = [[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]]` (comfy-kitchen
//! `_build_hadamard`). `H` is symmetric, so this is also `x @ H^T`.
//!
//! Kernel: one 64-thread workgroup per (row, group). The group sits in shared
//! memory as f32; four radix-4 stages each apply `H4` along one base-4 digit of
//! the column index (Kronecker factors act per digit), one 4-tuple per thread:
//! `out_j = s - 2 * in_(3-j)` with `s` the tuple sum. f32 math throughout,
//! one rounding at the store.
//!
//! Layout: 0 = x (read_write, `[rows, k]` act dtype, rotated in place),
//! 1 = uniform `{rows, k, _, _}`. Grid: `[k / 256, min(rows, 65535),
//! ceil(rows / 65535)]`.

use crate::backend::{BindingKind, BindingLayout};
use crate::ops::ActDtype;

/// Rotation group (the only `convrot_groupsize` in the wild).
pub const GROUP: u32 = 256;

macro_rules! hadamard_wgsl {
    ($prelude:expr, $word:expr, $ld2:expr, $st2:expr) => {
        concat!(
            $prelude,
            "struct U { rows: u32, k: u32, _a: u32, _b: u32 };\n",
            "@group(0) @binding(0) var<storage, read_write> x: array<",
            $word,
            ">;\n",
            "@group(0) @binding(1) var<uniform> u: U;\n",
            $ld2,
            $st2,
            r#"
var<workgroup> g: array<f32, 256>;

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = wid.z * 65535u + wid.y;
    if (row >= u.rows) { return; }
    let t = lid.x;
    // Word offset of this group (2 elements per word).
    let base = (row * u.k + wid.x * 256u) >> 1u;
    let a = ld2(base + t);
    let b = ld2(base + 64u + t);
    g[2u * t] = a.x;
    g[2u * t + 1u] = a.y;
    g[128u + 2u * t] = b.x;
    g[128u + 2u * t + 1u] = b.y;
    workgroupBarrier();
    // Stage s transforms digit s (stride 4^s): tuple = the 4 indices that
    // differ only in that digit.
    for (var s = 0u; s < 4u; s = s + 1u) {
        let stride = 1u << (2u * s);
        let lo = t % stride;
        let i0 = (t / stride) * stride * 4u + lo;
        let v0 = g[i0];
        let v1 = g[i0 + stride];
        let v2 = g[i0 + 2u * stride];
        let v3 = g[i0 + 3u * stride];
        let sum = v0 + v1 + v2 + v3;
        workgroupBarrier();
        g[i0] = sum - 2.0 * v3;
        g[i0 + stride] = sum - 2.0 * v2;
        g[i0 + 2u * stride] = sum - 2.0 * v1;
        g[i0 + 3u * stride] = sum - 2.0 * v0;
        workgroupBarrier();
    }
    let inv = 1.0 / 16.0;
    st2(base + t, vec2<f32>(g[2u * t], g[2u * t + 1u]) * inv);
    st2(base + 64u + t, vec2<f32>(g[128u + 2u * t], g[128u + 2u * t + 1u]) * inv);
}
"#
        )
    };
}

const WGSL_F32: &str = hadamard_wgsl!(
    "",
    "vec2<f32>",
    "fn ld2(i: u32) -> vec2<f32> { return x[i]; }\n",
    "fn st2(i: u32, v: vec2<f32>) { x[i] = v; }\n"
);

const WGSL_F16: &str = hadamard_wgsl!(
    "enable f16;\n",
    "vec2<f16>",
    "fn ld2(i: u32) -> vec2<f32> { return vec2<f32>(x[i]); }\n",
    "fn st2(i: u32, v: vec2<f32>) { x[i] = vec2<f16>(clamp(v, vec2<f32>(-65504.0), vec2<f32>(65504.0))); }\n"
);

const WGSL_BF16: &str = hadamard_wgsl!(
    crate::act_bf16_prelude!(),
    "u32",
    "fn ld2(i: u32) -> vec2<f32> { return unpack_bf16x2(x[i]); }\n",
    "fn st2(i: u32, v: vec2<f32>) { x[i] = pack_bf16x2(v.x, v.y); }\n"
);

pub fn wgsl(act: ActDtype) -> &'static str {
    match act {
        ActDtype::F32 => WGSL_F32,
        ActDtype::F16 => WGSL_F16,
        ActDtype::Bf16 => WGSL_BF16,
        ActDtype::I8 => unreachable!("hadamard runs on dense acts"),
    }
}

pub fn layout() -> &'static [BindingLayout] {
    &[
        BindingLayout {
            slot: 0,
            kind: BindingKind::StorageReadWrite,
        },
        BindingLayout {
            slot: 1,
            kind: BindingKind::Uniform,
        },
    ]
}

/// Launch grid for `rows` rows of width `k` (`k % 256 == 0`).
pub fn workgroups(rows: u32, k: u32) -> [u32; 3] {
    debug_assert!(k.is_multiple_of(GROUP));
    [k / GROUP, rows.min(65535), rows.div_ceil(65535)]
}

/// Host reference: `x @ H` per 256 group, in place (tests, CPU fallbacks).
pub fn rotate_host(x: &mut [f32], k: usize) {
    let g = GROUP as usize;
    for row in x.chunks_mut(k) {
        for grp in row.chunks_mut(g) {
            for s in 0..4 {
                let stride = 1usize << (2 * s);
                for t in 0..64 {
                    let i0 = (t / stride) * stride * 4 + t % stride;
                    let v = [
                        grp[i0],
                        grp[i0 + stride],
                        grp[i0 + 2 * stride],
                        grp[i0 + 3 * stride],
                    ];
                    let sum = v[0] + v[1] + v[2] + v[3];
                    for (j, vj) in v.iter().rev().enumerate() {
                        grp[i0 + j * stride] = sum - 2.0 * vj;
                    }
                }
            }
            for v in grp.iter_mut() {
                *v /= 16.0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The staged transform equals the dense product with the Kronecker matrix.
    #[test]
    fn staged_matches_dense_kron() {
        let h4 = [
            [1.0f32, 1., 1., -1.],
            [1., 1., -1., 1.],
            [1., -1., 1., 1.],
            [-1., 1., 1., 1.],
        ];
        let h = |i: usize, j: usize| -> f32 {
            (0..4)
                .map(|d| h4[(i >> (2 * d)) & 3][(j >> (2 * d)) & 3])
                .product::<f32>()
                / 16.0
        };
        let x: Vec<f32> = (0..512)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 7.0)
            .collect();
        let mut got = x.clone();
        rotate_host(&mut got, 512);
        for grp in 0..2 {
            for j in 0..256 {
                let want: f32 = (0..256).map(|i| x[grp * 256 + i] * h(i, j)).sum();
                assert!(
                    (got[grp * 256 + j] - want).abs() < 1e-4,
                    "{j}: {} vs {want}",
                    got[grp * 256 + j]
                );
            }
        }
    }

    /// Orthogonal and symmetric: rotating twice is the identity.
    #[test]
    fn involution() {
        let x: Vec<f32> = (0..256).map(|i| (i as f32).sin()).collect();
        let mut y = x.clone();
        rotate_host(&mut y, 256);
        rotate_host(&mut y, 256);
        for (a, b) in x.iter().zip(&y) {
            assert!((a - b).abs() < 1e-5);
        }
    }
}
