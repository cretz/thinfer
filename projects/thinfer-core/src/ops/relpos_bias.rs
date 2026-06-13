//! Expand a compact T5/umT5 relative-position bias into a dense per-head
//! additive attention mask `[H, S, S]` for SDPA's per-head mask mode
//! (`has_mask == 2`).
//!
//! Inputs:
//! - `table:      [num_buckets, H]` f32. The HF `relative_attention_bias.weight`
//!   (an `Embedding[num_buckets, n_heads]`), decoded to f32. Tiny (e.g. 32x64),
//!   uploaded once per layer.
//! - `bucket_map: [S, S]` u32. CPU-built bucket index per (query i, key j),
//!   `bucket_map[i*S + j] = relative_position_bucket(j - i)` (see
//!   [`relpos_bucket_map`]). Row-major, so the flat index is itself the
//!   (i, j) offset.
//! - `out:        [H, S, S]` act-dtype. `out[h,i,j] = table[bucket_map[i,j]*H + h]`.
//!
//! This is the same math HF computes as a dense `[1, H, S, S]` bias added to
//! the pre-softmax scores; here the upload stays compact (table + bucket_map)
//! and only the GPU-side `[H,S,S]` is materialized, fed straight to SDPA as the
//! per-head mask plane. umT5 attention scale is 1.0, so SDPA's `dot*scale +
//! bias` reduces to `dot + bias`.
//!
//! Layout: 0=Table, 1=BucketMap, 2=Out, 3=Uniform `{H, S, _, _}`.

use super::{ActDtype, WgslConfig};
use crate::backend::{Backend, BindingKind, BindingLayout, BufRef};
use crate::tensor::{ComputeDtype, F32};
use crate::{act_bf16_prelude, act_f16_prelude, act_store_bf16q, act_store_f32};

pub trait RelposBiasOp {
    const KERNEL_ID: &'static str;
    type Dtype: ComputeDtype;
    const TABLE: &'static str;
    const BUCKET_MAP: &'static str;
    const DIMS: &'static str;
    const OUTPUT: &'static str;

    fn wgsl(cfg: &WgslConfig) -> &'static str;
    fn layout() -> &'static [BindingLayout];

    fn workgroups(n: u32) -> [u32; 3] {
        super::linear_workgroups(n, 64)
    }
}

pub struct RelposBiasBufs<'a> {
    pub table: &'a BufRef,
    pub bucket_map: &'a BufRef,
    pub uniform: &'a BufRef,
    pub out: &'a BufRef,
}

/// `n` is the output element count for f32 acts, or the output WORD count
/// (`H*S*S / 2`) for packed (bf16/f16) acts. Caller sizes it to match the
/// chosen act dtype.
pub(crate) fn dispatch_relpos_bias<O: RelposBiasOp, B: Backend>(
    backend: &B,
    encoder: &mut B::CommandEncoder,
    pipeline: &B::Pipeline,
    bufs: &RelposBiasBufs<'_>,
    n: u32,
) -> Result<(), B::Error> {
    let bindings = [
        bufs.table.binding(0),
        bufs.bucket_map.binding(1),
        bufs.out.binding(2),
        bufs.uniform.binding(3),
    ];
    backend.dispatch(encoder, pipeline, &bindings, O::workgroups(n))
}

// f32 / bf16-writes share one body (both store through `act_store`), one
// output element per thread.
macro_rules! f32_body {
    () => {
        r#"
struct U { h: u32, s: u32, _p0: u32, _p1: u32 };

@group(0) @binding(0) var<storage, read> table: array<f32>;
@group(0) @binding(1) var<storage, read> bucket_map: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let i = gid.y * (ng.x * 64u) + gid.x;
    let plane = u.s * u.s;
    let total = u.h * plane;
    if (i >= total) { return; }
    let hh = i / plane;
    let ij = i - hh * plane;          // bucket_map is [S,S] row-major: flat == i*S+j
    let bucket = bucket_map[ij];
    out[i] = act_store(table[bucket * u.h + hh]);
}
"#
    };
}

const WGSL_F32: &str = concat!(act_store_f32!(), f32_body!());
const WGSL_F32_BF16: &str = concat!(act_store_bf16q!(), f32_body!());

// Packed paths: two output elements (keys j, j+1) per word. Requires S even
// (the same constraint SDPA's packed mask read imposes when has_mask != 0).
const WGSL_BF16_PACKED: &str = concat!(
    act_bf16_prelude!(),
    r#"
struct U { h: u32, s: u32, _p0: u32, _p1: u32 };

@group(0) @binding(0) var<storage, read> table: array<f32>;
@group(0) @binding(1) var<storage, read> bucket_map: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;
@group(0) @binding(3) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let w = gid.y * (ng.x * 64u) + gid.x;
    let words_per_qrow = u.s >> 1u;
    let total_words = u.h * u.s * words_per_qrow;
    if (w >= total_words) { return; }
    let row = w / words_per_qrow;    // hh*S + qi
    let kw  = w - row * words_per_qrow;
    let hh  = row / u.s;
    let qi  = row - hh * u.s;
    let bm  = qi * u.s + kw * 2u;
    let v0  = table[bucket_map[bm] * u.h + hh];
    let v1  = table[bucket_map[bm + 1u] * u.h + hh];
    out[w] = pack_bf16x2(v0, v1);
}
"#
);

const WGSL_F16_PACKED: &str = concat!(
    act_f16_prelude!(),
    r#"
struct U { h: u32, s: u32, _p0: u32, _p1: u32 };

@group(0) @binding(0) var<storage, read> table: array<f32>;
@group(0) @binding(1) var<storage, read> bucket_map: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<vec2<f16>>;
@group(0) @binding(3) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let w = gid.y * (ng.x * 64u) + gid.x;
    let words_per_qrow = u.s >> 1u;
    let total_words = u.h * u.s * words_per_qrow;
    if (w >= total_words) { return; }
    let row = w / words_per_qrow;    // hh*S + qi
    let kw  = w - row * words_per_qrow;
    let hh  = row / u.s;
    let qi  = row - hh * u.s;
    let bm  = qi * u.s + kw * 2u;
    let v0  = table[bucket_map[bm] * u.h + hh];
    let v1  = table[bucket_map[bm + 1u] * u.h + hh];
    out[w] = pack_f32_to_f16x2(v0, v1);
}
"#
);

const LAYOUT: &[BindingLayout] = &[
    BindingLayout {
        slot: 0,
        kind: BindingKind::StorageRead,
    },
    BindingLayout {
        slot: 1,
        kind: BindingKind::StorageRead,
    },
    BindingLayout {
        slot: 2,
        kind: BindingKind::StorageReadWrite,
    },
    BindingLayout {
        slot: 3,
        kind: BindingKind::Uniform,
    },
];

pub struct RelposBiasF32;

impl RelposBiasOp for RelposBiasF32 {
    const KERNEL_ID: &'static str = "relpos_bias.f32";
    type Dtype = F32;
    const TABLE: &'static str = "relpos_bias/table";
    const BUCKET_MAP: &'static str = "relpos_bias/bucket_map";
    const DIMS: &'static str = "relpos_bias/dims";
    const OUTPUT: &'static str = "relpos_bias/out";
    fn wgsl(cfg: &WgslConfig) -> &'static str {
        match (cfg.act_dtype, cfg.bf16_quant_writes) {
            (ActDtype::F32, false) => WGSL_F32,
            (ActDtype::F32, true) => WGSL_F32_BF16,
            (ActDtype::Bf16, _) => WGSL_BF16_PACKED,
            (ActDtype::F16, _) => WGSL_F16_PACKED,
            (ActDtype::I8, _) => unreachable!("ActDtype::I8 is never a block-level act dtype"),
        }
    }
    fn layout() -> &'static [BindingLayout] {
        LAYOUT
    }
}

// ---------------------------------------------------------------------------
// CPU bucket-map builder (HF T5 `_relative_position_bucket`)
// ---------------------------------------------------------------------------

/// HF T5/umT5 `_relative_position_bucket`. Maps a signed relative position
/// (`key - query`) to a bucket in `[0, num_buckets)`. Half the buckets cover
/// negative offsets, half positive (bidirectional); within each side, the
/// first `num_buckets/4` are exact and the rest are log-spaced out to
/// `max_distance`. Matches `transformers` PyTorch verbatim (trunc-toward-zero
/// on the log term, as `.to(torch.long)` does for the positive value).
pub fn relative_position_bucket(
    relative_position: i32,
    bidirectional: bool,
    num_buckets: u32,
    max_distance: u32,
) -> u32 {
    let mut ret = 0u32;
    let mut n = num_buckets;
    let rp_abs = if bidirectional {
        n /= 2;
        if relative_position > 0 {
            ret += n;
        }
        relative_position.unsigned_abs()
    } else {
        // unidirectional: only non-positive offsets, folded to >= 0.
        (-relative_position).max(0) as u32
    };
    let max_exact = n / 2;
    let bucket = if rp_abs < max_exact {
        rp_abs
    } else {
        let large = max_exact as f32
            + ((rp_abs as f32 / max_exact as f32).ln()
                / (max_distance as f32 / max_exact as f32).ln()
                * (n - max_exact) as f32);
        (large as u32).min(n - 1)
    };
    ret + bucket
}

/// Build the `[S, S]` row-major bucket map for [`RelposBiasF32`].
/// `bucket_map[i*S + j] = relative_position_bucket(j - i, ...)`.
pub fn relpos_bucket_map(
    s: usize,
    bidirectional: bool,
    num_buckets: u32,
    max_distance: u32,
) -> Vec<u32> {
    let mut m = vec![0u32; s * s];
    for i in 0..s {
        for j in 0..s {
            let rel = j as i32 - i as i32;
            m[i * s + j] = relative_position_bucket(rel, bidirectional, num_buckets, max_distance);
        }
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    // Spot-check against HF transformers reference values (bidirectional,
    // num_buckets=32, max_distance=128). n=16 per side, max_exact=8.
    #[test]
    fn bucket_matches_hf() {
        let f = |rp| relative_position_bucket(rp, true, 32, 128);
        assert_eq!(f(0), 0); // exact, non-positive side
        assert_eq!(f(-1), 1);
        assert_eq!(f(-7), 7);
        assert_eq!(f(1), 17); // positive side: 16 + abs (bucket 16 == rp 0, never positive)
        assert_eq!(f(7), 23);
        // Log-spaced region (abs >= max_exact=8): abs=8 -> log term 0 -> bucket 8.
        assert_eq!(f(-8), 8);
        assert_eq!(f(8), 24); // 16 + 8
        assert_eq!(f(-1000), 15); // saturates negative side top bucket
        assert_eq!(f(1000), 31); // saturates positive side top bucket
    }

    #[test]
    fn bucket_map_is_square_rowmajor() {
        let s = 4;
        let m = relpos_bucket_map(s, true, 32, 128);
        assert_eq!(m.len(), s * s);
        // Diagonal (rel=0) is bucket 0.
        for i in 0..s {
            assert_eq!(m[i * s + i], 0);
        }
        // m[i][j] for j>i is on the positive side (>= 16); j<i negative (< 16).
        assert!(m[1] >= 16); // row 0, col 1
        assert!(m[s] < 16); // row 1, col 0
    }
}
