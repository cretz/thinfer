//! Flash-attention small-D sdpa on packed-int8 activations with
//! per-(rows, D/32) f32 scale. Output O scale is `[B, S_q, H_q, D/32]`,
//! flattening to `[M=B*S_q, K/32]` where `K = H_q*D` - which is exactly the
//! per-(M, K/32) layout that `matmul_i8` consumes for o_proj. No transcode.
//!
//! Tile geometry: BR=64 Q rows per WG, BC=64 K rows per K/V tile, WG=64.
//! I8 byte density (1 B/elem vs F16's 2) lets BC double while keeping each
//! tile at 8 KiB (16 KiB total per WG) - matches the F16/Bf16 BC=32 ceiling.
//!
//! Binding fusion: Q/K/V/O each occupy one storage binding holding
//! `[data: i8-packed u32][params: (scale_f16, zero_f16) packed u32]`
//! contiguously in one GPU buffer. The kernel decodes the params segment
//! via `unpack2x16float(buf[scale_off + i])` (each word -> (s, z) as
//! vec2<f32>). Fewer bindings (5 storage + 1 uniform) keeps us under the
//! WebGPU `max_storage_buffers_per_shader_stage = 8` floor. Word-offsets
//! are derived from B/S/H/D so the uniform layout is unchanged. Same total
//! 4 bytes per (row, d_chunk) as the old f32-scale-only layout.
//!
//! Param layouts (within each fused buffer, after the data segment):
//! - Q params: `[B, S_q, H_q, D/32]` u32(=(scale_f16, zero_f16)).
//! - K params: `[B, S_k, H_kv, D/32]` u32.
//! - V params: `[B, S_k, H_kv, D/32]` u32.
//! - O params: `[B, S_q, H_q, D/32]` u32. Computed per d-chunk from
//!   `(min, max)` over the normalized fp32 O (each thread owns one full Q
//!   row in registers; the reduce over each 32-elem D-chunk is per-thread,
//!   no shared reduction).
//!
//! Mask binding is `vec2<f16>`. I8 acts require SHADER_F16 anyway; matching
//! F16_PACKED's mask dtype avoids carrying a parallel fp32 mask.
//!
//! Workgroup grid: `[ceil(rows/64), H_q, B]` where `rows` is this dispatch's
//! Q-row count (`u.row0 + rows <= u.s_q`); each thread owns one Q row in
//! registers. Chunking the query range across several dispatches is bit-exact
//! (each row is independent and all offsets derive from the GLOBAL row index
//! `u.row0 + local`), which keeps long-clip dispatches under the ~2s Windows
//! GPU watchdog. All buffers stay bound whole.
//!
//! Output form (`u.out_mode`):
//! - 0: paired i8 + per-(row, D/32) params, as described above.
//! - 1: dense packed f16 (`vec2<f16>` per u32) at the f16 act layout.
//! - 2: dense packed bf16 (2 x u16 per u32) at the bf16 act layout.
//!
//! The dense modes skip the output quantize entirely: consumers whose next
//! matmul is NOT a paired-A site (Wan's Q8 proj) read the normalized O
//! directly, with no dequant round-trip.

use crate::backend::{Backend, BindingKind, BindingLayout, BufRef};
use crate::{act_bf16_prelude, act_i8_prelude};

const LAYOUT: &[BindingLayout] = &[
    BindingLayout {
        slot: 0,
        kind: BindingKind::StorageRead,
    }, // q (fused: data || scale)
    BindingLayout {
        slot: 1,
        kind: BindingKind::StorageRead,
    }, // k (fused)
    BindingLayout {
        slot: 2,
        kind: BindingKind::StorageRead,
    }, // v (fused)
    BindingLayout {
        slot: 3,
        kind: BindingKind::StorageRead,
    }, // mask (vec2<f16>)
    BindingLayout {
        slot: 4,
        kind: BindingKind::StorageReadWrite,
    }, // out (fused: data || scale)
    BindingLayout {
        slot: 5,
        kind: BindingKind::Uniform,
    },
];

pub fn layout() -> &'static [BindingLayout] {
    LAYOUT
}
pub fn kernel_id() -> &'static str {
    "sdpa.i8"
}
pub fn hint() -> &'static str {
    "ai8-perblock-flashattn-br64-bc64-fused"
}

pub fn build_wgsl() -> String {
    let mut s = String::new();
    s.push_str("enable f16;\n");
    s.push_str(act_i8_prelude!());
    s.push_str(act_bf16_prelude!());
    s.push_str(BODY);
    s
}

const BODY: &str = r#"
struct U {
    b: u32, h_q: u32, h_kv: u32, s_q: u32,
    s_k: u32, d: u32, scale: f32, has_mask: u32,
    row0: u32, out_mode: u32, pad0: u32, pad1: u32,
};

@group(0) @binding(0) var<storage, read>       q:    array<u32>;
@group(0) @binding(1) var<storage, read>       k:    array<u32>;
@group(0) @binding(2) var<storage, read>       v:    array<u32>;
@group(0) @binding(3) var<storage, read>       mask: array<vec2<f16>>;
@group(0) @binding(4) var<storage, read_write> o:    array<u32>;
@group(0) @binding(5) var<uniform>             u: U;

const BR: u32 = 64u;
const BC: u32 = 64u;
const WG: u32 = 64u;
const BK: u32 = 32u;

// Tile: BC=64 K rows * D/4=32 u32 words = 2048 u32 = 8 KiB per side.
var<workgroup> k_tile: array<u32, 2048>;
var<workgroup> v_tile: array<u32, 2048>;
// Per-(kc, d_chunk) params (scale, zero): BC * (D/32) = 64 * 4 = 256 entries
// at D=128. Each entry is (s, z) as vec2<f32> (decoded from the f16-packed
// u32 word at scale-load time).
var<workgroup> k_tile_params: array<vec2<f32>, 256>;
var<workgroup> v_tile_params: array<vec2<f32>, 256>;

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let t  = lid.x;
    let qt = wgid.x;
    let hq = wgid.y;
    let bb = wgid.z;
    let sq = u.row0 + qt * BR + t;
    let valid = sq < u.s_q;

    let hkv = (hq * u.h_kv) / u.h_q;
    let d_w           = u.d >> 2u;          // D / 4 (u32 words per row)
    let chunks_per_row = u.d >> 5u;          // D / 32
    // Scale segments live immediately after each role's data segment in the
    // fused buffer. Word-offsets (4 B per u32 / 4 B per f32):
    //   Q/O: B * S_q * H_q  * D / 4 words of data
    //   K/V: B * S_k * H_kv * D / 4 words of data
    let q_scale_off = (u.b * u.s_q * u.h_q  * u.d) >> 2u;
    let kv_scale_off = (u.b * u.s_k * u.h_kv * u.d) >> 2u;
    let o_scale_off = q_scale_off;
    let q_w_off       = (((bb * u.s_q + sq) * u.h_q + hq) * u.d) >> 2u;
    let kv_b0_w       = ((bb * u.s_k * u.h_kv + hkv) * u.d) >> 2u;
    let kv_step_w     = (u.h_kv * u.d) >> 2u;
    let mask_w_base   = ((bb * u.s_q + sq) * u.s_k) >> 1u;
    let q_scale_row   = ((bb * u.s_q + sq) * u.h_q + hq) * chunks_per_row;

    // Load this thread's Q row (dequant with per-chunk (scale, zero) params).
    var q_local: array<f32, 128>;
    for (var d = 0u; d < u.d; d = d + 1u) { q_local[d] = 0.0; }
    if (valid) {
        for (var dc = 0u; dc < chunks_per_row; dc = dc + 1u) {
            let q_p = unpack2x16float(q[q_scale_off + q_scale_row + dc]);
            let cw_base = dc * 8u;
            for (var w = 0u; w < 8u; w = w + 1u) {
                let qv: vec4<f32> = unpack_i8x4_aff_raw(q[q_w_off + cw_base + w], q_p.x, q_p.y);
                let qe = dc * BK + w * 4u;
                q_local[qe + 0u] = qv.x;
                q_local[qe + 1u] = qv.y;
                q_local[qe + 2u] = qv.z;
                q_local[qe + 3u] = qv.w;
            }
        }
    }

    var o_local: array<f32, 128>;
    for (var d = 0u; d < u.d; d = d + 1u) { o_local[d] = 0.0; }
    // Large-negative finite f32, not -inf: Tint (browser WebGPU) rejects
    // const-exprs evaluating to -inf (bitcast<f32>(0xff800000u)) and ALSO
    // out-of-range decimals like -3.4028235e38 (Rust's f32::MIN print is
    // a hair beyond max-finite as an exact decimal); naga accepts both.
    // Equivalent as a running-max init.
    var m: f32 = -3.4e38;
    var l: f32 = 0.0;

    let n_tiles = (u.s_k + BC - 1u) / BC;
    let words_per_tile = BC * d_w;
    let words_per_thread = words_per_tile / WG;
    let scales_per_tile = BC * chunks_per_row;
    let scales_per_thread = scales_per_tile / WG;

    for (var kt = 0u; kt < n_tiles; kt = kt + 1u) {
        let kc_base = kt * BC;

        // Cooperative K/V data load.
        for (var i = 0u; i < words_per_thread; i = i + 1u) {
            let idx = i * WG + t;
            let kc  = idx / d_w;
            let dw  = idx % d_w;
            let key_global = kc_base + kc;
            var kw: u32 = 0u;
            var vw: u32 = 0u;
            if (key_global < u.s_k) {
                let base = kv_b0_w + key_global * kv_step_w + dw;
                kw = k[base];
                vw = v[base];
            }
            k_tile[idx] = kw;
            v_tile[idx] = vw;
        }
        // Cooperative K/V params load. Each scale-word packs (scale_f16,
        // zero_f16); decode via unpack2x16float once per (kc, dc) slot.
        for (var i = 0u; i < scales_per_thread; i = i + 1u) {
            let idx = i * WG + t;
            let kc  = idx / chunks_per_row;
            let dc  = idx % chunks_per_row;
            let key_global = kc_base + kc;
            var kp: vec2<f32> = vec2<f32>(0.0, 0.0);
            var vp: vec2<f32> = vec2<f32>(0.0, 0.0);
            if (key_global < u.s_k) {
                let so = ((bb * u.s_k + key_global) * u.h_kv + hkv) * chunks_per_row + dc;
                kp = unpack2x16float(k[kv_scale_off + so]);
                vp = unpack2x16float(v[kv_scale_off + so]);
            }
            k_tile_params[idx] = kp;
            v_tile_params[idx] = vp;
        }
        workgroupBarrier();

        if (valid) {
            for (var kc = 0u; kc < BC; kc = kc + 1u) {
                let key_global = kc_base + kc;
                if (key_global < u.s_k) {
                    let kc_words = kc * d_w;
                    let kc_scales = kc * chunks_per_row;

                    var dot: f32 = 0.0;
                    for (var dc = 0u; dc < chunks_per_row; dc = dc + 1u) {
                        let kp = k_tile_params[kc_scales + dc];
                        let cw_base = kc_words + dc * 8u;
                        let qe_base = dc * BK;
                        for (var w = 0u; w < 8u; w = w + 1u) {
                            let kv: vec4<f32> = unpack_i8x4_aff_raw(k_tile[cw_base + w], kp.x, kp.y);
                            let qe = qe_base + w * 4u;
                            dot = dot
                                + q_local[qe + 0u] * kv.x
                                + q_local[qe + 1u] * kv.y
                                + q_local[qe + 2u] * kv.z
                                + q_local[qe + 3u] * kv.w;
                        }
                    }
                    var bias: f32 = 0.0;
                    if (u.has_mask != 0u) {
                        let mw: vec2<f32> = vec2<f32>(mask[mask_w_base + (key_global >> 1u)]);
                        bias = select(mw.x, mw.y, (key_global & 1u) == 1u);
                    }
                    let s_j  = dot * u.scale + bias;
                    let m_new = max(m, s_j);
                    let alpha = exp(m - m_new);
                    let p_j   = exp(s_j - m_new);
                    for (var dc = 0u; dc < chunks_per_row; dc = dc + 1u) {
                        let vp = v_tile_params[kc_scales + dc];
                        let cw_base = kc_words + dc * 8u;
                        let qe_base = dc * BK;
                        for (var w = 0u; w < 8u; w = w + 1u) {
                            let vv: vec4<f32> = unpack_i8x4_aff_raw(v_tile[cw_base + w], vp.x, vp.y);
                            let qe = qe_base + w * 4u;
                            o_local[qe + 0u] = o_local[qe + 0u] * alpha + p_j * vv.x;
                            o_local[qe + 1u] = o_local[qe + 1u] * alpha + p_j * vv.y;
                            o_local[qe + 2u] = o_local[qe + 2u] * alpha + p_j * vv.z;
                            o_local[qe + 3u] = o_local[qe + 3u] * alpha + p_j * vv.w;
                        }
                    }
                    l = l * alpha + p_j;
                    m = m_new;
                }
            }
        }
        workgroupBarrier();
    }

    if (valid && u.out_mode != 0u) {
        // Dense output: normalized O packed 2 elems/u32 at the act layout
        // (mode 1 = f16, mode 2 = bf16). Element base == the paired data
        // layout's element index; dense packs 2 per word instead of 4.
        let inv_l = 1.0 / l;
        let ow = (((bb * u.s_q + sq) * u.h_q + hq) * u.d) >> 1u;
        for (var w = 0u; w < (u.d >> 1u); w = w + 1u) {
            let lo = o_local[2u * w] * inv_l;
            let hi = o_local[2u * w + 1u] * inv_l;
            if (u.out_mode == 1u) {
                o[ow + w] = pack2x16float(vec2<f32>(lo, hi));
            } else {
                o[ow + w] = pack_bf16x2(lo, hi);
            }
        }
        return;
    }
    if (valid) {
        let inv_l = 1.0 / l;
        let out_scale_row = q_scale_row;     // same shape as q scale
        let out_w_base    = q_w_off;
        for (var dc = 0u; dc < chunks_per_row; dc = dc + 1u) {
            let qe_base = dc * BK;
            var mn: f32 = 3.4e38;
            var mx: f32 = -3.4e38;
            for (var k_i = 0u; k_i < BK; k_i = k_i + 1u) {
                let v = o_local[qe_base + k_i] * inv_l;
                o_local[qe_base + k_i] = v;
                mn = min(mn, v);
                mx = max(mx, v);
            }
            let range = mx - mn;
            let s = select(range / 254.0, 1.0e-30, range <= 0.0);
            let z = mn + 127.0 * s;
            o[o_scale_off + out_scale_row + dc] = pack2x16float(vec2<f32>(s, z));
            let ow_base = out_w_base + dc * 8u;
            for (var w = 0u; w < 8u; w = w + 1u) {
                let qe = qe_base + w * 4u;
                let v4 = vec4<f32>(
                    o_local[qe + 0u],
                    o_local[qe + 1u],
                    o_local[qe + 2u],
                    o_local[qe + 3u],
                );
                o[ow_base + w] = pack_f32x4_aff_to_i8_raw(v4, s, z);
            }
        }
    }
}
"#;

/// One fused `(data || scale)` storage buffer per role plus mask + uniform.
/// Build each fused `BufRef` via `BufRef::view` covering the full data and
/// scale span of a single underlying GPU buffer. Producers (act_quant /
/// rope_i8 / qkv_split_i8 / sdpa output consumer) keep using their own
/// data-only and scale-only views into that same buffer.
pub struct SdpaI8Bufs<'a> {
    pub q: &'a BufRef,
    pub k: &'a BufRef,
    pub v: &'a BufRef,
    pub mask: &'a BufRef,
    pub out: &'a BufRef,
    pub uniform: &'a BufRef,
}

/// `rows` is this dispatch's Q-row count (the chunk size, == `s_q` for a
/// whole-tensor call); the uniform's `row0` carries the chunk's global row
/// offset. All buffers stay bound whole.
pub fn dispatch_sdpa_i8<B: Backend>(
    backend: &B,
    encoder: &mut B::CommandEncoder,
    pipeline: &B::Pipeline,
    bufs: &SdpaI8Bufs<'_>,
    b: u32,
    rows: u32,
    h_q: u32,
    d: u32,
) -> Result<(), B::Error> {
    assert!(d <= 128, "sdpa_i8: D={d} exceeds MAX_D=128");
    assert!(
        d.is_multiple_of(32),
        "sdpa_i8: D={d} must be multiple of 32"
    );
    let bindings = [
        bufs.q.binding(0),
        bufs.k.binding(1),
        bufs.v.binding(2),
        bufs.mask.binding(3),
        bufs.out.binding(4),
        bufs.uniform.binding(5),
    ];
    backend.dispatch(encoder, pipeline, &bindings, [rows.div_ceil(64), h_q, b])
}
