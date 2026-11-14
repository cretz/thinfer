//! Window attention over stacked key planes (comfy-kitchen `window_attention`,
//! the RelayVSR attention): query planes are the last `Tq` of the `Tk` key
//! planes; query plane `t` attends key planes `last - tw + 1 ..= last`
//! (`last = t + Tk - Tq`, `tw` the temporal window, clipped at plane 0). In key
//! plane `p` a query at `(y, x)` attends the `windows[p] = (wh, ww)` keys
//! centered on it: `lo = pos - win / 2`; with `clamp` the window first shrinks
//! to the frame and `lo` is shifted inside it (NATTEN), without it keys outside
//! the frame are cut off. Grouped-query: `kv_heads | num_heads`.
//!
//! Shapes (row-major, heads innermost like the dense SDPA's BSHD):
//! - Q   `[B, Tq, H, W, nh, hd]`, Out the same.
//! - K/V `[B, Tk, H, W, kvh, hd]`.
//!
//! Kernel: one 64-thread workgroup per (row segment of 64 queries, head,
//! batch); each thread owns one query (`hd <= 128` f32 registers, online
//! softmax in f32). Every query of a row shares its row window, so the
//! workgroup streams the UNION of its column windows through shared K/V tiles
//! of 32 keys, row by row, plane by plane, and each thread masks to its own
//! columns.
//!
//! Layout: 0 = Q, 1 = K, 2 = V, 3 = Out, 4 = uniform [`WindowAttnUniform`].
//! Grid: [`workgroups`].

use crate::backend::{BindingKind, BindingLayout};
use crate::ops::ActDtype;

/// Most key planes one call can carry per-plane windows for.
pub const MAX_PLANES: usize = 8;

/// Uniform block (std140: the window table is 16-byte strided).
#[derive(Clone, Copy, Debug)]
pub struct WindowAttnUniform {
    pub b: u32,
    pub tq: u32,
    pub tk: u32,
    pub h: u32,
    pub w: u32,
    pub nh: u32,
    pub kvh: u32,
    pub hd: u32,
    /// Key planes each query plane attends, counting its own.
    pub temporal_window: u32,
    pub clamp: bool,
    pub scale: f32,
    /// `(wh, ww)` per key plane; planes past the table reuse its last entry
    /// (one window for every plane: fill all with [`Self::same_window`]).
    pub windows: [(u32, u32); MAX_PLANES],
}

impl WindowAttnUniform {
    pub fn same_window(wh: u32, ww: u32) -> [(u32, u32); MAX_PLANES] {
        [(wh, ww); MAX_PLANES]
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let segs = self.w.div_ceil(64);
        let head = [
            self.b,
            self.tq,
            self.tk,
            self.h,
            self.w,
            self.nh,
            self.kvh,
            self.hd,
            self.temporal_window,
            self.clamp as u32,
            self.scale.to_bits(),
            segs,
        ];
        let mut out: Vec<u8> = head.iter().flat_map(|v| v.to_le_bytes()).collect();
        for (wh, ww) in self.windows {
            for v in [wh, ww, 0, 0] {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        out
    }
}

macro_rules! window_attn_wgsl {
    ($prelude:expr, $word:expr, $cv:expr, $pk:expr) => {
        concat!(
            $prelude,
            r#"
struct U {
    b: u32, tq: u32, tk: u32, h: u32,
    w: u32, nh: u32, kvh: u32, hd: u32,
    tw: u32, clamp: u32, scale: f32, segs: u32,
    win: array<vec4<u32>, 8>,
};
"#,
            "@group(0) @binding(0) var<storage, read> q: array<", $word, ">;\n",
            "@group(0) @binding(1) var<storage, read> k: array<", $word, ">;\n",
            "@group(0) @binding(2) var<storage, read> v: array<", $word, ">;\n",
            "@group(0) @binding(3) var<storage, read_write> out: array<", $word, ">;\n",
            "@group(0) @binding(4) var<uniform> u: U;\n",
            "var<workgroup> k_tile: array<", $word, ", 2048>;\n",
            "var<workgroup> v_tile: array<", $word, ", 2048>;\n",
            $cv,
            $pk,
            r#"
const BC: u32 = 32u;

// First attended index along an axis of `size` for a query at `pos`, and the
// (possibly clamped) window length. Without clamp the range may poke outside
// [0, size); callers intersect.
fn win_lo(pos: u32, size: u32, win: u32, shift_in: bool) -> i32 {
    var lo = i32(pos) - i32(win / 2u);
    if (shift_in) {
        lo = clamp(lo, 0, i32(size) - i32(win));
    }
    return lo;
}

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let t_l = lid.x;
    let seg = wid.x % u.segs;
    let y = (wid.x / u.segs) % u.h;
    let tq_i = wid.x / (u.segs * u.h);
    let hq = wid.y;
    let bb = wid.z;
    let hkv = hq / (u.nh / u.kvh);
    let clamp_on = u.clamp != 0u;

    let x0 = seg * 64u;
    let x = x0 + t_l;
    let valid = x < u.w;
    let x_last = min(x0 + 63u, u.w - 1u);

    let d_w = u.hd >> 1u;
    let q_tok = ((bb * u.tq + tq_i) * u.h + y) * u.w + x;
    let q_w_off = ((q_tok * u.nh + hq) * u.hd) >> 1u;
    let kv_tok_w = (u.kvh * u.hd) >> 1u;

    var q_local: array<f32, 128>;
    var o_local: array<f32, 128>;
    for (var d = 0u; d < 128u; d = d + 1u) { q_local[d] = 0.0; o_local[d] = 0.0; }
    if (valid) {
        for (var dw = 0u; dw < d_w; dw = dw + 1u) {
            let qv = cv(q[q_w_off + dw]);
            q_local[dw * 2u] = qv.x;
            q_local[dw * 2u + 1u] = qv.y;
        }
    }
    var m: f32 = -3.4e38;
    var l: f32 = 0.0;

    let last = tq_i + u.tk - u.tq;
    let first = u32(max(i32(last) - i32(u.tw) + 1, 0));
    for (var p = first; p <= last; p = p + 1u) {
        let wp = u.win[min(p, 7u)];
        var wh = wp.x;
        var ww = wp.y;
        if (clamp_on) {
            wh = min(wh, u.h);
            ww = min(ww, u.w);
        }
        let ylo = win_lo(y, u.h, wh, clamp_on);
        let y_start = u32(max(ylo, 0));
        let y_end = u32(min(ylo + i32(wh), i32(u.h)));
        // This thread's columns, and the workgroup's union (lo is monotone in x).
        let xlo = win_lo(x, u.w, ww, clamp_on);
        let my_c0 = max(xlo, 0);
        let my_c1 = min(xlo + i32(ww), i32(u.w));
        let c_start = u32(max(win_lo(x0, u.w, ww, clamp_on), 0));
        let c_end = u32(min(win_lo(x_last, u.w, ww, clamp_on) + i32(ww), i32(u.w)));
        let plane_tok0 = (bb * u.tk + p) * u.h;

        for (var ky = y_start; ky < y_end; ky = ky + 1u) {
            let row_tok0 = (plane_tok0 + ky) * u.w;
            for (var c0 = c_start; c0 < c_end; c0 = c0 + BC) {
                let n_keys = min(BC, c_end - c0);
                let words = BC * d_w;
                for (var i = t_l; i < words; i = i + 64u) {
                    let kc = i / d_w;
                    let dw = i % d_w;
                    if (kc < n_keys) {
                        let src = (row_tok0 + c0 + kc) * kv_tok_w + ((hkv * u.hd) >> 1u) + dw;
                        k_tile[i] = k[src];
                        v_tile[i] = v[src];
                    }
                }
                workgroupBarrier();
                if (valid) {
                    for (var kc = 0u; kc < n_keys; kc = kc + 1u) {
                        let col = i32(c0 + kc);
                        if (col >= my_c0 && col < my_c1) {
                            var dot: f32 = 0.0;
                            for (var dw = 0u; dw < d_w; dw = dw + 1u) {
                                let kv = cv(k_tile[kc * d_w + dw]);
                                dot = dot + q_local[dw * 2u] * kv.x + q_local[dw * 2u + 1u] * kv.y;
                            }
                            let s_j = dot * u.scale;
                            let m_new = max(m, s_j);
                            let alpha = exp(m - m_new);
                            let p_j = exp(s_j - m_new);
                            for (var dw = 0u; dw < d_w; dw = dw + 1u) {
                                let vv = cv(v_tile[kc * d_w + dw]);
                                o_local[dw * 2u] = o_local[dw * 2u] * alpha + p_j * vv.x;
                                o_local[dw * 2u + 1u] = o_local[dw * 2u + 1u] * alpha + p_j * vv.y;
                            }
                            l = l * alpha + p_j;
                            m = m_new;
                        }
                    }
                }
                workgroupBarrier();
            }
        }
    }

    if (valid) {
        let inv_l = 1.0 / l;
        for (var dw = 0u; dw < d_w; dw = dw + 1u) {
            out[q_w_off + dw] = pk(vec2<f32>(o_local[dw * 2u], o_local[dw * 2u + 1u]) * inv_l);
        }
    }
}
"#
        )
    };
}

const WGSL_F32: &str = window_attn_wgsl!(
    "",
    "vec2<f32>",
    "fn cv(w: vec2<f32>) -> vec2<f32> { return w; }\n",
    "fn pk(v: vec2<f32>) -> vec2<f32> { return v; }\n"
);

const WGSL_F16: &str = window_attn_wgsl!(
    "enable f16;\n",
    "vec2<f16>",
    "fn cv(w: vec2<f16>) -> vec2<f32> { return vec2<f32>(w); }\n",
    "fn pk(v: vec2<f32>) -> vec2<f16> { return vec2<f16>(clamp(v, vec2<f32>(-65504.0), vec2<f32>(65504.0))); }\n"
);

const WGSL_BF16: &str = window_attn_wgsl!(
    crate::act_bf16_prelude!(),
    "u32",
    "fn cv(w: u32) -> vec2<f32> { return unpack_bf16x2(w); }\n",
    "fn pk(v: vec2<f32>) -> u32 { return pack_bf16x2(v.x, v.y); }\n"
);

pub fn wgsl(act: ActDtype) -> &'static str {
    match act {
        ActDtype::F32 => WGSL_F32,
        ActDtype::F16 => WGSL_F16,
        ActDtype::Bf16 => WGSL_BF16,
        ActDtype::I8 => unreachable!("window attention runs on dense acts"),
    }
}

pub fn layout() -> &'static [BindingLayout] {
    const L: &[BindingLayout] = &[
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
            kind: BindingKind::StorageRead,
        },
        BindingLayout {
            slot: 3,
            kind: BindingKind::StorageReadWrite,
        },
        BindingLayout {
            slot: 4,
            kind: BindingKind::Uniform,
        },
    ];
    L
}

pub fn workgroups(u: &WindowAttnUniform) -> [u32; 3] {
    [u.w.div_ceil(64) * u.h * u.tq, u.nh, u.b]
}

/// Host reference (f32), the comfy-kitchen eager semantics. Tests + bring-up.
pub fn window_attn_host(q: &[f32], k: &[f32], v: &[f32], u: &WindowAttnUniform) -> Vec<f32> {
    let (h, w, nh, kvh, hd) = (
        u.h as i64,
        u.w as i64,
        u.nh as usize,
        u.kvh as usize,
        u.hd as usize,
    );
    let mut out = vec![0.0f32; q.len()];
    let lo = |pos: i64, size: i64, win: i64| -> i64 {
        let l = pos - win / 2;
        if u.clamp { l.clamp(0, size - win) } else { l }
    };
    for b in 0..u.b as usize {
        for t in 0..u.tq as usize {
            let last = t + (u.tk - u.tq) as usize;
            let first = (last as i64 - u.temporal_window as i64 + 1).max(0) as usize;
            for y in 0..h {
                for x in 0..w {
                    for hq in 0..nh {
                        let hk = hq / (nh / kvh);
                        let qo = ((((b * u.tq as usize + t) * h as usize + y as usize)
                            * w as usize
                            + x as usize)
                            * nh
                            + hq)
                            * hd;
                        let qv = &q[qo..qo + hd];
                        let mut scores = Vec::new();
                        let mut vals = Vec::new();
                        for p in first..=last {
                            let wp = u.windows[p.min(MAX_PLANES - 1)];
                            let (mut wh, mut ww) = (wp.0 as i64, wp.1 as i64);
                            if u.clamp {
                                wh = wh.min(h);
                                ww = ww.min(w);
                            }
                            let (y0, x0) = (lo(y, h, wh), lo(x, w, ww));
                            for ky in y0.max(0)..(y0 + wh).min(h) {
                                for kx in x0.max(0)..(x0 + ww).min(w) {
                                    let ko = ((((b * u.tk as usize + p) * h as usize
                                        + ky as usize)
                                        * w as usize
                                        + kx as usize)
                                        * kvh
                                        + hk)
                                        * hd;
                                    let s: f32 =
                                        qv.iter().zip(&k[ko..ko + hd]).map(|(a, b)| a * b).sum();
                                    scores.push(s * u.scale);
                                    vals.push(ko);
                                }
                            }
                        }
                        let mx = scores.iter().cloned().fold(f32::MIN, f32::max);
                        let e: Vec<f32> = scores.iter().map(|s| (s - mx).exp()).collect();
                        let z: f32 = e.iter().sum();
                        for (ei, &vo) in e.iter().zip(&vals) {
                            for d in 0..hd {
                                out[qo + d] += ei / z * v[vo + d];
                            }
                        }
                    }
                }
            }
        }
    }
    out
}
