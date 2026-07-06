//! WGSL kernels for the ONNX executor.
//!
//! Self-contained (not the `ops/` trait kernels, which are tuned for the
//! hand-coded LLM/diffusion models): all f32 storage, straightforward index
//! math, one kernel per ONNX op family. Binding convention per kernel:
//! storage-read inputs at 0.., the read_write output next, the uniform last.
//!
//! Perf note: convs are direct (one thread per output element), not the tuned
//! implicit-GEMM in `ops/conv2d.rs`. The face-swap models are small (256x256,
//! mostly low channel counts at full spatial), so this is fast enough for v1;
//! a tiled path is a later optimization if a trace shows conv dominating.

// Every elementwise kernel below recovers a linear element index from a 2D
// workgroup grid (WebGPU caps each grid dim at 65535) via
// `let i = gid.y * (ng.x * 64u) + gid.x;`, paired with `linear_workgroups(n, 64)`.

/// Direct NCHW conv2d with groups, dilation, stride, pad, bias.
/// Bindings: 0=x, 1=w, 2=bias, 3=out, 4=uniform.
pub const CONV2D: &str = r#"
struct U {
  n: u32, cin: u32, cout: u32, h: u32, w: u32, ho: u32, wo: u32,
  kh: u32, kw: u32, stride_h: u32, stride_w: u32, pad_h: u32, pad_w: u32,
  dil_h: u32, dil_w: u32, group: u32,
};
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> wgt: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> u: U;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
  let i = gid.y * (ng.x * 64u) + gid.x;
  let total = u.n * u.cout * u.ho * u.wo;
  if (i >= total) { return; }
  let ow = i % u.wo;
  let oh = (i / u.wo) % u.ho;
  let co = (i / (u.wo * u.ho)) % u.cout;
  let no = i / (u.wo * u.ho * u.cout);
  let cin_g = u.cin / u.group;
  let cout_g = u.cout / u.group;
  let g = co / cout_g;
  let ci0 = g * cin_g;
  var acc: f32 = bias[co];
  for (var cc: u32 = 0u; cc < cin_g; cc = cc + 1u) {
    let ci = ci0 + cc;
    for (var r: u32 = 0u; r < u.kh; r = r + 1u) {
      let ih_s = i32(oh * u.stride_h + r * u.dil_h) - i32(u.pad_h);
      if (ih_s < 0 || ih_s >= i32(u.h)) { continue; }
      let ih = u32(ih_s);
      for (var s: u32 = 0u; s < u.kw; s = s + 1u) {
        let iw_s = i32(ow * u.stride_w + s * u.dil_w) - i32(u.pad_w);
        if (iw_s < 0 || iw_s >= i32(u.w)) { continue; }
        let iw = u32(iw_s);
        let xv = x[((no * u.cin + ci) * u.h + ih) * u.w + iw];
        let wv = wgt[(((co * cin_g + cc) * u.kh + r) * u.kw + s)];
        acc = fma(xv, wv, acc);
      }
    }
  }
  out[i] = acc;
}
"#;

/// Direct NCHW conv-transpose (gather form) with groups, dilation, stride, pad,
/// output_padding, bias. Weight layout `[Cin, Cout/group, kH, kW]`.
/// Bindings: 0=x, 1=w, 2=bias, 3=out, 4=uniform.
pub const CONVT2D: &str = r#"
struct U {
  n: u32, cin: u32, cout: u32, h: u32, w: u32, ho: u32, wo: u32,
  kh: u32, kw: u32, stride_h: u32, stride_w: u32, pad_h: u32, pad_w: u32,
  dil_h: u32, dil_w: u32, group: u32,
};
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> wgt: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> u: U;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
  let i = gid.y * (ng.x * 64u) + gid.x;
  let total = u.n * u.cout * u.ho * u.wo;
  if (i >= total) { return; }
  let ow = i % u.wo;
  let oh = (i / u.wo) % u.ho;
  let co = (i / (u.wo * u.ho)) % u.cout;
  let no = i / (u.wo * u.ho * u.cout);
  let cin_g = u.cin / u.group;
  let cout_g = u.cout / u.group;
  let g = co / cout_g;
  let co_in_g = co % cout_g;
  let ci0 = g * cin_g;
  var acc: f32 = bias[co];
  for (var cc: u32 = 0u; cc < cin_g; cc = cc + 1u) {
    let ci = ci0 + cc;
    for (var r: u32 = 0u; r < u.kh; r = r + 1u) {
      let num_h = i32(oh + u.pad_h) - i32(r * u.dil_h);
      if (num_h < 0) { continue; }
      if (u32(num_h) % u.stride_h != 0u) { continue; }
      let ih = u32(num_h) / u.stride_h;
      if (ih >= u.h) { continue; }
      for (var s: u32 = 0u; s < u.kw; s = s + 1u) {
        let num_w = i32(ow + u.pad_w) - i32(s * u.dil_w);
        if (num_w < 0) { continue; }
        if (u32(num_w) % u.stride_w != 0u) { continue; }
        let iw = u32(num_w) / u.stride_w;
        if (iw >= u.w) { continue; }
        let xv = x[((no * u.cin + ci) * u.h + ih) * u.w + iw];
        let wv = wgt[(((ci * cout_g + co_in_g) * u.kh + r) * u.kw + s)];
        acc = fma(xv, wv, acc);
      }
    }
  }
  out[i] = acc;
}
"#;

/// Gemm: `out[M,N] = A[M,K] @ B + C`. `trans_b`: B is `[N,K]` (1) or `[K,N]`
/// (0). `has_bias`: add per-N bias C (broadcast over M). alpha/beta fixed to 1.
/// Bindings: 0=a, 1=b, 2=bias, 3=out, 4=uniform.
pub const GEMM: &str = r#"
struct U { m: u32, n: u32, k: u32, trans_b: u32, has_bias: u32, _p0: u32, _p1: u32, _p2: u32 };
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> u: U;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
  let i = gid.y * (ng.x * 64u) + gid.x;
  if (i >= u.m * u.n) { return; }
  let col = i % u.n;
  let row = i / u.n;
  var acc: f32 = 0.0;
  for (var kk: u32 = 0u; kk < u.k; kk = kk + 1u) {
    let av = a[row * u.k + kk];
    var bv: f32;
    if (u.trans_b == 1u) { bv = b[col * u.k + kk]; } else { bv = b[kk * u.n + col]; }
    acc = fma(av, bv, acc);
  }
  if (u.has_bias == 1u) { acc = acc + bias[col]; }
  out[i] = acc;
}
"#;

/// InstanceNormalization: per (n, c) normalize over H*W, then `scale[c]*xn +
/// bias[c]`. One workgroup (64 threads) per channel.
/// Bindings: 0=x, 1=scale, 2=bias, 3=out, 4=uniform.
pub const INSTANCE_NORM: &str = r#"
struct U { n: u32, c: u32, hw: u32, eps: f32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> scale: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> u: U;
var<workgroup> red_sum: array<f32, 64>;
var<workgroup> red_sq: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let chan = wid.x;            // 0 .. n*c
  let c = chan % u.c;
  let base = chan * u.hw;
  let tid = lid.x;
  var s: f32 = 0.0;
  var sq: f32 = 0.0;
  for (var j: u32 = tid; j < u.hw; j = j + 64u) {
    let v = x[base + j];
    s = s + v;
    sq = sq + v * v;
  }
  red_sum[tid] = s;
  red_sq[tid] = sq;
  workgroupBarrier();
  var stride: u32 = 32u;
  loop {
    if (tid < stride) {
      red_sum[tid] = red_sum[tid] + red_sum[tid + stride];
      red_sq[tid] = red_sq[tid] + red_sq[tid + stride];
    }
    workgroupBarrier();
    if (stride == 1u) { break; }
    stride = stride / 2u;
  }
  let mean = red_sum[0] / f32(u.hw);
  let var_ = red_sq[0] / f32(u.hw) - mean * mean;
  let inv = inverseSqrt(var_ + u.eps);
  let sc = scale[c];
  let bi = bias[c];
  for (var j: u32 = tid; j < u.hw; j = j + 64u) {
    out[base + j] = (x[base + j] - mean) * inv * sc + bi;
  }
}
"#;

/// Per-channel affine `out = x * a[c] + b[c]` over NCHW. Used for folded
/// BatchNormalization (a, b precomputed on host).
/// Bindings: 0=x, 1=a, 2=b, 3=out, 4=uniform.
pub const CHANNEL_AFFINE: &str = r#"
struct U { total: u32, c: u32, hw: u32, _p: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> u: U;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
  let i = gid.y * (ng.x * 64u) + gid.x;
  if (i >= u.total) { return; }
  let c = (i / u.hw) % u.c;
  out[i] = x[i] * a[c] + b[c];
}
"#;

/// PRelu with per-channel slope: `x>0 ? x : slope[c]*x`.
/// Bindings: 0=x, 1=slope, 2=out, 3=uniform.
pub const PRELU: &str = r#"
struct U { total: u32, c: u32, hw: u32, slope_len: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> slope: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> u: U;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
  let i = gid.y * (ng.x * 64u) + gid.x;
  if (i >= u.total) { return; }
  let c = (i / u.hw) % u.c;
  let s = select(slope[c], slope[0], u.slope_len == 1u);
  let v = x[i];
  out[i] = select(s * v, v, v > 0.0);
}
"#;

/// Elementwise unary activation. `kind`: 0 relu, 1 sigmoid, 2 tanh, 3 leakyrelu
/// (alpha), 4 identity-clip[0,1], 5 sqrt, 6 hardsigmoid (clip(alpha*x+beta,0,1)),
/// 7 clip (clamp to [alpha, beta]). Bindings: 0=x, 1=out, 2=uniform.
pub const UNARY: &str = r#"
struct U { total: u32, kind: u32, alpha: f32, beta: f32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> u: U;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
  let i = gid.y * (ng.x * 64u) + gid.x;
  if (i >= u.total) { return; }
  let v = x[i];
  var r: f32 = v;
  switch (u.kind) {
    case 0u: { r = max(v, 0.0); }
    case 1u: { r = 1.0 / (1.0 + exp(-v)); }
    case 2u: { r = tanh(v); }
    case 3u: { r = select(u.alpha * v, v, v > 0.0); }
    case 4u: { r = clamp(v, 0.0, 1.0); }
    case 5u: { r = sqrt(v); }
    case 6u: { r = clamp(u.alpha * v + u.beta, 0.0, 1.0); }
    default: { r = clamp(v, u.alpha, u.beta); }
  }
  out[i] = r;
}
"#;

/// Strided slice / gather over up-to-4D NCHW. Output coord `oc` maps to input
/// coord `start[k] + oc[k]*step[k]`. Also serves ONNX Split (one dispatch per
/// output block). Bindings: 0=x, 1=out, 2=uniform.
pub const SLICE: &str = r#"
struct U { total: u32, _p0: u32, _p1: u32, _p2: u32, id: vec4<u32>, od: vec4<u32>, start: vec4<u32>, step: vec4<u32> };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> u: U;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
  let i = gid.y * (ng.x * 64u) + gid.x;
  if (i >= u.total) { return; }
  let oc3 = i % u.od.w;
  let oc2 = (i / u.od.w) % u.od.z;
  let oc1 = (i / (u.od.w * u.od.z)) % u.od.y;
  let oc0 = i / (u.od.w * u.od.z * u.od.y);
  let ic0 = u.start.x + oc0 * u.step.x;
  let ic1 = u.start.y + oc1 * u.step.y;
  let ic2 = u.start.z + oc2 * u.step.z;
  let ic3 = u.start.w + oc3 * u.step.w;
  out[i] = x[((ic0 * u.id.y + ic1) * u.id.z + ic2) * u.id.w + ic3];
}
"#;

/// GlobalAveragePool (NCHW): mean over H*W -> `[N, C, 1, 1]`. One thread per
/// (n, c) output. Bindings: 0=x, 1=out, 2=uniform.
pub const GLOBAL_AVG_POOL: &str = r#"
struct U { nc: u32, hw: u32, _p0: u32, _p1: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> u: U;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
  let i = gid.y * (ng.x * 64u) + gid.x;
  if (i >= u.nc) { return; }
  let base = i * u.hw;
  var s: f32 = 0.0;
  for (var j: u32 = 0u; j < u.hw; j = j + 1u) { s = s + x[base + j]; }
  out[i] = s / f32(u.hw);
}
"#;

/// ReduceSum over a single axis (keepdims), on an up-to-4D tensor. `axis` is the
/// reduced axis; `astride`/`alen` are that axis's stride and length; the output
/// has that axis collapsed to 1. One thread per output element. Bindings: 0=x,
/// 1=out, 2=uniform.
pub const REDUCE_SUM: &str = r#"
struct U { total: u32, alen: u32, astride: u32, _p: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> u: U;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
  let i = gid.y * (ng.x * 64u) + gid.x;
  if (i >= u.total) { return; }
  // Base input index: split the output linear index around the (collapsed) axis.
  let inner = i % u.astride;
  let outer = i / u.astride;
  let base = outer * u.alen * u.astride + inner;
  var s: f32 = 0.0;
  for (var j: u32 = 0u; j < u.alen; j = j + 1u) { s = s + x[base + j * u.astride]; }
  out[i] = s;
}
"#;

/// Broadcasting binary elementwise over up-to-4D NCHW (right-aligned). `op`:
/// 0 add, 1 sub, 2 mul, 3 div. Operand dims of 1 broadcast (stride 0).
/// Bindings: 0=a, 1=b, 2=out, 3=uniform.
pub const BINARY: &str = r#"
struct U {
  op: u32, total: u32, _p0: u32, _p1: u32,
  od: vec4<u32>,   // output dims (d0,d1,d2,d3)
  ad: vec4<u32>,   // operand A dims
  bd: vec4<u32>,   // operand B dims
};
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> u: U;
fn lin(d: vec4<u32>, c: vec4<u32>) -> u32 {
  // index with broadcast: where dim==1, force coord 0.
  let i0 = select(c.x, 0u, d.x == 1u);
  let i1 = select(c.y, 0u, d.y == 1u);
  let i2 = select(c.z, 0u, d.z == 1u);
  let i3 = select(c.w, 0u, d.w == 1u);
  return ((i0 * d.y + i1) * d.z + i2) * d.w + i3;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
  let i = gid.y * (ng.x * 64u) + gid.x;
  if (i >= u.total) { return; }
  let c3 = i % u.od.w;
  let c2 = (i / u.od.w) % u.od.z;
  let c1 = (i / (u.od.w * u.od.z)) % u.od.y;
  let c0 = i / (u.od.w * u.od.z * u.od.y);
  let coord = vec4<u32>(c0, c1, c2, c3);
  let av = a[lin(u.ad, coord)];
  let bv = b[lin(u.bd, coord)];
  var r: f32;
  switch (u.op) {
    case 0u: { r = av + bv; }
    case 1u: { r = av - bv; }
    case 2u: { r = av * bv; }
    default: { r = av / bv; }
  }
  out[i] = r;
}
"#;

/// Broadcast-copy (Expand): replicate input (4D, dims-of-1 broadcast) into the
/// output shape. Bindings: 0=x, 1=out, 2=uniform.
pub const EXPAND: &str = r#"
struct U { total: u32, _p0: u32, _p1: u32, _p2: u32, od: vec4<u32>, xd: vec4<u32> };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> u: U;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
  let i = gid.y * (ng.x * 64u) + gid.x;
  if (i >= u.total) { return; }
  let c3 = i % u.od.w;
  let c2 = (i / u.od.w) % u.od.z;
  let c1 = (i / (u.od.w * u.od.z)) % u.od.y;
  let c0 = i / (u.od.w * u.od.z * u.od.y);
  let i0 = select(c0, 0u, u.xd.x == 1u);
  let i1 = select(c1, 0u, u.xd.y == 1u);
  let i2 = select(c2, 0u, u.xd.z == 1u);
  let i3 = select(c3, 0u, u.xd.w == 1u);
  out[i] = x[((i0 * u.xd.y + i1) * u.xd.z + i2) * u.xd.w + i3];
}
"#;

/// General 4D transpose by `perm`. Input dims `id`, output dims `od = id[perm]`.
/// `inv[k]` = which output axis the input axis k went to (so we can map an
/// output coord back to an input linear index). Bindings: 0=x, 1=out, 2=uniform.
pub const TRANSPOSE: &str = r#"
struct U { total: u32, _p0: u32, _p1: u32, _p2: u32, id: vec4<u32>, od: vec4<u32>, perm: vec4<u32> };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> u: U;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
  let i = gid.y * (ng.x * 64u) + gid.x;
  if (i >= u.total) { return; }
  let oc3 = i % u.od.w;
  let oc2 = (i / u.od.w) % u.od.z;
  let oc1 = (i / (u.od.w * u.od.z)) % u.od.y;
  let oc0 = i / (u.od.w * u.od.z * u.od.y);
  let oc = array<u32, 4>(oc0, oc1, oc2, oc3);
  // output axis a came from input axis perm[a]; scatter the output coords back.
  var ic = array<u32, 4>(0u, 0u, 0u, 0u);
  ic[u.perm.x] = oc[0];
  ic[u.perm.y] = oc[1];
  ic[u.perm.z] = oc[2];
  ic[u.perm.w] = oc[3];
  let idx = ((ic[0] * u.id.y + ic[1]) * u.id.z + ic[2]) * u.id.w + ic[3];
  out[i] = x[idx];
}
"#;

/// DepthToSpace. `mode`: 0 DCR (default), 1 CRD. Output `[N, C/(b*b), H*b,
/// W*b]`. Bindings: 0=x, 1=out, 2=uniform.
pub const DEPTH_TO_SPACE: &str = r#"
struct U { total: u32, n: u32, oc: u32, oh: u32, ow: u32, ic: u32, ih: u32, iw: u32, b: u32, mode: u32, _p0: u32, _p1: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> u: U;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
  let i = gid.y * (ng.x * 64u) + gid.x;
  if (i >= u.total) { return; }
  let ow = i % u.ow;
  let oh = (i / u.ow) % u.oh;
  let oc = (i / (u.ow * u.oh)) % u.oc;
  let no = i / (u.ow * u.oh * u.oc);
  let bh = oh % u.b;
  let bw = ow % u.b;
  let ih = oh / u.b;
  let iw = ow / u.b;
  var ici: u32;
  if (u.mode == 0u) { ici = (bh * u.b + bw) * u.oc + oc; }   // DCR
  else { ici = oc * (u.b * u.b) + (bh * u.b + bw); }          // CRD
  out[i] = x[((no * u.ic + ici) * u.ih + ih) * u.iw + iw];
}
"#;

/// Resize (2D, NCHW). `mode`: 0 nearest, 1 bilinear. `coord`: 0 asymmetric,
/// 1 half_pixel. Bindings: 0=x, 1=out, 2=uniform.
pub const RESIZE: &str = r#"
struct U { total: u32, n: u32, c: u32, ih: u32, iw: u32, oh: u32, ow: u32, mode: u32, coord: u32, _p0: u32, _p1: u32, _p2: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> u: U;
fn src_coord(o: u32, scale: f32, coord: u32) -> f32 {
  if (coord == 1u) { return (f32(o) + 0.5) / scale - 0.5; }  // half_pixel
  return f32(o) / scale;                                      // asymmetric
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
  let i = gid.y * (ng.x * 64u) + gid.x;
  if (i >= u.total) { return; }
  let ow = i % u.ow;
  let oh = (i / u.ow) % u.oh;
  let c = (i / (u.ow * u.oh)) % u.c;
  let no = i / (u.ow * u.oh * u.c);
  let base = (no * u.c + c) * u.ih * u.iw;
  let scale_h = f32(u.oh) / f32(u.ih);
  let scale_w = f32(u.ow) / f32(u.iw);
  let fy = src_coord(oh, scale_h, u.coord);
  let fx = src_coord(ow, scale_w, u.coord);
  if (u.mode == 0u) {
    let sy = clamp(i32(floor(fy)), 0, i32(u.ih) - 1);
    let sx = clamp(i32(floor(fx)), 0, i32(u.iw) - 1);
    out[i] = x[base + u32(sy) * u.iw + u32(sx)];
  } else {
    let y0 = floor(fy);
    let x0 = floor(fx);
    let dy = fy - y0;
    let dx = fx - x0;
    let iy0 = clamp(i32(y0), 0, i32(u.ih) - 1);
    let iy1 = clamp(i32(y0) + 1, 0, i32(u.ih) - 1);
    let ix0 = clamp(i32(x0), 0, i32(u.iw) - 1);
    let ix1 = clamp(i32(x0) + 1, 0, i32(u.iw) - 1);
    let v00 = x[base + u32(iy0) * u.iw + u32(ix0)];
    let v01 = x[base + u32(iy0) * u.iw + u32(ix1)];
    let v10 = x[base + u32(iy1) * u.iw + u32(ix0)];
    let v11 = x[base + u32(iy1) * u.iw + u32(ix1)];
    let top = mix(v00, v01, dx);
    let bot = mix(v10, v11, dx);
    out[i] = mix(top, bot, dy);
  }
}
"#;

/// Concat of two NCHW tensors along `axis` (0..3). Output `[od]`; `a` occupies
/// `[0, a_axis)` along `axis`, `b` the rest. Batch-safe (the contiguous-copy
/// path in exec only holds at batch 1). Bindings: 0=a, 1=b, 2=out, 3=uniform.
pub const CONCAT2: &str = r#"
struct U { total: u32, axis: u32, a_axis: u32, _p: u32, od: vec4<u32>, ad: vec4<u32>, bd: vec4<u32> };
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> u: U;
fn lin(d: vec4<u32>, c: vec4<u32>) -> u32 {
  return ((c.x * d.y + c.y) * d.z + c.z) * d.w + c.w;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
  let i = gid.y * (ng.x * 64u) + gid.x;
  if (i >= u.total) { return; }
  let c3 = i % u.od.w;
  let c2 = (i / u.od.w) % u.od.z;
  let c1 = (i / (u.od.w * u.od.z)) % u.od.y;
  let c0 = i / (u.od.w * u.od.z * u.od.y);
  var coord = vec4<u32>(c0, c1, c2, c3);
  let ax = coord[u.axis];
  if (ax < u.a_axis) {
    out[i] = a[lin(u.ad, coord)];
  } else {
    coord[u.axis] = ax - u.a_axis;
    out[i] = b[lin(u.bd, coord)];
  }
}
"#;

/// Zero-upsample (NCHW): place each input pixel at `(y*s_h, x*s_w)` in a larger
/// grid `[(h-1)*s_h+1, (w-1)*s_w+1]`, zeros elsewhere. Used to turn a strided
/// ConvTranspose into a stride-1 conv (with a flipped/transposed kernel), so it
/// can run through the tuned conv path. Bindings: 0=x, 1=out, 2=uniform.
pub const ZERO_UPSAMPLE: &str = r#"
struct U { total: u32, n: u32, c: u32, h: u32, w: u32, oh: u32, ow: u32, s_h: u32, s_w: u32, _p0: u32, _p1: u32, _p2: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> u: U;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
  let i = gid.y * (ng.x * 64u) + gid.x;
  if (i >= u.total) { return; }
  let ow = i % u.ow;
  let oh = (i / u.ow) % u.oh;
  let c = (i / (u.ow * u.oh)) % u.c;
  let no = i / (u.ow * u.oh * u.c);
  if (ow % u.s_w == 0u && oh % u.s_h == 0u) {
    let ih = oh / u.s_h;
    let iw = ow / u.s_w;
    out[i] = x[((no * u.c + c) * u.h + ih) * u.w + iw];
  } else {
    out[i] = 0.0;
  }
}
"#;

/// MaxPool 2D (NCHW) with kernel/stride/pad/dilation. Bindings: 0=x, 1=out,
/// 2=uniform.
pub const MAXPOOL: &str = r#"
struct U {
  total: u32, n: u32, c: u32, h: u32, w: u32, ho: u32, wo: u32,
  kh: u32, kw: u32, stride_h: u32, stride_w: u32, pad_h: u32, pad_w: u32,
  dil_h: u32, dil_w: u32,
};
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> u: U;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
  let i = gid.y * (ng.x * 64u) + gid.x;
  if (i >= u.total) { return; }
  let ow = i % u.wo;
  let oh = (i / u.wo) % u.ho;
  let c = (i / (u.wo * u.ho)) % u.c;
  let no = i / (u.wo * u.ho * u.c);
  let base = (no * u.c + c) * u.h * u.w;
  var m: f32 = -3.4e38;
  for (var r: u32 = 0u; r < u.kh; r = r + 1u) {
    let ih_s = i32(oh * u.stride_h + r * u.dil_h) - i32(u.pad_h);
    if (ih_s < 0 || ih_s >= i32(u.h)) { continue; }
    for (var s: u32 = 0u; s < u.kw; s = s + 1u) {
      let iw_s = i32(ow * u.stride_w + s * u.dil_w) - i32(u.pad_w);
      if (iw_s < 0 || iw_s >= i32(u.w)) { continue; }
      m = max(m, x[base + u32(ih_s) * u.w + u32(iw_s)]);
    }
  }
  out[i] = m;
}
"#;
