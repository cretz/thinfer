//! Quant scheme descriptors. Single source of truth for block layout +
//! WGSL helpers across the source-side ([`StorageEncoding::Quant`]) and
//! kernel-side ([`WeightDtype::Quant`]) enums.
//!
//! Adding a new quant: new variant + entries in every `match self {}`. The
//! closed enum is deliberate (no `dyn`): all schemes ship in-tree, derives
//! stay clean, pipeline cache keys are stable.

// GGUF canonical names use underscores (Q8_0, Q4_K_M, ...). Match the
// upstream identifiers verbatim rather than CamelCase-rename them.
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum QuantKind {
    /// 32 elements/block. Block = `f16 scale` + `i8 qs[32]` = 34 bytes.
    /// Dequant: `f = qs[i] * f32(scale)`. Lossless-ish baseline.
    Q8_0,
    /// 32 elements/block. Block = `f16 scale` + `u4 qs[32]` (packed two
    /// per byte, low nibble first) = 18 bytes. Dequant:
    /// `f = (i8(qs[i]) - 8) * f32(scale)`. WGSL stubbed.
    Q4_0,
    /// 256 elements/super-block. Block = 144 bytes:
    /// `f16 d` + `f16 dmin` + `u8 scales[12]` (6-bit packed; 8 sub-blocks
    /// of 32) + `u4 qs[256]` (packed). Dequant per sub-block uses 6-bit
    /// scale + 6-bit min decoded from `scales[]`. WGSL stubbed.
    Q4_K,
}

impl QuantKind {
    /// Number of fp32 elements one block decodes to.
    pub const fn block_size(self) -> u32 {
        match self {
            Self::Q8_0 => 32,
            Self::Q4_0 => 32,
            Self::Q4_K => 256,
        }
    }

    /// On-disk + on-GPU bytes per block. Matches GGUF / llama.cpp layout
    /// verbatim — we never repack (web path DMAs OPFS bytes straight to
    /// GPU, no WASM-heap detour).
    pub const fn bytes_per_block(self) -> u32 {
        match self {
            Self::Q8_0 => 34,
            Self::Q4_0 => 18,
            Self::Q4_K => 144,
        }
    }

    /// Total GPU bytes for a tensor of `n` elements. Asserts the row count
    /// divides `block_size` (GGUF tensors satisfy this by construction).
    pub const fn bytes_for_elements(self, n: u64) -> u64 {
        let bs = self.block_size() as u64;
        assert!(
            n.is_multiple_of(bs),
            "quant tensor element count must be a multiple of block_size"
        );
        (n / bs) * (self.bytes_per_block() as u64)
    }

    /// Short tag for pipeline cache discriminator. Distinct per quant.
    pub const fn hint(self) -> &'static str {
        match self {
            Self::Q8_0 => "q80",
            Self::Q4_0 => "q40",
            Self::Q4_K => "q4k",
        }
    }

    /// WGSL declaration for the B (weight) binding. Always `array<u32>`
    /// — quant blocks are bag-of-bytes, accessed with shifts. Caller
    /// supplies the binding index (matmul uses 1).
    pub const fn storage_decl(self, binding: u32) -> &'static str {
        let _ = self;
        let _ = binding;
        // All quants share the same storage decl shape; the kernel reads
        // raw u32 words. binding=1 is the only call site today; if we
        // ever vary it, swap to a `format!` at the use site.
        "@group(0) @binding(1) var<storage, read> b: array<u32>;"
    }

    /// WGSL identifier of the per-scheme dequant function emitted by
    /// [`load_b_block_fn`]. Used by the matmul kernel body to call the
    /// right unpack helper after the prelude has been pasted in.
    pub const fn load_b_block_call(self) -> &'static str {
        match self {
            Self::Q8_0 => "load_b_block_q8_0",
            Self::Q4_0 => "load_b_block_q4_0",
            Self::Q4_K => "load_b_block_q4_k",
        }
    }

    /// WGSL identifier of the scheme's "read block scale" helper
    /// (`fn block_scale_<scheme>(byte0: u32) -> f32`). Lets the matmul
    /// kernel split a block's dequant across TPB threads: each thread
    /// reads the scale once, then unpacks its sub-range of elements via
    /// [`block_elem_call`]. Keeps `bk=block_size` viable at 16 KiB
    /// shared-mem (WebGPU baseline) while still saturating a 256-thread
    /// WG on the B-load.
    pub const fn block_scale_call(self) -> &'static str {
        match self {
            Self::Q8_0 => "block_scale_q8_0",
            Self::Q4_0 => "block_scale_q4_0",
            Self::Q4_K => "block_scale_q4_k",
        }
    }

    /// WGSL identifier of the scheme's per-element dequant helper
    /// (`fn block_elem_<scheme>(byte0: u32, scale: f32, elem: u32) -> f32`).
    /// `elem` is the in-block element index `[0, block_size)`. Paired
    /// with [`block_scale_call`] in the matmul kernel's cooperative
    /// B-load loop.
    pub const fn block_elem_call(self) -> &'static str {
        match self {
            Self::Q8_0 => "block_elem_q8_0",
            Self::Q4_0 => "block_elem_q4_0",
            Self::Q4_K => "block_elem_q4_k",
        }
    }

    /// WGSL `fn load_b_block(block_idx: u32, out: ptr<function, array<f32, BS>>)`
    /// — dequants one block of `block_size()` elements into the caller's
    /// register array. Inner K-loop iterates K in chunks of `block_size`
    /// to amortize the unaligned-byte-load math at the block boundary.
    ///
    /// Q4_0 and Q4_K return `todo!()` placeholders; their unpack logic
    /// lands when those quants get wired up. The signature is real so
    /// callers can compose the kernel skeleton today.
    pub fn load_b_block_fn(self) -> String {
        match self {
            Self::Q8_0 => Q8_0_LOAD_B_BLOCK.to_string(),
            Self::Q4_0 => "// Q4_0 load_b_block: STUB. Same shape as Q8_0; nibble unpack \
                          + `(i8 - 8) * scale`. Fill in when wiring Q4_0.\n"
                .to_string(),
            Self::Q4_K => "// Q4_K load_b_block: STUB. Super-block 256, 8 sub-blocks of 32, \
                          6-bit scale/min codes packed in scales[12]. Fill in when wiring \
                          Q4_K.\n"
                .to_string(),
        }
    }
}

/// Q8_0 block layout (34 bytes):
///   bytes [0..2]    : f16 scale
///   bytes [2..34]   : i8 qs[32]
///
/// Blocks pack end-to-end (block i starts at byte 34*i). The buffer is
/// declared `array<u32>`; we read by computing the byte offset, then the
/// containing u32 word(s) and shifting. The block-aligned inner loop
/// means each block's 34 bytes touch words `[34*i / 4 .. (34*i+34) / 4]`
/// — at most 10 words. We unpack the whole block once into a function-
/// scope `array<f32, 32>`, then the K-loop reads from that.
///
/// fp16 -> f32 expansion is the IEEE bit-pattern path (sign+exp+mantissa
/// shift). No subnormal / denormal handling needed for the scale — Q8_0
/// scales are never subnormal in practice (training-time `amax/127`).
/// Keep it simple: assume normal/zero/inf passthrough.
const Q8_0_LOAD_B_BLOCK: &str = r#"
fn f16_bits_to_f32(h: u32) -> f32 {
    let h_masked: u32 = h & 0xFFFFu;
    let sign: u32 = (h_masked & 0x8000u) << 16u;
    let exp: u32  = (h_masked >> 10u) & 0x1Fu;
    let mant: u32 = h_masked & 0x3FFu;
    if (exp == 0u) {
        // zero or subnormal -> treat as zero (fast path; Q8_0 scales
        // are never subnormal in trained checkpoints).
        if (mant == 0u) {
            return bitcast<f32>(sign);
        }
        // Subnormal: renormalize. Slow path, rarely hit.
        var m: u32 = mant;
        var e: i32 = -1;
        loop {
            if ((m & 0x400u) != 0u) { break; }
            m = m << 1u;
            e = e - 1;
            if (e < -10) { break; }
        }
        let f_exp: u32 = u32(127 + e - 14);
        let f_mant: u32 = (m & 0x3FFu) << 13u;
        return bitcast<f32>(sign | (f_exp << 23u) | f_mant);
    }
    if (exp == 0x1Fu) {
        // inf/NaN
        return bitcast<f32>(sign | 0x7F800000u | (mant << 13u));
    }
    let f_exp: u32 = (exp + 112u) << 23u;
    let f_mant: u32 = mant << 13u;
    return bitcast<f32>(sign | f_exp | f_mant);
}

// Extract byte `byte_offset` from the u32-typed buffer `b`.
fn b_byte(byte_offset: u32) -> u32 {
    let wi: u32 = byte_offset >> 2u;
    let sh: u32 = (byte_offset & 3u) * 8u;
    return (b[wi] >> sh) & 0xFFu;
}

// Sign-extend a u8 (low 8 bits of `x`) to i32.
fn sext_i8(x: u32) -> i32 {
    let lo: u32 = x & 0xFFu;
    if ((lo & 0x80u) != 0u) {
        return i32(lo) - 256;
    }
    return i32(lo);
}

// Read the f16 scale at the start of a Q8_0 block.
fn block_scale_q8_0(byte0: u32) -> f32 {
    let lo: u32 = b_byte(byte0);
    let hi: u32 = b_byte(byte0 + 1u);
    return f16_bits_to_f32(lo | (hi << 8u));
}

// Dequant one element of a Q8_0 block. `elem` in `[0, 32)`. Caller
// supplies the precomputed scale so a workgroup of TPB threads
// cooperating on one block only reads the f16 scale TPB times in
// parallel rather than redoing the whole block per thread.
fn block_elem_q8_0(byte0: u32, scale: f32, elem: u32) -> f32 {
    let q: i32 = sext_i8(b_byte(byte0 + 2u + elem));
    return f32(q) * scale;
}

// Whole-block dequant. Retained for callers that need a register-
// resident block array (e.g. CPU-side parity helpers exported via the
// same name). The matmul kernel uses the split helpers above.
fn load_b_block_q8_0(byte0: u32, dst: ptr<function, array<f32, 32u>>) {
    let scale: f32 = block_scale_q8_0(byte0);
    for (var i: u32 = 0u; i < 32u; i = i + 1u) {
        (*dst)[i] = block_elem_q8_0(byte0, scale, i);
    }
}
"#;

/// Dequantize one block (32 elements) of Q8_0 data from `src` (must be
/// `>= 34` bytes; only the first 34 are read) into `dst[0..32]`.
///
/// Port of llama.cpp's `dequantize_row_q8_0` for a single block:
///
/// ```c
/// const float d = GGML_FP16_TO_FP32(blk.d);
/// for (j = 0; j < 32; ++j) y[j] = blk.qs[j] * d;
/// ```
///
/// Bit-clean reference for the GPU kernel. We expand the fp16 scale via
/// `half::f16::to_f32` (IEEE 754 conversion); subnormal scales round
/// through the same machinery as the WGSL helper.
pub fn dequantize_block_q8_0(src: &[u8], dst: &mut [f32; 32]) {
    assert!(src.len() >= 34, "Q8_0 block needs 34 bytes");
    let scale = half::f16::from_le_bytes([src[0], src[1]]).to_f32();
    for j in 0..32 {
        let q = src[2 + j] as i8;
        dst[j] = (q as f32) * scale;
    }
}

/// Dequantize a contiguous Q8_0 buffer. `dst.len()` must equal
/// `src.len() / 34 * 32` (no partial blocks).
pub fn dequantize_row_q8_0(src: &[u8], dst: &mut [f32]) {
    assert!(
        src.len().is_multiple_of(34),
        "Q8_0 buffer must be a whole number of 34-byte blocks"
    );
    let n_blocks = src.len() / 34;
    assert_eq!(
        dst.len(),
        n_blocks * 32,
        "dst.len() must match block count * 32"
    );
    for i in 0..n_blocks {
        let s = i * 34;
        let mut blk = [0f32; 32];
        dequantize_block_q8_0(&src[s..s + 34], &mut blk);
        dst[i * 32..(i + 1) * 32].copy_from_slice(&blk);
    }
}

/// Encode `src.len() / 32` blocks of fp32 data as Q8_0. Per-block scale
/// = max(|src|) / 127, with zero-scale blocks emitting all-zero qs.
/// Mirrors llama.cpp's `quantize_row_q8_0_ref` for testing.
pub fn quantize_row_q8_0(src: &[f32], dst: &mut Vec<u8>) {
    assert!(
        src.len().is_multiple_of(32),
        "Q8_0 quantize input must be multiple of 32"
    );
    let n_blocks = src.len() / 32;
    dst.clear();
    dst.reserve(n_blocks * 34);
    for i in 0..n_blocks {
        let block = &src[i * 32..(i + 1) * 32];
        let amax = block.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        let scale = amax / 127.0;
        let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        let scale_h = half::f16::from_f32(scale);
        dst.extend_from_slice(&scale_h.to_le_bytes());
        for &v in block {
            let q = (v * inv).round().clamp(-127.0, 127.0) as i8;
            dst.push(q as u8);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q8_0_layout() {
        assert_eq!(QuantKind::Q8_0.block_size(), 32);
        assert_eq!(QuantKind::Q8_0.bytes_per_block(), 34);
        assert_eq!(QuantKind::Q8_0.bytes_for_elements(64), 68);
        assert_eq!(QuantKind::Q8_0.bytes_for_elements(1024), 1024 / 32 * 34);
    }

    #[test]
    fn q4_0_layout() {
        assert_eq!(QuantKind::Q4_0.block_size(), 32);
        assert_eq!(QuantKind::Q4_0.bytes_per_block(), 18);
    }

    #[test]
    fn q4_k_layout() {
        assert_eq!(QuantKind::Q4_K.block_size(), 256);
        assert_eq!(QuantKind::Q4_K.bytes_per_block(), 144);
    }

    #[test]
    #[should_panic(expected = "multiple of block_size")]
    fn q8_0_rejects_non_multiple() {
        let _ = QuantKind::Q8_0.bytes_for_elements(33);
    }

    #[test]
    fn hints_distinct() {
        let mut s = std::collections::HashSet::new();
        for k in [QuantKind::Q8_0, QuantKind::Q4_0, QuantKind::Q4_K] {
            assert!(s.insert(k.hint()));
        }
    }

    #[test]
    fn q8_0_wgsl_nonempty() {
        let w = QuantKind::Q8_0.load_b_block_fn();
        assert!(w.contains("load_b_block_q8_0"));
        assert!(w.contains("f16_bits_to_f32"));
    }

    /// Hand-crafted Q8_0 block: scale=1.0 (fp16 bits 0x3C00), qs spans the
    /// full i8 range. Dequant produces exact f32 = i8 cast.
    #[test]
    fn dequant_q8_0_handcrafted() {
        let mut block = Vec::with_capacity(34);
        // fp16 1.0 = 0x3C00, LE bytes [0x00, 0x3C]
        block.push(0x00);
        block.push(0x3C);
        let qs: [i8; 32] = [
            -127, -64, -32, -16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16, 32, 64, 127, 100, 50, 25, 12,
            6, 3, -3, -6, -12, -25, -50, -100, -127, 42, -42,
        ];
        for &q in &qs {
            block.push(q as u8);
        }
        let mut out = [0f32; 32];
        dequantize_block_q8_0(&block, &mut out);
        for (i, &q) in qs.iter().enumerate() {
            assert_eq!(out[i], q as f32, "idx {i}: q={q}");
        }
    }

    /// Scale=0 block: every dequant output must be 0.0 regardless of qs.
    #[test]
    fn dequant_q8_0_zero_scale() {
        let mut block = vec![0u8; 34]; // fp16 +0.0 has bits 0x0000
        // Put nonzero qs to confirm they don't leak through.
        for i in 0..32 {
            block[2 + i] = ((i as i32) - 16) as u8;
        }
        let mut out = [1f32; 32];
        dequantize_block_q8_0(&block, &mut out);
        for &v in &out {
            assert_eq!(v, 0.0);
        }
    }

    /// Round-trip: quantize known fp32 → dequant → original within 1/2
    /// step. Each block's max-abs gets mapped to ±127, so step = amax/127
    /// and per-element error bound = step/2.
    #[test]
    fn q8_0_roundtrip_within_half_step() {
        // Two blocks: [0..32) sin-like and [32..64) decay.
        let mut src = Vec::with_capacity(64);
        for i in 0..32 {
            src.push(((i as f32) * 0.3).sin() * 12.5);
        }
        for i in 0..32 {
            src.push((-(i as f32) * 0.07).exp() * 100.0);
        }
        let mut q = Vec::new();
        quantize_row_q8_0(&src, &mut q);
        assert_eq!(q.len(), 2 * 34);
        let mut deq = vec![0f32; 64];
        dequantize_row_q8_0(&q, &mut deq);

        for (block_idx, block) in src.chunks(32).enumerate() {
            let amax = block.iter().fold(0f32, |a, &v| a.max(v.abs()));
            let step = amax / 127.0;
            for (j, &orig) in block.iter().enumerate() {
                let got = deq[block_idx * 32 + j];
                let err = (orig - got).abs();
                // Step/2 plus a small slack for fp16 scale rounding.
                let tol = step * 0.51 + 1e-4;
                assert!(
                    err <= tol,
                    "block {block_idx} idx {j}: orig={orig} got={got} err={err} tol={tol}"
                );
            }
        }
    }
}
