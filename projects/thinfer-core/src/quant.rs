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
    /// 32 elements/block. `f16 d` + `i8 qs[32]` = 34 bytes.
    /// Dequant: `f = qs[i] * f32(d)`.
    Q8_0,
    /// 32 elements/block. `f16 d` + `u4 qs[32]` (packed, low-nibble first)
    /// = 18 bytes. Dequant: `f = (i8(nib) - 8) * f32(d)`.
    Q4_0,
    /// 256-elem super-block, 8 sub-blocks of 32. `f16 d` + `f16 dmin` +
    /// `u8 scales[12]` (6-bit packed sc/m, 8 each) + `u4 qs[256]` = 144 B.
    /// Per sub-block: `f = d*sc[s]*nib - dmin*m[s]`.
    Q4_K,
    /// 256-elem super-block, 8 sub-blocks of 32. Q4_K layout + extra
    /// `u8 qh[32]` high bits (one per element, 8-per-byte) = 176 B.
    /// `nib5 = low_nib | (high_bit << 4)`. Dequant: same formula as Q4_K.
    Q5_K,
    /// 256-elem super-block, 16 sub-blocks of 16. `u8 ql[128]` (low 4 bits)
    /// + `u8 qh[64]` (high 2 bits) + `i8 sc[16]` + `f16 d` = 210 B.
    ///   Per sub-block (16 elements): `q = ((ql & 0xF) | ((qh & 3) << 4)) - 32;
    ///   f = d * sc[is] * q`.
    Q6_K,
}

impl QuantKind {
    /// Number of fp32 elements one block decodes to.
    pub const fn block_size(self) -> u32 {
        match self {
            Self::Q8_0 => 32,
            Self::Q4_0 => 32,
            Self::Q4_K => 256,
            Self::Q5_K => 256,
            Self::Q6_K => 256,
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
            Self::Q5_K => 176,
            Self::Q6_K => 210,
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
            Self::Q5_K => "q5k",
            Self::Q6_K => "q6k",
        }
    }

    /// WGSL declaration for the B (weight) binding. Always `array<u32>`
    /// — quant blocks are bag-of-bytes, accessed with shifts.
    pub const fn storage_decl(self, _binding: u32) -> &'static str {
        "@group(0) @binding(1) var<storage, read> b: array<u32>;"
    }

    /// WGSL identifier of the per-scheme whole-block dequant function
    /// (`fn load_b_block_<k>(byte0: u32, dst: ptr<function, array<f32, BS>>)`).
    /// Retained for parity helpers; the matmul uses the split init/elem
    /// helpers below for cooperative dequant.
    pub const fn load_b_block_call(self) -> &'static str {
        match self {
            Self::Q8_0 => "load_b_block_q8_0",
            Self::Q4_0 => "load_b_block_q4_0",
            Self::Q4_K => "load_b_block_q4_k",
            Self::Q5_K => "load_b_block_q5_k",
            Self::Q6_K => "load_b_block_q6_k",
        }
    }

    /// WGSL identifier of the per-block state initializer
    /// (`fn block_init_<k>(byte0) -> BlockState_<k>`). Called once per
    /// (block, cooperating thread) at the top of the cooperative dequant
    /// loop. For Q8_0/Q4_0 the state is a single f32 scale; for K-family
    /// it holds `d`/`dmin` plus the 8 (or 16) sub-block scale/min values
    /// decoded once so per-element calls don't reread the 6-bit packed
    /// scales table.
    pub const fn block_state_call(self) -> &'static str {
        match self {
            Self::Q8_0 => "block_init_q8_0",
            Self::Q4_0 => "block_init_q4_0",
            Self::Q4_K => "block_init_q4_k",
            Self::Q5_K => "block_init_q5_k",
            Self::Q6_K => "block_init_q6_k",
        }
    }

    /// WGSL identifier of the per-element dequant helper
    /// (`fn block_elem_<k>(byte0, state, elem) -> f32`). `state` is the
    /// per-scheme type returned by [`block_state_call`]; the matmul
    /// template lets WGSL infer the type via `let st = init(...)` so the
    /// caller does not need to know the per-scheme struct name.
    pub const fn block_elem_call(self) -> &'static str {
        match self {
            Self::Q8_0 => "block_elem_q8_0",
            Self::Q4_0 => "block_elem_q4_0",
            Self::Q4_K => "block_elem_q4_k",
            Self::Q5_K => "block_elem_q5_k",
            Self::Q6_K => "block_elem_q6_k",
        }
    }

    /// WGSL identifier of the 4-wide bulk dequant helper
    /// (`fn block_elem4_<k>(byte0, state, elem_start) -> vec4<f32>`).
    /// Reads one u32 (4 weight bytes) per call instead of four `b_byte`
    /// calls. Caller must satisfy `elem_start % 4 == 0` and stay within
    /// scheme-specific sub-block alignment (see each fn's body comment).
    pub const fn block_elem4_call(self) -> &'static str {
        match self {
            Self::Q8_0 => "block_elem4_q8_0",
            Self::Q4_0 => "block_elem4_q4_0",
            Self::Q4_K => "block_elem4_q4_k",
            Self::Q5_K => "block_elem4_q5_k",
            Self::Q6_K => "block_elem4_q6_k",
        }
    }

    /// Full WGSL preamble for this scheme: state struct decl,
    /// `block_init_<k>`, `block_elem_<k>`, and `load_b_block_<k>` (the
    /// whole-block helper retained for parity callers). Pasted into the
    /// matmul kernel once when `cfg.weight_dtype = Quant(self)`.
    pub fn load_b_block_fn(self) -> String {
        self.wgsl().to_string()
    }
}

/// CPU-side Q8_0 encoder over bf16 source bytes. Used by the load-time
/// transcode path (`WeightMeta::transcode`): tensors stored bf16 in the
/// file but consumed through the quant matmul path get requantized once
/// at upload.
///
/// Source is row-major `[N, K]` bf16 with `K % 32 == 0`, so 32-element
/// blocks never straddle rows and the tensor encodes as one flat block
/// stream. Output is the GGUF/llama.cpp `block_q8_0` layout verbatim
/// (`f16 d` + `i8 qs[32]` = 34 bytes), exactly what `dequant_i8` and the
/// matmul B-load WGSL consume. Quantization matches llama.cpp
/// `quantize_row_q8_0_ref`: `d = amax / 127`, `q = round(x / d)`.
pub fn encode_q8_0_from_bf16(src: &[u8], dst: &mut [u8]) {
    const SRC_BLOCK: usize = 64; // 32 bf16 elements
    const DST_BLOCK: usize = 34;
    assert!(
        src.len().is_multiple_of(SRC_BLOCK),
        "bf16 source must be whole 32-element blocks"
    );
    let blocks = src.len() / SRC_BLOCK;
    assert_eq!(dst.len(), blocks * DST_BLOCK, "dst must be q8_0-sized");
    for (s, o) in src
        .chunks_exact(SRC_BLOCK)
        .zip(dst.chunks_exact_mut(DST_BLOCK))
    {
        let mut x = [0f32; 32];
        for (xi, p) in x.iter_mut().zip(s.chunks_exact(2)) {
            *xi = half::bf16::from_bits(u16::from_le_bytes([p[0], p[1]])).to_f32();
        }
        let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d == 0.0 { 0.0 } else { 1.0 / d };
        o[..2].copy_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
        for (qo, &v) in o[2..].iter_mut().zip(x.iter()) {
            *qo = (v * id).round() as i8 as u8;
        }
    }
}

/// Common WGSL prelude shared by every scheme: f16-bits to f32 and a
/// byte-extraction helper over the `array<u32>` B buffer. Each scheme's
/// WGSL string starts with this so the same kernel never sees two
/// definitions of the same fn (only one quant scheme is active per
/// pipeline; the strings don't co-exist in a single shader).
const COMMON_PRELUDE: &str = r#"
fn f16_bits_to_f32(h: u32) -> f32 {
    let h_masked: u32 = h & 0xFFFFu;
    let sign: u32 = (h_masked & 0x8000u) << 16u;
    let exp: u32  = (h_masked >> 10u) & 0x1Fu;
    let mant: u32 = h_masked & 0x3FFu;
    if (exp == 0u) {
        if (mant == 0u) {
            return bitcast<f32>(sign);
        }
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
        return bitcast<f32>(sign | 0x7F800000u | (mant << 13u));
    }
    let f_exp: u32 = (exp + 112u) << 23u;
    let f_mant: u32 = mant << 13u;
    return bitcast<f32>(sign | f_exp | f_mant);
}

fn b_byte(byte_offset: u32) -> u32 {
    let wi: u32 = byte_offset >> 2u;
    let sh: u32 = (byte_offset & 3u) * 8u;
    return (b[wi] >> sh) & 0xFFu;
}

// Read a u32 (4 contiguous bytes) starting at `byte_offset`. Tolerates any
// byte alignment: when byte0 is misaligned (Q8_0/Q4_0/Q6_K block strides),
// the 4 bytes straddle two u32 words; we read both and shift-merge.
// vec4 dequant helpers use this to cut byte-by-byte b_byte calls 4x.
fn b_word_at(byte_offset: u32) -> u32 {
    let wi: u32 = byte_offset >> 2u;
    let sh: u32 = (byte_offset & 3u) * 8u;
    let w0: u32 = b[wi];
    if (sh == 0u) { return w0; }
    return (w0 >> sh) | (b[wi + 1u] << (32u - sh));
}

fn sext_i8(x: u32) -> i32 {
    let lo: u32 = x & 0xFFu;
    if ((lo & 0x80u) != 0u) {
        return i32(lo) - 256;
    }
    return i32(lo);
}

fn f16_at(byte0: u32) -> f32 {
    let lo: u32 = b_byte(byte0);
    let hi: u32 = b_byte(byte0 + 1u);
    return f16_bits_to_f32(lo | (hi << 8u));
}
"#;

/// Q8_0: 34 bytes. `f16 d` + `i8 qs[32]`. State carries one f32.
const Q8_0_BODY: &str = r#"
struct BlockState_Q8_0 {
    d: f32,
}

fn block_init_q8_0(byte0: u32) -> BlockState_Q8_0 {
    return BlockState_Q8_0(f16_at(byte0));
}

fn block_elem_q8_0(byte0: u32, st: BlockState_Q8_0, elem: u32) -> f32 {
    let q: i32 = sext_i8(b_byte(byte0 + 2u + elem));
    return f32(q) * st.d;
}

// vec4 dequant: 4 contiguous i8s in one u32 read. `elem_start % 4 == 0`
// required; the matmul cooperative loader satisfies this for ept >= 4.
// Q8_0 block stride is 34 bytes so byte0 may be misaligned -> b_word_at.
fn block_elem4_q8_0(byte0: u32, st: BlockState_Q8_0, elem_start: u32) -> vec4<f32> {
    let word: u32 = b_word_at(byte0 + 2u + elem_start);
    let b0: u32 = word & 0xFFu;
    let b1: u32 = (word >> 8u) & 0xFFu;
    let b2: u32 = (word >> 16u) & 0xFFu;
    let b3: u32 = (word >> 24u) & 0xFFu;
    let q0: i32 = select(i32(b0), i32(b0) - 256, (b0 & 0x80u) != 0u);
    let q1: i32 = select(i32(b1), i32(b1) - 256, (b1 & 0x80u) != 0u);
    let q2: i32 = select(i32(b2), i32(b2) - 256, (b2 & 0x80u) != 0u);
    let q3: i32 = select(i32(b3), i32(b3) - 256, (b3 & 0x80u) != 0u);
    return vec4<f32>(
        f32(q0) * st.d,
        f32(q1) * st.d,
        f32(q2) * st.d,
        f32(q3) * st.d,
    );
}

// Legacy whole-block helper. Kept for parity test callers.
fn load_b_block_q8_0(byte0: u32, dst: ptr<function, array<f32, 32u>>) {
    let st = block_init_q8_0(byte0);
    for (var i: u32 = 0u; i < 32u; i = i + 1u) {
        (*dst)[i] = block_elem_q8_0(byte0, st, i);
    }
}
"#;

/// Q4_0: 18 bytes. `f16 d` + 16 packed-nibble bytes. State carries f32 d.
const Q4_0_BODY: &str = r#"
struct BlockState_Q4_0 {
    d: f32,
}

fn block_init_q4_0(byte0: u32) -> BlockState_Q4_0 {
    return BlockState_Q4_0(f16_at(byte0));
}

fn block_elem_q4_0(byte0: u32, st: BlockState_Q4_0, elem: u32) -> f32 {
    let pair: u32 = elem & 15u;
    let hi: u32 = (elem >> 4u) & 1u;
    let byte: u32 = b_byte(byte0 + 2u + pair);
    let nib: u32 = (byte >> (hi * 4u)) & 0xFu;
    return f32(i32(nib) - 8) * st.d;
}

// vec4 dequant. `elem_start % 4 == 0` and the 4 elements stay within one
// 16-element nibble-half (low for [0,16), high for [16,32)) — guaranteed
// because 16 % 4 == 0 so a 4-aligned start never straddles the boundary.
fn block_elem4_q4_0(byte0: u32, st: BlockState_Q4_0, elem_start: u32) -> vec4<f32> {
    let hi_half: u32 = elem_start >> 4u;
    let pair: u32 = elem_start & 15u;
    let word: u32 = b_word_at(byte0 + 2u + pair);
    let nib_shift: u32 = hi_half * 4u;
    let n0: i32 = i32((word >> nib_shift) & 0xFu) - 8;
    let n1: i32 = i32((word >> (8u + nib_shift)) & 0xFu) - 8;
    let n2: i32 = i32((word >> (16u + nib_shift)) & 0xFu) - 8;
    let n3: i32 = i32((word >> (24u + nib_shift)) & 0xFu) - 8;
    return vec4<f32>(
        f32(n0) * st.d,
        f32(n1) * st.d,
        f32(n2) * st.d,
        f32(n3) * st.d,
    );
}

fn load_b_block_q4_0(byte0: u32, dst: ptr<function, array<f32, 32u>>) {
    let st = block_init_q4_0(byte0);
    for (var i: u32 = 0u; i < 32u; i = i + 1u) {
        (*dst)[i] = block_elem_q4_0(byte0, st, i);
    }
}
"#;

/// Q4_K: 144 bytes.
///   byte 0..2   : f16 d
///   byte 2..4   : f16 dmin
///   byte 4..16  : scales[12] (6-bit sc[0..8], 6-bit m[0..8], packed)
///   byte 16..144: qs[128] (32 nibbles per sub-block; sub_idx = elem>>5,
///                  low/high nibble per element pair within the sub-block).
///
/// llama.cpp scales[12] decode (`get_scale_min_k4`):
///   sc[s], m[s] for s in 0..8:
///     if s < 4:
///       sc = scales[s]   & 0x3F;       m = scales[s+4] & 0x3F;
///     else:
///       sc = (scales[s+4] & 0x0F) | ((scales[s-4] >> 6) << 4);
///       m  = (scales[s+4] >> 4)   | ((scales[s  ] >> 6) << 4);
///
/// State precomputes d*sc[s] and dmin*m[s] for all 8 sub-blocks so the
/// per-element call is a nibble extract + FMA.
const Q4_K_BODY: &str = r#"
struct BlockState_Q4_K {
    dsc: array<f32, 8>,
    dmm: array<f32, 8>,
}

fn block_init_q4_k(byte0: u32) -> BlockState_Q4_K {
    let d: f32 = f16_at(byte0);
    let dmin: f32 = f16_at(byte0 + 2u);
    // Load the 12-byte scales region into 12 u32 byte values.
    var sb: array<u32, 12>;
    for (var i: u32 = 0u; i < 12u; i = i + 1u) {
        sb[i] = b_byte(byte0 + 4u + i);
    }
    var st: BlockState_Q4_K;
    for (var s: u32 = 0u; s < 8u; s = s + 1u) {
        var sc: u32 = 0u;
        var m: u32 = 0u;
        if (s < 4u) {
            sc = sb[s] & 0x3Fu;
            m  = sb[s + 4u] & 0x3Fu;
        } else {
            let s4: u32 = s - 4u;
            sc = (sb[s + 4u] & 0x0Fu) | ((sb[s4] >> 6u) << 4u);
            m  = (sb[s + 4u] >> 4u)   | ((sb[s] >> 6u) << 4u);
        }
        st.dsc[s] = d * f32(sc);
        st.dmm[s] = dmin * f32(m);
    }
    return st;
}

fn block_elem_q4_k(byte0: u32, st: BlockState_Q4_K, elem: u32) -> f32 {
    // sub-block s (32 elems) ranges over elem r in 0..32. Pair of subs
    // (2s, 2s+1) shares 32 bytes of qs: low nibble = 2s, high nibble = 2s+1.
    // Per llama.cpp dequantize_row_q4_K: j-step covers 64 elems, q += 32.
    let sub: u32 = elem >> 5u;
    let r: u32 = elem & 31u;
    let half: u32 = sub & 1u;                  // 0 = low nibble, 1 = high
    let qs_base: u32 = byte0 + 16u + (sub >> 1u) * 32u;
    let byte: u32 = b_byte(qs_base + r);
    let nib: u32 = (byte >> (half * 4u)) & 0xFu;
    return st.dsc[sub] * f32(nib) - st.dmm[sub];
}

// vec4 dequant. `elem_start % 4 == 0` and 4 elements must stay within one
// sub-block (32 elems) — true because the matmul thread chunk is 16-aligned
// in elem space and one sub spans 32, so a 4-elem step never straddles.
// Q4_K block stride is 144 bytes (4-aligned), so b[] direct word load works.
fn block_elem4_q4_k(byte0: u32, st: BlockState_Q4_K, elem_start: u32) -> vec4<f32> {
    let sub: u32 = elem_start >> 5u;
    let r0: u32 = elem_start & 31u;
    let half: u32 = sub & 1u;
    let qs_byte_off: u32 = byte0 + 16u + (sub >> 1u) * 32u + r0;
    let word: u32 = b[qs_byte_off >> 2u];
    let shift: u32 = half * 4u;
    let dsc: f32 = st.dsc[sub];
    let dmm: f32 = st.dmm[sub];
    let n0: f32 = f32((word >> shift) & 0xFu);
    let n1: f32 = f32((word >> (8u + shift)) & 0xFu);
    let n2: f32 = f32((word >> (16u + shift)) & 0xFu);
    let n3: f32 = f32((word >> (24u + shift)) & 0xFu);
    return vec4<f32>(
        dsc * n0 - dmm,
        dsc * n1 - dmm,
        dsc * n2 - dmm,
        dsc * n3 - dmm,
    );
}

fn load_b_block_q4_k(byte0: u32, dst: ptr<function, array<f32, 256u>>) {
    let st = block_init_q4_k(byte0);
    for (var i: u32 = 0u; i < 256u; i = i + 1u) {
        (*dst)[i] = block_elem_q4_k(byte0, st, i);
    }
}
"#;

/// Q5_K: 176 bytes. Layout = Q4_K + 32-byte qh appended at the start of
/// the qs region (llama.cpp: `f16 d, f16 dmin, scales[12], qh[32], qs[128]`).
///
/// Element decode: 4-bit nibble from qs same as Q4_K, then bit-5 from qh.
/// qh layout: qh[32] is 32 bytes where bit `s%8` of byte `s/8 * 4 + (sub & 1)*4`
/// is the high bit for sub-block `sub` ... actually that's wrong, let me
/// follow llama.cpp `dequantize_row_q5_K` directly:
///
/// ```c
/// for (j = 0; j < QK_K; j += 64) {
///     uint8_t * ql = qs + j/2;        // 32 bytes for 64 elems (low nibbles)
///     uint8_t * qh = blk.qh;          // 32 bytes for whole super-block
///     uint8_t mask_h = 1 << is_l;     // is_l = sub_lo bit position
///     ...
/// }
/// ```
///
/// The simpler element-level form (used here): for element `elem` in [0,256),
/// sub = elem>>5, low/high nibble within sub determined by `(elem & 31) >> 4`,
/// pair = (elem & 31) & 15. qs byte = qs[sub*16 + pair]. qh byte = qh[pair +
/// (sub_pair * 16)] ... no. Element-by-element, the high bit lives at
/// `qh[(elem & 31) + (sub >> 1) * 32]`? No — qh is 32 bytes shared across the
/// super-block, one bit per element, indexed `qh[elem & 31] >> sub`.
///
/// Verified against llama.cpp: high bit for elem in sub=s, position r in sub
/// is `(qh[r % 32] >> s) & 1` — no, qh is per-element packed 8 elements per byte
/// across the super-block in column order: bit `s` of `qh[r]` is the high bit
/// for element `s*32 + r` for r in 0..32, s in 0..8.
const Q5_K_BODY: &str = r#"
struct BlockState_Q5_K {
    dsc: array<f32, 8>,
    dmm: array<f32, 8>,
}

fn block_init_q5_k(byte0: u32) -> BlockState_Q5_K {
    let d: f32 = f16_at(byte0);
    let dmin: f32 = f16_at(byte0 + 2u);
    var sb: array<u32, 12>;
    for (var i: u32 = 0u; i < 12u; i = i + 1u) {
        sb[i] = b_byte(byte0 + 4u + i);
    }
    var st: BlockState_Q5_K;
    for (var s: u32 = 0u; s < 8u; s = s + 1u) {
        var sc: u32 = 0u;
        var m: u32 = 0u;
        if (s < 4u) {
            sc = sb[s] & 0x3Fu;
            m  = sb[s + 4u] & 0x3Fu;
        } else {
            let s4: u32 = s - 4u;
            sc = (sb[s + 4u] & 0x0Fu) | ((sb[s4] >> 6u) << 4u);
            m  = (sb[s + 4u] >> 4u)   | ((sb[s] >> 6u) << 4u);
        }
        st.dsc[s] = d * f32(sc);
        st.dmm[s] = dmin * f32(m);
    }
    return st;
}

fn block_elem_q5_k(byte0: u32, st: BlockState_Q5_K, elem: u32) -> f32 {
    // Q5_K layout: f16 d, f16 dmin, scales[12], qh[32], qs[128].
    // qs paired as in Q4_K (sub pair shares 32 bytes); qh is one high
    // bit per element, bit `sub` of qh[r_in_sub].
    let sub: u32 = elem >> 5u;
    let r: u32 = elem & 31u;
    let half: u32 = sub & 1u;
    let qs_base: u32 = byte0 + 48u + (sub >> 1u) * 32u;
    let byte: u32 = b_byte(qs_base + r);
    let nib_lo: u32 = (byte >> (half * 4u)) & 0xFu;
    let qh_byte: u32 = b_byte(byte0 + 16u + r);
    let high_bit: u32 = (qh_byte >> sub) & 1u;
    let nib: u32 = nib_lo | (high_bit << 4u);
    return st.dsc[sub] * f32(nib) - st.dmm[sub];
}

// vec4 dequant. Q5_K block stride is 176 bytes (4-aligned). qs at byte0+48
// and qh at byte0+16 are both 4-aligned; one u32 read each gives 4 nibbles
// + 4 high bits with one storage access per region.
fn block_elem4_q5_k(byte0: u32, st: BlockState_Q5_K, elem_start: u32) -> vec4<f32> {
    let sub: u32 = elem_start >> 5u;
    let r0: u32 = elem_start & 31u;
    let half: u32 = sub & 1u;
    let qs_byte_off: u32 = byte0 + 48u + (sub >> 1u) * 32u + r0;
    let qs_word: u32 = b[qs_byte_off >> 2u];
    let qh_word: u32 = b[(byte0 + 16u + r0) >> 2u];
    let shift: u32 = half * 4u;
    let dsc: f32 = st.dsc[sub];
    let dmm: f32 = st.dmm[sub];
    let lo0: u32 = (qs_word >> shift) & 0xFu;
    let lo1: u32 = (qs_word >> (8u + shift)) & 0xFu;
    let lo2: u32 = (qs_word >> (16u + shift)) & 0xFu;
    let lo3: u32 = (qs_word >> (24u + shift)) & 0xFu;
    let h0: u32 = (qh_word >> sub) & 1u;
    let h1: u32 = (qh_word >> (8u + sub)) & 1u;
    let h2: u32 = (qh_word >> (16u + sub)) & 1u;
    let h3: u32 = (qh_word >> (24u + sub)) & 1u;
    return vec4<f32>(
        dsc * f32(lo0 | (h0 << 4u)) - dmm,
        dsc * f32(lo1 | (h1 << 4u)) - dmm,
        dsc * f32(lo2 | (h2 << 4u)) - dmm,
        dsc * f32(lo3 | (h3 << 4u)) - dmm,
    );
}

fn load_b_block_q5_k(byte0: u32, dst: ptr<function, array<f32, 256u>>) {
    let st = block_init_q5_k(byte0);
    for (var i: u32 = 0u; i < 256u; i = i + 1u) {
        (*dst)[i] = block_elem_q5_k(byte0, st, i);
    }
}
"#;

/// Q6_K: 210 bytes.
///   byte 0..128 : ql[128] (low 4 bits per element)
///   byte 128..192: qh[64] (high 2 bits per element, 4-per-byte)
///   byte 192..208: i8 sc[16] (per 16-elem sub-block scale)
///   byte 208..210: f16 d
///
/// llama.cpp `dequantize_row_q6_K`:
/// ```c
/// for (j = 0; j < 32; ++j) {
///     y[j+ 0] = d * sc[is+0] * ((int8_t)((ql[j+ 0] & 0xF) | ((qh[j] & 0x03) << 4)) - 32);
///     y[j+32] = d * sc[is+2] * ((int8_t)((ql[j+32] & 0xF) | ((qh[j] & 0x0C) << 2)) - 32);
///     y[j+64] = d * sc[is+4] * ((int8_t)((ql[j+ 0] >> 4)  | ((qh[j] & 0x30) << 0)) - 32);
///     y[j+96] = d * sc[is+6] * ((int8_t)((ql[j+32] >> 4)  | ((qh[j] & 0xC0) >> 2)) - 32);
/// }
/// // is steps by 8 every 128 elements.
/// ```
///
/// State precomputes d*sc[is] for is in 0..16. Per-element call extracts
/// the 4+2 bit value and applies the precomputed scale.
const Q6_K_BODY: &str = r#"
struct BlockState_Q6_K {
    dsc: array<f32, 16>,
}

fn block_init_q6_k(byte0: u32) -> BlockState_Q6_K {
    let d: f32 = f16_at(byte0 + 208u);
    var st: BlockState_Q6_K;
    for (var i: u32 = 0u; i < 16u; i = i + 1u) {
        let s: i32 = sext_i8(b_byte(byte0 + 192u + i));
        st.dsc[i] = d * f32(s);
    }
    return st;
}

// vec4 dequant. Q6_K block stride is 210 bytes (NOT 4-aligned), so b_word_at
// handles misaligned reads. 4 contiguous elements share (half, bk, is): bk
// changes at j=32 boundary; elem_start % 4 == 0 keeps j..j+3 within one bk.
// is = j>>4 is constant within {0..15} and {16..31} groups, and elem_start
// at 4-aligned positions inside {0..15} or {16..31} keeps is constant too.
fn block_elem4_q6_k(byte0: u32, st: BlockState_Q6_K, elem_start: u32) -> vec4<f32> {
    let half: u32 = elem_start >> 7u;
    let k: u32 = elem_start & 127u;
    let bk: u32 = k >> 5u;
    let j: u32 = k & 31u;
    let ql_byte_off: u32 = byte0 + half * 64u + (bk & 1u) * 32u + j;
    let ql_word: u32 = b_word_at(ql_byte_off);
    let qh_byte_off: u32 = byte0 + 128u + half * 32u + j;
    let qh_word: u32 = b_word_at(qh_byte_off);
    let nib_shift: u32 = select(4u, 0u, bk < 2u); // bk<2: low nibble; bk>=2: high
    let high2_shift: u32 = bk * 2u;
    let is: u32 = half * 8u + bk * 2u + (j >> 4u);
    let scale: f32 = st.dsc[is];
    let nib0: u32 = (ql_word >> nib_shift) & 0xFu;
    let nib1: u32 = (ql_word >> (8u + nib_shift)) & 0xFu;
    let nib2: u32 = (ql_word >> (16u + nib_shift)) & 0xFu;
    let nib3: u32 = (ql_word >> (24u + nib_shift)) & 0xFu;
    let h0: u32 = (qh_word >> high2_shift) & 0x3u;
    let h1: u32 = (qh_word >> (8u + high2_shift)) & 0x3u;
    let h2: u32 = (qh_word >> (16u + high2_shift)) & 0x3u;
    let h3: u32 = (qh_word >> (24u + high2_shift)) & 0x3u;
    let q0: i32 = i32(nib0 | (h0 << 4u)) - 32;
    let q1: i32 = i32(nib1 | (h1 << 4u)) - 32;
    let q2: i32 = i32(nib2 | (h2 << 4u)) - 32;
    let q3: i32 = i32(nib3 | (h3 << 4u)) - 32;
    return vec4<f32>(
        scale * f32(q0),
        scale * f32(q1),
        scale * f32(q2),
        scale * f32(q3),
    );
}

fn block_elem_q6_k(byte0: u32, st: BlockState_Q6_K, elem: u32) -> f32 {
    // Per llama.cpp dequantize_row_q6_K:
    //   y[l+ 0] = sc[is+0] * d * ((ql[l   ] & 0xF) | ((qh[l]>>0 & 3)<<4) - 32)
    //   y[l+32] = sc[is+2] * d * ((ql[l+32] & 0xF) | ((qh[l]>>2 & 3)<<4) - 32)
    //   y[l+64] = sc[is+4] * d * ((ql[l   ] >> 4)  | ((qh[l]>>4 & 3)<<4) - 32)
    //   y[l+96] = sc[is+6] * d * ((ql[l+32] >> 4)  | ((qh[l]>>6 & 3)<<4) - 32)
    //   is = l / 16 (advances within the 32-elem inner block).
    //   ql += 64, qh += 32, sc += 8 per 128-elem half.
    let half: u32 = elem >> 7u;
    let k: u32 = elem & 127u;
    let bk: u32 = k >> 5u;
    let j: u32 = k & 31u;
    // ql_off: bk=0,2 use ql[j]; bk=1,3 use ql[j+32].
    let ql_byte: u32 = b_byte(byte0 + half * 64u + (bk & 1u) * 32u + j);
    // Nibble half: bk=0,1 use LOW nibble; bk=2,3 use HIGH nibble.
    let nib_lo: u32 = select(ql_byte >> 4u, ql_byte & 0xFu, bk < 2u);
    let qh_byte: u32 = b_byte(byte0 + 128u + half * 32u + j);
    let high2: u32 = (qh_byte >> (bk * 2u)) & 0x3u;
    let q: i32 = i32(nib_lo | (high2 << 4u)) - 32;
    let is: u32 = half * 8u + bk * 2u + (j >> 4u);
    return st.dsc[is] * f32(q);
}

fn load_b_block_q6_k(byte0: u32, dst: ptr<function, array<f32, 256u>>) {
    let st = block_init_q6_k(byte0);
    for (var i: u32 = 0u; i < 256u; i = i + 1u) {
        (*dst)[i] = block_elem_q6_k(byte0, st, i);
    }
}
"#;

use std::sync::OnceLock;

fn cached(slot: &'static OnceLock<String>, body: &str) -> &'static str {
    slot.get_or_init(|| {
        let mut s = String::with_capacity(COMMON_PRELUDE.len() + body.len());
        s.push_str(COMMON_PRELUDE);
        s.push_str(body);
        s
    })
}

impl QuantKind {
    /// Cached, owned WGSL string for this scheme. Same content as
    /// [`load_b_block_fn`]; reading it does not allocate after the first call.
    pub fn wgsl(self) -> &'static str {
        static Q8_0_S: OnceLock<String> = OnceLock::new();
        static Q4_0_S: OnceLock<String> = OnceLock::new();
        static Q4_K_S: OnceLock<String> = OnceLock::new();
        static Q5_K_S: OnceLock<String> = OnceLock::new();
        static Q6_K_S: OnceLock<String> = OnceLock::new();
        match self {
            Self::Q8_0 => cached(&Q8_0_S, Q8_0_BODY),
            Self::Q4_0 => cached(&Q4_0_S, Q4_0_BODY),
            Self::Q4_K => cached(&Q4_K_S, Q4_K_BODY),
            Self::Q5_K => cached(&Q5_K_S, Q5_K_BODY),
            Self::Q6_K => cached(&Q6_K_S, Q6_K_BODY),
        }
    }
}

// =============================================================================
// CPU dequant + quant helpers. Used by parity tests; mirror llama.cpp
// reference implementations bit-for-bit so the WGSL kernels can be
// validated GPU-vs-CPU.
// =============================================================================

/// Dequantize one Q8_0 block (32 elements, 34 bytes).
pub fn dequantize_block_q8_0(src: &[u8], dst: &mut [f32; 32]) {
    assert!(src.len() >= 34, "Q8_0 block needs 34 bytes");
    let scale = half::f16::from_le_bytes([src[0], src[1]]).to_f32();
    for j in 0..32 {
        let q = src[2 + j] as i8;
        dst[j] = (q as f32) * scale;
    }
}

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

pub fn dequantize_block_q4_0(src: &[u8], dst: &mut [f32; 32]) {
    assert!(src.len() >= 18, "Q4_0 block needs 18 bytes");
    let scale = half::f16::from_le_bytes([src[0], src[1]]).to_f32();
    for j in 0..16 {
        let byte = src[2 + j];
        let lo = (byte & 0x0F) as i32 - 8;
        let hi = ((byte >> 4) & 0x0F) as i32 - 8;
        dst[j] = (lo as f32) * scale;
        dst[j + 16] = (hi as f32) * scale;
    }
}

pub fn dequantize_row_q4_0(src: &[u8], dst: &mut [f32]) {
    assert!(
        src.len().is_multiple_of(18),
        "Q4_0 buffer must be a whole number of 18-byte blocks"
    );
    let n_blocks = src.len() / 18;
    assert_eq!(dst.len(), n_blocks * 32);
    for i in 0..n_blocks {
        let s = i * 18;
        let mut blk = [0f32; 32];
        dequantize_block_q4_0(&src[s..s + 18], &mut blk);
        dst[i * 32..(i + 1) * 32].copy_from_slice(&blk);
    }
}

pub fn quantize_row_q4_0(src: &[f32], dst: &mut Vec<u8>) {
    assert!(src.len().is_multiple_of(32));
    let n_blocks = src.len() / 32;
    dst.clear();
    dst.reserve(n_blocks * 18);
    for i in 0..n_blocks {
        let block = &src[i * 32..(i + 1) * 32];
        let mut amax = 0f32;
        let mut signed_max = 0f32;
        for &v in block {
            if v.abs() > amax {
                amax = v.abs();
                signed_max = v;
            }
        }
        let d = signed_max / -8.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let d_h = half::f16::from_f32(d);
        dst.extend_from_slice(&d_h.to_le_bytes());
        for j in 0..16 {
            let x0 = block[j] * id;
            let x1 = block[j + 16] * id;
            let xi0 = ((x0 + 8.5).floor() as i32).clamp(0, 15) as u8;
            let xi1 = ((x1 + 8.5).floor() as i32).clamp(0, 15) as u8;
            dst.push(xi0 | (xi1 << 4));
        }
    }
}

/// llama.cpp `get_scale_min_k4` port. Reads 12-byte packed scales/mins
/// and produces 8 6-bit scale codes + 8 6-bit min codes.
fn q_k_get_scale_min(scales: &[u8; 12]) -> ([u8; 8], [u8; 8]) {
    let mut sc = [0u8; 8];
    let mut m = [0u8; 8];
    for s in 0..4 {
        sc[s] = scales[s] & 0x3F;
        m[s] = scales[s + 4] & 0x3F;
    }
    for s in 4..8 {
        let s4 = s - 4;
        sc[s] = (scales[s + 4] & 0x0F) | ((scales[s4] >> 6) << 4);
        m[s] = (scales[s + 4] >> 4) | ((scales[s] >> 6) << 4);
    }
    (sc, m)
}

pub fn dequantize_block_q4_k(src: &[u8], dst: &mut [f32; 256]) {
    assert!(src.len() >= 144, "Q4_K block needs 144 bytes");
    let d = half::f16::from_le_bytes([src[0], src[1]]).to_f32();
    let dmin = half::f16::from_le_bytes([src[2], src[3]]).to_f32();
    let mut scales12 = [0u8; 12];
    scales12.copy_from_slice(&src[4..16]);
    let (sc, m) = q_k_get_scale_min(&scales12);
    let qs = &src[16..144];
    for sub in 0..8usize {
        let dsc = d * sc[sub] as f32;
        let dmm = dmin * m[sub] as f32;
        let half = sub & 1; // 0 = low nibble, 1 = high
        let qs_base = (sub >> 1) * 32;
        for r in 0..32 {
            let byte = qs[qs_base + r];
            let nib = if half == 0 { byte & 0x0F } else { byte >> 4 };
            dst[sub * 32 + r] = dsc * nib as f32 - dmm;
        }
    }
}

pub fn dequantize_row_q4_k(src: &[u8], dst: &mut [f32]) {
    assert!(src.len().is_multiple_of(144));
    let n_blocks = src.len() / 144;
    assert_eq!(dst.len(), n_blocks * 256);
    for i in 0..n_blocks {
        let s = i * 144;
        let mut blk = [0f32; 256];
        dequantize_block_q4_k(&src[s..s + 144], &mut blk);
        dst[i * 256..(i + 1) * 256].copy_from_slice(&blk);
    }
}

pub fn dequantize_block_q5_k(src: &[u8], dst: &mut [f32; 256]) {
    assert!(src.len() >= 176, "Q5_K block needs 176 bytes");
    let d = half::f16::from_le_bytes([src[0], src[1]]).to_f32();
    let dmin = half::f16::from_le_bytes([src[2], src[3]]).to_f32();
    let mut scales12 = [0u8; 12];
    scales12.copy_from_slice(&src[4..16]);
    let (sc, m) = q_k_get_scale_min(&scales12);
    let qh = &src[16..48];
    let qs = &src[48..176];
    for sub in 0..8usize {
        let dsc = d * sc[sub] as f32;
        let dmm = dmin * m[sub] as f32;
        let half = sub & 1;
        let qs_base = (sub >> 1) * 32;
        for r in 0..32 {
            let byte = qs[qs_base + r];
            let lo_nib = if half == 0 { byte & 0x0F } else { byte >> 4 };
            let high_bit = ((qh[r] >> sub) & 1) as u32;
            let nib = lo_nib as u32 | (high_bit << 4);
            dst[sub * 32 + r] = dsc * nib as f32 - dmm;
        }
    }
}

pub fn dequantize_row_q5_k(src: &[u8], dst: &mut [f32]) {
    assert!(src.len().is_multiple_of(176));
    let n_blocks = src.len() / 176;
    assert_eq!(dst.len(), n_blocks * 256);
    for i in 0..n_blocks {
        let s = i * 176;
        let mut blk = [0f32; 256];
        dequantize_block_q5_k(&src[s..s + 176], &mut blk);
        dst[i * 256..(i + 1) * 256].copy_from_slice(&blk);
    }
}

pub fn dequantize_block_q6_k(src: &[u8], dst: &mut [f32; 256]) {
    assert!(src.len() >= 210, "Q6_K block needs 210 bytes");
    let ql = &src[0..128];
    let qh = &src[128..192];
    let sc = &src[192..208];
    let d = half::f16::from_le_bytes([src[208], src[209]]).to_f32();
    // Port of llama.cpp `dequantize_row_q6_K` inner loop. `is = l/16`
    // shifts the sc index inside each 32-elem inner block.
    for half in 0..2usize {
        let ql_off = half * 64;
        let qh_off = half * 32;
        let is_base = half * 8;
        for j in 0..32usize {
            let is = j >> 4; // 0 for j<16, 1 for j>=16
            let ql0 = ql[ql_off + j] as u32;
            let ql1 = ql[ql_off + 32 + j] as u32;
            let qhb = qh[qh_off + j] as u32;
            let q0 = ((ql0 & 0x0F) | ((qhb & 0x03) << 4)) as i32 - 32;
            let q1 = ((ql1 & 0x0F) | ((qhb & 0x0C) << 2)) as i32 - 32;
            let q2 = ((ql0 >> 4) | (qhb & 0x30)) as i32 - 32;
            let q3 = ((ql1 >> 4) | ((qhb & 0xC0) >> 2)) as i32 - 32;
            let base = half * 128;
            dst[base + j] = d * (sc[is_base + is] as i8 as f32) * q0 as f32;
            dst[base + 32 + j] = d * (sc[is_base + is + 2] as i8 as f32) * q1 as f32;
            dst[base + 64 + j] = d * (sc[is_base + is + 4] as i8 as f32) * q2 as f32;
            dst[base + 96 + j] = d * (sc[is_base + is + 6] as i8 as f32) * q3 as f32;
        }
    }
}

pub fn dequantize_row_q6_k(src: &[u8], dst: &mut [f32]) {
    assert!(src.len().is_multiple_of(210));
    let n_blocks = src.len() / 210;
    assert_eq!(dst.len(), n_blocks * 256);
    for i in 0..n_blocks {
        let s = i * 210;
        let mut blk = [0f32; 256];
        dequantize_block_q6_k(&src[s..s + 210], &mut blk);
        dst[i * 256..(i + 1) * 256].copy_from_slice(&blk);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q8_0_layout() {
        assert_eq!(QuantKind::Q8_0.block_size(), 32);
        assert_eq!(QuantKind::Q8_0.bytes_per_block(), 34);
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
    fn q5_k_layout() {
        assert_eq!(QuantKind::Q5_K.block_size(), 256);
        assert_eq!(QuantKind::Q5_K.bytes_per_block(), 176);
    }

    #[test]
    fn q6_k_layout() {
        assert_eq!(QuantKind::Q6_K.block_size(), 256);
        assert_eq!(QuantKind::Q6_K.bytes_per_block(), 210);
    }

    #[test]
    fn hints_distinct() {
        let mut s = std::collections::HashSet::new();
        for k in [
            QuantKind::Q8_0,
            QuantKind::Q4_0,
            QuantKind::Q4_K,
            QuantKind::Q5_K,
            QuantKind::Q6_K,
        ] {
            assert!(s.insert(k.hint()));
        }
    }

    #[test]
    fn wgsl_strings_have_required_symbols() {
        for k in [
            QuantKind::Q8_0,
            QuantKind::Q4_0,
            QuantKind::Q4_K,
            QuantKind::Q5_K,
            QuantKind::Q6_K,
        ] {
            let w = k.wgsl();
            assert!(w.contains(k.block_state_call()), "{:?} init fn", k);
            assert!(w.contains(k.block_elem_call()), "{:?} elem fn", k);
            assert!(w.contains(k.load_b_block_call()), "{:?} load_b fn", k);
            assert!(w.contains("f16_bits_to_f32"));
        }
    }

    fn approx(a: f32, b: f32, tol: f32) {
        assert!(
            (a - b).abs() <= tol,
            "approx fail: a={a} b={b} diff={}",
            (a - b).abs()
        );
    }

    /// Hand-built Q4_K block: d=1.0, dmin=0.0. sc[0..4]=1, sc[4..8]=0,
    /// m[0..8]=0. Sub 0 reads low nibbles of qs[0..32]; sub 1 reads high
    /// nibbles of qs[0..32]. Pack qs[r] = (r & 0xF) | ((15 - (r & 0xF)) << 4)
    /// so element r of sub 0 = r&0xF, element r of sub 1 = 15 - r&0xF.
    #[test]
    fn dequant_q4_k_handcrafted() {
        let mut blk = vec![0u8; 144];
        blk[0] = 0x00;
        blk[1] = 0x3C; // d=1.0
        blk[2] = 0x00;
        blk[3] = 0x00; // dmin=0.0
        for i in 0..4 {
            blk[4 + i] = 1;
        }
        // qs[0..32] for sub-pair (0,1). Other sub-pairs zero.
        for r in 0..32 {
            let lo = (r & 0xF) as u8;
            let hi = ((15 - (r & 0xF)) & 0xF) as u8;
            blk[16 + r] = lo | (hi << 4);
        }
        let mut out = [0f32; 256];
        dequantize_block_q4_k(&blk, &mut out);
        for r in 0..32 {
            assert_eq!(out[r], (r & 0xF) as f32, "sub0 elem {r}");
            assert_eq!(out[32 + r], (15 - (r & 0xF)) as f32, "sub1 elem {r}");
        }
        // Subs 2..3 still sc=1 but qs[32..64] is zero -> dequant zero.
        for (e, v) in out.iter().enumerate().take(128).skip(64) {
            assert_eq!(*v, 0.0, "subs 2..3 zero at {e}");
        }
        // Subs 4..7 have sc=0 -> all zero.
        for (e, v) in out.iter().enumerate().skip(128) {
            assert_eq!(*v, 0.0, "subs 4..7 zero at {e}");
        }
    }

    /// Q5_K with d=1.0, dmin=0.0, sc[0]=1: element 0 lives in sub=0,
    /// uses low nibble of qs[0]=1, high bit = bit 0 of qh[0]=1.
    /// Combined nibble = 1 | (1 << 4) = 17.
    #[test]
    fn dequant_q5_k_high_bit() {
        let mut blk = vec![0u8; 176];
        blk[0] = 0x00;
        blk[1] = 0x3C; // d=1.0
        blk[2] = 0x00;
        blk[3] = 0x00; // dmin=0.0
        blk[4] = 1; // sc[0]=1
        blk[16] = 1; // qh[0] bit 0 set -> high bit for elem 0
        blk[48] = 1; // qs[0] = 1
        let mut out = [0f32; 256];
        dequantize_block_q5_k(&blk, &mut out);
        approx(out[0], 17.0, 1e-6);
    }

    /// Q6_K with d=1.0, sc[0]=1: element 0 -> ql[0] low nibble | (qh[0] low2 << 4) - 32.
    /// Set ql[0]=3, qh[0]=0 -> q = 3 - 32 = -29. Result = 1.0 * 1 * -29 = -29.
    #[test]
    fn dequant_q6_k_handcrafted() {
        let mut blk = vec![0u8; 210];
        blk[208] = 0x00;
        blk[209] = 0x3C; // d=1.0
        blk[192] = 1; // sc[0]=1
        blk[0] = 3; // ql[0]=3
        // qh[0]=0 -> high bits 0, q = 3 - 32 = -29
        let mut out = [0f32; 256];
        dequantize_block_q6_k(&blk, &mut out);
        approx(out[0], -29.0, 1e-6);
    }

    /// encode_q8_0_from_bf16 -> dequantize_block_q8_0 roundtrip: error per
    /// element bounded by half a quant step (d/2) plus the f16 rounding of
    /// the stored scale (up to 127 * d * 2^-11, since q is computed against
    /// the f32 scale exactly as llama.cpp's quantize_row_q8_0_ref does).
    #[test]
    fn encode_q8_0_roundtrip() {
        let vals: Vec<f32> = (0..64).map(|i| ((i as f32) - 31.5) * 0.37).collect();
        let mut src = Vec::with_capacity(vals.len() * 2);
        for &v in &vals {
            src.extend_from_slice(&half::bf16::from_f32(v).to_bits().to_le_bytes());
        }
        let mut q = vec![0u8; 2 * 34];
        encode_q8_0_from_bf16(&src, &mut q);
        for (bi, blk) in q.chunks_exact(34).enumerate() {
            let mut out = [0f32; 32];
            dequantize_block_q8_0(blk, &mut out);
            let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
            for (i, &got) in out.iter().enumerate() {
                let want = half::bf16::from_f32(vals[bi * 32 + i]).to_f32();
                assert!(
                    (got - want).abs() <= d * (0.5 + 127.0 / 2048.0) + 1e-6,
                    "block {bi} elem {i}: got {got}, want {want}, d {d}"
                );
            }
        }
    }

    /// All-zero input encodes to d=0, q=0 (no NaN from 0/0).
    #[test]
    fn encode_q8_0_zero_block() {
        let src = vec![0u8; 64];
        let mut q = vec![0u8; 34];
        encode_q8_0_from_bf16(&src, &mut q);
        assert!(q.iter().all(|&b| b == 0));
    }
}
