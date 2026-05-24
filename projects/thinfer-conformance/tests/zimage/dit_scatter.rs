//! Smoke test for `dit::scatter_pad_rows`. Worklog flagged it unverified;
//! this confirms masked rows are overwritten with the pad-token row and
//! unmasked rows are preserved bit-exactly. No python fixture.
//!
//! Pad token is uploaded as bf16-packed (two halves per u32), matching the
//! residency-managed GPU storage layout. The kernel decodes bf16 -> fp32 on
//! the fly, so the comparison is against the fp32 round-trip of the test
//! values (rounded to bf16 then re-expanded).

#![cfg(feature = "conformance")]

use std::sync::Arc;

use thinfer_core::arbiter::MemArbiter;
use thinfer_core::backend::{Backend, BufRef, WgpuBackend, WgpuError};
use thinfer_core::ops::WgslConfig;
use thinfer_core::workspace::Workspace;
use thinfer_models::z_image::block::{ActBufRef, BlockPipelines, BlockWgslConfigs};
use thinfer_models::z_image::dit::scatter_pad_rows;

const N_ROWS: usize = 5;
const ROW_ELEMS: usize = 4;

#[test]
fn scatter_pad_rows_replaces_masked_and_preserves_others() {
    let backend = Arc::new(
        pollster::block_on(WgpuBackend::new()).expect("wgpu adapter unavailable for tests"),
    );
    let workspace = Workspace::new(backend.clone(), MemArbiter::unlimited());
    let pipelines = pollster::block_on(BlockPipelines::compile(
        &backend,
        &BlockWgslConfigs::uniform(WgslConfig {
            bf16_quant_writes: false,
            act_dtype: thinfer_core::ops::ActDtype::F32,
            weight_dtype: thinfer_core::ops::WeightDtype::Bf16,
        }),
    ))
    .expect("compile block pipelines");

    // dst = row_index * 10 + col, so each value is uniquely identifiable.
    let mut dst_init = vec![0f32; N_ROWS * ROW_ELEMS];
    for r in 0..N_ROWS {
        for c in 0..ROW_ELEMS {
            dst_init[r * ROW_ELEMS + c] = (r * 10 + c) as f32;
        }
    }
    let dst_bytes = bytes(&dst_init);
    let dst_id = backend
        .allocate(dst_bytes.len() as u64)
        .expect("allocate dst");
    backend
        .write_buffer(dst_id, 0, &dst_bytes)
        .expect("write dst");
    let dst = BufRef::new(dst_id, dst_bytes.len() as u64);

    // pad_token: distinct sentinel values uploaded as bf16-packed bytes.
    let pad_token_vals = [-1.0_f32, -2.0, -3.0, -4.0];
    let pad_bf16_bytes = bf16_packed_bytes(&pad_token_vals);
    let pad_id = backend
        .allocate(pad_bf16_bytes.len() as u64)
        .expect("allocate pad");
    backend
        .write_buffer(pad_id, 0, &pad_bf16_bytes)
        .expect("write pad");
    let pad_token = BufRef::new(pad_id, pad_bf16_bytes.len() as u64);

    // Mask rows 1 and 3.
    let mask = [0u8, 1, 0, 1, 0];

    run(&backend, &pipelines, &workspace, dst, pad_token, &mask).expect("scatter");

    let got_bytes = pollster::block_on(backend.read_buffer(dst_id, 0, dst_bytes.len() as u64))
        .expect("readback");
    let got = floats(&got_bytes);

    let pad_decoded = bf16_round_trip(&pad_token_vals);
    let mut expected = dst_init.clone();
    for (r, &m) in mask.iter().enumerate() {
        if m != 0 {
            expected[r * ROW_ELEMS..(r + 1) * ROW_ELEMS].copy_from_slice(&pad_decoded);
        }
    }
    assert_eq!(got, expected, "scatter mismatch");

    backend.free(dst_id);
    backend.free(pad_id);
}

#[test]
fn scatter_pad_rows_no_op_when_mask_all_zero() {
    let backend = Arc::new(
        pollster::block_on(WgpuBackend::new()).expect("wgpu adapter unavailable for tests"),
    );
    let workspace = Workspace::new(backend.clone(), MemArbiter::unlimited());
    let pipelines = pollster::block_on(BlockPipelines::compile(
        &backend,
        &BlockWgslConfigs::uniform(WgslConfig {
            bf16_quant_writes: false,
            act_dtype: thinfer_core::ops::ActDtype::F32,
            weight_dtype: thinfer_core::ops::WeightDtype::Bf16,
        }),
    ))
    .expect("compile block pipelines");

    let init: Vec<f32> = (0..(N_ROWS * ROW_ELEMS)).map(|i| i as f32).collect();
    let init_bytes = bytes(&init);
    let dst_id = backend
        .allocate(init_bytes.len() as u64)
        .expect("allocate dst");
    backend
        .write_buffer(dst_id, 0, &init_bytes)
        .expect("write dst");
    let dst = BufRef::new(dst_id, init_bytes.len() as u64);

    let pad_vals = [9.9_f32; ROW_ELEMS];
    let pad_bf16_bytes = bf16_packed_bytes(&pad_vals);
    let pad_id = backend
        .allocate(pad_bf16_bytes.len() as u64)
        .expect("allocate pad");
    backend
        .write_buffer(pad_id, 0, &pad_bf16_bytes)
        .expect("write pad");
    let pad_token = BufRef::new(pad_id, pad_bf16_bytes.len() as u64);

    run(
        &backend,
        &pipelines,
        &workspace,
        dst,
        pad_token,
        &[0u8; N_ROWS],
    )
    .expect("scatter");

    let got_bytes = pollster::block_on(backend.read_buffer(dst_id, 0, init_bytes.len() as u64))
        .expect("readback");
    assert_eq!(floats(&got_bytes), init, "no-mask path mutated dst");

    backend.free(dst_id);
    backend.free(pad_id);
}

fn run(
    backend: &WgpuBackend,
    pipelines: &BlockPipelines,
    workspace: &Workspace<WgpuBackend>,
    dst: BufRef,
    pad_token: BufRef,
    mask: &[u8],
) -> Result<(), WgpuError> {
    pollster::block_on(scatter_pad_rows(
        backend,
        pipelines,
        workspace,
        ActBufRef::dense(dst),
        pad_token,
        N_ROWS as u32,
        ROW_ELEMS as u32,
        mask,
    ))
}

fn bytes(vs: &[f32]) -> Vec<u8> {
    let mut out = vec![0u8; vs.len() * 4];
    for (i, v) in vs.iter().enumerate() {
        out[i * 4..(i + 1) * 4].copy_from_slice(&v.to_le_bytes());
    }
    out
}

fn floats(bs: &[u8]) -> Vec<f32> {
    bs.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Truncate-toward-zero bf16 of `f`: take the upper 16 bits of the f32 bit
/// pattern. Sufficient for these tiny integer-valued test inputs which are
/// exactly representable in bf16.
fn bf16_truncate(f: f32) -> u16 {
    (f.to_bits() >> 16) as u16
}

fn bf16_packed_bytes(vs: &[f32]) -> Vec<u8> {
    let halves: Vec<u16> = vs.iter().copied().map(bf16_truncate).collect();
    let n_words = halves.len().div_ceil(2);
    let mut out = vec![0u8; n_words * 4];
    for (i, &h) in halves.iter().enumerate() {
        let word_idx = i >> 1;
        let lane = i & 1;
        let off = word_idx * 4 + lane * 2;
        out[off..off + 2].copy_from_slice(&h.to_le_bytes());
    }
    out
}

fn bf16_round_trip(vs: &[f32]) -> Vec<f32> {
    vs.iter()
        .map(|&v| f32::from_bits((bf16_truncate(v) as u32) << 16))
        .collect()
}
