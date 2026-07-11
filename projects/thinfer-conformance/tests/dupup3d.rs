//! `thinfer_core::ops::dupup3d::DupUp3dF32` (Wan2.2 residual VAE up-shortcut)
//! vs an independently-written CPU gather reference. The worklog flagged the
//! op as built-but-unverified (the channel `repeat_interleave` + t/h/w
//! duplicate-reshape + `first_chunk` temporal trim are the risk). No python
//! fixture: the op is a parameter-free gather, so the reference is exact.
//!
//! Inputs encode their flat source index (`x[i] = i`), so each output cell's
//! value directly names which input cell it gathered; a wrong index math
//! surfaces as a specific off-by gather, not just a magnitude drift.

#![cfg(feature = "conformance")]

mod i8_common;

use i8_common::{alloc_with, alloc_zero};
use thinfer_core::backend::{Backend, WgpuBackend};
use thinfer_core::ops::{ActDtype, DupUp3dF32, DupUp3dOp, WeightDtype, WgslConfig};

/// One DupUp3D geometry. Mirrors the `U` struct field order in the kernel.
struct Case {
    in_c: u32,
    out_c: u32,
    ft: u32,
    fs: u32,
    t_in: u32,
    h_in: u32,
    w_in: u32,
    repeats: u32,
    t_drop: u32,
}

impl Case {
    fn t_out(&self) -> u32 {
        self.t_in * self.ft - self.t_drop
    }
    fn h_out(&self) -> u32 {
        self.h_in * self.fs
    }
    fn w_out(&self) -> u32 {
        self.w_in * self.fs
    }
    fn out_elems(&self) -> usize {
        (self.out_c * self.t_out() * self.h_out() * self.w_out()) as usize
    }
    fn in_elems(&self) -> usize {
        (self.in_c * self.t_in * self.h_in * self.w_in) as usize
    }

    /// 48-byte uniform: 12 u32 (9 fields + 3 pad), matching `dupup3d_uniform`.
    fn uniform_bytes(&self) -> [u8; 48] {
        let fields: [u32; 12] = [
            self.in_c,
            self.out_c,
            self.ft,
            self.fs,
            self.t_in,
            self.h_in,
            self.w_in,
            self.repeats,
            self.t_drop,
            0,
            0,
            0,
        ];
        let mut bytes = [0u8; 48];
        for (i, v) in fields.iter().enumerate() {
            bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        bytes
    }

    /// Independent CPU reference for the documented gather. For output
    /// `(oc, T, H, W)`: undo the temporal trim (`tf = T + t_drop`), split each
    /// axis into base/offset, fold the offsets into the expanded channel index
    /// `e`, invert the `repeat_interleave` (`ic = e / repeats`), then read the
    /// source cell `x[ic, t, h, w]`.
    fn reference(&self, x: &[f32]) -> Vec<f32> {
        let (ft, fs) = (self.ft, self.fs);
        let (t_in, h_in, w_in) = (self.t_in, self.h_in, self.w_in);
        let (t_out, h_out, w_out) = (self.t_out(), self.h_out(), self.w_out());
        let mut out = vec![0f32; self.out_elems()];
        for oc in 0..self.out_c {
            for to in 0..t_out {
                for ho in 0..h_out {
                    for wo in 0..w_out {
                        let tf = to + self.t_drop;
                        let (t, a) = (tf / ft, tf % ft);
                        let (h, b) = (ho / fs, ho % fs);
                        let (w, c) = (wo / fs, wo % fs);
                        let e = ((oc * ft + a) * fs + b) * fs + c;
                        let ic = e / self.repeats;
                        let in_idx = ic * (t_in * h_in * w_in) + t * (h_in * w_in) + h * w_in + w;
                        let out_idx = ((oc * t_out + to) * h_out + ho) * w_out + wo;
                        out[out_idx as usize] = x[in_idx as usize];
                    }
                }
            }
        }
        out
    }
}

async fn run(case: &Case) -> (Vec<f32>, Vec<f32>) {
    let backend = WgpuBackend::new()
        .await
        .expect("wgpu adapter unavailable for tests");

    // x[i] = i: the value of each output cell is the flat index it gathered.
    let x: Vec<f32> = (0..case.in_elems()).map(|i| i as f32).collect();
    let exp = case.reference(&x);

    let x_bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
    let x_buf = alloc_with(&backend, &x_bytes);
    let u_buf = alloc_with(&backend, &case.uniform_bytes());
    let out_buf = alloc_zero(&backend, (case.out_elems() * 4) as u64);

    let cfg = WgslConfig {
        bf16_quant_writes: false,
        act_dtype: ActDtype::F32,
        weight_dtype: WeightDtype::Bf16,
    };
    let pipeline = backend
        .create_pipeline(
            "dupup3d_conf",
            <DupUp3dF32 as DupUp3dOp>::wgsl(&cfg),
            "main",
            <DupUp3dF32 as DupUp3dOp>::layout(),
        )
        .await
        .expect("pipeline");

    let mut enc = backend.create_command_encoder();
    // Layout: 0=X, 1=Out, 2=Uniform.
    let bindings = [x_buf.binding(0), out_buf.binding(1), u_buf.binding(2)];
    backend
        .dispatch(
            &mut enc,
            &pipeline,
            &bindings,
            <DupUp3dF32 as DupUp3dOp>::workgroups(case.out_elems() as u32),
        )
        .expect("dispatch");
    backend.submit(enc).await.expect("submit");

    let out_bytes = backend
        .read_buffer(out_buf.id, out_buf.offset, out_buf.len)
        .await
        .expect("read out");
    let got: Vec<f32> = out_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    for buf in [x_buf, u_buf, out_buf] {
        backend.free(buf.id);
    }
    (got, exp)
}

fn assert_exact(case: &Case, label: &str) {
    let (got, exp) = pollster::block_on(run(case));
    assert_eq!(got.len(), exp.len(), "{label}: length mismatch");
    // Pure gather of exactly-representable integers -> bit-exact, no tolerance.
    if let Some((i, (g, e))) = got.iter().zip(&exp).enumerate().find(|(_, (g, e))| g != e) {
        panic!("{label}: mismatch at out[{i}]: got {g} exp {e}");
    }
}

#[test]
fn dupup3d_residual_shortcut() {
    // The real residual-up shape: 2x temporal + 2x spatial upsample with a
    // channel `repeat_interleave(2)` and the `first_chunk` temporal trim
    // (t_drop = ft - 1 = 1). out_c*ft*fs*fs = 16 expanded channels / 2 repeats
    // = 8 = in_c.
    assert_exact(
        &Case {
            in_c: 8,
            out_c: 2,
            ft: 2,
            fs: 2,
            t_in: 3,
            h_in: 2,
            w_in: 2,
            repeats: 2,
            t_drop: 1,
        },
        "residual_shortcut",
    );
}

#[test]
fn dupup3d_identity_copy() {
    // Degenerate: no upsample (ft=fs=1), no channel expansion (repeats=1), no
    // temporal trim -> a pure copy. Catches a stray index term that the
    // upsampling case would mask.
    let case = Case {
        in_c: 4,
        out_c: 4,
        ft: 1,
        fs: 1,
        t_in: 2,
        h_in: 3,
        w_in: 3,
        repeats: 1,
        t_drop: 0,
    };
    let (got, _) = pollster::block_on(run(&case));
    let expect: Vec<f32> = (0..case.in_elems()).map(|i| i as f32).collect();
    assert_eq!(
        got, expect,
        "identity_copy: output must equal input verbatim"
    );
}
