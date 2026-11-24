//! The device top-k sampler (`LlmOp::SampleTopK`) picks what the host
//! `Sampler` picks from the same logits and the same draw stream (top-p 1, no
//! penalties), and argmax at temperature 0. Skips without a CUDA device.
#![cfg(not(target_os = "macos"))]

use std::sync::Arc;

use cudarc::driver::{CudaContext, sys};
use thinfer_core::ops::llm::LlmOp;
use thinfer_cuda::llm::LlmKernels;
use thinfer_models::llm::sampler::{Sampler, SamplingParams};

fn upload(ctx: &Arc<CudaContext>, bytes: &[u8]) -> sys::CUdeviceptr {
    ctx.bind_to_thread().unwrap();
    let mut p = 0;
    unsafe {
        sys::cuMemAlloc_v2(&mut p, bytes.len().max(16))
            .result()
            .unwrap();
        sys::cuMemcpyHtoD_v2(p, bytes.as_ptr().cast(), bytes.len())
            .result()
            .unwrap();
    }
    p
}

#[test]
fn device_sampler_matches_host() {
    if !thinfer_cuda::driver_present() {
        eprintln!("skipped[cuda sample]: no CUDA driver");
        return;
    }
    let Ok(ctx) = CudaContext::new(0) else {
        eprintln!("skipped[cuda sample]: no device");
        return;
    };
    let sm = ctx.compute_capability().unwrap();
    let Ok(k) = LlmKernels::new(ctx.clone(), sm) else {
        eprintln!("skipped[cuda sample]: no kernels");
        return;
    };
    let n = 2048usize;
    let mut state = 42u64;
    let mut rnd = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
    };
    for (temp, top_k) in [(0.9f32, 50u32), (1.0, 1), (0.7, 64), (0.0, 50)] {
        let params = SamplingParams {
            temperature: temp,
            top_k,
            top_p: 1.0,
            min_p: 0.0,
            presence_penalty: 0.0,
            repetition_penalty: 1.0,
            seed: 7,
        };
        let mut host = Sampler::new(params);
        let mut draws = Sampler::new(params);
        let mut same = 0;
        let trials = 200;
        for _ in 0..trials {
            let logits: Vec<f32> = (0..n).map(|_| rnd() * 6.0).collect();
            let want = host.sample(&mut logits.clone());
            let u = if temp > 0.0 { draws.uniform() } else { 0.0 };
            let pl = upload(&ctx, bytemuck::cast_slice(&logits));
            let pu = upload(&ctx, bytemuck::bytes_of(&u));
            let po = upload(&ctx, &[0xffu8; 4]);
            let ran = k
                .run(&LlmOp::SampleTopK {
                    logits: pl,
                    n: n as u32,
                    k: top_k,
                    temperature: temp,
                    uniform: pu,
                    out: po,
                })
                .unwrap();
            assert!(ran);
            let mut got = 0u32;
            unsafe {
                sys::cuCtxSynchronize().result().unwrap();
                sys::cuMemcpyDtoH_v2((&raw mut got).cast(), po, 4)
                    .result()
                    .unwrap();
                for p in [pl, pu, po] {
                    sys::cuMemFree_v2(p).result().unwrap();
                }
            }
            same += (got == want) as u32;
        }
        eprintln!("temp {temp} top_k {top_k}: {same}/{trials} identical picks");
        // expf may differ from the host's by an ulp: a draw landing on a
        // boundary can pick the neighbour, nothing more.
        assert!(
            same >= trials - 2,
            "temp {temp} top_k {top_k}: {same}/{trials}"
        );
    }
}
