//! MiniMax-H3 AdaLN precompute health check.
//!
//! The precompute is the load-time step that makes H3 fit at all: it replaces
//! ~7 GiB of `adaln_proj` weights with a ~48 MB table, dropping the streamed DiT
//! from ~18.5 GiB to ~11.5 GiB per step. It is also where the checkpoint's
//! superblock reshape bites (`[1016064, 256]` on disk, `[96768, 2688]`
//! logically), so this drives the real GGUF and checks the table is the right
//! shape, finite, and actually varies with the timestep.
//!
//! No GPU: the precompute is host-side by construction.

#![cfg(feature = "minimax-e2e")]

use thinfer_core::format::gguf;
use thinfer_models::minimax::DitConfig;
use thinfer_models::minimax::adaln::PARAMS_PER_BLOCK;
use thinfer_models::minimax::loader::build_modulation_tables;
use thinfer_models::minimax::manifest::{self, role};
use thinfer_native::MmapFileOpener;
use thinfer_native::cache;

/// Four turbo steps present five distinct levels once video and audio
/// schedules are unioned; two is enough to prove the table varies.
const TIMESTEPS: [f32; 3] = [0.0, 0.5, 1.0];

#[tokio::test(flavor = "current_thread")]
async fn adaln_precompute() {
    let _trace = thinfer_core::trace::init_from_env();

    let fr = manifest::MANIFEST
        .get(role::DIT_GGUF_Q4_K_M)
        .expect("dit role");
    let Some(path) = cache::resolve(fr) else {
        eprintln!("skipped[minimax adaln_precompute]: DiT GGUF not in HF cache");
        return;
    };

    let cfg = DitConfig::h3_33b();
    let opener = MmapFileOpener::new(&path)
        .await
        .unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
    let src = gguf::GgufSource::open(opener)
        .await
        .expect("parse dit gguf");

    let started = std::time::Instant::now();
    let tables = build_modulation_tables(&cfg, &src, &TIMESTEPS)
        .await
        .expect("build modulation tables");
    let elapsed = started.elapsed().as_secs_f64();

    assert_eq!(tables.blocks.len(), cfg.n_layers, "one table per block");
    for (i, t) in tables.blocks.iter().enumerate() {
        assert_eq!(
            t.len(),
            TIMESTEPS.len() * cfg.adaln_out,
            "block {i} table size"
        );
        let bad = t.iter().filter(|v| !v.is_finite()).count();
        assert_eq!(bad, 0, "block {i} has {bad} non-finite modulation values");
    }
    assert_eq!(
        tables.final_layer.len(),
        TIMESTEPS.len() * cfg.final_adaln_out
    );

    // The table MUST depend on the timestep. A constant table means the
    // projection read the wrong bytes (the reshape trap) or `temb` collapsed.
    let row = cfg.adaln_out;
    let b0 = &tables.blocks[0];
    let d: f32 = (0..row).map(|i| (b0[i] - b0[row + i]).abs()).sum();
    assert!(d > 1e-3, "block 0 modulation does not vary with timestep");

    // Modulation lives at a sane scale: shift/gate near 0, scale near 0 too
    // (the block applies `1 + scale`). Orders of magnitude off means the
    // dequantized projection was misread.
    let mut worst = 0.0f32;
    for t in &tables.blocks {
        for v in t {
            worst = worst.max(v.abs());
        }
    }
    eprintln!(
        "---- minimax adaln precompute ----\n\
         {} blocks x {} timesteps x {} params, |max| {worst:.3}, {elapsed:.1}s",
        cfg.n_layers,
        TIMESTEPS.len(),
        PARAMS_PER_BLOCK
    );
    assert!(
        worst < 1.0e3,
        "modulation magnitude {worst} is implausible; suspect a misread projection"
    );
    eprintln!(
        "table {:.1} MB replaces ~7 GB of adaln_proj",
        tables.bytes(4) as f64 / 1.0e6
    );
}
