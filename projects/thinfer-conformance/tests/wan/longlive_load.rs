//! LongLive-2.0-5B DiT ingestion proof against the real `model_bf16.pt`.
//!
//! GPU-free and parity-free: it opens the actual 10GB `.pt` via the runtime
//! `PytorchSource`, applies the canonical rename, and asserts the rename is a
//! perfect bijection onto the checkpoint - every `.pt` tensor is consumed and
//! every canonical diffusers id the DiT loader requests resolves through the
//! renamed source. This is the durable counterpart to the structural unit test
//! in `wan/source.rs` (which can't see the real file) and the `thinfer-native`
//! `pt_smoke` dump (which can't see the rename map).
//!
//! Gated on `THINFER_LONGLIVE_PT` pointing at `model_bf16.pt` (skips when unset),
//! so it never runs under default `cargo test`. Resolve the cached path with:
//!   `ls ~/.cache/huggingface/hub/models--Efficient-Large-Model--LongLive-2.0-5B/snapshots/*/model_bf16.pt`
//!
//! Run:
//!   `THINFER_LONGLIVE_PT=<path> cargo test -p thinfer-conformance --features \
//!    wan-e2e --release longlive_pt -- --nocapture`

#![cfg(feature = "wan-e2e")]

use std::collections::HashSet;

use thinfer_core::format::pytorch::PytorchSource;
use thinfer_core::weight::{WeightId, WeightSource};
use thinfer_models::wan::dit_block::WanDitConfig;
use thinfer_models::wan::source::longlive_dit_renames;
use thinfer_native::MmapFileOpener;

#[test]
fn longlive_pt_rename_is_total_against_real_checkpoint() {
    let Some(path) = std::env::var_os("THINFER_LONGLIVE_PT") else {
        eprintln!("longlive_load: THINFER_LONGLIVE_PT unset; skipping");
        return;
    };
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    rt.block_on(async {
        let opener = MmapFileOpener::new(&path).await.expect("open .pt");
        let pt = PytorchSource::open(opener).await.expect("parse .pt");

        let pt_names: HashSet<String> = pt.catalog().entries.keys().map(|k| k.0.clone()).collect();
        let map = longlive_dit_renames(WanDitConfig::fastwan_ti2v_5b().num_layers);
        let map_keys: HashSet<String> = map.keys().map(|k| k.0.clone()).collect();

        // Every checkpoint tensor must be named by the map, and every map key
        // must exist in the checkpoint: a perfect set equality.
        let uncovered: Vec<&String> = pt_names.difference(&map_keys).collect();
        let missing: Vec<&String> = map_keys.difference(&pt_names).collect();
        assert!(
            uncovered.is_empty() && missing.is_empty(),
            "rename/.pt mismatch:\n  .pt tensors with no mapping: {uncovered:?}\n  \
             map keys absent from .pt: {missing:?}"
        );
        assert_eq!(pt_names.len(), 825, "LongLive DiT tensor count");

        // The renamed source must expose every canonical id under its new name.
        use thinfer_core::format::union::RenamedSource;
        let renamed = RenamedSource::with_passthrough(pt, map.clone());
        for canon in map.values() {
            assert!(
                renamed.catalog().get(canon).is_some(),
                "canonical id {canon:?} did not resolve through the renamed source"
            );
        }
        eprintln!(
            "longlive_load: OK - {} tensors, rename bijective, all canonical ids resolve",
            pt_names.len()
        );
        // Touch a known weight end to end to prove a tensor actually opens.
        let probe = WeightId("blocks.0.attn1.to_q.weight".to_string());
        renamed.open(&probe).await.expect("open probe tensor");
    });
}
