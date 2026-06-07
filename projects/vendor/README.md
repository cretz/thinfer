# Vendored crates

## `wgpu-29.0.3/` — web-subgroups patch

Upstream `wgpu` 29.0.3 facade crate, copied verbatim plus one change: it lets the
browser (web) backend report/request the `subgroups` feature, which upstream only
exposes on native (gfx-rs/wgpu #5555, #8202, both open). Without it our matmul/sdpa
shaders take a no-subgroup path in the browser (~2.3x slower per DiT step).

Only the thin facade is vendored; `wgpu-core/hal/types` + `naga` stay on crates.io
29.0.3. Substituted via `[patch.crates-io]` in the workspace `Cargo.toml`.

Two edits, both marked `// THINFER-PATCH(web-subgroups #5555)`:
- `webgpu/webgpu_sys/gen_GpuFeatureName.rs` — add `Subgroups = "subgroups"`.
- `webgpu.rs` — add the `SUBGROUP -> Subgroups` row to `FEATURES_MAPPING` (16->17).

Our side (in `thinfer-core`): prepend `enable subgroups;` to matmul_i8 + sdpa WGSL
only on the web backend (Tint needs it; naga rejects it).

On wgpu upgrade: re-copy the facade, `grep -rn THINFER-PATCH`, re-apply, update the
patch path. If upstream ships web subgroups, delete this vendor and the `[patch]`.
