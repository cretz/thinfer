//! Browser bindings. The `#[wasm_bindgen]` surface here must mirror
//! `src/wasm-pkg.d.ts` (the hand-maintained TS contract); both are internal
//! to the npm package, which wraps them in the typed public API.

#[cfg(target_arch = "wasm32")]
mod bindings;
#[cfg(target_arch = "wasm32")]
mod trace_bridge;
#[cfg(target_arch = "wasm32")]
mod weight_file;

#[cfg(target_arch = "wasm32")]
pub use bindings::*;
#[cfg(target_arch = "wasm32")]
pub use trace_bridge::*;
