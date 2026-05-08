//! Per-model integration suite for Z-Image-Turbo. Each submodule below is one
//! (formerly stand-alone) integration test file. We bundle them into a single
//! test binary because cargo only auto-discovers integration tests directly
//! under `tests/`, and we don't want a Cargo.toml `[[test]]` stanza per file.
//! `common` holds shared upload helpers.

mod dit_scatter;
mod e2e_parity;
