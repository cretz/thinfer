//! Weight format parsers. Pure: bytes in, catalog out. No IO.
//!
//! IO lives in platform crates (`thinfer-native` reads from disk;
//! `thinfer-web` from OPFS). Format parsers live here so both targets share
//! one implementation per format.

pub mod safetensors;
