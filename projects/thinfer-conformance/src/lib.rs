//! Conformance driver: regenerates per-dtype safetensors fixtures from the
//! op-test registry's spec, by shelling out to the bundled Python reference
//! (`python/thinfer_pytorch_ref/gen_from_spec.py` via `uv run`).
//!
//! Native test (`tests/native.rs`) calls `regenerate_fixtures` then runs the
//! registry against `WgpuBackend` and diffs. Web mirror in `thinfer-web/tests/`
//! consumes the committed fixtures via `include_bytes!`.

use std::path::{Path, PathBuf};
use thinfer_core::conformance::{SpecCase, SpecPayload};

pub fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

pub fn fixtures_dir() -> PathBuf {
    crate_root().join("fixtures")
}

pub fn python_dir() -> PathBuf {
    crate_root().join("python")
}

/// Serialize `cases` (each tagged with the dtypes it should be generated
/// in) to `<fixtures>/spec.json`, then invoke
/// `uv run python -m thinfer_pytorch_ref.gen_from_spec` to write
/// `<fixtures>/<dtype>.safetensors`. Panics on failure.
pub fn regenerate_fixtures(cases: &[SpecCase]) -> PathBuf {
    let fixtures = fixtures_dir();
    std::fs::create_dir_all(&fixtures).expect("create fixtures dir");
    let spec_json = fixtures.join("spec.json");
    let payload = SpecPayload { cases };
    std::fs::write(&spec_json, serde_json::to_vec_pretty(&payload).unwrap()).expect("write spec");
    run_python(&spec_json, &fixtures);
    fixtures
}

fn run_python(spec_json: &Path, out_dir: &Path) {
    let py_dir = python_dir();
    let status = std::process::Command::new("uv")
        .args([
            "run",
            "--directory",
            py_dir.to_str().unwrap(),
            "python",
            "-m",
            "thinfer_pytorch_ref.gen_from_spec",
            "--spec",
            spec_json.to_str().unwrap(),
            "--out-dir",
            out_dir.to_str().unwrap(),
        ])
        .status()
        .expect("failed to spawn `uv run` (is uv installed?)");
    assert!(status.success(), "python fixture gen failed");
}
