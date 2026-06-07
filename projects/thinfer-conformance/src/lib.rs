//! Conformance driver: regenerates per-dtype safetensors fixtures from the
//! op-test registry's spec, by shelling out to the bundled Python reference
//! (`python/thinfer_pytorch_ref/gen_from_spec.py` via `uv run`).
//!
//! Native test (`tests/native.rs`) calls `regenerate_fixtures` then runs the
//! registry against `WgpuBackend` and diffs. Web mirror in `thinfer-web/tests/`
//! consumes the committed fixtures via `include_bytes!`.

use std::collections::BTreeSet;
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
///
/// Cached: the fixtures are a pure function of the spec and the generator
/// sources, so the python invocation is skipped when the spec bytes match
/// the on-disk `spec.json`, every needed `<dtype>.safetensors` exists, and
/// no generator `.py` is newer than the fixtures (same idea as the
/// qwen3_parity pyref dump cache). `THINFER_CONFORMANCE_REGEN=1` forces a
/// regen; CI's fixture-drift check should set it.
pub fn regenerate_fixtures(cases: &[SpecCase]) -> PathBuf {
    let fixtures = fixtures_dir();
    std::fs::create_dir_all(&fixtures).expect("create fixtures dir");
    let spec_json = fixtures.join("spec.json");
    let payload = SpecPayload { cases };
    let spec_bytes = serde_json::to_vec_pretty(&payload).unwrap();
    if fixtures_cached(&spec_json, &spec_bytes, cases, &fixtures) {
        eprintln!("conformance: fixtures cached, skipping python regen");
        return fixtures;
    }
    std::fs::write(&spec_json, &spec_bytes).expect("write spec");
    run_python(&spec_json, &fixtures);
    fixtures
}

fn fixtures_cached(
    spec_json: &Path,
    spec_bytes: &[u8],
    cases: &[SpecCase],
    fixtures: &Path,
) -> bool {
    if std::env::var_os("THINFER_CONFORMANCE_REGEN").is_some() {
        return false;
    }
    if !std::fs::read(spec_json).is_ok_and(|prev| prev == spec_bytes) {
        return false;
    }
    let mtime = |p: &Path| std::fs::metadata(p).and_then(|m| m.modified()).ok();
    let mut newest_py = None;
    let Ok(rd) = std::fs::read_dir(python_dir().join("thinfer_pytorch_ref")) else {
        return false;
    };
    for ent in rd.flatten() {
        let p = ent.path();
        if p.extension().is_some_and(|e| e == "py") {
            newest_py = newest_py.max(mtime(&p));
        }
    }
    let Some(newest_py) = newest_py else {
        return false;
    };
    let dtypes: BTreeSet<&str> = cases
        .iter()
        .flat_map(|c| c.dtypes.iter().map(|d| d.as_str()))
        .collect();
    dtypes
        .iter()
        .all(|d| mtime(&fixtures.join(format!("{d}.safetensors"))).is_some_and(|t| t >= newest_py))
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
