//! Native conformance: regenerate Python fixtures, run the op registry on
//! WgpuBackend, diff against the reference. CI guards fixture drift via
//! `git diff --exit-code thinfer-conformance/fixtures/` after this test.

#![cfg(feature = "conformance")]

use safetensors::SafeTensors;
use std::collections::BTreeMap;
use thinfer_conformance::regenerate_fixtures;
use thinfer_core::backend::WgpuBackend;
use thinfer_core::conformance::{
    Dtype, OpTestContext, SpecCase, TestCase, diff_max_abs, registry, tol,
};

#[test]
fn ops_match_pytorch_reference() {
    let registry = registry();
    // Per-op (dtypes, test_cases) pairs. dtypes come from the op (default
    // `[Fp32, Bf16Writes]`; VAE-only ops override to `[Fp32]`); each case
    // gets paired with each supported dtype on the run loop.
    let by_op: Vec<(&'static [Dtype], Vec<TestCase>)> = registry
        .iter()
        .map(|op| (op.dtypes(), op.test_cases()))
        .collect();
    // Build the python spec: every case carries its op's dtypes list so the
    // generator writes an `out` tensor per dtype in the matching safetensors.
    let spec_cases: Vec<SpecCase> = by_op
        .iter()
        .flat_map(|(dtypes, cases)| {
            cases.iter().map(|c| SpecCase {
                name: c.name,
                op: c.op.clone(),
                inputs: c.inputs.clone(),
                dtypes: dtypes.to_vec(),
            })
        })
        .collect();

    let fixtures = regenerate_fixtures(&spec_cases);

    let backend =
        pollster::block_on(WgpuBackend::new()).expect("wgpu adapter unavailable for tests");

    // Load each dtype's fixture exactly once.
    let mut by_dtype: BTreeMap<Dtype, Vec<u8>> = BTreeMap::new();
    for (dtypes, _) in &by_op {
        for &d in *dtypes {
            by_dtype.entry(d).or_default();
        }
    }
    for dtype in by_dtype.keys().copied().collect::<Vec<_>>() {
        let path = fixtures.join(format!("{}.safetensors", dtype.as_str()));
        let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
        by_dtype.insert(dtype, bytes);
    }

    let mut failures: Vec<String> = Vec::new();
    let mut ran = 0;

    for (op, (dtypes, cases)) in registry.iter().zip(&by_op) {
        for case in cases {
            for &dtype in *dtypes {
                let bytes = &by_dtype[&dtype];
                let st = SafeTensors::deserialize(bytes).unwrap();
                let ctx = OpTestContext {
                    backend: &backend,
                    st: &st,
                    case,
                    dtype,
                };
                let got = pollster::block_on(op.run_test(&ctx));
                let key = format!("{}/out", case.name);
                let expected = st
                    .tensor(&key)
                    .unwrap_or_else(|e| panic!("missing {key}: {e}"))
                    .data();
                let max_abs = diff_max_abs(dtype, &got, expected);
                let limit = tol(dtype);
                if max_abs > limit {
                    failures.push(format!(
                        "{}.{}: max abs {:.3e} > tol {:.0e}",
                        case.name,
                        dtype.as_str(),
                        max_abs,
                        limit
                    ));
                } else {
                    eprintln!(
                        "ok  {}.{}  max_abs={:.3e} (tol {:.0e})",
                        case.name,
                        dtype.as_str(),
                        max_abs,
                        limit
                    );
                }
                ran += 1;
            }
        }
    }

    eprintln!("ran={ran}");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
