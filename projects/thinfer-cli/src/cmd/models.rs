//! `thinfer models`: every model id, one line each, from the same `ValueEnum`s
//! `--model` parses (so the list cannot drift from what the CLI accepts).

use clap::ValueEnum;
use thinfer_app::model::{ImageModelId, SwapModel, VideoModelId};
use thinfer_core::manifest::ModelManifest;
use thinfer_native::sources;

pub fn run() {
    section("image", ImageModelId::DEFAULT, |m| {
        (m.title(), notes(Some(m.manifest()), false))
    });
    section("video", VideoModelId::DEFAULT, |m| {
        (m.title(), notes(Some(m.manifest()), m.is_minimax()))
    });
    section("face-swap", SwapModel::DEFAULT, |m| {
        (m.title(), notes(None, false))
    });
    println!("\nUse one with `thinfer generate <kind> --model <id>`; `--help` there has details.");
}

fn section<M: ValueEnum + PartialEq>(
    kind: &str,
    default: M,
    describe: impl Fn(&M) -> (&'static str, String),
) {
    println!("{kind}:");
    for m in M::value_variants() {
        let Some(v) = m.to_possible_value() else {
            continue;
        };
        let (title, notes) = describe(m);
        let mark = if *m == default { " (default)" } else { "" };
        println!("  {:<24}{title}{mark}{notes}", v.get_name());
    }
}

/// Requirements beyond a plain download.
fn notes(manifest: Option<&ModelManifest>, cuda: bool) -> String {
    let user_weights =
        manifest.is_some_and(|m| m.iter().any(|(_, f)| sources::effective(f).is_unfilled()));
    let mut out = String::new();
    if user_weights {
        out.push_str(" [needs your weights: thinfer-sources.toml]");
    }
    if cuda {
        out.push_str(" [NVIDIA + CUDA 13]");
    }
    out
}
