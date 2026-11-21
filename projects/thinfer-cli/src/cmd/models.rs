//! `thinfer models`: every model id, one line each, from the same `ValueEnum`s
//! `--model` parses (so the list cannot drift from what the CLI accepts).

use clap::ValueEnum;
use thinfer_app::model::{
    ImageModelEngine, ImageModelId, SwapModel, TextModelEngine, TextModelId, VideoModelEngine,
    VideoModelId,
};
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
    section("text", TextModelId::DEFAULT, |m| {
        (m.title(), notes(Some(m.manifest()), false))
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The catalog lists models from the `ALL` arrays (no clap on the
    /// server); they must hold every variant.
    #[test]
    fn all_lists_cover_every_variant() {
        assert_eq!(ImageModelId::ALL.as_slice(), ImageModelId::value_variants());
        assert_eq!(VideoModelId::ALL.as_slice(), VideoModelId::value_variants());
        assert_eq!(SwapModel::ALL.as_slice(), SwapModel::value_variants());
        // The text list predates this check and keeps its own order.
        let mut text = TextModelId::ALL.to_vec();
        text.sort_by_key(|m| m.to_string());
        let mut parsed = TextModelId::value_variants().to_vec();
        parsed.sort_by_key(|m| m.to_string());
        assert_eq!(text, parsed);
    }
}
