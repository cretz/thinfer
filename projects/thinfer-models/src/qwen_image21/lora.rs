//! Qwen-Image-2.1 LoRAs onto the unsloth GGUF. Community adapters come in
//! several layouts, all normalized here into one in-memory adapter the generic
//! fold ([`crate::common::lora`]) consumes:
//!
//! - key prefixes: `transformer.` (diffusers / lightx2v), `diffusion_model.`
//!   (ComfyUI), none (ai-toolkit), `base_model.model.` (PEFT), and kohya's
//!   `lora_unet_` with underscores for dots;
//! - A / B suffixes: `lora_A` / `lora_B`, `lora_down` / `lora_up`, `lora.down` /
//!   `lora.up`, PEFT's `.default.weight`; a per-site `alpha` scales by
//!   `alpha / rank` (baked into B here);
//! - the MLP: the GGUF fuses `img_mlp.gate_up = [gate_layer; proj]`. A fused
//!   `gate_up` LoRA passes through; split `gate_layer` / `proj` LoRAs merge into
//!   one: A = `[A_gate; A_proj]`, B block-diagonal over the gate rows then the
//!   proj rows, which is exactly the sum of the two deltas.
//!
//! Every emitted key is `diffusion_model.{base key}.{lora_A|lora_B}.weight` (or
//! `.diff` / `.diff_b`) against the GGUF's `model.diffusion_model.` names, with
//! the scale baked in, so the fold matches each site by name.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use thinfer_core::tensor::{Shape, StorageEncoding};
use thinfer_core::weight::{
    WeightCatalog, WeightEntry, WeightId, WeightReader, WeightSource, decode_into,
};

use crate::common::lora::VecReader;

use super::config;
use super::loader::PREFIX;

/// Site names under each block, in the GGUF's (dotted) form.
const BLOCK_SITES: [&str; 10] = [
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out.0",
    "attn.norm_q",
    "attn.norm_k",
    "img_mlp.gate_layer",
    "img_mlp.proj",
    "img_mlp.gate_up",
    "img_mlp.out",
];
/// Top-level site names.
const TOP_SITES: [&str; 9] = [
    "img_in",
    "txt_in.in_layer",
    "txt_in.out_layer",
    "txt_in.text_norm",
    "modulation.1",
    "norm_out.linear",
    "proj_out",
    "time_text_embed.timestep_embedder.linear_1",
    "time_text_embed.timestep_embedder.linear_2",
];
const KEY_PREFIXES: [&str; 4] = [
    "base_model.model.",
    "transformer.",
    "diffusion_model.",
    "model.",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Part {
    A,
    B,
    Alpha,
    Diff,
    DiffB,
}

const SUFFIXES: [(&str, Part); 11] = [
    (".lora_A.default.weight", Part::A),
    (".lora_B.default.weight", Part::B),
    (".lora_A.weight", Part::A),
    (".lora_B.weight", Part::B),
    (".lora_down.weight", Part::A),
    (".lora_up.weight", Part::B),
    (".lora.down.weight", Part::A),
    (".lora.up.weight", Part::B),
    (".alpha", Part::Alpha),
    (".diff_b", Part::DiffB),
    (".diff", Part::Diff),
];

/// A kohya site (`transformer_blocks_3_attn_to_q`) in dotted form.
fn dotted_kohya(site: &str) -> Option<String> {
    if let Some(rest) = site.strip_prefix("transformer_blocks_") {
        let (idx, name) = rest.split_once('_')?;
        idx.parse::<usize>().ok()?;
        let s = BLOCK_SITES.iter().find(|s| s.replace('.', "_") == name)?;
        return Some(format!("transformer_blocks.{idx}.{s}"));
    }
    TOP_SITES
        .iter()
        .find(|s| s.replace('.', "_") == site)
        .map(|s| s.to_string())
}

/// `(site, part)` of an adapter key, the site in the GGUF's dotted form
/// without its `model.diffusion_model.` prefix.
fn parse_key(key: &str) -> Option<(String, Part)> {
    let (stem, part) = SUFFIXES
        .iter()
        .find_map(|&(s, p)| key.strip_suffix(s).map(|stem| (stem, p)))?;
    let mut site = stem;
    if let Some(k) = site.strip_prefix("lora_unet_") {
        return dotted_kohya(k).map(|s| (s, part));
    }
    // Strip any stack of known prefixes (`diffusion_model.model.diffusion_model.`).
    while let Some(rest) = KEY_PREFIXES.iter().find_map(|p| site.strip_prefix(p)) {
        site = rest;
    }
    Some((site.to_string(), part))
}

#[derive(Default)]
struct SiteParts {
    a: Option<(Vec<f32>, Vec<usize>)>,
    b: Option<(Vec<f32>, Vec<usize>)>,
    alpha: Option<f32>,
    diff: Option<(Vec<f32>, Vec<usize>)>,
    diff_b: Option<(Vec<f32>, Vec<usize>)>,
}

/// One site's low-rank delta: A `[r, k]`, B `[n, r]` with `alpha / r` baked in.
struct LowRank {
    a: Vec<f32>,
    b: Vec<f32>,
    r: usize,
    k: usize,
    n: usize,
}

impl SiteParts {
    /// The site's low-rank delta, when both halves are present.
    fn low_rank(&self) -> Option<LowRank> {
        let ((a, ash), (b, bsh)) = (self.a.as_ref()?, self.b.as_ref()?);
        let (r, k, n) = (ash[0], ash[1], bsh[0]);
        let scale = self.alpha.map_or(1.0, |al| al / r as f32);
        Some(LowRank {
            a: a.clone(),
            b: b.iter().map(|v| v * scale).collect(),
            r,
            k,
            n,
        })
    }
}

/// The normalized in-memory adapter (f32 tensors).
pub struct AdaptedLora {
    catalog: WeightCatalog,
    data: HashMap<WeightId, Arc<[u8]>>,
}

impl AdaptedLora {
    fn push(&mut self, key: String, vals: Vec<f32>, shape: Vec<usize>) {
        let bytes: Arc<[u8]> = Arc::from(bytemuck::cast_slice::<f32, u8>(&vals));
        let id = WeightId(key);
        self.catalog.entries.insert(
            id.clone(),
            WeightEntry {
                offset: 0,
                size: bytes.len() as u64,
                encoding: Some(StorageEncoding::F32),
                encoding_label: "F32".into(),
                shape: Shape(shape),
            },
        );
        self.data.insert(id, bytes);
    }

    fn push_site(
        &mut self,
        site: &str,
        a: Vec<f32>,
        b: Vec<f32>,
        (r, k, n): (usize, usize, usize),
    ) {
        let base = format!("diffusion_model.{PREFIX}{site}");
        self.push(format!("{base}.lora_A.weight"), a, vec![r, k]);
        self.push(format!("{base}.lora_B.weight"), b, vec![n, r]);
    }
}

impl WeightSource for AdaptedLora {
    type Reader = VecReader;
    type Error = String;

    fn catalog(&self) -> &WeightCatalog {
        &self.catalog
    }

    async fn open(&self, id: &WeightId) -> Result<VecReader, String> {
        self.data
            .get(id)
            .map(|b| VecReader::from_vec(b.to_vec()))
            .ok_or_else(|| format!("adapter has no tensor {}", id.0))
    }
}

async fn read_f32<L: WeightSource>(
    lora: &L,
    id: &WeightId,
) -> Result<(Vec<f32>, Vec<usize>), String> {
    let e = lora
        .catalog()
        .get(id)
        .ok_or_else(|| format!("{}: missing", id.0))?;
    let enc = e
        .encoding
        .ok_or_else(|| format!("{}: undecodable dtype {}", id.0, e.encoding_label))?;
    let shape = e.shape.0.clone();
    let elems: usize = shape.iter().product::<usize>().max(1);
    let mut r = lora
        .open(id)
        .await
        .map_err(|e| format!("{}: {e:?}", id.0))?;
    let mut bytes = vec![0u8; r.len() as usize];
    r.read_at(0, &mut bytes)
        .await
        .map_err(|e| format!("{}: {e:?}", id.0))?;
    let mut out = vec![0f32; elems];
    decode_into(enc, &bytes, bytemuck::cast_slice_mut(&mut out))
        .map_err(|e| format!("{}: {e:?}", id.0))?;
    Ok((out, shape))
}

/// Normalize `lora` onto the GGUF's keys (see the module note). Keys that are
/// not LoRA parts pass nothing through; sites the GGUF lacks are dropped by
/// the fold's discovery.
pub async fn adapt<L: WeightSource>(lora: &L) -> Result<AdaptedLora, String> {
    let mut sites: BTreeMap<String, SiteParts> = BTreeMap::new();
    for id in lora.catalog().entries.keys() {
        let Some((site, part)) = parse_key(&id.0) else {
            continue;
        };
        let (vals, shape) = read_f32(lora, id).await?;
        let p = sites.entry(site).or_default();
        match part {
            Part::A => p.a = Some((vals, shape)),
            Part::B => p.b = Some((vals, shape)),
            Part::Alpha => p.alpha = vals.first().copied(),
            Part::Diff => p.diff = Some((vals, shape)),
            Part::DiffB => p.diff_b = Some((vals, shape)),
        }
    }

    let mut out = AdaptedLora {
        catalog: WeightCatalog::new(),
        data: HashMap::new(),
    };
    let inner = config::MLP_INNER;
    // Split MLP halves, per block: (gate_layer, proj).
    let mut split: BTreeMap<String, [Option<SiteParts>; 2]> = BTreeMap::new();
    for (site, parts) in sites {
        let half = [".img_mlp.gate_layer", ".img_mlp.proj"]
            .iter()
            .position(|s| site.ends_with(s));
        if let Some(h) = half {
            let block =
                site[..site.len() - [".img_mlp.gate_layer", ".img_mlp.proj"][h].len()].to_string();
            split.entry(block).or_default()[h] = Some(parts);
            continue;
        }
        if let Some(LowRank { a, b, r, k, n }) = parts.low_rank() {
            out.push_site(&site, a, b, (r, k, n));
        }
        let base = format!("diffusion_model.{PREFIX}{site}");
        if let Some((d, sh)) = parts.diff {
            out.push(format!("{base}.diff"), d, sh);
        }
        if let Some((d, sh)) = parts.diff_b {
            out.push(format!("{base}.diff_b"), d, sh);
        }
    }

    for (block, halves) in split {
        let site = format!("{block}.img_mlp.gate_up");
        // Low-rank: A = [A_gate; A_proj], B block-diagonal over the row ranges.
        let lr: Vec<_> = halves
            .iter()
            .map(|h| h.as_ref().and_then(SiteParts::low_rank))
            .collect();
        let k = lr.iter().flatten().map(|t| t.k).next();
        if let Some(k) = k {
            let ranks: Vec<usize> = lr.iter().map(|t| t.as_ref().map_or(0, |t| t.r)).collect();
            let r = ranks[0] + ranks[1];
            let n = 2 * inner;
            let mut a = Vec::with_capacity(r * k);
            let mut b = vec![0f32; n * r];
            let mut col = 0;
            for (h, t) in lr.iter().enumerate() {
                let Some(LowRank {
                    a: ah,
                    b: bh,
                    r: rh,
                    k: kh,
                    n: nh,
                }) = t
                else {
                    continue;
                };
                if *kh != k || *nh != inner {
                    return Err(format!(
                        "{block}: MLP LoRA half is [{nh}, {kh}], expected [{inner}, {k}]"
                    ));
                }
                a.extend_from_slice(ah);
                for row in 0..inner {
                    let dst = (h * inner + row) * r + col;
                    b[dst..dst + rh].copy_from_slice(&bh[row * rh..(row + 1) * rh]);
                }
                col += rh;
            }
            out.push_site(&site, a, b, (r, k, n));
        }
        // Dense weight deltas stack the same way (zeros for a missing half).
        let diffs: Vec<_> = halves
            .iter()
            .map(|h| h.as_ref().and_then(|p| p.diff.as_ref()))
            .collect();
        if let Some(cols) = diffs.iter().flatten().map(|(_, sh)| sh[1]).next() {
            let mut d = vec![0f32; 2 * inner * cols];
            for (h, part) in diffs.iter().enumerate() {
                if let Some((v, _)) = part {
                    d[h * inner * cols..(h + 1) * inner * cols].copy_from_slice(v);
                }
            }
            out.push(
                format!("diffusion_model.{PREFIX}{site}.diff"),
                d,
                vec![2 * inner, cols],
            );
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_key_convention_lands_on_the_gguf_site() {
        let want = |s: &str| Some((s.to_string(), Part::A));
        for key in [
            "transformer.transformer_blocks.3.attn.to_q.lora.down.weight",
            "diffusion_model.transformer_blocks.3.attn.to_q.lora_down.weight",
            "transformer_blocks.3.attn.to_q.lora_A.weight",
            "base_model.model.transformer_blocks.3.attn.to_q.lora_A.default.weight",
            "lora_unet_transformer_blocks_3_attn_to_q.lora_down.weight",
        ] {
            assert_eq!(
                parse_key(key),
                want("transformer_blocks.3.attn.to_q"),
                "{key}"
            );
        }
        assert_eq!(
            parse_key("lora_unet_time_text_embed_timestep_embedder_linear_1.alpha"),
            Some((
                "time_text_embed.timestep_embedder.linear_1".into(),
                Part::Alpha
            ))
        );
        assert_eq!(
            parse_key("diffusion_model.transformer_blocks.0.attn.norm_q.diff"),
            Some(("transformer_blocks.0.attn.norm_q".into(), Part::Diff))
        );
    }

    /// Poll an in-memory future (it never pends) to completion.
    fn ready<F: core::future::Future>(fut: F) -> F::Output {
        let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
        match std::pin::pin!(fut).poll(&mut cx) {
            std::task::Poll::Ready(v) => v,
            std::task::Poll::Pending => unreachable!("in-memory adapter reads never pend"),
        }
    }

    /// A merged `gate_up` delta equals the stacked `gate_layer` and `proj`
    /// deltas, each with its own alpha scale.
    #[test]
    fn split_mlp_halves_merge_into_the_fused_site() {
        ready(async {
            let (k, inner) = (4, config::MLP_INNER);
            let mut src = AdaptedLora {
                catalog: WeightCatalog::new(),
                data: HashMap::new(),
            };
            let p = "transformer_blocks.0.img_mlp";
            // gate: rank 1, alpha 2 (scale 2); proj: rank 2, no alpha.
            src.push(
                format!("{p}.gate_layer.lora_A.weight"),
                vec![1.0, 0.0, 0.0, 0.0],
                vec![1, k],
            );
            src.push(
                format!("{p}.gate_layer.lora_B.weight"),
                vec![1.0; inner],
                vec![inner, 1],
            );
            src.push(format!("{p}.gate_layer.alpha"), vec![2.0], vec![]);
            src.push(
                format!("{p}.proj.lora_A.weight"),
                vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                vec![2, k],
            );
            src.push(
                format!("{p}.proj.lora_B.weight"),
                vec![3.0; inner * 2],
                vec![inner, 2],
            );
            let out = adapt(&src).await.unwrap();
            let key = |s: &str| WeightId(format!("diffusion_model.{PREFIX}{p}.gate_up.{s}"));
            let a = read_f32(&out, &key("lora_A.weight")).await.unwrap();
            let b = read_f32(&out, &key("lora_B.weight")).await.unwrap();
            assert_eq!(a.1, vec![3, k]);
            assert_eq!(b.1, vec![2 * inner, 3]);
            // A gate row (scale 2 on column 0), a proj row (columns 1, 2).
            assert_eq!(&b.0[0..3], &[2.0, 0.0, 0.0]);
            assert_eq!(&b.0[inner * 3..inner * 3 + 3], &[0.0, 3.0, 3.0]);
        });
    }
}
