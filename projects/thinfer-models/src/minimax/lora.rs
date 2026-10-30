//! Runtime LoRAs for the H3 DiT (the lightx2v turbo distillation and user
//! adapters), stacked and re-presented under the checkpoint's own names so
//! [`super::dit::register_handles`] finds ONE `lora_A`/`lora_B` pair per site.
//!
//! **Layouts.** H3 adapters come in three spellings, all accepted:
//!
//! - original/ComfyUI: `blocks.N.attn.qkv_proj`, `attn.out_proj`, `mlp.fc1`,
//!   `mlp.fc2` (`diffusion_model.` prefixed). The checkpoint's own layout.
//! - diffusers/PEFT (the turbo, FastVideo-native): `transformer_blocks.N.attn.
//!   to_q|to_k|to_v`, `attn.to_out(.0)`, `ff.net.0.proj`/`ff.fc_in`,
//!   `ff.net.2`/`ff.fc_out`. Separate q/k/v land on the fused `qkv_proj`, which
//!   is CONTIGUOUS `[q; k; v]` rows (measured against the official FL2VA
//!   checkpoint 2026-10-29: corr 0.997 contiguous vs 0.014 interleaved). The
//!   diffusers SwiGLU reads `[value; gate]` where the checkpoint's `mlp.fc1` is
//!   `[gate; value]`, so its `B` rows swap halves.
//! - kohya: the same names flattened to `_` under `lora_unet_`.
//!
//! Keys may carry `lora_A/lora_B` or `lora_down/lora_up`, an adapter name
//! (`.default`) and a `.alpha`. AdaLN and boundary-projection pairs are
//! reported as skipped: the runtime path covers the four block linears.
//!
//! **Stacking.** Every pair is a block-diagonal piece of one stacked pair:
//!
//! ```text
//! delta = sum_j P_j s_j B_j A_j = [P_1 s_1 B_1 | P_2 s_2 B_2 | ...] @ [A_1; A_2; ...]
//! ```
//!
//! where `P_j` places piece `j`'s rows (all `N` of them, or one third of a
//! fused qkv) and `s_j = strength * alpha / rank`. One fused qkv pair is one
//! piece; separate q/k/v are three; more adapters are more pieces. The dense
//! delta is never materialized, and tensors are synthesized on demand.
//!
//! **Layout.** Both stacked factors are emitted PRE-TRANSPOSED into the dense
//! matmul's `[K, N]` operand layout (`A^T` `[k, R]`, `B^T` `[R, n]`), so they
//! upload as-is: a per-upload GPU transpose (8 per block, every step) cost
//! more than the adapter's own GEMMs.
//!
//! **Trimmed checkpoints.** Adapters are trained on the released 50-block H3;
//! on a trim (FastH3) each block index maps through
//! [`DitConfig::block_for_base`] and pairs on removed blocks are skipped.

use std::collections::{BTreeMap, HashMap};

use half::{bf16, f16};

use thinfer_core::tensor::{Shape, StorageEncoding};
use thinfer_core::weight::{WeightCatalog, WeightEntry, WeightId, WeightReader, WeightSource};

use super::DitConfig;
use crate::common::lora::VecReader;

/// PEFT `lora_alpha` the turbo adapter was trained with (lightx2v
/// `inference_minimax_h3.py`, `DEFAULT_LORA_ALPHA`). Not stored in the file.
pub const TURBO_ALPHA: f32 = 8.0;

/// One adapter to stack.
pub struct Adapter<L> {
    pub source: L,
    /// User blend weight (1.0 = as trained).
    pub strength: f32,
    /// Overrides the file's `alpha` (the turbo ships none). With neither, the
    /// pair runs unscaled (`alpha = rank`), the ComfyUI convention.
    pub alpha: Option<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Site {
    Qkv,
    Out,
    Fc1,
    Fc2,
}

impl Site {
    fn name(self) -> &'static str {
        match self {
            Site::Qkv => "attn.qkv_proj",
            Site::Out => "attn.out_proj",
            Site::Fc1 => "mlp.fc1",
            Site::Fc2 => "mlp.fc2",
        }
    }

    /// `(N, K)` of the site's base linear.
    fn dims(self, cfg: &DitConfig) -> (usize, usize) {
        match self {
            Site::Qkv => (3 * cfg.inner_dim(), cfg.hidden),
            Site::Out => (cfg.hidden, cfg.inner_dim()),
            Site::Fc1 => (2 * cfg.ffn_hidden, cfg.hidden),
            Site::Fc2 => (cfg.hidden, cfg.ffn_hidden),
        }
    }
}

/// Where a source module's `B` rows land in its site.
#[derive(Clone, Copy)]
enum Part {
    /// All `N` rows; `swap` = exchange the two row halves (diffusers SwiGLU).
    Whole { swap: bool },
    /// One third of the contiguous fused qkv: 0 = q, 1 = k, 2 = v.
    QkvThird(usize),
}

/// A source module resolved to a site of the checkpoint.
struct Target {
    refiner: bool,
    block: usize,
    site: Site,
    part: Part,
}

/// Map a module name (any layout, trainer prefixes already stripped) to its
/// site, or `None` for a module the runtime path does not cover.
fn classify(module: &str) -> Option<Target> {
    // Separator-free, so `attn.to_q`, `attn_to_q` (kohya) resolve alike.
    let flat: String = module.chars().filter(|c| *c != '.' && *c != '_').collect();
    let (refiner, rest) = [
        ("tokenrefinerrefinerblocks", true),
        ("tokenrefinerblocks", true),
        ("transformerblocks", false),
        ("blocks", false),
    ]
    .iter()
    .find_map(|(p, r)| flat.strip_prefix(p).map(|rest| (*r, rest)))?;
    let digits = rest.chars().take_while(char::is_ascii_digit).count();
    let block = rest[..digits].parse().ok()?;
    let (site, part) = match &rest[digits..] {
        "attnqkvproj" => (Site::Qkv, Part::Whole { swap: false }),
        "attntoq" => (Site::Qkv, Part::QkvThird(0)),
        "attntok" => (Site::Qkv, Part::QkvThird(1)),
        "attntov" => (Site::Qkv, Part::QkvThird(2)),
        "attnoutproj" | "attntoout" | "attntoout0" => (Site::Out, Part::Whole { swap: false }),
        "mlpfc1" => (Site::Fc1, Part::Whole { swap: false }),
        "ffnet0proj" | "fffcin" => (Site::Fc1, Part::Whole { swap: true }),
        "mlpfc2" | "ffnet2" | "fffcout" => (Site::Fc2, Part::Whole { swap: false }),
        _ => return None,
    };
    Some(Target {
        refiner,
        block,
        site,
        part,
    })
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    A,
    B,
    Alpha,
}

const STRIP_PREFIXES: [&str; 6] = [
    "model.diffusion_model.",
    "diffusion_model.",
    "base_model.model.",
    "transformer.",
    "lora_unet_",
    "lora_transformer_",
];

/// Split a tensor key into `(module, kind)`:
/// `<prefixes><module>.<kind>[.<adapter name>][.weight]`.
fn parse_key(key: &str) -> Option<(String, Kind)> {
    let mut k = key;
    while let Some(rest) = STRIP_PREFIXES.iter().find_map(|p| k.strip_prefix(p)) {
        k = rest;
    }
    let segs: Vec<&str> = k.split('.').collect();
    let at = segs.iter().position(|s| {
        matches!(
            *s,
            "lora_A" | "lora_B" | "lora_down" | "lora_up" | "alpha" | "lora_alpha"
        )
    })?;
    if at == 0 || segs.len() - at > 3 {
        return None;
    }
    let kind = match segs[at] {
        "lora_A" | "lora_down" => Kind::A,
        "lora_B" | "lora_up" => Kind::B,
        _ => Kind::Alpha,
    };
    Some((segs[..at].join("."), kind))
}

/// One pair's contribution to a stacked site.
#[derive(Clone)]
struct Piece {
    adapter: usize,
    a: WeightId,
    b: WeightId,
    rank: usize,
    /// First row of the site this piece's `B` covers, and how many.
    row: usize,
    rows: usize,
    swap: bool,
    scale: f32,
}

/// Every piece stacked onto one site: `A` `[R, k]`, `B` `[n, R]`.
#[derive(Clone)]
struct Stack {
    n: usize,
    k: usize,
    pieces: Vec<Piece>,
}

impl Stack {
    fn rank(&self) -> usize {
        self.pieces.iter().map(|p| p.rank).sum()
    }
}

#[derive(Debug)]
pub enum LoraError<LE: core::fmt::Debug> {
    Source(LE),
    Read(String),
    /// A pair that cannot belong to this checkpoint (wrong model).
    Shape(String),
    Encoding(String),
    Missing(String),
}

/// The stacked adapters in the checkpoint's own tensor naming:
/// `diffusion_model.{blocks.N|token_refiner.blocks.N}.{site}.lora_{A,B}.weight`,
/// holding `A^T` `[k, R]` and `B^T` `[R, n]` (see the module docs).
pub struct LoraSource<L: WeightSource> {
    adapters: Vec<L>,
    catalog: WeightCatalog,
    /// Synthetic id -> (stack, is `B`).
    plan: HashMap<WeightId, (String, bool)>,
    stacks: BTreeMap<String, Stack>,
    skipped: Vec<String>,
}

impl<L: WeightSource> LoraSource<L> {
    /// The lightx2v turbo adapter alone.
    pub async fn turbo(cfg: &DitConfig, source: L) -> Result<Self, LoraError<L::Error>> {
        Self::new(
            cfg,
            vec![Adapter {
                source,
                strength: 1.0,
                alpha: Some(TURBO_ALPHA),
            }],
        )
        .await
    }

    /// Resolve and validate every pair; tensors are synthesized on `open`.
    pub async fn new(
        cfg: &DitConfig,
        adapters: Vec<Adapter<L>>,
    ) -> Result<Self, LoraError<L::Error>> {
        let mut stacks: BTreeMap<String, Stack> = BTreeMap::new();
        let mut skipped = Vec::new();
        let mut sources = Vec::with_capacity(adapters.len());
        for (ai, adapter) in adapters.into_iter().enumerate() {
            // module -> [A, B, alpha]
            let mut pairs: BTreeMap<String, [Option<WeightId>; 3]> = BTreeMap::new();
            for id in adapter.source.catalog().entries.keys() {
                // Dense payloads (`.diff`, `.diff_b`) are not low-rank.
                if let Some((module, kind)) = parse_key(&id.0) {
                    pairs.entry(module).or_default()[kind as usize] = Some(id.clone());
                } else if !id.0.ends_with(".diff") && !id.0.ends_with(".diff_b") {
                    skipped.push(id.0.clone());
                }
            }
            for (module, [a, b, alpha]) in pairs {
                let target = classify(&module).and_then(|t| {
                    let block = if t.refiner {
                        (t.block < cfg.token_refiner_layers).then_some(t.block)
                    } else {
                        cfg.block_for_base(t.block)
                    }?;
                    Some(Target { block, ..t })
                });
                let (Some(t), Some(a), Some(b)) = (target, a, b) else {
                    skipped.push(module);
                    continue;
                };
                let cat = adapter.source.catalog();
                let shape = |id: &WeightId| cat.get(id).map(|e| e.shape.0.clone());
                let (Some(sa), Some(sb)) = (shape(&a), shape(&b)) else {
                    skipped.push(module);
                    continue;
                };
                let (n, k) = t.site.dims(cfg);
                let (row, rows, swap) = match t.part {
                    Part::Whole { swap } => (0, n, swap),
                    Part::QkvThird(x) => (x * n / 3, n / 3, false),
                };
                let [rank, a_k] = sa[..] else {
                    return Err(LoraError::Shape(format!("{module}: lora_A is {sa:?}")));
                };
                if a_k != k || sb[..] != [rows, rank] {
                    return Err(LoraError::Shape(format!(
                        "{module}: lora_A {sa:?} / lora_B {sb:?} do not fit {} [{n}, {k}] \
                         (is this an adapter for a different model?)",
                        t.site.name(),
                    )));
                }
                let alpha = match (adapter.alpha, alpha) {
                    (Some(v), _) => v,
                    (None, Some(id)) => {
                        let v = read_f32(&adapter.source, &id).await?;
                        *v.first()
                            .ok_or_else(|| LoraError::Read(format!("{} empty", id.0)))?
                    }
                    (None, None) => rank as f32,
                };
                let base = if t.refiner {
                    format!("token_refiner.blocks.{}", t.block)
                } else {
                    format!("blocks.{}", t.block)
                };
                stacks
                    .entry(format!("{base}.{}", t.site.name()))
                    .or_insert_with(|| Stack {
                        n,
                        k,
                        pieces: Vec::new(),
                    })
                    .pieces
                    .push(Piece {
                        adapter: ai,
                        a,
                        b,
                        rank,
                        row,
                        rows,
                        swap,
                        scale: adapter.strength * alpha / rank as f32,
                    });
            }
            sources.push(adapter.source);
        }

        let mut catalog = WeightCatalog::new();
        let mut plan = HashMap::new();
        for (key, stack) in &stacks {
            let r = stack.rank();
            for (suffix, shape, is_b) in [
                ("lora_A", vec![stack.k, r], false),
                ("lora_B", vec![r, stack.n], true),
            ] {
                let id = WeightId(format!("diffusion_model.{key}.{suffix}.weight"));
                catalog.entries.insert(id.clone(), entry(shape));
                plan.insert(id, (key.clone(), is_b));
            }
        }
        Ok(Self {
            adapters: sources,
            catalog,
            plan,
            stacks,
            skipped,
        })
    }

    /// Sites carrying a stacked adapter.
    pub fn site_count(&self) -> usize {
        self.stacks.len()
    }

    /// Modules (or stray keys) no site covers: AdaLN and boundary
    /// projections, blocks a trim removed, incomplete pairs.
    pub fn skipped(&self) -> &[String] {
        &self.skipped
    }
}

/// Decode a float tensor (f32 / f16 / bf16) to f32.
async fn read_f32<L: WeightSource>(
    src: &L,
    id: &WeightId,
) -> Result<Vec<f32>, LoraError<L::Error>> {
    let e = src
        .catalog()
        .get(id)
        .ok_or_else(|| LoraError::Missing(id.0.clone()))?;
    let mut bytes = vec![0u8; e.size as usize];
    let mut reader = src.open(id).await.map_err(LoraError::Source)?;
    reader
        .read_at(0, &mut bytes)
        .await
        .map_err(|err| LoraError::Read(format!("{}: {err:?}", id.0)))?;
    let elems: usize = e.shape.0.iter().product::<usize>().max(1);
    let le2 = |c: &[u8]| u16::from_le_bytes([c[0], c[1]]);
    let out: Vec<f32> = match e.encoding {
        Some(StorageEncoding::F32) => bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        Some(StorageEncoding::F16) => bytes
            .chunks_exact(2)
            .map(|c| f16::from_bits(le2(c)).to_f32())
            .collect(),
        Some(StorageEncoding::Bf16) => bytes
            .chunks_exact(2)
            .map(|c| bf16::from_bits(le2(c)).to_f32())
            .collect(),
        other => {
            return Err(LoraError::Encoding(format!(
                "{}: {other:?} (adapters must be f32, f16 or bf16)",
                id.0
            )));
        }
    };
    if out.len() < elems {
        return Err(LoraError::Read(format!("{}: short tensor", id.0)));
    }
    Ok(out)
}

fn entry(shape: Vec<usize>) -> WeightEntry {
    WeightEntry {
        offset: 0,
        size: (shape.iter().product::<usize>() * 2) as u64,
        encoding: Some(StorageEncoding::Bf16),
        encoding_label: "BF16".to_string(),
        shape: Shape(shape),
    }
}

impl<L: WeightSource> WeightSource for LoraSource<L> {
    type Reader = VecReader;
    type Error = LoraError<L::Error>;

    fn catalog(&self) -> &WeightCatalog {
        &self.catalog
    }

    async fn open(&self, id: &WeightId) -> Result<Self::Reader, Self::Error> {
        let (key, is_b) = self
            .plan
            .get(id)
            .ok_or_else(|| LoraError::Missing(id.0.clone()))?;
        let stack = &self.stacks[key];
        let r = stack.rank();
        let mut out = vec![0f32; if *is_b { r * stack.n } else { stack.k * r }];
        let mut col = 0;
        for p in &stack.pieces {
            let src = &self.adapters[p.adapter];
            if *is_b {
                // `B` `[rows, rank]` -> `B^T` rows `col..`, columns `row..`.
                let b = read_f32(src, &p.b).await?;
                let half = p.rows / 2;
                for r_in in 0..p.rows {
                    let r_src = if p.swap { (r_in + half) % p.rows } else { r_in };
                    for j in 0..p.rank {
                        out[(col + j) * stack.n + p.row + r_in] = b[r_src * p.rank + j] * p.scale;
                    }
                }
            } else {
                // `A` `[rank, k]` -> `A^T` columns `col..`.
                let a = read_f32(src, &p.a).await?;
                for j in 0..p.rank {
                    for kk in 0..stack.k {
                        out[kk * r + col + j] = a[j * stack.k + kk];
                    }
                }
            }
            col += p.rank;
        }
        let bytes: Vec<u8> = out
            .iter()
            .flat_map(|v| bf16::from_f32(*v).to_bits().to_le_bytes())
            .collect();
        Ok(VecReader::from_vec(bytes))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every layout spelling of a site resolves to the same target; the
    /// diffusers SwiGLU swaps halves, the checkpoint's own `mlp.fc1` does not.
    #[test]
    fn layouts_resolve_to_the_checkpoint_sites() {
        let site = |key: &str| {
            let (module, _) = parse_key(key).expect(key);
            let t = classify(&module).expect(key);
            let part = match t.part {
                Part::Whole { swap } => (9, swap),
                Part::QkvThird(x) => (x, false),
            };
            (t.refiner, t.block, t.site, part)
        };
        let fused = (false, 3, Site::Qkv, (9, false));
        assert_eq!(
            site("diffusion_model.blocks.3.attn.qkv_proj.lora_A.weight"),
            fused
        );
        assert_eq!(
            site("lora_unet_blocks_3_attn_qkv_proj.lora_down.weight"),
            fused
        );
        assert_eq!(
            site("transformer_blocks.3.attn.to_k.lora_B.default.weight"),
            (false, 3, Site::Qkv, (1, false))
        );
        assert_eq!(
            site("transformer.transformer_blocks.3.ff.net.0.proj.lora_A.weight"),
            (false, 3, Site::Fc1, (9, true))
        );
        assert_eq!(
            site("diffusion_model.blocks.3.mlp.fc1.lora_up.weight"),
            (false, 3, Site::Fc1, (9, false))
        );
        assert_eq!(
            site("token_refiner.refiner_blocks.1.attn.to_out.0.lora_A.weight"),
            (true, 1, Site::Out, (9, false))
        );
        assert_eq!(
            site("diffusion_model.token_refiner.blocks.1.mlp.fc2.alpha"),
            (true, 1, Site::Fc2, (9, false))
        );
        assert!(classify("blocks.3.adaln_proj.linear").is_none());
        assert!(classify("video_patch_proj").is_none());
    }

    /// FastH3-Trim's kept-block renumbering.
    #[test]
    fn trim_maps_base_blocks() {
        let c = DitConfig::fasth3_trim();
        assert_eq!(c.block_for_base(5), Some(5));
        assert_eq!(c.block_for_base(6), None);
        assert_eq!(c.block_for_base(8), Some(6));
        assert_eq!(c.block_for_base(49), Some(41));
        assert_eq!(c.block_for_base(50), None);
        assert_eq!(DitConfig::h3_33b().block_for_base(6), Some(6));
    }

    /// The stacked pair reproduces the sum of the pieces' scaled deltas, each
    /// in its own rows: a wrong placement is silent (a plausible wrong model).
    #[test]
    fn stacked_pair_reproduces_the_piece_sum() {
        // Site n = 6 (three thirds of 2), k = 3. Piece 0: whole, rank 1, scale 2.
        // Piece 1: third 1 (rows 2..4), rank 2, scale 0.5.
        let (n, k) = (6usize, 3usize);
        let a0 = [1.0f32, 2.0, 3.0];
        let b0 = [1.0f32, -1.0, 2.0, 0.5, 3.0, 1.0];
        let a1 = [0.0f32, 1.0, 0.0, 1.0, 0.0, 1.0];
        let b1 = [1.0f32, 2.0, 3.0, 4.0];
        let mut want = vec![0f32; n * k];
        for r in 0..n {
            for c in 0..k {
                want[r * k + c] += 2.0 * b0[r] * a0[c];
            }
        }
        for r in 0..2 {
            for c in 0..k {
                for j in 0..2 {
                    want[(2 + r) * k + c] += 0.5 * b1[r * 2 + j] * a1[j * k + c];
                }
            }
        }
        // Stacked as `open` builds it: A_cat [3, k], B_cat [n, 3].
        let rr = 3;
        let a_cat: Vec<f32> = a0.iter().chain(&a1).copied().collect();
        let mut b_cat = vec![0f32; n * rr];
        for r in 0..n {
            b_cat[r * rr] = 2.0 * b0[r];
        }
        for r in 0..2 {
            for j in 0..2 {
                b_cat[(2 + r) * rr + 1 + j] = 0.5 * b1[r * 2 + j];
            }
        }
        let got: Vec<f32> = (0..n * k)
            .map(|i| {
                let (r, c) = (i / k, i % k);
                (0..rr).map(|j| b_cat[r * rr + j] * a_cat[j * k + c]).sum()
            })
            .collect();
        assert_eq!(got, want);
    }

    /// PEFT scaling for the turbo is alpha 8 / rank 128 = 1/16, exactly
    /// representable in bf16 (so baking it into `B` is lossless). Left at 1.0
    /// the adapter runs 16x too strong.
    #[test]
    fn turbo_scale_is_exact() {
        let s = TURBO_ALPHA / 128.0;
        assert_eq!(s, 1.0 / 16.0);
        assert_eq!(bf16::from_f32(s).to_f32(), s);
    }
}
