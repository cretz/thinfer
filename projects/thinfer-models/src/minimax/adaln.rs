//! Precomputed AdaLN modulation.
//!
//! ~13B of MiniMax-H3's 33B parameters live in the per-block `adaln_proj.linear`
//! projections (50 blocks x `[96768, 2688]`). Their input is the timestep
//! embedding alone, which does not depend on the packed sequence, so for a fixed
//! sampler schedule every modulation tensor the whole denoising run will ever
//! need can be computed once, up front, and the projections then dropped. The
//! model card notes this explicitly: the AdaLN outputs can be precomputed and
//! cached, so those parameters need not be loaded for inference-only deployment.
//!
//! The saving is large and lopsided. For a 4-step turbo run the table holds
//!
//! ```text
//! 50 blocks x (5 timesteps x 3 modalities) x 6 x 5376  =  24.2M values
//! ```
//!
//! which is ~48 MB in bf16, against ~7 GB of Q4 weights it replaces. The DiT's
//! streamed working set drops from ~18.5 GB to ~11.5 GB per step.
//!
//! **Precision.** The table is built from the f32 timestep embedding through the
//! DEQUANTIZED projection. Every block reads the same `temb`, so an error
//! introduced here biases every block's modulation identically at every sampling
//! step and accumulates coherently along the denoising trajectory. Build the
//! table in f32 and narrow only the stored result.

use super::{DitConfig, MODALITY_NUM};

/// Modulation parameters per block, in checkpoint order.
///
/// One projection is shared by `norm1` and `norm2` AND by the three modalities,
/// so it cannot be folded into either norm.
pub const PARAMS_PER_BLOCK: usize = 6;

/// Index of each parameter within a table row.
pub const SHIFT_MSA: usize = 0;
pub const SCALE_MSA: usize = 1;
pub const GATE_MSA: usize = 2;
pub const SHIFT_MLP: usize = 3;
pub const SCALE_MLP: usize = 4;
pub const GATE_MLP: usize = 5;

/// Layout of one block's precomputed modulation table.
///
/// Rows are `[t0_mod0, t0_mod1, t0_mod2, t1_mod0, ...]`, addressed by
/// `timestep_index * MODALITY_NUM + max(tag, 0)`. Within a row the six
/// parameters are contiguous, each `hidden` wide, so a shader given a row index
/// reads all six from one base offset.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ModulationLayout {
    pub num_timesteps: usize,
    pub hidden: usize,
}

impl ModulationLayout {
    pub fn new(num_timesteps: usize, hidden: usize) -> Self {
        Self {
            num_timesteps,
            hidden,
        }
    }

    /// Rows in the table: one per `(timestep, modality)` pair.
    pub fn rows(&self) -> usize {
        self.num_timesteps * MODALITY_NUM
    }

    /// Elements per row: the six parameters laid out back to back.
    pub fn row_width(&self) -> usize {
        PARAMS_PER_BLOCK * self.hidden
    }

    /// Elements in one block's table.
    pub fn elements(&self) -> usize {
        self.rows() * self.row_width()
    }

    /// Offset of `param` for `(timestep_index, tag)`.
    pub fn offset(&self, timestep_index: usize, tag: i32, param: usize) -> usize {
        debug_assert!(param < PARAMS_PER_BLOCK);
        let row = timestep_index * MODALITY_NUM + tag.max(0) as usize;
        debug_assert!(row < self.rows());
        row * self.row_width() + param * self.hidden
    }

    /// Total bytes for all blocks at `bytes_per_element`, plus the final layer.
    pub fn total_bytes(&self, n_layers: usize, bytes_per_element: usize) -> usize {
        let blocks = n_layers * self.elements() * bytes_per_element;
        let final_layer = self.num_timesteps * 2 * self.hidden * bytes_per_element;
        blocks + final_layer
    }
}

/// Sinusoidal timestep embedding matching diffusers `get_timestep_embedding`
/// with `flip_sin_to_cos = true` and `downscale_freq_shift = 0`.
///
/// Timesteps are consumed UNSCALED in `[0, 1]`; there is no 1000x convention
/// here. The `[cos, sin]` order is the flipped one: diffusers builds `[sin, cos]`
/// then swaps the halves.
pub fn timestep_embedding(timesteps: &[f32], dim: usize) -> Vec<f32> {
    let half = dim / 2;
    let mut out = vec![0.0f32; timesteps.len() * dim];
    for (row, &t) in timesteps.iter().enumerate() {
        for j in 0..half {
            // Computed in f64 then narrowed: the exponent is a ratio of small
            // integers scaled by ln(10000) and the f32 path loses the low bits.
            let exponent = -(10_000.0f64).ln() * j as f64 / half as f64;
            let arg = t as f64 * exponent.exp();
            out[row * dim + j] = arg.cos() as f32;
            out[row * dim + half + j] = arg.sin() as f32;
        }
    }
    out
}

/// The distinct noise levels a run presents, and the mapping from a per-row
/// timestep index at each step into that global table.
///
/// The transformer is handed the SAME table at every step, so a single
/// modulation table covers the whole run.
#[derive(Clone, Debug)]
pub struct ModulationPlan {
    /// Distinct timesteps, sorted. The table is built over exactly these.
    pub timesteps: Vec<f32>,
    /// Per step, the table row index of every packed-sequence row.
    pub per_step_row_index: Vec<Vec<u32>>,
    pub layout: ModulationLayout,
}

impl ModulationPlan {
    /// Build the plan from each step's `(distinct, per-row index)` pair as
    /// produced by [`super::packing::build_row_timesteps`], remapping every
    /// step's local indices onto one global timestep table.
    pub fn build(cfg: &DitConfig, per_step: &[(Vec<f32>, Vec<u32>)]) -> Self {
        let mut timesteps: Vec<f32> = per_step
            .iter()
            .flat_map(|(d, _)| d.iter().copied())
            .collect();
        timesteps.sort_by(|a, b| a.partial_cmp(b).expect("timesteps are finite"));
        timesteps.dedup();

        let per_step_row_index = per_step
            .iter()
            .map(|(distinct, index)| {
                let remap: Vec<u32> = distinct
                    .iter()
                    .map(|v| {
                        timesteps
                            .binary_search_by(|p| p.partial_cmp(v).expect("finite"))
                            .expect("every local timestep is in the global table")
                            as u32
                    })
                    .collect();
                index.iter().map(|&i| remap[i as usize]).collect()
            })
            .collect();

        let layout = ModulationLayout::new(timesteps.len(), cfg.hidden);
        Self {
            timesteps,
            per_step_row_index,
            layout,
        }
    }

    /// Per-row index into the modulation table for `step`, combining the row's
    /// timestep index with its modality tag.
    pub fn adaln_row_indices(&self, step: usize, token_tags: &[i32]) -> Vec<u32> {
        let ts = &self.per_step_row_index[step];
        debug_assert_eq!(ts.len(), token_tags.len());
        ts.iter()
            .zip(token_tags)
            .map(|(&t, &tag)| t * MODALITY_NUM as u32 + tag.max(0) as u32)
            .collect()
    }

    /// Run-length encode the per-row table indices into contiguous segments.
    ///
    /// This is what lets the DiT skip a gather kernel entirely. `build_packed_
    /// sequence` groups the layout by modality (`[text | conditions | audio |
    /// video]`) and every row of a modality shares one timestep, so rows that
    /// share a modulation table row are CONTIGUOUS. A t2va request collapses to
    /// exactly three segments, and the block can drive the existing broadcast
    /// `bcast_modulate` / `bcast_fma` kernels once per segment against an offset
    /// slice of the table.
    ///
    /// Keyframe requests add a conditioning segment, plus one per vision block
    /// embedded in the text rows (those are tagged video). The count stays small;
    /// callers that care about dispatch count should check [`Vec::len`].
    pub fn segments(&self, step: usize, token_tags: &[i32]) -> Vec<ModSegment> {
        let rows = self.adaln_row_indices(step, token_tags);
        let mut out: Vec<ModSegment> = Vec::new();
        for (i, &r) in rows.iter().enumerate() {
            match out.last_mut() {
                Some(seg) if seg.table_row == r => seg.len += 1,
                _ => out.push(ModSegment {
                    start: i as u32,
                    len: 1,
                    table_row: r,
                }),
            }
        }
        out
    }
}

/// A maximal run of consecutive packed-sequence rows sharing one modulation
/// table row. See [`ModulationPlan::segments`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModSegment {
    /// First packed-sequence row.
    pub start: u32,
    pub len: u32,
    /// Row of the modulation table these rows read.
    pub table_row: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::minimax::{TAG_AUDIO, TAG_PAD, TAG_TEXT, TAG_VIDEO};

    fn cfg() -> DitConfig {
        DitConfig::h3_33b()
    }

    #[test]
    fn row_width_matches_the_checkpoint_projection() {
        let l = ModulationLayout::new(5, cfg().hidden);
        // The projection emits 6 params x 3 modalities x hidden per timestep.
        assert_eq!(l.row_width() * MODALITY_NUM, cfg().adaln_out);
    }

    #[test]
    fn offsets_are_distinct_and_in_bounds() {
        let l = ModulationLayout::new(4, 8);
        let mut seen = std::collections::HashSet::new();
        for t in 0..4 {
            for tag in [TAG_VIDEO, TAG_TEXT, TAG_AUDIO] {
                for p in 0..PARAMS_PER_BLOCK {
                    let o = l.offset(t, tag, p);
                    assert!(o + l.hidden <= l.elements());
                    assert!(seen.insert(o), "offset {o} collided");
                }
            }
        }
        assert_eq!(seen.len(), 4 * MODALITY_NUM * PARAMS_PER_BLOCK);
    }

    #[test]
    fn padding_tag_clamps_to_video() {
        let l = ModulationLayout::new(2, 4);
        assert_eq!(l.offset(1, TAG_PAD, 0), l.offset(1, TAG_VIDEO, 0));
    }

    /// The whole point of the port: the table must be tiny next to the weights.
    #[test]
    fn table_is_megabytes_not_gigabytes() {
        let c = cfg();
        let l = ModulationLayout::new(5, c.hidden);
        let bytes = l.total_bytes(c.n_layers, 2);
        assert!(bytes < 64 << 20, "table is {bytes} bytes, expected < 64 MB");
        // The projections it replaces: 50 x [96768, 2688] at ~0.5 byte/param.
        let projected = c.n_layers * c.adaln_out * c.time_embed_dim / 2;
        assert!(
            projected / bytes > 100,
            "expected a >100x saving, got {}x",
            projected / bytes
        );
    }

    #[test]
    fn timestep_embedding_shape_and_range() {
        let e = timestep_embedding(&[0.0, 0.5, 1.0], 256);
        assert_eq!(e.len(), 3 * 256);
        assert!(e.iter().all(|v| (-1.0..=1.0).contains(v)));
    }

    /// `flip_sin_to_cos`: the first half is cos, the second sin. At t = 0 every
    /// angle is 0, so cos is all ones and sin all zeros. Getting the halves
    /// backwards is silent and would mismodulate every block.
    #[test]
    fn timestep_embedding_is_cos_then_sin() {
        let e = timestep_embedding(&[0.0], 8);
        assert_eq!(&e[..4], &[1.0, 1.0, 1.0, 1.0]);
        assert_eq!(&e[4..], &[0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn timestep_embedding_varies_with_t() {
        let a = timestep_embedding(&[0.25], 64);
        let b = timestep_embedding(&[0.75], 64);
        assert!(a != b);
    }

    #[test]
    fn plan_merges_per_step_tables() {
        // Two steps with different local distinct sets must land on one global
        // table, and each step's rows must resolve back to their own values.
        let per_step = vec![
            (vec![0.2f32, 0.6], vec![0u32, 1, 1, 0]),
            (vec![0.4f32, 0.8], vec![0u32, 1, 1, 0]),
        ];
        let plan = ModulationPlan::build(&cfg(), &per_step);
        assert_eq!(plan.timesteps, vec![0.2, 0.4, 0.6, 0.8]);
        assert_eq!(plan.per_step_row_index[0], vec![0, 2, 2, 0]);
        assert_eq!(plan.per_step_row_index[1], vec![1, 3, 3, 1]);
        assert_eq!(plan.layout.num_timesteps, 4);
    }

    #[test]
    fn adaln_row_combines_timestep_and_tag() {
        let per_step = vec![(vec![0.2f32, 0.6], vec![0u32, 1, 1])];
        let plan = ModulationPlan::build(&cfg(), &per_step);
        let rows = plan.adaln_row_indices(0, &[TAG_VIDEO, TAG_AUDIO, TAG_TEXT]);
        // row = timestep_index * MODALITY_NUM + tag:
        // (t0, video) -> 0, (t1, audio) -> 5, (t1, text) -> 4.
        assert_eq!(rows, vec![0, 5, 4]);
    }

    /// The property the whole no-gather design rests on: a t2va packed sequence
    /// collapses to exactly THREE contiguous modulation segments (text, audio,
    /// video). If this ever breaks, the DiT needs a gather kernel instead of
    /// per-segment broadcast dispatches.
    #[test]
    fn t2va_collapses_to_three_contiguous_segments() {
        use crate::minimax::packing::{build_packed_sequence, build_row_timesteps};

        let layout = build_packed_sequence(&[TAG_TEXT; 6], 3, 4, 4, 5, (1, 2, 2), &[]);
        let per_step = vec![build_row_timesteps(&layout, 0.7, 0.4, 0.999, 1.0)];
        let plan = ModulationPlan::build(&cfg(), &per_step);
        let segs = plan.segments(0, &layout.token_tags);

        assert_eq!(segs.len(), 3, "t2va must not need a gather: {segs:?}");
        // Segments tile the sequence with no gaps and no overlap.
        assert_eq!(segs[0].start, 0);
        let mut at = 0u32;
        for s in &segs {
            assert_eq!(s.start, at, "segments must be contiguous");
            at += s.len;
        }
        assert_eq!(at as usize, layout.seq_len, "segments must cover every row");

        // And they line up with the modality regions.
        assert_eq!(segs[0].len as usize, layout.text_indices.len());
        assert_eq!(segs[1].len as usize, layout.audio_indices.len());
        assert_eq!(segs[2].len as usize, layout.video_indices.len());
    }

    /// Every row of a segment must genuinely share the table row, otherwise the
    /// broadcast dispatch silently modulates rows with the wrong parameters.
    #[test]
    fn segments_agree_with_per_row_indices() {
        use crate::minimax::packing::{build_packed_sequence, build_row_timesteps};

        let layout = build_packed_sequence(&[TAG_TEXT; 4], 2, 4, 4, 3, (1, 2, 2), &[]);
        let per_step = vec![build_row_timesteps(&layout, 0.7, 0.4, 0.999, 1.0)];
        let plan = ModulationPlan::build(&cfg(), &per_step);
        let rows = plan.adaln_row_indices(0, &layout.token_tags);
        for s in plan.segments(0, &layout.token_tags) {
            for r in s.start..s.start + s.len {
                assert_eq!(rows[r as usize], s.table_row, "row {r} in {s:?}");
            }
        }
    }
}
