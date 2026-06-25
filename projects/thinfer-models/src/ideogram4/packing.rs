//! Single-stream sequence packing for the Ideogram-4 DiT (`_build_inputs`).
//!
//! One sample (batch=1, the engine's per-request shape): the sequence is
//! `[text tokens][image tokens]` (no left-pad, since a single prompt has the
//! max text length). Per-token:
//! - position_ids `(t,h,w)`: text tokens get `(p,p,p)` for `p=0..num_text`;
//!   image tokens get `(0, row, col)` over the `grid_h x grid_w` patch grid,
//!   plus `IMAGE_POSITION_OFFSET` so text/image position spaces never collide.
//! - indicator: LLM(3) for text, OUTPUT_IMAGE(2) for image.
//! - segment_ids: all 1 (one sample, no pad), so the block-diagonal attention
//!   mask is full bidirectional attention -> the DiT needs NO mask buffer.
//!
//! The grid is `H/(patch*ae) x W/(patch*ae)` = `/16`. Velocity is read back at
//! the image positions only; the unpatch is `(grid_h, grid_w, 2, 2, 32)`.

use super::config;

/// Image patch-grid dims and token count for a target resolution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageGrid {
    pub grid_h: usize,
    pub grid_w: usize,
}

impl ImageGrid {
    pub fn num_image_tokens(&self) -> usize {
        self.grid_h * self.grid_w
    }
}

/// `H/(patch*ae)` x `W/(patch*ae)`. Errors if not divisible by the patch.
pub fn image_grid(height: usize, width: usize) -> Result<ImageGrid, String> {
    let patch = config::PATCH_SIZE * config::AE_SCALE_FACTOR; // 16
    if !height.is_multiple_of(patch) || !width.is_multiple_of(patch) {
        return Err(format!(
            "height/width must be divisible by patch_size*ae_scale_factor={patch} (got {height}x{width})"
        ));
    }
    Ok(ImageGrid {
        grid_h: height / patch,
        grid_w: width / patch,
    })
}

/// `[num_text + num_image, 3]` row-major `(t,h,w)` position ids for the packed
/// stream `[text][image]`.
pub fn build_position_ids(num_text: usize, grid: ImageGrid) -> Vec<i64> {
    let num_image = grid.num_image_tokens();
    let total = num_text + num_image;
    let mut pos = vec![0_i64; total * 3];
    for p in 0..num_text {
        let o = p * 3;
        pos[o] = p as i64;
        pos[o + 1] = p as i64;
        pos[o + 2] = p as i64;
    }
    let off = config::IMAGE_POSITION_OFFSET;
    for r in 0..grid.grid_h {
        for c in 0..grid.grid_w {
            let img_idx = r * grid.grid_w + c;
            let o = (num_text + img_idx) * 3;
            pos[o] = off; // t = 0 + offset
            pos[o + 1] = r as i64 + off;
            pos[o + 2] = c as i64 + off;
        }
    }
    pos
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_divisibility() {
        assert_eq!(
            image_grid(1024, 1024).unwrap(),
            ImageGrid {
                grid_h: 64,
                grid_w: 64
            }
        );
        assert_eq!(
            image_grid(512, 768).unwrap(),
            ImageGrid {
                grid_h: 32,
                grid_w: 48
            }
        );
        assert!(image_grid(1000, 1024).is_err());
    }

    #[test]
    fn text_positions_are_ppp() {
        let grid = ImageGrid {
            grid_h: 2,
            grid_w: 2,
        };
        let pos = build_position_ids(3, grid);
        // 3 text + 4 image = 7 tokens.
        assert_eq!(pos.len(), 7 * 3);
        for p in 0..3 {
            assert_eq!(&pos[p * 3..p * 3 + 3], &[p as i64, p as i64, p as i64]);
        }
    }

    #[test]
    fn image_positions_offset_and_grid_ordered() {
        let off = config::IMAGE_POSITION_OFFSET;
        let grid = ImageGrid {
            grid_h: 2,
            grid_w: 3,
        };
        let pos = build_position_ids(1, grid);
        // image token (r=1,c=2) is img_idx=5 -> seq index 1+5=6.
        let o = 6 * 3;
        assert_eq!(&pos[o..o + 3], &[off, 1 + off, 2 + off]);
        // first image token (r=0,c=0) follows the 1 text token -> all = offset.
        let o0 = 3; // seq index 1 * 3 axes
        assert_eq!(&pos[o0..o0 + 3], &[off, off, off]);
    }
}
