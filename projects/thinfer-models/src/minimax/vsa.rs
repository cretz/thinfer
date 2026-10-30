//! Tile geometry for video sparse attention (VSA) over the packed sequence,
//! the attention FastH3 was distilled with (FastVideo
//! `minimax_h3_vsa.build_h3_segment_geometry`): every dense segment (text,
//! audio) is chunked into its own 64-row tiles, then the video grid into
//! `4 x 4 x 4` latent cubes (ragged at the edges), tile-major in `(t, h, w)`.
//! The kernels gather rows through this table, so the packed sequence itself
//! is never permuted.

/// Rows per tile, and the video cube that fills one.
pub const TILE: usize = 64;
pub const TILE_SHAPE: [usize; 3] = [4, 4, 4];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VsaGeometry {
    /// `n_tiles * TILE` packed row ids (slots past a tile's size are 0).
    pub tile_rows: Vec<u32>,
    pub tile_sizes: Vec<u32>,
    pub n_prefix: usize,
}

impl VsaGeometry {
    /// `prefix`: the dense segments as `(first_row, rows)` in packed order;
    /// `video_at`: the first video row; `grid`: the video rows' `(t, h, w)`
    /// extent, raster order.
    pub fn new(prefix: &[(usize, usize)], video_at: usize, grid: [usize; 3]) -> Self {
        let mut g = Self {
            tile_rows: Vec::new(),
            tile_sizes: Vec::new(),
            n_prefix: 0,
        };
        for &(at, n) in prefix {
            let rows: Vec<u32> = (at..at + n).map(|r| r as u32).collect();
            for chunk in rows.chunks(TILE) {
                g.push(chunk);
            }
        }
        g.n_prefix = g.n_tiles();
        let [t, h, w] = grid;
        let [st, sh, sw] = TILE_SHAPE;
        for t0 in (0..t).step_by(st) {
            for h0 in (0..h).step_by(sh) {
                for w0 in (0..w).step_by(sw) {
                    let mut rows = Vec::with_capacity(TILE);
                    for tt in t0..(t0 + st).min(t) {
                        for hh in h0..(h0 + sh).min(h) {
                            for ww in w0..(w0 + sw).min(w) {
                                rows.push((video_at + (tt * h + hh) * w + ww) as u32);
                            }
                        }
                    }
                    g.push(&rows);
                }
            }
        }
        g
    }

    fn push(&mut self, rows: &[u32]) {
        debug_assert!(!rows.is_empty() && rows.len() <= TILE);
        self.tile_sizes.push(rows.len() as u32);
        self.tile_rows.extend_from_slice(rows);
        self.tile_rows.resize(self.tile_sizes.len() * TILE, 0);
    }

    pub fn n_tiles(&self) -> usize {
        self.tile_sizes.len()
    }

    pub fn n_video(&self) -> usize {
        self.n_tiles() - self.n_prefix
    }

    /// Video tiles each video query keeps at `sparsity` (the reference's
    /// `compute_topk`: `ceil((1 - sparsity) * n)` clamped to `[1, n]`).
    pub fn topk(&self, sparsity: f64) -> usize {
        let n = self.n_video();
        (((1.0 - sparsity) * n as f64).ceil() as usize).clamp(1.min(n), n)
    }

    /// The device table: row ids, then sizes, little-endian u32.
    pub fn bytes(&self) -> Vec<u8> {
        self.tile_rows
            .iter()
            .chain(&self.tile_sizes)
            .flat_map(|v| v.to_le_bytes())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segments_and_cubes_match_the_reference_counts() {
        // 640x352x124 t2va: 32 text + 207 audio rows, then a 37x11x20 grid.
        let g = VsaGeometry::new(&[(0, 32), (32, 207)], 239, [37, 11, 20]);
        // Text: one 32-row tile. Audio: 64, 64, 64, 15.
        assert_eq!(g.n_prefix, 5);
        assert_eq!(&g.tile_sizes[..5], &[32, 64, 64, 64, 15]);
        // ceil(37/4) x ceil(11/4) x ceil(20/4) cubes.
        assert_eq!(g.n_video(), 10 * 3 * 5);
        assert_eq!(g.topk(0.8), 30);
        // Every packed row lands in exactly one tile.
        let mut seen = vec![0u32; 239 + 37 * 11 * 20];
        for (i, &n) in g.tile_sizes.iter().enumerate() {
            for &r in &g.tile_rows[i * TILE..i * TILE + n as usize] {
                seen[r as usize] += 1;
            }
        }
        assert!(seen.iter().all(|&c| c == 1));
        // The last cube is ragged on every axis: 1 x 3 x 4.
        assert_eq!(*g.tile_sizes.last().unwrap(), 12);
    }

    #[test]
    fn cube_rows_are_raster_within_the_cube() {
        let g = VsaGeometry::new(&[], 0, [4, 4, 8]);
        assert_eq!(g.n_prefix, 0);
        assert_eq!(g.n_tiles(), 2);
        // First cube: w 0..4 of every (t, h).
        assert_eq!(&g.tile_rows[..5], &[0, 1, 2, 3, 8]);
        // Second cube starts at w = 4.
        assert_eq!(g.tile_rows[TILE], 4);
    }
}
