//! Tile geometry for video sparse attention (VSA) over the packed sequence,
//! the attention FastH3 was distilled with (FastVideo
//! `minimax_h3_vsa.build_h3_segment_geometry`): every dense segment (text,
//! audio, reference images) is chunked into its own tiles, then the video grid
//! into latent cubes (ragged at the edges), tile-major in `(t, h, w)`. The
//! kernels gather rows through this table, so the packed sequence itself is
//! never permuted.

/// Rows per tile and the video cube that fills one (`vsa_tile_size`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VsaTiling {
    pub rows: usize,
    pub cube: [usize; 3],
}

impl VsaTiling {
    /// FastH3-Trim, and the 64-row tiles the VAE / vision towers attend in.
    pub const T64: Self = Self {
        rows: 64,
        cube: [4, 4, 4],
    };
    /// FastH3-Omni (`vsa_tile_size` 128).
    pub const T128: Self = Self {
        rows: 128,
        cube: [4, 4, 8],
    };
}

/// A run of whole tiles covering one contiguous packed-row range.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Slab {
    pub rows: std::ops::Range<usize>,
    pub tiles: std::ops::Range<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VsaGeometry {
    pub tiling: VsaTiling,
    /// `n_tiles * tiling.rows` packed row ids (slots past a tile's size are 0).
    pub tile_rows: Vec<u32>,
    pub tile_sizes: Vec<u32>,
    pub n_prefix: usize,
}

impl VsaGeometry {
    /// `prefix`: the dense segments as `(first_row, rows)` in packed order;
    /// `video_at`: the first video row; `grid`: the video rows' `(t, h, w)`
    /// extent, raster order.
    pub fn new(
        tiling: VsaTiling,
        prefix: &[(usize, usize)],
        video_at: usize,
        grid: [usize; 3],
    ) -> Self {
        let mut g = Self {
            tiling,
            tile_rows: Vec::new(),
            tile_sizes: Vec::new(),
            n_prefix: 0,
        };
        for &(at, n) in prefix {
            let rows: Vec<u32> = (at..at + n).map(|r| r as u32).collect();
            for chunk in rows.chunks(tiling.rows) {
                g.push(chunk);
            }
        }
        g.n_prefix = g.n_tiles();
        let [t, h, w] = grid;
        let [st, sh, sw] = tiling.cube;
        for t0 in (0..t).step_by(st) {
            for h0 in (0..h).step_by(sh) {
                for w0 in (0..w).step_by(sw) {
                    let mut rows = Vec::with_capacity(tiling.rows);
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
        debug_assert!(!rows.is_empty() && rows.len() <= self.tiling.rows);
        self.tile_sizes.push(rows.len() as u32);
        self.tile_rows.extend_from_slice(rows);
        self.tile_rows
            .resize(self.tile_sizes.len() * self.tiling.rows, 0);
    }

    pub fn n_tiles(&self) -> usize {
        self.tile_sizes.len()
    }

    pub fn n_video(&self) -> usize {
        self.n_tiles() - self.n_prefix
    }

    /// The packed rows of tile `t`.
    pub fn rows_of(&self, t: usize) -> &[u32] {
        let at = t * self.tiling.rows;
        &self.tile_rows[at..at + self.tile_sizes[t] as usize]
    }

    /// Video tiles each video query keeps at `sparsity` (the reference's
    /// `compute_topk`: `ceil((1 - sparsity) * n)` clamped to `[1, n]`).
    pub fn topk(&self, sparsity: f64) -> usize {
        let n = self.n_video();
        (((1.0 - sparsity) * n as f64).ceil() as usize).clamp(1.min(n), n)
    }

    /// Consecutive tile groups that each own one contiguous packed-row range,
    /// in order, covering every tile: the units the block can run row-local
    /// work and query-side attention over without materializing the whole
    /// sequence. The smallest such groups (a prefix tile, or a whole video
    /// time slab: cubes of one slab interleave their rows) are packed in order
    /// up to `max_rows` (one larger than that stands alone).
    pub fn slabs(&self, max_rows: usize) -> Vec<Slab> {
        let mut atoms = Vec::new();
        let (mut t0, mut lo, mut hi, mut count) = (0, usize::MAX, 0, 0);
        for t in 0..self.n_tiles() {
            let rows = self.rows_of(t);
            lo = lo.min(rows.iter().copied().min().unwrap_or(0) as usize);
            hi = hi.max(rows.iter().copied().max().unwrap_or(0) as usize + 1);
            count += rows.len();
            if count == hi - lo {
                atoms.push(Slab {
                    rows: lo..hi,
                    tiles: t0..t + 1,
                });
                (t0, lo, hi, count) = (t + 1, usize::MAX, 0, 0);
            }
        }
        let mut out: Vec<Slab> = Vec::new();
        for a in atoms {
            match out.last_mut() {
                Some(s) if s.rows.len() + a.rows.len() <= max_rows => {
                    s.rows.end = a.rows.end;
                    s.tiles.end = a.tiles.end;
                }
                _ => out.push(a),
            }
        }
        out
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
        let g = VsaGeometry::new(VsaTiling::T64, &[(0, 32), (32, 207)], 239, [37, 11, 20]);
        // Text: one 32-row tile. Audio: 64, 64, 64, 15.
        assert_eq!(g.n_prefix, 5);
        assert_eq!(&g.tile_sizes[..5], &[32, 64, 64, 64, 15]);
        // ceil(37/4) x ceil(11/4) x ceil(20/4) cubes.
        assert_eq!(g.n_video(), 10 * 3 * 5);
        assert_eq!(g.topk(0.8), 30);
        // Every packed row lands in exactly one tile.
        let mut seen = vec![0u32; 239 + 37 * 11 * 20];
        for t in 0..g.n_tiles() {
            for &r in g.rows_of(t) {
                seen[r as usize] += 1;
            }
        }
        assert!(seen.iter().all(|&c| c == 1));
        // The last cube is ragged on every axis: 1 x 3 x 4.
        assert_eq!(*g.tile_sizes.last().unwrap(), 12);
    }

    #[test]
    fn slabs_tile_the_sequence_in_contiguous_row_ranges() {
        // 832x480 5s Omni: 200 prefix rows, a 37x15x26 grid (time slabs of 4
        // frames = 1560 rows, the last of 1 frame).
        let g = VsaGeometry::new(VsaTiling::T128, &[(0, 200)], 200, [37, 15, 26]);
        // At any cap: rows and tiles covered once, in order, every tile's rows
        // inside its slab, and each slab's tiles filling its rows exactly.
        for cap in [1, 1500, 3200, usize::MAX] {
            let (mut row, mut tile) = (0, 0);
            for s in g.slabs(cap) {
                assert_eq!((s.rows.start, s.tiles.start), (row, tile), "cap {cap}");
                let mut n = 0;
                for t in s.tiles.clone() {
                    let rows = g.rows_of(t);
                    assert!(rows.iter().all(|&r| s.rows.contains(&(r as usize))));
                    n += rows.len();
                }
                assert_eq!(n, s.rows.len(), "cap {cap}");
                (row, tile) = (s.rows.end, s.tiles.end);
            }
            assert_eq!((row, tile), (200 + 37 * 15 * 26, g.n_tiles()));
        }
        // The prefix tiles (128 + 72) pack together; a 1560-row slab would
        // pass 1500, so it stands alone.
        let slabs = g.slabs(1500);
        assert_eq!(slabs[0].rows, 0..200);
        assert_eq!(slabs[1].rows, 200..1760);
        // Under 3200: prefix + one slab, then two slabs per group.
        let wide = g.slabs(3200);
        assert_eq!(
            (wide[0].rows.clone(), wide[1].rows.clone()),
            (0..1760, 1760..1760 + 2 * 1560)
        );
    }

    #[test]
    fn cube_rows_are_raster_within_the_cube() {
        let g = VsaGeometry::new(VsaTiling::T64, &[], 0, [4, 4, 8]);
        assert_eq!(g.n_prefix, 0);
        assert_eq!(g.n_tiles(), 2);
        // First cube: w 0..4 of every (t, h).
        assert_eq!(&g.tile_rows[..5], &[0, 1, 2, 3, 8]);
        // Second cube starts at w = 4.
        assert_eq!(g.tile_rows[64], 4);
    }

    #[test]
    fn omni_tiles_are_128_row_4x4x8_cubes() {
        // 832x480 5s: 37x15x26 grid; 200 prefix rows = 128 + 72.
        let g = VsaGeometry::new(VsaTiling::T128, &[(0, 200)], 200, [37, 15, 26]);
        assert_eq!(&g.tile_sizes[..2], &[128, 72]);
        assert_eq!(g.n_video(), 10 * 4 * 4);
        assert_eq!(g.topk(0.9), 16);
        // Full cube: w 0..8 of every (t, h).
        assert_eq!(g.tile_sizes[2], 128);
        assert_eq!(
            &g.rows_of(2)[..9],
            &[200, 201, 202, 203, 204, 205, 206, 207, 226]
        );
        // Last cube ragged: t 1, h 3, w 2.
        assert_eq!(*g.tile_sizes.last().unwrap(), 6);
    }
}
