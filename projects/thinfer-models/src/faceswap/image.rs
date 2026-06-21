//! Host-side image + affine geometry for the face-swap pipeline. Ported from
//! intabai `web/src/video-face-swap/cv.ts` (the OpenCV.js-free helpers).
//!
//! Images are HWC interleaved RGB `f32` in `[0, 255]`. The detector/embedder/
//! swapper run on the GPU; everything here (warp, paste-back, masks) is cheap
//! host work that the per-frame swap interleaves with GPU dispatch.

/// HWC interleaved RGB image, `f32` in `[0, 255]`.
#[derive(Clone)]
pub struct Image {
    pub w: usize,
    pub h: usize,
    /// `w * h * 3`, row-major, channel-interleaved (R,G,B).
    pub data: Vec<f32>,
}

impl Image {
    pub fn new(w: usize, h: usize) -> Self {
        Self {
            w,
            h,
            data: vec![0.0; w * h * 3],
        }
    }

    /// Build from interleaved RGB u8 (e.g. a decoded video frame).
    pub fn from_rgb8(w: usize, h: usize, rgb: &[u8]) -> Self {
        assert_eq!(rgb.len(), w * h * 3);
        Self {
            w,
            h,
            data: rgb.iter().map(|&b| b as f32).collect(),
        }
    }

    /// Interleaved RGB u8 copy (rounded + clamped), for encode / paste sinks.
    pub fn to_rgb8(&self) -> Vec<u8> {
        self.data
            .iter()
            .map(|&v| v.clamp(0.0, 255.0).round() as u8)
            .collect()
    }

    #[inline]
    fn at(&self, x: usize, y: usize, c: usize) -> f32 {
        self.data[(y * self.w + x) * 3 + c]
    }

    /// Bilinear sample with border-replicate, returning `[r, g, b]`.
    fn sample(&self, fx: f32, fy: f32) -> [f32; 3] {
        let sx = fx.clamp(0.0, (self.w - 1) as f32);
        let sy = fy.clamp(0.0, (self.h - 1) as f32);
        let x0 = sx.floor() as usize;
        let y0 = sy.floor() as usize;
        let x1 = (x0 + 1).min(self.w - 1);
        let y1 = (y0 + 1).min(self.h - 1);
        let dx = sx - x0 as f32;
        let dy = sy - y0 as f32;
        let mut out = [0.0f32; 3];
        for (c, o) in out.iter_mut().enumerate() {
            *o = self.at(x0, y0, c) * (1.0 - dx) * (1.0 - dy)
                + self.at(x1, y0, c) * dx * (1.0 - dy)
                + self.at(x0, y1, c) * (1.0 - dx) * dy
                + self.at(x1, y1, c) * dx * dy;
        }
        out
    }

    /// Bilinear resize to `(nw, nh)`.
    pub fn resize(&self, nw: usize, nh: usize) -> Image {
        let mut out = Image::new(nw, nh);
        // Map output pixel centers back to input (align_corners=false style:
        // src = (o + 0.5) * (in/out) - 0.5), matching browser drawImage scaling.
        let sx = self.w as f32 / nw as f32;
        let sy = self.h as f32 / nh as f32;
        for y in 0..nh {
            let fy = (y as f32 + 0.5) * sy - 0.5;
            for x in 0..nw {
                let fx = (x as f32 + 0.5) * sx - 0.5;
                let p = self.sample(fx, fy);
                let di = (y * nw + x) * 3;
                out.data[di..di + 3].copy_from_slice(&p);
            }
        }
        out
    }
}

/// A 2x3 affine `[a, b, tx, c, d, ty]`: `x' = a*x + b*y + tx`, `y' = c*x + d*y + ty`.
pub type Affine = [f32; 6];

/// Estimate a similarity transform (rotation + uniform scale + translation)
/// from `src` to `dst` points by least squares (cv.ts `estimateSimilarityTransform`).
/// Returns the forward matrix `src -> dst`.
pub fn estimate_similarity_transform(src: &[[f32; 2]], dst: &[[f32; 2]]) -> Affine {
    let n = src.len();
    let (mut s00, mut s02, mut s03) = (0.0f64, 0.0f64, 0.0f64);
    let (mut b0, mut b1, mut b2, mut b3) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        let (sx, sy) = (src[i][0] as f64, src[i][1] as f64);
        let (dx, dy) = (dst[i][0] as f64, dst[i][1] as f64);
        s00 += sx * sx + sy * sy;
        s02 += sx;
        s03 += sy;
        b0 += sx * dx + sy * dy;
        b1 += -sy * dx + sx * dy;
        b2 += dx;
        b3 += dy;
    }
    let a = [
        [s00, 0.0, s02, s03],
        [0.0, s00, -s03, s02],
        [s02, -s03, n as f64, 0.0],
        [s03, s02, 0.0, n as f64],
    ];
    let x = solve_linear_4x4(a, [b0, b1, b2, b3]);
    let (av, bv, tx, ty) = (x[0], x[1], x[2], x[3]);
    [
        av as f32, -bv as f32, tx as f32, bv as f32, av as f32, ty as f32,
    ]
}

#[allow(clippy::needless_range_loop)] // augmented-matrix index math reads clearer indexed
fn solve_linear_4x4(a: [[f64; 4]; 4], b: [f64; 4]) -> [f64; 4] {
    let mut aug = [[0.0f64; 5]; 4];
    for i in 0..4 {
        aug[i][..4].copy_from_slice(&a[i]);
        aug[i][4] = b[i];
    }
    for col in 0..4 {
        let mut max_row = col;
        for row in (col + 1)..4 {
            if aug[row][col].abs() > aug[max_row][col].abs() {
                max_row = row;
            }
        }
        aug.swap(col, max_row);
        let pivot = aug[col][col];
        if pivot.abs() < 1e-12 {
            continue;
        }
        for row in (col + 1)..4 {
            let f = aug[row][col] / pivot;
            for j in col..=4 {
                aug[row][j] -= f * aug[col][j];
            }
        }
    }
    let mut x = [0.0f64; 4];
    for i in (0..4).rev() {
        x[i] = aug[i][4];
        for j in (i + 1)..4 {
            x[i] -= aug[i][j] * x[j];
        }
        x[i] /= aug[i][i];
    }
    x
}

/// Invert a 2x3 affine.
pub fn invert_affine(m: &Affine) -> Affine {
    let [a, b, tx, c, d, ty] = *m;
    let det = a * d - b * c;
    let id = 1.0 / det;
    [
        d * id,
        -b * id,
        (b * ty - d * tx) * id,
        -c * id,
        a * id,
        (c * tx - a * ty) * id,
    ]
}

/// Warp `src` by forward affine `matrix` (src->dst) into a `(w, h)` output with
/// bilinear sampling + border-replicate.
pub fn warp_affine(src: &Image, matrix: &Affine, w: usize, h: usize) -> Image {
    let inv = invert_affine(matrix);
    let mut out = Image::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let fx = inv[0] * x as f32 + inv[1] * y as f32 + inv[2];
            let fy = inv[3] * x as f32 + inv[4] * y as f32 + inv[5];
            let p = src.sample(fx, fy);
            let di = (y * w + x) * 3;
            out.data[di..di + 3].copy_from_slice(&p);
        }
    }
    out
}

/// Feathered elliptical mask (cv.ts `createFeatheredMask`). `w*h` in `[0, 1]`.
pub fn feathered_mask(w: usize, h: usize, erode_x: f32, erode_y: f32) -> Vec<f32> {
    let mut mask = vec![0.0f32; w * h];
    let cx = w as f32 / 2.0;
    let cy = h as f32 / 2.0;
    let rx = w as f32 / 2.0 - erode_x;
    let ry = h as f32 / 2.0 - erode_y;
    let blur = erode_x.min(erode_y);
    for y in 0..h {
        for x in 0..w {
            let dx = (x as f32 - cx) / rx;
            let dy = (y as f32 - cy) / ry;
            let dist = (dx * dx + dy * dy).sqrt();
            mask[y * w + x] = if dist <= 1.0 {
                1.0
            } else {
                (1.0 - (dist - 1.0) * (rx / blur)).max(0.0)
            };
        }
    }
    mask
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A known scale+rotation+translation is recovered by the least-squares
    /// similarity solver, and maps the source points onto the destination.
    #[test]
    fn similarity_recovers_known_transform() {
        // dst = R(30deg)*1.5*src + (10, -5).
        let theta: f32 = 30f32.to_radians();
        let (s, c) = (theta.sin(), theta.cos());
        let scale = 1.5f32;
        let src = [[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0], [5.0, 3.0]];
        let mut dst = [[0.0f32; 2]; 5];
        for (i, p) in src.iter().enumerate() {
            dst[i][0] = scale * (c * p[0] - s * p[1]) + 10.0;
            dst[i][1] = scale * (s * p[0] + c * p[1]) - 5.0;
        }
        let m = estimate_similarity_transform(&src, &dst);
        for (i, p) in src.iter().enumerate() {
            let x = m[0] * p[0] + m[1] * p[1] + m[2];
            let y = m[3] * p[0] + m[4] * p[1] + m[5];
            assert!((x - dst[i][0]).abs() < 1e-3, "x[{i}] {x} vs {}", dst[i][0]);
            assert!((y - dst[i][1]).abs() < 1e-3, "y[{i}] {y} vs {}", dst[i][1]);
        }
    }

    /// invert_affine composes to identity.
    #[test]
    fn invert_affine_roundtrip() {
        let m = [1.3, -0.4, 12.0, 0.4, 1.3, -7.0];
        let inv = invert_affine(&m);
        // Forward then inverse maps a point back to itself.
        let (px, py) = (3.0f32, 9.0f32);
        let fx = m[0] * px + m[1] * py + m[2];
        let fy = m[3] * px + m[4] * py + m[5];
        let bx = inv[0] * fx + inv[1] * fy + inv[2];
        let by = inv[3] * fx + inv[4] * fy + inv[5];
        assert!((bx - px).abs() < 1e-3 && (by - py).abs() < 1e-3);
    }

    /// Warp by identity reproduces the source (interior pixels, exact-ish).
    #[test]
    fn warp_identity() {
        let mut img = Image::new(8, 8);
        for i in 0..img.data.len() {
            img.data[i] = (i % 256) as f32;
        }
        let id = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let out = warp_affine(&img, &id, 8, 8);
        for i in 0..img.data.len() {
            assert!((out.data[i] - img.data[i]).abs() < 1e-3);
        }
    }
}

/// Paste `crop` back onto `frame` using the forward affine `matrix` (frame->crop)
/// with a feathered elliptical mask (cv.ts `pasteBack`, no occlusion mask).
pub fn paste_back(frame: &Image, crop: &Image, matrix: &Affine) -> Image {
    let (fw, fh) = (frame.w, frame.h);
    let (cw, ch) = (crop.w, crop.h);
    let mask = feathered_mask(cw, ch, 15.0, 15.0);
    let mut out = frame.clone();
    let m = matrix;
    for y in 0..fh {
        for x in 0..fw {
            let cx = m[0] * x as f32 + m[1] * y as f32 + m[2];
            let cy = m[3] * x as f32 + m[4] * y as f32 + m[5];
            if cx < 0.0 || cx >= (cw - 1) as f32 || cy < 0.0 || cy >= (ch - 1) as f32 {
                continue;
            }
            let x0 = cx.floor() as usize;
            let y0 = cy.floor() as usize;
            let x1 = x0 + 1;
            let y1 = y0 + 1;
            let fx = cx - x0 as f32;
            let fy = cy - y0 as f32;
            let m00 = mask[y0 * cw + x0];
            let m10 = mask[y0 * cw + x1];
            let m01 = mask[y1 * cw + x0];
            let m11 = mask[y1 * cw + x1];
            let alpha = m00 * (1.0 - fx) * (1.0 - fy)
                + m10 * fx * (1.0 - fy)
                + m01 * (1.0 - fx) * fy
                + m11 * fx * fy;
            if alpha < 0.001 {
                continue;
            }
            let cp = crop.sample(cx, cy);
            let di = (y * fw + x) * 3;
            for (c, &v) in cp.iter().enumerate() {
                out.data[di + c] = out.data[di + c] * (1.0 - alpha) + v * alpha;
            }
        }
    }
    out
}
