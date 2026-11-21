//! SCRFD face detection: letterbox -> run -> distance-decode -> NMS. Mirrors
//! intabai `pipeline.ts::detectFacesScrfd`. Produces faces with a bbox + 5
//! landmarks in original-image pixel space.

use std::collections::HashMap;

use thinfer_core::onnx::{OnnxError, OnnxModel};

use super::image::Image;

#[derive(Clone, Debug)]
pub struct Face {
    /// `[x1, y1, x2, y2]` in original-image pixels.
    pub bbox: [f32; 4],
    pub score: f32,
    /// 5 landmarks (left eye, right eye, nose, mouth-L, mouth-R), image pixels.
    pub landmark5: [[f32; 2]; 5],
}

const DETECTOR_SIZE: usize = 640;
const STRIDES: [usize; 3] = [8, 16, 32];
const NUM_ANCHORS: usize = 2;
const SCORE_THRESHOLD: f32 = 0.5;
const NMS_IOU: f32 = 0.4;

/// Load the SCRFD detector on its fixed letterbox input.
pub async fn load(
    backend: std::sync::Arc<thinfer_core::backend::WgpuBackend>,
    onnx: &[u8],
) -> Result<OnnxModel, OnnxError> {
    let shapes = super::input_shape_map(onnx, &[1, 3, DETECTOR_SIZE as i64, DETECTOR_SIZE as i64])?;
    OnnxModel::load(backend, onnx, &shapes).await
}

/// Detect faces in `img`. Returns NMS-filtered faces sorted by descending score.
pub async fn detect(model: &OnnxModel, img: &Image) -> Result<Vec<Face>, OnnxError> {
    // Letterbox: scale preserving aspect ratio, origin top-left, zero pad.
    let scale = (DETECTOR_SIZE as f32 / img.w as f32).min(DETECTOR_SIZE as f32 / img.h as f32);
    let new_w = (img.w as f32 * scale).round() as usize;
    let new_h = (img.h as f32 * scale).round() as usize;
    let resized = img.resize(new_w.max(1), new_h.max(1));

    // NCHW BGR, normalized (p - 127.5) / 128, zero-padded to 640x640.
    let hw = DETECTOR_SIZE * DETECTOR_SIZE;
    let mut input = vec![0.0f32; 3 * hw];
    for y in 0..new_h {
        for x in 0..new_w {
            let si = (y * resized.w + x) * 3;
            let di = y * DETECTOR_SIZE + x;
            // channel order BGR: out0=B(src2), out1=G(src1), out2=R(src0)
            input[di] = (resized.data[si + 2] - 127.5) / 128.0;
            input[hw + di] = (resized.data[si + 1] - 127.5) / 128.0;
            input[2 * hw + di] = (resized.data[si] - 127.5) / 128.0;
        }
    }

    let mut feeds = HashMap::new();
    feeds.insert(model.input_names[0].clone(), input);
    let out = model.run(&feeds).await?;

    // Group outputs by stride (from row count) and kind (from last dim): scores
    // last-dim 1, bbox 4, kps 10. SCRFD output names are numeric, so key on shape.
    // Per stride: (scores, bbox, kps).
    type StrideHeads = (Vec<f32>, Vec<f32>, Vec<f32>);
    let mut by_stride: HashMap<usize, StrideHeads> = HashMap::new();
    for (shape, data) in out.values() {
        let rows = shape[0] as usize;
        let last = *shape.last().unwrap() as usize;
        let stride = match rows {
            r if r == (DETECTOR_SIZE / 8) * (DETECTOR_SIZE / 8) * NUM_ANCHORS => 8,
            r if r == (DETECTOR_SIZE / 16) * (DETECTOR_SIZE / 16) * NUM_ANCHORS => 16,
            r if r == (DETECTOR_SIZE / 32) * (DETECTOR_SIZE / 32) * NUM_ANCHORS => 32,
            _ => continue,
        };
        let entry = by_stride.entry(stride).or_insert((vec![], vec![], vec![]));
        match last {
            1 => entry.0 = data.clone(),
            4 => entry.1 = data.clone(),
            10 => entry.2 = data.clone(),
            _ => {}
        }
    }

    let ratio_w = img.w as f32 / new_w as f32;
    let ratio_h = img.h as f32 / new_h as f32;
    let mut faces: Vec<Face> = Vec::new();
    for stride in STRIDES {
        let Some((scores, bbox, kps)) = by_stride.get(&stride) else {
            continue;
        };
        let grid = DETECTOR_SIZE / stride;
        let n = grid * grid * NUM_ANCHORS;
        let s = stride as f32;
        for k in 0..n {
            let score = scores[k];
            if score < SCORE_THRESHOLD {
                continue;
            }
            let cell = k / NUM_ANCHORS;
            let col = (cell % grid) as f32;
            let row = (cell / grid) as f32;
            let ax = col * s;
            let ay = row * s;
            let x1 = (ax - bbox[k * 4] * s) * ratio_w;
            let y1 = (ay - bbox[k * 4 + 1] * s) * ratio_h;
            let x2 = (ax + bbox[k * 4 + 2] * s) * ratio_w;
            let y2 = (ay + bbox[k * 4 + 3] * s) * ratio_h;
            let mut landmark5 = [[0.0f32; 2]; 5];
            for (i, lm) in landmark5.iter_mut().enumerate() {
                lm[0] = (ax + kps[k * 10 + i * 2] * s) * ratio_w;
                lm[1] = (ay + kps[k * 10 + i * 2 + 1] * s) * ratio_h;
            }
            faces.push(Face {
                bbox: [x1, y1, x2, y2],
                score,
                landmark5,
            });
        }
    }
    Ok(nms(faces, NMS_IOU))
}

/// [`detect`] that also finds faces too small for one whole-image pass (which
/// shrinks `img` to 640): the whole image, plus overlapping 640 tiles at
/// `img`'s own resolution, merged by NMS. Cost grows with `img`'s area, so
/// callers bound its size first.
pub async fn detect_tiled(model: &OnnxModel, img: &Image) -> Result<Vec<Face>, OnnxError> {
    let mut faces = detect(model, img).await?;
    let (w, h) = (DETECTOR_SIZE.min(img.w), DETECTOR_SIZE.min(img.h));
    if (w, h) == (img.w, img.h) {
        return Ok(faces); // One tile would be the whole image.
    }
    // A face cut by an inner tile edge is seen whole by its neighbour (the
    // overlap is wider than the faces this pass is for): drop the partial box.
    let margin = 2.0;
    for (x, y) in tile_origins(img.w, DETECTOR_SIZE)
        .flat_map(|x| tile_origins(img.h, DETECTOR_SIZE).map(move |y| (x, y)))
    {
        for mut f in detect(model, &img.crop(x, y, w, h)).await? {
            let [x1, y1, x2, y2] = f.bbox;
            let cut = (x > 0 && x1 < margin)
                || (y > 0 && y1 < margin)
                || (x + w < img.w && x2 > w as f32 - margin)
                || (y + h < img.h && y2 > h as f32 - margin);
            if cut {
                continue;
            }
            let (dx, dy) = (x as f32, y as f32);
            f.bbox = [
                f.bbox[0] + dx,
                f.bbox[1] + dy,
                f.bbox[2] + dx,
                f.bbox[3] + dy,
            ];
            for lm in &mut f.landmark5 {
                lm[0] += dx;
                lm[1] += dy;
            }
            faces.push(f);
        }
    }
    Ok(nms(faces, NMS_IOU))
}

/// Tile starts along an edge of `len` for `tile`-wide tiles overlapping by a
/// quarter, the last flush with the far edge (a single 0 when it fits).
fn tile_origins(len: usize, tile: usize) -> impl Iterator<Item = usize> {
    let step = tile - tile / 4;
    let last = len.saturating_sub(tile);
    let n = last.div_ceil(step) + 1;
    (0..n).map(move |i| (i * step).min(last))
}

fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let x1 = a[0].max(b[0]);
    let y1 = a[1].max(b[1]);
    let x2 = a[2].min(b[2]);
    let y2 = a[3].min(b[3]);
    let inter = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let area_a = (a[2] - a[0]) * (a[3] - a[1]);
    let area_b = (b[2] - b[0]) * (b[3] - b[1]);
    inter / (area_a + area_b - inter)
}

fn nms(mut faces: Vec<Face>, thr: f32) -> Vec<Face> {
    faces.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut keep: Vec<Face> = Vec::new();
    'outer: for f in faces {
        for k in &keep {
            if iou(&f.bbox, &k.bbox) > thr {
                continue 'outer;
            }
        }
        keep.push(f);
    }
    keep
}

#[cfg(test)]
mod tests {
    use super::tile_origins;

    #[test]
    fn tiles_cover_the_edge_with_overlap() {
        assert_eq!(tile_origins(640, 640).collect::<Vec<_>>(), [0]);
        assert_eq!(tile_origins(300, 640).collect::<Vec<_>>(), [0]);
        for len in [641, 1000, 1440, 1920, 2561] {
            let t: Vec<usize> = tile_origins(len, 640).collect();
            assert_eq!(t[0], 0);
            assert_eq!(*t.last().unwrap(), len - 640, "flush with the far edge");
            // Consecutive tiles overlap by at least a quarter tile.
            assert!(
                t.windows(2)
                    .all(|p| p[1] > p[0] && p[0] + 640 - p[1] >= 160)
            );
        }
    }
}
