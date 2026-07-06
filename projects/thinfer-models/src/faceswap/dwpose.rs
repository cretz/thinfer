//! DWPose face-mask preprocessing for DreamID-V.
//!
//! Derives a per-frame binary FACE MASK video from a target video, matching the
//! reference `DreamID-V/pose/extract.py` pipeline:
//!   1. `yolox_l` person detection on the (letterboxed) frame -> person bboxes.
//!   2. `dw-ll_ucoco_384` (RTMPose whole-body) keypoints on each person crop ->
//!      133 COCO-wholebody keypoints via SimCC (two 1D heatmaps, argmax).
//!   3. Face keypoints (`candidate[24:92]`, the 68 face points) with score > 0.3.
//!   4. Convex hull of the largest face, filled white, dilated by a 15x15 kernel.
//!
//! Both ONNX nets run on the thinfer GPU executor (`thinfer_core::onnx`); the
//! detector/keypoint post-processing and hull/fill/dilate are host-side here (no
//! OpenCV). Geometry helpers (warp, resize) are reused from `image`.

use std::collections::HashMap;
use std::sync::Arc;

use thinfer_core::backend::WgpuBackend;
use thinfer_core::onnx::{OnnxError, OnnxModel};

use super::image::{self, Affine, Image};

const DET_SIZE: usize = 640;
const POSE_W: usize = 288;
const POSE_H: usize = 384;
const SIMCC_RATIO: f32 = 2.0;
const RESIZE_TARGET: f32 = 1024.0;
/// Face keypoints in the raw 133-keypoint RTMPose output. The reference inserts
/// a synthetic neck at index 17 (shifting feet/face/hands by 1) and then reads
/// `candidate[24:92]`; before the shift that is `[23:91]` (68 face points).
const FACE_LO: usize = 23;
const FACE_HI: usize = 91;
const KPT_SCORE_THR: f32 = 0.3;
/// RTMPose ImageNet-style normalization (RGB mean/std).
const POSE_MEAN: [f32; 3] = [123.675, 116.28, 103.53];
const POSE_STD: [f32; 3] = [58.395, 57.12, 57.375];
/// Person-detection thresholds (mirror `onnxdet.inference_detector`).
const DET_SCORE_THR: f32 = 0.3;
const DET_NMS_IOU: f32 = 0.45;
const DILATE_RADIUS: i32 = 7; // 15x15 kernel -> (15-1)/2

/// A loaded DWPose face-mask extractor (detector + keypoint nets resident).
pub struct DwPose {
    det: OnnxModel,
    pose: OnnxModel,
    det_input: String,
    pose_input: String,
}

impl DwPose {
    /// Load both ONNX nets at their fixed input shapes (yolox 640x640, RTMPose
    /// 384x288, batch 1 - one person crop per run).
    pub async fn load(
        backend: Arc<WgpuBackend>,
        yolox_onnx: &[u8],
        dwpose_onnx: &[u8],
    ) -> Result<Self, OnnxError> {
        let det = OnnxModel::load(
            Arc::clone(&backend),
            yolox_onnx,
            &single_input(yolox_onnx, &[1, 3, DET_SIZE as i64, DET_SIZE as i64])?,
        )
        .await?;
        let pose = OnnxModel::load(
            backend,
            dwpose_onnx,
            &single_input(dwpose_onnx, &[1, 3, POSE_H as i64, POSE_W as i64])?,
        )
        .await?;
        let det_input = det.input_names[0].clone();
        let pose_input = pose.input_names[0].clone();
        Ok(Self {
            det,
            pose,
            det_input,
            pose_input,
        })
    }

    /// Build the face mask for one original-resolution frame: white (255) face
    /// region on black, RGB, same size as `frame`.
    pub async fn face_mask(&self, frame: &Image) -> Result<Image, OnnxError> {
        let (ori_w, ori_h) = (frame.w, frame.h);
        // Reference `resize_image(., 1024)`: the detector/pose working image.
        let work = resize_work(frame);
        let boxes = self.detect(&work).await?;

        // Collect each person's face polygon (original-pixel int coords).
        let mut polys: Vec<Vec<[i32; 2]>> = Vec::new();
        for b in &boxes {
            let (kpts, scores) = self.pose_one(&work, b).await?;
            let mut pts: Vec<[i32; 2]> = Vec::new();
            for i in FACE_LO..FACE_HI {
                if scores[i] < KPT_SCORE_THR {
                    continue; // invisible -> reference sets coord -1, dropped
                }
                // Normalize by the working-image size, then scale to original.
                let nx = kpts[i][0] / work.w as f32;
                let ny = kpts[i][1] / work.h as f32;
                pts.push([(nx * ori_w as f32) as i32, (ny * ori_h as f32) as i32]);
            }
            if pts.len() > 3 {
                polys.push(convex_hull(&pts));
            }
        }

        // Keep the largest-area hull, fill it, dilate 15x15.
        let mut mask = vec![0u8; ori_w * ori_h];
        if let Some(hull) = polys.into_iter().max_by(|a, b| {
            polygon_area(a)
                .partial_cmp(&polygon_area(b))
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            fill_poly(&mut mask, ori_w, ori_h, &hull);
            mask = dilate(&mask, ori_w, ori_h, DILATE_RADIUS);
        }
        Ok(mask_to_image(&mask, ori_w, ori_h))
    }

    /// Run yolox on the working image, decode + NMS, return person bboxes
    /// `[x1,y1,x2,y2]` in working-image pixels.
    async fn detect(&self, work: &Image) -> Result<Vec<[f32; 4]>, OnnxError> {
        // Preprocess: resize preserving aspect to fit 640, pad bottom-right with
        // 114, keep RGB, raw [0,255] (yolox onnx has its own normalization).
        let r = (DET_SIZE as f32 / work.h as f32).min(DET_SIZE as f32 / work.w as f32);
        let nw = (work.w as f32 * r) as usize;
        let nh = (work.h as f32 * r) as usize;
        let resized = work.resize(nw.max(1), nh.max(1));
        let hw = DET_SIZE * DET_SIZE;
        let mut input = vec![114.0f32; 3 * hw];
        for y in 0..nh {
            for x in 0..nw {
                let si = (y * resized.w + x) * 3;
                for c in 0..3 {
                    input[c * hw + y * DET_SIZE + x] = resized.data[si + c];
                }
            }
        }
        let mut feeds = HashMap::new();
        feeds.insert(self.det_input.clone(), input);
        let out = self.det.run(&feeds).await?;
        let (_shape, pred) = &out[&self.det.output_names[0]]; // [1, 8400, 85]
        Ok(decode_yolox(pred, r))
    }

    /// RTMPose on one bbox: affine-warp the crop, normalize, run, SimCC-decode.
    /// Returns 133 keypoints (working-image pixels) + per-keypoint scores.
    async fn pose_one(
        &self,
        work: &Image,
        bbox: &[f32; 4],
    ) -> Result<(Vec<[f32; 2]>, Vec<f32>), OnnxError> {
        let (center, scale) = bbox_to_center_scale(bbox);
        let warp = pose_warp_matrix(center, scale);
        let crop = image::warp_affine(work, &warp, POSE_W, POSE_H);

        // Normalize (x-mean)/std per channel -> CHW [1,3,384,288].
        let chw = POSE_W * POSE_H;
        let mut input = vec![0.0f32; 3 * chw];
        for i in 0..chw {
            for c in 0..3 {
                input[c * chw + i] = (crop.data[i * 3 + c] - POSE_MEAN[c]) / POSE_STD[c];
            }
        }
        let mut feeds = HashMap::new();
        feeds.insert(self.pose_input.clone(), input);
        let out = self.pose.run(&feeds).await?;
        let (sx_shape, sx) = &out["simcc_x"]; // [1, 133, Wx]
        let (_sy_shape, sy) = &out["simcc_y"]; // [1, 133, Wy]
        let k = sx_shape[1] as usize;
        let wx = sx_shape[2] as usize;
        let wy = sy.len() / k;
        Ok(decode_simcc(sx, sy, k, wx, wy, center, scale))
    }
}

/// Run the whole video: load once, mask each frame. `frames` are original-
/// resolution RGB `Image`s; returns one mask `Image` per frame.
pub async fn face_mask_video(
    backend: Arc<WgpuBackend>,
    yolox_onnx: &[u8],
    dwpose_onnx: &[u8],
    frames: &[Image],
) -> Result<Vec<Image>, OnnxError> {
    let dw = DwPose::load(backend, yolox_onnx, dwpose_onnx).await?;
    let mut out = Vec::with_capacity(frames.len());
    for f in frames {
        out.push(dw.face_mask(f).await?);
    }
    Ok(out)
}

/// Reference `resize_image(img, 1024)`: scale so the short side hits 1024, then
/// round each side to a multiple of 64.
fn resize_work(img: &Image) -> Image {
    let (h, w) = (img.h as f32, img.w as f32);
    let k = RESIZE_TARGET / h.min(w);
    let nh = ((h * k / 64.0).round() * 64.0) as usize;
    let nw = ((w * k / 64.0).round() * 64.0) as usize;
    img.resize(nw.max(64), nh.max(64))
}

/// Decode yolox output `[1, 8400, 85]` -> person bboxes in working-image pixels.
/// Mirrors `demo_postprocess` + person-class `multiclass_nms`.
fn decode_yolox(pred: &[f32], ratio: f32) -> Vec<[f32; 4]> {
    const STRIDES: [usize; 3] = [8, 16, 32];
    let step = 85; // 4 box + 1 obj + 80 class
    let mut cand: Vec<([f32; 4], f32)> = Vec::new();
    let mut a = 0usize; // running anchor index
    for &s in &STRIDES {
        let grid = DET_SIZE / s;
        let sf = s as f32;
        for gy in 0..grid {
            for gx in 0..grid {
                let o = a * step;
                a += 1;
                let obj = pred[o + 4];
                let cls0 = pred[o + 5]; // class 0 = person
                let score = obj * cls0;
                if score <= 0.1 {
                    continue;
                }
                let cx = (pred[o] + gx as f32) * sf;
                let cy = (pred[o + 1] + gy as f32) * sf;
                let bw = pred[o + 2].exp() * sf;
                let bh = pred[o + 3].exp() * sf;
                let bbox = [
                    (cx - bw * 0.5) / ratio,
                    (cy - bh * 0.5) / ratio,
                    (cx + bw * 0.5) / ratio,
                    (cy + bh * 0.5) / ratio,
                ];
                cand.push((bbox, score));
            }
        }
    }
    // NMS, then keep score > 0.3.
    cand.sort_by(|x, y| y.1.total_cmp(&x.1));
    let mut keep: Vec<[f32; 4]> = Vec::new();
    'outer: for (bbox, score) in cand {
        for k in &keep {
            if box_iou(&bbox, k) > DET_NMS_IOU {
                continue 'outer;
            }
        }
        if score > DET_SCORE_THR {
            keep.push(bbox);
        }
    }
    keep
}

/// IoU with the reference's `+1` box convention (`onnxdet.nms`).
fn box_iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let x1 = a[0].max(b[0]);
    let y1 = a[1].max(b[1]);
    let x2 = a[2].min(b[2]);
    let y2 = a[3].min(b[3]);
    let inter = (x2 - x1 + 1.0).max(0.0) * (y2 - y1 + 1.0).max(0.0);
    let area_a = (a[2] - a[0] + 1.0) * (a[3] - a[1] + 1.0);
    let area_b = (b[2] - b[0] + 1.0) * (b[3] - b[1] + 1.0);
    inter / (area_a + area_b - inter)
}

/// `bbox_xyxy2cs(padding=1.25)` + `_fix_aspect_ratio(288/384)`.
fn bbox_to_center_scale(b: &[f32; 4]) -> ([f32; 2], [f32; 2]) {
    let center = [(b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5];
    let mut sw = (b[2] - b[0]) * 1.25;
    let mut sh = (b[3] - b[1]) * 1.25;
    let aspect = POSE_W as f32 / POSE_H as f32;
    if sw > sh * aspect {
        sh = sw / aspect;
    } else {
        sw = sh * aspect;
    }
    (center, [sw, sh])
}

/// `get_warp_matrix(center, scale, rot=0, output_size=(288,384))`: the src->dst
/// affine, solved from three point pairs (cv2 `getAffineTransform`).
fn pose_warp_matrix(center: [f32; 2], scale: [f32; 2]) -> Affine {
    let src_w = scale[0];
    let (dst_w, dst_h) = (POSE_W as f32, POSE_H as f32);
    let src_dir = [0.0, src_w * -0.5];
    let dst_dir = [0.0, dst_w * -0.5];
    let third = |a: [f32; 2], b: [f32; 2]| [b[0] - (a[1] - b[1]), b[1] + (a[0] - b[0])];
    let s0 = center;
    let s1 = [center[0] + src_dir[0], center[1] + src_dir[1]];
    let s2 = third(s0, s1);
    let d0 = [dst_w * 0.5, dst_h * 0.5];
    let d1 = [d0[0] + dst_dir[0], d0[1] + dst_dir[1]];
    let d2 = third(d0, d1);
    affine_from_3pts([s0, s1, s2], [d0, d1, d2])
}

/// Solve the 2x3 affine mapping `src[i] -> dst[i]` for three pairs.
fn affine_from_3pts(src: [[f32; 2]; 3], dst: [[f32; 2]; 3]) -> Affine {
    // Two independent 3x3 systems: [x y 1] * [a,b,tx]^T = x', likewise y'.
    let m = [
        [src[0][0], src[0][1], 1.0],
        [src[1][0], src[1][1], 1.0],
        [src[2][0], src[2][1], 1.0],
    ];
    let sol_x = solve3(m, [dst[0][0], dst[1][0], dst[2][0]]);
    let sol_y = solve3(m, [dst[0][1], dst[1][1], dst[2][1]]);
    [sol_x[0], sol_x[1], sol_x[2], sol_y[0], sol_y[1], sol_y[2]]
}

/// Gaussian elimination for a 3x3 system `a*x = b`.
#[allow(clippy::needless_range_loop)] // augmented-matrix column math reads clearer indexed
fn solve3(a: [[f32; 3]; 3], b: [f32; 3]) -> [f32; 3] {
    let mut m = [
        [a[0][0], a[0][1], a[0][2], b[0]],
        [a[1][0], a[1][1], a[1][2], b[1]],
        [a[2][0], a[2][1], a[2][2], b[2]],
    ];
    for col in 0..3 {
        let mut piv = col;
        for r in (col + 1)..3 {
            if m[r][col].abs() > m[piv][col].abs() {
                piv = r;
            }
        }
        m.swap(col, piv);
        let d = m[col][col];
        if d.abs() < 1e-12 {
            continue;
        }
        for r in 0..3 {
            if r == col {
                continue;
            }
            let f = m[r][col] / d;
            for c in col..4 {
                m[r][c] -= f * m[col][c];
            }
        }
    }
    [m[0][3] / m[0][0], m[1][3] / m[1][1], m[2][3] / m[2][2]]
}

/// SimCC decode: argmax over each 1D heatmap, `/simcc_split_ratio`, then map the
/// input-space keypoint back to working-image pixels via `scale`/`center`.
fn decode_simcc(
    sx: &[f32],
    sy: &[f32],
    k: usize,
    wx: usize,
    wy: usize,
    center: [f32; 2],
    scale: [f32; 2],
) -> (Vec<[f32; 2]>, Vec<f32>) {
    let mut kpts = vec![[0.0f32; 2]; k];
    let mut scores = vec![0.0f32; k];
    for j in 0..k {
        let (mx, vx) = argmax(&sx[j * wx..j * wx + wx]);
        let (my, vy) = argmax(&sy[j * wy..j * wy + wy]);
        let val = vx.min(vy);
        // keypoint in model-input pixels, then affine back to working image.
        let kx = mx as f32 / SIMCC_RATIO;
        let ky = my as f32 / SIMCC_RATIO;
        let px = kx / POSE_W as f32 * scale[0] + center[0] - scale[0] * 0.5;
        let py = ky / POSE_H as f32 * scale[1] + center[1] - scale[1] * 0.5;
        kpts[j] = [px, py];
        scores[j] = val;
    }
    (kpts, scores)
}

fn argmax(v: &[f32]) -> (usize, f32) {
    let mut bi = 0;
    let mut bv = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > bv {
            bv = x;
            bi = i;
        }
    }
    (bi, bv)
}

/// Andrew's monotone-chain convex hull of integer points (CCW, no collinear).
fn convex_hull(pts: &[[i32; 2]]) -> Vec<[i32; 2]> {
    let mut p = pts.to_vec();
    p.sort_by(|a, b| a[0].cmp(&b[0]).then(a[1].cmp(&b[1])));
    p.dedup();
    if p.len() < 3 {
        return p;
    }
    let cross = |o: [i32; 2], a: [i32; 2], b: [i32; 2]| -> i64 {
        (a[0] - o[0]) as i64 * (b[1] - o[1]) as i64 - (a[1] - o[1]) as i64 * (b[0] - o[0]) as i64
    };
    let mut hull: Vec<[i32; 2]> = Vec::with_capacity(p.len() + 1);
    // Lower hull.
    for &pt in &p {
        while hull.len() >= 2 && cross(hull[hull.len() - 2], hull[hull.len() - 1], pt) <= 0 {
            hull.pop();
        }
        hull.push(pt);
    }
    // Upper hull.
    let lower = hull.len() + 1;
    for &pt in p.iter().rev() {
        while hull.len() >= lower && cross(hull[hull.len() - 2], hull[hull.len() - 1], pt) <= 0 {
            hull.pop();
        }
        hull.push(pt);
    }
    hull.pop();
    hull
}

/// Shoelace area (absolute) of a polygon.
fn polygon_area(poly: &[[i32; 2]]) -> f64 {
    let n = poly.len();
    if n < 3 {
        return 0.0;
    }
    let mut acc = 0i64;
    for i in 0..n {
        let a = poly[i];
        let b = poly[(i + 1) % n];
        acc += a[0] as i64 * b[1] as i64 - b[0] as i64 * a[1] as i64;
    }
    (acc.abs() as f64) * 0.5
}

/// Scanline fill of a (convex) polygon into a binary mask.
fn fill_poly(mask: &mut [u8], w: usize, h: usize, poly: &[[i32; 2]]) {
    let n = poly.len();
    if n < 3 {
        return;
    }
    let ymin = poly.iter().map(|p| p[1]).min().unwrap().max(0);
    let ymax = poly.iter().map(|p| p[1]).max().unwrap().min(h as i32 - 1);
    for y in ymin..=ymax {
        let yf = y as f32 + 0.5;
        let mut xs: Vec<f32> = Vec::new();
        for i in 0..n {
            let a = poly[i];
            let b = poly[(i + 1) % n];
            let (y0, y1) = (a[1] as f32, b[1] as f32);
            if (y0 <= yf && y1 > yf) || (y1 <= yf && y0 > yf) {
                let t = (yf - y0) / (y1 - y0);
                xs.push(a[0] as f32 + t * (b[0] - a[0]) as f32);
            }
        }
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut i = 0;
        while i + 1 < xs.len() {
            let x0 = xs[i].ceil().max(0.0) as i32;
            let x1 = (xs[i + 1].floor() as i32).min(w as i32 - 1);
            for x in x0..=x1 {
                mask[y as usize * w + x as usize] = 255;
            }
            i += 2;
        }
    }
}

/// Binary dilation by a `(2r+1)x(2r+1)` square (separable max), matching a cv2
/// `dilate` with a ones kernel.
fn dilate(src: &[u8], w: usize, h: usize, r: i32) -> Vec<u8> {
    // Horizontal pass.
    let mut tmp = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut on = false;
            let x0 = (x as i32 - r).max(0) as usize;
            let x1 = (x as i32 + r).min(w as i32 - 1) as usize;
            for xx in x0..=x1 {
                if src[y * w + xx] != 0 {
                    on = true;
                    break;
                }
            }
            tmp[y * w + x] = on as u8 * 255;
        }
    }
    // Vertical pass.
    let mut out = vec![0u8; w * h];
    for y in 0..h {
        let y0 = (y as i32 - r).max(0) as usize;
        let y1 = (y as i32 + r).min(h as i32 - 1) as usize;
        for x in 0..w {
            let mut on = false;
            for yy in y0..=y1 {
                if tmp[yy * w + x] != 0 {
                    on = true;
                    break;
                }
            }
            out[y * w + x] = on as u8 * 255;
        }
    }
    out
}

/// White-on-black single-channel mask -> RGB `Image` (all three channels equal).
fn mask_to_image(mask: &[u8], w: usize, h: usize) -> Image {
    let mut img = Image::new(w, h);
    for (i, &m) in mask.iter().enumerate() {
        let v = m as f32;
        img.data[i * 3] = v;
        img.data[i * 3 + 1] = v;
        img.data[i * 3 + 2] = v;
    }
    img
}

/// `{first_input_name: shape}` for a single-input ONNX model.
fn single_input(onnx: &[u8], shape: &[i64]) -> Result<HashMap<String, Vec<i64>>, OnnxError> {
    let graph =
        thinfer_core::onnx::proto::parse_model(onnx).map_err(|e| OnnxError::Io(e.to_string()))?;
    let name = graph
        .inputs
        .iter()
        .find(|vi| !graph.initializers.iter().any(|t| t.name == vi.name))
        .map(|vi| vi.name.clone())
        .ok_or_else(|| OnnxError::Unsupported("model has no input".into()))?;
    let mut m = HashMap::new();
    m.insert(name, shape.to_vec());
    Ok(m)
}
