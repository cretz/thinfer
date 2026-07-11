//! Full face-swap pipeline e2e: SCRFD detect -> ArcFace embed -> HyperSwap ->
//! paste-back, on real photos. Functional (not numeric-parity, which the
//! per-op `faceswap_onnx` test covers): asserts faces are detected at sane
//! locations, the swap is NaN-free, and the change is localized to the face
//! (face region changes a lot; a corner patch barely changes). Writes the
//! swapped frame to `THINFER_FS_OUT` (if set) for visual inspection.
//!
//! Opt-in (`faceswap-e2e`); skips unless the models + images are provided:
//! ```text
//! THINFER_FS_SCRFD=.. THINFER_FS_ARCFACE=.. THINFER_FS_HYPERSWAP=..
//! THINFER_FS_SRC=src.png THINFER_FS_DST=dst.png THINFER_FS_OUT=out.png
//! cargo test -p thinfer-conformance --features faceswap-e2e --test faceswap_pipeline -- --nocapture
//! ```

#![cfg(feature = "faceswap-e2e")]

use std::path::Path;
use std::sync::Arc;

use thinfer_core::backend::WgpuBackend;
use thinfer_models::faceswap::FaceSwapper;
use thinfer_models::faceswap::image::Image;

fn load_png_rgb(path: &Path) -> Image {
    let decoder = png::Decoder::new(std::fs::File::open(path).expect("open png"));
    let mut reader = decoder.read_info().expect("png header");
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).expect("png frame");
    let (w, h) = (info.width as usize, info.height as usize);
    let rgb: Vec<u8> = match info.color_type {
        png::ColorType::Rgb => buf[..w * h * 3].to_vec(),
        png::ColorType::Rgba => buf[..w * h * 4]
            .chunks_exact(4)
            .flat_map(|p| [p[0], p[1], p[2]])
            .collect(),
        other => panic!("unsupported png color type {other:?}"),
    };
    Image::from_rgb8(w, h, &rgb)
}

fn write_png_rgb(path: &Path, img: &Image) {
    let file = std::fs::File::create(path).expect("create png");
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), img.w as u32, img.h as u32);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header()
        .expect("png header")
        .write_image_data(&img.to_rgb8())
        .expect("png data");
}

/// Mean absolute per-pixel difference over an axis-aligned region.
fn region_diff(a: &Image, b: &Image, x0: usize, y0: usize, x1: usize, y1: usize) -> f32 {
    let mut sum = 0.0f64;
    let mut n = 0u64;
    for y in y0..y1.min(a.h) {
        for x in x0..x1.min(a.w) {
            for c in 0..3 {
                let i = (y * a.w + x) * 3 + c;
                sum += (a.data[i] - b.data[i]).abs() as f64;
                n += 1;
            }
        }
    }
    (sum / n.max(1) as f64) as f32
}

#[test]
fn faceswap_pipeline_e2e() {
    let env = |k: &str| std::env::var(k).ok();
    let (Some(scrfd), Some(arcface), Some(hyperswap), Some(src), Some(dst)) = (
        env("THINFER_FS_SCRFD"),
        env("THINFER_FS_ARCFACE"),
        env("THINFER_FS_HYPERSWAP"),
        env("THINFER_FS_SRC"),
        env("THINFER_FS_DST"),
    ) else {
        eprintln!(
            "faceswap_pipeline_e2e skipped (set THINFER_FS_{{SCRFD,ARCFACE,HYPERSWAP,SRC,DST}})"
        );
        return;
    };

    pollster::block_on(async {
        let backend = Arc::new(WgpuBackend::new().await.expect("wgpu"));
        let swapper = FaceSwapper::load(
            backend,
            &std::fs::read(&scrfd).unwrap(),
            &std::fs::read(&arcface).unwrap(),
            &std::fs::read(&hyperswap).unwrap(),
        )
        .await
        .expect("load FaceSwapper");

        let src_img = load_png_rgb(Path::new(&src));
        let dst_img = load_png_rgb(Path::new(&dst));

        // Detection finds real faces at sane locations (validates SCRFD decode).
        let src_faces = swapper.detect(&src_img).await.expect("detect src");
        let dst_faces = swapper.detect(&dst_img).await.expect("detect dst");
        eprintln!(
            "src faces: {}, dst faces: {}",
            src_faces.len(),
            dst_faces.len()
        );
        assert!(!src_faces.is_empty(), "no face in source image");
        assert!(!dst_faces.is_empty(), "no face in target image");
        let f0 = &dst_faces[0];
        eprintln!("dst face0 bbox {:?} score {:.3}", f0.bbox, f0.score);
        assert!(f0.score > 0.5);
        assert!(f0.bbox[0] >= -5.0 && f0.bbox[2] <= dst_img.w as f32 + 5.0);

        // Source embedding is finite and ~unit norm.
        let emb = swapper.source_embedding(&src_img).await.expect("embedding");
        assert_eq!(emb.len(), 512);
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-3, "embedding norm {norm}");
        assert!(emb.iter().all(|x| x.is_finite()));

        // Swap, then check NaN-free + localized change.
        let out = swapper.swap_frame(&dst_img, &emb).await.expect("swap");
        assert_eq!((out.w, out.h), (dst_img.w, dst_img.h));
        assert!(
            out.data.iter().all(|x| x.is_finite()),
            "swap produced non-finite pixels"
        );

        let (bx0, by0) = (f0.bbox[0].max(0.0) as usize, f0.bbox[1].max(0.0) as usize);
        let (bx1, by1) = (f0.bbox[2].max(0.0) as usize, f0.bbox[3].max(0.0) as usize);
        let face_diff = region_diff(&dst_img, &out, bx0, by0, bx1, by1);
        let corner_diff = region_diff(&dst_img, &out, 0, 0, 40, 40);
        eprintln!("face-region mean|diff| {face_diff:.2}, corner mean|diff| {corner_diff:.4}");
        assert!(face_diff > 5.0, "face region barely changed ({face_diff})");
        assert!(
            corner_diff < 1.0,
            "corner changed too much ({corner_diff}) - paste leaked"
        );

        if let Some(out_path) = env("THINFER_FS_OUT") {
            write_png_rgb(Path::new(&out_path), &out);
            eprintln!("wrote swapped frame to {out_path}");
        }
    });
}

/// i8 DP4A conv quality eyeball: swap the SAME frame with HyperSwap in f32 and
/// in i8 (THINFER_ONNX_I8_CONV, scoped to HyperSwap by the loader), write both
/// swapped frames + a cropped face patch for each, and report the face-region
/// MAD between them. The detector/embedder stay f32 in both runs, so any
/// difference is purely the i8 conv. Run --ignored --nocapture; writes into the
/// dir given by THINFER_FS_OUT.
#[test]
#[ignore]
fn faceswap_i8_eyeball() {
    let env = |k: &str| std::env::var(k).ok();
    let (Some(scrfd), Some(arcface), Some(hyperswap), Some(src), Some(dst), Some(outdir)) = (
        env("THINFER_FS_SCRFD"),
        env("THINFER_FS_ARCFACE"),
        env("THINFER_FS_HYPERSWAP"),
        env("THINFER_FS_SRC"),
        env("THINFER_FS_DST"),
        env("THINFER_FS_OUT"),
    ) else {
        eprintln!("faceswap_i8_eyeball skipped (needs FS_{{SCRFD,ARCFACE,HYPERSWAP,SRC,DST,OUT}})");
        return;
    };

    pollster::block_on(async {
        let cfg = thinfer_core::backend::WgpuConfig {
            power_preference: thinfer_core::backend::PowerPreference::HighPerformance,
            ..Default::default()
        };
        let backend = Arc::new(WgpuBackend::new_with_config(cfg).await.expect("wgpu"));
        let (sb, ab, hb) = (
            std::fs::read(&scrfd).unwrap(),
            std::fs::read(&arcface).unwrap(),
            std::fs::read(&hyperswap).unwrap(),
        );
        let src_img = load_png_rgb(Path::new(&src));
        let dst_img = load_png_rgb(Path::new(&dst));

        // Load both swappers up front so the env toggle only affects the i8 one.
        let sw_f32 = FaceSwapper::load(Arc::clone(&backend), &sb, &ab, &hb)
            .await
            .expect("load f32");
        unsafe { std::env::set_var("THINFER_ONNX_I8_CONV", "1") };
        let sw_i8 = FaceSwapper::load(Arc::clone(&backend), &sb, &ab, &hb)
            .await
            .expect("load i8");
        unsafe { std::env::remove_var("THINFER_ONNX_I8_CONV") };

        let emb = sw_f32.source_embedding(&src_img).await.expect("embedding");
        let faces = sw_f32.detect(&dst_img).await.expect("detect");
        assert!(!faces.is_empty(), "no face in target");
        let f0 = &faces[0];

        let out_f32 = sw_f32.swap_frame(&dst_img, &emb).await.expect("swap f32");
        let out_i8 = sw_i8.swap_frame(&dst_img, &emb).await.expect("swap i8");
        assert!(
            out_i8.data.iter().all(|x| x.is_finite()),
            "i8 swap non-finite"
        );

        let (bx0, by0) = (f0.bbox[0].max(0.0) as usize, f0.bbox[1].max(0.0) as usize);
        let (bx1, by1) = (
            (f0.bbox[2] as usize).min(dst_img.w),
            (f0.bbox[3] as usize).min(dst_img.h),
        );
        let i8_vs_f32 = region_diff(&out_f32, &out_i8, bx0, by0, bx1, by1);
        let f32_vs_orig = region_diff(&dst_img, &out_f32, bx0, by0, bx1, by1);
        eprintln!(
            "[i8-eyeball] face bbox {:?}: f32-vs-i8 MAD={i8_vs_f32:.2}/255 \
             (swap magnitude f32-vs-orig={f32_vs_orig:.2}) -> i8 perturbs the swap by {:.1}%",
            f0.bbox,
            100.0 * i8_vs_f32 / f32_vs_orig.max(1e-3)
        );

        let d = Path::new(&outdir);
        std::fs::create_dir_all(d).ok();
        write_png_rgb(&d.join("swap_f32.png"), &out_f32);
        write_png_rgb(&d.join("swap_i8.png"), &out_i8);
        // Cropped face patches (padded) for a zoomed side-by-side.
        let crop = |img: &Image| -> Image {
            let pad = ((bx1 - bx0) / 4).max(10);
            let (cx0, cy0) = (bx0.saturating_sub(pad), by0.saturating_sub(pad));
            let (cx1, cy1) = ((bx1 + pad).min(img.w), (by1 + pad).min(img.h));
            let (cw, ch) = (cx1 - cx0, cy1 - cy0);
            let mut out = Image::new(cw, ch);
            for y in 0..ch {
                for x in 0..cw {
                    for c in 0..3 {
                        out.data[(y * cw + x) * 3 + c] =
                            img.data[((cy0 + y) * img.w + cx0 + x) * 3 + c];
                    }
                }
            }
            out
        };
        write_png_rgb(&d.join("face_f32.png"), &crop(&out_f32));
        write_png_rgb(&d.join("face_i8.png"), &crop(&out_i8));
        eprintln!("wrote swap_{{f32,i8}}.png + face_{{f32,i8}}.png to {outdir}");
    });
}
