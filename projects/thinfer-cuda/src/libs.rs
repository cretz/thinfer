//! Locating the optional CUDA libraries (Windows `.dll`, Linux `.so`). cudarc
//! opens each library by bare file name (normal OS search), which misses a
//! cuDNN install directory or a "next to the exe" bundle. So each one is
//! preloaded once by full path from the known locations; the OS loader then
//! resolves cudarc's by-name load to the already-loaded module. A library that
//! is not found stays unloaded and the ops needing it decline (wgpu runs).

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// The newest entry of `dir` whose name passes `keep`, by version order.
fn newest(dir: &Path, keep: &dyn Fn(&str) -> bool) -> Option<PathBuf> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok()?.file_name().into_string().ok())
        .filter(|n| keep(n))
        .collect();
    names.sort_by_key(|n| version_key(n));
    names.pop().map(|n| dir.join(n))
}

fn version_key(name: &str) -> Vec<u32> {
    name.split(|c: char| !c.is_ascii_digit())
        .filter(|p| !p.is_empty())
        .map(|p| p.parse().unwrap_or(0))
        .collect()
}

/// The CUDA 13 toolkit root: `CUDA_PATH` / `CUDA_HOME`, else the default
/// install location (a process started before the install has no env var).
fn toolkit_dir() -> Option<PathBuf> {
    for var in ["CUDA_PATH", "CUDA_HOME"] {
        if let Some(p) = std::env::var_os(var) {
            return Some(PathBuf::from(p));
        }
    }
    if cfg!(windows) {
        let root = PathBuf::from(std::env::var_os("ProgramFiles")?)
            .join("NVIDIA GPU Computing Toolkit")
            .join("CUDA");
        newest(&root, &|n| n.starts_with("v13."))
    } else {
        let p = PathBuf::from("/usr/local/cuda");
        p.is_dir().then_some(p)
    }
}

/// Windows: `C:/Program Files/NVIDIA/CUDNN/v<ver>/bin/<cuda 13.x>/x64`. Linux
/// packages install cuDNN into the system library path, which the default
/// search already covers.
fn cudnn_dir() -> Option<PathBuf> {
    if !cfg!(windows) {
        return None;
    }
    let root = PathBuf::from(std::env::var_os("ProgramFiles")?)
        .join("NVIDIA")
        .join("CUDNN");
    let ver = newest(&root, &|n| n.starts_with('v'))?;
    let cuda = newest(&ver.join("bin"), &|n| n.starts_with("13."))?;
    Some(cuda.join("x64"))
}

/// Where to look, in order: `<exe dir>/cuda` (a redistributable bundle), the
/// toolkit's library dirs, then cuDNN's own install dir.
fn search_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Some(exe_dir) = std::env::current_exe()
        .ok()
        .and_then(|e| e.parent().map(Path::to_path_buf))
    {
        dirs.push(exe_dir.join("cuda"));
    }
    if let Some(cuda) = toolkit_dir() {
        if cfg!(windows) {
            dirs.push(cuda.join("bin").join("x64"));
            dirs.push(cuda.join("bin"));
        } else {
            dirs.push(cuda.join("lib64"));
            dirs.push(cuda.join("lib"));
        }
    }
    dirs.extend(cudnn_dir());
    dirs
}

/// The CUDA toolkit include dir (NVRTC needs `mma.h` / `cuda_fp16.h`).
pub fn include_dir() -> Option<PathBuf> {
    let dir = toolkit_dir()?.join("include");
    dir.is_dir().then_some(dir)
}

/// Preload the newest library named `<prefix>*` containing `ext` from the
/// first search dir that has one. Returns the path it loaded from.
fn preload(prefix: &str, ext: &str) -> Option<PathBuf> {
    for dir in search_dirs() {
        let Some(path) = newest(&dir, &|n| n.starts_with(prefix) && n.contains(ext)) else {
            continue;
        };
        // SAFETY: loading an NVIDIA runtime library; its initializers are
        // trusted. Kept loaded for the process lifetime (leaked handle).
        if let Ok(lib) = unsafe { libloading::Library::new(&path) } {
            std::mem::forget(lib);
            return Some(path);
        }
    }
    None
}

/// What was found. Preloads once per process.
pub struct Libs {
    pub nvrtc: Option<PathBuf>,
}

pub fn libs() -> &'static Libs {
    static LIBS: OnceLock<Libs> = OnceLock::new();
    LIBS.get_or_init(|| {
        // NVRTC resolves its builtins library by name at compile time, so the
        // builtins go first, from the same search.
        let nvrtc = if cfg!(windows) {
            let _ = preload("nvrtc-builtins64_", ".dll");
            preload("nvrtc64_13", ".dll")
        } else {
            let _ = preload("libnvrtc-builtins.so.13", ".so");
            preload("libnvrtc.so.13", ".so")
        };
        tracing::info!(target: thinfer_core::trace::ADAPTER, nvrtc = ?nvrtc, "cuda libs");
        Libs { nvrtc }
    })
}
