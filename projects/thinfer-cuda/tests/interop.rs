//! Shared-memory round trip: bytes written by wgpu are read back by CUDA from
//! the same allocation, and vice versa. Skips (passes, with a note) when the
//! machine has no NVIDIA Vulkan adapter with CUDA.
#![cfg(not(target_os = "macos"))]

use cudarc::driver::sys;

fn device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))
    .ok()?;
    // Windows gates Win32 memory export behind this feature; elsewhere it is
    // absent and fd export needs no feature (`probe` checks either way).
    let required_features = adapter.features() & wgpu::Features::VULKAN_EXTERNAL_MEMORY_WIN32;
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_features,
        ..Default::default()
    }))
    .ok()
}

#[test]
fn wgpu_and_cuda_see_the_same_bytes() {
    let Some((device, queue)) = device() else {
        eprintln!("skipped[cuda interop]: no Vulkan adapter with external memory");
        return;
    };
    let cuda = match thinfer_cuda::probe(&device) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skipped[cuda interop]: {e}");
            return;
        }
    };
    eprintln!(
        "cuda interop on {} sm_{}{}",
        cuda.name, cuda.sm.0, cuda.sm.1
    );

    const N: usize = 1 << 20;
    let shared = thinfer_cuda::alloc_shared(&device, &cuda, N as u64, "interop-test").unwrap();

    // wgpu -> CUDA.
    let pattern: Vec<u8> = (0..N).map(|i| (i * 31 + 7) as u8).collect();
    queue.write_buffer(&shared.buffer, 0, &pattern);
    queue.submit([]);
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    cuda.ctx.bind_to_thread().unwrap();
    let mut seen = vec![0u8; N];
    // SAFETY: `seen` holds N bytes; `shared.mapping.ptr` maps N bytes.
    unsafe { sys::cuMemcpyDtoH_v2(seen.as_mut_ptr().cast(), shared.mapping.ptr, N) }
        .result()
        .unwrap();
    assert!(seen == pattern, "CUDA did not see wgpu's write");

    // CUDA -> wgpu.
    // SAFETY: as above.
    unsafe { sys::cuMemsetD8_v2(shared.mapping.ptr, 0xA5, N) }
        .result()
        .unwrap();
    unsafe { sys::cuCtxSynchronize() }.result().unwrap();
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("interop-readback"),
        size: N as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(&shared.buffer, 0, &readback, 0, N as u64);
    queue.submit([enc.finish()]);
    readback
        .slice(..)
        .map_async(wgpu::MapMode::Read, |r| r.unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let view = readback.slice(..).get_mapped_range();
    assert!(
        view.iter().all(|b| *b == 0xA5),
        "wgpu did not see CUDA's write"
    );
}

#[test]
fn engine_buffers_are_cuda_addressable_with_the_allocator_installed() {
    use std::sync::Arc;
    use thinfer_core::backend::{Backend, PowerPreference, WgpuBackend, WgpuConfig};

    // The discrete GPU (hybrid laptops default to the integrated one, which
    // correctly probes as "no CUDA match").
    let backend = pollster::block_on(WgpuBackend::new_with_config(WgpuConfig {
        power_preference: PowerPreference::HighPerformance,
        ..Default::default()
    }))
    .expect("wgpu adapter");
    let cuda = match thinfer_cuda::probe(backend.wgpu_device()) {
        Ok(c) => Arc::new(c),
        Err(e) => {
            eprintln!("skipped[cuda engine alloc]: {e}");
            return;
        }
    };
    // Before install: plain buffers, no CUDA address.
    let plain = backend.allocate(4096).unwrap();
    assert_eq!(backend.external_address(plain), None);
    assert!(
        backend.set_external_allocator(Arc::new(thinfer_cuda::CudaAllocator::new(cuda.clone())))
    );

    const N: usize = 1 << 16;
    let id = backend.allocate(N as u64).unwrap();
    let ptr = backend
        .external_address(id)
        .expect("allocation went through CUDA");
    let pattern: Vec<u8> = (0..N).map(|i| (i * 13 + 1) as u8).collect();
    backend.write_buffer(id, 0, &pattern).unwrap();
    // The engine's own readback both flushes the write and checks wgpu's view.
    assert_eq!(
        pollster::block_on(backend.read_buffer(id, 0, N as u64)).unwrap(),
        pattern
    );
    cuda.ctx.bind_to_thread().unwrap();
    let mut seen = vec![0u8; N];
    // SAFETY: `seen` holds N bytes; `ptr` maps the N-byte allocation.
    unsafe { sys::cuMemcpyDtoH_v2(seen.as_mut_ptr().cast(), ptr, N) }
        .result()
        .unwrap();
    assert!(seen == pattern, "CUDA view differs from the engine's write");
    backend.free(id);
    backend.free(plain);
}
