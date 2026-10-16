//! Metal backend (macOS / Apple GPUs).

use super::{Backend, DeviceInfo, Direction, Kernel, build_chase};
use anyhow::{Context, Result, anyhow, bail};
use metal::objc::{msg_send, sel, sel_impl};
use metal::{
    Buffer, CommandBufferRef, CommandQueue, CompileOptions, ComputePipelineState, Device, MTLCommandBufferStatus,
    MTLResourceOptions, MTLSize, NSRange,
};
use std::ffi::c_void;

const SHADERS: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct Params { uint n; uint passes; float value; };

// Every pass shifts each thread's starting element by one SIMD width, so the
// set of addresses a thread touches changes from pass to pass. That keeps the
// compiler from hoisting/merging loads or stores across passes while keeping
// accesses coalesced.

kernel void bw_read(device const float4* src [[buffer(0)]],
                    device float* out          [[buffer(1)]],
                    constant Params& p         [[buffer(2)]],
                    uint gid [[thread_position_in_grid]],
                    uint gsz [[threads_per_grid]]) {
    float4 acc = 0.0f;
    for (uint pass = 0; pass < p.passes; ++pass) {
        uint start = (gid + pass * 32u) % gsz;
        for (uint i = start; i < p.n; i += gsz) acc += src[i];
    }
    float s = acc.x + acc.y + acc.z + acc.w;
    if (s == p.value) out[0] = s; // never true; defeats dead-code elimination
}

kernel void bw_write(device float4* dst   [[buffer(0)]],
                     constant Params& p   [[buffer(2)]],
                     uint gid [[thread_position_in_grid]],
                     uint gsz [[threads_per_grid]]) {
    for (uint pass = 0; pass < p.passes; ++pass) {
        uint start = (gid + pass * 32u) % gsz;
        float4 v = float4(p.value + float(pass));
        for (uint i = start; i < p.n; i += gsz) dst[i] = v;
    }
}

kernel void bw_copy(device const float4* src [[buffer(0)]],
                    device float4* dst       [[buffer(1)]],
                    constant Params& p       [[buffer(2)]],
                    uint gid [[thread_position_in_grid]],
                    uint gsz [[threads_per_grid]]) {
    for (uint pass = 0; pass < p.passes; ++pass) {
        uint start = (gid + pass * 32u) % gsz;
        for (uint i = start; i < p.n; i += gsz) dst[i] = src[i];
    }
}

kernel void chase(device const uint* next [[buffer(0)]],
                  device uint* out        [[buffer(1)]],
                  constant uint& steps    [[buffer(2)]]) {
    uint p = 0;
    for (uint i = 0; i < steps; ++i) p = next[p];
    out[0] = p;
}
"#;

#[repr(C)]
struct Params {
    n: u32,
    passes: u32,
    value: f32,
}

struct DevBufs {
    bytes: usize,
    a: Buffer,
    b: Buffer,
}

struct ChaseBuf {
    key: (usize, usize),
    buf: Buffer,
}

pub struct MetalBackend {
    device: Device,
    queue: CommandQueue,
    read: ComputePipelineState,
    write: ComputePipelineState,
    copy: ComputePipelineState,
    chase: ComputePipelineState,
    out: Buffer,
    dev: Option<DevBufs>,
    host: Option<(usize, Buffer)>,
    chase_buf: Option<ChaseBuf>,
    info: DeviceInfo,
}

pub fn list_devices() -> Vec<String> {
    Device::all().iter().map(|d| d.name().to_string()).collect()
}

impl MetalBackend {
    pub fn new(index: usize) -> Result<Self> {
        let all = Device::all();
        if all.is_empty() {
            bail!("no Metal devices found");
        }
        let device = all
            .get(index)
            .cloned()
            .ok_or_else(|| anyhow!("device index {index} out of range ({} devices)", all.len()))?;

        let library = device
            .new_library_with_source(SHADERS, &CompileOptions::new())
            .map_err(|e| anyhow!("shader compile failed: {e}"))?;
        let pipeline = |name: &str| -> Result<ComputePipelineState> {
            let f = library.get_function(name, None).map_err(|e| anyhow!("{name}: {e}"))?;
            device
                .new_compute_pipeline_state_with_function(&f)
                .map_err(|e| anyhow!("{name}: {e}"))
        };

        let info = DeviceInfo {
            backend: "Metal",
            name: device.name().to_string(),
            memory_bytes: device.recommended_max_working_set_size(),
            unified_memory: device.has_unified_memory(),
            details: vec![
                ("Max buffer".into(), crate::units::fmt_bytes(device.max_buffer_length() as usize)),
                ("Low power".into(), device.is_low_power().to_string()),
            ],
        };

        Ok(Self {
            queue: device.new_command_queue(),
            read: pipeline("bw_read")?,
            write: pipeline("bw_write")?,
            copy: pipeline("bw_copy")?,
            chase: pipeline("chase")?,
            out: device.new_buffer(16, MTLResourceOptions::StorageModeShared),
            dev: None,
            host: None,
            chase_buf: None,
            info,
            device,
        })
    }

    fn dev_bufs(&mut self, bytes: usize) -> Result<(Buffer, Buffer)> {
        if self.dev.as_ref().map(|d| d.bytes) != Some(bytes) {
            self.dev = None; // release the old pair before allocating
            let opts = MTLResourceOptions::StorageModePrivate;
            let a = self.device.new_buffer(bytes as u64, opts);
            let b = self.device.new_buffer(bytes as u64, opts);
            // Zero-fill so reads see well-defined data (and pages get committed).
            let cb = self.queue.new_command_buffer();
            let blit = cb.new_blit_command_encoder();
            blit.fill_buffer(&a, NSRange::new(0, bytes as u64), 0);
            blit.fill_buffer(&b, NSRange::new(0, bytes as u64), 0);
            blit.end_encoding();
            commit_and_wait(cb)?;
            self.dev = Some(DevBufs { bytes, a, b });
        }
        let d = self.dev.as_ref().unwrap();
        Ok((d.a.clone(), d.b.clone()))
    }

    fn host_buf(&mut self, bytes: usize) -> Buffer {
        if self.host.as_ref().map(|h| h.0) != Some(bytes) {
            self.host = None;
            let buf = self.device.new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared);
            // Touch every page from the CPU so first-use faults aren't timed.
            unsafe { std::ptr::write_bytes(buf.contents() as *mut u8, 1, bytes) };
            self.host = Some((bytes, buf));
        }
        self.host.as_ref().unwrap().1.clone()
    }
}

#[allow(unexpected_cfgs)] // old `objc` msg_send! macro
fn commit_and_wait(cb: &CommandBufferRef) -> Result<f64> {
    cb.commit();
    cb.wait_until_completed();
    if cb.status() == MTLCommandBufferStatus::Error {
        bail!("Metal command buffer failed (GPU timeout or out of memory?)");
    }
    // SAFETY: GPUStartTime / GPUEndTime are CFTimeInterval (f64) properties
    // of MTLCommandBuffer, valid once the buffer has completed.
    let (start, end): (f64, f64) = unsafe { (msg_send![cb, GPUStartTime], msg_send![cb, GPUEndTime]) };
    Ok(end - start)
}

impl Backend for MetalBackend {
    fn info(&self) -> &DeviceInfo {
        &self.info
    }

    fn default_threads(&self) -> usize {
        1 << 18
    }

    fn run_kernel(&mut self, kernel: Kernel, bytes: usize, passes: u32, threads: usize) -> Result<f64> {
        let (a, b) = self.dev_bufs(bytes)?;
        let n = (bytes / 16) as u64;
        let n32 = u32::try_from(n).context("buffer too large for 32-bit indexing")?;
        let params = Params { n: n32, passes, value: -1.0 };
        let pso = match kernel {
            Kernel::Read => &self.read,
            Kernel::Write => &self.write,
            Kernel::Copy => &self.copy,
        };

        let grid = (threads as u64).clamp(1, n);
        let tg = pso.max_total_threads_per_threadgroup().min(256).min(grid);

        let cb = self.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pso);
        match kernel {
            Kernel::Read => {
                enc.set_buffer(0, Some(&a), 0);
                enc.set_buffer(1, Some(&self.out), 0);
            }
            Kernel::Write => enc.set_buffer(0, Some(&b), 0),
            Kernel::Copy => {
                enc.set_buffer(0, Some(&a), 0);
                enc.set_buffer(1, Some(&b), 0);
            }
        }
        enc.set_bytes(2, size_of::<Params>() as u64, &params as *const Params as *const c_void);
        enc.dispatch_threads(MTLSize::new(grid, 1, 1), MTLSize::new(tg, 1, 1));
        enc.end_encoding();
        commit_and_wait(cb)
    }

    fn run_transfer(&mut self, dir: Direction, bytes: usize, reps: u32) -> Result<f64> {
        // On Apple silicon both buffers live in the same physical DRAM; this
        // measures a GPU blit between CPU-visible (shared) and GPU-private memory.
        let host = self.host_buf(bytes);
        let (dev, _) = self.dev_bufs(bytes)?;
        let (src, dst) = match dir {
            Direction::HostToDevice => (&host, &dev),
            Direction::DeviceToHost => (&dev, &host),
        };
        let cb = self.queue.new_command_buffer();
        let blit = cb.new_blit_command_encoder();
        for _ in 0..reps {
            blit.copy_from_buffer(src, 0, dst, 0, bytes as u64);
        }
        blit.end_encoding();
        commit_and_wait(cb)
    }

    fn run_latency(&mut self, bytes: usize, steps: u32, stride: usize) -> Result<f64> {
        let key = (bytes, stride);
        if self.chase_buf.as_ref().map(|c| c.key) != Some(key) {
            self.chase_buf = None;
            let data = build_chase(bytes, stride, 0x9E37_79B9_7F4A_7C15);
            let buf = self.device.new_buffer_with_data(
                data.as_ptr() as *const c_void,
                (data.len() * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            self.chase_buf = Some(ChaseBuf { key, buf });
        }
        let buf = &self.chase_buf.as_ref().unwrap().buf;

        let cb = self.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.chase);
        enc.set_buffer(0, Some(buf), 0);
        enc.set_buffer(1, Some(&self.out), 0);
        enc.set_bytes(2, 4, &steps as *const u32 as *const c_void);
        enc.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
        enc.end_encoding();
        commit_and_wait(cb)
    }
}
