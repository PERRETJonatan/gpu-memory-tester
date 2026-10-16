//! CUDA backend (NVIDIA GPUs). The CUDA driver and NVRTC are loaded at
//! runtime, so building this backend does not require the CUDA toolkit.

use super::{Backend, DeviceInfo, Direction, Kernel, build_chase};
use anyhow::{Context, Result, anyhow};
use cudarc::driver::sys::{CUdevice_attribute as Attr, CUevent_flags};
use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PinnedHostSlice, PushKernelArg};
use cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};
use std::sync::Arc;

const KERNELS: &str = r#"
// Every pass shifts each thread's starting element by one warp, so the set of
// addresses a thread touches changes from pass to pass. That keeps the
// compiler from hoisting/merging loads or stores across passes while keeping
// accesses coalesced.

extern "C" __global__ void bw_read(const float4* __restrict__ src, float* out,
                                   unsigned long long n, unsigned int passes, float value) {
    unsigned long long gid = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    unsigned long long gsz = (unsigned long long)gridDim.x * blockDim.x;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    for (unsigned int pass = 0; pass < passes; ++pass) {
        unsigned long long start = (gid + pass * 32ull) % gsz;
        for (unsigned long long i = start; i < n; i += gsz) {
            float4 v = src[i];
            acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
        }
    }
    float s = acc.x + acc.y + acc.z + acc.w;
    if (s == value) out[0] = s; // never true; defeats dead-code elimination
}

extern "C" __global__ void bw_write(float4* __restrict__ dst,
                                    unsigned long long n, unsigned int passes, float value) {
    unsigned long long gid = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    unsigned long long gsz = (unsigned long long)gridDim.x * blockDim.x;
    for (unsigned int pass = 0; pass < passes; ++pass) {
        unsigned long long start = (gid + pass * 32ull) % gsz;
        float f = value + (float)pass;
        float4 v = make_float4(f, f, f, f);
        for (unsigned long long i = start; i < n; i += gsz) dst[i] = v;
    }
}

extern "C" __global__ void bw_copy(const float4* __restrict__ src, float4* __restrict__ dst,
                                   unsigned long long n, unsigned int passes) {
    unsigned long long gid = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    unsigned long long gsz = (unsigned long long)gridDim.x * blockDim.x;
    for (unsigned int pass = 0; pass < passes; ++pass) {
        unsigned long long start = (gid + pass * 32ull) % gsz;
        for (unsigned long long i = start; i < n; i += gsz) dst[i] = src[i];
    }
}

extern "C" __global__ void chase(const unsigned int* next, unsigned int* out, unsigned int steps) {
    unsigned int p = 0;
    for (unsigned int i = 0; i < steps; ++i) p = __ldcg(&next[p]); // bypass L1, like a GPU-wide access
    out[0] = p;
}
"#;

pub fn list_devices() -> Vec<String> {
    let Ok(count) = CudaContext::device_count() else { return Vec::new() };
    (0..count as usize)
        .filter_map(|i| CudaContext::new(i).ok()?.name().ok())
        .collect()
}

pub struct CudaBackend {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    read: CudaFunction,
    write: CudaFunction,
    copy: CudaFunction,
    chase: CudaFunction,
    out: CudaSlice<u32>,
    dev: Option<(usize, CudaSlice<f32>, CudaSlice<f32>)>,
    host: Option<(usize, PinnedHostSlice<u8>)>,
    dev_bytes: Option<(usize, CudaSlice<u8>)>,
    chase_buf: Option<((usize, usize), CudaSlice<u32>)>,
    sm_count: usize,
    info: DeviceInfo,
}

impl CudaBackend {
    pub fn new(ordinal: usize) -> Result<Self> {
        let ctx = CudaContext::new(ordinal).context("failed to open CUDA device")?;
        let stream = ctx.default_stream();

        let ptx = compile_ptx_with_opts(KERNELS, CompileOptions { options: vec!["-O3".into()], ..Default::default() })
            .map_err(|e| anyhow!("NVRTC compile failed: {e:?}"))?;
        let module = ctx.load_module(ptx)?;

        let attr = |a| ctx.attribute(a).unwrap_or(0);
        let sm_count = attr(Attr::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT) as usize;
        let (major, minor) = ctx.compute_capability()?;
        let mem_clock_khz = attr(Attr::CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE) as f64;
        let bus_bits = attr(Attr::CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH) as f64;
        let mut details = vec![
            ("Compute capability".into(), format!("{major}.{minor}")),
            ("SMs".into(), sm_count.to_string()),
        ];
        if mem_clock_khz > 0.0 && bus_bits > 0.0 {
            // DDR: two transfers per clock.
            let peak = 2.0 * mem_clock_khz * 1e3 * bus_bits / 8.0 / 1e9;
            details.push(("Memory bus".into(), format!("{bus_bits} bit @ {:.0} MHz", mem_clock_khz / 1e3)));
            details.push(("Theoretical peak".into(), format!("{peak:.1} GB/s")));
        }
        let info = DeviceInfo {
            backend: "CUDA",
            name: ctx.name()?,
            memory_bytes: ctx.total_mem()? as u64,
            unified_memory: attr(Attr::CU_DEVICE_ATTRIBUTE_INTEGRATED) != 0,
            details,
        };

        Ok(Self {
            read: module.load_function("bw_read")?,
            write: module.load_function("bw_write")?,
            copy: module.load_function("bw_copy")?,
            chase: module.load_function("chase")?,
            out: stream.alloc_zeros::<u32>(4)?,
            dev: None,
            host: None,
            dev_bytes: None,
            chase_buf: None,
            sm_count: sm_count.max(1),
            info,
            ctx,
            stream,
        })
    }

    /// Times whatever `f` enqueues on the stream using CUDA events.
    fn timed(&self, f: impl FnOnce() -> Result<()>) -> Result<f64> {
        let start = self.stream.record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
        f()?;
        let end = self.stream.record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
        Ok(start.elapsed_ms(&end)? as f64 / 1e3)
    }
}

impl Backend for CudaBackend {
    fn info(&self) -> &DeviceInfo {
        &self.info
    }

    fn default_threads(&self) -> usize {
        self.sm_count * 2048
    }

    fn run_kernel(&mut self, kernel: Kernel, bytes: usize, passes: u32, threads: usize) -> Result<f64> {
        if self.dev.as_ref().map(|d| d.0) != Some(bytes) {
            self.dev = None;
            let a = self.stream.alloc_zeros::<f32>(bytes / 4)?;
            let b = self.stream.alloc_zeros::<f32>(bytes / 4)?;
            self.dev = Some((bytes, a, b));
        }
        let n = (bytes / 16) as u64;
        let grid = (threads as u64).clamp(1, n);
        let block = grid.min(256);
        let cfg = LaunchConfig {
            grid_dim: (grid.div_ceil(block) as u32, 1, 1),
            block_dim: (block as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let value = -1.0f32;

        let (_, a, b) = self.dev.as_mut().unwrap();
        let (stream, out) = (&self.stream, &mut self.out);
        let (read, write, copy) = (&self.read, &self.write, &self.copy);
        let start = stream.record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
        unsafe {
            match kernel {
                Kernel::Read => stream.launch_builder(read).arg(&*a).arg(out).arg(&n).arg(&passes).arg(&value).launch(cfg),
                Kernel::Write => stream.launch_builder(write).arg(b).arg(&n).arg(&passes).arg(&value).launch(cfg),
                Kernel::Copy => stream.launch_builder(copy).arg(&*a).arg(b).arg(&n).arg(&passes).launch(cfg),
            }?;
        }
        let end = stream.record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
        Ok(start.elapsed_ms(&end)? as f64 / 1e3)
    }

    fn run_transfer(&mut self, dir: Direction, bytes: usize, reps: u32) -> Result<f64> {
        if self.host.as_ref().map(|h| h.0) != Some(bytes) {
            self.host = None;
            // Plain page-locked memory (not write-combined) so both directions run at full DMA speed.
            let pinned = unsafe { self.ctx.alloc_pinned_with_flags::<u8>(bytes, 0)? };
            self.host = Some((bytes, pinned));
        }
        if self.dev_bytes.as_ref().map(|d| d.0) != Some(bytes) {
            self.dev_bytes = None;
            self.dev_bytes = Some((bytes, self.stream.alloc_zeros::<u8>(bytes)?));
        }
        let mut host = self.host.take().unwrap();
        let mut dev = self.dev_bytes.take().unwrap();
        let res = self.timed(|| {
            for _ in 0..reps {
                match dir {
                    Direction::HostToDevice => self.stream.memcpy_htod(&host.1, &mut dev.1)?,
                    Direction::DeviceToHost => self.stream.memcpy_dtoh(&dev.1, &mut host.1)?,
                }
            }
            Ok(())
        });
        self.host = Some(host);
        self.dev_bytes = Some(dev);
        res
    }

    fn run_latency(&mut self, bytes: usize, steps: u32, stride: usize) -> Result<f64> {
        let key = (bytes, stride);
        if self.chase_buf.as_ref().map(|c| c.0) != Some(key) {
            self.chase_buf = None;
            let data = build_chase(bytes, stride, 0x9E37_79B9_7F4A_7C15);
            self.chase_buf = Some((key, self.stream.clone_htod(&data)?));
        }
        let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1, 1, 1), shared_mem_bytes: 0 };
        let (stream, out, chase) = (&self.stream, &mut self.out, &self.chase);
        let buf = &self.chase_buf.as_ref().unwrap().1;
        let start = stream.record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
        unsafe { stream.launch_builder(chase).arg(buf).arg(out).arg(&steps).launch(cfg)? };
        let end = stream.record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
        Ok(start.elapsed_ms(&end)? as f64 / 1e3)
    }
}
