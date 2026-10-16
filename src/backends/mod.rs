//! GPU backend abstraction. Each backend times work on the GPU itself
//! (command-buffer timestamps on Metal, events on CUDA), so CPU-side
//! submission overhead is excluded from the measurements.

use anyhow::{Result, bail};

#[cfg(all(feature = "metal", target_os = "macos"))]
pub mod metal;

#[cfg(feature = "cuda")]
pub mod cuda;

/// On-device bandwidth kernels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kernel {
    Read,
    Write,
    Copy,
}

/// Host <-> device transfer direction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    HostToDevice,
    DeviceToHost,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct DeviceInfo {
    pub backend: &'static str,
    pub name: String,
    pub memory_bytes: u64,
    pub unified_memory: bool,
    /// Extra free-form details (compute units, clocks, ...).
    pub details: Vec<(String, String)>,
}

pub trait Backend {
    fn info(&self) -> &DeviceInfo;

    /// Default number of threads launched by the grid-stride kernels.
    fn default_threads(&self) -> usize;

    /// Runs `kernel` over a `bytes`-sized buffer, sweeping it `passes` times
    /// inside a single dispatch. Returns GPU time in seconds.
    fn run_kernel(&mut self, kernel: Kernel, bytes: usize, passes: u32, threads: usize) -> Result<f64>;

    /// Performs `reps` transfers of `bytes` each. Returns GPU time in seconds.
    fn run_transfer(&mut self, dir: Direction, bytes: usize, reps: u32) -> Result<f64>;

    /// Single-threaded dependent-load chase through a `bytes`-sized random
    /// cycle, `steps` hops long. Returns GPU time in seconds.
    fn run_latency(&mut self, bytes: usize, steps: u32, stride: usize) -> Result<f64>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum BackendKind {
    Auto,
    Metal,
    Cuda,
}

pub fn list_devices() -> Vec<String> {
    #[allow(unused_mut)]
    let mut out = Vec::new();
    #[cfg(all(feature = "metal", target_os = "macos"))]
    out.extend(metal::list_devices().into_iter().map(|n| format!("metal:{n}")));
    #[cfg(feature = "cuda")]
    out.extend(cuda::list_devices().into_iter().map(|n| format!("cuda:{n}")));
    out
}

#[allow(unused_variables)]
pub fn create(kind: BackendKind, device: usize) -> Result<Box<dyn Backend>> {
    let try_cuda = matches!(kind, BackendKind::Auto | BackendKind::Cuda);
    let try_metal = matches!(kind, BackendKind::Auto | BackendKind::Metal);
    #[allow(unused_mut)]
    let mut errors: Vec<String> = Vec::new();

    if try_cuda {
        #[cfg(feature = "cuda")]
        match cuda::CudaBackend::new(device) {
            Ok(b) => return Ok(Box::new(b)),
            Err(e) => errors.push(format!("cuda: {e:#}")),
        }
        #[cfg(not(feature = "cuda"))]
        if kind == BackendKind::Cuda {
            errors.push("cuda: not compiled in (rebuild with `--features cuda`)".into());
        }
    }
    if try_metal {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        match metal::MetalBackend::new(device) {
            Ok(b) => return Ok(Box::new(b)),
            Err(e) => errors.push(format!("metal: {e:#}")),
        }
        #[cfg(not(all(feature = "metal", target_os = "macos")))]
        if kind == BackendKind::Metal {
            errors.push("metal: not available (macOS only, `metal` feature)".into());
        }
    }
    if errors.is_empty() {
        bail!("no GPU backend compiled in; enable the `metal` or `cuda` feature");
    }
    bail!("no usable GPU backend:\n  {}", errors.join("\n  "))
}

/// Builds a single random cycle over `bytes / stride` nodes (Sattolo's
/// algorithm), laid out in a u32 index array where node `k` lives at element
/// `k * stride / 4` and stores the element index of its successor.
pub fn build_chase(bytes: usize, stride: usize, seed: u64) -> Vec<u32> {
    let words_per_node = (stride / 4).max(1);
    let nodes = (bytes / 4 / words_per_node).max(2);
    let mut order: Vec<u32> = (0..nodes as u32).collect();
    let mut rng = seed | 1;
    let mut next = || {
        // xorshift64*
        rng ^= rng >> 12;
        rng ^= rng << 25;
        rng ^= rng >> 27;
        rng.wrapping_mul(0x2545_F491_4F6C_DD1D)
    };
    for i in (1..nodes).rev() {
        let j = (next() % i as u64) as usize; // j < i => single cycle
        order.swap(i, j);
    }
    let mut data = vec![0u32; nodes * words_per_node];
    for k in 0..nodes {
        let from = order[k] as usize;
        let to = order[(k + 1) % nodes] as usize;
        data[from * words_per_node] = (to * words_per_node) as u32;
    }
    data
}

#[cfg(test)]
mod tests {
    use super::build_chase;

    #[test]
    fn chase_is_single_cycle() {
        let stride = 64;
        let data = build_chase(1 << 16, stride, 42);
        let nodes = data.len() / (stride / 4);
        let mut seen = vec![false; nodes];
        let mut p = 0usize;
        for _ in 0..nodes {
            let node = p / (stride / 4);
            assert!(!seen[node], "revisited node before covering all");
            seen[node] = true;
            p = data[p] as usize;
        }
        assert_eq!(p, 0);
        assert!(seen.iter().all(|&s| s));
    }
}
