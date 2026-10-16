//! Benchmark harness: sizing, repetitions, statistics.

use crate::backends::{Backend, Direction, Kernel};
use anyhow::{Result, bail};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, clap::ValueEnum, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Test {
    Read,
    Write,
    Copy,
    H2d,
    D2h,
    Latency,
}

impl Test {
    pub const ALL: [Test; 6] = [Test::Read, Test::Write, Test::Copy, Test::H2d, Test::D2h, Test::Latency];

    pub fn label(self) -> &'static str {
        match self {
            Test::Read => "Read",
            Test::Write => "Write",
            Test::Copy => "Copy",
            Test::H2d => "Host->Dev",
            Test::D2h => "Dev->Host",
            Test::Latency => "Latency",
        }
    }

    pub fn unit(self) -> &'static str {
        if self == Test::Latency { "ns" } else { "GB/s" }
    }

    /// For latency lower is better; for bandwidth higher is better.
    pub fn higher_is_better(self) -> bool {
        self != Test::Latency
    }
}

pub struct Config {
    pub threads: usize,
    pub iterations: u32,
    /// Approximate bytes moved per timed sample (bandwidth tests).
    pub target_bytes: f64,
    pub latency_steps: u32,
    pub latency_stride: usize,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct Measurement {
    pub test: Test,
    pub size_bytes: usize,
    pub unit: &'static str,
    pub best: f64,
    pub median: f64,
    pub worst: f64,
    pub mean: f64,
    pub stddev: f64,
    pub samples: Vec<f64>,
}

pub fn run(backend: &mut dyn Backend, test: Test, bytes: usize, cfg: &Config) -> Result<Measurement> {
    let bytes = if test == Test::Latency { bytes } else { bytes / 16 * 16 };
    if bytes < 16 {
        bail!("size too small");
    }

    // One sample = (seconds, bytes moved or steps taken).
    let sample = |b: &mut dyn Backend| -> Result<(f64, f64)> {
        Ok(match test {
            Test::Read | Test::Write | Test::Copy => {
                let (kernel, factor) = match test {
                    Test::Read => (Kernel::Read, 1.0),
                    Test::Write => (Kernel::Write, 1.0),
                    _ => (Kernel::Copy, 2.0), // read + write, STREAM convention
                };
                let per_pass = bytes as f64 * factor;
                let passes = (cfg.target_bytes / per_pass).round().clamp(1.0, 100_000.0) as u32;
                let t = b.run_kernel(kernel, bytes, passes, cfg.threads)?;
                (t, per_pass * passes as f64)
            }
            Test::H2d | Test::D2h => {
                let dir = if test == Test::H2d { Direction::HostToDevice } else { Direction::DeviceToHost };
                let reps = (cfg.target_bytes / 2.0 / bytes as f64).round().clamp(1.0, 1000.0) as u32;
                let t = b.run_transfer(dir, bytes, reps)?;
                (t, bytes as f64 * reps as f64)
            }
            Test::Latency => {
                let t = b.run_latency(bytes, cfg.latency_steps, cfg.latency_stride)?;
                (t, cfg.latency_steps as f64)
            }
        })
    };

    sample(backend)?; // warm-up: allocation, page faults, clocks ramping up

    let mut values = Vec::with_capacity(cfg.iterations as usize);
    for _ in 0..cfg.iterations {
        let (secs, amount) = sample(backend)?;
        if secs <= 0.0 {
            bail!("GPU reported a non-positive duration");
        }
        values.push(if test == Test::Latency { secs * 1e9 / amount } else { amount / secs / 1e9 });
    }
    Ok(summarize(test, bytes, values))
}

fn summarize(test: Test, bytes: usize, samples: Vec<f64>) -> Measurement {
    let mut sorted = samples.clone();
    sorted.sort_by(f64::total_cmp);
    if test.higher_is_better() {
        sorted.reverse();
    }
    let n = sorted.len() as f64;
    let mean = sorted.iter().sum::<f64>() / n;
    let var = sorted.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    let median = if sorted.len() % 2 == 0 {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
    } else {
        sorted[sorted.len() / 2]
    };
    Measurement {
        test,
        size_bytes: bytes,
        unit: test.unit(),
        best: sorted[0],
        median,
        worst: *sorted.last().unwrap(),
        mean,
        stddev: var.sqrt(),
        samples,
    }
}

/// Power-of-two sizes from `min` to `max` inclusive.
pub fn sweep_sizes(min: usize, max: usize) -> Vec<usize> {
    let mut sizes = Vec::new();
    let mut s = min.next_power_of_two();
    while s <= max {
        sizes.push(s);
        s *= 2;
    }
    sizes
}
