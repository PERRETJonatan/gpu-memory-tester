mod backends;
mod bench;
mod units;

use anyhow::Result;
use backends::{BackendKind, DeviceInfo};
use bench::{Measurement, Test};
use clap::Parser;
use std::io::{IsTerminal, Write};
use units::{fmt_bytes, parse_size};

/// GPU memory bandwidth & latency tester (Metal / CUDA).
///
/// Bandwidth is reported in decimal GB/s (1e9 bytes/s). Copy counts bytes
/// read + bytes written (STREAM convention). All timings come from the GPU.
#[derive(Parser, Debug)]
#[command(version, about)]
struct Cli {
    /// GPU backend to use.
    #[arg(short, long, value_enum, default_value_t = BackendKind::Auto)]
    backend: BackendKind,

    /// Device index (see --list).
    #[arg(short, long, default_value_t = 0)]
    device: usize,

    /// List available devices and exit.
    #[arg(short, long)]
    list: bool,

    /// Tests to run (comma separated).
    #[arg(short, long, value_enum, value_delimiter = ',', default_values_t = Test::ALL)]
    tests: Vec<Test>,

    /// Buffer size for a single-size run (e.g. 256M, 1G). Default: 1 GiB, capped to 1/4 of device memory.
    #[arg(short, long, value_parser = parse_size)]
    size: Option<usize>,

    /// Sweep power-of-two buffer sizes from --min-size to --max-size to expose the cache hierarchy.
    #[arg(long)]
    sweep: bool,

    #[arg(long, value_parser = parse_size, default_value = "16K")]
    min_size: usize,

    /// Default: same as --size.
    #[arg(long, value_parser = parse_size)]
    max_size: Option<usize>,

    /// Timed samples per measurement (after one warm-up).
    #[arg(short, long, default_value_t = 10)]
    iterations: u32,

    /// Threads launched by the bandwidth kernels (grid-stride). Default depends on the device.
    #[arg(long)]
    threads: Option<usize>,

    /// Approximate traffic per timed sample, so small buffers still produce measurable runs.
    #[arg(long, value_parser = parse_size, default_value = "2G")]
    target: usize,

    /// Dependent loads per latency sample.
    #[arg(long, default_value_t = 100_000)]
    latency_steps: u32,

    /// Distance between pointer-chase nodes (use >= cache line size).
    #[arg(long, value_parser = parse_size, default_value = "128")]
    latency_stride: usize,

    /// Output format.
    #[arg(short, long, value_enum, default_value_t = Format::Table)]
    format: Format,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
enum Format {
    Table,
    Json,
    Csv,
}

fn main() {
    if let Err(e) = real_main() {
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}

fn real_main() -> Result<()> {
    let cli = Cli::parse();

    if cli.list {
        let devices = backends::list_devices();
        if devices.is_empty() {
            println!("No GPU devices found.");
        }
        for d in devices {
            println!("{d}");
        }
        return Ok(());
    }

    let mut backend = backends::create(cli.backend, cli.device)?;
    let info = backend.info().clone();

    let default_size = (1usize << 30).min((info.memory_bytes / 4) as usize).max(1 << 20);
    let size = cli.size.unwrap_or(default_size);
    let sizes = if cli.sweep {
        let s = bench::sweep_sizes(cli.min_size, cli.max_size.unwrap_or(size));
        anyhow::ensure!(!s.is_empty(), "empty sweep range");
        s
    } else {
        vec![size]
    };

    let cfg = bench::Config {
        threads: cli.threads.unwrap_or_else(|| backend.default_threads()),
        iterations: cli.iterations.max(1),
        target_bytes: cli.target as f64,
        latency_steps: cli.latency_steps.max(1),
        latency_stride: cli.latency_stride.max(4),
    };

    let mut tests = cli.tests.clone();
    tests.dedup();

    if cli.format == Format::Table {
        print_header(&info, &cfg);
    }

    let progress = std::io::stderr().is_terminal();
    let total = tests.len() * sizes.len();
    let mut results = Vec::with_capacity(total);
    for &test in &tests {
        for &bytes in &sizes {
            if progress {
                eprint!("\r\x1b[2K  [{}/{}] {} {} ...", results.len() + 1, total, test.label(), fmt_bytes(bytes));
                let _ = std::io::stderr().flush();
            }
            let m = bench::run(backend.as_mut(), test, bytes, &cfg)
                .map_err(|e| e.context(format!("{} @ {}", test.label(), fmt_bytes(bytes))))?;
            results.push(m);
        }
    }
    if progress {
        eprint!("\r\x1b[2K");
    }

    match cli.format {
        Format::Table if cli.sweep => print_sweep(&tests, &sizes, &results),
        Format::Table => print_table(&results),
        Format::Json => print_json(&info, &cfg, &results)?,
        Format::Csv => print_csv(&info, &results),
    }
    Ok(())
}

fn print_header(info: &DeviceInfo, cfg: &bench::Config) {
    println!("GPU Memory Tester");
    println!("  Backend  : {}", info.backend);
    println!("  Device   : {}", info.name);
    println!(
        "  Memory   : {}{}",
        fmt_bytes(info.memory_bytes as usize),
        if info.unified_memory { " (unified)" } else { "" }
    );
    for (k, v) in &info.details {
        println!("  {k:<9}: {v}");
    }
    println!("  Threads  : {}   Iterations: {}", cfg.threads, cfg.iterations);
    if info.unified_memory {
        println!("  Note     : unified memory; Host<->Dev is a GPU blit within the same DRAM.");
    }
    println!();
}

/// Formats a measurement with its unit, scaling up when the value gets large
/// (GB/s -> TB/s, ns -> µs).
fn fmt_val(v: f64, test: Test) -> String {
    let (base, big) = if test == Test::Latency { ("ns", "µs") } else { ("GB/s", "TB/s") };
    if v >= 1000.0 { format!("{:.2} {big}", v / 1000.0) } else { format!("{v:.1} {base}") }
}

fn print_table(results: &[Measurement]) {
    println!("{:<10} {:>10} {:>13} {:>13} {:>13} {:>8}", "Test", "Size", "Best", "Median", "Worst", "StdDev");
    println!("{}", "-".repeat(73));
    for m in results {
        println!(
            "{:<10} {:>10} {:>13} {:>13} {:>13} {:>8}",
            m.test.label(),
            fmt_bytes(m.size_bytes),
            fmt_val(m.best, m.test),
            fmt_val(m.median, m.test),
            fmt_val(m.worst, m.test),
            format!("±{:.1}%", m.stddev / m.mean * 100.0),
        );
    }
}

fn print_sweep(tests: &[Test], sizes: &[usize], results: &[Measurement]) {
    print!("{:>10}", "Size");
    for t in tests {
        print!(" {:>13}", t.label());
    }
    println!();
    println!("{}", "-".repeat(10 + 14 * tests.len()));
    for &size in sizes {
        print!("{:>10}", fmt_bytes(size));
        for &t in tests {
            let m = results.iter().find(|m| m.test == t && (m.size_bytes == size || m.size_bytes == size / 16 * 16));
            match m {
                Some(m) => print!(" {:>13}", fmt_val(m.best, t)),
                None => print!(" {:>13}", "-"),
            }
        }
        println!();
    }
    println!("\n(best of each measurement shown; use --format csv/json for full statistics)");
}

fn print_json(info: &DeviceInfo, cfg: &bench::Config, results: &[Measurement]) -> Result<()> {
    let out = serde_json::json!({
        "device": {
            "backend": info.backend,
            "name": info.name,
            "memory_bytes": info.memory_bytes,
            "unified_memory": info.unified_memory,
            "details": info.details.iter().cloned().collect::<std::collections::BTreeMap<_, _>>(),
        },
        "config": {
            "threads": cfg.threads,
            "iterations": cfg.iterations,
            "target_bytes": cfg.target_bytes,
            "latency_steps": cfg.latency_steps,
            "latency_stride": cfg.latency_stride,
        },
        "results": results,
    });
    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}

fn print_csv(info: &DeviceInfo, results: &[Measurement]) {
    println!("backend,device,test,size_bytes,unit,best,median,worst,mean,stddev");
    for m in results {
        println!(
            "{},\"{}\",{},{},{},{:.3},{:.3},{:.3},{:.3},{:.3}",
            info.backend,
            info.name,
            format!("{:?}", m.test).to_lowercase(),
            m.size_bytes,
            m.unit,
            m.best,
            m.median,
            m.worst,
            m.mean,
            m.stddev
        );
    }
}
