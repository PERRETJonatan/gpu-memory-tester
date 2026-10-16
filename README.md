# gpu-memory-tester

GPU memory bandwidth & latency tester written in Rust, with **Metal** (macOS / Apple silicon)
and **CUDA** (NVIDIA) backends. All timings are taken on the GPU (Metal command-buffer
timestamps, CUDA events), so CPU submission overhead is not measured.

## Tests

| Test        | What it measures |
|-------------|------------------|
| `read`      | Kernel streaming reads (`float4`, coalesced, grid-stride) |
| `write`     | Kernel streaming writes |
| `copy`      | Kernel `dst[i] = src[i]`; counts read + write bytes (STREAM convention) |
| `h2d`/`d2h` | Host <-> device transfer (CUDA: pinned memcpy over PCIe/NVLink; Metal: blit between shared and private buffers — same DRAM on Apple silicon) |
| `latency`   | Single-thread pointer chase through a random cycle (Sattolo), ns per dependent load |

Bandwidth is decimal GB/s (1e9 B/s). Each sample sweeps the buffer enough times to move
~2 GB (`--target`), so small, cache-resident buffers still yield meaningful timings.

## Build

```sh
cargo build --release
```

The same command works on every OS: Metal is used on macOS, CUDA wherever an NVIDIA
driver is present. CUDA is loaded at runtime, so no CUDA toolkit is needed to compile.

To **run** the CUDA backend you need:
- the NVIDIA proprietary driver (`nvidia-smi` should work), which provides `libcuda.so`;
- NVRTC from CUDA 12.x (`libnvrtc.so.12`), e.g. the `cuda-nvrtc-12-x` package or
  `pip install nvidia-cuda-nvrtc-cu12`, with its lib directory on `LD_LIBRARY_PATH`.

Run `gpu-memory-tester --list` to see which backends are usable and why not.

## Usage

```sh
gpu-memory-tester                          # all tests at 1 GiB
gpu-memory-tester --list                   # list devices
gpu-memory-tester -t read,copy -s 4G       # pick tests and size
gpu-memory-tester --sweep --max-size 2G    # 16 KiB .. 2 GiB: shows caches vs DRAM
gpu-memory-tester --sweep -f csv > out.csv # full stats (best/median/worst/mean/stddev)
gpu-memory-tester -b cuda -d 1             # backend/device selection
```

See `--help` for `--iterations`, `--threads`, `--latency-steps`, `--latency-stride`, `--target`.

## What it does (and doesn't)

The tool only allocates GPU/host buffers, runs the benchmark kernels shown in
`src/backends/`, and prints results to stdout. It does not access the network, read or
write files, spawn processes, or need elevated privileges. Memory use is bounded by the
chosen buffer size (default 1 GiB, capped to 1/4 of device memory).

## License

MIT — see [LICENSE](LICENSE).
