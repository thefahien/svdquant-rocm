# SVDQuant W4A4 and RDNA WMMA on AMD Radeon GPUs (RX 7900 XTX / RDNA3 / RDNA4)

This repo benchmarks and patches **SVDQuant W4A4** (`v_wmma_i32_16x16x16_iu4`), **INT8 ConvRot**
(`v_wmma_i32_16x16x16_iu8`), and **BF16/FP16** (`rocBLAS` / `hipBLASLt` / `TunableOp`) matrix kernels on AMD
Radeon GPUs at Flux and Qwen-Image diffusion-transformer (DiT) shapes.

The patch and build script target **RDNA3, RDNA3.5, and RDNA4 GPUs** (`gfx1100`–`gfx1151`, `gfx1200`–`gfx1201`;
`gfx1030` RDNA2 compiles with a non-WMMA `fp32` fallback for `LoRA-down` only). Benchmarks in this repo were
measured on an **AMD Radeon RX 7900 XTX (`gfx1100`)**.

It contains the `gfx1100` build recipe, the unmodified [Comfy-Org/comfy-kitchen#262](https://github.com/Comfy-Org/comfy-kitchen/pull/262)
patch (`08216d9`), the benchmark scripts, the `.jsonl` benchmark logs, and primary-source RDNA3 WMMA roofline and
register-layout notes.

Short version:

- **SVDQuant W4A4 (`iu4` WMMA):** The `comfy-kitchen` wheel shipped in ComfyUI today (through `v0.2.41`) runs W4A4
  on `gfx1100` at **15.3–19.4 TOPS** (4–7x slower than BF16). Applying `0001-hip-svdquant-w4a4-perf.patch` (which
  was only run-tested on `gfx1200` and closed unmerged) brings the RX 7900 XTX to **123–139 INT4 TOPS (50–56% of
  the 246 TOPS hardware peak)** — **7.4–9.2x faster than shipped**, **1.57–1.76x faster than default PyTorch BF16
  (`rocBLAS`)**, and **1.08–1.41x faster than `TunableOp`-tuned BF16** while using **3.6x less weight memory**.
- **BF16/FP16 BLAS baseline trap:** PyTorch defaults to `rocBLAS` on `gfx1100` (53–72 TFLOPS). Switching to
  `hipBLASLt` gives +15–50%, and `PYTORCH_TUNABLEOP_ENABLED=1` reaches **88–92 TFLOPS (72–75% of the 123 TFLOPS
  WMMA peak)**.
- **INT8 ConvRot (`int8_linear`):** `comfy-kitchen`'s HIP INT8 WMMA kernel already runs at **97.6–104.2 TOPS
  (79–85% of the 123 TOPS peak)** across all 192 Qwen-Image-2.1 DiT linears (`606 ms` of the `763 ms` step).
  Because INT8 and BF16 share the same 123 TOPS matrix ceiling on RDNA3, INT4 (`246 TOPS` peak) is the only
  hardware path with higher matrix throughput.

## Results

RX 7900 XTX (`gfx1100`, stock 339 W), `torch 2.13.0+rocm7.1`, `HIP 7.1.52802`, rank 32, group size 64, `bf16`.
`"total"` = activation smoothing + INT4 quantize + `LoRA-down` (`M x K x 32`) + W4A4 GEMM + `LoRA-up` (`M x 32 x N`).
Full tables and methodology: [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

| M x K x N | shipped 0.2.36: total / GEMM TOPS | patched: total / GEMM TOPS | vs default BF16 (`rocBLAS`) | vs `hipBLASLt` BF16 | vs `TunableOp` BF16 |
|---|---|---|---|---|---|
| 4096 x 3072 x 3072 | 9.14 ms / 15.3 | **0.81 ms / 123.0** | **1.76x** (1.44 ms) | **1.19x** (0.97 ms) | **1.08x** (0.88 ms) |
| 4096 x 3072 x 12288 | 22.86 ms / 16.6 | **2.50 ms / 128.4** | **1.69x** (4.23 ms) | **1.49x** (3.72 ms) | **1.41x** (3.53 ms) |
| 4096 x 12288 x 3072 | 35.96 ms / 19.4 | **3.08 ms / 138.7** | **1.69x** (5.20 ms) | **1.19x** (3.67 ms) | **1.14x** (3.50 ms) |
| 4096 x 4096 x 4096 | 13.99 ms / 16.5 | **1.35 ms / 127.8** | **1.57x** (2.12 ms) | **1.18x** (1.59 ms) | **1.12x** (1.50 ms) |
| 8192 x 3072 x 12288 | 45.14 ms / 16.7 | **5.18 ms / 127.7** | **1.65x** (8.56 ms) | **1.45x** (7.50 ms) | **1.36x** (7.02 ms) |
| 256 x 3072 x 3072 | 1.55 ms / 5.9 | 0.17 ms / 72.7 | 0.44x (0.074 ms) | — | — |

Raw data: [`results/7900xtx-svdquant-w4a4-shipped.jsonl`](results/7900xtx-svdquant-w4a4-shipped.jsonl),
[`results/7900xtx-svdquant-w4a4-patched.jsonl`](results/7900xtx-svdquant-w4a4-patched.jsonl),
[`results/7900xtx-bf16-fp16-blas-tune.jsonl`](results/7900xtx-bf16-fp16-blas-tune.jsonl).

- **Correctness:** HIP W4A4 relative error vs `fp32` ground truth matches the `fp32`-accumulating eager reference
  to 4 significant figures on every shape (`0.17315` vs `0.17319`).
- **Small `M` crossover:** At `M = 256`, the W4A4 GEMM runs at 72.7 TOPS (`0.066 ms`), but activation quantize +
  LoRA overhead (`0.100 ms`) makes total layer time slower than a single BF16 GEMM (`0.074 ms`). Below `M ~= 64`,
  all paths become memory-bound (see [docs/ROOFLINE.md](docs/ROOFLINE.md)).
- **Negative control (Qwen-Image-2.1 `int8_convrot`):** Enabling `hipBLASLt` or `TunableOp` changes end-to-end
  ComfyUI sampling speed by **0.0%** (`0.763 s/step` across all 5 arms), because all 192 DiT linear layers dispatch
  to `comfy-kitchen`'s custom HIP `int8_linear` kernel rather than BLAS.

## Supported AMD GPUs

The HIP WMMA kernels dispatch per architecture. Pass `COMFY_HIP_ARCHS` to `scripts/build_patched_kitchen.sh`
for your card (defaults to `gfx1100`). Only `gfx1100` (measured here) and `gfx1200` (measured in PR #262) have
been run-verified on hardware; other `gfx11xx`/`gfx12xx` targets share the same ISA instructions:

| Architecture | GPUs | `COMFY_HIP_ARCHS` | Hardware W4A4 Status |
|---|---|---|---|
| **RDNA3** | RX 7900 XTX, 7900 XT, 7900 GRE | `gfx1100` (default) | `v_wmma_i32_16x16x16_iu4` (**run-verified:** 123–139 TOPS) |
| **RDNA3** | RX 7800 XT, 7700 XT | `gfx1101` | `v_wmma_i32_16x16x16_iu4` (compile-verified) |
| **RDNA3** | RX 7600 XT, 7600 | `gfx1102` | `v_wmma_i32_16x16x16_iu4` (compile-verified) |
| **RDNA3 / 3.5 APU** | Radeon 780M, 890M, Ryzen AI Max+ (Strix Halo) | `gfx1103` / `gfx1150` / `gfx1151` | `v_wmma_i32_16x16x16_iu4` (compile-verified) |
| **RDNA4** | RX 9070 XT, RX 9070 | `gfx1200` / `gfx1201` | `v_wmma_i32_16x16x16_iu4` (**run-verified in PR #262:** 73–84 TOPS) |
| **RDNA2** | RX 6950 XT, 6900 XT, 6800 XT, 6800 | `gfx1030` | Compile-verified (`LoRA-down` falls back to `fp32`; no WMMA for W4A4 GEMM) |

## Setup and Reproducing

Builds in an isolated directory (~25 seconds with 8 jobs) using AMD's pip ROCm SDK if system `hipcc` is absent:

```bash
git clone https://github.com/thefahien/svdquant-rocm && cd svdquant-rocm

# Default (gfx1100: RX 7900 XTX / XT / GRE). For other cards, prefix e.g. COMFY_HIP_ARCHS=gfx1101 or gfx1200:
bash scripts/build_patched_kitchen.sh ./build/ck-rocm

PYTHONPATH=./build/ck-rocm/site python3 bench/bench_svdquant_w4a4.py
PYTHONPATH=./build/ck-rocm/site python3 bench/bench_int8_convrot.py
bash scripts/run_blas_tune.sh
```

## Documentation

- [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md): Full W4A4, BF16/FP16 BLAS, INT8 ConvRot, and ComfyUI end-to-end tables.
- [`docs/ROOFLINE.md`](docs/ROOFLINE.md): RX 7900 XTX (`gfx1100`) WMMA peak throughput verification, LLVM
  `AMDGPUWMMAIntrinsicsGFX11` proof, and arithmetic-intensity roofline (`M = 1..4096`).
- [`docs/WMMA_LAYOUT.md`](docs/WMMA_LAYOUT.md): Wave32 per-lane VGPR and matrix-element mappings for
  `v_wmma_i32_16x16x16_iu4`, `iu8`, and `f16` from AMD MR-ISA XML and `amd_matrix_instruction_calculator`.
- [`docs/NUNCHAKU_SPEC.md`](docs/NUNCHAKU_SPEC.md): Nunchaku CUDA kernel inventory, `SVDQW4A4Linear` checkpoint
  format, packed `qweight` `mma.sync.m16n8k64` fragment layout, and ROCm port feasibility.
- [`patches/README.md`](patches/README.md): Details on `0001-hip-svdquant-w4a4-perf.patch`.

## Credits and Licenses

- [`Comfy-Org/comfy-kitchen`](https://github.com/Comfy-Org/comfy-kitchen) and unmodified commit `08216d9` (PR #262)
  by `0xDELUXA` ([Apache-2.0](LICENSES/Apache-2.0.txt)).
- SVDQuant / Nunchaku format reference from [`nunchaku-ai/nunchaku`](https://github.com/nunchaku-ai/nunchaku)
  ([Apache-2.0](LICENSES/Apache-2.0.txt)).
- Excerpts from AMD GPUOpen and AMD RDNA3 ISA documentation in `docs/` remain the property of Advanced Micro Devices,
  Inc.
- Scripts, benchmarks, and original documentation in this repo: [MIT](LICENSE).
