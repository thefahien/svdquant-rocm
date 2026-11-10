# Benchmarks

All measurements were run on an AMD Radeon RX 7900 XTX (`gfx1100`, 96 CUs, 24 GB GDDR6, stock 339 W power limit)
with `torch 2.13.0+rocm7.1` and `HIP 7.1.52802`.

Hardware matrix peaks on the RX 7900 XTX (at 2500 MHz boost, from the official AMD spec sheet and GPUOpen WMMA
table; see [ROOFLINE.md](ROOFLINE.md)):

- **INT4 WMMA (`v_wmma_i32_16x16x16_iu4`):** 246 TOPS (1024 OPS/clock/CU)
- **INT8 WMMA (`v_wmma_i32_16x16x16_iu8`):** 123 TOPS (512 OPS/clock/CU)
- **BF16 / FP16 WMMA (`v_wmma_f32_16x16x16_{bf16,f16}`):** 123 TFLOPS (512 FLOPS/clock/CU)
- **FP16 / FP32 Vector VALU:** 61.4 TFLOPS

---

## 1. SVDQuant W4A4 (`comfy-kitchen` HIP) on `gfx1100`

Script: [`bench/bench_svdquant_w4a4.py`](../bench/bench_svdquant_w4a4.py) (synthetic DiT-shaped layers, rank 32,
`bf16`, group size 64, 3 warm-up + 20 timed iterations).
`"total"` includes activation smoothing + INT4 quantization + `LoRA-down` (`M x K x 32`) + W4A4 GEMM + `LoRA-up`
(`M x 32 x N`). `"GEMM TOPS"` is `2 * M * K * N / t_gemm`.

### Correctness

On every tested shape, the HIP output's relative error against an `fp32` ground truth matches the
`fp32`-accumulating eager reference to 4 significant figures (e.g. `0.17315` vs `0.17319` at `4096x3072x3072`).
All outputs are finite. About 0.5% of activation codes differ by one step from eager on rounding boundaries.
(The ~17% absolute error against `fp32` comes from uncalibrated random synthetic weights without offline SVD
outlier extraction; it is not a measure of image quality on a trained checkpoint.)

### Speed: shipped `0.2.36` vs patched (`08216d9`) vs BF16 baselines

Upstream `comfy-kitchen` (up through `0.2.41`) ships the initial `ec72ea1` `svdquant_w4a4.hip` kernel.
Patch [`0001-hip-svdquant-w4a4-perf.patch`](../patches/0001-hip-svdquant-w4a4-perf.patch) applies
[Comfy-Org/comfy-kitchen#262](https://github.com/Comfy-Org/comfy-kitchen/pull/262) (`08216d9`, which was
run-verified by its author on `gfx1200` only and closed unmerged on 2026-11-05) and builds it for `gfx1100`.

| M x K x N | shipped 0.2.36: total / GEMM TOPS | patched (`08216d9`): total / GEMM TOPS | vs default BF16 (`rocBLAS`) | vs `hipBLASLt` BF16 | vs `TunableOp` BF16 |
|---|---|---|---|---|---|
| 4096 x 3072 x 3072 | 9.14 ms / 15.3 | **0.81 ms / 123.0** | **1.76x** (1.44 ms) | **1.19x** (0.97 ms) | **1.08x** (0.88 ms) |
| 4096 x 3072 x 12288 | 22.86 ms / 16.6 | **2.50 ms / 128.4** | **1.69x** (4.23 ms) | **1.49x** (3.72 ms) | **1.41x** (3.53 ms) |
| 4096 x 12288 x 3072 | 35.96 ms / 19.4 | **3.08 ms / 138.7** | **1.69x** (5.20 ms) | **1.19x** (3.67 ms) | **1.14x** (3.50 ms) |
| 4096 x 4096 x 4096 | 13.99 ms / 16.5 | **1.35 ms / 127.8** | **1.57x** (2.12 ms) | **1.18x** (1.59 ms) | **1.12x** (1.50 ms) |
| 8192 x 3072 x 12288 | 45.14 ms / 16.7 | **5.18 ms / 127.7** | **1.65x** (8.56 ms) | **1.45x** (7.50 ms) | **1.36x** (7.02 ms) |
| 256 x 3072 x 3072 | 1.55 ms / 5.9 | 0.17 ms / 72.7 | 0.44x (0.074 ms) | — | — |

Raw data:
[`results/7900xtx-svdquant-w4a4-shipped.jsonl`](../results/7900xtx-svdquant-w4a4-shipped.jsonl),
[`results/7900xtx-svdquant-w4a4-patched.jsonl`](../results/7900xtx-svdquant-w4a4-patched.jsonl).

Reading it:

- The shipped `comfy-kitchen` wheel runs W4A4 on `gfx1100`, but at **15.3–19.4 TOPS (4–7x slower than BF16)**.
- With the patch built for `gfx1100`, the W4A4 GEMM reaches **123–139 INT4 TOPS (50–56% of the 246 TOPS peak)** —
  a **7.4–9.2x kernel speedup** over the shipped wheel.
- Including activation quantization and the rank-32 LoRA branch, total layer time is **1.57–1.76x faster** than
  PyTorch's default BF16 (`rocBLAS`), and **1.08–1.41x faster** than `TunableOp`-tuned BF16, while using **~3.6x
  less weight memory**.
- At `M = 256`, the GEMM itself still runs at 72.7 TOPS (`0.066 ms`), but activation quantization + LoRA overhead
  (`0.100 ms`) makes the total call slower than a single BF16 GEMM (`0.074 ms`).

---

## 2. BF16 and FP16 BLAS Tuning (`rocBLAS` vs `hipBLASLt` vs `TunableOp`)

Script: [`bench/bench_blas_modes.py`](../bench/bench_blas_modes.py), runner:
[`scripts/run_blas_tune.sh`](../scripts/run_blas_tune.sh).
5 warm-up + 30 timed iterations; each arm runs in a fresh Python process.
Max relative error vs an `fp32` reference slice is identical across all settings (`0.0021–0.0036` for BF16,
`0.00027–0.00031` for FP16), confirming that tuning changes kernel selection, not numerical precision.

### BF16 TFLOPS (peak = 123 TFLOPS)

| M x K x N | default (`rocBLAS`) | `hipBLASLt` | `TunableOp` | default again |
|---|---|---|---|---|
| 4096 x 3072 x 3072 | 52.9 | 79.7 | **88.3** | 41.9 |
| 4096 x 3072 x 12288 | 68.5 | 83.1 | **87.7** | 67.6 |
| 4096 x 12288 x 3072 | 61.9 | 84.2 | **88.5** | 61.7 |
| 4096 x 4096 x 4096 | 65.4 | 86.3 | **91.4** | 64.5 |
| 8192 x 3072 x 12288 | 72.3 | 82.5 | **88.1** | 71.6 |
| 4608 x 3072 x 9216 | 69.1 | 84.2 | **89.4** | 68.5 |
| 1024 x 3072 x 3072 | 63.3 | 72.8 | **92.0** | 50.6 |

### FP16 TFLOPS (peak = 123 TFLOPS)

| M x K x N | default (`rocBLAS`) | `hipBLASLt` | `TunableOp` | default again |
|---|---|---|---|---|
| 4096 x 3072 x 3072 | 54.0 | 77.4 | **77.8** | 47.0 |
| 4096 x 3072 x 12288 | 68.9 | 84.0 | **87.2** | 67.1 |
| 4096 x 12288 x 3072 | 59.3 | 78.0 | **85.8** | 59.4 |
| 4096 x 4096 x 4096 | 61.1 | 80.9 | **90.7** | 61.1 |
| 8192 x 3072 x 12288 | 71.3 | 80.4 | **88.2** | 71.4 |
| 4608 x 3072 x 9216 | 64.7 | 84.0 | **85.9** | 64.6 |
| 1024 x 3072 x 3072 | 57.2 | 72.3 | **78.2** | 47.7 |

Raw data: [`results/7900xtx-bf16-fp16-blas-tune.jsonl`](../results/7900xtx-bf16-fp16-blas-tune.jsonl).

- PyTorch defaults to `rocBLAS` on `gfx1100`. Switching to `hipBLASLt` adds **+15–50%**.
- Enabling `PYTORCH_TUNABLEOP_ENABLED=1` benchmarks every candidate kernel once per shape, writes a CSV, and
  reaches **88–92 BF16 TFLOPS (72–75% of the 123 TFLOPS hardware peak)**. Tuning these 14 shapes took ~1 minute.

---

## 3. INT8 ConvRot Kernel (`int8_linear`) on Qwen-Image-2.1 Shapes

Script: [`bench/bench_int8_convrot.py`](../bench/bench_int8_convrot.py).
Layer shapes scraped from `qwen_image_2.1_int8_convrot.safetensors`: 32 transformer blocks x 6 `int8` linears
(`q, k, v, out` at `4096x4096`; `img_mlp.gate_up` at `4096->24576`; `img_mlp.out` at `12288->4096`), at
`M = 4128` image tokens (`768x1376`). 5 warm-up + 30 timed iterations.
Hardware peak for INT8 WMMA and BF16 WMMA on RDNA3 is identical (**123 TOPS / TFLOPS**).

| Layer (x32 blocks) | `int8_linear` total | rotate + quant | `int8` GEMM | `int8` GEMM TOPS | BF16 `hipBLASLt` | BF16 `rocBLAS` |
|---|---|---|---|---|---|---|
| `attn.to_q` (`4096x4096`) | 1.51 ms | 0.13 ms | 1.37 ms | 101.3 | 1.67 ms | 2.16 ms |
| `attn.to_k` (`4096x4096`) | 1.58 ms | 0.14 ms | 1.42 ms | 97.6 | 1.70 ms | 2.11 ms |
| `attn.to_v` (`4096x4096`) | 1.51 ms | 0.13 ms | 1.37 ms | 101.4 | 1.67 ms | 2.15 ms |
| `attn.to_out.0` (`4096x4096`) | 1.55 ms | 0.14 ms | 1.36 ms | 101.8 | 1.57 ms | 2.00 ms |
| `img_mlp.gate_up` (`4096->24576`) | 8.47 ms | 0.14 ms | 8.33 ms | 99.7 | 10.11 ms | 11.67 ms |
| `img_mlp.out` (`12288->4096`) | 4.33 ms | 0.32 ms | 3.99 ms | 104.2 | 5.00 ms | 7.19 ms |
| **One forward (all 192 linears)** | **606.3 ms** | | | | **695.3 ms** | **872.7 ms** |

Raw data: [`results/7900xtx-int8-convrot-layers.jsonl`](../results/7900xtx-int8-convrot-layers.jsonl).

- `comfy-kitchen`'s HIP `int8_gemm` already runs at **97.6–104.2 TOPS (79–85% of the 123 TOPS hardware peak)** on
  the RX 7900 XTX. Walsh-Hadamard rotation + dynamic INT8 quantization adds ~9% on 4096-wide layers.
- Across all 192 image-stream linears, `int8_linear` takes **606 ms out of the 763 ms end-to-end sampling step
  (79% of step time)**, beating `hipBLASLt` BF16 (`695 ms`) by 1.15x and default `rocBLAS` BF16 (`873 ms`) by 1.44x.
- Because INT8 WMMA and BF16 WMMA share the same 123 TOPS ceiling on RDNA3, **4-bit (`iu4` WMMA at 246 TOPS peak)**
  is the only hardware path with higher matrix throughput (~123–139 achieved TOPS vs ~100 INT8 TOPS).

---

## 4. Negative Control: End-to-End Qwen-Image-2.1 (`int8_convrot`) with BLAS Tuning

Model: UNet `qwen_image_2.1_int8_convrot`, text encoder `qwen3vl_8b_w4a8`, VAE `bf16`. `768x1376`, 30 steps,
`euler/simple`, `cfg 1`. Each arm ran in a fresh ComfyUI process with 1 warm-up render + 2 timed renders
(seeds 2222, 3333).

| Arm | Sampling s/step (timed 1 / timed 2) | Wall time per image (timed 1 / timed 2) |
|---|---|---|
| default (`rocBLAS`) | 0.758 / 0.763 s | 33.1 / 33.0 s |
| `hipBLASLt` | 0.763 / 0.763 s | 33.0 / 33.0 s |
| `TunableOp` (tuning in warm-up: 438 s) | 0.763 / 0.763 s | 33.0 / 33.0 s |
| `TunableOp` (read tuned CSV, fresh process) | 0.763 / 0.763 s | 33.0 / 33.0 s |
| default again | 0.763 / 0.763 s | 39.2 / 42.4 s |

Raw data: [`results/7900xtx-qwen-image-blas-ab.jsonl`](../results/7900xtx-qwen-image-blas-ab.jsonl).

- **Result:** 0.0% change in sampling step time (`0.763 s/step` in every arm).
- **Why:** All 192 heavy DiT linear layers use `int8_convrot` and dispatch to `comfy-kitchen`'s HIP `int8_linear`
  kernel, bypassing `rocBLAS`/`hipBLASLt`. Inspection of the generated `TunableOp` CSV showed 81 small or odd shapes
  (text-encoder GEMMs at `M=45..61` and VAE `N=144/288` GEMMs), none of the 4128-token DiT layers.
- `hipBLASLt` / `TunableOp` only speeds up unquantized `bf16`/`fp16` diffusion models on RDNA3.

---

## Raw data index

| File | Contents |
|---|---|
| [`results/7900xtx-svdquant-w4a4-shipped.jsonl`](../results/7900xtx-svdquant-w4a4-shipped.jsonl) | Shipped `comfy-kitchen 0.2.36` SVDQuant W4A4 vs BF16 across 6 DiT shapes |
| [`results/7900xtx-svdquant-w4a4-patched.jsonl`](../results/7900xtx-svdquant-w4a4-patched.jsonl) | Patched `comfy-kitchen` (`08216d9`, built for `gfx1100`) SVDQuant W4A4 across 6 DiT shapes |
| [`results/7900xtx-bf16-fp16-blas-tune.jsonl`](../results/7900xtx-bf16-fp16-blas-tune.jsonl) | BF16 and FP16 GEMM sweep (`default`, `rocblas`, `hipblaslt`, `default+tunableop`, `default-again`) across 7 shapes |
| [`results/7900xtx-int8-convrot-layers.jsonl`](../results/7900xtx-int8-convrot-layers.jsonl) | Qwen-Image-2.1 `int8_linear` per-layer breakdown (shipped vs patched vs `rocBLAS`/`hipBLASLt` BF16) |
| [`results/7900xtx-qwen-image-blas-ab.jsonl`](../results/7900xtx-qwen-image-blas-ab.jsonl) | End-to-end ComfyUI Qwen-Image-2.1 5-arm BLAS A/B test |
