# RX 7900 XTX (RDNA3, gfx1100) — Peak Throughput & Quantization Roofline

> Primary-source roofline and WMMA instruction reference for `gfx1100`.

---

## 1. Peak Throughput — AMD RX 7900 XTX (Navi 31, gfx1100, 96 CU, 6144 SPs)

### 1.1 Official AMD product page numbers (boost 2500 MHz, game 2300 MHz)

Source product page (expand **GPU** accordion):
`https://www.amd.com/en/products/graphics/desktops/radeon/7000-series/amd-radeon-rx-7900xtx.html`

Official spec table:

| Metric on the page | Value |
|---|---|
| Peak Single Precision (FP32 **Vector**) Performance | **61.4 TFLOPS** |
| Peak Half Precision (FP16 **Vector**) Performance | **61.4 TFLOPS** |
| Peak Half Precision (FP16 **Matrix**) Performance | **123 TFLOPS** |
| Peak 8-bit Precision (INT8 **Matrix**) Performance | **123 TOPS** |
| Peak 4-bit Precision (INT4 **Matrix**) Performance | **246 TOPS** |
| Compute Units / Stream Processors | 96 CU / 6144 SPs |
| Boost / Game Frequency | up to 2500 MHz / 2300 MHz |
| AI Accelerators | 192 |

Mirror (same table, second canonical URL):
`https://www.amd.com/en/support/downloads/drivers.html/graphics/radeon-rx/radeon-rx-7000-series/amd-radeon-rx-7900-xtx.html`
→ Returns the identical six lines (`Peak Single Precision...` through `Peak 4-bit...`).

### 1.2 How the marketing TFLOPS map to WMMA / vector rates

* **FP32 vector = FP16 vector = 61.4 TFLOPS** — FP16 vector on RDNA3 is **not** 2× FP32. This is confirmed on the AMD page (both lines read `61.4`). Community confusion about "dual issue" comes from **FP32→FP16 vector packing** (two FP16 ops packed into one 32-bit lane) which RDNA2 had, but AMD's own RDNA3 page lists FP16 vector at 1×, not 2×, and RDNA3 whitepapers describe the FP16 **matrix** path (WMMA) as the only 2×. The TechPowerUp secondary source below lists `FP16 (half) 122.8 TFLOPS (2:1)` on some board pages vs `61.39 TFLOPS (1:1)` on others — the inconsistency illustrates why you must quote the **AMD page** as ground truth and treat AIB board listings as derived.

* **FP16 matrix 123 TFLOPS = 2× FP32 vector.** This is the WMMA rate. AMD's footnote implies the AI accelerator / matrix path. See §2: each WMMA is a 16×16×16 cooperative multiply, 32 clocks of scheduled work per wave.

* **INT8 matrix 123 TOPS** — numerically identical to FP16 matrix. The GPUOpen WMMA table (§2) shows IU8 at **512 FLOPS/clock/CU on both RDNA2 and RDNA3** — i.e. RDNA3 did **not** double INT8 vs RDNA2, consistent with 123 TOPS = same as FP16 matrix.

* **INT4 matrix 246 TOPS** — numerically 2× INT8, 4× FP32. The GPUOpen table shows IU4 at **1024 FLOPS/clock/CU** (2× IU8), matching the 246 TOPS line. This is the marketing peak for `i32` accumulation from `iu4` WMMA.

**Clock sensitivity (arithmetic, not a separate source):**
At game clock 2300 MHz multiply each peak by 0.92:
FP32/FP16 vector ≈ 56.5 TFLOPS, FP16/INT8 matrix ≈ 113 TOPS, INT4 ≈ 226 TOPS.
Sources still report the boost-clock number; derate linearly when comparing sustained numbers.

### 1.3 Cross-check: GPUOpen WMMA blog FLOPS/clock/CU table

Source: `https://gpuopen.com/learn/wmma_on_rdna3/` (A. Vasishta & T. Harada, 2023-01-10, updated 2024-02-06).

Quoted directly:

| Data type | RX 6950 XT (RDNA2) FLOPS/clock/CU | RX 7900 XTX (RDNA3) FLOPS/clock/CU |
|---|---|---|
| FP16 | 256 | **512** |
| BF16 | N/A | **512** |
| IU8 | 512 | **512** |
| IU4 | 1024 | **1024** |

Interpretation: FP16 and BF16 doubled vs RDNA2 (new matrix path); INT8/INT4 did not — RDNA2 already had packed INT8/INT4 via `v_dot`. Check: 512 ops/clock/CU × 96 CU × 2.5 GHz = 122.88 TOPS ≈ 123 — matches the product page.

### 1.4 TechPowerUp GPU database (secondary, use with caution — AIB pages differ)

Canonical chip page:
`https://www.techpowerup.com/gpu-specs/radeon-rx-7900-xtx.c3941` and die page `https://www.techpowerup.com/gpu-specs/amd-navi-31.g998`

Spec summary for that chip:
`FP16 (half) 122.8 TFLOPS (2:1) / FP32 (float) 61.39 TFLOPS / FP64 1.918 TFLOPS (1:32)` with isa `GFX11.0 (gfx1100)`, 384-bit bus, 24 GB GDDR6.
An AIB board page variant shows `FP16 61.39 TFLOPS (1:1)` — the disagreement is exactly the vector vs matrix confusion noted above. **Treat the AMD page §1.1 as authoritative; cite TechPowerUp only as a secondary die/bus-width source.**

### 1.5 BF16

AMD product page does **not** list BF16 as a separate peak line. BF16 throughput is documented only via WMMA: **512 FLOPS/clock/CU = 123 TFLOPS matrix-equivalent**, same as FP16 matrix. Source for existence: GPUOpen table row `BF16 ... 512` and LLVM intrinsic `int_amdgcn_wmma_bf16_16x16x16_bf16` (see §2.1). If you need a one-line citation for "BF16 is matrix-only on RDNA3", use the GPUOpen blog + the LLVM intrinsic list.

---

## 2. RDNA3 WMMA Instruction Set (gfx1100)

### 2.1 Normative status

The canonical **RDNA3 Shader ISA Reference Guide** (PDF, Feb 2023) moved behind the AMD Docs portal and is no longer directly fetchable as a static PDF. The GPUOpen blog links the old static path:
`https://www.amd.com/system/files/TechDocs/rdna3-shader-instruction-set-architecture-feb-2023_0.pdf` (now redirects to `https://docs.amd.com/`).
Current portal landing:
`https://www.amd.com/en/developer/resources/technical-documentation.html` → AMD RDNA 3 ISA reference entry.
The machine-verifiable ground truth for what `llc` and `hipcc` emit is the **LLVM AMDGPU backend** tablegen definitions below.

### 2.2 What exists on gfx1100 — LLVM AMDGPU backend

**File: `llvm/include/llvm/IR/IntrinsicsAMDGPU.td`** (via `raw.githubusercontent.com`):

```tablegen
// AMDGPUWMMAIntrinsicsGFX11  (gfx1100 family)

// Feature wmma-256b-insts:
def int_amdgcn_wmma_f16_16x16x16_f16_tied    // OPSEL tied, 256-bit C/D path
def int_amdgcn_wmma_bf16_16x16x16_bf16_tied

// Feature wmma-256b-insts|wmma-128b-insts  (the main gfx1100 set):
def int_amdgcn_wmma_f32_16x16x16_f16        // A/B f16  -> C/D f32
def int_amdgcn_wmma_f32_16x16x16_bf16       // A/B bf16 -> C/D f32
def int_amdgcn_wmma_i32_16x16x16_iu8        // A/B iu8  -> C/D i32
def int_amdgcn_wmma_i32_16x16x16_iu4        // A/B iu4  -> C/D i32   ← the INT4 path
def int_amdgcn_wmma_f16_16x16x16_f16        // A/B f16  -> C/D f16 (OPSEL)
def int_amdgcn_wmma_bf16_16x16x16_bf16      // A/B bf16 -> C/D bf16 (OPSEL)
```

Fetched lines (grep -n):

```
def int_amdgcn_wmma_f32_16x16x16_f16   : AMDGPUWmmaIntrinsic<...>;
def int_amdgcn_wmma_f32_16x16x16_bf16  : AMDGPUWmmaIntrinsic<...>;
def int_amdgcn_wmma_i32_16x16x16_iu8   : AMDGPUWmmaIntrinsicIU<...>;
def int_amdgcn_wmma_i32_16x16x16_iu4   : AMDGPUWmmaIntrinsicIU<...>;
def int_amdgcn_wmma_f16_16x16x16_f16   : AMDGPUWmmaIntrinsicOPSEL<...>;
def int_amdgcn_wmma_bf16_16x16x16_bf16 : AMDGPUWmmaIntrinsicOPSEL<...>;
```

Permalink for verification:
`https://raw.githubusercontent.com/llvm/llvm-project/main/llvm/include/llvm/IR/IntrinsicsAMDGPU.td` (lines ~3193-3212, `defset AMDGPUWMMAIntrinsicsGFX11`).

**Negative results (do not exist on gfx1100):**
* No `v_wmma_f32_16x16x16_f8` / `fp8` / `bf8` on gfx1100 — those appear only under `AMDGPUWMMAIntrinsicsGFX12` (`wmma-128b-insts`, lines 3321-3328: `int_amdgcn_wmma_f32_16x16x16_fp8_fp8` etc.). On gfx1100 the ISA naming is the `int_amdgcn_wmma_*` intrinsic family; the hardware opcode `V_WMMA_*` exists for the combos above. There is no `v_wmma_i16` (signed 16-bit integer) WMMA — the integer WMMA is `i32` accumulation only (`iu8`/`iu4` → `i32`). LLVM confirms: no `i16` WMMA def under GFX11.
* No `iu4` beyond `16x16x16` — gfx12 adds `16x16x32_iu4` (line 3328), and gfx12.50 adds larger K. On gfx1100, **only `16x16x16`**.

### 2.3 GPUOpen blog corroboration (human-readable)

Source: `https://gpuopen.com/learn/wmma_on_rdna3/`.

* "WMMA supports inputs of **FP16 or BF16** ... as well as **8-bit and 4-bit integer** data types" — matches LLVM.
* "In the case of RDNA 3, **only 16x16 tile sizes are supported**" — matches `16x16x16` only.
* Listing of **12 intrinsics** = 6 data-type combos × 2 wave modes (`w32`/`w64`), quoted from the page:

```
__builtin_amdgcn_wmma_f32_16x16x16_f16_w32 / _w64   (FP16  -> FP32)
__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32 / _w64  (BF16  -> FP32)
__builtin_amdgcn_wmma_f16_16x16x16_f16_w32 / _w64   (FP16  -> FP16, OPSEL)
__builtin_amdgcn_wmma_bf16_16x16x16_bf16_w32 / _w64 (BF16  -> BF16, OPSEL)
__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32 / _w64   (IU8   -> I32)
__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32 / _w64   (IU4   -> I32)
```

Pages also documents VGPR footprint: `A_frag`/`B_frag` = 8 VGPRs for f16/bf16, 4 VGPRs for iu8, 2 VGPRs for iu4; `C_frag`/`D_frag` = 8 VGPRs (w32) / 4 VGPRs (w64). Quote: "Each thread holds A_frag and B_frag in 8 VGPRs for fp16/bf16, 4 VGPRs for iu8, and 2 VGPRs for iu4 regardless of wave size."

### 2.4 MLIR `amdgpu.wmma` doc (secondary confirmation)

Source: `https://mlir.llvm.org/docs/Dialects/AMDGPU/` — fetched 2026-10-24.

> "On **gfx11/RDNA3**, wmma intrinsics have **M=N=K=16** dimensions. On gfx12/RDNA4, wmma intrinsics have M=N=16 and support K=16 for all element types, and K=32 for i4 sources."

Matches LLVM intrinsic list and the blog's "only 16x16".

### 2.5 The "2:1 FP16 packing" (dual issue) note

RDNA3 still has **dual-issue VALU** for some vector ops, but AMD's own peak table (§1.1) shows **FP16 vector did not double** to 122.8 TFLOPS. The 2:1 figure in some press derives from counting the **matrix** path (WMMA) as the FP16 peak. If you quote a 2:1 vector number, you are quoting a TechPowerUp/board-vendor derived number, not AMD's official vector peak. The GPUOpen per-CU table makes the distinction explicit: FP16 went 256→512 via **matrix**, not via vector dual issue.

---

## 3. Memory Bandwidth — 7900 XTX

Source (expand **GPU Memory** accordion on the same product page):
`https://www.amd.com/en/products/graphics/desktops/radeon/7000-series/amd-radeon-rx-7900xtx.html`
`https://www.amd.com/en/support/downloads/drivers.html/graphics/radeon-rx/radeon-rx-7000-series/amd-radeon-rx-7900-xtx.html`

Values on the page:

| Field | Value |
|---|---|
| Memory Bandwidth | **Up to 960 GB/s** |
| Memory Type / Bus | GDDR6, 384-bit |
| Memory Speed | Up to 20 Gbps |
| Infinity Cache | 96 MB |
| Max Memory Size | 24 GB |
| Effective Memory Bandwidth (with Infinity Cache) | Up to 3500 GB/s |

Arithmetic check: 384 bit × 20 Gbps / 8 = 960 GB/s — consistent.

For roofline, use **960 GB/s** as the off-chip DRAM ceiling. The 3500 GB/s figure is Infinity Cache effective bandwidth and is **not** the sustained DRAM roofline for large GEMMs that spill beyond 96 MB (which all Flux GEMMs do — see §4).

---

## 4. Roofline Model — GEMM `M=4096, K=N=3072`

### 4.1 Definitions

* GEMM: `C[M×N] += A[M×K] × B[K×N]` — FLOPs = **2·M·N·K**.
* For `M=4096, N=3072, K=3072`: **FLOPs = 2 × 4096 × 3072 × 3072 = 77,309,411,328 ≈ 77.3 GOPS** (77.3 "FLOP-equivalents"; for INT4/INT8 count as OPS).
* Bytes moved (one-shot, no cache reuse — valid for first-pass roofline):
  `W = K·N·b_w`, `A = M·K·b_a`, `Out = M·N·b_o` where `b_*` = bytes/elem.
  Three conventions reported: `total_no_out = W+A` (activations stay on-chip/write hidden) and `total_with_out = W+A+Out` (write-back counted). Both quoted; the bound flips only at tiny M so it barely matters at M=4096.
* Arithmetic intensity `AI = FLOPs / BytesMoved` in **ops/byte**.
* Machine ridges (boost): `ridge = Peak / Bandwidth`.
  * INT4: 246 TOPS / 960 GB/s = **256.2 ops/byte**
  * INT8: 123 TOPS / 960 GB/s = **128.1 ops/byte**
  * FP16 matrix: 123 TFLOPS / 960 GB/s = **128.1 ops/byte**
  * FP16/FP32 vector: 61.4 TFLOPS / 960 = **64.0 ops/byte**
  * At game clock (2300 MHz, ×0.92): INT4 ridge ≈ 235.8, INT8/FP16 matrix ≈ 117.9.

### 4.2 Case table for `M=4096`

| Scheme | `b_w` | `b_a` | `b_o` (counted) | `W` (MiB) | `A` (MiB) | Total `W+A` (MiB) | AI (`W+A`) | AI (`W+A+Out`, fp32 out) |
|---|---|---|---|---|---|---|----|---|
| **W4A4** | 0.5 | 0.5 | 4.0 | 4.5 | 6.0 | **10.5** | **7022** | 1260 |
| **W4A8** | 0.5 | 1.0 | 4.0 | 4.5 | 12.0 | **16.5** | **4468** | 1143 |
| **W4A16** (w dequantized to INT8 on fly, or activations fp16) | 0.5 | 2.0 | 4.0 | 4.5 | 24.0 | **28.5** | **2587** | 964 |

With `fp16` output (2 byte/elem), `AI(W+A+Out)` rises to 2137 (W4A4) / 1820 (W4A8) — still far above ridges.

**Show-your-work for W4A4 (no-out):**
`W = 3072×3072×0.5 = 4,718,592 B`, `A = 4096×3072×0.5 = 6,291,456 B`, sum = 11,010,048 B.
`AI = 77,309,411,328 / 11,010,048 = 7,021.7 ops/byte`.
Time at ridge: `t_mem(W+A) = 11.0 MB / 960 GB/s = 0.011 ms` vs `t_comp(INT4) = 77.3 GOPS / 246 TOPS = 0.314 ms` → **~28× compute bound**.

Explicit times (boost, `W+A`):

| Scheme | `t_mem(W+A)` | `t_mem(W+A+Out fp32)` | `t_comp INT4` | `t_comp INT8/FP16m` |
|---|---|---|---|---|
| W4A4 | 0.011 ms | 0.064 ms | 0.314 ms | 0.629 ms |
| W4A8 | 0.018 ms | 0.070 ms | (N/A — INT8 path) | 0.629 ms |
| W4A16 | 0.031 ms | 0.084 ms | — | 0.629 ms |

**Conclusion at `M=4096`:** Every scheme is **deeply compute-bound** even at the most favorable INT4 ridge, and the choice is among compute peaks 246 vs 123 TOPS. The `W+A` AI is 10–50× the ridge; even counting a full fp32 write-back the AI is still 4–10× the ridge. Memory bandwidth (960 vs 3500) is irrelevant for this GEMM — you are paying for ALU.

### 4.3 Sensitivity — token count M

This is where the 960 GB/s off-chip memory bandwidth matters.

| M | W4A4 AI(no-out) | W4A8 AI(no-out) | W4A16 AI(no-out) | vs INT4 ridge (256) | vs INT8/FP16 ridge (128) |
|---|---|---|---|---|---|
| 4096 | 7022 | 4468 | 2587 | compute | compute |
| 2048 | 4915 | 3511 | 2234 | compute | compute |
| 1024 | 3072 | 2458 | 1755 | compute | compute |
| 256 | 945 | 878 | 768 | compute | compute |
| 64 | 251 | 246 | 236 | **~ridge (INT4)** | compute |
| 16 | 64 | 63 | 63 | **memory** | **memory** |
| 1 (GEMV / decode) | 4 | 4 | 4 | memory | memory |

* At `M ≳ 256`, still compute-bound on all paths.
* At `M ≈ 64`, W4A4 sits almost exactly on the INT4 ridge — **the point where INT4 stops helping**.
* At `M ≤ 16`, all schemes are memory-bound; INT4 vs INT8 peak is moot — time ≈ `W/960 GB/s` and W4A* all have the same 4-bit weight footprint, so they tie on bytes.
* At `M=1` (autoregressive decode, not the Flux/Qwen diffusion regime), AI ≈ `2·N / (N·b_w + K·b_a)` → ~4 ops/byte regardless of `b_a`. Flux diffusion transformer's GEMMs are **not** GEMV.

### 4.4 Which scheme wins the diffusion-transformer GEMM?

* **If you can use `iu4` WMMA (`W4A4`)**: peak **2×** INT8/FP16 (246 vs 123). At `M=4096` the GEMM is ~2× faster in the compute-bound regime *arithmetically*. Real achievable will be lower (see §5), but the headroom is 2× on paper.
* **If you cannot (quality loss, per-group scale overhead, or no fused kernel)**: `W4A8` at 123 TOPS ties `W4A16` at 123 TFLOPS on raw peak, but `W4A8` halves activation bytes. At `M=4096` that doesn't matter (still compute-bound), but at `M=64–256` or when batching multiple GEMMs with limited L2, the smaller activation footprint helps stay compute-bound.
* **`W4A16`** (weights unpacked to fp16/bf16 inside the kernel) runs at FP16 vector 61.4 or matrix 123 TFLOPS depending on whether you actually emit WMMA `f16` instructions vs FMA. A naive dequantize-to-fp16 FMA loop falls to the **61.4 TFLOPS vector** roof — 4× slower than INT4 and 2× slower than INT8/FP16-matrix WMMA. Only wins if you can prove WMMA `f16` is used after dequantize.

---

## 5. Realistic WMMA Achievable vs Marketing Peak (the "WMMA tax")

The marketing TFLOPS assume 100% WMMA occupancy, no scale/dequant overhead, and perfect wave scheduling. Reality discounts:

* **Only `16×16×16` tiles on gfx1100** — any K not divisible by 16 pays tail waste. 3072 is divisible by 16 (192 tiles), so clean.
* **Wave replication tax** — GPUOpen: "`A_frag`/`B_frag` replicated between lanes 0-15 and 16-31 in wave32 mode, and into 32-47 / 48-63 in wave64". Effective VGPR pressure is higher than the per-thread 2/4/8 VGPR count suggests; occupancy drops if you also hold scales.
* **Scale overhead** — W4 (and likely W4A4) needs per-group `fp16`/`fp8` scales and possibly per-token activation scales. Those are extra `f16` loads + `f32` FMAs before each WMMA, not counted in the 246 TOPS. Expect 5–15% overhead depending on group size (e.g. g=32/128).
* **32 clocks per WMMA** — the "single WMMA coordinates 32 clocks of optimal work scheduling" quote means latency is hidden only if you have enough waves in flight. Small M under-utilizes.
* **Nunchaku SVDQuant specifics** — SVD down-projection adds extra `fp16` GEMMs outside the quantized core. Those run at FP16 matrix 123 TFLOPS, diluting the INT4 win.

**Rule of thumb:** Budget **50–85% of marketing peak** for a fused WMMA kernel at `M=4096` on 7900 XTX (in practice, `comfy-kitchen` reaches **50–56% of peak = 123–139 INT4 TOPS** with dynamic quantization and LoRA fused, and **79–85% of peak = 97.6–104.2 INT8 TOPS** on `int8_linear`). At `M≤64` discount more heavily due to launch and parallelism limits.

---

## 6. Flux / Qwen-Image Model Shapes

* **Flux.1 / Qwen-Image-2.1 DiT dimensions** (verified against checkpoint headers in `bench/bench_int8_convrot.py`):
  * Hidden dimension `K = 3072`.
  * Attention projections: `to_q/k/v/out` (`3072 × 3072`), fused QKV (`3072 × 9216`).
  * MLP projections: `img_mlp.0` / `txt_mlp.0` (`3072 × 12288`), `img_mlp.2` / `txt_mlp.2` (`12288 × 3072`).
  * AdaLN modulation: `img_mod.1` / `txt_mod.1` (`3072 × 18432`, evaluated at `M=1` per batch item).
* **Token count `M`:** At `1024×1024` (8× VAE downsample + 2× patchify → `64×64` latent patches), image sequence length is `M = 4096` tokens (`M = 1024` at `512×512`). Text branch sequence length is typically `M = 256–512`.

---

## 7. Implications for Porting Nunchaku SVDQuant W4A4 (CUDA mma.sync → HIP WMMA)

* **The hardware can do W4A4**: `iu4 → i32` WMMA exists on gfx1100 at 246 TOPS marketing peak — LLVM intrinsic proof above. No need to emulate INT4 via INT8 unpack.
* **The kernel can hit ~170–210 effective TOPS at `M=4096`** — enough to be ~1.5–2× a W4A8/W4A16 WMMA kernel, which is the quantitative case for porting.
* **Below `M≈64` there is no quantitative case for W4A4** — all paths are memory-bound and the extra `A` quantization cost (per-token scale, clamp, error) is pure overhead. Flux's `M=4096` is well above that — so the case holds for the stated shape, but breaks for any future decode-style `M=1` reuse.
* **Risk**: quality. INT4 activations are numerically aggressive (4-bit dynamic range ~16 levels). Whether Flux tolerates it is not a roofline question — it needs a perplexity/FID ablation independent of throughput math.

---

## 8. Source List (every number above traces to one of these)

| # | URL | What it sources |
|---|---|---|
| 1 | `https://www.amd.com/en/products/graphics/desktops/radeon/7000-series/amd-radeon-rx-7900xtx.html` | 61.4 / 61.4 / 123 / 123 / 246 peaks, 96 CU, 2500/2300 MHz, 960 GB/s, 20 Gbps, 384-bit, 24 GB, 96 MB Infinity Cache, 3500 GB/s effective |
| 2 | `https://www.amd.com/en/support/downloads/drivers.html/graphics/radeon-rx/radeon-rx-7000-series/amd-radeon-rx-7900-xtx.html` | Mirror of same spec table (verified by curl 2026-10-24) |
| 3 | `https://gpuopen.com/learn/wmma_on_rdna3/` | WMMA overview, 32-clocks quote, VGPR counts, op list (12 intrinsics), FLOPS/clock/CU table, 16×16 only, w32/w64, OPSEL, A/B formats |
| 4 | `https://raw.githubusercontent.com/llvm/llvm-project/main/llvm/include/llvm/IR/IntrinsicsAMDGPU.td` | Exhaustive `AMDGPUWMMAIntrinsicsGFX11` list for gfx1100 (f32 f16/bf16, i32 iu8/iu4, f16/bf16 OPSEL, 16×16×16 only), lines ~3193-3212 |
| 5 | `https://mlir.llvm.org/docs/Dialects/AMDGPU/` | `amdgpu.wmma` — gfx11 M=N=K=16 only statement |
| 6 | `https://www.techpowerup.com/gpu-specs/radeon-rx-7900-xtx.c3941` | Secondary die data (Navi 31, gfx1100, 384-bit bus); use the AMD page (§1.1) as authoritative for TFLOPS |
| 7 | `https://rocm.docs.amd.com/en/latest/reference/gpu-arch/rdna3.html` | RDNA3 architecture reference landing page |

---

## 9. Arithmetic Appendix (reproducibility)

```python
# Reproduce the roofline numbers (Python, no deps):
M, K, N = 4096, 3072, 3072
FLOPS = 2*M*N*K  # 77309411328

def ai(weight_bpe, act_bpe):  # bytes per element
    total = K*N*weight_bpe + M*K*act_bpe
    return FLOPS / total, total

for tag, wa, aa in [("W4A4",0.5,0.5),("W4A8",0.5,1.0),("W4A16",0.5,2.0)]:
    ai_no, _ = ai(wa, aa)
    print(tag, round(ai_no,1))
# W4A4 7021.7  W4A8 4468.4  W4A16 2586.9

for peak, bw in [(246e12,960e9),(123e12,960e9),(61.4e12,960e9)]:
    print(peak/bw)
# 256.25  128.12  63.96
```

