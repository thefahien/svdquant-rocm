# Nunchaku (SVDQuant W4A4) Kernel Inventory and Packed INT4 Layout Reference

Reference guide to the CUDA kernel inventory and on-disk packed INT4 tensor format in
[`nunchaku-ai/nunchaku`](https://github.com/nunchaku-ai/nunchaku) (commit `302e0e9`, 2026-09-06)
and how it maps to [`Comfy-Org/comfy-kitchen`](https://github.com/Comfy-Org/comfy-kitchen)'s
HIP `svdquant_w4a4.hip` backend on RDNA3 (`gfx1100`) and RDNA4 (`gfx1200`).

Companion docs in this repository:
- [`ROOFLINE.md`](ROOFLINE.md) — RX 7900 XTX (`gfx1100`) peak throughput and quantization roofline math.
- [`WMMA_LAYOUT.md`](WMMA_LAYOUT.md) — RDNA3 wave32 WMMA register fragment layout (`iu4`, `iu8`, `f16`).
- [`BENCHMARKS.md`](BENCHMARKS.md) — Measured `gfx1100` kernel and end-to-end ComfyUI benchmarks.

---

## 1. Architecture Overview: Why `comfy-kitchen` HIP Replaces Direct PTX Translation

1. **RDNA3 Hardware W4A4 Support:** `gfx1100` provides a native 4-bit integer WMMA instruction,
   `int_amdgcn_wmma_i32_16x16x16_iu4` (`v_wmma_i32_16x16x16_iu4`), with a boost peak of **246 TOPS**
   versus **123 TOPS** for INT8 and FP16/BF16 matrix operations (see [`ROOFLINE.md`](ROOFLINE.md) §1–2).
2. **Compute-Bound Regime at DiT Shapes:** At diffusion-transformer shapes (`M=4096, K=N=3072`),
   arithmetic intensity is 7,022 ops/byte (W4A4) vs a 256 ops/byte INT4 ridge on the RX 7900 XTX
   (960 GB/s DRAM), making the GEMM ~28× compute-bound down to `M ≈ 256`.
3. **Existing HIP Implementation in `comfy-kitchen`:** `Comfy-Org/comfy-kitchen` (Apache-2.0) ships
   a HIP backend (`comfy_kitchen/backends/hip/ops/svdquant_w4a4.hip`) implementing
   `quantize_svdquant_w4a4` and `scaled_mm_svdquant_w4a4` with fused rank-$R$ LoRA projections.
   With commit `08216d9` ([PR #262](https://github.com/Comfy-Org/comfy-kitchen/pull/262), closed
   unmerged on 2026-11-05 and included in [`../patches/0001-hip-svdquant-w4a4-perf.patch`](../patches/0001-hip-svdquant-w4a4-perf.patch)),
   the HIP W4A4 path reaches **123.0–138.7 INT4 TOPS** on `gfx1100` (50–56% of peak) and **73–84 TOPS**
   on `gfx1200`.
4. **Why `rocWMMA` Alone Cannot Run W4A4:** `rocWMMA` supports `int8` on `gfx11` but does not expose
   `int4` (`iu4`) fragments (`https://rocm.docs.amd.com/projects/rocWMMA/en/latest/api-reference/data-type-support.html`),
   so 4-bit WMMA on RDNA3/RDNA4 must call `__builtin_amdgcn_wmma_i32_16x16x16_iu4` directly.

---

## 2. Related ROCm / HIP Implementations

| Component | Description | RDNA3 (`gfx1100`) Support | Role in SVDQuant / Diffusion |
|---|---|---|---|
| `comfy-kitchen` HIP SVDQuant W4A4 (`backends/hip/ops/svdquant_w4a4.hip` + PR #262 `08216d9`) | HIP SVDQuant W4A4 dynamic quantizer + W4A4 GEMM + fused LoRA-down/up using raw WMMA builtins | **Run-verified on `gfx1100`** (this repo: 123–139 TOPS) and `gfx1200` (PR #262: 73–84 TOPS) | Primary W4A4 inference kernel for ComfyUI `TensorCoreSVDQuantW4A4Layout` |
| `comfy-kitchen` PR #99 (`w4a8` fused requantize) | Weight-requantization kernel (ConvRot codebook assignment + ALS + scales + stochastic rounding + int4 packing) | HIP port in upstream CI | Offline checkpoint conversion tool, not an inference GEMM |
| `comfy-kitchen` HIP INT8 WMMA GEMM (PR #259, merged) | `v_wmma_i32_16x16x16_iu8` GEMM + fused ConvRot / RMS / modulation / SwiGLU | **Run-verified on `gfx1100`** (this repo: 97.6–104.2 TOPS across Qwen-Image-2.1 layers) | Used by `TensorCoreInt8ConvRotLayout` checkpoints |
| `rocWMMA` 2.2+ | Official AMD C++ WMMA template library | Supports `f16`/`bf16` and `i8 -> i32` `16x16x16` on `gfx11`; **no `int4` (`iu4`) support** | Covers FP16/BF16/INT8, not W4A4 |
| Triton `tl.dot` | Python GPU kernel compiler (`https://triton-lang.org/main/python-api/generated/triton.language.dot.html`) | Supports `{int8, fp8_e5m2, fp16, bf16, fp32}` inputs; **no `int4` `tl.dot` input type** | Suitable for W4A8 prototypes, not native `iu4` W4A4 |

---

## 3. Upstream `nunchaku` CUDA Kernel Inventory (`nunchaku-ai/nunchaku` @ `302e0e9`)

All file and line references below refer to `https://github.com/nunchaku-ai/nunchaku` at commit
`302e0e97024ebd68688fe890e5df83731edf7b54` (~11.5k lines under `src/kernels/`).

### 3.1 Kernel Files

| File (`src/kernels/`) | Lines | Purpose | Used by Flux / Qwen-Image SVDQuant Inference? | CUDA / PTX Intrinsics |
|---|---|---|---|---|
| `zgemm/gemm_w4a4.cu` + `gemm_w4a4.cuh` | 162 + 1187 | W4A4 GEMM core (SVDQuant): tiled `m16n8k64` int4 MMA, fused LoRA/RoPE/norm/QKV-pack/SiLU epilogues, on-the-fly activation + weight quantizers | **Yes** — every quantized linear (`src/Linear.cpp:191-219`, `nunchaku/ops/gemm.py:11`) | `mma.sync` (via `mma.cuh`), `ldmatrix` (`gemm_w4a4.cuh:619`), `cvt.pack.sat.s4/u4.s32.b32`, `prmt`, `lop3`, `ld/st.shared` |
| `zgemm/gemm_w4a4_launch_*.cu` (5 files) + `gemm_w4a4_launch_impl.cuh` + `gemm_w4a4_launch.cuh` | 572 (impl) | Compile-time instantiations + host launch/dispatch of W4A4 GEMM and `quantize_w4a4_act_fuse_lora` | **Yes** | Instantiates device kernels above |
| `zgemm/mma.cuh` | 175 | Inline-PTX wrappers: `mma.sync.aligned.m16n8k16` (`f16`/`bf16`/`f32`) and `mma.sync.aligned.m16n8kx.row.col.s32.{s4,u4,s8,u8}` (`K=64` for 4-bit, `32` for 8-bit) | **Yes** | Raw `mma.sync` PTX |
| `zgemm/mma_earlycuda.cuh` | 227 | Fallback MMA for pre-Ampere (`sm_70`/`sm_75`): decomposed `mma.sync.m8n8*` sequences | Only when built for `sm < 80` | Raw `mma.sync` PTX |
| `zgemm/gemm_base.cuh` | 923 | GEMM framework: `GEMMConfig_W4A4`/`W8A8` tile config (`:34-70`), warp loaders (`load_act`/`load_wgt` `:265-292`), `fp16`↔`fp32` psum conversion, default epilogue | **Yes** | `ldmatrix` (`:640`), `ld.shared.v2/v4.b32`, `prmt.b32`, `lop3.b32` |
| `zgemm/gemm_utils.cuh` | 428 | Load/store helpers, `ldmatrix`/`movmatrix` wrappers (`:191-203`), activation quantizers `quantize_float2<4/8>` (`:206-252`), fast `int->half2` (`:384-426`) | **Yes** | `ldmatrix.sync.x4.m8n8`, `movmatrix.sync.m8n8.trans`, `cvt.pack.sat.s4/u4/s8.s32.b32`, `cvt.rn.satfinite.e2m1x2/e4m3x2`, `prmt`, `lop3`, `tanh.approx.f32`, `red.relaxed.gpu.global.add.f32` |
| `zgemm/epilogues.cuh` | 823 | Fused GEMM epilogues: bias, RMSNorm-Q/K + RoPE (`:53-160`), QKV split/pack, pooling, GLU (SiLU-gated) quantized output | **Yes** (`gemm_w4a4_launch_impl.cuh:271-294`) | `ldmatrix` (`:321`), `half2` SIMD |
| `zgemm/lora.cuh` | 356 | Fused LoRA (SVD low-rank) down/up projections folded into the W4A4 GEMM prologue/epilogue | **Yes** (`src/Linear.cpp:199-219`) | `half2` SIMD |
| `zgemm/attention.cuh` + `attention.cu` | 703 + 96 | Fused `fp16` attention (`attention_fp16`), `m16n8k16` `f32` accumulate | **Yes** for Flux (`src/FluxModel.cpp:371,631,1076`); **No** for Qwen-Image (uses PyTorch SDPA) | `ldmatrix`, `mma.sync.m16n8k16.f16`, `prmt` |
| `zgemm/gemm_w8a8.cu` + `.cuh` | 193 + 504 | W8A8 GEMM (`int8` weights/activations, `INSN_K=32`) | Non-4-bit builds only | `mma.sync.m16n8k32.s8`, `ldmatrix` |
| `awq/gemm_awq.cu` | 1369 | W4A16 GEMM (AWQ `int4` weights, `fp16` activations) for `M>1` | **Yes** (Qwen-Image `AWQW4A16Linear` and Flux modulation layers) | `cp.async`, `ldmatrix`, byte-permute `int4` unpack |
| `awq/gemv_awq.cu` | 294 | W4A16 GEMV (`M=1`) | **Yes** (small-`M` path of `AWQW4A16Linear.forward`) | `int4` unpack + warp shuffle reduction |
| `awq/dequantize.cuh` | 131 | AWQ `int4 -> half` dequantization helpers | **Yes** | `ldmatrix`-style unpack, `prmt` |
| `layernorm_kernels.cu` + `_impl.cuh` | 217 + 362 | LayerNorm, RMSNorm, fused residual-add + RMSNorm + requant | **Yes** | Warp-shuffle reductions |
| `activation_kernels.cu` + `_impl.cuh` | 106 | `silu_and_mul`, `dequant_silu_and_mul_quant`, GELU | Sana/SDXL MLP paths | Standard CUDA |
| `misc_kernels.cu` + `_impl.cuh` | 326 + 270 | `add`, `mul_add`, `split_mod`, `splitqkv`, `quant_kernel_static`, `cast`, `topk` | **Yes** | Standard CUDA |
| `gemm_f16.cu` | 135 | CUTLASS `fp16`/`bf16` dense GEMM for unquantized linears | **Yes** (fallback linears) | CUTLASS |

### 3.2 CUDA PTX to ROCm / `gfx1100` Mapping

| CUDA Primitive | Location in `nunchaku` | NVIDIA Feature | ROCm / `gfx1100` Equivalent |
|---|---|---|---|
| `mma.sync.aligned.m16n8k64.row.col.s32.{s4,u4,s4,u4}` | `mma.cuh:109-135` | Sub-byte `int4` Tensor Core (`16x8x64`) | Loop of four `16x16x16` `__builtin_amdgcn_wmma_i32_16x16x16_iu4` tiles over `K=64` |
| `mma.sync.aligned.m16n8k16.f16/bf16/f32` | `mma.cuh:36-108` | `fp16`/`bf16` Tensor Core | `__builtin_amdgcn_wmma_f32_16x16x16_f16` / `bf16` |
| `ldmatrix.sync.aligned.x4.m8n8.shared.b16` | `gemm_utils.cuh:191`, `gemm_w4a4.cuh:619` | Shared-to-register warp matrix load | Vectorized LDS `ds_read_b64` / `ds_read_b128` into per-lane WMMA layout |
| `cvt.pack.sat.s4/u4/s8.s32.b32` | `gemm_utils.cuh:215-235` | Sub-byte saturating pack | Bitwise shifts/masks + `__builtin_amdgcn_perm` |
| `prmt.b32`, `lop3.b32` | `gemm_utils.cuh:388-425` | Byte permute, 3-input LUT logic | `__builtin_amdgcn_perm`, bitwise boolean expressions |
| `cvt.rn.satfinite.e2m1x2/e4m3x2.f32` | `gemm_utils.cuh:239-252` | Blackwell `sm_120` NVFP4/FP8 convert | Not applicable on RDNA3 (`gfx1100` has INT4 `iu4` WMMA, not FP4/FP8 WMMA) |

---

## 4. SVDQuant Checkpoint Format & Packed INT4 Layout

Two quantized linear layer types appear in SVDQuant checkpoints:
- **`SVDQW4A4Linear`** — the SVDQuant W4A4 core (`qweight`, `wscales`, `smooth_factor`, `proj_down`, `proj_up`).
- **`AWQW4A16Linear`** — AWQ `int4` weights with `fp16`/`bf16` activations (used for `M=1` AdaLN modulation layers).

### 4.1 `SVDQW4A4Linear` Tensor Signatures (`nunchaku/models/linear.py:89-128`, `src/Linear.cpp:92-117`)

| Checkpoint Tensor Name | C++ Name (`Linear.cpp`) | Checkpoint Shape | Dtype | Padded C++ Shape | Description |
|---|---|---|---|---|---|
| `qweight` | `qweight` | `(N, K/2)` | `int8` (2×`int4`/byte) | `(N_pad, K_pad/2)` | Packed `int4` weights |
| `wscales` | `wscales` | `(K/G, N)` | `bf16`/`fp16` (`int4`) or `fp8_e4m3` (`nvfp4`) | `(K_pad/64, N_pad)` | Per-group weight scales (`G=64` for `int4`, `G=16` for `nvfp4`) |
| `smooth_factor` | `smooth` | `(K,)` | `bf16`/`fp16` | `(K_pad,)` | Per-channel smoothing factor absorbed during activation quantization |
| `smooth_factor_orig` | *(unused at runtime)* | `(K,)` | `bf16`/`fp16` | — | Pre-absorption smoothing factor |
| `proj_down` | `lora_down` | `(K, R)` | `bf16`/`fp16` | `(K_pad, R)` | Low-rank SVD down-projection (default rank `R = 32`) |
| `proj_up` | `lora_up` | `(N, R)` | `bf16`/`fp16` | `(N_pad, R)` | Low-rank SVD up-projection |
| `bias` | `bias` | `(N,)` | `bf16`/`fp16` | `(N_pad,)` | Optional bias vector |

* Dimension padding: `N_pad = ceil(N / 128) * 128`, `K_pad = ceil(K / 128) * 128`, and `M_pad = ceil(M / 256) * 256` (`BLOCK_M = 256`).
* `act_unsigned`: boolean layer attribute selecting unsigned (`u4`, `[0..15]`) vs signed (`s4`, `[-8..7]`) activation quantization (shifted by `+0.171875` after GELU in `nunchaku/ops/fused.py:43`).

### 4.2 `AWQW4A16Linear` Tensor Signatures (`nunchaku/models/linear.py:300-345`, `src/Linear.cpp:25-38`)

| Tensor | Shape | Dtype | Notes |
|---|---|---|---|
| `qweight` | `(N/4, K/2)` | `int32` | 8×`int4` per `int32` word |
| `wscales` | `(K/64, N)` | `bf16`/`fp16` | Per-group weight scales (`G=64`) |
| `wzeros` | `(K/64, N)` | `bf16`/`fp16` | Per-group zero points |
| `bias` | `(N,)` | `bf16`/`fp16` | Optional bias |
| `lora_down` | `(R, K)` | `bf16`/`fp16` | Transposed relative to `SVDQW4A4Linear` |
| `lora_up` | `(N, R)` | `bf16`/`fp16` | `(N, R)` |

### 4.3 Packed INT4 Layout of `qweight` (`SVDQW4A4Linear`)

In `nunchaku`, `qweight` is **not** stored in simple row-major `(N, K/2)` order. Its `(N, K/2)` `int8`
shape is a flat view over a buffer pre-permuted into the register fragments consumed by
`mma.sync.m16n8k64.row.col.s32.{s4,u4}`, matching the output of `quantize_w4a4_wgt_kernel`
(`src/kernels/zgemm/gemm_w4a4.cuh:662-689`).

Tiling constants (`GEMMConfig_W4A4`, `gemm_base.cuh:34-46, 112-116`):
`BLOCK_M = 256`, `BLOCK_N = 128`, `WARP_SIZE = 32`, `NUM_WARPS = 8`, `INSN_M = 16`, `INSN_N = 16`, `INSN_K = 64`,
`WARP_M = 32`, `WARP_N = 128`, `WARP_K = 64`, `WARP_M_TILES = 2`, `WARP_N_TILES = 8`.

1. **Sub-word Nibble Packing (`uint32`):**
   Eight `int4` values are packed into each `uint32` word, with element `j` (`j = 0..7`) in bits
   `[4*j, 4*j + 4)` (low nibble first, `gemm_utils.cuh:205-227`).
2. **Per-Tile Warp Permutation (`16 × 64` Weight Tile, `quantize_w4a4_warp`, `gemm_w4a4.cuh:535-623`):**
   - `PACK_SIZE = 8`, `NUM_PACKS_PER_ROW = 8`, `NUM_ROWS_PER_PACKWARP = 4`, `NUM_PACKWARPS = 4`.
   - Lane `L` (`0..31`) in pack slot `i` (`0..3`) loads 8 consecutive values from row
     `rowId = i * 4 + L / 8` and columns `colId = (L % 8) * 8 .. (L % 8) * 8 + 7`.
   - Across each 64-element row (8 lanes × 8 values), a warp reduction computes
     `scale = maxabs / 7.0`, emitting one scale per 64 K-elements (`K / 64` groups).
   - Each lane packs its 8 quantized `int4` values into a `uint32` word and writes to shared memory
     `uint32_t mat[16][8]`.
   - Lane `L` executes `ldmatrix.sync.aligned.x4.m8n8.shared.b16` from `&mat[L % 16][(L / 16) * 4]`,
     yielding a `uint4` (16 bytes = 32 `int4` nibbles) per lane.
   - **Weight-only word swap (`gemm_w4a4.cuh:682`):** For weights only, the middle 32-bit words of
     each lane's `uint4` are swapped (`std::swap(tmpout.y, tmpout.z)`) because the B-fragment of
     `m16n8k64` consumes `(wgt.x, wgt.y)` for the first `N=8` half and `(wgt.z, wgt.w)` for the
     second `N=8` half (`gemm_w4a4.cuh:409-417`). Activations do not swap `.y` and `.z`.
3. **Global On-Disk Tile Order (`gemm_base.cuh:278-292`):**
   `qweight` is laid out as `[N / 128, K / 64, 8 (WARP_N_TILES), 32 (WARP_SIZE)]` of `uint4`
   (16 bytes per `uint4`):
   ```
   uint4_index = ((n_blk * (K / 64) + k64) * 8 + n_tile) * 32 + lane
   byte_offset = uint4_index * 16
   ```
   where `n_blk = n_row // 128`, `n_tile = (n_row % 128) // 16`, and `k64 = k_col // 64`.
4. **Packed Scale Layout (`wscales` and `ascales`, `gemm_base.cuh:105-154`):**
   - `wscales` (`[K/64, N]`): packed with `WSCALES_PACK_SIZE = 4` (`fp16`/`bf16`), indexed per
     warp via `load_wmscale` (`gemm_w4a4.cuh:69-83`).
   - `ascales` (`[K/64, M_pad]`): packed with `ASCALES_PACK_SIZE = 2` (`fp16`/`bf16`), emitted by
     `pack_ascales` in the activation quantizer.
5. **How `comfy-kitchen` HIP (`08216d9`) Handles the Layout on RDNA3/RDNA4:**
   In `comfy_kitchen/backends/hip/ops/svdquant_w4a4.hip`, `unpack_nunchaku_frag` (lines 48–69)
   translates each lane's pre-permuted Nunchaku `uint4` (`ldmatrix` fragment) into two linear
   `uint32` words (16 `int4` nibbles along `K`) in shared memory (LDS), and each RDNA wave32 lane
   then loads its contiguous 64-bit (`int32x2`) `A_frag` and `B_frag` directly into
   `__builtin_amdgcn_wmma_i32_16x16x16_iu4`.
