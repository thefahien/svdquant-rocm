# Patches

## 0001-hip-svdquant-w4a4-perf.patch

[`0001-hip-svdquant-w4a4-perf.patch`](0001-hip-svdquant-w4a4-perf.patch)

From [Comfy-Org/comfy-kitchen#262](https://github.com/Comfy-Org/comfy-kitchen/pull/262) (`08216d9`, by
`0xDELUXA`, Apache-2.0). The PR was tested on `gfx1200` and closed unmerged on 2026-11-05; upstream
`comfy-kitchen` (through `v0.2.41`) still ships the initial `ec72ea1` HIP SVDQuant W4A4 kernel, which runs at
15.3–19.4 TOPS on an RX 7900 XTX (`gfx1100`) — 4–7x slower than BF16 matmul.

Building with this patch for `COMFY_HIP_ARCHS=gfx1100` raises `gfx1100` W4A4 GEMM throughput to **123–139 INT4
TOPS** (50–56% of the 246 TOPS hardware peak):

1. **WMMA LoRA-down (`MmaBf16::load` + `svdquant_lora_down`):** runs the low-rank `proj_down` branch on the
   RDNA3 WMMA GEMM core (`v_wmma_f32_16x16x16_bf16`) instead of falling back to an untiled path.
2. **Wave-per-row quantizer (`quantize_svdquant_w4a4`):** processes 8 rows per block (one wave32 per row, two
   elements per lane), matching the CUDA quantizer layout.
3. **LDS scale and LoRA-up staging (`scaled_mm_svdquant_w4a4`):** stages per-tile activation/weight scales (`fp32`)
   and rank-chunked LoRA-up operands in LDS alongside the INT4 tiles (`v_wmma_i32_16x16x16_iu4`), removing
   global-memory stalls during per-group scale flushes.

## License

`0001-hip-svdquant-w4a4-perf.patch` is an unmodified `git format-patch` export of commit `08216d9c25fe932fc612cb4e7d148daa8efabd08` by `0xDELUXA` from `Comfy-Org/comfy-kitchen` under the [Apache-2.0 license](../LICENSES/Apache-2.0.txt).
