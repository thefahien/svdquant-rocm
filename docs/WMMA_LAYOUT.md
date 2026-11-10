# RDNA3 (gfx11, wave32) WMMA Fragment Layout — Primary-Source Notes

> **Purpose:** Exact per-lane register/fragment mapping for `v_wmma_i32_16x16x16_iu4`,
> `v_wmma_i32_16x16x16_iu8`, and `v_wmma_f16_16x16x16_f16` on RDNA3 `gfx1100`
> (RX 7900 XTX), wave32.

## 0. Sources (authority order)

| # | Source | File / URL | What was extracted |
|---|--------|------------|--------------------|
| 1 | **GPUOpen blog** “How to accelerate AI applications on RDNA 3 using WMMA” | <https://gpuopen.com/learn/wmma_on_rdna3/> | Fragment VGPR counts, lane replication rule, row-major/column-major convention, OPSEL semantics, `lane = tid % 16` idiom |
| 2 | **Machine-readable ISA** (GPUOpen MR-ISA bundle 2026-08-06) | `amdgpu_isa_rdna3.xml` · <https://gpuopen.com/machine-readable-isa/> → `AMD_GPU_MR_ISA_XML_2026_08_06.zip` | Operand `OperandSize`, `FMT_WMMA_*` BitCounts, DataFormat descriptions; VOP3P opcodes |
| 3 | **AMD Matrix Instruction Calculator** (`ROCm/amd_matrix_instruction_calculator`) | `matrix_calculator.py` · <https://github.com/ROCm/amd_matrix_instruction_calculator> | Per-lane GPR/bit→(row,col) formulas for every WMMA variant, including lane replication |
| 4 | **LLVM AMDGPU backend** | `VOP3PInstructions.td`, `IntrinsicsAMDGPU.td`, `gfx11_asm_wmma.s` · <https://github.com/llvm/llvm-project> | VOP profiles, intrinsic signatures, assembly operand layout, `clamp`/`neg_lo`/`op_sel` modifiers |
| 5 | **rocWMMA** | `wmma_impl.hpp`, `io_traits.hpp`, `mapping_util.hpp` · <https://github.com/ROCm/rocWMMA> | `VRegI32x2`/`AccRegI32x8` type aliases confirming VGPR counts; `__builtin_amdgcn_wmma_*` call sites |

> **RDNA3 ISA PDF** (`RDNA3_Shader_ISA_December2022.pdf`, referenced by the blog) is
> behind `docs.amd.com` authentication; the machine-readable XML (`amdgpu_isa_rdna3.xml`)
> is the canonical machine-verifiable replacement.

---

## 1. Instruction overview (what WMMA does)

```
D[M][N] = A[M][K] * B[K][N] + C[M][N]    // M=N=K=16, fused multiply-add
```

- `A` is **M×K** (16×16), `B` is **K×N** (16×16), `C`/`D` are **M×N** (16×16).
- The GFX11 table in `VOP3PInstructions.td` lists exactly six 16×16×16 variants (wave32 *and* wave64 forms):
  - `v_wmma_f32_16x16x16_f16` / `v_wmma_f32_16x16x16_bf16` (f32 accumulate)
  - `v_wmma_f16_16x16x16_f16` / `v_wmma_bf16_16x16x16_bf16` (f16/bf16 accumulate, OPSEL)
  - `v_wmma_i32_16x16x16_iu8` / `v_wmma_i32_16x16x16_iu4` (i32 accumulate, clamp, signedness via NEG)
- VOP3P encodings (from ISA XML `amdgpu_isa_rdna3.xml` and `matrix_calculator.py --detail`):
  - `v_wmma_f32_16x16x16_f16` opcode 0x40, `bf16` 0x41, `f16` 0x42, `bf16` 0x43, `iu8` 0x44, `iu4` 0x45 (ENC_VOP3P).

> **Quote — ISA XML** (`V_WMMA_I32_16X16X16_IU4`):
> “Multiply the 16×16 matrix in the first input by the 16×16 matrix in the second
> input and add the 16×16 matrix in the third input using fused multiply add.
> Store the resulting matrix into vector registers.”

---
## 2. Register counts per lane (VGPRs, wave32)

All counts are **per lane** (per thread) in wave32; each VGPR is 32 bits.
Counts are confirmed by three independent sources: the GPUOpen blog, the
calculator `--detail-instruction`, and the ISA XML `OperandSize` /
`FMT_WMMA_*` `BitCount` fields.

| Variant | A (SRC0) | B (SRC1) | C (SRC2) | D (VDST) | Source |
|---------|--------|--------|----------|----------|--------|
| **iu4** (`V_WMMA_I32_16X16X16_IU4`) | **2 VGPRs** (64 b) — `v0:1` | **2 VGPRs** (64 b) — `v2:3` | **8 VGPRs** (256 b) | **8 VGPRs** (256 b) | blog + calc + ISA XML + LLVM |
| **iu8** (`V_WMMA_I32_16X16X16_IU8`) | **4 VGPRs** (128 b) — `v0:3` | **4 VGPRs** (128 b) — `v4:7` | **8 VGPRs** (256 b) | **8 VGPRs** (256 b) | calc + ISA XML + LLVM |
| **f16→f16** (`V_WMMA_F16_16X16X16_F16`) | **8 VGPRs** (256 b) — `v0:7` | **8 VGPRs** (256 b) — `v8:15` | **8 VGPRs** (256 b, only low *or* high half used) | **8 VGPRs** (256 b, same) | blog + calc + ISA XML + LLVM |
| **f16→f32** (`V_WMMA_F32_16X16X16_F16`) | **8 VGPRs** (256 b) | **8 VGPRs** (256 b) | **8 VGPRs** (256 b) | **8 VGPRs** (256 b) | blog + calc + ISA XML + LLVM |
| **bf16→bf16** | 8 | 8 | 8 | 8 | calc + LLVM |
| **bf16→f32** | 8 | 8 | 8 | 8 | calc + LLVM |

Wave64 for comparison (same A/B VGPRs; C/D halves): iu4 C/D **4 VGPRs**, iu8 C/D **4 VGPRs**,
f16→f16 C/D **4 VGPRs** (but still 8 VGPRs of storage with half-wave packing), f32 C/D **4 VGPRs**.
The calculator `--detail-instruction` output explicitly states this:

```
# iu4 wave32 (excerpt from calculator --detail-instruction):
    Wave32 register usage:
        GPRs required for A: 2
        GPRs required for B: 2
        GPRs required for C: 8
        GPRs required for D: 8
    Wave64 register usage:
        GPRs required for A: 2
        GPRs required for B: 2
        GPRs required for C: 4
        GPRs required for D: 4
    Wave32 matrix element to register mapping with no modifiers:
    Wave64 matrix element to register mapping with no modifiers:
    Wave32 register to matrix element mapping with no modifiers:
    Wave64 register to matrix element mapping with no modifiers:
```

```
# iu8 wave32:
    Wave32 register usage:
        GPRs required for A: 4
        GPRs required for B: 4
        GPRs required for C: 8
        GPRs required for D: 8
    Wave64 register usage:
        GPRs required for A: 4
        GPRs required for B: 4
        GPRs required for C: 4
        GPRs required for D: 4
    Wave32 matrix element to register mapping with no modifiers:
    Wave64 matrix element to register mapping with no modifiers:
    Wave32 register to matrix element mapping with no modifiers:
    Wave64 register to matrix element mapping with no modifiers:
```

```
# f16->f16 wave32:
    Wave32 register usage:
        GPRs required for A: 8
        GPRs required for B: 8
        GPRs required for C: 8
        GPRs required for D: 8
    Wave64 register usage:
        GPRs required for A: 8
        GPRs required for B: 8
        GPRs required for C: 4
        GPRs required for D: 4
    Wave32 matrix element to register mapping with no modifiers:
    Wave64 matrix element to register mapping with no modifiers:
    Wave32 register to matrix element mapping with no modifiers:
    Wave64 register to matrix element mapping with no modifiers:
```

> **Quote — GPUOpen blog** (Loading elements into matrix fragments):
> “From the perspective of a single lane (thread) within a wave, each “fragment”
> is locally stored in VGPRs, with each VGPR being 32 bits wide. Each thread
> holds `A_frag` and `B_frag` in 8 VGPRs for fp16/bf16, 4 VGPRs for iu8, and
> 2 VGPRs for iu4 regardless of the wave size.”
> “`C_frag` and `D_frag` requires 8 VGPRs in wave32 mode and 4 VGPRs in
> wave64 mode, regardless of the datatype used by matrices C and D.”

> **Quote — ISA XML** (`FMT_WMMA_AB_16X16_IU4`):
> “WMMA AB matrices are 16×16 and the individual elements are packed; each
> VGPR lane can contain 2 16-bit elements, 4 8-bit elements or 8 4-bit
> elements. OPSEL is ignored for AB matrices. WMMA requires the AB matrix be
> replicated; 2 copies ... for wave32 mode and 4 copies ... for wave64 mode.”

> **ISA XML operand sizes (wave32, per lane — `OperandSize` is total bits across the wave’s VGPRs for that operand, but the lane VGPR count is OperandSize/32 per 32 lanes):**
> - `V_WMMA_I32_16X16X16_IU4`: VDST 256 b (8 VGPRs), SRC0 64 b (2 VGPRs), SRC1 64 b (2 VGPRs), SRC2 256 b (8 VGPRs)
> - `V_WMMA_I32_16X16X16_IU8`: VDST 256, SRC0 128 (4 VGPRs), SRC1 128, SRC2 256
> - `V_WMMA_F16_16X16X16_F16`: VDST 256, SRC0 256 (8 VGPRs), SRC1 256, SRC2 256
> (`FMT_WMMA_AB_16X16_IU4` BitCount 64 = 2 VGPRs; `FMT_WMMA_AB_16X16_IU8` BitCount 128 = 4 VGPRs;
> `FMT_WMMA_AB_16X16_F16` BitCount 256 = 8 VGPRs; all `FMT_WMMA_DC_*` BitCount 256 = 8 VGPRs)

> **LLVM `VOP3PInstructions.td`** confirms the same via VOP profiles:
> - `V_WMMA_I32_16X16X16_IU4` → `VOP_V8I32_V2I32_V2I32_V8I32` = `[v8i32, v2i32, v2i32, v8i32]` → 8×i32 D, 2×i32 A, 2×i32 B, 8×i32 C
> - `V_WMMA_I32_16X16X16_IU8` → `VOP_V8I32_V4I32_V4I32_V8I32` → 8×i32 D, 4×i32 A, 4×i32 B, 8×i32 C
> - `V_WMMA_F16_16X16X16_F16` → `VOP_V16F16_V16F16_V16F16_V16F16` → 16×f16 D, 16×f16 A, 16×f16 B, 16×f16 C (8 VGPRs of f16 packed 2 per VGPR)

> **rocWMMA `wmma_impl.hpp`** (gfx11): `VRegI32x2` (2 VGPRs) for iu8 K=8 stub,
> `VRegI32x4` (4 VGPRs) for iu8 K=16; `AccRegI32x8` (8 VGPRs) for C/D — matches.
> For gfx11 iu4 the calculator is authoritative; rocWMMA gfx11 iu4 path is
> **UNCONFIRMED** in `wmma_impl.hpp` (no explicit `int4_t` specialization was found
> for gfx11; the builtin `__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32` is still the
> hardware truth per LLVM).

---

## 2. Register counts per lane (VGPRs, wave32)

All counts are **per lane** (per thread) in wave32; each VGPR is 32 bits.
Confirmed by GPUOpen blog + calculator + ISA XML + LLVM.

| Variant | A (SRC0) | B (SRC1) | C (SRC2) | D (VDST) |
|---------|--------|--------|----------|----------|
| **iu4** (`V_WMMA_I32_16X16X16_IU4`) | **2 VGPRs** (64 b) — `v0:1` | **2 VGPRs** (64 b) — `v2:3` | **8 VGPRs** (256 b) | **8 VGPRs** (256 b) |
| **iu8** (`V_WMMA_I32_16X16X16_IU8`) | **4 VGPRs** (128 b) — `v0:3` | **4 VGPRs** (128 b) — `v4:7` | **8 VGPRs** (256 b) | **8 VGPRs** (256 b) |
| **f16→f16** (`V_WMMA_F16_16X16X16_F16`) | **8 VGPRs** (256 b) — `v0:7` | **8 VGPRs** (256 b) — `v8:15` | **8 VGPRs** (256 b, only low *or* high half used) | **8 VGPRs** |
| **f16→f32** (`V_WMMA_F32_16X16X16_F16`) | **8 VGPRs** | **8 VGPRs** | **8 VGPRs** | **8 VGPRs** |

Wave64: same A/B VGPRs; C/D halves: iu4 C/D **4 VGPRs**, iu8 C/D **4 VGPRs**, f16→f16 C/D **4 VGPRs**.

```
# iu4 wave32 GPR usage (calculator --detail-instruction):
    Wave32 register usage:
        GPRs required for A: 2
        GPRs required for B: 2
        GPRs required for C: 8
        GPRs required for D: 8
    Wave64 register usage:
        GPRs required for A: 2
        GPRs required for B: 2
        GPRs required for C: 4
        GPRs required for D: 4
    Wave32 matrix element to register mapping with no modifiers:
    Wave64 matrix element to register mapping with no modifiers:
    Wave32 register to matrix element mapping with no modifiers:
    Wave64 register to matrix element mapping with no modifiers:
```

```
# iu8 wave32 GPR usage:
    Wave32 register usage:
        GPRs required for A: 4
        GPRs required for B: 4
        GPRs required for C: 8
        GPRs required for D: 8
    Wave64 register usage:
        GPRs required for A: 4
        GPRs required for B: 4
        GPRs required for C: 4
        GPRs required for D: 4
    Wave32 matrix element to register mapping with no modifiers:
    Wave64 matrix element to register mapping with no modifiers:
    Wave32 register to matrix element mapping with no modifiers:
    Wave64 register to matrix element mapping with no modifiers:
```

```
# f16->f16 wave32 GPR usage:
    Wave32 register usage:
        GPRs required for A: 8
        GPRs required for B: 8
        GPRs required for C: 8
        GPRs required for D: 8
    Wave64 register usage:
        GPRs required for A: 8
        GPRs required for B: 8
        GPRs required for C: 4
        GPRs required for D: 4
    Wave32 matrix element to register mapping with no modifiers:
    Wave64 matrix element to register mapping with no modifiers:
    Wave32 register to matrix element mapping with no modifiers:
    Wave64 register to matrix element mapping with no modifiers:
```

> **Quote — GPUOpen blog** (Loading elements into matrix fragments):
> “From the perspective of a single lane (thread) within a wave, each \u201cfragment\u201d
> is locally stored in VGPRs, with each VGPR being 32 bits wide. Each thread
> holds `A_frag` and `B_frag` in 8 VGPRs for fp16/bf16, 4 VGPRs for iu8, and
> 2 VGPRs for iu4 regardless of the wave size.”
> “`C_frag` and `D_frag` requires 8 VGPRs in wave32 mode and 4 VGPRs in
> wave64 mode, regardless of the datatype used by matrices C and D.”

> **ISA XML** `FMT_WMMA_AB_16X16_IU4` BitCount 64 = 2 VGPRs; `FMT_WMMA_AB_16X16_IU8` BitCount 128 = 4 VGPRs;
> `FMT_WMMA_AB_16X16_F16` BitCount 256 = 8 VGPRs; all `FMT_WMMA_DC_*` BitCount 256 = 8 VGPRs.

> **LLVM `VOP3PInstructions.td`:** `V_WMMA_I32_16X16X16_IU4` → `VOP_V8I32_V2I32_V2I32_V8I32` = `[v8i32, v2i32, v2i32, v8i32]`;
> `V_WMMA_I32_16X16X16_IU8` → `VOP_V8I32_V4I32_V4I32_V8I32`; `V_WMMA_F16_16X16X16_F16` → `VOP_V16F16_V16F16_V16F16_V16F16`.

---

## 3. Lane replication (wave32 = 2 copies, wave64 = 4 copies)

WMMA on RDNA3 **requires** that the 16×16 A/B data be replicated across
half-waves. The C/D data is **not** replicated (each lane holds distinct output).

> **Quote — GPUOpen blog:**
> “WMMA on RDNA 3 requires that the contents of `A_frag` and `B_frag` are
> replicated between lanes 0-15 and lanes 16-31 of the wave in wave32 mode.
> This means that for wave32 mode, each VGPR in lane 0 must have the exact
> same matrix data as each VGPR in lane 16. It is similar for lane 1 into
> lane 17, and so on, all the way to lane 15 into lane 31. This effectively
> maintains two copies of matrix data between the two half-waves. In wave64
> mode, data from lanes 0-15 must also be replicated into lanes 32-47 and 48-63.”

> **Quote — ISA XML** (`FMT_WMMA_AB_16X16_*`):
> “WMMA requires the AB matrix be replicated; 2 copies of the matrix are
> required for wave32 mode and 4 copies are required for wave64 mode. The
> total number of VGPRs is the same regardless of wave size.”

> **Quote — calculator** (`--detail-instruction`, wave32 mapping):
> `A[i][k] Lane: i and i+16` (wave32); `A[i][k] Lane: i, i+16, i+32, and i+48` (wave64)
> `B[k][j] Lane: j and j+16` (wave32)

**Emulator consequence:** For correctness the math is identical with or without
replication (both copies contribute the same product), but a bit-accurate
emulator that wants to catch *packing bugs where the two halves diverge* should
assert `a_frag[lane] == a_frag[lane+16]` (and same for `b_frag`) on entry, or
mask to `lane % 16` as the blog’s example does:

```c
// from the blog’s f16 example — lane is 0..31 mod 16
const int lane = lIdx % 16;
for (int ele = 0; ele < 16; ++ele) {
    b_frag[ele] = b[16*ele + lane];   // B is row-major: B[k][lane]
    a_frag[ele] = a[16*lane + ele];   // A is column-major: A[lane][k]
}
```

---

## 4. Matrix memory layout (row-major vs column-major)

| Matrix | Layout | Meaning |
|--------|--------|---------|
| **A** (M×K) | **column-major** | `A[i][k]` at `a[16*lane + k]` in the blog’s load loop; the figures label A cells as `(i,j)` = `(row, col)` with rows along lanes |
| **B** (K×N) | **row-major** | `B[k][j]` at `b[16*k + lane]`; figures label B cells as `(i,j)` = `(col, row)` — i.e. transposed header |
| **C/D** (M×N) | **row-major** | `C[i][j]` at `c[16*i + j]`; figures show C/D row-major |

> **Quote — GPUOpen blog:**
> “WMMA requires a combination of row-major and column-major inputs for
> matrices A, B, C, and D. Matrix A is stored in column-major order, whereas
> matrices B, C, and D are all stored in row-major order.”

In the f16 input figure (`f16_16x16x16_f16_w32_input.png`) this is visible:
- **A (Column Major)** panel: cell `(i,j)` is `A[i-1][j-1]` with j=K. Each lane holds one **row** of A (16 K values).
- **B (Row Major)** panel: cell `(i,j)` is `B[j-1][i-1]` — the header is transposed, so each lane holds one **column** of B.

---

## 5. Per-lane fragment mapping — the exact formulas (wave32)

All formulas below are **wave32, no modifiers** (no OPSEL/NEG/clamp), as
reported by `matrix_calculator.py --detail-instruction` on `gfx1100` and
cross-checked against the ISA XML `FMT_WMMA_*` BitCount and the GPUOpen
figures. Nibble/byte order inside a VGPR is little-endian (low bits = low K).

### 5.1  iu4 — `v_wmma_i32_16x16x16_iu4` (D = i32, A/B = iu4 packed 8 per VGPR)

**VGPRs per lane:** A 2 (`v0:1`), B 2 (`v2:3` if we name them contiguously; the
ISA shows `VRegSrc_64` = 64 bits = 2 VGPRs), C/D 8 (`v4:11` or `v0:7` depending on
allocation; LLVM shows `v[4:11]` for w32).

**Packing (iu4):** Each VGPR holds **8 int4** values, 4 bits each. K=0 is in
bits `[3:0]`, K=1 in `[7:4]`, …, K=7 in `[31:28]` of VGPR 0; K=8 in `[3:0]` of
VGPR 1, …, K=15 in `[31:28]` of VGPR 1. The `FMT_WMMA_AB_16X16_IU4` BitMaps
confirm: Order 0→BitOffset 0, Order 1→4, …, Order 15→60 (4-bit steps).

```
Matrix-element → register (wave32, no modifiers):
  A[i][k]  GPR:  floor(k / 8).[4*(k%8)+3 : 4*(k%8)]    Lane: i and i+16
  B[k][j]  GPR:  floor(k / 8).[4*(k%8)+3 : 4*(k%8)]    Lane: j and j+16
  C/D[i][j] GPR: floor(i / 2)                         Lane: ((16*i) % 32) + j
                                                      // = (i%2==0 ? j : j+16) + 16*(i//2 is encoded in GPR)
```

```
Register → matrix-element (wave32, inverse):
  A  i: (lane % 16)
     k: 8*GPR_num + floor(GPR_bits / 4)
  B  j: (lane % 16)
     k: 8*GPR_num + floor(GPR_bits / 4)
  C/D i: 2*GPR_num + floor(lane / 16)
      j: (lane % 16)
```

**In plain English:**
- **A:** lane `L = i % 16` (with duplicate `L+16`) holds the entire **row** `i` of A.
  Within that lane, VGPR0 holds K=0..7 (nibble 0 = K0), VGPR1 holds K=8..15.
  So `A[5][9]` is in lane 5 (and 21), VGPR1 bits `[7:4]` (K=9 → second nibble of VGPR1).
- **B:** lane `L = j % 16` (with duplicate `L+16`) holds the entire **column** `j` of B.
  Same VGPR packing: `B[9][5]` is in lane 5 (and 21), VGPR1 bits `[7:4]` (K=9).
- **C/D:** each lane holds **8 distinct** i32 accumulators (one per VGPR, unpacked).
  VGPR `v` holds two rows: `v` even holds rows `2*v` and `2*v+1` split across the two half-waves.
  Explicitly (checked against `--matrix-layout`):

```
lane  0: v0=C[0][0]  v1=C[2][0]  v2=C[4][0]  v3=C[6][0]  v4=C[8][0]  v5=C[10][0]  v6=C[12][0]  v7=C[14][0]
lane  1: v0=C[0][1]  v1=C[2][1]  ...
...
lane 15: v0=C[0][15] ... v7=C[14][15]
lane 16: v0=C[1][0]  v1=C[3][0]  v2=C[5][0]  v3=C[7][0]  v4=C[9][0]  v5=C[11][0]  v6=C[13][0]  v7=C[15][0]
...
lane 31: v0=C[1][15] ... v7=C[15][15]
```

In formula form (0-based): for VGPR index `g` (0..7) and lane `L` (0..31),
```
  C/D[ i = 2*g + (L // 16) ][ j = L % 16 ]   is at  VGPR g, lane L   (one i32 per VGPR)
```
Equivalently, lane `L` VGPR `g` holds `D[2*g + (L>=16)][L%16]` — this matches both
the calculator `--matrix-layout` table and the f16 output figure
(`f16_16x16x16_f16_w32_output.png`) which shows VGPR0/lanes0-15 = row 1,
VGPR0/lanes16-31 = row 2, etc., generalized to i32 where each lane holds 8 rows strided by 2.

> **Sources for §5.1:**
> - Calculator: `matrix_calculator.py --architecture gfx1100 --instruction v_wmma_i32_16x16x16_iu4 --detail-instruction` → formulas above verbatim;
>   `--A-matrix --matrix-layout` shows lane 0 = `A[0][0..15]` across v0/v1 nibbles;
>   `--C-matrix --matrix-layout` shows lane 0 = `C[0,2,4,...,14][0]` across v0..v7.
> - ISA XML: `FMT_WMMA_AB_16X16_IU4` BitCount 64 (2 VGPRs), `FMT_WMMA_DC_16X16_I32` BitCount 256 (8 VGPRs).
> - LLVM: `VOP_V8I32_V2I32_V2I32_V8I32` (2+2+8 VGPRs).

---

### 5.2  iu8 — `v_wmma_i32_16x16x16_iu8` (D = i32, A/B = iu8 packed 4 per VGPR)

**VGPRs per lane:** A 4 (`v0:3`), B 4 (`v4:7`), C/D 8. ISA `OperandSize` 128 for SRC0/SRC1 = 4 VGPRs.

**Packing (iu8):** Each VGPR holds **4 int8** values, 8 bits each. K=0 in `[7:0]`,
K=1 in `[15:8]`, K=2 in `[23:16]`, K=3 in `[31:24]` of VGPR 0; K=4..7 in VGPR 1, etc.
`FMT_WMMA_AB_16X16_IU8` BitMaps confirm 8-bit steps at offsets 0,8,16,…,120.

```
Matrix-element → register (wave32, no modifiers):
  A[i][k]   GPR: floor(k/4).[8*(k%4)+7 : 8*(k%4)]    Lane: i and i+16
  B[k][j]   GPR: floor(k/4).[8*(k%4)+7 : 8*(k%4)]    Lane: j and j+16
  C/D[i][j] GPR: floor(i/2)                          Lane: ((16*i) % 32) + j
```

```
Register → matrix-element (wave32, inverse):
  A i: (lane % 16)
    k: 4*GPR_num + floor(GPR_bits / 8)
  B j: (lane % 16)
    k: 4*GPR_num + floor(GPR_bits / 8)
  C/D i: 2*GPR_num + floor(lane / 16)
      j: (lane % 16)
```

C/D mapping is **identical** to iu4 (same i32 accumulator). Only A/B packing differs
(4 bytes/VGPR instead of 8 nibbles/VGPR).

> **Sources for §5.2:** calculator `--detail` + `--matrix-layout` for `v_wmma_i32_16x16x16_iu8`;
> ISA XML `FMT_WMMA_AB_16X16_IU8` BitCount 128; LLVM `VOP_V8I32_V4I32_V4I32_V8I32`.

### 5.3  f16 — `v_wmma_f16_16x16x16_f16` (cross-check, D = f16 packed)

**VGPRs per lane:** A 8, B 8, C/D 8 (wave32). ISA `FMT_WMMA_DC_16X16_F16` BitCount 256 but
ComponentCount 8 with OPSEL — only **one** f16 per VGPR is active at a time (low or high half).

**Packing (f16):** Each VGPR holds **2 fp16** values, 16 bits each. K=0 in `[15:0]`,
K=1 in `[31:16]` of VGPR 0; K=2..3 in VGPR 1, …, K=14..15 in VGPR 7.

```
Matrix-element → register (wave32, no modifiers, OPSEL=0 → low half):
  A[i][k]   GPR: floor(k/2).[16*(k%2)+15 : 16*(k%2)]   Lane: i and i+16
  B[k][j]   GPR: floor(k/2).[16*(k%2)+15 : 16*(k%2)]   Lane: j and j+16
  C/D[i][j] GPR: (floor(i/2)).[15:0]                   Lane: ((16*i) % 32) + j
            // only bits [15:0] when OPSEL=0; OPSEL=1 → [31:16]
```

```
Register → matrix-element (wave32, inverse):
  A i: (lane % 16)
    k: 2*GPR_num + floor(GPR_bits / 16)
  B j: (lane % 16)
    k: 2*GPR_num + floor(GPR_bits / 16)
  C/D i: (2*GPR_num + floor(lane/16)).[15:0]   // OPSEL selects half
      j: (lane % 16)
```

The f16 input figure (`f16_16x16x16_f16_w32_input.png`) confirms: VGPR0 `[15:0]` = K0,
`[31:16]` = K1, VGPR1 `[15:0]` = K2, etc., with lane L = row (A) or column (B).
The f16 output figure (`f16_16x16x16_f16_w32_output.png`) confirms: VGPR `v` holds
rows `2*v` (lanes 0-15) and `2*v+1` (lanes 16-31), column = lane%16 — same pattern as i32
but with only one f16 per VGPR active (low half when `opsel=0`).

> **Quote — GPUOpen blog (OPSEL):**
> “The “OPSEL” parameter is a boolean flag required to be specified when using
> a 16-bit format for the C and D matrices. If this flag is set to true, the
> elements of C and D are stored in the upper half of the VGPR. However, when
> this flag is set to false, they are stored in the lower half of the VGPR.”

> **LLVM `IntrinsicsAMDGPU.td`:** `int_amdgcn_wmma_f16_16x16x16_f16` is
> `AMDGPUWmmaIntrinsicOPSEL` with an `i1 %high (op_sel)` argument; the `tied`
> variant is also OPSEL. `int_amdgcn_wmma_i32_*` are `AMDGPUWmmaIntrinsicIU`
> with **no** OPSEL argument (no half selection for i32).

---

## 6. How 4-bit values are packed (nibble order vs K)

All three integer/float variants pack **low K in low bits** (little-endian within the VGPR).

| Variant | Values per VGPR | Bits per value | K → bits | Order |
|---------|---------------|----------------|----------|-------|
| iu4 | 8 | 4 | K → `[4*(K%8)+3 : 4*(K%8)]` in VGPR `floor(K/8)` | low nibble = low K |
| iu8 | 4 | 8 | K → `[8*(K%4)+7 : 8*(K%4)]` in VGPR `floor(K/4)` | low byte = low K |
| f16/bf16 | 2 | 16 | K → `[16*(K%2)+15 : 16*(K%2)]` in VGPR `floor(K/2)` | low half = low K |

**Proof:** The ISA XML `FMT_WMMA_AB_*` `BitMap` Orders are 0..15 sequential with
`BitOffset` 0,4,8,… (iu4), 0,8,16,… (iu8), 0,16,32,… (f16). The calculator
`--matrix-layout` tables show `A[0][0]` at `v0.[3:0]` and `A[0][1]` at `v0.[7:4]` (iu4),
`A[0][0]` at `v0.[7:0]` and `A[0][1]` at `v0.[15:8]` (iu8), `A[0][0]` at `v0.[15:0]`
and `A[0][1]` at `v0.[31:16]` (f16).

**Signedness (iu4/iu8):** The 4-/8-bit values are **sign-extended or zero-extended**
based on the `neg_lo` modifier. NEG is *not* lane-dependent.

> **LLVM `VOP3PInstructions.td` / `IntrinsicsAMDGPU.td`:** The IU intrinsics are
> `AMDGPUWmmaIntrinsicIU` with `i1 %A_sign`, `i1 %B_sign`, `i1 %clamp` (all `ImmArg`).
> `neg_lo` bit 0 = A signed (sext), bit 1 = B signed, bit 2 controls C sign/clamp.
> See `gfx11_asm_wmma.s`: `v_wmma_i32_16x16x16_iu4 ... neg_lo:[1,0,0]` means A=signed, B=unsigned.

**Numpy emulator sketch (iu4):**

```python
import numpy as np

def unpack_iu4_vgprs(vgprs, signed):
    # vgprs: np.ndarray shape (2,) dtype=np.uint32, per lane
    # returns: np.ndarray shape (16,) of int32, K=0..15
    out = np.empty(16, dtype=np.int32)
    for k in range(16):
        g = k // 8          # VGPR index 0 or 1
        nib = (vgprs[g] >> (4*(k % 8))) & 0xF
        if signed:
            nib = np.int8(nib << 4) >> 4   # sign-extend 4b -> 8b -> 32b
        out[k] = int(nib)
    return out

def unpack_iu8_vgprs(vgprs, signed):
    out = np.empty(16, dtype=np.int32)
    for k in range(16):
        g = k // 4
        byte = (vgprs[g] >> (8*(k % 4))) & 0xFF
        if signed:
            byte = np.int8(byte)
        out[k] = int(byte)
    return out
```

---

