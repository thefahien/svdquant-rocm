#!/usr/bin/env python3
"""SVDQuant W4A4 (comfy-kitchen HIP) on gfx1100: correctness vs eager reference + speed vs BF16.

Synthetic layers at DiT (Flux / Qwen-Image) shapes, rank 32, group size 64.
Prints JSON lines.
"""
import json
import sys
import time
import torch
import comfy_kitchen
from comfy_kitchen.backends import hip as H
from comfy_kitchen.backends.eager import svdquant as E

torch.manual_seed(0)
dev = "cuda"
DT = torch.bfloat16
R = 32
G = 64
print(json.dumps({
    "device": torch.cuda.get_device_name(0),
    "torch": torch.__version__,
    "hip": torch.version.hip,
    "kitchen_version": getattr(comfy_kitchen, "__version__", "unknown"),
    "arch": torch.cuda.get_device_properties(0).gcnArchName,
}))


def quant_weight(w):
    """(N, K) fp -> qweight (N, K//2) int8 row-major nibbles, wscales (K//64, N). absmax/7."""
    n, k = w.shape
    g = w.float().view(n, k // G, G)
    s = g.abs().amax(-1).clamp(min=1e-10) / 7
    q = (g / s.unsqueeze(-1)).round().clamp(-7, 7).to(torch.int8).view(n, k)
    return E._pack_int4_row_major(q), s.t().contiguous().to(DT)


def make_layer(m, k, n):
    x = torch.randn(m, k, device=dev, dtype=DT)
    w = torch.randn(n, k, device=dev, dtype=DT) / k ** 0.5
    smooth = (torch.rand(k, device=dev) * 1.5 + 0.5).to(DT)
    down = (torch.randn(k, R, device=dev) / k ** 0.5).to(DT)
    up = (torch.randn(n, R, device=dev) / R ** 0.5).to(DT)
    qw, ws = quant_weight(w * smooth)
    return x, w, smooth, down, up, qw, ws


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def timeit(fn, iters=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1e3


shapes = [
    (4096, 3072, 3072),
    (4096, 3072, 12288),
    (4096, 12288, 3072),
    (4096, 4096, 4096),
    (8192, 3072, 12288),
    (256, 3072, 3072),
]
ok_all = True
for (m, k, n) in shapes:
    x, w, smooth, down, up, qw, ws = make_layer(m, k, n)
    qh, ash, lah = H.quantize_svdquant_w4a4(x, smooth, down)
    qe, ase, lae = E.quantize_svdquant_w4a4(x, smooth, down)
    torch.cuda.synchronize()
    q_mismatch = (E._unpack_int4_row_major(qh) != E._unpack_int4_row_major(qe)).float().mean().item()
    oh = H.scaled_mm_svdquant_w4a4(qh, qw, ash, ws, lah, up)
    oe = E.scaled_mm_svdquant_w4a4(qe, qw, ase, ws, lae, up)
    truth = (x.float() / smooth.float()) @ (w.float() * smooth.float()).t() + x.float() @ down.float() @ up.float().t()
    torch.cuda.synchronize()
    finite = bool(torch.isfinite(oh).all())
    r_hip_eager = rel(oh[:m], oe[:m])
    r_hip_truth = rel(oh[:m], truth)
    r_eager_truth = rel(oe[:m], truth)
    ok = finite and r_hip_truth <= r_eager_truth * 1.02
    ok_all &= ok

    t_bf16 = timeit(lambda: x @ w.t())

    def w4a4():
        q, a, l = H.quantize_svdquant_w4a4(x, smooth, down)
        return H.scaled_mm_svdquant_w4a4(q, qw, a, ws, l, up)

    t_w4a4 = timeit(w4a4)
    q0, a0, l0 = H.quantize_svdquant_w4a4(x, smooth, down)
    t_gemm = timeit(lambda: H.scaled_mm_svdquant_w4a4(q0, qw, a0, ws, l0, up))
    flops = 2 * m * k * n
    print(json.dumps({
        "shape": [m, k, n],
        "PASS": ok,
        "finite": finite,
        "act_q_mismatch": round(q_mismatch, 6),
        "rel_hip_vs_eager": round(r_hip_eager, 5),
        "rel_hip_vs_fp32": round(r_hip_truth, 5),
        "rel_eager_vs_fp32": round(r_eager_truth, 5),
        "ms_bf16": round(t_bf16, 3),
        "ms_w4a4_total": round(t_w4a4, 3),
        "ms_w4a4_gemm": round(t_gemm, 3),
        "speedup_total": round(t_bf16 / t_w4a4, 2),
        "TFLOPS_bf16": round(flops / t_bf16 / 1e9, 1),
        "TOPS_w4a4_gemm": round(flops / t_gemm / 1e9, 1),
    }), flush=True)
    del x, w, qw, ws, truth, oh, oe
    torch.cuda.empty_cache()

print(json.dumps({"ALL_PASS": ok_all, "peak_mem_GB": round(torch.cuda.max_memory_allocated() / 1e9, 2)}))
sys.exit(0 if ok_all else 1)
