#!/usr/bin/env python3
"""BF16/FP16 GEMM throughput on the RX 7900 XTX under different BLAS backends.

Usage: MODE=<default|rocblas|hipblaslt> python3 bench/bench_blas_modes.py
(Set PYTORCH_TUNABLEOP_ENABLED=1 via env or use scripts/run_blas_tune.sh.)
Prints one JSON line per shape.
"""
import json
import os
import time
import torch

mode = os.environ.get("MODE", "default")
if mode == "rocblas":
    torch.backends.cuda.preferred_blas_library("cublas")      # maps to rocBLAS on ROCm
elif mode == "hipblaslt":
    torch.backends.cuda.preferred_blas_library("cublaslt")    # maps to hipBLASLt on ROCm
tun = os.environ.get("PYTORCH_TUNABLEOP_ENABLED", "0")
label = mode + ("+tunableop" if tun == "1" else "")
torch.manual_seed(0)
SHAPES = [
    (4096, 3072, 3072),
    (4096, 3072, 12288),
    (4096, 12288, 3072),
    (4096, 4096, 4096),
    (8192, 3072, 12288),
    (4608, 3072, 9216),
    (1024, 3072, 3072),
]


def bench(a, b, iters=30):
    for _ in range(5):
        a @ b.t()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        c = a @ b.t()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1e3, c


for dt in (torch.bfloat16, torch.float16):
    for (m, k, n) in SHAPES:
        a = torch.randn(m, k, device="cuda", dtype=dt)
        b = torch.randn(n, k, device="cuda", dtype=dt)
        ms, c = bench(a, b)
        ref = a[:64].float() @ b[:256].float().t()
        err = ((c[:64, :256].float() - ref).abs().max() / ref.abs().max()).item()
        print(json.dumps({
            "label": label,
            "blas": str(torch.backends.cuda.preferred_blas_library()),
            "dtype": str(dt).split(".")[1],
            "shape": [m, k, n],
            "ms": round(ms, 3),
            "TFLOPS": round(2 * m * k * n / ms / 1e9, 1),
            "max_rel_err": float(f"{err:.2e}"),
        }), flush=True)
        del a, b, c
