#!/usr/bin/env python3
"""comfy-kitchen HIP int8_linear (ConvRot, the path Qwen-Image-2.1 int8_convrot uses) on gfx1100.

Shapes = one Qwen-Image-2.1 block at 768x1376 (4128 image tokens, hidden 4096, 32 blocks).
Optionally pass a safetensors checkpoint path to scrape the exact layer counts from its header;
otherwise uses the standard 32-block Qwen-Image-2.1 layer table.
"""
import argparse
import json
import re
import struct
import time
import torch
import comfy_kitchen
from comfy_kitchen.backends import hip as H

DEFAULT_LAYERS = {
    "attn.to_k.weight": [32, [4096, 4096]],
    "attn.to_out.0.weight": [32, [4096, 4096]],
    "attn.to_q.weight": [32, [4096, 4096]],
    "attn.to_v.weight": [32, [4096, 4096]],
    "img_mlp.gate_up.weight": [32, [24576, 4096]],
    "img_mlp.out.weight": [32, [4096, 12288]],
}


def load_layers(ckpt_path):
    if not ckpt_path:
        return DEFAULT_LAYERS
    with open(ckpt_path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    lin = {}
    for k, v in hdr.items():
        if k.endswith(".weight") and isinstance(v, dict) and v.get("dtype") == "I8":
            key = re.sub(r"transformer_blocks\.\d+\.", "", k)
            lin.setdefault(key, [0, v["shape"]])[0] += 1
    return lin


def timeit(fn, iters=30):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None, help="Optional safetensors checkpoint to scrape I8 linear shapes")
    ap.add_argument("--tokens", type=int, default=4128, help="Image token count M (default: 4128 for 768x1376)")
    args = ap.parse_args()

    dev = "cuda"
    torch.manual_seed(0)
    print(json.dumps({
        "kitchen_version": getattr(comfy_kitchen, "__version__", "unknown"),
        "arch": torch.cuda.get_device_properties(0).gcnArchName,
    }), flush=True)

    lin = load_layers(args.checkpoint)
    print(json.dumps({"int8_linears_per_kind": lin}), flush=True)

    total_ms = {"int8_linear": 0.0, "bf16_rocblas": 0.0, "bf16_hipblaslt": 0.0}
    for key, (count, (N, K)) in sorted(lin.items()):
        M = 64 if ("txt" in key or "add_" in key) else args.tokens
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        w8 = torch.randint(-127, 128, (N, K), device=dev, dtype=torch.int8)
        ws = torch.rand(N, 1, device=dev) * 0.01 + 1e-3
        wb = torch.randn(N, K, device=dev, dtype=torch.bfloat16)
        flops = 2 * M * N * K

        t_lin = timeit(lambda: H.int8_linear(x, w8, ws, None, torch.bfloat16, True, 256))
        t_quant = timeit(lambda: H._rotate_quant_int8(x, 256, None, None, 0.0))
        q, xs = H._rotate_quant_int8(x, 256, None, None, 0.0)
        xs = xs.reshape(-1).contiguous()
        w8a = H._aligned(w8.contiguous())
        wsf = ws.reshape(-1).contiguous()
        out = torch.empty((M, N), dtype=torch.bfloat16, device=dev)
        code = H.DTYPE_TO_CODE[torch.bfloat16]
        t_gemm = timeit(lambda: H._C.int8_gemm(
            H._dl(q), H._dl(w8a), H._dl(out), H._dl(xs), H._dl(wsf), 1,
            None, M, N, K, code, H._stream(x)
        ))

        torch.backends.cuda.preferred_blas_library("cublas")
        t_rb = timeit(lambda: x @ wb.t())
        torch.backends.cuda.preferred_blas_library("cublaslt")
        t_lt = timeit(lambda: x @ wb.t())
        torch.backends.cuda.preferred_blas_library("cublas")

        total_ms["int8_linear"] += t_lin * count
        total_ms["bf16_rocblas"] += t_rb * count
        total_ms["bf16_hipblaslt"] += t_lt * count
        print(json.dumps({
            "layer": key,
            "count": count,
            "MxKxN": [M, K, N],
            "ms_int8_linear": round(t_lin, 3),
            "ms_rotate_quant": round(t_quant, 3),
            "ms_int8_gemm": round(t_gemm, 3),
            "TOPS_int8_gemm": round(flops / t_gemm / 1e9, 1),
            "ms_bf16_rocblas": round(t_rb, 3),
            "ms_bf16_hipblaslt": round(t_lt, 3),
            "TFLOPS_bf16_hipblaslt": round(flops / t_lt / 1e9, 1),
        }), flush=True)
        del x, w8, wb, out, q
        torch.cuda.empty_cache()

    print(json.dumps({
        "per_forward_ms": {k: round(v, 1) for k, v in total_ms.items()},
        "note": "image-stream int8 linears only, one forward = one sampling step at cfg 1",
    }), flush=True)


if __name__ == "__main__":
    main()
