#!/usr/bin/env bash
# Benchmark BF16/FP16 GEMM throughput across default (rocBLAS), hipBLASLt, and TunableOp.
# Each setting runs in a fresh Python process.
# Usage: bash scripts/run_blas_tune.sh [PYTHON_BIN]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${1:-python3}"
TUNE_CSV="${TUNE_CSV:-/tmp/tunable_gfx1100.csv}"

for MODE in default rocblas hipblaslt; do
    MODE="$MODE" "$PY" "$REPO_ROOT/bench/bench_blas_modes.py"
done

rm -f "$TUNE_CSV" "${TUNE_CSV%.csv}"*.csv
MODE=default PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=1 \
    PYTORCH_TUNABLEOP_FILENAME="$TUNE_CSV" \
    "$PY" "$REPO_ROOT/bench/bench_blas_modes.py" > /dev/null

MODE=default PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=0 \
    PYTORCH_TUNABLEOP_FILENAME="$TUNE_CSV" \
    "$PY" "$REPO_ROOT/bench/bench_blas_modes.py"

MODE=default "$PY" "$REPO_ROOT/bench/bench_blas_modes.py" \
    | sed 's/"label": "default"/"label": "default-again"/'
