#!/usr/bin/env bash
# Build comfy-kitchen with 0001-hip-svdquant-w4a4-perf.patch (PR #262, 08216d9) for RDNA3/RDNA4 (default: gfx1100).
# Usage: COMFY_HIP_ARCHS=gfx1100 bash scripts/build_patched_kitchen.sh [BUILD_DIR] [PYTHON_BIN]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${1:-$REPO_ROOT/build/ck-rocm}"
BASE_PY="${2:-python3}"
ARCH="${COMFY_HIP_ARCHS:-gfx1100}"

export PIP_DISABLE_PIP_VERSION_CHECK=1
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

if [ ! -x "$BUILD_DIR/venv/bin/python" ]; then
    "$BASE_PY" -m venv "$BUILD_DIR/venv"
fi
PY="$BUILD_DIR/venv/bin/python"

echo "[1/4] Installing build tools and AMD ROCm SDK for $ARCH..."
"$PY" -m pip install -q cmake ninja setuptools wheel
if ! command -v hipcc >/dev/null 2>&1; then
    PRIMARY_ARCH="${ARCH%%;*}"
    "$PY" -m pip install -q "rocm[libraries,devel,device-$PRIMARY_ARCH]" \
        --index-url https://repo.amd.com/rocm/whl-multi-arch/ \
        --extra-index-url https://pypi.org/simple
    ROOT="$("$PY" -m rocm_sdk path --root)"
    export ROCM_HOME="$ROOT" ROCM_PATH="$ROOT" PATH="$ROOT/bin:$BUILD_DIR/venv/bin:$PATH"
fi

echo "[2/4] Fetching comfy-kitchen base commit (9064367) and applying patch..."
if [ ! -d ck ]; then
    git clone -q https://github.com/Comfy-Org/comfy-kitchen.git ck
fi
cd ck
git fetch -q origin 90643673e4c2d1501d1bea2de8998b80d49ac8ca
git checkout -q --force 90643673e4c2d1501d1bea2de8998b80d49ac8ca
git apply "$REPO_ROOT/patches/0001-hip-svdquant-w4a4-perf.patch"

echo "[3/4] Building wheel for COMFY_HIP_ARCHS=$ARCH..."
export COMFY_KITCHEN_BUILD_HIP=1
export COMFY_HIP_ARCHS="$ARCH"
export MAX_JOBS="${MAX_JOBS:-8}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-8}"
rm -rf dist
"$PY" -m pip wheel . --no-deps -w dist

echo "[4/4] Installing into isolated site directory: $BUILD_DIR/site"
rm -rf "$BUILD_DIR/site"
"$PY" -m pip install -q --no-deps --target "$BUILD_DIR/site" dist/*.whl

echo "Done. Run benchmarks with:"
echo "  PYTHONPATH=$BUILD_DIR/site python3 $REPO_ROOT/bench/bench_svdquant_w4a4.py"
