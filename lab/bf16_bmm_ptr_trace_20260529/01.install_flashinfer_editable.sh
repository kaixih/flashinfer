#!/usr/bin/env bash
set -euo pipefail

FLASHINFER_SRC="${FLASHINFER_SRC:-/scratch/repo/flashinfer}"
cd "${FLASHINFER_SRC}"

echo "Installing FlashInfer from ${FLASHINFER_SRC}"
echo "HEAD=$(git rev-parse HEAD 2>/dev/null || true)"
echo "BMM pointer trace markers:"
grep -n "FI_TRTLLM_BMM_PTR\\|FLASHINFER_DEBUG_TRTLLM_BMM_PTRS" \
  csrc/trtllm_fused_moe_runner.cu csrc/trtllm_batched_gemm_runner.cu | head -n 80
echo "Forced JIT marker:"
grep -n "FLASHINFER_FORCE_JIT_FUSED_MOE_TRTLLM" flashinfer/jit/core.py

export FLASHINFER_CUDA_ARCH_LIST="${FLASHINFER_CUDA_ARCH_LIST:-10.0a}"
export MAX_JOBS="${MAX_JOBS:-16}"
export FLASHINFER_NVCC_THREADS="${FLASHINFER_NVCC_THREADS:-1}"
export FLASHINFER_DISABLE_VERSION_CHECK="${FLASHINFER_DISABLE_VERSION_CHECK:-1}"
export FLASHINFER_FORCE_JIT_FUSED_MOE_TRTLLM="${FLASHINFER_FORCE_JIT_FUSED_MOE_TRTLLM:-1}"

python3 -m pip install -e . -v --no-build-isolation --no-deps

python3 - <<'PY'
import os
import flashinfer

path = os.path.abspath(getattr(flashinfer, "__file__", ""))
force_jit = os.environ.get("FLASHINFER_FORCE_JIT_FUSED_MOE_TRTLLM") == "1"
print("flashinfer:", path)
print("FLASHINFER_FORCE_JIT_FUSED_MOE_TRTLLM:", os.environ.get("FLASHINFER_FORCE_JIT_FUSED_MOE_TRTLLM"))
if "/scratch/repo/flashinfer" not in path and not force_jit:
    raise SystemExit("flashinfer is not loading from /scratch/repo/flashinfer and forced JIT is not active")
print("install check ok")
PY
