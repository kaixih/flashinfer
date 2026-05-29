#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-/scratch/models/Qwen3-30B-A3B}
TP=${TP:-8}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-30000}
CUDA_GRAPH_MAX_BS=${CUDA_GRAPH_MAX_BS:-1024}
TIMEOUT_SEC=${TIMEOUT_SEC:-2400}
OUT_ROOT=${OUT_ROOT:-/scratch/repro/miles-b200-qwen3-30b/logs}
RUN_ID=${RUN_ID:-sglang-pure-piecewise-maxbs${CUDA_GRAPH_MAX_BS}-bf16-valid-configs-$(date +%Y%m%d-%H%M%S)}
RUN_DIR=${OUT_ROOT}/${RUN_ID}

mkdir -p "${RUN_DIR}"

export FLASHINFER_FORCE_JIT_FUSED_MOE_TRTLLM="${FLASHINFER_FORCE_JIT_FUSED_MOE_TRTLLM:-1}"
export FLASHINFER_AUTOTUNER_LOAD_FROM_FILE="${FLASHINFER_AUTOTUNER_LOAD_FROM_FILE:-0}"
export FLASHINFER_DEBUG_TRTLLM_BMM_PTRS="${FLASHINFER_DEBUG_TRTLLM_BMM_PTRS:-1}"
export FLASHINFER_DEBUG_TRTLLM_BMM_PTRS_LIMIT="${FLASHINFER_DEBUG_TRTLLM_BMM_PTRS_LIMIT:-50000}"
export FLASHINFER_DEBUG_TRTLLM_BF16_VALID_CONFIGS="${FLASHINFER_DEBUG_TRTLLM_BF16_VALID_CONFIGS:-1}"
export FLASHINFER_DEBUG_TRTLLM_BF16_VALID_CONFIGS_LIMIT="${FLASHINFER_DEBUG_TRTLLM_BF16_VALID_CONFIGS_LIMIT:-80}"
export FLASHINFER_DEBUG_TRTLLM_BF16_VALID_CONFIGS_MAX_ITEMS="${FLASHINFER_DEBUG_TRTLLM_BF16_VALID_CONFIGS_MAX_ITEMS:-400}"

echo "RUN_ID=${RUN_ID}"
echo "RUN_DIR=${RUN_DIR}"
python3 -m pip list | grep -E 'flashinfer|sglang' | tee "${RUN_DIR}/versions.txt" || true

if [[ "${CLEAN_SERVER:-1}" == "1" ]]; then
  pkill -f "sglang.launch_server" || true
  sleep 5
fi

CMD=(
  python3 -m sglang.launch_server
  --model-path "${MODEL_PATH}"
  --tensor-parallel-size "${TP}"
  --host "${HOST}"
  --port "${PORT}"
  --trust-remote-code
  --moe-runner-backend flashinfer_trtllm
  --cuda-graph-max-bs "${CUDA_GRAPH_MAX_BS}"
  --disable-flashinfer-autotune
)

printf '%q ' "${CMD[@]}" | tee "${RUN_DIR}/server.cmd"
echo | tee -a "${RUN_DIR}/server.cmd"
env | grep -E 'FLASHINFER_(FORCE|DEBUG|AUTOTUNER)' | sort | tee "${RUN_DIR}/flashinfer_env.txt"

set +e
timeout "${TIMEOUT_SEC}" "${CMD[@]}" > "${RUN_DIR}/server.log" 2>&1
status=$?
set -e

echo "exit_status=${status}" | tee "${RUN_DIR}/status.txt"
grep -E "FI_TRTLLM_BF16_VALID_TACTICS|FI_TRTLLM_MOE_SELECTED_CONFIG|FI_TRTLLM_BMM_PTR|Capture cuda graph|Capture piecewise|Piecewise CUDA Graph failed|illegal memory|CUDA error|Traceback|RuntimeError|server is fired up|Uvicorn running" \
  "${RUN_DIR}/server.log" | tail -220 | tee "${RUN_DIR}/summary.txt" || true

echo "RUN_DIR=${RUN_DIR}"
exit "${status}"
