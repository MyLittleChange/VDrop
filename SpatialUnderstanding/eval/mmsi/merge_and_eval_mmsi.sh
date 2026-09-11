#!/bin/bash
#
# Merge sharded MMSI-Bench results and optionally run vLLM LLM judge evaluation.
#
# Usage:
#   bash merge_and_eval_mmsi.sh
#

set -euo pipefail

SCRIPT_DIR="/path/to/ThinkMorph-BAGEL-release"

VENV_PATH="${VENV_PATH:-/path/to/home/verl/.venv/bin/activate}"
if command -v module >/dev/null 2>&1; then
    module load cuda/12.6.0
fi
unset ROCR_VISIBLE_DEVICES
source "${VENV_PATH}"

# ==================== Configuration ====================
# Choose ONE of the following result directories (comment/uncomment as needed):

# BAGEL baseline (24 shards)
# RESULTS_DIR="/path/to/scratch/VisualCoT/BAGEL_mmsi_eval"
# NUM_SHARDS=24

# BAGEL sg_2000 (48 shards)
RESULTS_DIR="${RESULTS_DIR:-/path/to/scratch/VisualCoT/training_data_mix_balance_matterport_point_matching_no_think_lora_mmsi_eval}"
NUM_SHARDS="${NUM_SHARDS:-24}"

# Output directory (default: same as results_dir)
OUTPUT_DIR="${OUTPUT_DIR:-${RESULTS_DIR}}"

# Run number
RUN_NUM=1

# ==================== Mode ====================
# Set MODE to one of: "merge", "summary", "judge"
#   merge   - merge shards + print basic metrics (no LLM judge)
#   summary - print metrics from an already-evaluated JSON file
#   judge   - merge shards + run LLM judge + print accuracy
MODE="${MODE:-judge}"

# ==================== Summary mode settings ====================
# Path to an already-evaluated JSON file (only used when MODE=summary)
SUMMARY_FILE="${RESULTS_DIR}/evaluated_model_results_run_${RUN_NUM}.json"

# ==================== LLM Judge settings (only used when MODE=judge) ====================
JUDGE_BACKEND="${JUDGE_BACKEND:-vllm}"
NUM_PROCESSES="${NUM_PROCESSES:-16}"

# vLLM settings for Qwen3-VL judge. Set START_VLLM=0 to reuse an already-running server.
START_VLLM="${START_VLLM:-1}"
VLLM_MODEL_PATH="${VLLM_MODEL_PATH:-Qwen/Qwen3-VL-235B-A22B-Instruct-FP8}"
VLLM_MODEL_NAME="${VLLM_MODEL_NAME:-Qwen/Qwen3-VL-235B-A22B-Instruct-FP8}"
JUDGE_MODEL="${JUDGE_MODEL:-${VLLM_MODEL_NAME}}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://${VLLM_HOST}:${VLLM_PORT}/v1}"
VLLM_GPUS="${VLLM_GPUS:-0,1,2,3}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-4}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.9}"
VLLM_READY_RETRIES="${VLLM_READY_RETRIES:-500}"
VLLM_READY_SLEEP="${VLLM_READY_SLEEP:-3}"
VLLM_LOG="${VLLM_LOG:-${OUTPUT_DIR}/vllm_mmsi_judge.log}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

VLLM_PID=""
cleanup() {
    if [ -n "${VLLM_PID}" ]; then
        log "Stopping vLLM server (PID: ${VLLM_PID})..."
        kill "${VLLM_PID}" 2>/dev/null || true
        wait "${VLLM_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

start_vllm_if_needed() {
    if [ "${JUDGE_BACKEND}" != "vllm" ] || [ "${START_VLLM}" = "0" ]; then
        return
    fi

    mkdir -p "$(dirname "${VLLM_LOG}")"

    if curl -s "http://${VLLM_HOST}:${VLLM_PORT}/health" >/dev/null 2>&1; then
        log "Using existing vLLM server at ${VLLM_BASE_URL}"
        return
    fi

    log "Starting vLLM judge server"
    log "  Model path: ${VLLM_MODEL_PATH}"
    log "  Served name: ${VLLM_MODEL_NAME}"
    log "  GPUs: ${VLLM_GPUS}"
    log "  Base URL: ${VLLM_BASE_URL}"
    log "  Log: ${VLLM_LOG}"

    CUDA_VISIBLE_DEVICES="${VLLM_GPUS}" vllm serve "${VLLM_MODEL_PATH}" \
        --host "${VLLM_HOST}" \
        --port "${VLLM_PORT}" \
        --trust-remote-code \
        --max-model-len "${VLLM_MAX_MODEL_LEN}" \
        --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
        --disable-log-stats \
        --tensor-parallel-size "${VLLM_TENSOR_PARALLEL_SIZE}" \
        --served-model-name "${VLLM_MODEL_NAME}" \
        --disable-mm-preprocessor-cache \
        > "${VLLM_LOG}" 2>&1 &

    VLLM_PID=$!
    log "vLLM server started with PID: ${VLLM_PID}"

    READY=false
    i=1
    while [ "${i}" -le "${VLLM_READY_RETRIES}" ]; do
        if curl -s "http://${VLLM_HOST}:${VLLM_PORT}/health" >/dev/null 2>&1; then
            log "vLLM server is ready"
            READY=true
            break
        fi
        if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
            log "ERROR: vLLM server process died"
            tail -n 200 "${VLLM_LOG}" || true
            exit 1
        fi
        sleep "${VLLM_READY_SLEEP}"
        i=$((i + 1))
    done

    if [ "${READY}" = false ]; then
        log "ERROR: vLLM server failed to start"
        tail -n 200 "${VLLM_LOG}" || true
        exit 1
    fi
}

# ==================== Run ====================
case "${MODE}" in
    merge)
        python3 "/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/eval/mmsi/merge_and_eval_mmsi.py" \
            --results_dir "${RESULTS_DIR}" \
            --output_dir "${OUTPUT_DIR}" \
            --run_num "${RUN_NUM}" \
            --num_shards "${NUM_SHARDS}"
        ;;
    summary)
        python3 "/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/eval/mmsi/merge_and_eval_mmsi.py" \
            --summary "${SUMMARY_FILE}"
        ;;
    judge)
        start_vllm_if_needed
        python3 "/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/eval/mmsi/merge_and_eval_mmsi.py" \
            --results_dir "${RESULTS_DIR}" \
            --output_dir "${OUTPUT_DIR}" \
            --run_num "${RUN_NUM}" \
            --num_shards "${NUM_SHARDS}" \
            --run_judge \
            --judge_backend "${JUDGE_BACKEND}" \
            --judge_model "${JUDGE_MODEL}" \
            --base_url "${VLLM_BASE_URL}" \
            --num_processes "${NUM_PROCESSES}"
        ;;
    *)
        echo "ERROR: Unknown MODE '${MODE}'. Use 'merge', 'summary', or 'judge'."
        exit 1
        ;;
esac
