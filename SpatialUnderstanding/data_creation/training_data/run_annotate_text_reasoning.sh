#!/bin/bash
#SBATCH --job-name=annotate_text_reasoning
#SBATCH --partition=short-unkillable
#SBATCH --gres=gpu:a100l:4
#SBATCH -c 24
#SBATCH --output=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/data_creation/training_data/slurm_logs/annotate_text_reasoning-%A_%a.txt
#SBATCH --error=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/data_creation/training_data/slurm_logs/annotate_text_reasoning_error-%A_%a.txt
#SBATCH --ntasks=1
#SBATCH --time=3:00:00
#SBATCH --mem=512Gb

# ==============================================================================
# Unified text-reasoning annotation job.
#
# Starts Qwen3-VL-235B-FP8 with vLLM on this Slurm allocation, waits for the
# local OpenAI-compatible endpoint, runs annotate_text_reasoning.py, then stops
# the server on exit. This mirrors run_datagen_spatial_gpu.sh so annotation no
# longer depends on a separately submitted vLLM server job.
#
# Usage:
#   sbatch SpatialUnderstanding/data_creation/training_data/run_annotate_text_reasoning.sh
#   sbatch --array=0-7 SpatialUnderstanding/data_creation/training_data/run_annotate_text_reasoning.sh
#
# Useful overrides:
#   TOTAL_SHARDS=8 WORKERS=8 MAX_SAMPLES=50 sbatch --array=0-7 ...
#   VLLM_PORT=4877 TEMPERATURE=0.2 MAX_TOKENS=768 sbatch ...
# ==============================================================================

set -euo pipefail

module load cuda/12.6.0
unset ROCR_VISIBLE_DEVICES
source /path/to/home/verl/.venv/bin/activate

################################################################################
# CONFIGURATION
################################################################################

SCRIPT_DIR="/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/data_creation/training_data"
REPO_DIR="/path/to/ThinkMorph-BAGEL-release"
LOG_DIR="${SCRIPT_DIR}/slurm_logs"

VLLM_MODEL="${VLLM_MODEL:-/path/to/scratch/models/Qwen3-VL-235B-A22B-Instruct-FP8}"
VLLM_MODEL_NAME="${VLLM_MODEL_NAME:-qwen3_vl_235b_fp8}"
VLLM_PORT="${VLLM_PORT:-8765}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-16384}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.95}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-16}"

TOTAL_SHARDS="${TOTAL_SHARDS:-1}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
WORKERS="${WORKERS:-8}"
TEMPERATURE="${TEMPERATURE:-0.4}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
NETWORK_RETRIES="${NETWORK_RETRIES:-4}"

INPUT="${INPUT:-/path/to/scratch/infinigen/training_data_mix_all_balance/no_thinking/no_thinking.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/path/to/scratch/infinigen/training_data_mix_all_balance/text_thinking}"

VLLM_LOG="${LOG_DIR}/annotate_text_reasoning_vllm-${SLURM_JOB_ID:-local}.log"
SERVER_PID=""

mkdir -p "$LOG_DIR"

################################################################################
# UTILITIES
################################################################################

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

cleanup() {
    log "Cleaning up..."
    if [ -n "$SERVER_PID" ]; then
        log "Stopping vLLM server (PID: $SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=""
    fi
}

trap cleanup EXIT INT TERM

################################################################################
# STEP 1: Start vLLM server
################################################################################

NUM_GPUS="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
if [ "$NUM_GPUS" -lt 1 ]; then
    log "ERROR: no GPUs visible to this job."
    exit 1
fi

log "Starting vLLM server for $VLLM_MODEL on port $VLLM_PORT with $NUM_GPUS GPUs..."

vllm serve "$VLLM_MODEL" \
    --served-model-name "$VLLM_MODEL_NAME" \
    --port "$VLLM_PORT" \
    --host 0.0.0.0 \
    --trust-remote-code \
    --tensor-parallel-size "$NUM_GPUS" \
    --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
    --max-model-len "$VLLM_MAX_MODEL_LEN" \
    --max-num-seqs "$VLLM_MAX_NUM_SEQS" \
    --mm-encoder-tp-mode data \
    --dtype auto \
    --async-scheduling \
    --limit-mm-per-prompt.image 2 \
    --limit-mm-per-prompt.video 0 \
    --disable-log-stats \
    --disable-mm-preprocessor-cache \
    > "$VLLM_LOG" 2>&1 &

SERVER_PID=$!
log "vLLM server started with PID: $SERVER_PID"
log "vLLM log: $VLLM_LOG"

log "Waiting for vLLM server to be ready..."
READY=false
for i in $(seq 1 1800); do
    if curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
        READY=true
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        log "ERROR: vLLM server exited before becoming ready. Last log lines:"
        tail -n 80 "$VLLM_LOG" || true
        exit 1
    fi
    sleep 2
done

if [ "$READY" = false ]; then
    log "ERROR: vLLM server did not start within 20 minutes. Last log lines:"
    tail -n 80 "$VLLM_LOG" || true
    exit 1
fi
log "vLLM server is ready."

################################################################################
# STEP 2: Run annotation
################################################################################

API_BASE="http://localhost:${VLLM_PORT}/v1"
log "Starting text-reasoning annotation with API_BASE=$API_BASE"
log "Shard ${SHARD_INDEX}/${TOTAL_SHARDS}; workers=$WORKERS"

cd "$REPO_DIR"

ANNOTATE_ARGS=(
    --input "$INPUT"
    --output_dir "$OUTPUT_DIR"
    --api_base "$API_BASE"
    --model_name "$VLLM_MODEL_NAME"
    --shard "${SHARD_INDEX}/${TOTAL_SHARDS}"
    --workers "$WORKERS"
    --temperature "$TEMPERATURE"
    --max_tokens "$MAX_TOKENS"
    --network_retries "$NETWORK_RETRIES"
)

if [ -n "${MAX_SAMPLES:-}" ]; then
    ANNOTATE_ARGS+=(--max_samples "$MAX_SAMPLES")
fi

if [ "${NO_RESUME:-0}" = "1" ]; then
    ANNOTATE_ARGS+=(--no_resume)
fi

python "$SCRIPT_DIR/annotate_text_reasoning.py" "${ANNOTATE_ARGS[@]}"

log "Text-reasoning annotation complete."
log "Outputs under: $OUTPUT_DIR"
