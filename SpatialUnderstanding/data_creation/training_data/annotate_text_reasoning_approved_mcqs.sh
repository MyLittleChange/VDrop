#!/bin/bash
#SBATCH --job-name=anno_cot_mcqs
#SBATCH --partition=short-unkillable
#SBATCH --gres=gpu:a100l:4
#SBATCH -c 24
#SBATCH --output=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/data_creation/training_data/slurm_logs/anno_cot_mcqs-%A_%a.txt
#SBATCH --error=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/data_creation/training_data/slurm_logs/anno_cot_mcqs_error-%A_%a.txt
#SBATCH --ntasks=1
#SBATCH --time=3:00:00
#SBATCH --mem=512Gb

# T_cot annotation for the COSMIC test split (Two-Reader Informativeness experiment).
# Spins up Qwen3-VL-235B-FP8 on this allocation, runs annotate_text_reasoning_approved_mcqs.py,
# stops the server on exit. Resumable: rerun and it picks up where it left off.

set -euo pipefail

module load cuda/12.6.0
unset ROCR_VISIBLE_DEVICES
source /path/to/home/verl/.venv/bin/activate

SCRIPT_DIR="/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/data_creation/training_data"
LOG_DIR="${SCRIPT_DIR}/slurm_logs"

VLLM_MODEL="${VLLM_MODEL:-/path/to/scratch/models/Qwen3-VL-235B-A22B-Instruct-FP8}"
VLLM_MODEL_NAME="${VLLM_MODEL_NAME:-qwen3_vl_235b_fp8}"
VLLM_PORT="${VLLM_PORT:-8765}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-16384}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.95}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-16}"

WORKERS="${WORKERS:-8}"
TEMPERATURE="${TEMPERATURE:-0.4}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
SUBTASK="${SUBTASK:-}"   # blank = all four
OUTPUT_DIR="${OUTPUT_DIR:-/path/to/scratch/VisualCoT/infinigen/two_reader_informativeness/text_thinking_approved_mcqs}"

VLLM_LOG="${LOG_DIR}/anno_cot_mcqs_vllm-${SLURM_JOB_ID:-local}.log"
SERVER_PID=""

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

cleanup() {
    log "Cleaning up..."
    if [ -n "$SERVER_PID" ]; then
        log "Stopping vLLM server (PID: $SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

NUM_GPUS="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
if [ "$NUM_GPUS" -lt 1 ]; then
    log "ERROR: no GPUs visible."
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
log "vLLM server PID: $SERVER_PID, log: $VLLM_LOG"

log "Waiting for vLLM server to be ready..."
READY=false
for i in $(seq 1 1800); do
    if curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
        READY=true; break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        log "ERROR: vLLM exited before ready. Tail:"
        tail -n 80 "$VLLM_LOG" || true
        exit 1
    fi
    sleep 2
done
[ "$READY" = false ] && { log "ERROR: vLLM did not start within 1h."; tail -n 80 "$VLLM_LOG" || true; exit 1; }
log "vLLM ready."

API_BASE="http://localhost:${VLLM_PORT}/v1"
ARGS=(
    --api_base "$API_BASE"
    --model_name "$VLLM_MODEL_NAME"
    --output_dir "$OUTPUT_DIR"
    --workers "$WORKERS"
    --temperature "$TEMPERATURE"
    --max_tokens "$MAX_TOKENS"
)
[ -n "$SUBTASK" ] && ARGS+=(--subtask "$SUBTASK")

log "Annotating with: ${ARGS[*]}"
python "$SCRIPT_DIR/annotate_text_reasoning_approved_mcqs.py" "${ARGS[@]}"

log "Done. Outputs under: $OUTPUT_DIR"
