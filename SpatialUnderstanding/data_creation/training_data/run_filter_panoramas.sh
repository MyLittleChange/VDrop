#!/bin/bash
#SBATCH --job-name=FilterPanoramas
#SBATCH --gres=gpu:a100l:2
#SBATCH -c 4
#SBATCH --partition=main
#SBATCH --mem=32Gb
#SBATCH --time=2:00:00
#SBATCH --output=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/data_creation/training_data/slurm_logs/filter_panoramas_output-%j.txt
#SBATCH --error=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/data_creation/training_data/slurm_logs/filter_panoramas_error-%j.txt

# ==============================================================================
# Filter bad panoramas using Qwen3-VL-32B via vLLM
# Writes bad_panoramas.json, then used by create_mix_all_sft_data.py
# ==============================================================================

set -e

module load cuda/12.6.0
unset ROCR_VISIBLE_DEVICES
source /path/to/home/verl/.venv/bin/activate

VLLM_MODEL="/path/to/scratch/models/Qwen3-VL-32B-Instruct"
VLLM_PORT=8766

SCRIPT_DIR="/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/data_creation/training_data"
LOG_DIR="${SCRIPT_DIR}/slurm_logs"
VLLM_LOG="${LOG_DIR}/vllm_filter_panoramas-${SLURM_JOB_ID}.log"

mkdir -p "$LOG_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

cleanup() {
    log "Cleaning up..."
    if [ -n "$SERVER_PID" ]; then
        log "Stopping vLLM server (PID: $SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true
        sleep 5
    fi
}

trap cleanup EXIT INT TERM

################################################################################
# Step 1: Start vLLM server (1 GPU)
################################################################################

log "Starting vLLM server for $VLLM_MODEL on port $VLLM_PORT..."

vllm serve "$VLLM_MODEL" \
    --port $VLLM_PORT \
    --trust-remote-code \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 8192 \
    --disable-log-stats \
    --disable-mm-preprocessor-cache \
    > "$VLLM_LOG" 2>&1 &

SERVER_PID=$!
log "vLLM server started with PID: $SERVER_PID"

# Wait for server to be ready
log "Waiting for vLLM server to be ready..."
READY=false
for i in $(seq 1 300); do
    if curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
        READY=true
        break
    fi
    sleep 2
done

if [ "$READY" = false ]; then
    log "ERROR: vLLM server did not start within 10 minutes."
    exit 1
fi
log "vLLM server is ready."

################################################################################
# Step 2: Run panorama filter
################################################################################

log "Filtering panoramas..."

python "$SCRIPT_DIR/filter_panoramas_vlm.py" \
    --api_base "http://localhost:${VLLM_PORT}/v1" \
    --output_file /path/to/scratch/infinigen/bad_panoramas.json \
    --workers 4

log "Panorama filtering complete."
log "Results: /path/to/scratch/infinigen/bad_panoramas.json"
