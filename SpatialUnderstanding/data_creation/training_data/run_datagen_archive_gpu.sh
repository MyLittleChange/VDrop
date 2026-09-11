#!/bin/bash
#SBATCH --job-name=DatagenArchiveGPU
#SBATCH --gres=gpu:a100l:4
#SBATCH -c 8
#SBATCH --mem=120Gb
#SBATCH --time=2:00:00
#SBATCH --output=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/data_creation/training_data/slurm_logs/archive_gpu_output-%j.txt
#SBATCH --error=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/data_creation/training_data/slurm_logs/archive_gpu_error-%j.txt

# ==============================================================================
# GPU stage: scene_llm_visible_objects (Qwen3-VL-235B via vLLM)
# Run AFTER run_datagen_archive_pregpu.sh and BEFORE run_datagen_archive_postgpu.sh.
# Usage: sbatch run_datagen_archive_gpu.sh
# ==============================================================================

set -e

module load cuda/12.6.0
unset ROCR_VISIBLE_DEVICES
source /path/to/home/verl/.venv/bin/activate

################################################################################
# CONFIGURATION
################################################################################

BASE_DIR="/path/to/scratch/infinigen/spatial"
SCENE_DATAFILE="/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/data_creation/training_data/dataset_archive_scenes_v1.json"
QUESTION_VERSION="V4"

VLLM_MODEL="/path/to/scratch/models/Qwen3-VL-235B-A22B-Instruct-FP8"
VLLM_PORT=8765

SCRIPT_DIR="/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/data_creation/training_data"
LOG_DIR="${SCRIPT_DIR}/slurm_logs"
VLLM_LOG="${LOG_DIR}/vllm_server-${SLURM_JOB_ID}.log"

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
        sleep 5
    fi
    rm -f "$DUMMY_KEY_FILE"
}

trap cleanup EXIT INT TERM

################################################################################
# STEP 1: Start vLLM server (4 GPUs, tensor-parallel)
################################################################################

log "Starting vLLM server for $VLLM_MODEL on port $VLLM_PORT..."

vllm serve "$VLLM_MODEL" \
    --port $VLLM_PORT \
    --trust-remote-code \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 32768 \
    --disable-log-stats \
    --disable-mm-preprocessor-cache \
    > "$VLLM_LOG" 2>&1 &

SERVER_PID=$!
log "vLLM server started with PID: $SERVER_PID"

# Wait for server to be ready (up to 10 minutes)
log "Waiting for vLLM server to be ready..."
READY=false
for i in $(seq 1 600); do
    if curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
        READY=true
        break
    fi
    sleep 2
done

if [ "$READY" = false ]; then
    log "ERROR: vLLM server did not start within 20 minutes."
    exit 1
fi
log "vLLM server is ready."

################################################################################
# STEP 2: Run GPU stage
################################################################################

log "Starting GPU stage: scene_llm_visible_objects..."

DUMMY_KEY_FILE="/tmp/dummy_key_${SLURM_JOB_ID}.txt"
echo "dummy" > "$DUMMY_KEY_FILE"

export PYTHONPATH="/path/to/ThinkMorph-BAGEL-release/MultiAgent_Spatial:${PYTHONPATH}"
cd "$SCRIPT_DIR"

python datagen_pipeline.py \
    --base_dir "$BASE_DIR" \
    --scene_datafile "$SCENE_DATAFILE" \
    --stages_to_run \
        scene_llm_visible_objects \
    --question_version "$QUESTION_VERSION" \
    --client_vis_objects vllm \
    --model_name_vis_objects "$VLLM_MODEL" \
    --api_base_vis_objects "http://localhost:${VLLM_PORT}/v1" \
    --api_key "$DUMMY_KEY_FILE"

log "GPU stage complete."
log "Next step: sbatch run_datagen_archive_postgpu.sh"
