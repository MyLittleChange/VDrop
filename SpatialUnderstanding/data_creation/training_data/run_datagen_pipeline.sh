#!/bin/bash
#SBATCH --job-name=DatagenPipeline
#SBATCH --partition=short-unkillable
#SBATCH --gres=gpu:a100l:4
#SBATCH -c 16
#SBATCH --mem=128Gb
#SBATCH --time=3:00:00
#SBATCH --output=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/data_creation/training_data/slurm_logs/datagen_output-%j.txt
#SBATCH --error=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/data_creation/training_data/slurm_logs/datagen_error-%j.txt

# ==============================================================================
# Full 14-stage QA generation pipeline for rendered Infinigen scenes.
# Requires: GEMINI_API_KEY env var set before submitting.
# Usage: sbatch run_datagen_pipeline.sh
# ==============================================================================

set -e

module load cuda/12.6.0
unset ROCR_VISIBLE_DEVICES
source /path/to/scratch/morph_env/bin/activate

################################################################################
# CONFIGURATION
################################################################################

BASE_DIR="/path/to/scratch/infinigen/outputs_rendered"
SCENE_DATAFILE="/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/data_creation/training_data/dataset_new_scenes_v1.json"
QUESTION_VERSION="V4"

GEMINI_MODEL="gemini-2.5-flash-preview-04-17"
GEMINI_API_BASE="https://generativelanguage.googleapis.com/v1beta/openai/"

VLLM_MODEL="Qwen/Qwen3-VL-235B-A22B-Instruct-FP8"
VLLM_PORT=8765
VLLM_LOG="/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/data_creation/training_data/slurm_logs/vllm_server-${SLURM_JOB_ID}.log"

SCRIPT_DIR="/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/data_creation/training_data"
LOG_DIR="${SCRIPT_DIR}/slurm_logs"

mkdir -p "$LOG_DIR"

################################################################################
# API KEY SETUP
################################################################################

if [ -z "$VisualCoT_GEMINI" ]; then
    echo "ERROR: VisualCoT_GEMINI env var is not set. Make sure ~/.bashrc exports it."
    exit 1
fi

GEMINI_KEY_FILE="/tmp/gemini_key_${SLURM_JOB_ID}.txt"
echo "$VisualCoT_GEMINI" > "$GEMINI_KEY_FILE"
export OPENAI_API_KEY="$VisualCoT_GEMINI"

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
    rm -f "$GEMINI_KEY_FILE"
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
    --disable-log-stats \
    --disable-mm-preprocessor-cache \
    > "$VLLM_LOG" 2>&1 &

SERVER_PID=$!
log "vLLM server started with PID: $SERVER_PID"

# Wait for server to be ready (up to 10 minutes)
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
# STEP 2: Run datagen pipeline
################################################################################

log "Starting datagen pipeline..."
log "Base dir: $BASE_DIR"
log "Scene datafile: $SCENE_DATAFILE"

cd "$SCRIPT_DIR"

python datagen_pipeline.py \
    --base_dir "$BASE_DIR" \
    --scene_datafile "$SCENE_DATAFILE" \
    --stages_to_run \
        find_all_scenes \
        scene_filtering \
        scene_object_info \
        scene_camera_info \
        scene_blender_color_info \
        scene_llm_visible_objects \
        scene_bound_objects \
        scene_obj_color_info \
        scene_generate_descriptions \
        scene_solve_perception \
        scene_generate_questions \
        scene_generate_maps \
        scene_generate_paraphrase \
        aggregate_data \
        filter_questions \
    --question_version "$QUESTION_VERSION" \
    --client_scene_filtering openai \
    --model_name_scene_filtering "$GEMINI_MODEL" \
    --api_base_scene_filtering "$GEMINI_API_BASE" \
    --client_color openai \
    --model_name_color "$GEMINI_MODEL" \
    --api_base_color "$GEMINI_API_BASE" \
    --client_paraphrase openai \
    --model_name_paraphrase "$GEMINI_MODEL" \
    --api_base_paraphrase "$GEMINI_API_BASE" \
    --client_vis_objects vllm \
    --model_name_vis_objects "$VLLM_MODEL" \
    --api_base_vis_objects "http://localhost:${VLLM_PORT}/v1" \
    --api_key "$GEMINI_KEY_FILE"

log "Datagen pipeline complete."
