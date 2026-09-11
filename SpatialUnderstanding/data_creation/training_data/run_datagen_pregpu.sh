#!/bin/bash
#SBATCH --job-name=DatagenPreGPU
#SBATCH --partition=main
#SBATCH -c 4
#SBATCH --mem=32Gb
#SBATCH --time=24:00:00
#SBATCH --output=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/data_creation/training_data/slurm_logs/pregpu_output-%j.txt
#SBATCH --error=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/data_creation/training_data/slurm_logs/pregpu_error-%j.txt

# ==============================================================================
# Pre-GPU stages: find_all_scenes, scene_filtering, scene_object_info,
#                 scene_camera_info, scene_blender_color_info
# Requires: VisualCoT_GEMINI env var set before submitting.
# Run BEFORE run_datagen_gpu.sh.
# Usage: sbatch run_datagen_pregpu.sh
# ==============================================================================

set -e

source /path/to/scratch/morph_env/bin/activate

################################################################################
# CONFIGURATION
################################################################################

BASE_DIR="/path/to/scratch/infinigen/outputs_rendered"
SCENE_DATAFILE="/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/data_creation/training_data/dataset_new_scenes_v1.json"
QUESTION_VERSION="V4"

GEMINI_MODEL="gemini-3-flash-preview"
GEMINI_API_BASE="https://generativelanguage.googleapis.com/v1beta/openai/"

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
    rm -f "$GEMINI_KEY_FILE"
}

trap cleanup EXIT INT TERM

################################################################################
# RUN PRE-GPU STAGES
################################################################################

log "Starting pre-GPU datagen stages..."
log "Base dir: $BASE_DIR"
log "Scene datafile: $SCENE_DATAFILE"

export PYTHONPATH="/path/to/ThinkMorph-BAGEL-release/MultiAgent_Spatial:${PYTHONPATH}"
export PATH="/path/to/scratch/blender-4.2.0-linux-x64:${PATH}"
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
    --api_key "$GEMINI_KEY_FILE"

log "Pre-GPU stages complete."
log "Next step: sbatch run_datagen_gpu.sh"
