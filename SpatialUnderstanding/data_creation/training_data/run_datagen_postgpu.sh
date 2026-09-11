#!/bin/bash
#SBATCH --job-name=DatagenPostGPU
#SBATCH --partition=long
#SBATCH -c 4
#SBATCH --mem=32Gb
#SBATCH --time=12:00:00
#SBATCH --output=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/data_creation/training_data/slurm_logs/postgpu_output-%j.txt
#SBATCH --error=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/data_creation/training_data/slurm_logs/postgpu_error-%j.txt

# ==============================================================================
# Post-GPU stages: scene_bound_objects, scene_obj_color_info,
#                  scene_generate_descriptions, scene_solve_perception,
#                  scene_generate_questions, scene_generate_maps,
#                  scene_generate_paraphrase, aggregate_data, filter_questions
# Requires: VisualCoT_GEMINI env var set before submitting.
# Run AFTER run_datagen_gpu.sh.
# Usage: sbatch run_datagen_postgpu.sh
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
# RUN POST-GPU STAGES
################################################################################

log "Starting post-GPU datagen stages..."
log "Base dir: $BASE_DIR"
log "Scene datafile: $SCENE_DATAFILE"

export PYTHONPATH="/path/to/ThinkMorph-BAGEL-release/MultiAgent_Spatial:${PYTHONPATH}"
export PATH="/path/to/scratch/blender-4.2.0-linux-x64:${PATH}"
cd "$SCRIPT_DIR"

python datagen_pipeline.py \
    --base_dir "$BASE_DIR" \
    --scene_datafile "$SCENE_DATAFILE" \
    --stages_to_run \
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
    --api_key "$GEMINI_KEY_FILE"

log "Post-GPU stages complete."
log "Check dataset_*_questions_filtered_${QUESTION_VERSION}.json in $SCRIPT_DIR"
