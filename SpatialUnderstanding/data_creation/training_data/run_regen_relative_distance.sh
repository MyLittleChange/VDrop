#!/bin/bash
#SBATCH --job-name=regen_reldist
#SBATCH --output=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/data_creation/training_data/slurm_logs/regen_reldist_output-%j.txt
#SBATCH --error=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/data_creation/training_data/slurm_logs/regen_reldist_error-%j.txt
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
# ==============================================================================
# Regenerate ONLY relative_distance questions with dist_threshold=0.5,
# patch into existing questions_paraphrased.json, then re-run aggregate + filter.
#
# Requires: VisualCoT_GEMINI env var set before submitting.
# Usage:
#   sbatch run_regen_relative_distance.sh                          # V5 (spatial/)
#   sbatch run_regen_relative_distance.sh outputs_rendered V4      # V4
# ==============================================================================

BASE_NAME="${1:-spatial}"   # "spatial" or "outputs_rendered"
QUESTION_VERSION="${2:-V5}"

SCRATCH="/path/to/scratch/infinigen"
BASE_DIR="$SCRATCH/$BASE_NAME"

SCRIPT_DIR="/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/data_creation/training_data"

if [ "$QUESTION_VERSION" = "V5" ]; then
    SCENE_DATAFILE="$SCRIPT_DIR/dataset_new_scenes_v5.json"
else
    SCENE_DATAFILE="$SCRIPT_DIR/dataset_new_scenes_v1.json"
fi
REPO_ROOT="/path/to/ThinkMorph-BAGEL-release"

GEMINI_MODEL="gemini-3-flash-preview"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# --- Validate env ---
if [ -z "$VisualCoT_GEMINI" ]; then
    echo "ERROR: VisualCoT_GEMINI env var not set."
    exit 1
fi

GEMINI_KEY_FILE="/tmp/gemini_key_reldist_${SLURM_JOB_ID}.txt"
echo "$VisualCoT_GEMINI" > "$GEMINI_KEY_FILE"

cleanup() { rm -f "$GEMINI_KEY_FILE"; }
trap cleanup EXIT

# --- Activate env ---
source /path/to/scratch/morph_env/bin/activate

export PYTHONPATH="$REPO_ROOT/MultiAgent_Spatial:${PYTHONPATH}"
cd "$REPO_ROOT"

log "Regenerating relative_distance questions"
log "BASE_DIR:         $BASE_DIR"
log "QUESTION_VERSION: $QUESTION_VERSION"
log "dist_threshold:   0.5"

python "$SCRIPT_DIR/regen_relative_distance.py" \
    --base_dir "$BASE_DIR" \
    --question_version "$QUESTION_VERSION" \
    --scene_datafile "$SCENE_DATAFILE" \
    --dist_threshold 0.5 \
    --model_name_paraphrase "$GEMINI_MODEL" \
    --api_key "$GEMINI_KEY_FILE"

log "Done. Check slurm_logs/ for results."
log "Verify: dataset_relative_distance_questions_filtered_${QUESTION_VERSION}.json"
