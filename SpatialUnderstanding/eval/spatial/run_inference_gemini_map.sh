#!/bin/bash
#SBATCH --job-name=gemini_map_infer
#SBATCH --output=/path/to/ThinkMorph-BAGEL-release/run_inference_gemini_map_output.txt
#SBATCH --error=/path/to/ThinkMorph-BAGEL-release/run_inference_gemini_map_error.txt
#SBATCH --ntasks=1

set -e  # Exit on error

VENV_PATH="/path/to/home/verl/.venv/bin/activate"
source "$VENV_PATH"

################################################################################
# CONFIGURATION - Edit these settings
################################################################################

# Gemini model configuration
MODEL_NAME="gemini-3-pro-preview"
THINKING_LEVEL="high"  # Options: low, high

# Data path
DATA_FILE="/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_dataset_map_questions_normalized.json"

# Output directory
OUTPUT_DIR="/path/to/scratch/spatial_collab/Gemini"

# Panorama mode: set to "true" to include panorama as additional input (4 images total)
USE_PANORAMA="true"

# Inference parameters
NUM_SAMPLES=""  # Leave empty to run all samples
TEMPERATURE=0.3
MAX_TOKENS=26384  # Needs to be high when thinking is enabled (thinking tokens count against this limit)
MAX_WORKERS=64  # Adjust based on API rate limits

################################################################################
# DO NOT EDIT BELOW THIS LINE
################################################################################

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

################################################################################
# Main Script
################################################################################

log "=============================================="
log "Gemini Map Verification Inference Script"
log "=============================================="
log "Model: $MODEL_NAME"
log "Thinking Level: $THINKING_LEVEL"
log "Data File: $DATA_FILE"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Generate output filename
DATASET_BASENAME=$(basename "$DATA_FILE" .json)
MODEL_SHORT_NAME=$(echo "$MODEL_NAME" | sed 's/[^a-zA-Z0-9]/_/g')
OUTPUT_FILE="${OUTPUT_DIR}/${MODEL_SHORT_NAME}_${DATASET_BASENAME}.json"

log "Output: $OUTPUT_FILE"

NUM_SAMPLES_ARG=""
if [ -n "$NUM_SAMPLES" ]; then
    NUM_SAMPLES_ARG="--num_samples $NUM_SAMPLES"
fi

PANORAMA_ARG=""
if [ "$USE_PANORAMA" = "true" ]; then
    PANORAMA_ARG="--use_panorama"
    log "Panorama mode: ENABLED (4 images: 2 views + panorama + map)"
else
    log "Panorama mode: DISABLED (3 images: 2 views + map)"
fi

log "=============================================="
log "Running inference..."
log "=============================================="

python /path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/eval/spatial/run_inference_gemini_map.py \
    --data_file "$DATA_FILE" \
    --model_name "$MODEL_NAME" \
    --thinking_level "$THINKING_LEVEL" \
    --output_file "$OUTPUT_FILE" \
    $NUM_SAMPLES_ARG \
    $PANORAMA_ARG \
    --temperature $TEMPERATURE \
    --max_tokens $MAX_TOKENS \
    --max_workers $MAX_WORKERS

INFERENCE_EXIT_CODE=$?

if [ $INFERENCE_EXIT_CODE -eq 0 ]; then
    log "Inference completed successfully!"
    log "Results saved to: $OUTPUT_FILE"
else
    log "WARNING: Inference failed with exit code $INFERENCE_EXIT_CODE"
    exit 1
fi
