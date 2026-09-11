#!/bin/bash
#SBATCH --job-name=gemini_infer
#SBATCH --output=/path/to/ThinkMorph-BAGEL-release/run_inference_gemini_output.txt
#SBATCH --error=/path/to/ThinkMorph-BAGEL-release/run_inference_gemini_error.txt
#SBATCH --ntasks=1

set -e  # Exit on error

VENV_PATH="/path/to/home/verl/.venv/bin/activate"
source "$VENV_PATH"

################################################################################
# CONFIGURATION - Edit these settings
################################################################################

# Gemini model configuration
MODEL_NAME="gemini-3-pro-preview"
THINKING_LEVEL="high"  # Options:  low, high

# Data paths
DATA_DIR="/path/to/scratch/VisualCoT/spatial_collab_dataset"

# List of dataset files to test
DATASET_FILES=(
    "approved_mcqs_anchor_normalized.json"
    "approved_mcqs_counting_normalized.json"
    "approved_mcqs_relative_distance_normalized.json"
)

# Output directory
OUTPUT_DIR="/path/to/scratch/spatial_collab/Gemini"

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
log "Gemini Spatial Collab Inference Script"
log "=============================================="
log "Model: $MODEL_NAME"
log "Thinking Level: $THINKING_LEVEL"

# Create output directory
mkdir -p "$OUTPUT_DIR"

################################################################################
# Run inference for each dataset
################################################################################

log "=============================================="
log "Running inference on ${#DATASET_FILES[@]} datasets"
log "=============================================="

NUM_SAMPLES_ARG=""
if [ -n "$NUM_SAMPLES" ]; then
    NUM_SAMPLES_ARG="--num_samples $NUM_SAMPLES"
fi


TOTAL_DATASETS=${#DATASET_FILES[@]}
CURRENT=0
FAILED_DATASETS=()

for DATASET_FILE in "${DATASET_FILES[@]}"; do
    CURRENT=$((CURRENT + 1))

    # Generate output filename based on dataset name
    DATASET_BASENAME=$(basename "$DATASET_FILE" .json)
    MODEL_SHORT_NAME=$(echo "$MODEL_NAME" | sed 's/[^a-zA-Z0-9]/_/g')
    OUTPUT_FILE="${OUTPUT_DIR}/${MODEL_SHORT_NAME}_${DATASET_BASENAME}.json"

    log "=============================================="
    log "Processing dataset [$CURRENT/$TOTAL_DATASETS]: $DATASET_FILE"
    log "Output: $OUTPUT_FILE"
    log "=============================================="

    python /path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/eval/spatial/run_inference_gemini.py \
        --data_dir "$DATA_DIR" \
        --dataset_files "$DATASET_FILE" \
        --model_name "$MODEL_NAME" \
        --thinking_level "$THINKING_LEVEL" \
        --output_file "$OUTPUT_FILE" \
        $NUM_SAMPLES_ARG \
        --temperature $TEMPERATURE \
        --max_tokens $MAX_TOKENS \
        --max_workers $MAX_WORKERS

    INFERENCE_EXIT_CODE=$?

    if [ $INFERENCE_EXIT_CODE -eq 0 ]; then
        log "Dataset $DATASET_FILE completed successfully!"
        log "Results saved to: $OUTPUT_FILE"
    else
        log "WARNING: Dataset $DATASET_FILE failed with exit code $INFERENCE_EXIT_CODE"
        FAILED_DATASETS+=("$DATASET_FILE")
    fi

    log ""
done

################################################################################
# Summary
################################################################################

log "=============================================="
log "All datasets processed!"
log "=============================================="
log "Total datasets: $TOTAL_DATASETS"
log "Failed datasets: ${#FAILED_DATASETS[@]}"

if [ ${#FAILED_DATASETS[@]} -gt 0 ]; then
    log "Failed datasets list:"
    for FAILED in "${FAILED_DATASETS[@]}"; do
        log "  - $FAILED"
    done
    exit 1
else
    log "All datasets completed successfully!"
fi
