#!/bin/bash
#SBATCH --job-name=gemini_vsibench
#SBATCH --output=/path/to/ThinkMorph-BAGEL-release/run_inference_gemini_vsibench_output.txt
#SBATCH --error=/path/to/ThinkMorph-BAGEL-release/run_inference_gemini_vsibench_error.txt
#SBATCH --ntasks=1
#SBATCH --mem=40G
set -e  # Exit on error

VENV_PATH="/path/to/home/verl/.venv/bin/activate"
source "$VENV_PATH"

################################################################################
# CONFIGURATION - Edit these settings
################################################################################

# Gemini model configuration
MODEL_NAME="gemini-3-pro-preview"
THINKING_LEVEL="high"  # Options: low, high

# Data paths
DATA_DIR="/path/to/scratch/datasets/VSI-Bench"
DATASET_FILE="test_debiased.parquet"

# Output
OUTPUT_FILE="/path/to/scratch/vsibench/Gemini/inference_results_gemini_vsibench.json"

# Inference parameters
NUM_SAMPLES=""  # Leave empty to run all samples
TEMPERATURE=0.3
MAX_TOKENS=26384  # Needs to be high when thinking is enabled
MAX_WORKERS=32    # Adjust based on API rate limits
NUM_FRAMES=8

# Environment
SCRIPT_DIR="/path/to/ThinkMorph-BAGEL-release"

################################################################################
# DO NOT EDIT BELOW THIS LINE
################################################################################

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

log "=============================================="
log "Gemini VSI-Bench Inference Script"
log "=============================================="
log "Model: $MODEL_NAME"
log "Thinking Level: $THINKING_LEVEL"
log "Data Dir: $DATA_DIR"
log "Dataset: $DATASET_FILE"
log "Output: $OUTPUT_FILE"
log "Frames: $NUM_FRAMES"

mkdir -p "$(dirname "$OUTPUT_FILE")"

cd "$SCRIPT_DIR"

NUM_SAMPLES_ARG=""
if [ -n "$NUM_SAMPLES" ]; then
    NUM_SAMPLES_ARG="--num_samples $NUM_SAMPLES"
fi

python run_inference_gemini_vsibench.py \
    --data_dir "$DATA_DIR" \
    --dataset_file "$DATASET_FILE" \
    --model_name "$MODEL_NAME" \
    --thinking_level "$THINKING_LEVEL" \
    --output_file "$OUTPUT_FILE" \
    $NUM_SAMPLES_ARG \
    --temperature $TEMPERATURE \
    --max_tokens $MAX_TOKENS \
    --max_workers $MAX_WORKERS \
    --num_frames $NUM_FRAMES

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    log "Inference completed successfully!"
    log "Results saved to: $OUTPUT_FILE"
else
    log "Inference failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi
