#!/bin/bash
#SBATCH --job-name=bagel_no_train_center_view
#SBATCH --output=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/bagel/bagel_center_view_output_%a.txt
#SBATCH --error=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/bagel/bagel_center_view_error_%a.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --array=0-23

#
# BAGEL Center View Generation Test Script
#
# Tests BAGEL's ability to generate center view images given two input views
# and a spatial annotation. Uses job arrays for parallel execution.
#
# Usage:
#   # Submit all shards as a job array
#   sbatch run_bagel_center_view.sh
#
#   # Run a single shard interactively
#   SLURM_ARRAY_TASK_ID=0 bash run_bagel_center_view.sh
#
#   # Quick test with 5 samples on shard 0
#   NUM_SAMPLES=5 SLURM_ARRAY_TASK_ID=0 bash run_bagel_center_view.sh
#

set -euo pipefail

SCRIPT_DIR="/path/to/ThinkMorph-BAGEL-release"

# ==================== Configuration ====================
MODEL_PATH="${MODEL_PATH:-/path/to/scratch/models/BAGEL-7B-MoT}"

ANNOTATION_FILE="${ANNOTATION_FILE:-/path/to/scratch/VisualCoT/annotations/gemini_3_flash_preview_rendered_v3_anchor_annotations_center_view_no_angle.json}"

OUTPUT_DIR="${OUTPUT_DIR:-/path/to/scratch/VisualCoT/BAGEL_center_view_eval}"

MAX_MEM_PER_GPU="${MAX_MEM_PER_GPU:-70GiB}"

# Number of shards (should match --array range, e.g., 0-47 means 48 shards)
NUM_SHARDS="${NUM_SHARDS:-24}"

SEED="${SEED:-42}"

THINK="${THINK:-true}"

# Set to a number to limit samples, or leave empty for all
NUM_SAMPLES="${NUM_SAMPLES:-}"

# ==================== Determine Shard ====================
SHARD_IDX="${SLURM_ARRAY_TASK_ID:-0}"
SHARD_SPEC="${SHARD_IDX}/${NUM_SHARDS}"

# ==================== Print Configuration ====================
echo "=============================================="
echo "BAGEL Center View Generation Test"
echo "=============================================="
echo "Configuration:"
echo "  Model Path:       $MODEL_PATH"
echo "  Annotation File:  $ANNOTATION_FILE"
echo "  Output Dir:       $OUTPUT_DIR"
echo "  Max Mem/GPU:      $MAX_MEM_PER_GPU"
echo "  Shard:            $SHARD_SPEC"
echo "  Think Mode:       $THINK"
echo "  Seed:             $SEED"
echo "  Num Samples:      ${NUM_SAMPLES:-all}"
echo "=============================================="
echo ""

# ==================== Validation ====================
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "ERROR: Model directory not found: $MODEL_PATH"
    exit 1
fi

if [[ ! -f "$ANNOTATION_FILE" ]]; then
    echo "ERROR: Annotation file not found: $ANNOTATION_FILE"
    exit 1
fi

# ==================== Environment Setup ====================
module load cuda/12.6.0
source /path/to/scratch/morph_env/bin/activate

echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo ""

# ==================== Create Output Directory ====================
mkdir -p "$OUTPUT_DIR"

# ==================== Build Command ====================
THINK_FLAG=""
if [[ "$THINK" == "false" ]]; then
    THINK_FLAG="--no_think"
fi

NUM_SAMPLES_FLAG=""
if [[ -n "$NUM_SAMPLES" ]]; then
    NUM_SAMPLES_FLAG="--num_samples $NUM_SAMPLES"
fi

# ==================== Run Inference ====================
echo "Starting center view generation for shard ${SHARD_IDX}/${NUM_SHARDS}..."
echo ""

python3 "${SCRIPT_DIR}/SpatialUnderstanding/test_bagel_center_view.py" \
    --model_path "$MODEL_PATH" \
    --annotation_file "$ANNOTATION_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --max_mem_per_gpu "$MAX_MEM_PER_GPU" \
    --random_seed "$SEED" \
    --shard "$SHARD_SPEC" \
    $THINK_FLAG \
    $NUM_SAMPLES_FLAG

echo ""
echo "=============================================="
echo "Shard ${SHARD_IDX}/${NUM_SHARDS} Complete!"
echo "=============================================="
echo "Results saved to: $OUTPUT_DIR"
