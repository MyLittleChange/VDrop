#!/bin/bash
#SBATCH --job-name=bagel_cv_1gpu
#SBATCH --output=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/bagel/bagel_center_view_1gpu_output.txt
#SBATCH --error=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/bagel/bagel_center_view_1gpu_error.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --partition=unkillable

#
# BAGEL Center View Generation — 1 GPU on unkillable
#
# Runs all shards sequentially on a single GPU.
#
set -euo pipefail

SCRIPT_DIR="/path/to/ThinkMorph-BAGEL-release"

# ==================== Configuration ====================
# MODEL_PATH="${MODEL_PATH:-/path/to/scratch/VisualCoT/BAGEL_checkpoints/BAGEL_format_2000}"

MODEL_PATH="${MODEL_PATH:-/path/to/scratch/VisualCoT/BAGEL_checkpoints/BAGEL_format_visual_only}"

ANNOTATION_FILE="${ANNOTATION_FILE:-/path/to/scratch/VisualCoT/annotations/gemini_3_flash_preview_orbit_spatial_test_annotations_no_angle_parsed_qa.json}"

OUTPUT_DIR="${OUTPUT_DIR:-/path/to/scratch/VisualCoT/parsed_qa_anchor_center_spatial_rear_mcqa_visual_only_spatial_test_set_center_view_parsed_qa}"

MAX_MEM_PER_GPU="${MAX_MEM_PER_GPU:-70GiB}"

SEED="${SEED:-42}"

THINK="${THINK:-true}"

NUM_SAMPLES="${NUM_SAMPLES:-50}"

# ==================== Print Configuration ====================
echo "=============================================="
echo "BAGEL Center View Generation (1-GPU)"
echo "=============================================="
echo "Configuration:"
echo "  Model Path:       $MODEL_PATH"
echo "  Annotation File:  $ANNOTATION_FILE"
echo "  Output Dir:       $OUTPUT_DIR"
echo "  Max Mem/GPU:      $MAX_MEM_PER_GPU"
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
unset ROCR_VISIBLE_DEVICES
source /path/to/scratch/morph_env/bin/activate

echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo ""

mkdir -p "$OUTPUT_DIR"

# ==================== Build flags ====================
THINK_FLAG=""
if [[ "$THINK" == "false" ]]; then
    THINK_FLAG="--no_think"
fi

NUM_SAMPLES_FLAG=""
if [[ -n "$NUM_SAMPLES" ]]; then
    NUM_SAMPLES_FLAG="--num_samples $NUM_SAMPLES"
fi

# ==================== Run on 1 GPU ====================
echo "Running all samples on GPU 0..."

CUDA_VISIBLE_DEVICES=0 python3 "${SCRIPT_DIR}/SpatialUnderstanding/test_bagel_center_view.py" \
    --model_path "$MODEL_PATH" \
    --annotation_file "$ANNOTATION_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --max_mem_per_gpu "$MAX_MEM_PER_GPU" \
    --random_seed "$SEED" \
    $THINK_FLAG \
    $NUM_SAMPLES_FLAG

echo "=============================================="
echo "Done!"
echo "=============================================="
echo "Results saved to: $OUTPUT_DIR"
