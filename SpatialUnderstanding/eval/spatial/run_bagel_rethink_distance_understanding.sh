#!/bin/bash
#SBATCH --job-name=rethink_und_distance
#SBATCH --output=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/bagel/rethink_und_distance_output.txt
#SBATCH --error=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/bagel/rethink_und_distance_error.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=unkillable

#
# Understanding Rethink Inference — 1 GPU on unkillable
#
# Feeds pre-generated thinking images back as 3rd input (no new generation).
#
# Usage:
#   sbatch run_bagel_rethink_distance_understanding.sh
#
set -euo pipefail

SCRIPT_DIR="/path/to/ThinkMorph-BAGEL-release"

# ==================== Configuration ====================
MODEL_PATH="${MODEL_PATH:-/path/to/scratch/VisualCoT/BAGEL_checkpoints/BAGEL_format_training_data_mix_all_balance_understanding}"

DATA_DIR="${DATA_DIR:-/path/to/scratch/VisualCoT/spatial_collab_dataset}"
DATASET_FILES="${DATASET_FILES:-approved_mcqs_relative_distance_normalized.json}"

# Prior inference results containing saved generated images
RESULTS_FILE="${RESULTS_FILE:-/path/to/scratch/VisualCoT/BAGEL_format_training_data_mix_all_rotation_balance_visual_only_mcqs_relative_distance_normalized/inference_results_bagel_merged.json}"

OUTPUT_DIR="${OUTPUT_DIR:-/path/to/scratch/VisualCoT/rethink_understanding_mcqs_relative_distance_normalized}"

MAX_MEM_PER_GPU="${MAX_MEM_PER_GPU:-70GiB}"

SEED="${SEED:-42}"

THINKING_MODE="${THINKING_MODE:-understanding}"
VIT_MIN_SIZE="${VIT_MIN_SIZE:-512}"

# ==================== Print Configuration ====================
echo "=============================================="
echo "Understanding Rethink Eval (1-GPU) — Distance"
echo "=============================================="
echo "Configuration:"
echo "  Model Path:      $MODEL_PATH"
echo "  Results File:    $RESULTS_FILE"
echo "  Data Dir:        $DATA_DIR"
echo "  Dataset Files:   $DATASET_FILES"
echo "  Output Dir:      $OUTPUT_DIR"
echo "  Max Mem/GPU:     $MAX_MEM_PER_GPU"
echo "  Thinking Mode:   $THINKING_MODE"
echo "  Seed:            $SEED"
echo "=============================================="
echo ""

# ==================== Validation ====================
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "ERROR: Model directory not found: $MODEL_PATH"
    exit 1
fi

if [[ ! -f "$RESULTS_FILE" ]]; then
    echo "ERROR: Prior results file not found: $RESULTS_FILE"
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

OUTPUT_FILE="${OUTPUT_DIR}/rethink_results_bagel.json"

echo "Running inference -> ${OUTPUT_FILE}"
echo ""

python3 "${SCRIPT_DIR}/inference/run_inference_bagel_rethink.py" \
    --model_path "$MODEL_PATH" \
    --results_file "$RESULTS_FILE" \
    --data_dir "$DATA_DIR" \
    --dataset_files $DATASET_FILES \
    --output_file "$OUTPUT_FILE" \
    --max_mem_per_gpu "$MAX_MEM_PER_GPU" \
    --random_seed "$SEED" \
    --thinking_mode "$THINKING_MODE" \
    --vit_min_size "$VIT_MIN_SIZE"

echo ""
echo "=============================================="
echo "Done! Results saved to: $OUTPUT_FILE"
echo "=============================================="
