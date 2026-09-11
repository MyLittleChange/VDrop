#!/bin/bash
#SBATCH --job-name=mmsi_rethink
#SBATCH --output=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/mmsi_log/mmsi_rethink_output.txt
#SBATCH --error=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/mmsi_log/mmsi_rethink_error.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --partition=unkillable

#
# MMSI-Bench Rethink Inference — 1 GPU
#
# Feeds pre-generated thinking images back as additional input (no new generation).
#
# Usage:
#   sbatch eval_mmsi_bench_rethink.sh
#
set -euo pipefail

SCRIPT_DIR="/path/to/ThinkMorph-BAGEL-release"

# ==================== Configuration ====================
MODEL_PATH="${MODEL_PATH:-/path/to/scratch/VisualCoT/BAGEL_checkpoints/BAGEL_format_panorama_qa_visual_only}"

# Directory with prior inference results (contains model_results_run_1*.json with GeneratedImages)
RESULTS_DIR="${RESULTS_DIR:-/path/to/scratch/VisualCoT/panorama_qa_visual_only_mmsi_correct_shape}"

OUTPUT_DIR="${OUTPUT_DIR:-/path/to/scratch/VisualCoT/rethink_panorama_qa_visual_only_mmsi}"

MAX_MEM_PER_GPU="${MAX_MEM_PER_GPU:-70GiB}"
SEED="${SEED:-42}"
THINKING_MODE="${THINKING_MODE:-no_thinking}"
VIT_MIN_SIZE="${VIT_MIN_SIZE:-512}"

# ==================== Print Configuration ====================
echo "=============================================="
echo "MMSI-Bench Rethink Inference (1-GPU)"
echo "=============================================="
echo "Configuration:"
echo "  Model Path:      $MODEL_PATH"
echo "  Results Dir:     $RESULTS_DIR"
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

if [[ ! -d "$RESULTS_DIR" ]]; then
    echo "ERROR: Prior results directory not found: $RESULTS_DIR"
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

# ==================== Run Inference ====================
echo "Running rethink inference..."
echo ""

python3 "${SCRIPT_DIR}/inference/eval_mmsi_bench_rethink.py" \
    --mode inference \
    --model_path "$MODEL_PATH" \
    --results_dir "$RESULTS_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --max_mem_per_gpu "$MAX_MEM_PER_GPU" \
    --seed "$SEED" \
    --thinking_mode "$THINKING_MODE" \
    --vit_min_size "$VIT_MIN_SIZE"

echo ""
echo "=============================================="
echo "Done! Results saved to: $OUTPUT_DIR"
echo "=============================================="
