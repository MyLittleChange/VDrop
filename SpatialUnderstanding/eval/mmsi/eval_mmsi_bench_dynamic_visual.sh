#!/bin/bash
#SBATCH --job-name=mmsi_dynamic_visual
#SBATCH --output=/path/to/ThinkMorph-BAGEL-release/mmsi_log/mmsi_dynamic_visual_inference_output_%a.txt
#SBATCH --error=/path/to/ThinkMorph-BAGEL-release/mmsi_log/mmsi_dynamic_visual_inference_error_%a.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2:00:00
#SBATCH --array=0-23
#SBATCH --partition=long

#
# MMSI-Bench Dynamic Visual Thinking Inference
#
# The model self-selects the visual mode:
#   <panoramic> <image_start> ... <image_end>  → 720×2048
#   <BEV> <image_start> ... <image_end>        → 720×720
#
# Usage:
#   sbatch eval_mmsi_bench_dynamic_visual.sh
#   SLURM_ARRAY_TASK_ID=0 bash eval_mmsi_bench_dynamic_visual.sh
#

set -euo pipefail

SCRIPT_DIR="/path/to/ThinkMorph-BAGEL-release"

# ==================== Configuration ====================
MODEL_PATH="${MODEL_PATH:-/path/to/scratch/VisualCoT/BAGEL_checkpoints/BAGEL_format_mix_view_qa_dynamic_visual_only}"

DATA_SOURCE="${DATA_SOURCE:-parquet}"
PARQUET_PATH="${PARQUET_PATH:-/path/to/scratch/datasets/MMSI-Bench}"

OUTPUT_DIR="${OUTPUT_DIR:-/path/to/scratch/VisualCoT/dynamic_visual_only_mmsi}"

EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-/path/to/scratch/VisualCoT/mmsi_evaluated}"

MAX_MEM_PER_GPU="${MAX_MEM_PER_GPU:-80GiB}"

NUM_PASSES="${NUM_PASSES:-1}"

NUM_SHARDS="${NUM_SHARDS:-24}"

SEED="${SEED:-42}"

# dynamic_visual_thinking: model generates <panoramic>/<BEV> before <image_start>
THINKING_MODE="${THINKING_MODE:-dynamic_visual_thinking}"
THINK="${THINK:-false}"

# Fallback image size when no mode token is detected (H W)
IMAGE_SHAPES="${IMAGE_SHAPES:-720 720}"

# ==================== Determine Shard ====================
SHARD_IDX="${SLURM_ARRAY_TASK_ID:-0}"
SHARD_SPEC="${SHARD_IDX}/${NUM_SHARDS}"

# ==================== Validation ====================
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "ERROR: Model directory not found: $MODEL_PATH"
    exit 1
fi

# ==================== Display Configuration ====================
echo "=============================================="
echo "MMSI-Bench Dynamic Visual Thinking Inference"
echo "=============================================="
echo "Configuration:"
echo "  Model Path:      $MODEL_PATH"
echo "  Data Source:     $DATA_SOURCE"
echo "  Parquet Path:    $PARQUET_PATH"
echo "  Output Dir:      $OUTPUT_DIR"
echo "  Max Mem/GPU:     $MAX_MEM_PER_GPU"
echo "  Num Passes:      $NUM_PASSES"
echo "  Shard:           $SHARD_SPEC"
echo "  Seed:            $SEED"
echo "  Thinking Mode:   $THINKING_MODE"
echo "  Think:           $THINK"
echo "  Image Shapes:    $IMAGE_SHAPES (fallback)"
echo "=============================================="
echo ""

module load cuda/12.6.0
unset ROCR_VISIBLE_DEVICES
source /path/to/scratch/morph_env/bin/activate

echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo ""

# ==================== Run Inference ====================
echo "Starting inference for shard ${SHARD_IDX}/${NUM_SHARDS}..."
echo ""

THINK_FLAG=""
if [[ "$THINK" == "false" ]]; then
    THINK_FLAG="--no_think"
fi

python3 "${SCRIPT_DIR}/inference/eval_mmsi_bench.py" \
    --mode inference \
    --model_path "$MODEL_PATH" \
    --data_source "$DATA_SOURCE" \
    --parquet_path "$PARQUET_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --max_mem_per_gpu "$MAX_MEM_PER_GPU" \
    --num_passes "$NUM_PASSES" \
    --seed "$SEED" \
    --shard "$SHARD_SPEC" \
    --thinking_mode "$THINKING_MODE" \
    ${IMAGE_SHAPES:+--image_shapes $IMAGE_SHAPES} \
    $THINK_FLAG

echo ""
echo "=============================================="
echo "Shard ${SHARD_IDX}/${NUM_SHARDS} Complete!"
echo "=============================================="
echo "Results saved to: $OUTPUT_DIR"

# ==================== Run Evaluation with LLM Judge ====================
# Note: Only run evaluation after ALL shards are complete.
# Evaluation should be run separately (not as part of the array job).
# Example:
#   VisualCoT_GEMINI=your-key python inference/eval_mmsi_bench.py \
#       --mode evaluate --results_dir $OUTPUT_DIR --output_dir $EVAL_OUTPUT_DIR
