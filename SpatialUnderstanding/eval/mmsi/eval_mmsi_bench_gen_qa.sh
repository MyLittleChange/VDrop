#!/bin/bash
#SBATCH --job-name=mmsi_inference
#SBATCH --output=/path/to/ThinkMorph-BAGEL-release/mmsi_log/mmsi_visual_only_inference_output_%a.txt
#SBATCH --error=/path/to/ThinkMorph-BAGEL-release/mmsi_log/mmsi_visual_only_inference_error_%a.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=3:00:00
#SBATCH --array=0-23
#SBATCH --partition=long

#
# MMSI-Bench Inference Script
#
# Runs ThinkMorph inference on MMSI-Bench dataset using job arrays
# for parallel execution across multiple GPUs.
#
# Usage:
#   # Submit all shards as a job array
#   sbatch eval_mmsi_bench.sh
#
#   # Run a single shard interactively
#   SLURM_ARRAY_TASK_ID=0 bash eval_mmsi_bench.sh
#
#   # Submit with custom number of shards (edit --array above)
#   sbatch eval_mmsi_bench.sh
#

set -euo pipefail

SCRIPT_DIR="/path/to/ThinkMorph-BAGEL-release"

# ==================== Configuration ====================
# Model path
MODEL_PATH="${MODEL_PATH:-/path/to/scratch/VisualCoT/BAGEL_checkpoints/BAGEL_format_gen_qa}"
# MODEL_PATH="${MODEL_PATH:-/path/to/scratch/models/BAGEL-7B-MoT}"

# Data source
DATA_SOURCE="${DATA_SOURCE:-parquet}"
PARQUET_PATH="${PARQUET_PATH:-/path/to/scratch/datasets/MMSI-Bench}"

# Output directory for inference results
OUTPUT_DIR="${OUTPUT_DIR:-/path/to/scratch/VisualCoT/parsed_qa_anchor_cenetr_spatial_rear_mmsi_eval}"

# Output directory for evaluated results
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-/path/to/scratch/VisualCoT/mmsi_evaluated}"

# GPU memory per card
MAX_MEM_PER_GPU="${MAX_MEM_PER_GPU:-80GiB}"

# Number of evaluation passes
NUM_PASSES="${NUM_PASSES:-1}"

# Number of shards (should match --array range, e.g., 0-47 means 48 shards)
NUM_SHARDS="${NUM_SHARDS:-24}"

# Random seed
SEED="${SEED:-42}"

# LLM Judge Configuration (Gemini)
JUDGE_API_KEY="${JUDGE_API_KEY:-${VisualCoT_GEMINI:-}}"
JUDGE_MODEL_NAME="${JUDGE_MODEL_NAME:-gemini-3-flash-preview}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"

# ==================== Determine Shard ====================
# Use SLURM_ARRAY_TASK_ID if available, otherwise default to 0
SHARD_IDX="${SLURM_ARRAY_TASK_ID:-0}"
SHARD_SPEC="${SHARD_IDX}/${NUM_SHARDS}"

# ==================== Validation ====================
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "ERROR: Model directory not found: $MODEL_PATH"
    exit 1
fi

# ==================== Display Configuration ====================
echo "=============================================="
echo "MMSI-Bench Inference with ThinkMorph"
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
echo "=============================================="
echo ""

module load cuda/12.6.0
unset ROCR_VISIBLE_DEVICES
source /path/to/scratch/morph_env/bin/activate

# Show GPU info
echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo ""

# ==================== Run Inference ====================
echo "Starting inference for shard ${SHARD_IDX}/${NUM_SHARDS}..."
echo ""

python3 "${SCRIPT_DIR}/inference/eval_mmsi_bench.py" \
    --mode inference \
    --model_path "$MODEL_PATH" \
    --data_source "$DATA_SOURCE" \
    --parquet_path "$PARQUET_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --max_mem_per_gpu "$MAX_MEM_PER_GPU" \
    --num_passes "$NUM_PASSES" \
    --seed "$SEED" \
    --shard "$SHARD_SPEC"

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
