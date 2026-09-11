#!/bin/bash
#SBATCH --job-name=mmsi_missing
#SBATCH --output=/path/to/ThinkMorph-BAGEL-release/mmsi_log/mmsi_inference_output_%a.txt
#SBATCH --error=/path/to/ThinkMorph-BAGEL-release/mmsi_log/mmsi_inference_error_%a.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2:00:00
#SBATCH --array=0-1
#SBATCH --partition=main

#
# Re-run missing MMSI-Bench shards (12, 18)
#
# Usage:
#   sbatch eval_mmsi_bench_missing.sh
#

set -euo pipefail

# Map SLURM array index to the actual missing shard indices
MISSING_SHARDS=(2)
SHARD_IDX="${MISSING_SHARDS[${SLURM_ARRAY_TASK_ID:-0}]}"

SCRIPT_DIR="/path/to/ThinkMorph-BAGEL-release"

# ==================== Configuration ====================
MODEL_PATH="${MODEL_PATH:-/path/to/scratch/VisualCoT/BAGEL_checkpoints/BAGEL_format_2000}"
DATA_SOURCE="${DATA_SOURCE:-parquet}"
PARQUET_PATH="${PARQUET_PATH:-/path/to/scratch/datasets/MMSI-Bench}"
OUTPUT_DIR="${OUTPUT_DIR:-/path/to/scratch/VisualCoT/parsed_qa_anchor_cenetr_spatial_rear_mmsi_eval}"
MAX_MEM_PER_GPU="${MAX_MEM_PER_GPU:-80GiB}"
NUM_PASSES="${NUM_PASSES:-1}"
NUM_SHARDS="${NUM_SHARDS:-24}"
SEED="${SEED:-42}"

SHARD_SPEC="${SHARD_IDX}/${NUM_SHARDS}"

# ==================== Validation ====================
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "ERROR: Model directory not found: $MODEL_PATH"
    exit 1
fi

# ==================== Display Configuration ====================
echo "=============================================="
echo "MMSI-Bench Inference - Missing Shards"
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
