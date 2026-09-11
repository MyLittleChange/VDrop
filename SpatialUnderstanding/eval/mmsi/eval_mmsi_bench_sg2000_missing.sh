#!/bin/bash
#SBATCH --job-name=mmsi_sg2k_miss
#SBATCH --output=/path/to/ThinkMorph-BAGEL-release/mmsi_sg2k_missing_output.txt
#SBATCH --error=/path/to/ThinkMorph-BAGEL-release/mmsi_sg2k_missing_error.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100l:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --time=3:00:00
#SBATCH --partition=short-unkillable

#
# Re-run missing MMSI-Bench shards (31, 37, 43) for BAGEL_sg_2000
# Uses short-unkillable partition (requires 4 GPUs).
# Runs 3 shards sequentially, each on a single GPU via CUDA_VISIBLE_DEVICES.
#
# Usage:
#   sbatch eval_mmsi_bench_sg2000_missing.sh
#

set -euo pipefail

MISSING_SHARDS=(31 37 43)

SCRIPT_DIR="/path/to/ThinkMorph-BAGEL-release"

# ==================== Configuration ====================
MODEL_PATH="${MODEL_PATH:-/path/to/scratch/models/BAGEL-7B-MoT}"
DATA_SOURCE="${DATA_SOURCE:-parquet}"
PARQUET_PATH="${PARQUET_PATH:-/path/to/scratch/datasets/MMSI-Bench}"
OUTPUT_DIR="${OUTPUT_DIR:-/path/to/scratch/VisualCoT/BAGEL_sg_2000_mmsi_eval}"
MAX_MEM_PER_GPU="${MAX_MEM_PER_GPU:-80GiB}"
NUM_PASSES="${NUM_PASSES:-1}"
NUM_SHARDS="${NUM_SHARDS:-48}"
SEED="${SEED:-42}"

# ==================== Validation ====================
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "ERROR: Model directory not found: $MODEL_PATH"
    exit 1
fi

module load cuda/12.6.0
unset ROCR_VISIBLE_DEVICES
source /path/to/scratch/morph_env/bin/activate

# Show GPU info
echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo ""

# ==================== Run each missing shard on a separate GPU ====================
PIDS=()
for i in "${!MISSING_SHARDS[@]}"; do
    SHARD_IDX="${MISSING_SHARDS[$i]}"
    SHARD_SPEC="${SHARD_IDX}/${NUM_SHARDS}"
    GPU_ID="$i"

    echo "=============================================="
    echo "Launching shard ${SHARD_IDX}/${NUM_SHARDS} on GPU ${GPU_ID}"
    echo "=============================================="
    echo "  Model Path:      $MODEL_PATH"
    echo "  Output Dir:      $OUTPUT_DIR"
    echo "  Shard:           $SHARD_SPEC"
    echo ""

    CUDA_VISIBLE_DEVICES="${GPU_ID}" python3 "${SCRIPT_DIR}/inference/eval_mmsi_bench.py" \
        --mode inference \
        --model_path "$MODEL_PATH" \
        --data_source "$DATA_SOURCE" \
        --parquet_path "$PARQUET_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --max_mem_per_gpu "$MAX_MEM_PER_GPU" \
        --num_passes "$NUM_PASSES" \
        --seed "$SEED" \
        --shard "$SHARD_SPEC" &

    PIDS+=($!)
done

# Wait for all to finish
echo ""
echo "Waiting for all 3 shards to complete..."
FAIL=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        echo "ERROR: Process $pid failed"
        FAIL=1
    fi
done

echo ""
echo "=============================================="
if [[ "$FAIL" -eq 0 ]]; then
    echo "All missing shards complete!"
else
    echo "Some shards failed — check output above."
fi
echo "=============================================="
echo "Results saved to: $OUTPUT_DIR"
