#!/bin/bash
#SBATCH --job-name=bagel_pano_4gpu
#SBATCH --output=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/bagel/bagel_panorama_view_4gpu_output.txt
#SBATCH --error=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/bagel/bagel_panorama_view_4gpu_error.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100l:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=3:00:00
#SBATCH --partition=short-unkillable

#
# BAGEL Panorama View Generation — 4 GPUs on short-unkillable
#
# Runs 4 shards in parallel, one per GPU, using CUDA_VISIBLE_DEVICES.
# Repeat with different SHARD_BATCH to cover all shards.
#
set -euo pipefail

SCRIPT_DIR="/path/to/ThinkMorph-BAGEL-release"

# ==================== Configuration ====================
MODEL_PATH="${MODEL_PATH:-/path/to/scratch/VisualCoT/BAGEL_checkpoints/BAGEL_format_panorama_view_generation}"

TEST_FILE="${TEST_FILE:-/path/to/scratch/VisualCoT/training_data/panorama_view_generation/test_samples.json}"

OUTPUT_DIR="${OUTPUT_DIR:-/path/to/scratch/VisualCoT/panorama_view_generation_test_set}"

MAX_MEM_PER_GPU="${MAX_MEM_PER_GPU:-70GiB}"

NUM_SHARDS="${NUM_SHARDS:-4}"

SEED="${SEED:-42}"

THINK="${THINK:-true}"

NUM_SAMPLES="${NUM_SAMPLES:-}"

# Which batch of 4 shards to run (0 = shards 0-3, 1 = shards 4-7, etc.)
SHARD_BATCH="${SHARD_BATCH:-0}"
GPUS_PER_JOB=4

# ==================== Compute shard indices for this batch ====================
START_SHARD=$((SHARD_BATCH * GPUS_PER_JOB))
SHARD_INDICES=()
for i in $(seq 0 $((GPUS_PER_JOB - 1))); do
    IDX=$((START_SHARD + i))
    if [[ $IDX -lt $NUM_SHARDS ]]; then
        SHARD_INDICES+=($IDX)
    fi
done

if [[ ${#SHARD_INDICES[@]} -eq 0 ]]; then
    echo "ERROR: SHARD_BATCH=${SHARD_BATCH} exceeds total shards (${NUM_SHARDS}). Nothing to do."
    exit 1
fi

# ==================== Print Configuration ====================
echo "=============================================="
echo "BAGEL Panorama View Generation (4-GPU)"
echo "=============================================="
echo "Configuration:"
echo "  Model Path:       $MODEL_PATH"
echo "  Test File:        $TEST_FILE"
echo "  Output Dir:       $OUTPUT_DIR"
echo "  Max Mem/GPU:      $MAX_MEM_PER_GPU"
echo "  Total Shards:     $NUM_SHARDS"
echo "  Shard Batch:      $SHARD_BATCH (shards ${SHARD_INDICES[*]})"
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

if [[ ! -f "$TEST_FILE" ]]; then
    echo "ERROR: Test file not found: $TEST_FILE"
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

# ==================== Launch shards in parallel, one per GPU ====================
PIDS=()
for i in "${!SHARD_INDICES[@]}"; do
    SHARD_IDX="${SHARD_INDICES[$i]}"
    SHARD_SPEC="${SHARD_IDX}/${NUM_SHARDS}"
    GPU_ID="$i"

    echo "Launching shard ${SHARD_IDX}/${NUM_SHARDS} on GPU ${GPU_ID}"

    CUDA_VISIBLE_DEVICES="${GPU_ID}" python3 "${SCRIPT_DIR}/SpatialUnderstanding/test_bagel_panorama_view.py" \
        --model_path "$MODEL_PATH" \
        --test_file "$TEST_FILE" \
        --output_dir "$OUTPUT_DIR" \
        --max_mem_per_gpu "$MAX_MEM_PER_GPU" \
        --random_seed "$SEED" \
        --shard "$SHARD_SPEC" \
        $THINK_FLAG \
        $NUM_SAMPLES_FLAG &

    PIDS+=($!)
done

# ==================== Wait for all to finish ====================
echo ""
echo "Waiting for ${#PIDS[@]} shards to complete..."
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
    echo "All shards (${SHARD_INDICES[*]}) complete!"
else
    echo "Some shards failed — check output above."
fi
echo "=============================================="
echo "Results saved to: $OUTPUT_DIR"
