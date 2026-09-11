#!/bin/bash
#SBATCH --job-name=mmsi_dynamic_visual_2gpu
#SBATCH --output=/path/to/ThinkMorph-BAGEL-release/mmsi_log/mmsi_dynamic_visual_2gpu_output.txt
#SBATCH --error=/path/to/ThinkMorph-BAGEL-release/mmsi_log/mmsi_dynamic_visual_2gpu_error.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100l:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --partition=main

#
# MMSI-Bench Dynamic Visual Thinking Inference — 2 GPUs per job
#
# Runs 2 shards in parallel, one per GPU, in a single job (no array).
#
# The model self-selects the visual mode:
#   <panoramic> <image_start> ... <image_end>  → 720×2048
#   <BEV> <image_start> ... <image_end>        → 720×720
#
# Usage:
#   sbatch eval_mmsi_bench_dynamic_visual_2gpu.sh
#   bash eval_mmsi_bench_dynamic_visual_2gpu.sh
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

NUM_SHARDS="${NUM_SHARDS:-2}"

SEED="${SEED:-42}"

# dynamic_visual_thinking: model generates <panoramic>/<BEV> before <image_start>
THINKING_MODE="${THINKING_MODE:-dynamic_visual_thinking}"
THINK="${THINK:-false}"

# Fallback image size when no mode token is detected (H W)
IMAGE_SHAPES="${IMAGE_SHAPES:-720 720}"

# ==================== Determine Shard Batch ====================
SHARD_BATCH="${SLURM_ARRAY_TASK_ID:-0}"
GPUS_PER_JOB=2

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

# ==================== Validation ====================
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "ERROR: Model directory not found: $MODEL_PATH"
    exit 1
fi

# ==================== Display Configuration ====================
echo "=============================================="
echo "MMSI-Bench Dynamic Visual Thinking Inference (2 GPUs)"
echo "=============================================="
echo "Configuration:"
echo "  Model Path:      $MODEL_PATH"
echo "  Data Source:     $DATA_SOURCE"
echo "  Parquet Path:    $PARQUET_PATH"
echo "  Output Dir:      $OUTPUT_DIR"
echo "  Max Mem/GPU:     $MAX_MEM_PER_GPU"
echo "  Num Passes:      $NUM_PASSES"
echo "  Total Shards:    $NUM_SHARDS"
echo "  Shard Batch:     $SHARD_BATCH (shards ${SHARD_INDICES[*]})"
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

# ==================== Build flags ====================
THINK_FLAG=""
if [[ "$THINK" == "false" ]]; then
    THINK_FLAG="--no_think"
fi

# ==================== Launch shards in parallel, one per GPU ====================
PIDS=()
for i in "${!SHARD_INDICES[@]}"; do
    SHARD_IDX="${SHARD_INDICES[$i]}"
    SHARD_SPEC="${SHARD_IDX}/${NUM_SHARDS}"
    GPU_ID="$i"

    echo "Launching shard ${SHARD_IDX}/${NUM_SHARDS} on GPU ${GPU_ID}"

    CUDA_VISIBLE_DEVICES="${GPU_ID}" python3 "${SCRIPT_DIR}/inference/eval_mmsi_bench.py" \
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
        $THINK_FLAG &

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

# ==================== Run Evaluation with LLM Judge ====================
# Note: Only run evaluation after ALL shards are complete.
# Evaluation should be run separately (not as part of the array job).
# Example:
#   VisualCoT_GEMINI=your-key python inference/eval_mmsi_bench.py \
#       --mode evaluate --results_dir $OUTPUT_DIR --output_dir $EVAL_OUTPUT_DIR
