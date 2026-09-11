#!/bin/bash
#SBATCH --job-name=mmsi_vanilla_interleaved
#SBATCH --output=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/mmsi_log/mmsi_vanilla_interleaved_4gpu_output_%a.txt
#SBATCH --error=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/mmsi_log/mmsi_vanilla_interleaved_4gpu_error_%a.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100l:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=3:00:00
#SBATCH --partition=short-unkillable
#SBATCH --exclude=cn-g[001-007,009-011,014,017-020,022,024,027]

#
# MMSI-Bench Inference — vanilla BAGEL-7B-MoT, think=true, interleaved_thinking
# 4 GPUs × 4 shards on short-unkillable (3 h cap).
#

set -euo pipefail

SCRIPT_DIR="/path/to/ThinkMorph-BAGEL-release"

MODEL_PATH="${MODEL_PATH:-/path/to/scratch/models/BAGEL-7B-MoT}"
DATA_SOURCE="${DATA_SOURCE:-parquet}"
PARQUET_PATH="${PARQUET_PATH:-/path/to/scratch/datasets/MMSI-Bench}"
OUTPUT_DIR="${OUTPUT_DIR:-/path/to/scratch/VisualCoT/BAGEL_vanilla_mmsi_interleaved_eval}"

MAX_MEM_PER_GPU="${MAX_MEM_PER_GPU:-80GiB}"
NUM_PASSES="${NUM_PASSES:-1}"
NUM_SHARDS="${NUM_SHARDS:-4}"
SEED="${SEED:-42}"

THINKING_MODE="${THINKING_MODE:-interleaved_thinking}"
VIT_MIN_SIZE="${VIT_MIN_SIZE:-512}"
MAX_ROUNDS="${MAX_ROUNDS:-3}"
THINK="${THINK:-true}"
IMAGE_SHAPES="${IMAGE_SHAPES:-720 720}"

SHARD_BATCH="${SLURM_ARRAY_TASK_ID:-0}"
GPUS_PER_JOB=4

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

if [[ ! -d "$MODEL_PATH" ]]; then
    echo "ERROR: Model directory not found: $MODEL_PATH"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "MMSI-Bench Inference — vanilla BAGEL, think=true, interleaved_thinking"
echo "=============================================="
echo "  Model Path:      $MODEL_PATH"
echo "  Output Dir:      $OUTPUT_DIR"
echo "  Thinking Mode:   $THINKING_MODE"
echo "  Think:           $THINK"
echo "  Shards:          ${SHARD_INDICES[*]} / $NUM_SHARDS"
echo "=============================================="

module load cuda/12.6.0
unset ROCR_VISIBLE_DEVICES
source /path/to/scratch/morph_env/bin/activate

nvidia-smi --query-gpu=index,name,memory.total --format=csv

THINK_FLAG=""
if [[ "$THINK" == "false" ]]; then
    THINK_FLAG="--no_think"
fi

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
        --vit_min_size "$VIT_MIN_SIZE" \
        --max_rounds "$MAX_ROUNDS" \
        ${IMAGE_SHAPES:+--image_shapes $IMAGE_SHAPES} \
        --resume \
        $THINK_FLAG &

    PIDS+=($!)
done

FAIL=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        echo "ERROR: Process $pid failed"
        FAIL=1
    fi
done

echo "=============================================="
if [[ "$FAIL" -eq 0 ]]; then
    echo "All shards (${SHARD_INDICES[*]}) complete!"
else
    echo "Some shards failed — check output above."
fi
echo "Results saved to: $OUTPUT_DIR"
