#!/bin/bash
#SBATCH --job-name=mmsi_cv_lora_7k
#SBATCH --output=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/mmsi_log/mmsi_training_data_corner_view_visual_only_lora_7k_batch%a_output.txt
#SBATCH --error=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/mmsi_log/mmsi_training_data_corner_view_visual_only_lora_7k_batch%a_error.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100l:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --partition=main

# MMSI-Bench inference for cv_lora_7k
# Strategy: 8 shards total, 4 GPUs per job, 2 chained jobs (SHARD_BATCH=0 covers shards 0-3;
# SHARD_BATCH=1 covers shards 4-7). 3h short-unkillable wasn't enough for a single 4-shard job
# (~250 samples per shard on 4 GPUs), so split into 8 shards (~125/shard) across 2 jobs.
#
# Submit as a Slurm array: --array=0-1 sets SLURM_ARRAY_TASK_ID to 0 then 1.
# CheckpointManager-style resume: each shard writes its own checkpoint file, so a re-submit
# of a TIMEOUT-ed job picks up where it left off.
set -euo pipefail
SCRIPT_DIR="/path/to/ThinkMorph-BAGEL-release"

MODEL_PATH="${MODEL_PATH:-/path/to/scratch/VisualCoT/BAGEL_checkpoints/BAGEL_format_training_data_corner_view_visual_only_lora_7k}"
DATA_SOURCE="${DATA_SOURCE:-parquet}"
PARQUET_PATH="${PARQUET_PATH:-/path/to/scratch/datasets/MMSI-Bench}"
OUTPUT_DIR="${OUTPUT_DIR:-/path/to/scratch/VisualCoT/eval_outputs/BAGEL_format_training_data_corner_view_visual_only_lora_7k_mmsi}"

MAX_MEM_PER_GPU="${MAX_MEM_PER_GPU:-70GiB}"
NUM_PASSES="${NUM_PASSES:-1}"
NUM_SHARDS="${NUM_SHARDS:-8}"
SEED="${SEED:-42}"
THINKING_MODE="${THINKING_MODE:-visual_only_thinking}"
VIT_MIN_SIZE="${VIT_MIN_SIZE:-512}"
MAX_ROUNDS="${MAX_ROUNDS:-3}"
THINK="${THINK:-false}"
IMAGE_SHAPES="${IMAGE_SHAPES:-720 1024}"

SHARD_BATCH="${SLURM_ARRAY_TASK_ID:-${SHARD_BATCH:-0}}"
GPUS_PER_JOB=2
START_SHARD=$((SHARD_BATCH * GPUS_PER_JOB))
SHARD_INDICES=()
for i in $(seq 0 $((GPUS_PER_JOB - 1))); do
    IDX=$((START_SHARD + i))
    if [[ $IDX -lt $NUM_SHARDS ]]; then SHARD_INDICES+=($IDX); fi
done
if [[ ${#SHARD_INDICES[@]} -eq 0 ]]; then echo "ERROR: SHARD_BATCH=${SHARD_BATCH} exceeds total shards (${NUM_SHARDS})"; exit 1; fi

if [[ ! -d "$MODEL_PATH" ]]; then echo "ERROR: Model not found: $MODEL_PATH"; exit 1; fi

echo "MMSI-Bench cv_lora_7k — Model: $MODEL_PATH | Shards: $NUM_SHARDS | Batch: $SHARD_BATCH (covers ${SHARD_INDICES[*]})"

module load cuda/12.6.0
unset ROCR_VISIBLE_DEVICES
source /path/to/scratch/morph_env/bin/activate

nvidia-smi --query-gpu=index,name,memory.total --format=csv
mkdir -p "$OUTPUT_DIR"

THINK_FLAG=""; [[ "$THINK" == "false" ]] && THINK_FLAG="--no_think"

PIDS=()
for i in "${!SHARD_INDICES[@]}"; do
    SHARD_IDX="${SHARD_INDICES[$i]}"
    SHARD_SPEC="${SHARD_IDX}/${NUM_SHARDS}"
    GPU_ID="$i"
    echo "Launching shard ${SHARD_IDX}/${NUM_SHARDS} on GPU ${GPU_ID}"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" python3 "${SCRIPT_DIR}/inference/eval_mmsi_bench.py" \
        --mode inference --model_path "$MODEL_PATH" \
        --data_source "$DATA_SOURCE" --parquet_path "$PARQUET_PATH" \
        --output_dir "$OUTPUT_DIR" --max_mem_per_gpu "$MAX_MEM_PER_GPU" \
        --num_passes "$NUM_PASSES" --seed "$SEED" --shard "$SHARD_SPEC" \
        --thinking_mode "$THINKING_MODE" --vit_min_size "$VIT_MIN_SIZE" \
        --max_rounds "$MAX_ROUNDS" \
        --resume \
        ${IMAGE_SHAPES:+--image_shapes $IMAGE_SHAPES} \
        $THINK_FLAG &
    PIDS+=($!)
done

FAIL=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then echo "ERROR: Process $pid failed"; FAIL=1; fi
done
[[ "$FAIL" -eq 0 ]] && echo "All shards complete!" || echo "Some shards failed."
echo "Results saved to: $OUTPUT_DIR"
