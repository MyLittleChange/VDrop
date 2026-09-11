#!/bin/bash
#SBATCH --job-name=bagel_omni_generic
#SBATCH --output=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/slurm_logs/omnispatial/%x_output.txt
#SBATCH --error=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/slurm_logs/omnispatial/%x_error.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100l:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --partition=main

# Generic BAGEL OmniSpatial Inference — 4 GPUs
# Required env vars: MODEL_PATH, OUTPUT_DIR
# Optional env vars: THINK (false), THINKING_MODE (no_thinking), NUM_SHARDS (4)
set -euo pipefail

SCRIPT_DIR="/path/to/ThinkMorph-BAGEL-release"

[[ -z "${MODEL_PATH:-}" ]] && { echo "ERROR: MODEL_PATH is required"; exit 1; }
[[ -z "${OUTPUT_DIR:-}"  ]] && { echo "ERROR: OUTPUT_DIR is required";  exit 1; }
[[ ! -d "$MODEL_PATH"    ]] && { echo "ERROR: Model not found: $MODEL_PATH"; exit 1; }

DATASET_PATH="${DATASET_PATH:-/path/to/scratch/datasets/OmniSpatial/OmniSpatial-test}"
MAX_MEM_PER_GPU="${MAX_MEM_PER_GPU:-70GiB}"
NUM_SHARDS="${NUM_SHARDS:-4}"
SEED="${SEED:-42}"
THINK="${THINK:-false}"
THINKING_MODE="${THINKING_MODE:-no_thinking}"
MAX_ROUNDS="${MAX_ROUNDS:-3}"
TASK_TYPE="${TASK_TYPE:-Complex_Logic,Perspective_Taking}"
SUB_TASK_TYPE="${SUB_TASK_TYPE:-}"
SHARD_BATCH="${SHARD_BATCH:-0}"
GPUS_PER_JOB=4

START_SHARD=$((SHARD_BATCH * GPUS_PER_JOB))
SHARD_INDICES=()
for i in $(seq 0 $((GPUS_PER_JOB - 1))); do
    IDX=$((START_SHARD + i))
    if [[ $IDX -lt $NUM_SHARDS ]]; then SHARD_INDICES+=($IDX); fi
done
[[ ${#SHARD_INDICES[@]} -eq 0 ]] && { echo "ERROR: SHARD_BATCH=${SHARD_BATCH} exceeds total shards"; exit 1; }

echo "OmniSpatial | Model: ${MODEL_PATH##*/} | Thinking: ${THINKING_MODE} | Shards: ${SHARD_INDICES[*]}/${NUM_SHARDS}"

module load cuda/12.6.0
unset ROCR_VISIBLE_DEVICES
source /path/to/scratch/morph_env/bin/activate
mkdir -p "$OUTPUT_DIR"

THINK_FLAG=""; [[ "$THINK" == "false" ]] && THINK_FLAG="--no_think"
FILTER_FLAGS=""
[[ -n "$TASK_TYPE"     ]] && FILTER_FLAGS="$FILTER_FLAGS --task_type $TASK_TYPE"
[[ -n "$SUB_TASK_TYPE" ]] && FILTER_FLAGS="$FILTER_FLAGS --sub_task_type $SUB_TASK_TYPE"

PIDS=()
for i in "${!SHARD_INDICES[@]}"; do
    SHARD_IDX="${SHARD_INDICES[$i]}"
    CUDA_VISIBLE_DEVICES="$i" python3 "${SCRIPT_DIR}/inference/run_inference_bagel_omnispatial.py" \
        --model_path "$MODEL_PATH" \
        --dataset_path "$DATASET_PATH" \
        --output_file "${OUTPUT_DIR}/inference_results_bagel_shard${SHARD_IDX}.json" \
        --max_mem_per_gpu "$MAX_MEM_PER_GPU" --shard "${SHARD_IDX}/${NUM_SHARDS}" \
        --random_seed "$SEED" --thinking_mode "$THINKING_MODE" \
        --max_rounds "$MAX_ROUNDS" \
        $THINK_FLAG $FILTER_FLAGS &
    PIDS+=($!)
done

FAIL=0
for pid in "${PIDS[@]}"; do wait "$pid" || FAIL=1; done
[[ "$FAIL" -eq 0 ]] && echo "All shards done." || echo "Some shards failed."
echo "Results: $OUTPUT_DIR"
