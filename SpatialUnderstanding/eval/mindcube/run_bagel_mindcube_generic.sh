#!/bin/bash
#SBATCH --job-name=bagel_mc_generic
#SBATCH --output=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/slurm_logs/mindcube/%x_output.txt
#SBATCH --error=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/slurm_logs/mindcube/%x_error.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100l:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=3:00:00
#SBATCH --partition=short-unkillable

# Generic BAGEL MindCube Inference — 4 GPUs, short-unkillable (3 h)
# Required env vars: MODEL_PATH, OUTPUT_DIR
# Optional env vars: THINK (false), THINKING_MODE (no_thinking), NUM_SHARDS (4)
set -euo pipefail

SCRIPT_DIR="/path/to/ThinkMorph-BAGEL-release"

[[ -z "${MODEL_PATH:-}" ]] && { echo "ERROR: MODEL_PATH is required"; exit 1; }
[[ -z "${OUTPUT_DIR:-}"  ]] && { echo "ERROR: OUTPUT_DIR is required";  exit 1; }
[[ ! -d "$MODEL_PATH"    ]] && { echo "ERROR: Model not found: $MODEL_PATH"; exit 1; }

DATA_DIR="${DATA_DIR:-/path/to/scratch/datasets/MindCube/data}"
DATASET_FILES="${DATASET_FILES:-raw/MindCube_tinybench.jsonl}"
GENERATED_IMAGES_DIR="${GENERATED_IMAGES_DIR:-${OUTPUT_DIR}/generated_images}"
MAX_MEM_PER_GPU="${MAX_MEM_PER_GPU:-70GiB}"
NUM_SHARDS="${NUM_SHARDS:-4}"
SEED="${SEED:-42}"
THINK="${THINK:-false}"
THINKING_MODE="${THINKING_MODE:-no_thinking}"
VIT_MIN_SIZE="${VIT_MIN_SIZE:-512}"
MAX_ROUNDS="${MAX_ROUNDS:-3}"
SHARD_BATCH="${SHARD_BATCH:-0}"
GPUS_PER_JOB=4

START_SHARD=$((SHARD_BATCH * GPUS_PER_JOB))
SHARD_INDICES=()
for i in $(seq 0 $((GPUS_PER_JOB - 1))); do
    IDX=$((START_SHARD + i))
    if [[ $IDX -lt $NUM_SHARDS ]]; then SHARD_INDICES+=($IDX); fi
done
[[ ${#SHARD_INDICES[@]} -eq 0 ]] && { echo "ERROR: SHARD_BATCH=${SHARD_BATCH} exceeds total shards"; exit 1; }

echo "MindCube | Model: ${MODEL_PATH##*/} | Thinking: ${THINKING_MODE} | Shards: ${SHARD_INDICES[*]}/${NUM_SHARDS}"

module load cuda/12.6.0
unset ROCR_VISIBLE_DEVICES
source /path/to/scratch/morph_env/bin/activate
mkdir -p "$OUTPUT_DIR"

THINK_FLAG=""; [[ "$THINK" == "false" ]] && THINK_FLAG="--no_think"
GENERATED_IMAGES_FLAG=""; [[ -n "${GENERATED_IMAGES_DIR:-}" ]] && GENERATED_IMAGES_FLAG="--generated_images_dir $GENERATED_IMAGES_DIR"

PIDS=()
for i in "${!SHARD_INDICES[@]}"; do
    SHARD_IDX="${SHARD_INDICES[$i]}"
    CUDA_VISIBLE_DEVICES="$i" python3 "${SCRIPT_DIR}/inference/run_inference_bagel_mindcube.py" \
        --model_path "$MODEL_PATH" --data_dir "$DATA_DIR" --dataset_files $DATASET_FILES \
        --output_file "${OUTPUT_DIR}/inference_results_shard${SHARD_IDX}.json" \
        --max_mem_per_gpu "$MAX_MEM_PER_GPU" --shard "${SHARD_IDX}/${NUM_SHARDS}" \
        --random_seed "$SEED" --thinking_mode "$THINKING_MODE" \
        --vit_min_size "$VIT_MIN_SIZE" --max_rounds "$MAX_ROUNDS" \
        --image_shapes 720 720 $THINK_FLAG $GENERATED_IMAGES_FLAG &
    PIDS+=($!)
done

FAIL=0
for pid in "${PIDS[@]}"; do wait "$pid" || FAIL=1; done
[[ "$FAIL" -eq 0 ]] && echo "All shards done." || echo "Some shards failed."
echo "Results: $OUTPUT_DIR"
