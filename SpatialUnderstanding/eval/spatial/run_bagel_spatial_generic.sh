#!/bin/bash
#SBATCH --job-name=bagel_spatial_generic
#SBATCH --output=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/slurm_logs/spatial/%x_output.txt
#SBATCH --error=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/slurm_logs/spatial/%x_error.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100l:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=3:00:00
#SBATCH --partition=short-unkillable

# Generic BAGEL Spatial Collaboration Inference — 4 GPUs
# Required env vars: MODEL_PATH, OUTPUT_DIR, TASK (anchor|counting|distance|direction)
# Optional env vars: THINK (false), THINKING_MODE (no_thinking), NUM_SHARDS (4), SHARD_BATCH (0)
set -euo pipefail

SCRIPT_DIR="/path/to/ThinkMorph-BAGEL-release"

[[ -z "${MODEL_PATH:-}" ]] && { echo "ERROR: MODEL_PATH is required"; exit 1; }
[[ -z "${OUTPUT_DIR:-}"  ]] && { echo "ERROR: OUTPUT_DIR is required";  exit 1; }
[[ -z "${TASK:-}"        ]] && { echo "ERROR: TASK is required (anchor|counting|distance|direction)"; exit 1; }
[[ ! -d "$MODEL_PATH"    ]] && { echo "ERROR: Model not found: $MODEL_PATH"; exit 1; }

case "$TASK" in
    anchor)    DATASET_FILES="${DATASET_FILES:-approved_mcqs_anchor_normalized.json}" ;;
    counting)  DATASET_FILES="${DATASET_FILES:-approved_mcqs_counting_normalized.json}" ;;
    distance)  DATASET_FILES="${DATASET_FILES:-approved_mcqs_relative_distance_normalized.json}" ;;
    direction) DATASET_FILES="${DATASET_FILES:-approved_mcqs_relative_direction_normalized.json}" ;;
    *) echo "ERROR: Unknown TASK=${TASK}. Use anchor|counting|distance|direction"; exit 1 ;;
esac

DATA_DIR="${DATA_DIR:-/path/to/scratch/VisualCoT/spatial_collab_dataset}"
GENERATED_IMAGES_DIR="${GENERATED_IMAGES_DIR:-${OUTPUT_DIR}/generated_images}"
MAX_MEM_PER_GPU="${MAX_MEM_PER_GPU:-70GiB}"
NUM_SHARDS="${NUM_SHARDS:-4}"
SEED="${SEED:-42}"
THINK="${THINK:-false}"
THINKING_MODE="${THINKING_MODE:-no_thinking}"
VIT_MIN_SIZE="${VIT_MIN_SIZE:-512}"
MAX_ROUNDS="${MAX_ROUNDS:-3}"
IMAGE_SHAPES="${IMAGE_SHAPES:-720 720}"
SHARD_BATCH="${SHARD_BATCH:-0}"
GPUS_PER_JOB=4

START_SHARD=$((SHARD_BATCH * GPUS_PER_JOB))
SHARD_INDICES=()
for i in $(seq 0 $((GPUS_PER_JOB - 1))); do
    IDX=$((START_SHARD + i))
    if [[ $IDX -lt $NUM_SHARDS ]]; then SHARD_INDICES+=($IDX); fi
done
[[ ${#SHARD_INDICES[@]} -eq 0 ]] && { echo "ERROR: SHARD_BATCH=${SHARD_BATCH} exceeds total shards"; exit 1; }

echo "Task: ${TASK} | Model: ${MODEL_PATH##*/} | Thinking: ${THINKING_MODE} | Shards: ${SHARD_INDICES[*]}/${NUM_SHARDS}"

module load cuda/12.6.0
unset ROCR_VISIBLE_DEVICES
source /path/to/scratch/morph_env/bin/activate
mkdir -p "$OUTPUT_DIR"

THINK_FLAG=""; [[ "$THINK" == "false" ]] && THINK_FLAG="--no_think"
GENERATED_IMAGES_FLAG=""; [[ -n "${GENERATED_IMAGES_DIR:-}" ]] && GENERATED_IMAGES_FLAG="--generated_images_dir $GENERATED_IMAGES_DIR"

PIDS=()
for i in "${!SHARD_INDICES[@]}"; do
    SHARD_IDX="${SHARD_INDICES[$i]}"
    CUDA_VISIBLE_DEVICES="$i" python3 "${SCRIPT_DIR}/inference/run_inference_bagel_spatial.py" \
        --model_path "$MODEL_PATH" --data_dir "$DATA_DIR" --dataset_files $DATASET_FILES \
        --output_file "${OUTPUT_DIR}/inference_results_bagel_shard${SHARD_IDX}.json" \
        --max_mem_per_gpu "$MAX_MEM_PER_GPU" --shard "${SHARD_IDX}/${NUM_SHARDS}" \
        --random_seed "$SEED" --thinking_mode "$THINKING_MODE" \
        --vit_min_size "$VIT_MIN_SIZE" --max_rounds "$MAX_ROUNDS" \
        --image_shapes $IMAGE_SHAPES $THINK_FLAG $GENERATED_IMAGES_FLAG &
    PIDS+=($!)
done

FAIL=0
for pid in "${PIDS[@]}"; do wait "$pid" || FAIL=1; done
[[ "$FAIL" -eq 0 ]] && echo "All shards done." || echo "Some shards failed."
echo "Results: $OUTPUT_DIR"
