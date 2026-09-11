#!/bin/bash
#SBATCH --job-name=bagel_mc_mat_vo9k
#SBATCH --output=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/eval/mindcube/bagel_mindcube_4gpu_mix_balance_matterport_visual_only_lora_9k_output.txt
#SBATCH --error=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/eval/mindcube/bagel_mindcube_4gpu_mix_balance_matterport_visual_only_lora_9k_error.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100l:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --partition=unkillable

# BAGEL MindCube Inference — 4 GPUs (visual_only needs more GPU parallelism)
# Model: BAGEL_format_training_data_mix_balance_matterport_visual_only_lora
# Thinking mode: visual_only_thinking
set -euo pipefail

SCRIPT_DIR="/path/to/ThinkMorph-BAGEL-release"

MODEL_PATH="${MODEL_PATH:-/path/to/scratch/VisualCoT/BAGEL_checkpoints/BAGEL_format_training_data_mix_balance_matterport_visual_only_lora}"
DATA_DIR="${DATA_DIR:-/path/to/scratch/datasets/MindCube/data}"
DATASET_FILES="${DATASET_FILES:-raw/MindCube_tinybench.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/path/to/scratch/mindcube/BAGEL_format_training_data_mix_balance_matterport_visual_only_lora}"
GENERATED_IMAGES_DIR="${GENERATED_IMAGES_DIR:-${OUTPUT_DIR}/generated_images}"
MAX_MEM_PER_GPU="${MAX_MEM_PER_GPU:-70GiB}"
NUM_SHARDS="${NUM_SHARDS:-4}"
SEED="${SEED:-42}"
THINK="${THINK:-false}"
THINKING_MODE="${THINKING_MODE:-visual_only_thinking}"
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

echo "Model: BAGEL_format_training_data_mix_balance_matterport_visual_only_lora"
echo "MindCube | Thinking: ${THINKING_MODE} | Shards: ${SHARD_INDICES[*]}/${NUM_SHARDS}"

[[ ! -d "$MODEL_PATH" ]] && { echo "ERROR: Model not found: $MODEL_PATH"; exit 1; }

module load cuda/12.6.0
unset ROCR_VISIBLE_DEVICES
source /path/to/scratch/morph_env/bin/activate
mkdir -p "$OUTPUT_DIR"

THINK_FLAG=""; [[ "$THINK" == "false" ]] && THINK_FLAG="--no_think"
GENERATED_IMAGES_FLAG=""; [[ -n "$GENERATED_IMAGES_DIR" ]] && GENERATED_IMAGES_FLAG="--generated_images_dir $GENERATED_IMAGES_DIR"

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
