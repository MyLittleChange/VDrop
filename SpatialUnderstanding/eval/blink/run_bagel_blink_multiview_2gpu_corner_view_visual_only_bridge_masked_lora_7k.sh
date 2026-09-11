#!/bin/bash
#SBATCH --job-name=blink_mv_cv_bm_lora_7k
#SBATCH --output=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/eval/blink/bagel_blink_multiview_2gpu_corner_view_visual_only_bridge_masked_lora_7k_output.txt
#SBATCH --error=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/eval/blink/bagel_blink_multiview_2gpu_corner_view_visual_only_bridge_masked_lora_7k_error.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100l:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --partition=main

# BAGEL BLINK Multi-view_Reasoning Inference — 2 GPUs
# Model: BAGEL_format_corner_view_visual_only_bridge_masked_lora_7k
# Thinking mode: visual_only_thinking
set -euo pipefail

SCRIPT_DIR="/path/to/ThinkMorph-BAGEL-release"

MODEL_PATH="${MODEL_PATH:-/path/to/scratch/VisualCoT/BAGEL_checkpoints/BAGEL_format_corner_view_visual_only_bridge_masked_lora_7k}"
DATA_DIR="${DATA_DIR:-/path/to/scratch/datasets/BLINK}"
TASK="${TASK:-Multi-view_Reasoning}"
SPLIT="${SPLIT:-val}"
OUTPUT_DIR="${OUTPUT_DIR:-/path/to/scratch/blink/BAGEL_format_corner_view_visual_only_bridge_masked_lora_7k}"
MAX_MEM_PER_GPU="${MAX_MEM_PER_GPU:-80GiB}"
NUM_SHARDS="${NUM_SHARDS:-2}"
SEED="${SEED:-42}"
THINK="${THINK:-false}"
THINKING_MODE="${THINKING_MODE:-visual_only_thinking}"
VIT_MIN_SIZE="${VIT_MIN_SIZE:-512}"
SHARD_BATCH="${SHARD_BATCH:-0}"
GPUS_PER_JOB=2

START_SHARD=$((SHARD_BATCH * GPUS_PER_JOB))
SHARD_INDICES=()
for i in $(seq 0 $((GPUS_PER_JOB - 1))); do
    IDX=$((START_SHARD + i))
    if [[ $IDX -lt $NUM_SHARDS ]]; then SHARD_INDICES+=($IDX); fi
done
[[ ${#SHARD_INDICES[@]} -eq 0 ]] && { echo "ERROR: SHARD_BATCH=${SHARD_BATCH} exceeds total shards"; exit 1; }

echo "Model: BAGEL_format_corner_view_visual_only_bridge_masked_lora_7k"
echo "BLINK | Thinking: ${THINKING_MODE} | Shards: ${SHARD_INDICES[*]}/${NUM_SHARDS}"

[[ ! -d "$MODEL_PATH" ]] && { echo "ERROR: Model not found: $MODEL_PATH"; exit 1; }

module load cuda/12.6.0
unset ROCR_VISIBLE_DEVICES
source /path/to/scratch/morph_env/bin/activate
mkdir -p "$OUTPUT_DIR"

THINK_FLAG=""; [[ "$THINK" == "false" ]] && THINK_FLAG="--no_think"

PIDS=()
for i in "${!SHARD_INDICES[@]}"; do
    SHARD_IDX="${SHARD_INDICES[$i]}"
    CUDA_VISIBLE_DEVICES="$i" python3 "${SCRIPT_DIR}/inference/run_inference_bagel_blink_multiview.py" \
        --model_path "$MODEL_PATH" \
        --data_dir "$DATA_DIR" --task "$TASK" --split "$SPLIT" \
        --output_file "${OUTPUT_DIR}/inference_results_shard${SHARD_IDX}.json" \
        --max_mem_per_gpu "$MAX_MEM_PER_GPU" --shard "${SHARD_IDX}/${NUM_SHARDS}" \
        --random_seed "$SEED" --thinking_mode "$THINKING_MODE" \
        --vit_min_size "$VIT_MIN_SIZE" \
        $THINK_FLAG &
    PIDS+=($!)
done

FAIL=0
for pid in "${PIDS[@]}"; do wait "$pid" || FAIL=1; done
[[ "$FAIL" -eq 0 ]] && echo "All shards done." || echo "Some shards failed."
echo "Results: $OUTPUT_DIR"
