#!/bin/bash
#SBATCH --job-name=bagel_map_all_bm_map_lora_7k
#SBATCH --output=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/bagel/bagel_map_4gpu_mix_all_balance_with_map_visual_only_bridge_masked_lora_7k_output.txt
#SBATCH --error=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/bagel/bagel_map_4gpu_mix_all_balance_with_map_visual_only_bridge_masked_lora_7k_error.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100l:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --partition=long

#
# BAGEL Map Verification Inference — 2 GPUs on main
# Model: BAGEL_format_mix_all_balance_with_map_visual_only_bridge_masked_lora_7k
# Think: false, Thinking mode: visual_only_thinking
#
set -euo pipefail

SCRIPT_DIR="/path/to/ThinkMorph-BAGEL-release"

MODEL_PATH="${MODEL_PATH:-/path/to/scratch/VisualCoT/BAGEL_checkpoints/BAGEL_format_mix_all_balance_with_map_visual_only_bridge_masked_lora_7k}"
DATA_FILE="${DATA_FILE:-/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_dataset_map_questions_normalized.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/path/to/scratch/VisualCoT/BAGEL_format_mix_all_balance_with_map_visual_only_bridge_masked_lora_7k_map}"
GENERATED_IMAGES_DIR="${GENERATED_IMAGES_DIR:-${OUTPUT_DIR}/generated_images}"

MAX_MEM_PER_GPU="${MAX_MEM_PER_GPU:-70GiB}"
NUM_SHARDS="${NUM_SHARDS:-2}"
SEED="${SEED:-42}"
THINK="${THINK:-false}"
THINKING_MODE="${THINKING_MODE:-visual_only_thinking}"

SHARD_BATCH="${SHARD_BATCH:-0}"
GPUS_PER_JOB=2

START_SHARD=$((SHARD_BATCH * GPUS_PER_JOB))
SHARD_INDICES=()
for i in $(seq 0 $((GPUS_PER_JOB - 1))); do
    IDX=$((START_SHARD + i))
    if [[ $IDX -lt $NUM_SHARDS ]]; then SHARD_INDICES+=($IDX); fi
done
if [[ ${#SHARD_INDICES[@]} -eq 0 ]]; then echo "ERROR: SHARD_BATCH=${SHARD_BATCH} exceeds total shards"; exit 1; fi

echo "BAGEL Map (visual_only_thinking) — Model: $MODEL_PATH | Shards: $NUM_SHARDS | Batch: $SHARD_BATCH"

if [[ ! -d "$MODEL_PATH" ]]; then echo "ERROR: Model not found: $MODEL_PATH"; exit 1; fi

module load cuda/12.6.0
unset ROCR_VISIBLE_DEVICES
source /path/to/scratch/morph_env/bin/activate
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"

nvidia-smi --query-gpu=index,name,memory.total --format=csv
mkdir -p "$OUTPUT_DIR"

THINK_FLAG=""; [[ "$THINK" == "false" ]] && THINK_FLAG="--no_think"
GENERATED_IMAGES_FLAG=""; [[ -n "$GENERATED_IMAGES_DIR" ]] && GENERATED_IMAGES_FLAG="--generated_images_dir $GENERATED_IMAGES_DIR"

PIDS=()
for i in "${!SHARD_INDICES[@]}"; do
    SHARD_IDX="${SHARD_INDICES[$i]}"
    GPU_ID="$i"
    OUTPUT_FILE="${OUTPUT_DIR}/inference_results_bagel_map_shard${SHARD_IDX}.json"
    echo "Launching shard ${SHARD_IDX}/${NUM_SHARDS} on GPU ${GPU_ID} -> ${OUTPUT_FILE}"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" python3 "${SCRIPT_DIR}/SpatialUnderstanding/eval/spatial/run_inference_bagel_map.py" \
        --model_path "$MODEL_PATH" --data_file "$DATA_FILE" \
        --output_file "$OUTPUT_FILE" --max_mem_per_gpu "$MAX_MEM_PER_GPU" \
        --shard "${SHARD_IDX}/${NUM_SHARDS}" --random_seed "$SEED" \
        --thinking_mode "$THINKING_MODE" $THINK_FLAG $GENERATED_IMAGES_FLAG &
    PIDS+=($!)
done

FAIL=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then echo "ERROR: Process $pid failed"; FAIL=1; fi
done
[[ "$FAIL" -eq 0 ]] && echo "All shards complete!" || echo "Some shards failed."
echo "Results saved to: $OUTPUT_DIR"
