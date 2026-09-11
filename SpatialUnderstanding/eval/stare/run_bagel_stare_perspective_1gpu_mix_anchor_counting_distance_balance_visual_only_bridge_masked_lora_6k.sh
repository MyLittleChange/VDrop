#!/bin/bash
#SBATCH --job-name=bagel_stare_all_bm_lora_6k
#SBATCH --output=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/eval/stare/slurm_logs/bagel_stare_perspective_mix_anchor_counting_distance_balance_visual_only_bridge_masked_lora_6k_output.txt
#SBATCH --error=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/eval/stare/slurm_logs/bagel_stare_perspective_mix_anchor_counting_distance_balance_visual_only_bridge_masked_lora_6k_error.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100l:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --partition=main

#
# BAGEL STARE Perspective Inference — 2 GPUs on main
# Model: BAGEL_format_mix_anchor_counting_distance_balance_visual_only_bridge_masked_lora_6k
# Think: false, Thinking mode: visual_only_thinking
# 2 parallel shards, one per GPU
#
set -euo pipefail

SCRIPT_DIR="/path/to/ThinkMorph-BAGEL-release"

MODEL_PATH="${MODEL_PATH:-/path/to/scratch/VisualCoT/BAGEL_checkpoints/BAGEL_format_mix_anchor_counting_distance_balance_visual_only_bridge_masked_lora_6k}"
DATA_FILE="${DATA_FILE:-/path/to/scratch/datasets/STARE/perspective/test-00000-of-00001.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-/path/to/scratch/VisualCoT/stare_perspective/BAGEL_format_mix_anchor_counting_distance_balance_visual_only_bridge_masked_lora_6k}"

MAX_MEM_PER_GPU="${MAX_MEM_PER_GPU:-70GiB}"
NUM_SHARDS="${NUM_SHARDS:-2}"
SEED="${SEED:-42}"
THINK="${THINK:-false}"
THINKING_MODE="${THINKING_MODE:-visual_only_thinking}"

SHARD_BATCH="${SHARD_BATCH:-0}"
GPUS_PER_JOB=2

# Compute shard indices for this batch
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

echo "BAGEL STARE Perspective (2-GPU main, visual_only_thinking) — Model: $MODEL_PATH | Shards: ${SHARD_INDICES[*]}/$NUM_SHARDS"

if [[ ! -d "$MODEL_PATH" ]]; then echo "ERROR: Model not found: $MODEL_PATH"; exit 1; fi
if [[ ! -f "$DATA_FILE" ]]; then echo "ERROR: Data file not found: $DATA_FILE"; exit 1; fi

module load cuda/12.6.0
unset ROCR_VISIBLE_DEVICES
source /path/to/scratch/morph_env/bin/activate
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"

nvidia-smi --query-gpu=index,name,memory.total --format=csv
mkdir -p "$OUTPUT_DIR"

THINK_FLAG=""; [[ "$THINK" == "false" ]] && THINK_FLAG="--no_think"

PIDS=()
for i in "${!SHARD_INDICES[@]}"; do
    SHARD_IDX="${SHARD_INDICES[$i]}"
    SHARD_SPEC="${SHARD_IDX}/${NUM_SHARDS}"
    GPU_ID="$i"
    OUTPUT_FILE="${OUTPUT_DIR}/inference_results_bagel_shard${SHARD_IDX}.json"

    echo "Launching shard ${SHARD_IDX}/${NUM_SHARDS} on GPU ${GPU_ID} -> ${OUTPUT_FILE}"

    CUDA_VISIBLE_DEVICES="${GPU_ID}" python3 "${SCRIPT_DIR}/inference/run_inference_bagel_stare_perspective.py" \
        --model_path "$MODEL_PATH" --data_file "$DATA_FILE" \
        --output_file "$OUTPUT_FILE" --max_mem_per_gpu "$MAX_MEM_PER_GPU" \
        --shard "$SHARD_SPEC" --random_seed "$SEED" \
        --thinking_mode "$THINKING_MODE" --image_shapes 720 1024 $THINK_FLAG &

    PIDS+=($!)
done

echo "Waiting for ${#PIDS[@]} shards to complete..."
FAIL=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        echo "ERROR: Process $pid failed"
        FAIL=1
    fi
done

if [[ "$FAIL" -eq 0 ]]; then
    echo "All shards (${SHARD_INDICES[*]}) complete!"
else
    echo "Some shards failed — check output above."
fi
echo "Results saved to: $OUTPUT_DIR"
