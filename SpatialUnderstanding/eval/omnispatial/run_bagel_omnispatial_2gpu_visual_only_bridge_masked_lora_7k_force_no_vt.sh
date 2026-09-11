#!/bin/bash
#SBATCH --job-name=fnvt_omni_visual_only_bridg
#SBATCH --output=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/eval/omnispatial/slurm_logs/bagel_omnispatial_2gpu_visual_only_bridge_masked_lora_7k_force_no_vt_output.txt
#SBATCH --error=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/eval/omnispatial/slurm_logs/bagel_omnispatial_2gpu_visual_only_bridge_masked_lora_7k_force_no_vt_error.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100l:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --partition=main

# BAGEL OmniSpatial - force_no_visual_thinking
# Model: visual_only_bridge_masked_lora_7k
# Subset: Complex_Logic + Perspective_Taking
set -euo pipefail
SCRIPT_DIR="/path/to/ThinkMorph-BAGEL-release"

MODEL_PATH="${MODEL_PATH:-/path/to/scratch/VisualCoT/BAGEL_checkpoints/BAGEL_format_mix_all_balance_visual_only_bridge_masked_lora_7k}"
DATASET_PATH="${DATASET_PATH:-/path/to/scratch/datasets/OmniSpatial/OmniSpatial-test}"
OUTPUT_DIR="${OUTPUT_DIR:-/path/to/scratch/VisualCoT/omnispatial/BAGEL_format_mix_all_balance_visual_only_bridge_masked_lora_7k_complex_logic_perspective_taking_force_no_vt}"

MAX_MEM_PER_GPU="${MAX_MEM_PER_GPU:-70GiB}"
NUM_SHARDS="${NUM_SHARDS:-2}"
SEED="${SEED:-42}"
THINK="${THINK:-false}"
THINKING_MODE="${THINKING_MODE:-visual_only_thinking}"
TASK_TYPE="${TASK_TYPE:-Complex_Logic,Perspective_Taking}"
SUB_TASK_TYPE="${SUB_TASK_TYPE:-}"

SHARD_BATCH="${SHARD_BATCH:-0}"
GPUS_PER_JOB=2
START_SHARD=$((SHARD_BATCH * GPUS_PER_JOB))
SHARD_INDICES=()
for i in $(seq 0 $((GPUS_PER_JOB - 1))); do
    IDX=$((START_SHARD + i))
    if [[ $IDX -lt $NUM_SHARDS ]]; then SHARD_INDICES+=($IDX); fi
done
if [[ ${#SHARD_INDICES[@]} -eq 0 ]]; then echo "ERROR: SHARD_BATCH=${SHARD_BATCH} exceeds total shards"; exit 1; fi

if [[ ! -d "$MODEL_PATH" ]]; then echo "ERROR: Model not found: $MODEL_PATH"; exit 1; fi
if [[ ! -f "${DATASET_PATH}/data.json" ]]; then echo "ERROR: data.json not found at ${DATASET_PATH}/data.json"; exit 1; fi

module load cuda/12.6.0
unset ROCR_VISIBLE_DEVICES
source /path/to/scratch/morph_env/bin/activate
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"

nvidia-smi --query-gpu=index,name,memory.total --format=csv
mkdir -p "$OUTPUT_DIR"

THINK_FLAG=""; [[ "$THINK" == "false" ]] && THINK_FLAG="--no_think"
FILTER_FLAGS=""
[[ -n "$TASK_TYPE" ]] && FILTER_FLAGS="$FILTER_FLAGS --task_type $TASK_TYPE"
[[ -n "$SUB_TASK_TYPE" ]] && FILTER_FLAGS="$FILTER_FLAGS --sub_task_type $SUB_TASK_TYPE"

PIDS=()
for i in "${!SHARD_INDICES[@]}"; do
    SHARD_IDX="${SHARD_INDICES[$i]}"
    SHARD_SPEC="${SHARD_IDX}/${NUM_SHARDS}"
    GPU_ID="$i"
    OUTPUT_FILE="${OUTPUT_DIR}/inference_results_bagel_shard${SHARD_IDX}.json"
    echo "Launching shard ${SHARD_IDX}/${NUM_SHARDS} on GPU ${GPU_ID} -> ${OUTPUT_FILE}"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" python3 "${SCRIPT_DIR}/inference/run_inference_bagel_omnispatial.py" \
        --model_path "$MODEL_PATH" --dataset_path "$DATASET_PATH" \
        --output_file "$OUTPUT_FILE" --max_mem_per_gpu "$MAX_MEM_PER_GPU" \
        --shard "$SHARD_SPEC" --random_seed "$SEED" --thinking_mode "$THINKING_MODE" \
        --image_shapes 720 1024 --force_no_visual_thinking $THINK_FLAG $FILTER_FLAGS &
    PIDS+=($!)
done

FAIL=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then echo "ERROR: Process $pid failed"; FAIL=1; fi
done
[[ "$FAIL" -eq 0 ]] && echo "All shards complete!" || echo "Some shards failed."
echo "Results saved to: $OUTPUT_DIR"
