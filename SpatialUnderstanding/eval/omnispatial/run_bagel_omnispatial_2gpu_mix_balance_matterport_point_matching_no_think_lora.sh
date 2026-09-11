#!/bin/bash
#SBATCH --job-name=bagel_omni_mat_pm_nt_lora
#SBATCH --output=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/eval/omnispatial/slurm_logs/bagel_omnispatial_mix_balance_matterport_point_matching_no_think_lora_output.txt
#SBATCH --error=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/eval/omnispatial/slurm_logs/bagel_omnispatial_mix_balance_matterport_point_matching_no_think_lora_error.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100l:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --partition=main

#
# BAGEL OmniSpatial Benchmark Inference — 2 GPUs on main
# Model: training_data_mix_balance_matterport_point_matching_no_think_lora
# Think: false, Thinking mode: no_thinking
# Usage:
#   sbatch run_bagel_omnispatial_2gpu_mix_balance_matterport_point_matching_no_think_lora.sh
#
set -euo pipefail

SCRIPT_DIR="/path/to/ThinkMorph-BAGEL-release"

# ==================== Configuration ====================
MODEL_PATH="${MODEL_PATH:-/path/to/scratch/VisualCoT/BAGEL_checkpoints/training_data_mix_balance_matterport_point_matching_no_think_lora}"

DATASET_PATH="${DATASET_PATH:-/path/to/scratch/datasets/OmniSpatial/OmniSpatial-test}"

OUTPUT_DIR="${OUTPUT_DIR:-/path/to/scratch/VisualCoT/omnispatial/training_data_mix_balance_matterport_point_matching_no_think_lora}"

MAX_MEM_PER_GPU="${MAX_MEM_PER_GPU:-70GiB}"

NUM_SHARDS="${NUM_SHARDS:-2}"

SEED="${SEED:-42}"

THINK="${THINK:-false}"
THINKING_MODE="${THINKING_MODE:-no_thinking}"

TASK_TYPE="${TASK_TYPE:-Complex_Logic,Perspective_Taking}"
SUB_TASK_TYPE="${SUB_TASK_TYPE:-}"

SHARD_BATCH="${SHARD_BATCH:-0}"
GPUS_PER_JOB=2

# ==================== Compute shard indices for this batch ====================
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

# ==================== Print Configuration ====================
echo "=============================================="
echo "BAGEL OmniSpatial Benchmark Inference (2-GPU main, no_thinking)"
echo "Model: training_data_mix_balance_matterport_point_matching_no_think_lora"
echo "=============================================="
echo "Configuration:"
echo "  Model Path:      $MODEL_PATH"
echo "  Dataset Path:    $DATASET_PATH"
echo "  Output Dir:      $OUTPUT_DIR"
echo "  Max Mem/GPU:     $MAX_MEM_PER_GPU"
echo "  Total Shards:    $NUM_SHARDS"
echo "  Shard Batch:     $SHARD_BATCH (shards ${SHARD_INDICES[*]})"
echo "  Think Mode:      $THINK"
echo "  Thinking Mode:   $THINKING_MODE"
echo "  Task Type:       ${TASK_TYPE:-<all>}"
echo "  Sub Task Type:   ${SUB_TASK_TYPE:-<all>}"
echo "  Seed:            $SEED"
echo "=============================================="
echo ""

# ==================== Validation ====================
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "ERROR: Model directory not found: $MODEL_PATH"
    exit 1
fi

if [[ ! -f "${DATASET_PATH}/data.json" ]]; then
    echo "ERROR: data.json not found at: ${DATASET_PATH}/data.json"
    exit 1
fi

# ==================== Environment Setup ====================
module load cuda/12.6.0
unset ROCR_VISIBLE_DEVICES
source /path/to/scratch/morph_env/bin/activate

export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"

echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo ""

mkdir -p "$OUTPUT_DIR"

# ==================== Build flags ====================
THINK_FLAG=""
if [[ "$THINK" == "false" ]]; then
    THINK_FLAG="--no_think"
fi

FILTER_FLAGS=""
if [[ -n "$TASK_TYPE" ]]; then
    FILTER_FLAGS="$FILTER_FLAGS --task_type $TASK_TYPE"
fi
if [[ -n "$SUB_TASK_TYPE" ]]; then
    FILTER_FLAGS="$FILTER_FLAGS --sub_task_type $SUB_TASK_TYPE"
fi

# ==================== Launch shards in parallel, one per GPU ====================
PIDS=()
for i in "${!SHARD_INDICES[@]}"; do
    SHARD_IDX="${SHARD_INDICES[$i]}"
    SHARD_SPEC="${SHARD_IDX}/${NUM_SHARDS}"
    GPU_ID="$i"
    OUTPUT_FILE="${OUTPUT_DIR}/inference_results_bagel_shard${SHARD_IDX}.json"

    echo "Launching shard ${SHARD_IDX}/${NUM_SHARDS} on GPU ${GPU_ID} -> ${OUTPUT_FILE}"

    CUDA_VISIBLE_DEVICES="${GPU_ID}" python3 "${SCRIPT_DIR}/inference/run_inference_bagel_omnispatial.py" \
        --model_path "$MODEL_PATH" \
        --dataset_path "$DATASET_PATH" \
        --output_file "$OUTPUT_FILE" \
        --max_mem_per_gpu "$MAX_MEM_PER_GPU" \
        --shard "$SHARD_SPEC" \
        --random_seed "$SEED" \
        --thinking_mode "$THINKING_MODE" \
        $THINK_FLAG \
        $FILTER_FLAGS &

    PIDS+=($!)
done

# ==================== Wait for all to finish ====================
echo ""
echo "Waiting for ${#PIDS[@]} shards to complete..."
FAIL=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        echo "ERROR: Process $pid failed"
        FAIL=1
    fi
done

echo ""
echo "=============================================="
if [[ "$FAIL" -eq 0 ]]; then
    echo "All shards (${SHARD_INDICES[*]}) complete!"
else
    echo "Some shards failed — check output above."
fi
echo "=============================================="
echo "Results saved to: $OUTPUT_DIR"
