#!/bin/bash
#SBATCH --job-name=mmsi_balance_nt_lora10k
#SBATCH --output=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/mmsi_log/mmsi_inference_balance_no_think_lora_10k_output_%a.txt
#SBATCH --error=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/mmsi_log/mmsi_inference_balance_no_think_lora_10k_error_%a.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100l:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --partition=main

#
# MMSI-Bench Inference Script — 2 GPUs per job
# Model: BAGEL_format_mix_all_balance_no_think_lora_10k
# Think: false, Thinking mode: no_thinking
#

set -euo pipefail

SCRIPT_DIR="/path/to/ThinkMorph-BAGEL-release"

# ==================== Configuration ====================
MODEL_PATH="${MODEL_PATH:-/path/to/scratch/VisualCoT/BAGEL_checkpoints/BAGEL_format_mix_all_balance_no_think_lora_10k}"

DATA_SOURCE="${DATA_SOURCE:-parquet}"
PARQUET_PATH="${PARQUET_PATH:-/path/to/scratch/datasets/MMSI-Bench}"

OUTPUT_DIR="${OUTPUT_DIR:-/path/to/scratch/VisualCoT/BAGEL_format_mix_all_balance_no_think_lora_10k_mmsi_eval}"

EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-/path/to/scratch/VisualCoT/mmsi_evaluated}"

MAX_MEM_PER_GPU="${MAX_MEM_PER_GPU:-80GiB}"

NUM_PASSES="${NUM_PASSES:-1}"

NUM_SHARDS="${NUM_SHARDS:-2}"

SEED="${SEED:-42}"

THINKING_MODE="${THINKING_MODE:-no_thinking}"
VIT_MIN_SIZE="${VIT_MIN_SIZE:-512}"
MAX_ROUNDS="${MAX_ROUNDS:-3}"
THINK="${THINK:-false}"
IMAGE_SHAPES="${IMAGE_SHAPES:-720 720}"

JUDGE_API_KEY="${JUDGE_API_KEY:-${VisualCoT_GEMINI:-}}"
JUDGE_MODEL_NAME="${JUDGE_MODEL_NAME:-gemini-3-flash-preview}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"

# ==================== Determine Shard Batch ====================
SHARD_BATCH="${SLURM_ARRAY_TASK_ID:-0}"
GPUS_PER_JOB=2

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

# ==================== Validation ====================
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "ERROR: Model directory not found: $MODEL_PATH"
    exit 1
fi

# ==================== Display Configuration ====================
echo "=============================================="
echo "MMSI-Bench Inference with ThinkMorph (no_thinking)"
echo "=============================================="
echo "Configuration:"
echo "  Model Path:      $MODEL_PATH"
echo "  Data Source:     $DATA_SOURCE"
echo "  Parquet Path:    $PARQUET_PATH"
echo "  Output Dir:      $OUTPUT_DIR"
echo "  Max Mem/GPU:     $MAX_MEM_PER_GPU"
echo "  Num Passes:      $NUM_PASSES"
echo "  Total Shards:    $NUM_SHARDS"
echo "  Shard Batch:     $SHARD_BATCH (shards ${SHARD_INDICES[*]})"
echo "  Seed:            $SEED"
echo "  Thinking Mode:   $THINKING_MODE"
echo "  Think:           $THINK"
echo "=============================================="
echo ""

module load cuda/12.6.0
unset ROCR_VISIBLE_DEVICES
source /path/to/scratch/morph_env/bin/activate

echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo ""

# ==================== Build flags ====================
THINK_FLAG=""
if [[ "$THINK" == "false" ]]; then
    THINK_FLAG="--no_think"
fi

# ==================== Launch shards in parallel, one per GPU ====================
PIDS=()
for i in "${!SHARD_INDICES[@]}"; do
    SHARD_IDX="${SHARD_INDICES[$i]}"
    SHARD_SPEC="${SHARD_IDX}/${NUM_SHARDS}"
    GPU_ID="$i"

    echo "Launching shard ${SHARD_IDX}/${NUM_SHARDS} on GPU ${GPU_ID}"

    CUDA_VISIBLE_DEVICES="${GPU_ID}" python3 "${SCRIPT_DIR}/inference/eval_mmsi_bench.py" \
        --mode inference \
        --model_path "$MODEL_PATH" \
        --data_source "$DATA_SOURCE" \
        --parquet_path "$PARQUET_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --max_mem_per_gpu "$MAX_MEM_PER_GPU" \
        --num_passes "$NUM_PASSES" \
        --seed "$SEED" \
        --shard "$SHARD_SPEC" \
        --thinking_mode "$THINKING_MODE" \
        --vit_min_size "$VIT_MIN_SIZE" \
        --max_rounds "$MAX_ROUNDS" \
        ${IMAGE_SHAPES:+--image_shapes $IMAGE_SHAPES} \
        $THINK_FLAG &

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
