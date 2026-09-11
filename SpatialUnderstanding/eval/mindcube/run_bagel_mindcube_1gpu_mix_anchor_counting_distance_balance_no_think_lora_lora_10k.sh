#!/bin/bash
#SBATCH --job-name=bagel_mindcube_1gpu_acd_nt_lora10k
#SBATCH --output=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/eval/mindcube/bagel_mindcube_1gpu_mix_anchor_counting_distance_balance_no_think_lora_lora_10k_output.txt
#SBATCH --error=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/eval/mindcube/bagel_mindcube_1gpu_mix_anchor_counting_distance_balance_no_think_lora_lora_10k_error.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --partition=unkillable

#
# BAGEL MindCube Inference — 1 GPU on unkillable
# Model: BAGEL_format_mix_anchor_counting_distance_balance_no_think_lora_lora_10k
# Think: false, Thinking mode: no_thinking
# Generates square images (720x720)
# Usage:
#   sbatch run_bagel_mindcube_1gpu_mix_anchor_counting_distance_balance_no_think_lora_lora_10k.sh
#
set -euo pipefail

SCRIPT_DIR="/path/to/ThinkMorph-BAGEL-release"

# ==================== Configuration ====================
MODEL_PATH="${MODEL_PATH:-/path/to/scratch/VisualCoT/BAGEL_checkpoints/BAGEL_format_mix_anchor_counting_distance_balance_no_think_lora_lora_10k}"

DATA_DIR="${DATA_DIR:-/path/to/scratch/datasets/MindCube/data}"
DATASET_FILES="${DATASET_FILES:-raw/MindCube_tinybench.jsonl}"

OUTPUT_DIR="${OUTPUT_DIR:-/path/to/scratch/mindcube/BAGEL_format_mix_anchor_counting_distance_balance_no_think_lora_lora_10k}"

GENERATED_IMAGES_DIR="${GENERATED_IMAGES_DIR:-${OUTPUT_DIR}/generated_images}"

MAX_MEM_PER_GPU="${MAX_MEM_PER_GPU:-70GiB}"

NUM_SHARDS="${NUM_SHARDS:-1}"

SEED="${SEED:-42}"

THINK="${THINK:-false}"

# Thinking mode: visual_only_thinking, interleaved_thinking, text_only_thinking, or no_thinking
THINKING_MODE="${THINKING_MODE:-no_thinking}"
VIT_MIN_SIZE="${VIT_MIN_SIZE:-512}"
MAX_ROUNDS="${MAX_ROUNDS:-3}"

# Which batch of 1 shard to run (0 = shard 0, 1 = shard 1, etc.)
SHARD_BATCH="${SHARD_BATCH:-0}"
GPUS_PER_JOB=1

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
echo "BAGEL MindCube Inference (1-GPU unkillable, no_thinking)"
echo "=============================================="
echo "Configuration:"
echo "  Model Path:      $MODEL_PATH"
echo "  Data Dir:        $DATA_DIR"
echo "  Dataset Files:   $DATASET_FILES"
echo "  Output Dir:      $OUTPUT_DIR"
echo "  Gen Images Dir:  $GENERATED_IMAGES_DIR"
echo "  Max Mem/GPU:     $MAX_MEM_PER_GPU"
echo "  Total Shards:    $NUM_SHARDS"
echo "  Shard Batch:     $SHARD_BATCH (shards ${SHARD_INDICES[*]})"
echo "  Think Mode:      $THINK"
echo "  Thinking Mode:   $THINKING_MODE"
echo "  Image Shapes:    720 720"
echo "  Seed:            $SEED"
echo "=============================================="
echo ""

# ==================== Validation ====================
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "ERROR: Model directory not found: $MODEL_PATH"
    exit 1
fi

# ==================== Environment Setup ====================
module load cuda/12.6.0
unset ROCR_VISIBLE_DEVICES
source /path/to/scratch/morph_env/bin/activate

echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo ""

mkdir -p "$OUTPUT_DIR"

# ==================== Build flags ====================
THINK_FLAG=""
if [[ "$THINK" == "false" ]]; then
    THINK_FLAG="--no_think"
fi

GENERATED_IMAGES_FLAG=""
if [[ -n "$GENERATED_IMAGES_DIR" ]]; then
    GENERATED_IMAGES_FLAG="--generated_images_dir $GENERATED_IMAGES_DIR"
fi

# ==================== Launch shards in parallel, one per GPU ====================
PIDS=()
for i in "${!SHARD_INDICES[@]}"; do
    SHARD_IDX="${SHARD_INDICES[$i]}"
    SHARD_SPEC="${SHARD_IDX}/${NUM_SHARDS}"
    GPU_ID="$i"
    OUTPUT_FILE="${OUTPUT_DIR}/inference_results_shard${SHARD_IDX}.json"

    echo "Launching shard ${SHARD_IDX}/${NUM_SHARDS} on GPU ${GPU_ID} -> ${OUTPUT_FILE}"

    CUDA_VISIBLE_DEVICES="${GPU_ID}" python3 "${SCRIPT_DIR}/inference/run_inference_bagel_mindcube.py" \
        --model_path "$MODEL_PATH" \
        --data_dir "$DATA_DIR" \
        --dataset_files $DATASET_FILES \
        --output_file "$OUTPUT_FILE" \
        --max_mem_per_gpu "$MAX_MEM_PER_GPU" \
        --shard "$SHARD_SPEC" \
        --random_seed "$SEED" \
        --thinking_mode "$THINKING_MODE" \
        --vit_min_size "$VIT_MIN_SIZE" \
        --max_rounds "$MAX_ROUNDS" \
        --image_shapes 720 720 \
        $THINK_FLAG \
        $GENERATED_IMAGES_FLAG &

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
