#!/bin/bash
#SBATCH --job-name=bagel_cv_2gpu
#SBATCH --output=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/bagel/bagel_center_view_2gpu_output.txt
#SBATCH --error=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/bagel/bagel_center_view_2gpu_error.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100l:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --partition=main

#
# BAGEL Center View Generation — 2 GPUs on main
#
# Runs 2 shards in parallel, one per GPU, using CUDA_VISIBLE_DEVICES.
# Repeat with different SHARD_BATCH to cover all shards.
#
set -euo pipefail

SCRIPT_DIR="/path/to/ThinkMorph-BAGEL-release"

# ==================== Configuration ====================
# MODEL_PATH="${MODEL_PATH:-/path/to/scratch/models/BAGEL-7B-MoT}"

MODEL_PATH="${MODEL_PATH:-/path/to/scratch/VisualCoT/BAGEL_checkpoints/BAGEL_format_orbit_orbit_annotations_parsed_qa_train_interleaved_thinking}"

ANNOTATION_FILE="${ANNOTATION_FILE:-/path/to/scratch/VisualCoT/annotations/gemini_3_flash_preview_rendered_orbit_orbit_annotations_parsed_qa_test.json}"

OUTPUT_DIR="${OUTPUT_DIR:-/path/to/scratch/VisualCoT/orbit_orbit_annotations_parsed_qa_train_interleaved_thinking_parsed_qa_test}"

MAX_MEM_PER_GPU="${MAX_MEM_PER_GPU:-70GiB}"

NUM_SHARDS="${NUM_SHARDS:-2}"

SEED="${SEED:-42}"

THINK="${THINK:-false}"

THINKING_MODE="${THINKING_MODE:-interleaved_thinking}"

NUM_SAMPLES="${NUM_SAMPLES:-}"

# Which batch of 2 shards to run (0 = shards 0-1, 1 = shards 2-3, etc.)
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
echo "BAGEL Center View Generation (2-GPU)"
echo "=============================================="
echo "Configuration:"
echo "  Model Path:       $MODEL_PATH"
echo "  Annotation File:  $ANNOTATION_FILE"
echo "  Output Dir:       $OUTPUT_DIR"
echo "  Max Mem/GPU:      $MAX_MEM_PER_GPU"
echo "  Total Shards:     $NUM_SHARDS"
echo "  Shard Batch:      $SHARD_BATCH (shards ${SHARD_INDICES[*]})"
echo "  Thinking Mode:    $THINKING_MODE"
echo "  Think Mode:       $THINK"
echo "  Seed:             $SEED"
echo "  Num Samples:      ${NUM_SAMPLES:-all}"
echo "=============================================="
echo ""

# ==================== Validation ====================
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "ERROR: Model directory not found: $MODEL_PATH"
    exit 1
fi

if [[ ! -f "$ANNOTATION_FILE" ]]; then
    echo "ERROR: Annotation file not found: $ANNOTATION_FILE"
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

NUM_SAMPLES_FLAG=""
if [[ -n "$NUM_SAMPLES" ]]; then
    NUM_SAMPLES_FLAG="--num_samples $NUM_SAMPLES"
fi

# ==================== Launch shards in parallel, one per GPU ====================
PIDS=()
for i in "${!SHARD_INDICES[@]}"; do
    SHARD_IDX="${SHARD_INDICES[$i]}"
    SHARD_SPEC="${SHARD_IDX}/${NUM_SHARDS}"
    GPU_ID="$i"

    echo "Launching shard ${SHARD_IDX}/${NUM_SHARDS} on GPU ${GPU_ID}"

    CUDA_VISIBLE_DEVICES="${GPU_ID}" python3 "${SCRIPT_DIR}/SpatialUnderstanding/test_bagel_center_view.py" \
        --model_path "$MODEL_PATH" \
        --annotation_file "$ANNOTATION_FILE" \
        --output_dir "$OUTPUT_DIR" \
        --max_mem_per_gpu "$MAX_MEM_PER_GPU" \
        --random_seed "$SEED" \
        --thinking_mode "$THINKING_MODE" \
        --shard "$SHARD_SPEC" \
        $THINK_FLAG \
        $NUM_SAMPLES_FLAG &

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
