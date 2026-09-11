#!/bin/bash
#SBATCH --job-name=bagel_vsibench_2gpu_panorama
#SBATCH --output=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/eval/vsibench/slurm_logs/bagel_vsibench_2gpu_panorama_output.txt
#SBATCH --error=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/eval/vsibench/slurm_logs/bagel_vsibench_2gpu_panorama_error.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=15:00:00
#SBATCH  --account=rrg-bengioy-ad

#
# BAGEL VSI-Bench Inference — 2 GPUs on Rorqual
# Samples 8 uniform frames per video, passes as multi-image input to BAGEL.
# Runs 2 shards in parallel, one per GPU.
# Usage:
#   sbatch run_bagel_vsibench_2gpu_rorqual.sh
#   SHARD_BATCH=1 sbatch run_bagel_vsibench_2gpu_rorqual.sh
#
set -euo pipefail

SCRIPT_DIR="/path/to/ThinkMorph-BAGEL-release"
SCRATCH="/path/to/scratch"

# ==================== Configuration ====================
MODEL_PATH="${MODEL_PATH:-$SCRATCH/VisualCoT/BAGEL_checkpoints/BAGEL_format_panorama_qa_visual_only}"

DATA_DIR="${DATA_DIR:-$SCRATCH/datasets/VSI-Bench}"
DATASET_FILE="${DATASET_FILE:-test_debiased.parquet}"

OUTPUT_DIR="${OUTPUT_DIR:-$SCRATCH/vsibench/BAGEL_format_panorama_qa_visual_only}"

MAX_MEM_PER_GPU="${MAX_MEM_PER_GPU:-70GiB}"

NUM_SHARDS="${NUM_SHARDS:-2}"

SEED="${SEED:-42}"

THINK="${THINK:-false}"

# Thinking mode: visual_only_thinking, interleaved_thinking, text_only_thinking, no_thinking
THINKING_MODE="${THINKING_MODE:-visual_only_thinking}"
VIT_MIN_SIZE="${VIT_MIN_SIZE:-512}"

NUM_FRAMES="${NUM_FRAMES:-8}"

IMAGE_SHAPES="${IMAGE_SHAPES:-320 1024}"
GENERATED_IMAGES_DIR="${GENERATED_IMAGES_DIR:-${OUTPUT_DIR}/generated_images}"

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
echo "BAGEL VSI-Bench Inference (2-GPU, Rorqual)"
echo "=============================================="
echo "Configuration:"
echo "  Model Path:      $MODEL_PATH"
echo "  Data Dir:        $DATA_DIR"
echo "  Dataset File:    $DATASET_FILE"
echo "  Output Dir:      $OUTPUT_DIR"
echo "  Max Mem/GPU:     $MAX_MEM_PER_GPU"
echo "  Total Shards:    $NUM_SHARDS"
echo "  Shard Batch:     $SHARD_BATCH (shards ${SHARD_INDICES[*]})"
echo "  Think Mode:      $THINK"
echo "  Thinking Mode:   $THINKING_MODE"
echo "  Num Frames:      $NUM_FRAMES"
echo "  Image Shapes:    $IMAGE_SHAPES"
echo "  Seed:            $SEED"
echo "=============================================="
echo ""

# ==================== Validation ====================
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "ERROR: Model directory not found: $MODEL_PATH"
    exit 1
fi

# ==================== Environment Setup ====================
module load cuda/12.6
unset ROCR_VISIBLE_DEVICES
source $SCRATCH/venv_vcot/bin/activate

echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo ""

mkdir -p "$OUTPUT_DIR"
mkdir -p "$(dirname "${SBATCH_STDOUT:-/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/eval/vsibench/slurm_logs/dummy}")"

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

    CUDA_VISIBLE_DEVICES="${GPU_ID}" python3 "${SCRIPT_DIR}/inference/run_inference_bagel_vsibench.py" \
        --model_path "$MODEL_PATH" \
        --data_dir "$DATA_DIR" \
        --dataset_file "$DATASET_FILE" \
        --output_file "$OUTPUT_FILE" \
        --max_mem_per_gpu "$MAX_MEM_PER_GPU" \
        --shard "$SHARD_SPEC" \
        --random_seed "$SEED" \
        --thinking_mode "$THINKING_MODE" \
        --vit_min_size "$VIT_MIN_SIZE" \
        --num_frames "$NUM_FRAMES" \
        --image_shapes $IMAGE_SHAPES \
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
