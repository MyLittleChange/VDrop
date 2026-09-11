#!/bin/bash
#SBATCH --job-name=bagel_vsibench_4gpu
#SBATCH --output=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/eval/vsibench/bagel_vsibench_4gpu_output.txt
#SBATCH --error=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/eval/vsibench/bagel_vsibench_4gpu_error.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100l:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=3:00:00
#SBATCH --partition=short-unkillable

#
# BAGEL VSI-Bench Inference — 4 GPUs on short-unkillable
# Samples 8 uniform frames per video, passes as multi-image input to BAGEL.
# Runs 4 shards in parallel, one per GPU.
# Usage:
#   sbatch run_bagel_vsibench_4gpu.sh
#   SHARD_BATCH=1 sbatch run_bagel_vsibench_4gpu.sh
#
set -euo pipefail

SCRIPT_DIR="/path/to/ThinkMorph-BAGEL-release"

# ==================== Configuration ====================
MODEL_PATH="${MODEL_PATH:-$SCRATCH/VisualCoT/BAGEL_checkpoints/BAGEL_format_training_data_mix_all_rotation_visual_only}"

DATA_DIR="${DATA_DIR:-/path/to/scratch/datasets/VSI-Bench}"
DATASET_FILE="${DATASET_FILE:-test_debiased.parquet}"

OUTPUT_DIR="${OUTPUT_DIR:-/path/to/scratch/vsibench/BAGEL_format_training_data_mix_all_rotation_visual_only}"

MAX_MEM_PER_GPU="${MAX_MEM_PER_GPU:-70GiB}"

NUM_SHARDS="${NUM_SHARDS:-4}"

SEED="${SEED:-42}"

THINK="${THINK:-false}"

# Thinking mode: visual_only_thinking, interleaved_thinking, text_only_thinking, no_thinking
THINKING_MODE="${THINKING_MODE:-visual_only_thinking}"
VIT_MIN_SIZE="${VIT_MIN_SIZE:-512}"

NUM_FRAMES="${NUM_FRAMES:-8}"

GENERATED_IMAGES_DIR="${GENERATED_IMAGES_DIR:-${OUTPUT_DIR}/generated_images}"

# Which batch of 4 shards to run (0 = shards 0-3, 1 = shards 4-7, etc.)
SHARD_BATCH="${SHARD_BATCH:-0}"
GPUS_PER_JOB=4

# Resume mode: set RESUME=true to filter out already-completed samples and split the
# remainder across NUM_SHARDS GPUs. Outputs go to inference_results_resume_shard{N}.json.
RESUME="${RESUME:-false}"
RESUME_SHARD0="${RESUME_SHARD0:-inference_results_shard0.json}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-inference_results_shard1_checkpoint.json}"
RESUME_DATASET_FILE="${RESUME_DATASET_FILE:-remaining_samples.jsonl}"

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
echo "BAGEL VSI-Bench Inference (4-GPU)"
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
echo "  Gen Images Dir:  $GENERATED_IMAGES_DIR"
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

# ==================== Resume mode: generate filtered JSONL ====================
if [[ "$RESUME" == "true" ]]; then
    echo "=== RESUME MODE: generating remaining samples JSONL ==="
    python3 "${SCRIPT_DIR}/SpatialUnderstanding/eval/vsibench/prepare_resume_dataset.py" \
        --output_dir "$OUTPUT_DIR" \
        --data_dir "$DATA_DIR" \
        --dataset_file "$DATASET_FILE" \
        --shard0_file "$RESUME_SHARD0" \
        --checkpoint_file "$RESUME_CHECKPOINT" \
        --out_file "$RESUME_DATASET_FILE"
    # Use absolute path so os.path.join(data_dir, dataset_file) resolves to the JSONL,
    # while DATA_DIR stays as-is for video lookups (data_dir/dataset/scene.mp4).
    DATASET_FILE="${OUTPUT_DIR}/${RESUME_DATASET_FILE}"
    echo "=== Resuming with filtered dataset: ${DATASET_FILE} ==="
fi

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
    if [[ "$RESUME" == "true" ]]; then
        OUTPUT_FILE="${OUTPUT_DIR}/inference_results_resume_shard${SHARD_IDX}.json"
    else
        OUTPUT_FILE="${OUTPUT_DIR}/inference_results_shard${SHARD_IDX}.json"
    fi

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
