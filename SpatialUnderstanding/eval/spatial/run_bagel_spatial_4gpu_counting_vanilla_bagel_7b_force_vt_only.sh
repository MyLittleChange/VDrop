#!/bin/bash
#SBATCH --job-name=fvt_counting_vanilla
#SBATCH --output=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/bagel/bagel_spatial_4gpu_counting_vanilla_bagel_7b_force_vt_only_output.txt
#SBATCH --error=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/bagel/bagel_spatial_4gpu_counting_vanilla_bagel_7b_force_vt_only_error.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100l:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --partition=main

#
# Vanilla BAGEL with force_visual_thinking ONLY (no bridge KV masking).
# Task: counting | Model: vanilla_bagel_7b
# Companion to the inference-time bridge-mask probe: this row gives the
# "force the model to generate a bridge AND let it use the bridge" baseline,
# so we can compare against the bridge-mask version to isolate the effect of
# blinding the bridge from the effect of forcing visual thinking itself.
#
set -euo pipefail

SCRIPT_DIR="/path/to/ThinkMorph-BAGEL-release"

# ==================== Configuration ====================
MODEL_PATH="${MODEL_PATH:-/path/to/scratch/models/BAGEL-7B-MoT}"

DATA_DIR="${DATA_DIR:-/path/to/scratch/VisualCoT/spatial_collab_dataset}"
DATASET_FILES="${DATASET_FILES:-approved_mcqs_counting_normalized.json}"

OUTPUT_DIR="${OUTPUT_DIR:-/path/to/scratch/VisualCoT/BAGEL_7B_MoT_pretrained_mcqs_counting_normalized_force_vt_only}"

GENERATED_IMAGES_DIR="${GENERATED_IMAGES_DIR:-${OUTPUT_DIR}/generated_images}"

MAX_MEM_PER_GPU="${MAX_MEM_PER_GPU:-70GiB}"
NUM_SHARDS="${NUM_SHARDS:-4}"
SEED="${SEED:-42}"
THINK="${THINK:-false}"

THINKING_MODE="${THINKING_MODE:-visual_only_thinking}"
VIT_MIN_SIZE="${VIT_MIN_SIZE:-512}"
MAX_ROUNDS="${MAX_ROUNDS:-3}"

SHARD_BATCH="${SHARD_BATCH:-0}"
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

echo "=============================================="
echo "BAGEL Spatial — vanilla force_visual_thinking ONLY (counting)"
echo "Model: vanilla_bagel_7b"
echo "=============================================="
echo "  Model Path:      $MODEL_PATH"
echo "  Output Dir:      $OUTPUT_DIR"
echo "  Total Shards:    $NUM_SHARDS"
echo "  Shard Batch:     $SHARD_BATCH (shards ${SHARD_INDICES[*]})"
echo ""

if [[ ! -d "$MODEL_PATH" ]]; then
    echo "ERROR: Model directory not found: $MODEL_PATH"
    exit 1
fi

module load cuda/12.6.0
unset ROCR_VISIBLE_DEVICES
source /path/to/scratch/morph_env/bin/activate

echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo ""

mkdir -p "$OUTPUT_DIR"

THINK_FLAG=""
if [[ "$THINK" == "false" ]]; then
    THINK_FLAG="--no_think"
fi

GENERATED_IMAGES_FLAG=""
if [[ -n "$GENERATED_IMAGES_DIR" ]]; then
    GENERATED_IMAGES_FLAG="--generated_images_dir $GENERATED_IMAGES_DIR"
fi

PIDS=()
for i in "${!SHARD_INDICES[@]}"; do
    SHARD_IDX="${SHARD_INDICES[$i]}"
    SHARD_SPEC="${SHARD_IDX}/${NUM_SHARDS}"
    GPU_ID="$i"
    OUTPUT_FILE="${OUTPUT_DIR}/inference_results_bagel_shard${SHARD_IDX}.json"

    echo "Launching shard ${SHARD_IDX}/${NUM_SHARDS} on GPU ${GPU_ID} -> ${OUTPUT_FILE}"

    CUDA_VISIBLE_DEVICES="${GPU_ID}" python3 "${SCRIPT_DIR}/inference/run_inference_bagel_spatial.py" \
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
        --image_shapes 720 1024 \
        --force_visual_thinking \
        $THINK_FLAG \
        $GENERATED_IMAGES_FLAG &

    PIDS+=($!)
done

echo ""
echo "Waiting for ${#PIDS[@]} shards to complete..."
FAIL=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        echo "ERROR: Process $pid failed"
        FAIL=1
    fi
done

echo "=============================================="
if [[ "$FAIL" -eq 0 ]]; then
    echo "All shards (${SHARD_INDICES[*]}) complete!"
else
    echo "Some shards failed — check output above."
fi
echo "=============================================="
echo "Results saved to: $OUTPUT_DIR"
