#!/bin/bash
#SBATCH --job-name=bagel_map_generic
#SBATCH --output=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/slurm_logs/spatial/%x_output.txt
#SBATCH --error=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/slurm_logs/spatial/%x_error.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --partition=unkillable

# Generic BAGEL Map Verification Inference — 1 GPU (OOD-Task)
# Required env vars: MODEL_PATH, OUTPUT_DIR
# Optional env vars: THINK (false), THINKING_MODE (no_thinking)
set -euo pipefail

SCRIPT_DIR="/path/to/ThinkMorph-BAGEL-release"

[[ -z "${MODEL_PATH:-}" ]] && { echo "ERROR: MODEL_PATH is required"; exit 1; }
[[ -z "${OUTPUT_DIR:-}"  ]] && { echo "ERROR: OUTPUT_DIR is required";  exit 1; }
[[ ! -d "$MODEL_PATH"    ]] && { echo "ERROR: Model not found: $MODEL_PATH"; exit 1; }

DATA_FILE="${DATA_FILE:-/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_dataset_map_questions_normalized.json}"
GENERATED_IMAGES_DIR="${GENERATED_IMAGES_DIR:-${OUTPUT_DIR}/generated_images}"
MAX_MEM_PER_GPU="${MAX_MEM_PER_GPU:-70GiB}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SEED="${SEED:-42}"
THINK="${THINK:-false}"
THINKING_MODE="${THINKING_MODE:-no_thinking}"
SHARD_IDX="${SHARD_IDX:-0}"

echo "Map | Model: ${MODEL_PATH##*/} | Thinking: ${THINKING_MODE}"

module load cuda/12.6.0
unset ROCR_VISIBLE_DEVICES
source /path/to/scratch/morph_env/bin/activate
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"
mkdir -p "$OUTPUT_DIR"

THINK_FLAG=""; [[ "$THINK" == "false" ]] && THINK_FLAG="--no_think"
GENERATED_IMAGES_FLAG=""; [[ -n "${GENERATED_IMAGES_DIR:-}" ]] && GENERATED_IMAGES_FLAG="--generated_images_dir $GENERATED_IMAGES_DIR"

python3 "${SCRIPT_DIR}/SpatialUnderstanding/eval/spatial/run_inference_bagel_map.py" \
    --model_path "$MODEL_PATH" \
    --data_file "$DATA_FILE" \
    --output_file "${OUTPUT_DIR}/inference_results_bagel_map_shard${SHARD_IDX}.json" \
    --max_mem_per_gpu "$MAX_MEM_PER_GPU" \
    --shard "${SHARD_IDX}/${NUM_SHARDS}" \
    --random_seed "$SEED" \
    --thinking_mode "$THINKING_MODE" \
    $THINK_FLAG \
    $GENERATED_IMAGES_FLAG

echo "Done. Results: $OUTPUT_DIR"
