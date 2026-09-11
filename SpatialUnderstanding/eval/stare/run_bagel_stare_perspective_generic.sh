#!/bin/bash
#SBATCH --job-name=bagel_stare_generic
#SBATCH --output=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/slurm_logs/stare/%x_output.txt
#SBATCH --error=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/slurm_logs/stare/%x_error.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --partition=unkillable

# Generic BAGEL STARE Perspective Inference — 1 GPU
# Required env vars: MODEL_PATH, OUTPUT_DIR
# Optional env vars: THINK (false), THINKING_MODE (no_thinking)
set -euo pipefail

SCRIPT_DIR="/path/to/ThinkMorph-BAGEL-release"

[[ -z "${MODEL_PATH:-}" ]] && { echo "ERROR: MODEL_PATH is required"; exit 1; }
[[ -z "${OUTPUT_DIR:-}"  ]] && { echo "ERROR: OUTPUT_DIR is required";  exit 1; }
[[ ! -d "$MODEL_PATH"    ]] && { echo "ERROR: Model not found: $MODEL_PATH"; exit 1; }

DATA_FILE="${DATA_FILE:-/path/to/scratch/datasets/STARE/perspective/test-00000-of-00001.parquet}"
MAX_MEM_PER_GPU="${MAX_MEM_PER_GPU:-70GiB}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SEED="${SEED:-42}"
THINK="${THINK:-false}"
THINKING_MODE="${THINKING_MODE:-no_thinking}"
MAX_ROUNDS="${MAX_ROUNDS:-3}"
SHARD_IDX="${SHARD_IDX:-0}"

echo "STARE | Model: ${MODEL_PATH##*/} | Thinking: ${THINKING_MODE}"

module load cuda/12.6.0
unset ROCR_VISIBLE_DEVICES
source /path/to/scratch/morph_env/bin/activate
mkdir -p "$OUTPUT_DIR"

THINK_FLAG=""; [[ "$THINK" == "false" ]] && THINK_FLAG="--no_think"

python3 "${SCRIPT_DIR}/inference/run_inference_bagel_stare_perspective.py" \
    --model_path "$MODEL_PATH" \
    --data_file "$DATA_FILE" \
    --output_file "${OUTPUT_DIR}/inference_results_bagel_shard${SHARD_IDX}.json" \
    --max_mem_per_gpu "$MAX_MEM_PER_GPU" \
    --shard "${SHARD_IDX}/${NUM_SHARDS}" \
    --random_seed "$SEED" \
    --thinking_mode "$THINKING_MODE" \
    --max_rounds "$MAX_ROUNDS" \
    $THINK_FLAG

echo "Done. Results: $OUTPUT_DIR"
