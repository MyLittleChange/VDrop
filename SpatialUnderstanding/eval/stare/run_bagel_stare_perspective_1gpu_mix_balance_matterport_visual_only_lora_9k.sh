#!/bin/bash
#SBATCH --job-name=bagel_stare_mat_vo9k
#SBATCH --output=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/eval/stare/slurm_logs/bagel_stare_perspective_mix_balance_matterport_visual_only_lora_9k_output.txt
#SBATCH --error=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/eval/stare/slurm_logs/bagel_stare_perspective_mix_balance_matterport_visual_only_lora_9k_error.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --partition=unkillable

# BAGEL STARE Perspective Inference — 1 GPU
# Model: BAGEL_format_training_data_mix_balance_matterport_visual_only_lora
# Thinking mode: visual_only_thinking
set -euo pipefail

SCRIPT_DIR="/path/to/ThinkMorph-BAGEL-release"

MODEL_PATH="${MODEL_PATH:-/path/to/scratch/VisualCoT/BAGEL_checkpoints/BAGEL_format_training_data_mix_balance_matterport_visual_only_lora}"
DATA_FILE="${DATA_FILE:-/path/to/scratch/datasets/STARE/perspective/test-00000-of-00001.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-/path/to/scratch/VisualCoT/stare_perspective/BAGEL_format_training_data_mix_balance_matterport_visual_only_lora}"
MAX_MEM_PER_GPU="${MAX_MEM_PER_GPU:-70GiB}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SEED="${SEED:-42}"
THINK="${THINK:-false}"
THINKING_MODE="${THINKING_MODE:-visual_only_thinking}"
SHARD_IDX="${SHARD_IDX:-0}"

echo "Model: BAGEL_format_training_data_mix_balance_matterport_visual_only_lora"
echo "STARE | Thinking: ${THINKING_MODE}"

[[ ! -d "$MODEL_PATH" ]] && { echo "ERROR: Model not found: $MODEL_PATH"; exit 1; }

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
    $THINK_FLAG

echo "Done. Results: $OUTPUT_DIR"
