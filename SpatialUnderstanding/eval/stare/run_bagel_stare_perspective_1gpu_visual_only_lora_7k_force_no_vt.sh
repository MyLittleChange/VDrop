#!/bin/bash
#SBATCH --job-name=fnvt_stare_visual_only_lora
#SBATCH --output=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/eval/stare/slurm_logs/bagel_stare_perspective_1gpu_visual_only_lora_7k_force_no_vt_output.txt
#SBATCH --error=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/eval/stare/slurm_logs/bagel_stare_perspective_1gpu_visual_only_lora_7k_force_no_vt_error.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --partition=unkillable

# BAGEL STARE Perspective - force_no_visual_thinking
# Model: visual_only_lora_7k
set -euo pipefail
SCRIPT_DIR="/path/to/ThinkMorph-BAGEL-release"

MODEL_PATH="${MODEL_PATH:-/path/to/scratch/VisualCoT/BAGEL_checkpoints/BAGEL_format_training_data_mix_all_balance_visual_only_lora_7k}"
DATA_FILE="${DATA_FILE:-/path/to/scratch/datasets/STARE/perspective/test-00000-of-00001.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-/path/to/scratch/VisualCoT/stare_perspective/BAGEL_format_training_data_mix_all_balance_visual_only_lora_7k_force_no_vt}"

MAX_MEM_PER_GPU="${MAX_MEM_PER_GPU:-70GiB}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SEED="${SEED:-42}"
THINK="${THINK:-false}"
THINKING_MODE="${THINKING_MODE:-visual_only_thinking}"
SHARD_IDX="${SHARD_IDX:-0}"

if [[ ! -d "$MODEL_PATH" ]]; then echo "ERROR: Model not found: $MODEL_PATH"; exit 1; fi
if [[ ! -f "$DATA_FILE" ]]; then echo "ERROR: Data file not found: $DATA_FILE"; exit 1; fi

module load cuda/12.6.0
unset ROCR_VISIBLE_DEVICES
source /path/to/scratch/morph_env/bin/activate
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"

nvidia-smi --query-gpu=index,name,memory.total --format=csv
mkdir -p "$OUTPUT_DIR"

THINK_FLAG=""; [[ "$THINK" == "false" ]] && THINK_FLAG="--no_think"
OUTPUT_FILE="${OUTPUT_DIR}/inference_results_bagel_shard${SHARD_IDX}.json"

echo "Launching shard ${SHARD_IDX}/${NUM_SHARDS} -> ${OUTPUT_FILE}"
python3 "${SCRIPT_DIR}/inference/run_inference_bagel_stare_perspective.py" \
    --model_path "$MODEL_PATH" --data_file "$DATA_FILE" \
    --output_file "$OUTPUT_FILE" --max_mem_per_gpu "$MAX_MEM_PER_GPU" \
    --shard "${SHARD_IDX}/${NUM_SHARDS}" --random_seed "$SEED" \
    --thinking_mode "$THINKING_MODE" --image_shapes 720 1024 \
    --force_no_visual_thinking $THINK_FLAG

echo "Done. Results saved to: $OUTPUT_DIR"
