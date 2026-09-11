#!/bin/bash
#SBATCH --job-name=bagel_stare_mat_pm_nt_lora
#SBATCH --output=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/eval/stare/slurm_logs/bagel_stare_perspective_mix_balance_matterport_point_matching_no_think_lora_output.txt
#SBATCH --error=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/eval/stare/slurm_logs/bagel_stare_perspective_mix_balance_matterport_point_matching_no_think_lora_error.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G

#SBATCH --partition=unkillable
#
# BAGEL STARE Perspective Inference — 1 GPU on unkillable
# Model: training_data_mix_balance_matterport_point_matching_no_think_lora
# Think: false, Thinking mode: no_thinking
# Usage:
#   sbatch run_bagel_stare_perspective_1gpu_mix_balance_matterport_point_matching_no_think_lora.sh
#
set -euo pipefail

SCRIPT_DIR="/path/to/ThinkMorph-BAGEL-release"

# ==================== Configuration ====================
MODEL_PATH="${MODEL_PATH:-/path/to/scratch/VisualCoT/BAGEL_checkpoints/training_data_mix_balance_matterport_point_matching_no_think_lora}"

DATA_FILE="${DATA_FILE:-/path/to/scratch/datasets/STARE/perspective/test-00000-of-00001.parquet}"

OUTPUT_DIR="${OUTPUT_DIR:-/path/to/scratch/VisualCoT/stare_perspective/training_data_mix_balance_matterport_point_matching_no_think_lora}"

MAX_MEM_PER_GPU="${MAX_MEM_PER_GPU:-70GiB}"

NUM_SHARDS="${NUM_SHARDS:-1}"

SEED="${SEED:-42}"

THINK="${THINK:-false}"
THINKING_MODE="${THINKING_MODE:-no_thinking}"

SHARD_IDX="${SHARD_IDX:-0}"

# ==================== Print Configuration ====================
echo "=============================================="
echo "BAGEL STARE Perspective Inference (1-GPU unkillable, no_thinking)"
echo "Model: training_data_mix_balance_matterport_point_matching_no_think_lora"
echo "=============================================="
echo "Configuration:"
echo "  Model Path:      $MODEL_PATH"
echo "  Data File:       $DATA_FILE"
echo "  Output Dir:      $OUTPUT_DIR"
echo "  Max Mem/GPU:     $MAX_MEM_PER_GPU"
echo "  Shard:           $SHARD_IDX/$NUM_SHARDS"
echo "  Think Mode:      $THINK"
echo "  Thinking Mode:   $THINKING_MODE"
echo "  Seed:            $SEED"
echo "=============================================="
echo ""

# ==================== Validation ====================
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "ERROR: Model directory not found: $MODEL_PATH"
    exit 1
fi

if [[ ! -f "$DATA_FILE" ]]; then
    echo "ERROR: Data file not found: $DATA_FILE"
    exit 1
fi

# ==================== Environment Setup ====================
module load cuda/12.6.0
unset ROCR_VISIBLE_DEVICES
source /path/to/scratch/morph_env/bin/activate

export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"

echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo ""

mkdir -p "$OUTPUT_DIR"

# ==================== Build flags ====================
THINK_FLAG=""
if [[ "$THINK" == "false" ]]; then
    THINK_FLAG="--no_think"
fi

OUTPUT_FILE="${OUTPUT_DIR}/inference_results_bagel_shard${SHARD_IDX}.json"

echo "Launching shard ${SHARD_IDX}/${NUM_SHARDS} -> ${OUTPUT_FILE}"

python3 "${SCRIPT_DIR}/inference/run_inference_bagel_stare_perspective.py" \
    --model_path "$MODEL_PATH" \
    --data_file "$DATA_FILE" \
    --output_file "$OUTPUT_FILE" \
    --max_mem_per_gpu "$MAX_MEM_PER_GPU" \
    --shard "${SHARD_IDX}/${NUM_SHARDS}" \
    --random_seed "$SEED" \
    --thinking_mode "$THINKING_MODE" \
    $THINK_FLAG

echo ""
echo "=============================================="
echo "Done. Results saved to: $OUTPUT_DIR"
echo "=============================================="
