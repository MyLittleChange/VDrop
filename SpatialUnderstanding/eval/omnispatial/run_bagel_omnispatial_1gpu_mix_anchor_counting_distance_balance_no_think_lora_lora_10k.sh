#!/bin/bash
#SBATCH --job-name=bagel_omni_acd_nt_lora10k
#SBATCH --output=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/eval/omnispatial/slurm_logs/bagel_omnispatial_mix_anchor_counting_distance_balance_no_think_lora_lora_10k_output.txt
#SBATCH --error=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/eval/omnispatial/slurm_logs/bagel_omnispatial_mix_anchor_counting_distance_balance_no_think_lora_lora_10k_error.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --partition=unkillable

#
# BAGEL OmniSpatial Benchmark Inference — 1 GPU on unkillable
# Model: BAGEL_format_mix_anchor_counting_distance_balance_no_think_lora_lora_10k
# Think: false, Thinking mode: no_thinking
# Subset: Complex_Logic + Perspective_Taking task_types (813 samples)
#
set -euo pipefail

SCRIPT_DIR="/path/to/ThinkMorph-BAGEL-release"

# ==================== Configuration ====================
MODEL_PATH="${MODEL_PATH:-/path/to/scratch/VisualCoT/BAGEL_checkpoints/BAGEL_format_mix_anchor_counting_distance_balance_no_think_lora_lora_10k}"

DATASET_PATH="${DATASET_PATH:-/path/to/scratch/datasets/OmniSpatial/OmniSpatial-test}"

OUTPUT_DIR="${OUTPUT_DIR:-/path/to/scratch/VisualCoT/omnispatial/BAGEL_format_mix_anchor_counting_distance_balance_no_think_lora_lora_10k_complex_logic_perspective_taking}"

MAX_MEM_PER_GPU="${MAX_MEM_PER_GPU:-70GiB}"

NUM_SHARDS="${NUM_SHARDS:-1}"

SEED="${SEED:-42}"

THINK="${THINK:-false}"

# Thinking mode: visual_only_thinking, interleaved_thinking, text_only_thinking, or no_thinking
THINKING_MODE="${THINKING_MODE:-no_thinking}"

# Filter dataset by task_type / sub_task_type (comma-separated; empty = no filter)
TASK_TYPE="${TASK_TYPE:-Complex_Logic,Perspective_Taking}"
SUB_TASK_TYPE="${SUB_TASK_TYPE:-}"

SHARD_IDX="${SHARD_IDX:-0}"

# ==================== Print Configuration ====================
echo "=============================================="
echo "BAGEL OmniSpatial Benchmark Inference (1-GPU unkillable, no_thinking)"
echo "=============================================="
echo "Configuration:"
echo "  Model Path:      $MODEL_PATH"
echo "  Dataset Path:    $DATASET_PATH"
echo "  Output Dir:      $OUTPUT_DIR"
echo "  Max Mem/GPU:     $MAX_MEM_PER_GPU"
echo "  Shard:           $SHARD_IDX/$NUM_SHARDS"
echo "  Think Mode:      $THINK"
echo "  Thinking Mode:   $THINKING_MODE"
echo "  Task Type:       ${TASK_TYPE:-<all>}"
echo "  Sub Task Type:   ${SUB_TASK_TYPE:-<all>}"
echo "  Seed:            $SEED"
echo "=============================================="
echo ""

# ==================== Validation ====================
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "ERROR: Model directory not found: $MODEL_PATH"
    exit 1
fi

if [[ ! -f "${DATASET_PATH}/data.json" ]]; then
    echo "ERROR: data.json not found at: ${DATASET_PATH}/data.json"
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

FILTER_FLAGS=""
if [[ -n "$TASK_TYPE" ]]; then
    FILTER_FLAGS="$FILTER_FLAGS --task_type $TASK_TYPE"
fi
if [[ -n "$SUB_TASK_TYPE" ]]; then
    FILTER_FLAGS="$FILTER_FLAGS --sub_task_type $SUB_TASK_TYPE"
fi

OUTPUT_FILE="${OUTPUT_DIR}/inference_results_bagel_shard${SHARD_IDX}.json"

echo "Launching shard ${SHARD_IDX}/${NUM_SHARDS} -> ${OUTPUT_FILE}"

python3 "${SCRIPT_DIR}/inference/run_inference_bagel_omnispatial.py" \
    --model_path "$MODEL_PATH" \
    --dataset_path "$DATASET_PATH" \
    --output_file "$OUTPUT_FILE" \
    --max_mem_per_gpu "$MAX_MEM_PER_GPU" \
    --shard "${SHARD_IDX}/${NUM_SHARDS}" \
    --random_seed "$SEED" \
    --thinking_mode "$THINKING_MODE" \
    $THINK_FLAG \
    $FILTER_FLAGS

echo ""
echo "=============================================="
echo "Done. Results saved to: $OUTPUT_DIR"
echo "=============================================="
