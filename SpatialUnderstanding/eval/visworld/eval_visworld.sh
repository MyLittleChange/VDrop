#!/bin/bash
#SBATCH --job-name=visworld_inference
#SBATCH --output=/path/to/ThinkMorph-BAGEL-release/visworld_inference_output.txt
#SBATCH --error=/path/to/ThinkMorph-BAGEL-release/visworld_inference_error.txt
#SBATCH --ntasks=1

#
# VisWorld-Eval Inference Script (Data-Parallel across 4 GPUs)
#
# Launches 4 inference processes, one per GPU, each handling a shard of the
# dataset. After all shards finish, merges results into final JSON files.
#
# Usage:
#   bash eval_visworld.sh
#   sbatch eval_visworld.sh
#

set -euo pipefail

SCRIPT_DIR="/path/to/ThinkMorph-BAGEL-release"

# ==================== Configuration ====================
# Model path
MODEL_PATH="${MODEL_PATH:-/path/to/scratch/models/ThinkMorph-7B}"

# Data source
DATA_SOURCE="${DATA_SOURCE:-parquet}"
PARQUET_PATH="${PARQUET_PATH:-/path/to/scratch/datasets/VisWorld-Eval}"

# Splits to evaluate (space-separated, or empty for all)
# Options: ballgame cube maze mmsi multihop paperfolding sokoban
SPLITS="${SPLITS:-}"

# Output directory for inference results
OUTPUT_DIR="${OUTPUT_DIR:-/path/to/scratch/VisualCoT/visworld_results}"

# Output directory for evaluated results
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-/path/to/scratch/VisualCoT/visworld_evaluated}"

# GPU memory per card (each process gets 1 GPU with 80GB)
MAX_MEM_PER_GPU="${MAX_MEM_PER_GPU:-80GiB}"

# Number of GPUs / shards for data parallelism
NUM_GPUS="${NUM_GPUS:-4}"

# Number of evaluation passes
NUM_PASSES="${NUM_PASSES:-1}"

# Random seed
SEED="${SEED:-42}"

# LLM Judge Configuration (Gemini)
JUDGE_API_KEY="${JUDGE_API_KEY:-${VisualCoT_GEMINI:-}}"
JUDGE_MODEL_NAME="${JUDGE_MODEL_NAME:-gemini-3-flash-preview}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"

# ==================== Validation ====================
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "ERROR: Model directory not found: $MODEL_PATH"
    exit 1
fi

# ==================== Display Configuration ====================
echo "=============================================="
echo "VisWorld-Eval Inference with ThinkMorph"
echo "  Data-Parallel: ${NUM_GPUS} GPUs"
echo "=============================================="
echo "Configuration:"
echo "  Model Path:      $MODEL_PATH"
echo "  Data Source:     $DATA_SOURCE"
echo "  Parquet Path:    $PARQUET_PATH"
echo "  Splits:          ${SPLITS:-all}"
echo "  Output Dir:      $OUTPUT_DIR"
echo "  Max Mem/GPU:     $MAX_MEM_PER_GPU"
echo "  Num GPUs:        $NUM_GPUS"
echo "  Num Passes:      $NUM_PASSES"
echo "  Seed:            $SEED"
echo "=============================================="
echo ""

module load cuda/12.6.0
unset ROCR_VISIBLE_DEVICES
source /path/to/scratch/morph_env/bin/activate

# Show GPU info
echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo ""

# ==================== Build splits argument ====================
SPLITS_ARG=""
if [[ -n "$SPLITS" ]]; then
    SPLITS_ARG="--splits $SPLITS"
fi

# ==================== Run Inference (Data-Parallel) ====================
mkdir -p "$OUTPUT_DIR"
echo "Launching ${NUM_GPUS} parallel inference processes..."
echo ""

PIDS=()
for SHARD_ID in $(seq 0 $((NUM_GPUS - 1))); do
    echo "  [Shard ${SHARD_ID}] Starting on GPU ${SHARD_ID}..."
    CUDA_VISIBLE_DEVICES=${SHARD_ID} python3 "${SCRIPT_DIR}/SpatialUnderstanding/eval/visworld/eval_visworld.py" \
        --mode inference \
        --model_path "$MODEL_PATH" \
        --data_source "$DATA_SOURCE" \
        --parquet_path "$PARQUET_PATH" \
        $SPLITS_ARG \
        --output_dir "$OUTPUT_DIR" \
        --max_mem_per_gpu "$MAX_MEM_PER_GPU" \
        --num_passes "$NUM_PASSES" \
        --seed "$SEED" \
        --shard_id "$SHARD_ID" \
        --num_shards "$NUM_GPUS" \
        > "${OUTPUT_DIR}/shard_${SHARD_ID}.log" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "Waiting for all ${NUM_GPUS} shards to complete..."

# Wait for all background processes; exit on first failure
FAIL=0
for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
        echo "ERROR: Shard $i (PID ${PIDS[$i]}) failed. Check ${OUTPUT_DIR}/shard_${i}.log"
        FAIL=1
    fi
done

if [[ "$FAIL" -eq 1 ]]; then
    echo "One or more shards failed. Aborting."
    exit 1
fi

echo "All shards completed successfully."
echo ""

# ==================== Merge Shard Results ====================
echo "Merging shard results..."

for PASS in $(seq 1 "$NUM_PASSES"); do
    MERGED_FILE="${OUTPUT_DIR}/model_results_run_${PASS}.json"
    # Use Python to merge JSON arrays from all shard files
    python3 -c "
import json, sys, os
merged = []
for shard_id in range(${NUM_GPUS}):
    path = os.path.join('${OUTPUT_DIR}', f'model_results_run_${PASS}_shard_{shard_id}.json')
    with open(path) as f:
        merged.extend(json.load(f))
with open('${MERGED_FILE}', 'w') as f:
    json.dump(merged, f, indent=2)
print(f'  Pass ${PASS}: merged {len(merged)} results -> ${MERGED_FILE}')
"
    # Clean up shard files
    for SHARD_ID in $(seq 0 $((NUM_GPUS - 1))); do
        rm -f "${OUTPUT_DIR}/model_results_run_${PASS}_shard_${SHARD_ID}.json"
    done
done

echo ""
echo "=============================================="
echo "VisWorld-Eval Inference Complete!"
echo "=============================================="
echo "Results saved to: $OUTPUT_DIR"

# ==================== Run Evaluation with LLM Judge ====================
if [[ -z "$JUDGE_API_KEY" ]]; then
    echo ""
    echo "WARNING: JUDGE_API_KEY or VisualCoT_GEMINI not set, skipping LLM judge evaluation"
    echo "To run evaluation later:"
    echo "  VisualCoT_GEMINI=your-key python eval_visworld.py --mode evaluate --results_dir $OUTPUT_DIR"
else
    echo ""
    echo "=============================================="
    echo "Running LLM Judge Evaluation (Gemini)"
    echo "=============================================="
    echo "Configuration:"
    echo "  Results Dir:  $OUTPUT_DIR"
    echo "  Eval Output:  $EVAL_OUTPUT_DIR"
    echo "  Judge Model:  $JUDGE_MODEL_NAME"
    echo "  Processes:    $NUM_PROCESSES"
    echo "=============================================="
    echo ""

    python3 "${SCRIPT_DIR}/SpatialUnderstanding/eval/visworld/eval_visworld.py" \
        --mode evaluate \
        --results_dir "$OUTPUT_DIR" \
        --output_dir "$EVAL_OUTPUT_DIR" \
        --api_key "$JUDGE_API_KEY" \
        --judge_model "$JUDGE_MODEL_NAME" \
        --num_processes "$NUM_PROCESSES"

    echo ""
    echo "=============================================="
    echo "VisWorld-Eval Evaluation Complete!"
    echo "=============================================="
    echo "Evaluated results saved to: $EVAL_OUTPUT_DIR"
fi
