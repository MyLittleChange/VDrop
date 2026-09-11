#!/bin/bash
#SBATCH --job-name=mmsi_generic
#SBATCH --output=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/slurm_logs/mmsi/%x_output.txt
#SBATCH --error=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/slurm_logs/mmsi/%x_error.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100l:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --partition=main

# Generic MMSI-Bench Inference — 2 GPUs, single job (no array)
# Required env vars: MODEL_PATH, OUTPUT_DIR
# Optional env vars: THINK (false), THINKING_MODE (no_thinking), NUM_SHARDS (2)
set -euo pipefail

SCRIPT_DIR="/path/to/ThinkMorph-BAGEL-release"

[[ -z "${MODEL_PATH:-}" ]] && { echo "ERROR: MODEL_PATH is required"; exit 1; }
[[ -z "${OUTPUT_DIR:-}"  ]] && { echo "ERROR: OUTPUT_DIR is required";  exit 1; }
[[ ! -d "$MODEL_PATH"    ]] && { echo "ERROR: Model not found: $MODEL_PATH"; exit 1; }

DATA_SOURCE="${DATA_SOURCE:-parquet}"
PARQUET_PATH="${PARQUET_PATH:-/path/to/scratch/datasets/MMSI-Bench}"
MAX_MEM_PER_GPU="${MAX_MEM_PER_GPU:-80GiB}"
NUM_PASSES="${NUM_PASSES:-1}"
NUM_SHARDS="${NUM_SHARDS:-2}"
SEED="${SEED:-42}"
THINK="${THINK:-false}"
THINKING_MODE="${THINKING_MODE:-no_thinking}"
VIT_MIN_SIZE="${VIT_MIN_SIZE:-512}"
MAX_ROUNDS="${MAX_ROUNDS:-3}"
IMAGE_SHAPES="${IMAGE_SHAPES:-720 720}"
GPUS_PER_JOB=2

echo "MMSI | Model: ${MODEL_PATH##*/} | Thinking: ${THINKING_MODE} | Shards: 0-$((NUM_SHARDS-1))/${NUM_SHARDS}"

module load cuda/12.6.0
unset ROCR_VISIBLE_DEVICES
source /path/to/scratch/morph_env/bin/activate
mkdir -p "$OUTPUT_DIR"

THINK_FLAG=""; [[ "$THINK" == "false" ]] && THINK_FLAG="--no_think"

PIDS=()
for i in $(seq 0 $((GPUS_PER_JOB - 1))); do
    if [[ $i -lt $NUM_SHARDS ]]; then
        CUDA_VISIBLE_DEVICES="$i" python3 "${SCRIPT_DIR}/inference/eval_mmsi_bench.py" \
            --mode inference \
            --model_path "$MODEL_PATH" \
            --data_source "$DATA_SOURCE" \
            --parquet_path "$PARQUET_PATH" \
            --output_dir "$OUTPUT_DIR" \
            --max_mem_per_gpu "$MAX_MEM_PER_GPU" \
            --num_passes "$NUM_PASSES" \
            --seed "$SEED" \
            --shard "${i}/${NUM_SHARDS}" \
            --thinking_mode "$THINKING_MODE" \
            --vit_min_size "$VIT_MIN_SIZE" \
            --max_rounds "$MAX_ROUNDS" \
            ${IMAGE_SHAPES:+--image_shapes $IMAGE_SHAPES} \
            --resume \
            $THINK_FLAG &
        PIDS+=($!)
    fi
done

FAIL=0
for pid in "${PIDS[@]}"; do wait "$pid" || FAIL=1; done
[[ "$FAIL" -eq 0 ]] && echo "All shards done." || echo "Some shards failed."
echo "Results: $OUTPUT_DIR"
