#!/bin/bash
#SBATCH --job-name=tm_spatial
#SBATCH --output=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/eval/bagel/thinkmorph/slurm_logs/tm_spatial_%j.out
#SBATCH --error=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/eval/bagel/thinkmorph/slurm_logs/tm_spatial_%j.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100l:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=3:00:00
#SBATCH --partition=short-unkillable

# ThinkMorph-7B on COSMIC spatial (4 subtypes) — think=True, interleaved_thinking
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/path/to/scratch/models/ThinkMorph-7B}"
MODEL_TAG="thinkmorph_7b"
OUTPUT_ROOT="${OUTPUT_ROOT:-/path/to/scratch/VisualCoT/eval_results/${MODEL_TAG}}"

export MODEL_PATH
export THINK="true"
export THINKING_MODE="interleaved_thinking"
export NUM_SHARDS="${NUM_SHARDS:-4}"

GENERIC="/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/eval/spatial/run_bagel_spatial_generic.sh"

FAILED=()
for TASK in anchor counting distance direction; do
    export TASK
    export OUTPUT_DIR="${OUTPUT_ROOT}/${TASK}"
    mkdir -p "$OUTPUT_DIR"
    echo "===== [$TASK] OUTPUT_DIR=$OUTPUT_DIR ====="
    bash "$GENERIC" || FAILED+=("$TASK")
done

if [ ${#FAILED[@]} -gt 0 ]; then echo "FAILED: ${FAILED[*]}"; exit 1; fi
echo "All 4 spatial subtypes done. Merge with merge_bagel_spatial_results.py per ${OUTPUT_ROOT}/<task>/"
