#!/bin/bash
#
# Merge sharded VSI-Bench results and optionally run LLM answer re-extraction.
#
# Usage:
#   bash merge_and_eval_vsibench.sh
#

set -euo pipefail

SCRIPT_DIR="/path/to/ThinkMorph-BAGEL-release"

source /path/to/scratch/morph_env/bin/activate

# ==================== Configuration ====================
# Set RESULTS_DIR to the directory containing inference_results_shard*.json files

# BAGEL text_only_thinking
RESULTS_DIR="/path/to/scratch/vsibench/BAGEL_format_training_data_mix_all_rotation_visual_only"

# Output directory (default: same as results_dir)
OUTPUT_DIR="${RESULTS_DIR}"

# ==================== Mode ====================
# Set MODE to one of: "merge", "summary", "judge"
#   merge   - merge shards + print basic metrics (no LLM re-extraction)
#   summary - print metrics from an already-combined JSON file
#   judge   - merge shards + run LLM answer re-extraction + print metrics
MODE="judge"

# ==================== Summary mode settings ====================
# Path to an already-combined JSON file (only used when MODE=summary)
SUMMARY_FILE="${RESULTS_DIR}/inference_results_combined.json"

# ==================== LLM Judge settings (only used when MODE=judge) ====================
JUDGE_MODEL="gemini-3-flash-preview"
NUM_PROCESSES=16

# ==================== Run ====================
case "${MODE}" in
    merge)
        python3 "${SCRIPT_DIR}/SpatialUnderstanding/eval/vsibench/merge_and_eval_vsibench.py" \
            --results_dir "${RESULTS_DIR}" \
            --output_dir "${OUTPUT_DIR}"
        ;;
    summary)
        python3 "${SCRIPT_DIR}/SpatialUnderstanding/eval/vsibench/merge_and_eval_vsibench.py" \
            --summary "${SUMMARY_FILE}"
        ;;
    judge)
        python3 "${SCRIPT_DIR}/SpatialUnderstanding/eval/vsibench/merge_and_eval_vsibench.py" \
            --results_dir "${RESULTS_DIR}" \
            --output_dir "${OUTPUT_DIR}" \
            --run_judge \
            --judge_model "${JUDGE_MODEL}" \
            --num_processes "${NUM_PROCESSES}"
        ;;
    *)
        echo "ERROR: Unknown MODE '${MODE}'. Use 'merge', 'summary', or 'judge'."
        exit 1
        ;;
esac
