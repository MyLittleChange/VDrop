#!/bin/bash
#
# Merge sharded MMSI-Bench results and optionally run LLM judge evaluation.
#
# Usage:
#   bash merge_and_eval_mmsi.sh
#

set -euo pipefail

SCRIPT_DIR="/path/to/ThinkMorph-BAGEL-release"

source /path/to/scratch/morph_env/bin/activate

# ==================== Configuration ====================
# Choose ONE of the following result directories (comment/uncomment as needed):

# BAGEL baseline (24 shards)
# RESULTS_DIR="/path/to/scratch/VisualCoT/BAGEL_mmsi_eval"
# NUM_SHARDS=24

# mat_pm_vo_lora (training_data_mix_balance_matterport_point_matching_visual_only_lora)
RESULTS_DIR="/path/to/scratch/VisualCoT/BAGEL_format_training_data_mix_balance_matterport_point_matching_visual_only_lora_mmsi_eval"
NUM_SHARDS=2

# Output directory (default: same as results_dir)
OUTPUT_DIR="${RESULTS_DIR}"

# Run number
RUN_NUM=1

# ==================== Mode ====================
# Set MODE to one of: "merge", "summary", "judge"
#   merge   - merge shards + print basic metrics (no LLM judge)
#   summary - print metrics from an already-evaluated JSON file
#   judge   - merge shards + run LLM judge + print accuracy
MODE="judge"

# ==================== Summary mode settings ====================
# Path to an already-evaluated JSON file (only used when MODE=summary)
SUMMARY_FILE="${RESULTS_DIR}/evaluated_model_results_run_${RUN_NUM}.json"

# ==================== LLM Judge settings (only used when MODE=judge) ====================
JUDGE_MODEL="gemini-3-flash-preview"
NUM_PROCESSES=16

# ==================== Run ====================
case "${MODE}" in
    merge)
        python3 "/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/eval/mmsi/merge_and_eval_mmsi.py" \
            --results_dir "${RESULTS_DIR}" \
            --output_dir "${OUTPUT_DIR}" \
            --run_num "${RUN_NUM}" \
            --num_shards "${NUM_SHARDS}"
        ;;
    summary)
        python3 "/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/eval/mmsi/merge_and_eval_mmsi.py" \
            --summary "${SUMMARY_FILE}"
        ;;
    judge)
        python3 "/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/eval/mmsi/merge_and_eval_mmsi.py" \
            --results_dir "${RESULTS_DIR}" \
            --output_dir "${OUTPUT_DIR}" \
            --run_num "${RUN_NUM}" \
            --num_shards "${NUM_SHARDS}" \
            --run_judge \
            --judge_model "${JUDGE_MODEL}" \
            --num_processes "${NUM_PROCESSES}"
        ;;
    *)
        echo "ERROR: Unknown MODE '${MODE}'. Use 'merge', 'summary', or 'judge'."
        exit 1
        ;;
esac
