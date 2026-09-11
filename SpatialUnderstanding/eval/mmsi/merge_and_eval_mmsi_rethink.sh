#!/bin/bash
#
# Evaluate MMSI-Bench rethink results (rethink_results_run_*.json).
# Uses --rethink mode of merge_and_eval_mmsi.py which reads RethinkLLMJudgeResult.
#
# Usage:
#   bash merge_and_eval_mmsi_rethink.sh
#

set -euo pipefail

source /path/to/scratch/morph_env/bin/activate

# ==================== Configuration ====================
# Directory containing rethink_results_run_1.json
RESULTS_DIR="/path/to/scratch/VisualCoT/rethink_panorama_qa_visual_only_mmsi"

# Run number
RUN_NUM=1

RETHINK_FILE="${RESULTS_DIR}/rethink_results_run_${RUN_NUM}.json"

# ==================== Run ====================
python3 "/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/eval/mmsi/merge_and_eval_mmsi.py" \
    --rethink "${RETHINK_FILE}"
