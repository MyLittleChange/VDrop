#!/bin/bash
#
# Merge sharded MindCube results and compute final accuracy.
#
# Usage:
#   bash merge_mindcube_results.sh
#   RESULTS_DIR=/path/to/dir NUM_SHARDS=4 bash merge_mindcube_results.sh
#

set -euo pipefail

SCRIPT_DIR="/path/to/ThinkMorph-BAGEL-release"

source /path/to/scratch/morph_env/bin/activate

# ==================== Configuration ====================
# Set RESULTS_DIR to the output directory of the eval job

# BAGEL-7B-MoT, text_only_thinking
# RESULTS_DIR="/path/to/scratch/mindcube/BAGEL"

# panorama_no_thinking
# RESULTS_DIR="/path/to/scratch/mindcube/BAGEL_format_panorama_qa_no_thinking"
RESULTS_DIR="/path/to/scratch/mindcube/training_data_mix_balance_matterport_point_matching_no_think_lora"

# panorama_visual_only
# RESULTS_DIR="/path/to/scratch/mindcube/BAGEL_format_panorama_qa_visual_only_720_1024"

# topdown_visual_only
# RESULTS_DIR="${RESULTS_DIR:-/path/to/scratch/mindcube/BAGEL_format_topdown_qa_visual_only}"

NUM_SHARDS="${NUM_SHARDS:-4}"

OUTPUT_FILE="${OUTPUT_FILE:-${RESULTS_DIR}/merged_results.json}"

# ==================== Run ====================
echo "Merging MindCube results..."
echo "  Results Dir: $RESULTS_DIR"
echo "  Num Shards:  $NUM_SHARDS"
echo "  Output File: $OUTPUT_FILE"
echo ""

python3 "${SCRIPT_DIR}/SpatialUnderstanding/eval/mindcube/merge_mindcube_results.py" \
    --results_dir "$RESULTS_DIR" \
    --num_shards "$NUM_SHARDS" \
    --output_file "$OUTPUT_FILE"
