#!/bin/bash
# Run bootstrap confidence interval evaluation on all merged inference result files.

BASE=/path/to/scratch/VisualCoT
SCRIPT=$(dirname "$0")/bootstrap_eval.py

DIRS=(
    # visual-only (panorama)
    "$BASE/panorama_qa_visual_only_mcqs_anchor_normalized"
    "$BASE/panorama_qa_visual_only_mcqs_counting_normalized"
    "$BASE/panorama_qa_visual_only_mcqs_relative_direction_normalized"
    "$BASE/panorama_qa_visual_only_mcqs_relative_distance_normalized"
    # visual-only (topdown)
    "$BASE/topdown_qa_visual_only_mcqs_anchor_normalized"
    "$BASE/topdown_qa_visual_only_mcqs_counting_normalized"
    "$BASE/topdown_qa_visual_only_mcqs_relative_distance_normalized"
    "$BASE/topdown_qa_visual_only/approved_mcqs_relative_direction_normalized"
    # no-thinking (panorama)
    "$BASE/panorama_qa_no_thinking/approved_mcqs_relative_direction_normalized"
    "$BASE/panorama_qa_no_thinking/panorama_qa_no_thinking_mcqs_anchor_normalized"
    "$BASE/panorama_qa_no_thinking/panorama_qa_no_thinking_mcqs_counting_normalized"
    "$BASE/panorama_qa_no_thinking/panorama_qa_no_thinking_mcqs_relative_distance_normalized"
)

for DIR in "${DIRS[@]}"; do
    JSON="$DIR/inference_results_bagel_merged.json"
    echo "=========================================="
    echo "$(basename "$DIR")"
    echo "=========================================="
    python "$SCRIPT" "$JSON"
    echo ""
done
