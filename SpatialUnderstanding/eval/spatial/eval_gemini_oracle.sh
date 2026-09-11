#!/bin/bash
# Run Gemini-3-Pro (thinking) oracle ablation on the COSMIC 800-sample subset.
# Sequential over 4 conditions (no parallel — single API account).
#
# Usage:
#   bash eval_gemini_oracle.sh                  # all 4 conditions
#   CONDITION=T_pano N_SAMPLES=5 bash ...       # smoke
#
# Env:
#   VisualCoT_GEMINI must be set (load from ~/.bashrc).

set -euo pipefail

# Source ~/.bashrc to get VisualCoT_GEMINI
if [ -z "${VisualCoT_GEMINI:-}" ]; then
    # shellcheck disable=SC1090
    source ~/.bashrc
fi
if [ -z "${VisualCoT_GEMINI:-}" ]; then
    echo "ERROR: VisualCoT_GEMINI not set"; exit 1
fi
export VisualCoT_GEMINI

SCRIPT_DIR="/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/eval/spatial"
OUTPUT_DIR="${OUTPUT_DIR:-/path/to/scratch/VisualCoT/eval_results/two_reader_oracle_gemini}"

MODEL_NAME="${MODEL_NAME:-gemini-3-pro-preview}"
THINKING_LEVEL="${THINKING_LEVEL:-high}"
WORKERS="${WORKERS:-16}"
TEMPERATURE="${TEMPERATURE:-0.7}"
MAX_TOKENS="${MAX_TOKENS:-16384}"

# Activate env with google.genai
source /path/to/scratch/morph_env/bin/activate

CONDITIONS_TO_RUN="${CONDITION:-none T_td_blender T_cor T_pano}"

N_SAMPLES_FLAG=""
if [ -n "${N_SAMPLES:-}" ]; then
    N_SAMPLES_FLAG="--max_samples ${N_SAMPLES}"
fi

for COND in $CONDITIONS_TO_RUN; do
    echo "=== Condition: $COND ==="
    python "$SCRIPT_DIR/run_inference_gemini_oracle.py" \
        --model_name "$MODEL_NAME" \
        --thinking_level "$THINKING_LEVEL" \
        --condition "$COND" \
        --subtasks anchor counting relative_distance relative_direction \
        --output_dir "$OUTPUT_DIR" \
        --workers "$WORKERS" \
        --temperature "$TEMPERATURE" \
        --max_tokens "$MAX_TOKENS" \
        $N_SAMPLES_FLAG
done

echo ""
echo "Done. Aggregate:"
echo "  python $SCRIPT_DIR/aggregate_oracle_ablation.py --output_dir $OUTPUT_DIR"
