#!/bin/bash
# =============================================================================
# Gemini-3-Pro (thinking) oracle ablation — Slurm long-cpu wrapper.
# Use this for runs that must survive the user logging out.
#
# Usage:
#   CONDITION="T_pano_gen T_cor_gen T_td_gen" sbatch eval_gemini_oracle_longcpu.sh
#   CONDITION=T_noise WORKERS=8 sbatch eval_gemini_oracle_longcpu.sh
# =============================================================================
#SBATCH --job-name=gemini_oracle
#SBATCH --partition=long-cpu
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/eval/spatial/slurm_logs/gemini_oracle_%j.out
#SBATCH --error=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/eval/spatial/slurm_logs/gemini_oracle_%j.err

set -euo pipefail
mkdir -p /path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/eval/spatial/slurm_logs

# Pick up the API key (from ~/.bashrc by default)
if [ -z "${VisualCoT_GEMINI:-}" ]; then
    # shellcheck disable=SC1090
    source ~/.bashrc
fi
if [ -z "${VisualCoT_GEMINI:-}" ]; then
    echo "ERROR: VisualCoT_GEMINI not set"; exit 1
fi
export VisualCoT_GEMINI

source /path/to/scratch/morph_env/bin/activate

SCRIPT_DIR="/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/eval/spatial"
OUTPUT_DIR="${OUTPUT_DIR:-/path/to/scratch/VisualCoT/eval_results/two_reader_oracle_gemini}"

MODEL_NAME="${MODEL_NAME:-gemini-3-pro-preview}"
THINKING_LEVEL="${THINKING_LEVEL:-high}"
WORKERS="${WORKERS:-16}"
TEMPERATURE="${TEMPERATURE:-0.7}"
MAX_TOKENS="${MAX_TOKENS:-16384}"

CONDITIONS_TO_RUN="${CONDITION:-none T_td_blender T_cor T_pano T_noise T_pano_gen T_cor_gen T_td_gen}"

N_SAMPLES_FLAG=""
if [ -n "${N_SAMPLES:-}" ]; then
    N_SAMPLES_FLAG="--max_samples ${N_SAMPLES}"
fi

echo "=== Conditions: $CONDITIONS_TO_RUN ==="
echo "=== Workers: $WORKERS ==="
echo "=== Output: $OUTPUT_DIR ==="

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
echo "  python $SCRIPT_DIR/aggregate_oracle_ablation.py --output_dir $OUTPUT_DIR --paired"
