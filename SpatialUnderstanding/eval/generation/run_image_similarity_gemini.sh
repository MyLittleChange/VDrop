#!/bin/bash
# Gemini-3-Flash autorater for generated-vs-GT pairs. API-only, long-cpu.
#
# Usage:
#   sbatch run_image_similarity_gemini.sh         # full run on all 3 pairings
#   PAIRING=pano NUM_SAMPLES=5 sbatch ...         # smoke
#SBATCH --job-name=img_sim_gemini
#SBATCH --partition=long-cpu
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH --time=6:00:00
#SBATCH --output=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/eval/generation/slurm_logs/img_sim_gemini_%j.out
#SBATCH --error=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/eval/generation/slurm_logs/img_sim_gemini_%j.err

set -euo pipefail
mkdir -p /path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/eval/generation/slurm_logs

if [ -z "${VisualCoT_GEMINI:-}" ]; then
    # shellcheck disable=SC1090
    source ~/.bashrc
fi
export VisualCoT_GEMINI

source /path/to/scratch/morph_env/bin/activate

SCRIPT_DIR="/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/eval/generation"
OUTPUT_DIR="${OUTPUT_DIR:-/path/to/scratch/VisualCoT/eval_results/image_similarity/gemini_judge}"
PAIRING="${PAIRING:-all}"
MODEL_NAME="${MODEL_NAME:-gemini-3-flash-preview}"
WORKERS="${WORKERS:-16}"
NUM_SAMPLES_FLAG=""
if [ -n "${NUM_SAMPLES:-}" ]; then
    NUM_SAMPLES_FLAG="--num_samples ${NUM_SAMPLES}"
fi

echo "=== pairing=$PAIRING  model=$MODEL_NAME  workers=$WORKERS ==="

python "$SCRIPT_DIR/gemini_image_pair_judge.py" \
    --pairing "$PAIRING" \
    --subtasks anchor counting relative_distance relative_direction \
    --output_dir "$OUTPUT_DIR" \
    --model_name "$MODEL_NAME" \
    --workers "$WORKERS" \
    $NUM_SAMPLES_FLAG

echo "Done. Aggregate:"
echo "  python $SCRIPT_DIR/aggregate_image_similarity.py"
