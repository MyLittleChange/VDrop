#!/bin/bash
# DINO + LPIPS + PSNR + SSIM for generated-vs-GT pairs (3 pairings × 4 subtasks).
#
# Usage:
#   sbatch run_image_similarity_pixel.sh                # full run on all 3 pairings
#   PAIRING=pano NUM_SAMPLES=5 sbatch ...               # smoke
#SBATCH --job-name=img_sim_pixel
#SBATCH --partition=unkillable
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --output=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/eval/generation/slurm_logs/img_sim_pixel_%j.out
#SBATCH --error=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/eval/generation/slurm_logs/img_sim_pixel_%j.err

set -euo pipefail
mkdir -p /path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/eval/generation/slurm_logs

module load cuda/12.6.0
unset ROCR_VISIBLE_DEVICES
source /path/to/scratch/morph_env/bin/activate

SCRIPT_DIR="/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/eval/generation"
OUTPUT_DIR="${OUTPUT_DIR:-/path/to/scratch/VisualCoT/eval_results/image_similarity/pixel}"
PAIRING="${PAIRING:-all}"
NUM_SAMPLES_FLAG=""
if [ -n "${NUM_SAMPLES:-}" ]; then
    NUM_SAMPLES_FLAG="--num_samples ${NUM_SAMPLES}"
fi

echo "=== pairing=$PAIRING  output=$OUTPUT_DIR ==="

python "$SCRIPT_DIR/compute_image_similarity.py" \
    --pairing "$PAIRING" \
    --subtasks anchor counting relative_distance relative_direction \
    --output_dir "$OUTPUT_DIR" \
    --device cuda \
    --lpips_net alex \
    --batch_size 32 \
    $NUM_SAMPLES_FLAG

echo "Done. Aggregate:"
echo "  python $SCRIPT_DIR/aggregate_image_similarity.py"
