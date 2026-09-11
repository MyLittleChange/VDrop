#!/bin/bash
#SBATCH --job-name=eval_gen
#SBATCH --output=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/bagel/eval_gen_output_%a.txt
#SBATCH --error=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/bagel/eval_gen_error_%a.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --array=0

#
# Evaluate spatial view generation quality
#
# Computes pixel metrics (PSNR, SSIM, LPIPS), depth consistency (SI-RMSE),
# and LLM-based instruction following (Gemini).
#
# Usage:
#   # Submit all shards (pixel + depth metrics)
#   sbatch run_evaluate_generation.sh
#
#   # Run a single shard interactively
#   SLURM_ARRAY_TASK_ID=0 bash run_evaluate_generation.sh
#
#   # LLM evaluation only (no GPU needed, override SBATCH with srun)
#   METRICS="llm" SLURM_ARRAY_TASK_ID=0 bash run_evaluate_generation.sh
#
#   # Merge sharded results after all jobs finish
#   MODE=merge bash run_evaluate_generation.sh
#
#   # Quick test with 5 samples
#   NUM_SAMPLES=5 SLURM_ARRAY_TASK_ID=0 bash run_evaluate_generation.sh
#

set -euo pipefail

SCRIPT_DIR="/path/to/ThinkMorph-BAGEL-release"

# ==================== Configuration ====================
RESULTS_DIR="${RESULTS_DIR:-/path/to/scratch/VisualCoT/description_based_generation_orbit_relative}"

METRICS="${METRICS:-psnr ssim lpips depth}"

MODE="${MODE:-evaluate}"

NUM_SHARDS="${NUM_SHARDS:-1}"

# LLM config
GEMINI_MODEL="${GEMINI_MODEL:-gemini-3-flash-preview}"
MAX_WORKERS="${MAX_WORKERS:-16}"

# Depth config
DEPTH_MODEL="${DEPTH_MODEL:-depth-anything/Depth-Anything-V2-Small-hf}"

# LPIPS config
LPIPS_NET="${LPIPS_NET:-alex}"

DEVICE="${DEVICE:-cuda}"

NUM_SAMPLES="${NUM_SAMPLES:-}"

# ==================== Determine Shard ====================
SHARD_IDX="${SLURM_ARRAY_TASK_ID:-0}"
SHARD_SPEC="${SHARD_IDX}/${NUM_SHARDS}"

# ==================== Print Configuration ====================
echo "=============================================="
echo "Evaluate Spatial View Generation"
echo "=============================================="
echo "Configuration:"
echo "  Results Dir:      $RESULTS_DIR"
echo "  Mode:             $MODE"
echo "  Metrics:          $METRICS"
echo "  Shard:            $SHARD_SPEC"
echo "  Device:           $DEVICE"
echo "  Depth Model:      $DEPTH_MODEL"
echo "  LPIPS Net:        $LPIPS_NET"
echo "  Gemini Model:     $GEMINI_MODEL"
echo "  Max Workers:      $MAX_WORKERS"
echo "  Num Samples:      ${NUM_SAMPLES:-all}"
echo "=============================================="
echo ""

# ==================== Validation ====================
if [[ "$MODE" != "merge" && ! -d "$RESULTS_DIR" ]]; then
    echo "ERROR: Results directory not found: $RESULTS_DIR"
    exit 1
fi

# ==================== Environment Setup ====================
module load cuda/12.6.0
source /path/to/scratch/morph_env/bin/activate

if [[ "$DEVICE" == "cuda" ]]; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
    echo ""
fi

# ==================== Build Command ====================
NUM_SAMPLES_FLAG=""
if [[ -n "$NUM_SAMPLES" ]]; then
    NUM_SAMPLES_FLAG="--num_samples $NUM_SAMPLES"
fi

# ==================== Run ====================
if [[ "$MODE" == "merge" ]]; then
    echo "Merging sharded evaluation results..."
    python3 "${SCRIPT_DIR}/SpatialUnderstanding/eval/generation/evaluate_generation.py" \
        --mode merge \
        --results_dir "$RESULTS_DIR"
else
    echo "Starting evaluation for shard ${SHARD_IDX}/${NUM_SHARDS}..."
    echo ""

    python3 "${SCRIPT_DIR}/SpatialUnderstanding/eval/generation/evaluate_generation.py" \
        --results_dir "$RESULTS_DIR" \
        --metrics $METRICS \
        --shard "$SHARD_SPEC" \
        --depth_model "$DEPTH_MODEL" \
        --lpips_net "$LPIPS_NET" \
        --gemini_model "$GEMINI_MODEL" \
        --max_workers "$MAX_WORKERS" \
        --device "$DEVICE" \
        $NUM_SAMPLES_FLAG
fi

echo ""
echo "=============================================="
echo "Shard ${SHARD_IDX}/${NUM_SHARDS} Complete!"
echo "=============================================="
echo "Results saved to: $RESULTS_DIR"
