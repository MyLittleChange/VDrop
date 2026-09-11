#!/bin/bash
#SBATCH --job-name=render_perturbed_v2
#SBATCH --partition=long-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --output=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/data_creation/training_data/slurm_logs/render_perturbed_v2-%A_%a.txt
#SBATCH --error=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/data_creation/training_data/slurm_logs/render_perturbed_v2_error-%A_%a.txt

# Submit:
#   sbatch --array=0-7 SpatialUnderstanding/data_creation/training_data/run_render_perturbed_topdowns_v2.sh
#
# Each array task takes one CRC32-determined shard of the ~2,140 raw scenes
# discovered under v4/v5/Ankur roots and renders correct + 3-tier-curriculum
# perturbed top-downs into:
#   /path/to/scratch/infinigen/map_questions_perturbed_v2/<scene>/

set -euo pipefail

source /path/to/scratch/morph_env/bin/activate

SCRIPT_DIR="/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/data_creation/training_data"
LOG_DIR="${SCRIPT_DIR}/slurm_logs"
mkdir -p "$LOG_DIR"

NUM_SHARDS="${NUM_SHARDS:-8}"
SHARD_IDX="${SLURM_ARRAY_TASK_ID:-0}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/path/to/scratch/infinigen/map_questions_perturbed_v2}"
WORKERS="${WORKERS:-8}"
MAX_PER_TYPE="${MAX_PER_TYPE:-5}"

echo "[$(date '+%F %T')] shard ${SHARD_IDX}/${NUM_SHARDS}; workers=$WORKERS; output=$OUTPUT_ROOT"

cd /path/to/ThinkMorph-BAGEL-release

python "$SCRIPT_DIR/render_perturbed_topdowns.py" \
    --scan_roots \
    --output_root "$OUTPUT_ROOT" \
    --workers "$WORKERS" \
    --max_per_type "$MAX_PER_TYPE" \
    --force \
    --num_shards "$NUM_SHARDS" \
    --shard_idx "$SHARD_IDX"

echo "[$(date '+%F %T')] shard ${SHARD_IDX} done."
