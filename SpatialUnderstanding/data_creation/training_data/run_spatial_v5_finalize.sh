#!/bin/bash
#SBATCH --job-name=V5Finalize
#SBATCH -c 32
#SBATCH --mem=64Gb
#SBATCH --time=2:00:00
#SBATCH --output=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/data_creation/training_data/slurm_logs/v5_finalize_output-%j.txt
#SBATCH --error=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/data_creation/training_data/slurm_logs/v5_finalize_error-%j.txt

# ==============================================================================
# V5 finalize: filter_rendered → create_mix_all
# Runs after panorama render array completes.
# ==============================================================================

set -e

source /path/to/scratch/morph_env/bin/activate

PROJECT_DIR="/path/to/ThinkMorph-BAGEL-release"
TRAINING_DATA_DIR="$PROJECT_DIR/SpatialUnderstanding/data_creation/training_data"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

cd "$PROJECT_DIR"

###############################################################################
# Step 1: Filter rendered panoramas
###############################################################################
log "Step 1: Filtering rendered panoramas..."
python "$TRAINING_DATA_DIR/prepare_panorama_v5_split.py" \
    --filter_rendered \
    --panorama_dir /path/to/scratch/infinigen/rendered_panorama_v5

###############################################################################
# Step 2: Create mixed training data (existing + V4 + V5)
###############################################################################
log "Step 2: Creating mixed SFT training data (visual_only mode)..."
python "$TRAINING_DATA_DIR/create_mix_all_sft_data.py" --thinking_mode visual_only --max_workers 32 \
    --output_dir /path/to/scratch/infinigen/training_data_mix_all_rotation

log "Step 3: Creating mixed SFT training data (no_thinking mode)..."
python "$TRAINING_DATA_DIR/create_mix_all_sft_data.py" --thinking_mode no_thinking --max_workers 32 \
    --output_dir /path/to/scratch/infinigen/training_data_mix_all_rotation

log "Step 4: Creating JSON export with images..."
python "$TRAINING_DATA_DIR/create_mix_all_sft_data.py" --thinking_mode json_export --max_workers 32 \
    --output_dir /path/to/scratch/infinigen/training_data_mix_all_rotation/training_data_mix_all_balance

log "V5 pipeline complete!"
log "Output: /path/to/scratch/infinigen/training_data_mix_all_rotation/"
