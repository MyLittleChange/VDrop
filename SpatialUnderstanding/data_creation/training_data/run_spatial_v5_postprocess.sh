#!/bin/bash
#SBATCH --job-name=V5PostProc
#SBATCH -c 4
#SBATCH --mem=32Gb
#SBATCH --time=4:00:00
#SBATCH --output=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/data_creation/training_data/slurm_logs/v5_postprocess_output-%j.txt
#SBATCH --error=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/data_creation/training_data/slurm_logs/v5_postprocess_error-%j.txt

# ==============================================================================
# V5 post-process: normalize → filter_test → prepare_panorama → submit render
# Runs after postgpu completes. Submits panorama render array + finalize job.
# ==============================================================================

set -e

source /path/to/scratch/morph_bpy_env/bin/activate

PROJECT_DIR="/path/to/ThinkMorph-BAGEL-release"
TRAINING_DATA_DIR="$PROJECT_DIR/SpatialUnderstanding/data_creation/training_data"
RENDERING_DIR="$PROJECT_DIR/SpatialUnderstanding/data_creation/rendering"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

cd "$PROJECT_DIR"

################################################################################
# Step 1: Normalize V5 datasets
################################################################################
log "Step 1: Normalizing V5 datasets..."
python tools/dataset_utils/normalize_v5_datasets.py

################################################################################
# Step 2: Filter test sets
################################################################################
log "Step 2: Filtering test sets for V5..."
python tools/dataset_utils/filter_test_set_v5.py

################################################################################
# Step 3: Prepare panorama split
################################################################################
log "Step 3: Preparing panorama V5 split..."
python "$TRAINING_DATA_DIR/prepare_panorama_v5_split.py"

################################################################################
# Step 4: Submit panorama render array job
################################################################################
log "Step 4: Submitting panorama render job..."
RENDER_JOB_ID=$(bash "$RENDERING_DIR/batch_render_panorama_v5.sh")
log "Panorama render array job: $RENDER_JOB_ID"

################################################################################
# Step 5: Submit finalize job with dependency on render completion
################################################################################
log "Step 5: Submitting finalize job (depends on render $RENDER_JOB_ID)..."
FINALIZE_JOB_ID=$(sbatch --parsable --dependency=afterok:${RENDER_JOB_ID} \
    "$TRAINING_DATA_DIR/run_spatial_v5_finalize.sh")
log "Finalize job: $FINALIZE_JOB_ID (depends on render $RENDER_JOB_ID)"

log "Post-processing complete. Pipeline continues:"
log "  Render: $RENDER_JOB_ID (array)"
log "  Finalize: $FINALIZE_JOB_ID (after render)"
