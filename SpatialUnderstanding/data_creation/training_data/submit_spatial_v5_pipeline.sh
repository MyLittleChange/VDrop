#!/bin/bash
# =============================================================================
# Submit the entire Spatial V5 pipeline with SLURM dependency chaining.
#
# Usage: bash SpatialUnderstanding/data_creation/training_data/submit_spatial_v5_pipeline.sh
#
# Pipeline:
#   1. pregpu  (main, CPU)     → scene_filtering, object_info, camera_info, blender_color
#   2. gpu     (4xA100L)       → scene_llm_visible_objects (vLLM Qwen3-VL-235B)
#   3. postgpu (main, CPU)     → bound_objects → filter_questions, produces V5 JSONs
#   4. postprocess (main, CPU) → normalize → filter_test → prepare_panorama → submit render
#      4a. render (long, CPU array) → panorama rendering
#      4b. finalize (main, CPU)     → filter_rendered → create_mix_all (Parquet + JSONL)
#
# Jobs 4a and 4b are submitted by the postprocess job itself (with dependencies).
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "================================================"
echo "Submitting Spatial V5 Full Pipeline"
echo "================================================"

# Step 1: Pre-GPU
JOB1=$(sbatch --parsable "$SCRIPT_DIR/run_datagen_spatial_pregpu.sh")
echo "1. PreGPU:      $JOB1"

# Step 2: GPU (depends on pregpu)
JOB2=$(sbatch --parsable --dependency=afterok:${JOB1} "$SCRIPT_DIR/run_datagen_spatial_gpu.sh")
echo "2. GPU:         $JOB2  (after $JOB1)"

# Step 3: PostGPU (depends on gpu)
JOB3=$(sbatch --parsable --dependency=afterok:${JOB2} "$SCRIPT_DIR/run_datagen_spatial_postgpu.sh")
echo "3. PostGPU:     $JOB3  (after $JOB2)"

# Step 4: Postprocess (depends on postgpu)
# This job will internally submit the panorama render array + finalize job
JOB4=$(sbatch --parsable --dependency=afterok:${JOB3} "$SCRIPT_DIR/run_spatial_v5_postprocess.sh")
echo "4. Postprocess: $JOB4  (after $JOB3)"
echo "   (will submit render array + finalize automatically)"

echo ""
echo "================================================"
echo "Pipeline submitted: $JOB1 → $JOB2 → $JOB3 → $JOB4 → [render] → [finalize]"
echo "================================================"
echo ""
echo "Monitor: squeue -u $USER"
echo "Cancel all: scancel $JOB1 $JOB2 $JOB3 $JOB4"
echo ""
echo "Note: The postprocess job ($JOB4) will submit additional jobs:"
echo "  - Panorama render array (long partition)"
echo "  - Finalize job (after render completes)"
echo "Check squeue after postprocess starts for their job IDs."
