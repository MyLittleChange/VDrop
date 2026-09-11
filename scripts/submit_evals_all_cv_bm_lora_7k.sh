#!/bin/bash
# Submit benchmark eval jobs for the corner-view bridge-masked LoRA 7k checkpoint:
#   BAGEL_format_corner_view_visual_only_bridge_masked_lora_7k
#
# Subset chosen by user:
#   - anchor, counting, distance, direction -> short-unkillable (4-GPU)
#   - BLINK Multi-view_Reasoning + STARE-Perspective -> main (2-GPU)
#   (Map, MMSI, MindCube, OmniSpatial run on another cluster.)
#
# Run after merge_lora has produced the BAGEL_format_* directory.
set -euo pipefail

SBATCH=/opt/slurm/bin/sbatch
EVAL=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/eval

CKPT=corner_view_visual_only_bridge_masked_lora_7k

for TASK in anchor counting distance direction; do
    $SBATCH "$EVAL/spatial/run_bagel_spatial_4gpu_${TASK}_${CKPT}.sh"
done

$SBATCH "$EVAL/blink/run_bagel_blink_multiview_2gpu_${CKPT}.sh"
$SBATCH "$EVAL/stare/run_bagel_stare_perspective_2gpu_${CKPT}.sh"

echo "All eval jobs submitted. Monitor with: /opt/slurm/bin/squeue -u $USER"
