#!/bin/bash
# Submit benchmark eval jobs for the mix_all_balance bridge-masked 0.8-fraction LoRA 7k checkpoint:
#   BAGEL_format_mix_all_balance_visual_only_bridge_masked_0.8_lora_7k
#
# Subset chosen by user (only anchor + counting + direction on Mila, others on other clusters):
#   - anchor    -> short-unkillable (4-GPU)
#   - counting  -> main (2-GPU)
#   - direction -> long (2-GPU)
#   (MMSI / MindCube / OmniSpatial / STARE / BLINK run on another cluster.)
#
# Run after merge_lora has produced the BAGEL_format_* directory.
set -euo pipefail

SBATCH=/opt/slurm/bin/sbatch
EVAL=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/eval

CKPT=mix_all_balance_visual_only_bridge_masked_0.8_lora_7k

$SBATCH "$EVAL/spatial/run_bagel_spatial_4gpu_anchor_${CKPT}.sh"
$SBATCH "$EVAL/spatial/run_bagel_spatial_4gpu_counting_${CKPT}.sh"
$SBATCH "$EVAL/spatial/run_bagel_spatial_4gpu_direction_${CKPT}.sh"

echo "All eval jobs submitted. Monitor with: /opt/slurm/bin/squeue -u $USER"
