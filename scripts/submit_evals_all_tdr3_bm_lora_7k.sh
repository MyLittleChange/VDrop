#!/bin/bash
# Submit benchmark eval jobs for the topdown_round3 bridge-masked LoRA 7k checkpoint:
#   BAGEL_format_topdown_round3_visual_only_bridge_masked_lora_7k
#
# Subset chosen by user (other benchmarks evaluated on different clusters):
#   - anchor + counting -> main (2-GPU)
#   - distance + direction + map -> short-unkillable (4-GPU)
#   (MMSI / MindCube / OmniSpatial / STARE / BLINK run on another cluster.)
#
# Run after merge_lora has produced the BAGEL_format_* directory.
set -euo pipefail

SBATCH=/opt/slurm/bin/sbatch
EVAL=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/eval

CKPT=topdown_round3_visual_only_bridge_masked_lora_7k

for TASK in anchor counting distance direction; do
    $SBATCH "$EVAL/spatial/run_bagel_spatial_4gpu_${TASK}_${CKPT}.sh"
done
$SBATCH "$EVAL/spatial/run_bagel_spatial_map_4gpu_${CKPT}.sh"

echo "All eval jobs submitted. Monitor with: /opt/slurm/bin/squeue -u $USER"
