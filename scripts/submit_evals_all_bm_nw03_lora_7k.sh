#!/bin/bash
# Submit benchmark eval jobs for the mix_all_balance bridge-masked no-warmup 0.3-fraction LoRA 7k checkpoint:
#   BAGEL_format_mix_all_balance_visual_only_bridge_masked_no_warmup_0.3_lora_7k
#
# Subset chosen by user (other benchmarks evaluated on different clusters):
#   - all 5 spatial scripts -> short-unkillable (4-GPU)
#   (MMSI / MindCube / OmniSpatial / STARE / BLINK run on another cluster.)
#
# Run after merge_lora has produced the BAGEL_format_* directory.
set -euo pipefail

SBATCH=/opt/slurm/bin/sbatch
EVAL=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/eval

CKPT=mix_all_balance_visual_only_bridge_masked_no_warmup_0.3_lora_7k

for TASK in anchor counting distance direction; do
    $SBATCH "$EVAL/spatial/run_bagel_spatial_4gpu_${TASK}_${CKPT}.sh"
done
$SBATCH "$EVAL/spatial/run_bagel_spatial_map_4gpu_${CKPT}.sh"

echo "All eval jobs submitted. Monitor with: /opt/slurm/bin/squeue -u $USER"
