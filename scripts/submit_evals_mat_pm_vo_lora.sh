#!/bin/bash
# Submit all 6 benchmark eval jobs for the new visual_only LoRA checkpoint:
#   BAGEL_format_training_data_mix_balance_matterport_point_matching_visual_only_lora
#
# Run after merge_lora has produced the BAGEL_format_* directory.
set -euo pipefail

SBATCH=/opt/slurm/bin/sbatch
EVAL=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/eval

for TASK in anchor counting distance direction; do
    $SBATCH "$EVAL/spatial/run_bagel_spatial_4gpu_${TASK}_mix_balance_matterport_point_matching_visual_only_lora.sh"
done
$SBATCH "$EVAL/spatial/run_bagel_spatial_map_4gpu_mix_balance_matterport_point_matching_visual_only_lora.sh"

$SBATCH "$EVAL/mmsi/eval_mmsi_bench_2gpu_mix_balance_matterport_point_matching_visual_only_lora.sh"
$SBATCH "$EVAL/mindcube/run_bagel_mindcube_4gpu_mix_balance_matterport_point_matching_visual_only_lora.sh"
$SBATCH "$EVAL/omnispatial/run_bagel_omnispatial_4gpu_mix_balance_matterport_point_matching_visual_only_lora.sh"
$SBATCH "$EVAL/stare/run_bagel_stare_perspective_1gpu_mix_balance_matterport_point_matching_visual_only_lora.sh"
$SBATCH "$EVAL/blink/run_bagel_blink_multiview_1gpu_mix_balance_matterport_point_matching_visual_only_lora.sh"

echo "All eval jobs submitted. Monitor with: /opt/slurm/bin/squeue -u $USER"
