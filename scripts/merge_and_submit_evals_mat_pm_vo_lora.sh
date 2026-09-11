#!/bin/bash
#SBATCH --job-name=merge_mat_pm_vo_lora
#SBATCH --output=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/bagel/merge_mat_pm_vo_lora_output.txt
#SBATCH --error=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/bagel/merge_mat_pm_vo_lora_error.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=long-cpu

#
# Merge the matterport_point_matching visual_only LoRA into base BAGEL,
# then submit the 6-benchmark eval suite. CPU-only (no GPU needed for merge).
#
set -euo pipefail

CKPT=training_data_mix_balance_matterport_point_matching_visual_only_lora
STEP=0009000
LORA_RANK=32
LORA_ALPHA=64.0

LOCAL_BASE=/path/to/scratch/VisualCoT/BAGEL_checkpoints
LOCAL_LORA_DIR=${LOCAL_BASE}/${CKPT}/${STEP}
LOCAL_MERGED=${LOCAL_BASE}/BAGEL_format_${CKPT}

REPO=/path/to/ThinkMorph-BAGEL-release

source /path/to/scratch/morph_env/bin/activate

echo "[$(date)] Merging LoRA into base model -> ${LOCAL_MERGED} ..."
python "$REPO/tools/merge_lora.py" \
    --base_model_path /path/to/scratch/models/BAGEL-7B-MoT \
    --lora_adapter_path "$LOCAL_LORA_DIR" \
    --output_path "$LOCAL_MERGED" \
    --lora_rank "$LORA_RANK" \
    --lora_alpha "$LORA_ALPHA"
echo "[$(date)] Merge done."

echo "[$(date)] Submitting eval suite ..."
bash "$REPO/scripts/submit_evals_mat_pm_vo_lora.sh"
echo "[$(date)] All eval jobs submitted."
