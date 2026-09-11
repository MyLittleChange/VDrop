#!/bin/bash
#SBATCH --job-name=merge_all_bm_map_lora_7k
#SBATCH --output=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/bagel/merge_all_bm_map_lora_7k_output.txt
#SBATCH --error=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/bagel/merge_all_bm_map_lora_7k_error.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=long-cpu

#
# Merge the mix_all_balance_with_map bridge-masked LoRA step 7000 into base BAGEL,
# then submit the 5-script spatial eval suite (all on main 2-GPU per user).
# CPU-only (no GPU needed for merge).
#
# Combines the mix_all_balance + map-task augmented training data with bridge-necessity
# attention masking. Source dir name says `_lora_6k/0007000/` but the step inside is
# 7000; merged output uses `_7k` to match the actual step.
#
set -euo pipefail

CKPT_SRC_DIR=mix_all_balance_with_map_visual_only_bridge_masked_lora_6k
CKPT_OUT_TAG=mix_all_balance_with_map_visual_only_bridge_masked_lora_7k
STEP=0007000
LORA_RANK=32
LORA_ALPHA=64.0

LORA_BASE=/path/to/scratch/infinigen
LOCAL_LORA_DIR=${LORA_BASE}/${CKPT_SRC_DIR}/${STEP}

LOCAL_BASE=/path/to/scratch/VisualCoT/BAGEL_checkpoints
LOCAL_MERGED=${LOCAL_BASE}/BAGEL_format_${CKPT_OUT_TAG}

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
bash "$REPO/scripts/submit_evals_all_bm_map_lora_7k.sh"
echo "[$(date)] All eval jobs submitted."
