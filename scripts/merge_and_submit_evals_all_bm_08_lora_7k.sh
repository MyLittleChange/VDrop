#!/bin/bash
#SBATCH --job-name=merge_all_bm_08_lora_7k
#SBATCH --output=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/bagel/merge_all_bm_08_lora_7k_output.txt
#SBATCH --error=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/bagel/merge_all_bm_08_lora_7k_error.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=long-cpu

#
# Merge the mix_all_balance_visual_only_bridge_masked_0.8 LoRA step 7000 into base BAGEL,
# then submit 3 spatial eval jobs (anchor on short-unkillable 4-GPU; counting on main 2-GPU;
# direction on long 2-GPU per user).
# CPU-only (no GPU needed for merge).
#
# Bridge-masking ablation: BRIDGE_MASK_DROP_FRACTION=0.8 (heavier mask than default 0.5
# — 80% of the chosen view's patch tokens hidden during reflect/answer decoding).
# Source dir name has no `_lora_Nk` suffix (just `..._0.8/0007000/`).
#
set -euo pipefail

CKPT_SRC_DIR=mix_all_balance_visual_only_bridge_masked_0.8
CKPT_OUT_TAG=mix_all_balance_visual_only_bridge_masked_0.8_lora_7k
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
bash "$REPO/scripts/submit_evals_all_bm_08_lora_7k.sh"
echo "[$(date)] All eval jobs submitted."
