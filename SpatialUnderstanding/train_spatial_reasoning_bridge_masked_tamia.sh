#!/bin/bash
#SBATCH --job-name=train_bridge_masked
#SBATCH --output=/path/to/ThinkMorph-BAGEL-release/scripts/train_spatial_bridge_masked_output.txt
#SBATCH --error=/path/to/ThinkMorph-BAGEL-release/scripts/train_spatial_bridge_masked_error.txt
#SBATCH --ntasks=1

set -e

VENV_PATH="/path/to/scratch/venv_morph/bin/activate"
module load cuda/12.6
unset ROCR_VISIBLE_DEVICES
source "$VENV_PATH"

export WANDB_MODE=offline
export WANDB_DIR="/path/to/scratch/VisualCoT/wandb_logs"
export WANDB_CACHE_DIR="/path/to/scratch/.cache/wandb"

################################################################################
# CONFIGURATION
################################################################################

SCRIPT_DIR="/path/to/ThinkMorph-BAGEL-release"
OUTPUT_PATH="/path/to/scratch/VisualCoT/training_outputs/spatial_reasoning_bridge_masked"
CKPT_PATH="/path/to/scratch/VisualCoT/BAGEL_checkpoints/spatial_reasoning_bridge_masked_1K4"

BAGEL_MODEL_PATH="/path/to/scratch/models/BAGEL-7B-MoT"

mkdir -p "$OUTPUT_PATH"
mkdir -p "$CKPT_PATH"

################################################################################
# BRIDGE-MASKING KNOBS (read by BridgeMaskedPackedDataset via env vars)
################################################################################

export BRIDGE_MASK_WARMUP_STEPS=500
export BRIDGE_MASK_ANNEAL_STEPS=1500
export BRIDGE_MASK_DROP_FRACTION=0.5
export BRIDGE_MASK_DROP_STRATEGY=region   # region | random_patches
export BRIDGE_MASK_DROP=random            # random | V1 | V2 | both | none
                                          # `both` masks both V1 AND V2 each step
                                          # (each independently partial-masked).

################################################################################
# FUNCTIONS
################################################################################

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

################################################################################
# RUN TRAINING
################################################################################

log "=============================================="
log "Starting Bridge-Masked Spatial Reasoning Training"
log "=============================================="

cd "$SCRIPT_DIR"

export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=4 \
  --master_addr=localhost \
  --master_port=29500 \
  train/pretrain_unified_navit_bridge_masked.py \
  --model_path "$BAGEL_MODEL_PATH" \
  --dataset_config_file ./configs/data/spatial_reasoning_bridge_masked.yaml \
  --layer_module Qwen2MoTDecoderLayer \
  --finetune_from_hf True \
  --auto_resume True \
  --finetune-from-ema True \
  --results_dir "$OUTPUT_PATH" \
  --checkpoint_dir "$CKPT_PATH" \
  --log_every 1 \
  --lr 1e-5 \
  --num_worker 1 \
  --max_latent_size 64 \
  --max_num_tokens 10000 \
  --vit_cond_dropout_prob 0 \
  --text_cond_dropout_prob 0 \
  --mse_weight 1 \
  --ce_weight 1 \
  --total_steps 2000 \
  --save_every 50 \
  --wandb_offline True \
  --num_shard 4

log "=============================================="
log "Training Complete!"
log "=============================================="
