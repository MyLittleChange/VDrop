#!/bin/bash
#SBATCH --job-name=train_spatial
#SBATCH --output=/path/to/ThinkMorph-BAGEL-release/scripts/train_spatial_output.txt
#SBATCH --error=/path/to/ThinkMorph-BAGEL-release/scripts/train_spatial_error.txt
#SBATCH --ntasks=1

set -e

VENV_PATH="/path/to/scratch/venv_morph/bin/activate"
module load cuda/12.6
unset ROCR_VISIBLE_DEVICES
source "$VENV_PATH"

export WANDB_MODE=offline

# Redirect W&B metadata and logs to scratch
export WANDB_DIR="/path/to/scratch/VisualCoT/wandb_logs"
export WANDB_CACHE_DIR="/path/to/scratch/.cache/wandb"

################################################################################
# CONFIGURATION
################################################################################

SCRIPT_DIR="/path/to/ThinkMorph-BAGEL-release"
OUTPUT_PATH="/path/to/scratch/VisualCoT/training_outputs/spatial_reasoning"
CKPT_PATH="/path/to/scratch/VisualCoT/BAGEL_checkpoints/spatial_reasoning_1K4"

# BAGEL pretrained model path (finetune from this checkpoint)
BAGEL_MODEL_PATH="/path/to/scratch/models/BAGEL-7B-MoT"

# Create output directories
mkdir -p "$OUTPUT_PATH"
mkdir -p "$CKPT_PATH"

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
log "Starting ThinkMorph Spatial Reasoning Training"
log "=============================================="

cd "$SCRIPT_DIR"

# Add ThinkMorph to Python path so 'data' module can be found
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Single node, 4 GPUs
torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=4 \
  --master_addr=localhost \
  --master_port=29500 \
  train/pretrain_unified_navit.py \
  --model_path "$BAGEL_MODEL_PATH" \
  --dataset_config_file ./data/configs/spatial_reasoning.yaml \
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
