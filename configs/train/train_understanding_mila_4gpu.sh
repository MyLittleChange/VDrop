#!/bin/bash
#SBATCH --job-name=train_understanding
#SBATCH --output=/path/to/ThinkMorph-BAGEL-release/configs/train/train_understanding_output.txt
#SBATCH --error=/path/to/ThinkMorph-BAGEL-release/configs/train/train_understanding_error.txt
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100l:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=0G
#SBATCH --partition=short-unkillable
#SBATCH --exclude=cn-g[001-007,009-011,014,017-020,022,024,027]

set -e

VENV_PATH="/path/to/scratch/morph_env/bin/activate"
module load cuda/12.6.0
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
OUTPUT_PATH="/path/to/scratch/VisualCoT/training_outputs/understanding"
CKPT_PATH="/path/to/scratch/VisualCoT/BAGEL_checkpoints/understanding"

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
log "Starting ThinkMorph Understanding Training (4x A100)"
log "=============================================="

cd "$SCRIPT_DIR"

# Add ThinkMorph to Python path so 'data' module can be found
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Memory optimization - helps with fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Single node, 4 GPUs
torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=4 \
  --master_addr=localhost \
  --master_port=29500 \
  train/pretrain_unified_navit.py \
  --model_path "$BAGEL_MODEL_PATH" \
  --dataset_config_file ./configs/data/understanding.yaml \
  --layer_module Qwen2MoTDecoderLayer \
  --finetune_from_hf True \
  --auto_resume True \
  --resume_from $BAGEL_MODEL_PATH \
  --finetune-from-ema True \
  --resume_model_only \
  --results_dir "$OUTPUT_PATH" \
  --checkpoint_dir "$CKPT_PATH" \
  --log_every 1 \
  --lr 1e-5 \
  --num_worker 1 \
  --max_latent_size 64 \
  --max_num_tokens 10240 \
  --visual_gen False \
  --wandb_offline True \
  --wandb_name Understanding_mila \
  --total_steps 6000 \
  --save_every 500 \
  --num_shard 4

log "=============================================="
log "Training Complete!"
log "=============================================="
