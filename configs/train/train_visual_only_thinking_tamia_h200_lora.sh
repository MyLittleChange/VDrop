#!/bin/bash
#SBATCH --job-name=train_visual_only_rotation_balance_lora
#SBATCH --output=/path/to/ThinkMorph-BAGEL-release/scripts/train_visual_only_rotation_balance_h200_lora_output.txt
#SBATCH --error=/path/to/ThinkMorph-BAGEL-release/scripts/train_visual_only_rotation_balance_h200_lora_error.txt
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1       # 每个节点运行一个任务（由 torchrun 管理多进程）
#SBATCH --gres=gpu:h200:8              # 每个节点申请 8 张 GPU
#SBATCH --cpus-per-task=64        # 根据你的集群配置调整 CPU 核心数
#SBATCH --mem=0G
#SBATCH --time=20:00:00

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
OUTPUT_PATH="/path/to/scratch/VisualCoT/training_outputs/training_data_mix_all_rotation_balance_visual_only_matterport_rotation_lora"
CKPT_PATH="/path/to/scratch/VisualCoT/BAGEL_checkpoints/training_data_mix_all_rotation_balance_visual_only_matterport_rotation_lora"

# BAGEL pretrained model path (frozen base for LoRA)
BAGEL_MODEL_PATH="/path/to/scratch/models/BAGEL-7B-MoT_training_data_mix_all_rotation_balance_visual_only"

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
log "Starting ThinkMorph Visual-Only rotation_balance LoRA Training"
log "=============================================="

cd "$SCRIPT_DIR"

# Add ThinkMorph to Python path so 'data' module can be found
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Single node, 8 GPUs
# Visual-only-thinking is a generation task — LoRA targets BOTH MoT paths:
# self_attn.q/k/v/o_proj + q/k/v/o_proj_moe_gen, and mlp + mlp_moe_gen MLPs.
# lr bumped from 1e-5 → 1e-4 (standard LoRA practice).
torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=8 \
  --master_addr=localhost \
  --master_port=29500 \
  train/pretrain_unified_navit.py \
  --model_path "$BAGEL_MODEL_PATH" \
  --dataset_config_file ./data/configs/visual_only_thinking_rotation_balance.yaml \
  --layer_module Qwen2MoTDecoderLayer \
  --finetune_from_hf True \
  --auto_resume True \
  --resume_from $BAGEL_MODEL_PATH \
  --finetune-from-ema True \
  --resume_model_only \
  --results_dir "$OUTPUT_PATH" \
  --checkpoint_dir "$CKPT_PATH" \
  --log_every 1 \
  --lr 1e-4 \
  --num_worker 1 \
  --max_latent_size 64 \
  --max_num_tokens_per_sample 20000 \
  --max_num_tokens 45000 \
  --vit_cond_dropout_prob 0 \
  --text_cond_dropout_prob 0 \
  --mse_weight 1 \
  --ce_weight 1 \
  --wandb_offline True \
  --wandb_name training_data_mix_all_rotation_balance_visual_only_matterport_rotation_lora \
  --total_steps 4000 \
  --save_every 2000 \
  --num_shard 8 \
  --use_lora True \
  --lora_rank 32 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --lora_target_regex '.*\.self_attn\.(q|k|v|o)_proj(_moe_gen)?$|.*\.mlp(_moe_gen)?\.(gate|up|down)_proj$'

log "=============================================="
log "Training Complete!"
log "=============================================="
