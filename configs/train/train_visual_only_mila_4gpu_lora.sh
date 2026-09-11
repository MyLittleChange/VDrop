#!/bin/bash
#SBATCH --job-name=train_vo_lora
#SBATCH --output=/path/to/ThinkMorph-BAGEL-release/configs/train/train_visual_only_lora_output.txt
#SBATCH --error=/path/to/ThinkMorph-BAGEL-release/configs/train/train_visual_only_lora_error.txt
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
OUTPUT_PATH="/path/to/scratch/VisualCoT/training_outputs/visual_only_mix_all_lora"
CKPT_PATH="/path/to/scratch/VisualCoT/BAGEL_checkpoints/visual_only_mix_all_lora"

BAGEL_MODEL_PATH="/path/to/scratch/models/BAGEL-7B-MoT"

mkdir -p "$OUTPUT_PATH"
mkdir -p "$CKPT_PATH"

################################################################################

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

log "=============================================="
log "Starting Visual-Only Thinking LoRA Training (mix_all, 4x A100)"
log "=============================================="

cd "$SCRIPT_DIR"

export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Visual-only-thinking is a generation task — LoRA targets both MoT paths:
# q_proj/k_proj/v_proj/o_proj (understanding) AND q_proj_moe_gen/... (generation),
# plus mlp.gate/up/down (und) AND mlp_moe_gen.gate/up/down (gen).
torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=4 \
  --master_addr=localhost \
  --master_port=29500 \
  train/pretrain_unified_navit.py \
  --model_path "$BAGEL_MODEL_PATH" \
  --dataset_config_file ./configs/data/visual_only_mix_all.yaml \
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
  --max_num_tokens 45000 \
  --vit_cond_dropout_prob 0 \
  --text_cond_dropout_prob 0 \
  --mse_weight 1 \
  --ce_weight 1 \
  --wandb_offline True \
  --wandb_name Visual_Only_mix_all_lora_mila \
  --total_steps 3000 \
  --save_every 300 \
  --num_shard 4 \
  --use_lora True \
  --lora_rank 32 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --lora_target_regex '.*\.self_attn\.(q|k|v|o)_proj(_moe_gen)?$|.*\.mlp(_moe_gen)?\.(gate|up|down)_proj$'

log "=============================================="
log "Training Complete!"
log "=============================================="
