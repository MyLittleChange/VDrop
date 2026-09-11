#!/bin/bash
#SBATCH --job-name=train_spatial
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# Experiment-specific settings
EXPERIMENT="spatial_reasoning"
OUTPUT_PATH="${OUTPUT_ROOT}/${EXPERIMENT}"
CKPT_PATH="${CKPT_ROOT}/spatial_reasoning_1K4"
mkdir -p "$OUTPUT_PATH" "$CKPT_PATH"

log "=============================================="
log "Starting ThinkMorph Spatial Reasoning Training"
log "=============================================="

cd "$SCRIPT_DIR"

torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=${NUM_GPUS} \
  --master_addr=localhost \
  --master_port=29500 \
  train/pretrain_unified_navit.py \
  --model_path "$BAGEL_MODEL_PATH" \
  --dataset_config_file ./configs/data/spatial_reasoning.yaml \
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
  --max_num_tokens 3000 \
  --vit_cond_dropout_prob 0 \
  --text_cond_dropout_prob 0 \
  --mse_weight 1 \
  --ce_weight 1 \
  --total_steps 2000 \
  --save_every 50 \
  --num_shard ${NUM_GPUS}

log "=============================================="
log "Training Complete!"
log "=============================================="
