#!/bin/bash
#SBATCH --job-name=train_spatial_sg
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# Experiment-specific settings
EXPERIMENT="spatial_reasoning_sg"
OUTPUT_PATH="${OUTPUT_ROOT}/${EXPERIMENT}"
CKPT_PATH="${CKPT_ROOT}/spatial_reasoning_1K4_sg"
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
  --dataset_config_file ./configs/data/spatial_reasoning_scence_graph.yaml \
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
  --max_num_tokens 45000 \
  --vit_cond_dropout_prob 0 \
  --text_cond_dropout_prob 0 \
  --mse_weight 1 \
  --ce_weight 1 \
  --wandb_offline True \
  --wandb_name Spatial_Understanding_sg_${CLUSTER_NAME} \
  --total_steps 2000 \
  --save_every 500 \
  --num_shard ${NUM_GPUS}

log "=============================================="
log "Training Complete!"
log "=============================================="
