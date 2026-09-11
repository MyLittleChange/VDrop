#!/bin/bash
#SBATCH --job-name=train_text_only_sg
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# Experiment-specific settings
EXPERIMENT="text_only_thinking_sg"
OUTPUT_PATH="${OUTPUT_ROOT}/${EXPERIMENT}"
CKPT_PATH="${CKPT_ROOT}/${EXPERIMENT}"
mkdir -p "$OUTPUT_PATH" "$CKPT_PATH"

log "=============================================="
log "Starting ThinkMorph Text-Only Thinking Training"
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
  --dataset_config_file ./configs/data/text_only_thinking_scence_graph.yaml \
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
  --ce_weight 1 \
  --wandb_offline True \
  --wandb_name Text_Only_Thinking_sg_${CLUSTER_NAME} \
  --total_steps 2000 \
  --save_every 500 \
  --num_shard ${NUM_GPUS}

log "=============================================="
log "Training Complete!"
log "=============================================="
