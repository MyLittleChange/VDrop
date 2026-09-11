#!/bin/bash
#SBATCH --job-name=train_spatial_multinode
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --mem=1000G

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# Experiment-specific settings
EXPERIMENT="spatial_reasoning"
OUTPUT_PATH="${OUTPUT_ROOT}/${EXPERIMENT}"
CKPT_PATH="${CKPT_ROOT}/spatial_reasoning_1K4"
mkdir -p "$OUTPUT_PATH" "$CKPT_PATH"

# Multi-node setup
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500
export MASTER_ADDR MASTER_PORT
export OUTPUT_PATH CKPT_PATH BAGEL_MODEL_PATH NUM_GPUS

log "Master address: $MASTER_ADDR"
log "Total nodes: $SLURM_NNODES"

log "=============================================="
log "Starting ThinkMorph Spatial Reasoning Training (Multi-Node)"
log "=============================================="

cd "$SCRIPT_DIR"

# Multi-node: use srun to launch torchrun on all nodes
srun --ntasks-per-node=1 bash -c "torchrun \
  --nnodes=$SLURM_NNODES \
  --node_rank=\$SLURM_PROCID \
  --nproc_per_node=${NUM_GPUS} \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  train/pretrain_unified_navit.py \
  --model_path $BAGEL_MODEL_PATH \
  --dataset_config_file ./configs/data/spatial_reasoning.yaml \
  --layer_module Qwen2MoTDecoderLayer \
  --finetune_from_hf True \
  --auto_resume True \
  --resume_from $BAGEL_MODEL_PATH \
  --finetune-from-ema True \
  --results_dir $OUTPUT_PATH \
  --checkpoint_dir $CKPT_PATH \
  --log_every 1 \
  --lr 1e-5 \
  --num_worker 1 \
  --max_latent_size 64 \
  --max_num_tokens 20000 \
  --vit_cond_dropout_prob 0 \
  --text_cond_dropout_prob 0 \
  --mse_weight 1 \
  --ce_weight 1 \
  --total_steps 2000 \
  --save_every 100 \
  --num_shard $((SLURM_NNODES * NUM_GPUS))"

log "=============================================="
log "Training Complete!"
log "=============================================="
