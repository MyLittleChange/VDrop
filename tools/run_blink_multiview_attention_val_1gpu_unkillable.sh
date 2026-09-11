#!/usr/bin/env bash
#SBATCH --job-name=blink_mv_attn_vanilla
#SBATCH --output=/path/to/ThinkMorph-BAGEL-release/artifacts/blink_multiview_attention_val_vanilla_unkillable_output.txt
#SBATCH --error=/path/to/ThinkMorph-BAGEL-release/artifacts/blink_multiview_attention_val_vanilla_unkillable_error.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --partition=unkillable

set -euo pipefail

module load cuda/12.6.0
unset ROCR_VISIBLE_DEVICES
source /path/to/scratch/morph_env/bin/activate

export GPUS_PER_JOB=1
export TOTAL_SHARDS="${TOTAL_SHARDS:-1}"
export RUN_SET=vanilla_force_bridge

exec /usr/bin/env bash /path/to/ThinkMorph-BAGEL-release/tools/run_blink_multiview_attention_val_common.sh
