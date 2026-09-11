#!/usr/bin/env bash
#SBATCH --job-name=blink_mv_attn_trained
#SBATCH --output=/path/to/ThinkMorph-BAGEL-release/artifacts/blink_multiview_attention_val_trained_short_unkillable_output.txt
#SBATCH --error=/path/to/ThinkMorph-BAGEL-release/artifacts/blink_multiview_attention_val_trained_short_unkillable_error.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100l:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=3:00:00
#SBATCH --partition=short-unkillable
#SBATCH --exclude=cn-g[001-007,009-011,014,017-020,022,024,027]

set -euo pipefail

module load cuda/12.6.0
unset ROCR_VISIBLE_DEVICES
source /path/to/scratch/morph_env/bin/activate

export GPUS_PER_JOB=4
export TOTAL_SHARDS="${TOTAL_SHARDS:-4}"
export RUN_SET=trained_pair

exec /usr/bin/env bash /path/to/ThinkMorph-BAGEL-release/tools/run_blink_multiview_attention_val_common.sh
