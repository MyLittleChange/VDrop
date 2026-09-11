#!/bin/bash
#SBATCH --job-name=bzc_mindcube
#SBATCH --output=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/eval/bagel/zebra_cot/slurm_logs/bzc_mindcube_%j.out
#SBATCH --error=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/eval/bagel/zebra_cot/slurm_logs/bzc_mindcube_%j.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100l:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=3:00:00
#SBATCH --partition=short-unkillable

# Bagel-Zebra-CoT zero-shot on MindCube
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/path/to/scratch/models/Bagel-Zebra-CoT}"
MODEL_TAG="bagel_zebra_cot"

export MODEL_PATH
export OUTPUT_DIR="${OUTPUT_DIR:-/path/to/scratch/VisualCoT/eval_results/${MODEL_TAG}/mindcube}"
export THINK="false"
export THINKING_MODE="zebra_cot"
export MAX_ROUNDS="${MAX_ROUNDS:-3}"
export NUM_SHARDS="${NUM_SHARDS:-4}"

mkdir -p "$OUTPUT_DIR"
exec bash /path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/eval/mindcube/run_bagel_mindcube_generic.sh
