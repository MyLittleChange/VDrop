#!/bin/bash
#SBATCH --job-name=tm_mindcube
#SBATCH --output=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/eval/bagel/thinkmorph/slurm_logs/tm_mindcube_%j.out
#SBATCH --error=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/eval/bagel/thinkmorph/slurm_logs/tm_mindcube_%j.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100l:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=3:00:00
#SBATCH --partition=short-unkillable

# ThinkMorph-7B on MindCube — think=True, interleaved_thinking
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/path/to/scratch/models/ThinkMorph-7B}"
MODEL_TAG="thinkmorph_7b"

export MODEL_PATH
export OUTPUT_DIR="${OUTPUT_DIR:-/path/to/scratch/VisualCoT/eval_results/${MODEL_TAG}/mindcube}"
export THINK="true"
export THINKING_MODE="interleaved_thinking"
export NUM_SHARDS="${NUM_SHARDS:-4}"

mkdir -p "$OUTPUT_DIR"
exec bash /path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/eval/mindcube/run_bagel_mindcube_generic.sh
