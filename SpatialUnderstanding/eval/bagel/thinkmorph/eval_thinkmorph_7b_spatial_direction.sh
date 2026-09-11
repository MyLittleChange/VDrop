#!/bin/bash
#SBATCH --job-name=tm_spat_dir
#SBATCH --output=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/eval/bagel/thinkmorph/slurm_logs/tm_spat_dir_%j.out
#SBATCH --error=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/eval/bagel/thinkmorph/slurm_logs/tm_spat_dir_%j.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100l:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --partition=main

# ThinkMorph-7B on COSMIC spatial: direction only
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/path/to/scratch/models/ThinkMorph-7B}"
MODEL_TAG="thinkmorph_7b"

export MODEL_PATH
export TASK="direction"
export OUTPUT_DIR="${OUTPUT_DIR:-/path/to/scratch/VisualCoT/eval_results/${MODEL_TAG}/${TASK}}"
export THINK="true"
export THINKING_MODE="interleaved_thinking"
export NUM_SHARDS="${NUM_SHARDS:-2}"

mkdir -p "$OUTPUT_DIR"
exec bash /path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/eval/spatial/run_bagel_spatial_generic.sh
