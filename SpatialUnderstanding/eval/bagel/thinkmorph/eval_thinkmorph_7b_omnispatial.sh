#!/bin/bash
#SBATCH --job-name=tm_omni
#SBATCH --output=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/eval/bagel/thinkmorph/slurm_logs/tm_omni_%j.out
#SBATCH --error=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/eval/bagel/thinkmorph/slurm_logs/tm_omni_%j.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100l:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --partition=short-unkillable

# ThinkMorph-7B on OmniSpatial (Complex_Logic + Perspective_Taking) — think=True
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/path/to/scratch/models/ThinkMorph-7B}"
MODEL_TAG="thinkmorph_7b"

export MODEL_PATH
export OUTPUT_DIR="${OUTPUT_DIR:-/path/to/scratch/VisualCoT/eval_results/${MODEL_TAG}/omnispatial}"
export THINK="true"
export THINKING_MODE="interleaved_thinking"
export NUM_SHARDS="${NUM_SHARDS:-4}"

mkdir -p "$OUTPUT_DIR"
exec bash /path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/eval/omnispatial/run_bagel_omnispatial_generic.sh
