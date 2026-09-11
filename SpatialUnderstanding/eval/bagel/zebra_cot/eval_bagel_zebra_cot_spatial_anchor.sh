#!/bin/bash
#SBATCH --job-name=bzc_spat_anc
#SBATCH --output=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/eval/bagel/zebra_cot/slurm_logs/bzc_spat_anc_%j.out
#SBATCH --error=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/eval/bagel/zebra_cot/slurm_logs/bzc_spat_anc_%j.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100l:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=12:00:00
#SBATCH --partition=long

# Bagel-Zebra-CoT on COSMIC spatial: anchor only — auto-resumes from checkpoint
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/path/to/scratch/models/Bagel-Zebra-CoT}"
MODEL_TAG="bagel_zebra_cot"

export MODEL_PATH
export TASK="anchor"
export OUTPUT_DIR="${OUTPUT_DIR:-/path/to/scratch/VisualCoT/eval_results/${MODEL_TAG}/${TASK}}"
export THINK="false"
export THINKING_MODE="zebra_cot"
export MAX_ROUNDS="${MAX_ROUNDS:-3}"
export NUM_SHARDS="${NUM_SHARDS:-4}"

mkdir -p "$OUTPUT_DIR"
exec bash /path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/eval/spatial/run_bagel_spatial_generic.sh
