#!/bin/bash
#SBATCH --job-name=tm_mmsi
#SBATCH --output=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/eval/bagel/thinkmorph/slurm_logs/tm_mmsi_%j.out
#SBATCH --error=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/eval/bagel/thinkmorph/slurm_logs/tm_mmsi_%j.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100l:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --partition=main

# ThinkMorph-7B on MMSI-Bench — think=True, interleaved_thinking
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/path/to/scratch/models/ThinkMorph-7B}"
MODEL_TAG="thinkmorph_7b"

export MODEL_PATH
export OUTPUT_DIR="${OUTPUT_DIR:-/path/to/scratch/VisualCoT/eval_results/${MODEL_TAG}/mmsi}"
export THINK="true"
export THINKING_MODE="interleaved_thinking"
export NUM_SHARDS="${NUM_SHARDS:-2}"

mkdir -p "$OUTPUT_DIR"
exec bash /path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/eval/mmsi/eval_mmsi_bench_generic.sh
