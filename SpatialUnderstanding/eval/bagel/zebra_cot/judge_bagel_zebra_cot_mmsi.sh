#!/bin/bash
#SBATCH --job-name=bzc_mmsi_judge
#SBATCH --output=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/eval/bagel/zebra_cot/slurm_logs/bzc_mmsi_judge_%j.out
#SBATCH --error=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/eval/bagel/zebra_cot/slurm_logs/bzc_mmsi_judge_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=long-cpu

# Gemini-3-Flash judge over MMSI-Bench Bagel-Zebra-CoT inference (1000 samples in 2 shards)
set -euo pipefail

source /path/to/scratch/morph_env/bin/activate

RESULTS_DIR="/path/to/scratch/VisualCoT/eval_results/bagel_zebra_cot/mmsi"
OUTPUT_DIR="${RESULTS_DIR}"
RUN_NUM=1
NUM_SHARDS=2
JUDGE_MODEL="gemini-3-flash-preview"
NUM_PROCESSES=16

python3 /path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/eval/mmsi/merge_and_eval_mmsi.py \
    --results_dir "${RESULTS_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --run_num "${RUN_NUM}" \
    --num_shards "${NUM_SHARDS}" \
    --run_judge \
    --judge_model "${JUDGE_MODEL}" \
    --num_processes "${NUM_PROCESSES}"

echo "Done. Evaluated file: ${OUTPUT_DIR}/evaluated_model_results_run_${RUN_NUM}.json"
