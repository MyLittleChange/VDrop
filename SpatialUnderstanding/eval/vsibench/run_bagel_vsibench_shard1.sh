#!/bin/bash
#SBATCH --job-name=bagel_vsibench_shard1
#SBATCH --output=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/eval/vsibench/bagel_vsibench_shard1_output.txt
#SBATCH --error=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/eval/vsibench/bagel_vsibench_shard1_error.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=unkillable

#
# Run VSI-Bench shard 1/2 on a single GPU.
# Shard 0 already completed; this finishes the remaining half.
#
set -euo pipefail

SCRIPT_DIR="/path/to/ThinkMorph-BAGEL-release"

MODEL_PATH="/path/to/scratch/VisualCoT/BAGEL_checkpoints/BAGEL_format_training_data_mix_all_rotation_no_thinking"
DATA_DIR="/path/to/scratch/datasets/VSI-Bench"
DATASET_FILE="test_debiased.parquet"
OUTPUT_DIR="/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/eval/vsibench/BAGEL_format_training_data_mix_all_rotation_no_thinking"
OUTPUT_FILE="${OUTPUT_DIR}/inference_results_shard1.json"

MAX_MEM_PER_GPU="70GiB"
SEED=42
THINKING_MODE="no_thinking"
VIT_MIN_SIZE=512
NUM_FRAMES=8
GENERATED_IMAGES_DIR="${OUTPUT_DIR}/generated_images"

echo "=============================================="
echo "BAGEL VSI-Bench — Shard 1/2 (single GPU)"
echo "=============================================="
echo "  Model:    $MODEL_PATH"
echo "  Output:   $OUTPUT_FILE"
echo "=============================================="

module load cuda/12.6.0
unset ROCR_VISIBLE_DEVICES
source /path/to/scratch/morph_env/bin/activate

nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo ""

mkdir -p "$OUTPUT_DIR"

CUDA_VISIBLE_DEVICES=0 python3 "${SCRIPT_DIR}/inference/run_inference_bagel_vsibench.py" \
    --model_path "$MODEL_PATH" \
    --data_dir "$DATA_DIR" \
    --dataset_file "$DATASET_FILE" \
    --output_file "$OUTPUT_FILE" \
    --max_mem_per_gpu "$MAX_MEM_PER_GPU" \
    --shard "1/2" \
    --random_seed "$SEED" \
    --thinking_mode "$THINKING_MODE" \
    --vit_min_size "$VIT_MIN_SIZE" \
    --num_frames "$NUM_FRAMES" \
    --no_think \
    --generated_images_dir "$GENERATED_IMAGES_DIR"

echo ""
echo "Shard 1/2 complete! Results: $OUTPUT_FILE"
