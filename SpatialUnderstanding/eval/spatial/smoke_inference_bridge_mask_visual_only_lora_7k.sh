#!/bin/bash
# Smoke test for the inference-time bridge-necessity probe.
#
# Runs the visual_only_lora_7k checkpoint on 8 counting-task samples with the
# `--inference_bridge_mask` flag enabled. The bridge image is generated as
# usual but its KV slice is zeroed before the answer decode, so answer tokens
# cannot attend to the bridge. Compare the resulting accuracy / answer texts
# with the corresponding flag-off run on the same samples to confirm the
# intervention has the expected effect.
#
# Submit with:
#   /opt/slurm/bin/sbatch SpatialUnderstanding/eval/spatial/smoke_inference_bridge_mask_visual_only_lora_7k.sh

#SBATCH --job-name=bagel_smoke_inf_bm
#SBATCH --output=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/bagel/bagel_smoke_inference_bridge_mask_output.txt
#SBATCH --error=/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/bagel/bagel_smoke_inference_bridge_mask_error.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --partition=short-unkillable

set -euo pipefail

SCRIPT_DIR="/path/to/ThinkMorph-BAGEL-release"
MODEL_PATH="${MODEL_PATH:-/path/to/scratch/VisualCoT/BAGEL_checkpoints/BAGEL_format_training_data_mix_all_balance_visual_only_lora_7k}"
DATA_DIR="${DATA_DIR:-/path/to/scratch/VisualCoT/spatial_collab_dataset}"
DATASET_FILES="${DATASET_FILES:-approved_mcqs_counting_normalized.json}"
OUTPUT_BASE="${OUTPUT_BASE:-/path/to/scratch/VisualCoT/smoke_inference_bridge_mask}"
NUM_SAMPLES="${NUM_SAMPLES:-8}"
SEED="${SEED:-42}"

if [[ ! -d "$MODEL_PATH" ]]; then echo "ERROR: Model not found: $MODEL_PATH"; exit 1; fi

module load cuda/12.6.0
unset ROCR_VISIBLE_DEVICES
source /path/to/scratch/morph_env/bin/activate

mkdir -p "$OUTPUT_BASE"

# Run flag-OFF and flag-ON on the same NUM_SAMPLES (random_seed pinned), so
# inference_results_*.json from the two runs cover the identical sample subset.
for MODE in baseline mask; do
    OUTPUT_FILE="${OUTPUT_BASE}/results_${MODE}.json"
    GENERATED_IMAGES_DIR="${OUTPUT_BASE}/images_${MODE}"
    MASK_FLAG=""
    [[ "$MODE" == "mask" ]] && MASK_FLAG="--inference_bridge_mask"

    echo "=== Smoke test: ${MODE} -> ${OUTPUT_FILE} ==="
    CUDA_VISIBLE_DEVICES=0 python3 "${SCRIPT_DIR}/inference/run_inference_bagel_spatial.py" \
        --model_path "$MODEL_PATH" \
        --data_dir "$DATA_DIR" \
        --dataset_files $DATASET_FILES \
        --output_file "$OUTPUT_FILE" \
        --generated_images_dir "$GENERATED_IMAGES_DIR" \
        --num_samples "$NUM_SAMPLES" \
        --random_seed "$SEED" \
        --thinking_mode "visual_only_thinking" \
        --vit_min_size 512 \
        --max_rounds 3 \
        --image_shapes 720 1024 \
        --no_think \
        --no_resume \
        $MASK_FLAG
done

echo "=== Smoke test complete. Compare:"
echo "  baseline: ${OUTPUT_BASE}/results_baseline.json"
echo "  mask:     ${OUTPUT_BASE}/results_mask.json"
echo "Expected: same set of sample_ids, but answer text differs on at least some."
