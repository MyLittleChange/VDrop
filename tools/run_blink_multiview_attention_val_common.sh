#!/usr/bin/env bash

set -euo pipefail

# Shared implementation for BLINK Multi-view_Reasoning val-split
# answer-token attention Slurm launchers. The wrapper script sets
# GPUS_PER_JOB and the Slurm partition/resources.

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-/path/to/ThinkMorph-BAGEL-release}}"
if [[ ! -f "$REPO_ROOT/tools/answer_attention_probe_batch.py" ]]; then
  REPO_ROOT="/path/to/ThinkMorph-BAGEL-release"
fi
cd "$REPO_ROOT"

PYTHON="${PYTHON:-/path/to/scratch/morph_env/bin/python}"
DATA_DIR="${DATA_DIR:-/path/to/scratch/datasets/BLINK}"
TASK="${TASK:-Multi-view_Reasoning}"
SPLIT="${SPLIT:-val}"

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/path/to/scratch/VisualCoT/BAGEL_checkpoints}"
VANILLA_MODEL_PATH="${VANILLA_MODEL_PATH:-/path/to/scratch/models/BAGEL-7B-MoT}"
VISUAL_MODEL_PATH="${VISUAL_MODEL_PATH:-$CHECKPOINT_ROOT/BAGEL_format_training_data_mix_all_balance_visual_only_lora_7k}"
BRIDGE_MASK_MODEL_PATH="${BRIDGE_MASK_MODEL_PATH:-$CHECKPOINT_ROOT/BAGEL_format_mix_all_balance_visual_only_bridge_masked_lora_7k}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/path/to/scratch/VisualCoT/artifacts/blink_multiview_attention_val133}"
SAMPLE_COUNT="${SAMPLE_COUNT:-133}"
MANIFEST="${MANIFEST:-$OUTPUT_ROOT/fixed_blink_multiview_${SPLIT}_n${SAMPLE_COUNT}.json}"

GPUS_PER_JOB="${GPUS_PER_JOB:-1}"
TOTAL_SHARDS="${TOTAL_SHARDS:-$GPUS_PER_JOB}"

# BAGEL-7B-MoT text decoder has 28 layers, indexed 0..27.
LAYERS="${LAYERS:-0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27}"
MAX_MEM_PER_GPU="${MAX_MEM_PER_GPU:-70GiB}"
NUM_TIMESTEPS="${NUM_TIMESTEPS:-2}"
MAX_THINK_TOKEN_N="${MAX_THINK_TOKEN_N:-64}"
MAX_ANSWER_TOKEN_N="${MAX_ANSWER_TOKEN_N:-32}"
VIT_MIN_SIZE="${VIT_MIN_SIZE:-512}"
TEXT_TEMPERATURE="${TEXT_TEMPERATURE:-0.3}"
DO_SAMPLE="${DO_SAMPLE:-0}"

RUN_PROBES="${RUN_PROBES:-1}"
FORCE_RERUN="${FORCE_RERUN:-0}"
RESAMPLE="${RESAMPLE:-0}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
SAVE_RAW_QK="${SAVE_RAW_QK:-0}"
SAVE_VISUALIZATIONS="${SAVE_VISUALIZATIONS:-0}"
RUN_SET="${RUN_SET:-trained_pair}"
AGGREGATE_RUN_NAMES="${AGGREGATE_RUN_NAMES:-vanilla_force_bridge visual_only_lora_7k bridge_mask_lora_7k}"

case "$RUN_SET" in
  trained_pair)
    RUN_NAMES=("visual_only_lora_7k" "bridge_mask_lora_7k")
    MODEL_PATHS=("$VISUAL_MODEL_PATH" "$BRIDGE_MASK_MODEL_PATH")
    NO_THINK_FLAGS=("1" "1")
    ;;
  vanilla_force_bridge)
    RUN_NAMES=("vanilla_force_bridge")
    MODEL_PATHS=("$VANILLA_MODEL_PATH")
    NO_THINK_FLAGS=("0")
    ;;
  all)
    RUN_NAMES=("vanilla_force_bridge" "visual_only_lora_7k" "bridge_mask_lora_7k")
    MODEL_PATHS=("$VANILLA_MODEL_PATH" "$VISUAL_MODEL_PATH" "$BRIDGE_MASK_MODEL_PATH")
    NO_THINK_FLAGS=("0" "1" "1")
    ;;
  *)
    echo "ERROR: unknown RUN_SET=$RUN_SET; use trained_pair, vanilla_force_bridge, or all" >&2
    exit 1
    ;;
esac
read -r -a AGGREGATE_RUN_ARGS <<< "$AGGREGATE_RUN_NAMES"

mkdir -p "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/runs"

if [[ "$RESAMPLE" == "1" ]]; then
  rm -f "$MANIFEST"
fi

if [[ ! -f "$MANIFEST" ]]; then
  echo "Creating BLINK fixed manifest: $MANIFEST"
  "$PYTHON" "$REPO_ROOT/tools/create_blink_multiview_manifest.py" \
    --data_dir "$DATA_DIR" \
    --task "$TASK" \
    --split "$SPLIT" \
    --sample_count "$SAMPLE_COUNT" \
    --output "$MANIFEST"
else
  echo "Reusing BLINK fixed manifest: $MANIFEST"
fi

read -r -a LAYER_ARGS <<< "$LAYERS"

run_shard() {
  local run_name="$1"
  local model_path="$2"
  local no_think="$3"
  local shard_idx="$4"
  local gpu_id="$5"
  local shard_spec="${shard_idx}/${TOTAL_SHARDS}"
  local log_path="$OUTPUT_ROOT/logs/${run_name}_shard${shard_idx}of${TOTAL_SHARDS}.log"

  local cmd=(
    "$PYTHON" "$REPO_ROOT/tools/answer_attention_probe_batch.py"
    --model_path "$model_path"
    --dataset_kind blink_multiview
    --data_dir "$DATA_DIR"
    --blink_task "$TASK"
    --blink_split "$SPLIT"
    --manifest "$MANIFEST"
    --run_name "$run_name"
    --output_root "$OUTPUT_ROOT"
    --shard "$shard_spec"
    --thinking_mode visual_only_thinking
    --bridge_mode normal
    --layers "${LAYER_ARGS[@]}"
    --max_mem_per_gpu "$MAX_MEM_PER_GPU"
    --vit_min_size "$VIT_MIN_SIZE"
    --num_timesteps "$NUM_TIMESTEPS"
    --max_think_token_n "$MAX_THINK_TOKEN_N"
    --max_answer_token_n "$MAX_ANSWER_TOKEN_N"
    --text_temperature "$TEXT_TEMPERATURE"
    --force_bridge
  )
  if [[ "$no_think" == "1" ]]; then
    cmd+=(--no_think)
  fi
  if [[ "$DO_SAMPLE" == "1" ]]; then
    cmd+=(--do_sample)
  fi
  if [[ "$FORCE_RERUN" == "1" ]]; then
    cmd+=(--force_rerun)
  fi
  if [[ "$CONTINUE_ON_ERROR" == "1" ]]; then
    cmd+=(--continue_on_error)
  fi
  if [[ "$SAVE_RAW_QK" != "1" ]]; then
    cmd+=(--skip_raw_qk)
  fi
  if [[ "$SAVE_VISUALIZATIONS" != "1" ]]; then
    cmd+=(--skip_visualizations)
  fi

  echo "Launching $run_name shard $shard_spec on GPU $gpu_id; log: $log_path"
  CUDA_VISIBLE_DEVICES="$gpu_id" "${cmd[@]}" >"$log_path" 2>&1 &
}

run_model() {
  local run_name="$1"
  local model_path="$2"
  local no_think="$3"

  if [[ ! -d "$model_path" ]]; then
    echo "ERROR: model not found for $run_name: $model_path" >&2
    return 1
  fi

  echo ""
  echo "============================================================"
  echo "Running $run_name"
  echo "Model: $model_path"
  echo "Shards: $TOTAL_SHARDS, GPUs/job: $GPUS_PER_JOB"
  echo "============================================================"

  local pids=()
  for ((shard_idx = 0; shard_idx < TOTAL_SHARDS; shard_idx++)); do
    local gpu_id=$((shard_idx % GPUS_PER_JOB))
    run_shard "$run_name" "$model_path" "$no_think" "$shard_idx" "$gpu_id"
    pids+=("$!")

    if [[ ${#pids[@]} -eq "$GPUS_PER_JOB" || $shard_idx -eq $((TOTAL_SHARDS - 1)) ]]; then
      local fail=0
      for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
          echo "ERROR: shard process $pid failed for $run_name" >&2
          fail=1
        fi
      done
      pids=()
      if [[ "$fail" != "0" && "$CONTINUE_ON_ERROR" != "1" ]]; then
        return 1
      fi
    fi
  done
}

echo "BLINK Multi-view attention probe"
echo "Output root: $OUTPUT_ROOT"
echo "Manifest: $MANIFEST"
echo "Sample count: $SAMPLE_COUNT"
echo "Run set: $RUN_SET"
echo "Runs in this job: ${RUN_NAMES[*]}"
echo "Runs in aggregate summary: ${AGGREGATE_RUN_ARGS[*]}"
echo "Layers: $LAYERS"
echo "Save visualizations: $SAVE_VISUALIZATIONS"
echo "Save raw Q/K: $SAVE_RAW_QK"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total --format=csv
fi

if [[ "$RUN_PROBES" == "1" ]]; then
  for i in "${!RUN_NAMES[@]}"; do
    run_model "${RUN_NAMES[$i]}" "${MODEL_PATHS[$i]}" "${NO_THINK_FLAGS[$i]}"
  done
fi

echo ""
echo "Aggregating attention distributions..."
"$PYTHON" "$REPO_ROOT/tools/aggregate_attention_distribution.py" \
  --output_root "$OUTPUT_ROOT" \
  --manifest "$MANIFEST" \
  --run_names "${AGGREGATE_RUN_ARGS[@]}"

echo "Done. Summary: $OUTPUT_ROOT/attention_distribution_summary.md"
