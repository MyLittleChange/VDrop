#!/usr/bin/env bash
# tools/onboard_model.sh
# ─────────────────────────────────────────────────────────────────────────────
# Onboard a new LoRA checkpoint: SCP → merge → submit full eval suite.
#
# Usage:
#   bash tools/onboard_model.sh \
#     --ckpt  <checkpoint_name>     # e.g. training_data_mix_balance_matterport_point_matching_no_think_lora
#     --step  <step_num>            # e.g. 0009000
#     --tag   <short_id>            # ≤15 chars, used in SLURM job names & log filenames
#     [--thinking_mode no_thinking] # or visual_only_thinking
#     [--think false]               # or true
#     [--rank 32]
#     [--alpha 64.0]
#     [--skip_scp]
#     [--skip_merge]
#     [--benchmarks all]            # comma-separated: spatial,map,mmsi,mindcube,omnispatial,stare,blink
#     [--dry_run]
#
# Example:
#   bash tools/onboard_model.sh \
#     --ckpt training_data_mix_balance_matterport_point_matching_no_think_lora \
#     --step 0009000 \
#     --tag  mat_pm_nt
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Constants ────────────────────────────────────────────────────────────────
REPO=/path/to/ThinkMorph-BAGEL-release
SCRATCH=/path/to/scratch
CKPT_BASE="${SCRATCH}/VisualCoT/BAGEL_checkpoints"
BASE_MODEL=/path/to/scratch/models/BAGEL-7B-MoT
MORPH_ENV=/path/to/scratch/morph_env/bin/activate
TAMIA=USER@REMOTE_HOST
SBATCH=/opt/slurm/bin/sbatch
LOG_BASE="${REPO}/SpatialUnderstanding/slurm_logs"
EVAL_DIR="${REPO}/SpatialUnderstanding/eval"

# ── Defaults ─────────────────────────────────────────────────────────────────
THINKING_MODE=no_thinking
THINK=false
RANK=32
ALPHA=64.0
SKIP_SCP=false
SKIP_MERGE=false
DRY_RUN=false
BENCHMARKS=all
CKPT=""
STEP=""
TAG=""

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ckpt)          CKPT="$2";          shift 2 ;;
        --step)          STEP="$2";          shift 2 ;;
        --tag)           TAG="$2";           shift 2 ;;
        --thinking_mode) THINKING_MODE="$2"; shift 2 ;;
        --think)         THINK="$2";         shift 2 ;;
        --rank)          RANK="$2";          shift 2 ;;
        --alpha)         ALPHA="$2";         shift 2 ;;
        --skip_scp)      SKIP_SCP=true;      shift   ;;
        --skip_merge)    SKIP_MERGE=true;    shift   ;;
        --dry_run)       DRY_RUN=true;       shift   ;;
        --benchmarks)    BENCHMARKS="$2";    shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

[[ -z "$CKPT" ]] && { echo "ERROR: --ckpt is required"; exit 1; }
[[ -z "$STEP" ]] && { echo "ERROR: --step is required"; exit 1; }
[[ -z "$TAG"  ]] && { echo "ERROR: --tag is required (short id ≤15 chars for SLURM names)"; exit 1; }

# ── Derived paths ─────────────────────────────────────────────────────────────
MERGED_PATH="${CKPT_BASE}/BAGEL_format_${CKPT}"
LOCAL_LORA="${CKPT_BASE}/${CKPT}/${STEP}/lora_adapter.safetensors"
TAMIA_LORA="${TAMIA}:/path/to/scratch/VisualCoT/BAGEL_checkpoints/${CKPT}/${STEP}/lora_adapter.safetensors"

# Output dirs matching existing naming conventions
DATA_DIR_SPATIAL="${SCRATCH}/VisualCoT/spatial_collab_dataset"

# ── Helpers ───────────────────────────────────────────────────────────────────
run() {
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  [DRY_RUN] $*"
    else
        "$@"
    fi
}

should_run() {
    [[ "$BENCHMARKS" == "all" ]] || [[ ",$BENCHMARKS," == *",$1,"* ]]
}

# Submit a SLURM job, passing MODEL_PATH/OUTPUT_DIR/THINK/THINKING_MODE via env.
# Usage: submit_job JOB_NAME BENCHMARK SCRIPT [extra sbatch flags...]
submit_job() {
    local job_name="$1"; shift
    local benchmark="$1"; shift
    local script="$1"; shift
    # remaining are extra sbatch flags (e.g. --array=0-11)

    mkdir -p "${LOG_BASE}/${benchmark}"
    local log_out="${LOG_BASE}/${benchmark}/${job_name}_output.txt"
    local log_err="${LOG_BASE}/${benchmark}/${job_name}_error.txt"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  [DRY_RUN] $SBATCH --job-name=${job_name} --output=${log_out} $* ${script}"
    else
        $SBATCH \
            --job-name="${job_name}" \
            --output="${log_out}" \
            --error="${log_err}" \
            --export=ALL \
            "$@" \
            "$script"
    fi
}

# ── Banner ────────────────────────────────────────────────────────────────────
echo "============================================================"
echo "  ThinkMorph — Onboard New Model"
echo "  Checkpoint : BAGEL_format_${CKPT}"
echo "  Step       : ${STEP}"
echo "  Tag        : ${TAG}"
echo "  Think mode : ${THINKING_MODE}  (think=${THINK})"
echo "  LoRA       : r=${RANK}, alpha=${ALPHA}"
echo "  Benchmarks : ${BENCHMARKS}"
[[ "$DRY_RUN" == "true" ]] && echo "  *** DRY RUN — no changes made ***"
echo "============================================================"

# ════════════════════════════════════════════════════════════════
# Step 1 — SCP lora_adapter.safetensors from Tamia
# ════════════════════════════════════════════════════════════════
if [[ "$SKIP_SCP" == "false" ]]; then
    echo ""
    echo "[1/3] SCP lora_adapter.safetensors from Tamia..."
    mkdir -p "$(dirname "$LOCAL_LORA")"
    run scp "$TAMIA_LORA" "$LOCAL_LORA"
    echo "  -> ${LOCAL_LORA}"
else
    echo "[1/3] SCP skipped (--skip_scp)"
fi

# ════════════════════════════════════════════════════════════════
# Step 2 — Merge LoRA into base model
# ════════════════════════════════════════════════════════════════
if [[ "$SKIP_MERGE" == "false" ]]; then
    echo ""
    echo "[2/3] Merging LoRA adapter into base model..."
    # shellcheck disable=SC1090
    source "$MORPH_ENV"
    run python "${REPO}/tools/merge_lora.py" \
        --base_model_path "$BASE_MODEL" \
        --lora_adapter_path "${CKPT_BASE}/${CKPT}/${STEP}" \
        --output_path "$MERGED_PATH" \
        --lora_rank  "$RANK" \
        --lora_alpha "$ALPHA"
    echo "  -> ${MERGED_PATH}/model.safetensors"
else
    echo "[2/3] Merge skipped (--skip_merge)"
fi

# ════════════════════════════════════════════════════════════════
# Step 3 — Submit eval jobs
# ════════════════════════════════════════════════════════════════
echo ""
echo "[3/3] Submitting eval jobs..."

# Export common env vars; each benchmark additionally sets OUTPUT_DIR and TASK
export MODEL_PATH="${MERGED_PATH}"
export THINK THINKING_MODE

# ── COSMIC spatial tasks (anchor / counting / distance / direction) ───────────
if should_run "spatial"; then
    TASK_MAP="anchor:anchor counting:counting distance:relative_distance direction:spatial"
    for entry in $TASK_MAP; do
        TASK="${entry%%:*}"
        DATASET_SUFFIX="${entry##*:}"
        export TASK
        export OUTPUT_DIR="${CKPT_BASE}/BAGEL_format_${CKPT}_mcqs_${DATASET_SUFFIX}_normalized"
        JOB="${TASK:0:4}_${TAG}"
        echo "  spatial/${TASK}  -> ${JOB}"
        submit_job "$JOB" "spatial" \
            "${EVAL_DIR}/spatial/run_bagel_spatial_generic.sh"
    done
fi

# ── COSMIC map (OOD-Task) ─────────────────────────────────────────────────────
if should_run "map"; then
    export OUTPUT_DIR="${CKPT_BASE}/BAGEL_format_${CKPT}_map_questions_normalized"
    JOB="map_${TAG}"
    echo "  map              -> ${JOB}"
    submit_job "$JOB" "spatial" \
        "${EVAL_DIR}/spatial/run_bagel_spatial_map_generic.sh"
fi

# ── MMSI-Bench (single 2-GPU job, 2 shards) ──────────────────────────────────
if should_run "mmsi"; then
    export OUTPUT_DIR="${SCRATCH}/VisualCoT/BAGEL_format_${CKPT}_mmsi_eval"
    JOB="mmsi_${TAG}"
    echo "  mmsi             -> ${JOB}"
    submit_job "$JOB" "mmsi" \
        "${EVAL_DIR}/mmsi/eval_mmsi_bench_generic.sh"
fi

# ── MindCube ──────────────────────────────────────────────────────────────────
if should_run "mindcube"; then
    export OUTPUT_DIR="${SCRATCH}/mindcube/BAGEL_format_${CKPT}"
    JOB="mc_${TAG}"
    echo "  mindcube         -> ${JOB}"
    submit_job "$JOB" "mindcube" \
        "${EVAL_DIR}/mindcube/run_bagel_mindcube_generic.sh"
fi

# ── OmniSpatial ───────────────────────────────────────────────────────────────
if should_run "omnispatial"; then
    export OUTPUT_DIR="${SCRATCH}/VisualCoT/omnispatial/BAGEL_format_${CKPT}"
    JOB="omni_${TAG}"
    echo "  omnispatial      -> ${JOB}"
    submit_job "$JOB" "omnispatial" \
        "${EVAL_DIR}/omnispatial/run_bagel_omnispatial_generic.sh"
fi

# ── STARE Perspective ─────────────────────────────────────────────────────────
if should_run "stare"; then
    export OUTPUT_DIR="${SCRATCH}/VisualCoT/stare_perspective/BAGEL_format_${CKPT}"
    JOB="stare_${TAG}"
    echo "  stare            -> ${JOB}"
    submit_job "$JOB" "stare" \
        "${EVAL_DIR}/stare/run_bagel_stare_perspective_generic.sh"
fi

# ── BLINK MultiView ───────────────────────────────────────────────────────────
if should_run "blink"; then
    export OUTPUT_DIR="${SCRATCH}/blink/BAGEL_format_${CKPT}"
    JOB="blink_${TAG}"
    echo "  blink            -> ${JOB}"
    submit_job "$JOB" "blink" \
        "${EVAL_DIR}/blink/run_bagel_blink_multiview_generic.sh"
fi

echo ""
echo "============================================================"
echo "  Done!  Logs -> ${LOG_BASE}/"
echo "  Merged model -> ${MERGED_PATH}/"
echo "============================================================"
