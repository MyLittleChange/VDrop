#!/bin/bash
# Sync a LoRA adapter from tamia, merge it into a full BAGEL checkpoint,
# and generate per-benchmark eval SLURM scripts pointing at the merged model.
#
# Usage:
#   bash tools/sync_merge_gen_eval.sh \
#     --lora-subpath training_data_mix_anchor_counting_distance_balance_visual_only_lora/0006000 \
#     --new-tag mix_anchor_counting_distance_balance_visual_only_lora_6k
#
# Flags:
#   --lora-subpath <name>/<step>     (required) e.g. training_data_foo_lora/0006000
#   --new-tag <tag>                  (required) tag used as BAGEL_format_<tag> and in generated filenames
#   --base-model-path <path>         default: /path/to/scratch/models/BAGEL-7B-MoT
#   --lora-rank <int>                default: 32
#   --lora-alpha <float>             default: 64
#   --skip-scp                       skip step 1 (sync) if already present
#   --skip-merge                     skip step 2 (merge) if already present
#   --skip-gen                       skip step 3 (eval-script generation)
#   --force                          overwrite existing generated eval scripts
#   --dry-run                        print actions without executing

set -euo pipefail

REPO_ROOT="/path/to/ThinkMorph-BAGEL-release"
TAMIA_HOST="USER@REMOTE_HOST"
TAMIA_ROOT="/path/to/scratch/VisualCoT/BAGEL_checkpoints"
MILA_ROOT="/path/to/scratch/VisualCoT/BAGEL_checkpoints"
BASE_MODEL_PATH="/path/to/scratch/models/BAGEL-7B-MoT"
LORA_RANK=32
LORA_ALPHA=64

LORA_SUBPATH=""
NEW_TAG=""
SKIP_SCP=0
SKIP_MERGE=0
SKIP_GEN=0
FORCE=0
DRY_RUN=0

# Each entry: "<template_relative_path>|<tag_in_template>"
TEMPLATES=(
    "SpatialUnderstanding/eval/mmsi/eval_mmsi_bench_2gpu_mix_all_rotation_balance_visual_only.sh|mix_all_rotation_balance_visual_only"
    "SpatialUnderstanding/eval/mindcube/run_bagel_mindcube_1gpu_mix_all_balance_no_think_lora_10k.sh|mix_all_balance_no_think_lora_10k"
    "SpatialUnderstanding/eval/omnispatial/run_bagel_omnispatial_1gpu_mix_anchor_counting_distance_balance_no_think_lora_lora_10k.sh|mix_anchor_counting_distance_balance_no_think_lora_lora_10k"
    "SpatialUnderstanding/eval/stare/run_bagel_stare_perspective_1gpu_mix_all_rotation_no_thinking.sh|mix_all_rotation_no_thinking"
    "SpatialUnderstanding/eval/spatial/run_bagel_spatial_4gpu_anchor_mix_all_rotation_visual_only.sh|mix_all_rotation_visual_only"
    "SpatialUnderstanding/eval/spatial/run_bagel_spatial_4gpu_counting_mix_all_rotation_visual_only.sh|mix_all_rotation_visual_only"
    "SpatialUnderstanding/eval/spatial/run_bagel_spatial_4gpu_direction_mix_all_rotation_visual_only.sh|mix_all_rotation_visual_only"
    "SpatialUnderstanding/eval/spatial/run_bagel_spatial_4gpu_distance_mix_all_rotation_visual_only.sh|mix_all_rotation_visual_only"
    "SpatialUnderstanding/eval/spatial/run_bagel_spatial_map_mix_all_rotation_balance_visual_only.sh|mix_all_rotation_balance_visual_only"
)

usage() {
    sed -n '2,22p' "$0"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --lora-subpath)     LORA_SUBPATH="$2"; shift 2 ;;
        --new-tag)          NEW_TAG="$2"; shift 2 ;;
        --base-model-path)  BASE_MODEL_PATH="$2"; shift 2 ;;
        --lora-rank)        LORA_RANK="$2"; shift 2 ;;
        --lora-alpha)       LORA_ALPHA="$2"; shift 2 ;;
        --skip-scp)         SKIP_SCP=1; shift ;;
        --skip-merge)       SKIP_MERGE=1; shift ;;
        --skip-gen)         SKIP_GEN=1; shift ;;
        --force)            FORCE=1; shift ;;
        --dry-run)          DRY_RUN=1; shift ;;
        -h|--help)          usage ;;
        *) echo "Unknown flag: $1"; usage ;;
    esac
done

if [[ -z "$LORA_SUBPATH" || -z "$NEW_TAG" ]]; then
    echo "ERROR: --lora-subpath and --new-tag are required"
    usage
fi

TAMIA_SRC="${TAMIA_HOST}:${TAMIA_ROOT}/${LORA_SUBPATH}/lora_adapter.safetensors"
MILA_DST_DIR="${MILA_ROOT}/${LORA_SUBPATH}"
MILA_DST="${MILA_DST_DIR}/lora_adapter.safetensors"
MERGED_DIR="${MILA_ROOT}/BAGEL_format_${NEW_TAG}"

run() {
    echo "+ $*"
    if [[ "$DRY_RUN" -eq 0 ]]; then
        "$@"
    fi
}

echo "=============================================="
echo "Sync + Merge + Gen Eval"
echo "=============================================="
echo "  LoRA subpath:     $LORA_SUBPATH"
echo "  New tag:          $NEW_TAG"
echo "  Base model:       $BASE_MODEL_PATH"
echo "  Rank / Alpha:     $LORA_RANK / $LORA_ALPHA"
echo "  Tamia source:     $TAMIA_SRC"
echo "  Mila destination: $MILA_DST"
echo "  Merged output:    $MERGED_DIR"
echo "  Dry run:          $DRY_RUN"
echo "=============================================="
echo ""

# ---------- Step 1: SCP ----------
if [[ "$SKIP_SCP" -eq 1 ]]; then
    echo "[1/3] SCP: skipped"
elif [[ -f "$MILA_DST" && "$FORCE" -eq 0 ]]; then
    echo "[1/3] SCP: destination already exists, skipping ($MILA_DST)"
else
    echo "[1/3] SCP from tamia..."
    run mkdir -p "$MILA_DST_DIR"
    run scp "$TAMIA_SRC" "$MILA_DST"
fi
echo ""

# ---------- Step 2: Merge LoRA ----------
if [[ "$SKIP_MERGE" -eq 1 ]]; then
    echo "[2/3] Merge: skipped"
elif [[ -f "$MERGED_DIR/model.safetensors" && "$FORCE" -eq 0 ]]; then
    echo "[2/3] Merge: output already exists, skipping ($MERGED_DIR/model.safetensors)"
else
    echo "[2/3] Merging LoRA into base..."
    run python "$REPO_ROOT/tools/merge_lora.py" \
        --base_model_path "$BASE_MODEL_PATH" \
        --lora_adapter_path "$MILA_DST_DIR" \
        --output_path "$MERGED_DIR" \
        --lora_rank "$LORA_RANK" \
        --lora_alpha "$LORA_ALPHA"
fi
echo ""

# ---------- Step 3: Generate eval scripts ----------
if [[ "$SKIP_GEN" -eq 1 ]]; then
    echo "[3/3] Eval gen: skipped"
    exit 0
fi

echo "[3/3] Generating eval scripts..."
GEN_COUNT=0
SKIP_COUNT=0
for entry in "${TEMPLATES[@]}"; do
    TEMPLATE_REL="${entry%%|*}"
    OLD_TAG="${entry##*|}"
    TEMPLATE="$REPO_ROOT/$TEMPLATE_REL"

    if [[ ! -f "$TEMPLATE" ]]; then
        echo "  WARN: template not found, skipping: $TEMPLATE"
        continue
    fi

    TEMPLATE_DIR="$(dirname "$TEMPLATE")"
    TEMPLATE_BASE="$(basename "$TEMPLATE")"
    NEW_BASE="${TEMPLATE_BASE//${OLD_TAG}/${NEW_TAG}}"
    NEW_PATH="$TEMPLATE_DIR/$NEW_BASE"

    if [[ "$NEW_BASE" == "$TEMPLATE_BASE" ]]; then
        echo "  WARN: old tag '$OLD_TAG' not in filename $TEMPLATE_BASE — skipping"
        continue
    fi

    if [[ -f "$NEW_PATH" && "$FORCE" -eq 0 ]]; then
        echo "  skip (exists): $NEW_PATH"
        SKIP_COUNT=$((SKIP_COUNT + 1))
        continue
    fi

    echo "  $TEMPLATE_REL"
    echo "    -> $NEW_PATH  (tag: $OLD_TAG -> $NEW_TAG)"
    if [[ "$DRY_RUN" -eq 0 ]]; then
        sed "s|${OLD_TAG}|${NEW_TAG}|g" "$TEMPLATE" > "$NEW_PATH"
        chmod +x "$NEW_PATH"
    fi
    GEN_COUNT=$((GEN_COUNT + 1))
done

echo ""
echo "=============================================="
echo "Done. Generated: $GEN_COUNT, skipped: $SKIP_COUNT"
echo "=============================================="
