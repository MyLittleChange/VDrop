#!/bin/bash
# Shared preamble for all ThinkMorph training scripts.
# Source this at the top of every training script:
#   source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

set -e

# ── Resolve repo root and load cluster config ───────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [ ! -f "$REPO_ROOT/cluster.env" ]; then
    echo "ERROR: cluster.env not found at $REPO_ROOT/cluster.env"
    echo "  cp cluster.env.example cluster.env"
    echo "  # then edit cluster.env with your cluster-specific paths"
    exit 1
fi
source "$REPO_ROOT/cluster.env"

# ── Use SCRIPT_DIR from cluster.env, fallback to REPO_ROOT ──────────────────
SCRIPT_DIR="${SCRIPT_DIR:-$REPO_ROOT}"

# ── Load CUDA module ────────────────────────────────────────────────────────
if [ -n "$CUDA_MODULE" ]; then
    module load "$CUDA_MODULE"
fi
unset ROCR_VISIBLE_DEVICES

# ── Activate Python environment ──────────────────────────────────────────────
source "$VENV_PATH"

# ── Export environment variables ─────────────────────────────────────────────
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR
export WANDB_CACHE_DIR
export SCRATCH_ROOT
export DATA_ROOT
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

if [ -n "$PYTORCH_CUDA_ALLOC_CONF" ]; then
    export PYTORCH_CUDA_ALLOC_CONF
fi

# ── Helper functions ─────────────────────────────────────────────────────────
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}
