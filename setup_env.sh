#!/bin/bash
# ThinkMorph Environment Setup
# Usage: bash setup_env.sh
#
# Prerequisites: cluster.env must exist (copy from cluster.env.example)
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Load cluster config ─────────────────────────────────────────────────────
if [ ! -f "$REPO_ROOT/cluster.env" ]; then
    echo "ERROR: cluster.env not found."
    echo "  cp cluster.env.example cluster.env"
    echo "  # then edit cluster.env with your cluster-specific paths"
    exit 1
fi
source "$REPO_ROOT/cluster.env"

echo "=== ThinkMorph Environment Setup ==="
echo "Cluster:    ${CLUSTER_NAME}"
echo "Venv path:  ${VENV_PATH}"
echo ""

# ── Install uv if needed ────────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    echo "uv installed: $(uv --version)"
else
    echo "uv found: $(uv --version)"
fi

# ── Load CUDA module if specified ────────────────────────────────────────────
if [ -n "$CUDA_MODULE" ]; then
    if command -v module &>/dev/null; then
        echo "Loading CUDA module: $CUDA_MODULE"
        module load "$CUDA_MODULE"
    else
        echo "WARNING: 'module' command not available, skipping CUDA module load."
        echo "  Make sure CUDA is already in your PATH."
    fi
fi

# ── Create venv ──────────────────────────────────────────────────────────────
VENV_DIR="${VENV_PATH%/bin/activate}"
if [ -f "$VENV_PATH" ]; then
    echo "Venv already exists at: $VENV_DIR"
else
    echo "Creating venv at: $VENV_DIR"
    uv venv "$VENV_DIR" --python 3.10
fi
source "$VENV_PATH"
echo "Python: $(python --version) at $(which python)"

# ── Install PyTorch with CUDA ────────────────────────────────────────────────
echo ""
echo "Installing PyTorch (CUDA 12.6)..."
uv pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu126

# ── Install project dependencies ─────────────────────────────────────────────
echo ""
echo "Installing project dependencies..."
uv pip install -r "$REPO_ROOT/requirements.txt"

# ── Try installing flash-attn (optional, may fail without build tools) ───────
echo ""
echo "Attempting flash-attn install (optional)..."
if uv pip install flash-attn --no-build-isolation 2>/dev/null; then
    echo "flash-attn installed successfully."
else
    echo "WARNING: flash-attn install failed. You can install it later with:"
    echo "  uv pip install flash-attn --no-build-isolation"
fi

# ── Create directory structure ───────────────────────────────────────────────
echo ""
echo "Creating directory structure..."
mkdir -p "$DATA_ROOT" "$OUTPUT_ROOT" "$CKPT_ROOT" "$WANDB_DIR" "$WANDB_CACHE_DIR"

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "=== Setup Complete ==="
echo "  Cluster:      ${CLUSTER_NAME}"
echo "  Venv:         ${VENV_DIR}"
echo "  Data root:    ${DATA_ROOT}"
echo "  Output root:  ${OUTPUT_ROOT}"
echo "  Ckpt root:    ${CKPT_ROOT}"
echo "  Model root:   ${MODEL_ROOT}"
echo ""
echo "Next steps:"
echo "  1. Copy/symlink training data to ${DATA_ROOT}"
echo "  2. Download model weights to ${BAGEL_MODEL_PATH}"
echo "  3. Run training: sbatch configs/train/<script>.sh"
