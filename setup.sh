#!/bin/bash
# RunPod setup script for semantic-id-recommender

set -e

echo "=== Setting up RunPod instance ==="

# Use fast local storage for caches
export HF_HOME=/root/.cache/huggingface
export TORCH_HOME=/root/.cache/torch
export UV_CACHE_DIR=/root/.cache/uv
export PIP_CACHE_DIR=/root/.cache/pip

mkdir -p $HF_HOME $TORCH_HOME $UV_CACHE_DIR $PIP_CACHE_DIR

# Install system packages
echo "=== Installing system packages ==="
apt-get update && apt-get install -y tmux htop nvtop

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "=== Installing uv ==="
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env
fi

# Setup project directory and venv
echo "=== Setting up Python environment ==="
PROJECT_DIR=/workspace/semantic_id_recommender
cd $PROJECT_DIR

if [ ! -d ".venv" ]; then
    uv venv .venv --python 3.12
fi

source .venv/bin/activate

# Install dependencies
echo "=== Installing project dependencies ==="
uv pip install -e .
uv pip install flash-attn --no-build-isolation

# Persist environment variables
echo "=== Configuring shell ==="
cat >> ~/.bashrc << 'EOF'

# Semantic ID Recommender setup
export HF_HOME=/root/.cache/huggingface
export TORCH_HOME=/root/.cache/torch
export UV_CACHE_DIR=/root/.cache/uv
export PIP_CACHE_DIR=/root/.cache/pip
cd /workspace/semantic_id_recommender && source .venv/bin/activate
EOF

# Verify installation
echo "=== Verifying installation ==="
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import unsloth; print('Unsloth: OK')"

echo "=== Setup complete ==="
echo "Run 'source ~/.bashrc' or start a new shell to activate the environment."
