# Use RunPod's PyTorch base image with CUDA support
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Set working directory
WORKDIR /workspace/semantic_id_recommender

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to PATH
ENV PATH="/root/.local/bin:$PATH"

# Optimization: Copy only dependency files first (for better caching)
COPY pyproject.toml uv.lock* ./

# Create venv and install dependencies
RUN uv sync

# Copy the rest of the source code
COPY . .

# Create data directory (will be mounted or populated at runtime)
RUN mkdir -p data checkpoints

# Set up shell to activate venv by default
RUN echo 'source /workspace/semantic_id_recommender/.venv/bin/activate' >> /root/.bashrc

# Default command
CMD ["/bin/bash"]
