# Semantic ID Recommender - Setup & Usage Guide

This guide walks you through setting up and running the semantic ID recommender pipeline.

## Prerequisites

### Required Accounts

1. **HuggingFace Account** (free)
   - Sign up at https://huggingface.co/join
   - Create an access token at https://huggingface.co/settings/tokens
   - Token needs "Write" permission for pushing models
   - **Create private repositories for your models**:
     - Go to https://huggingface.co/new
     - Repository name: e.g., `semantic-rqvae` or `semantic-recommender`
     - Type: **Model**
     - Visibility: **Private** (recommended to keep your models private)
     - Click "Create repository"
     - Note your full repo ID: `your-username/semantic-rqvae`

2. **Weights & Biases Account** (free, optional)
   - Sign up at https://wandb.ai/authorize
   - Used for training logging

3. **Modal Account** (free tier available)
   - Sign up at https://modal.com/signup
   - $30 free credits for new accounts


## Running on RunPod

This repo is designed for running on runpod.

### Step 1: Create a RunPod Account

1. Sign up at https://www.runpod.io/
2. Add credits to your account (pay-as-you-go)

### Step 2: Configure Environment Variables (Secrets)

Before launching your pod, set up your API tokens as environment variables so they're available in your pod.

1. Go to **Settings** → **Environment Variables** in RunPod
2. Add the following environment variables:

| Variable Name | Description | How to Get |
|---------------|-------------|------------|
| `HF_TOKEN` | HuggingFace API token | Get from https://huggingface.co/settings/tokens (needs "Write" permission) |
| `WANDB_API_KEY` | Weights & Biases API key (optional) | Get from https://wandb.ai/authorize |

### Step 4: Connect to Your Pod

Make sure to add your public SSH key to RunPod first (Settings → SSH Keys).

In your pod's **Connect** menu, use **SSH over exposed TCP** (not the proxy method).

### Step 5: Verify Environment Variables

After connecting to your pod, verify that your environment variables are set:

```bash
# Check if HuggingFace token is set
echo $HF_TOKEN

# Check if Weights & Biases key is set (optional)
echo $WANDB_API_KEY
```

### Step 6: Setup the Environment

On the RunPod pod, clone the repository and install dependencies:

```bash
# Clone the code repository
cd /workspace
git clone https://github.com/charleslow/semantic_id_recommender
cd semantic_id_recommender

curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.bashrc
uv sync
source .venv/bin/activate

# Install Jupyter kernel
uv pip install ipykernel
python -m ipykernel install --user --name=semantic-id-recommender --display-name="Python (semantic-id-recommender)"
```

### Step 7: Transfer Data to RunPod

From your **local machine**, transfer your data files using scp:

```bash
# Transfer data files to RunPod using direct TCP connection
# Use the POD_IP and POD_PORT from Step 4
scp -P ${POD_PORT} -i ~/.ssh/id_ed25519 \
  data/* \
  root@${POD_IP}:/workspace/semantic_id_recommender/data/

# Verify files transferred
ssh root@${POD_IP} -p ${POD_PORT} -i ~/.ssh/id_ed25519 \
  "ls -lh /workspace/semantic_id_recommender/data/"
```

**Note**: Use the **SSH over exposed TCP** connection (direct IP and port) from Step 4.


### Step 8: Run Training with tmux

**Important**: Always use `tmux` for long-running training jobs. This ensures training continues even if your SSH connection drops.

#### Start a tmux session

```bash
# Create a new tmux session named "training"
tmux new -s training

# Inside tmux, activate the environment
source .venv/bin/activate
```

#### Train RQ-VAE

```bash
# Using your config file
python -m scripts.train_rqvae --config configs/rqvae_config.yaml

# Or test with dummy data
python -m scripts.train_rqvae --config configs/rqvae_config.yaml --create-dummy --dummy-size 1000
```

#### Fine-tune LLM

```bash
# Stage 1 only (embedding training)
python -m scripts.train_llm --config configs/stage1_config.yaml --stage 1

# Stage 2 only (LoRA fine-tuning)
python -m scripts.train_llm --config configs/stage2_config.yaml --stage 2

# Both stages sequentially (recommended)
python -m scripts.train_llm --config configs/stage1_config.yaml --stage 1,2 --stage2-config configs/stage2_config.yaml
```

#### tmux Commands Reference

```bash
# Detach from tmux (training continues in background)
# Press: Ctrl+B, then D

# Reattach to running session
tmux attach -t training

# List all sessions
tmux ls

# Kill a session when done
tmux kill-session -t training

# Create a new window in the same session
# Press: Ctrl+B, then C

# Switch between windows
# Press: Ctrl+B, then window number (0, 1, 2...)
```

### Step 9: Save Models to HuggingFace Hub

After training completes, push your models to HuggingFace Hub (required for deployment).

**Note**: Your `HF_TOKEN` environment variable from Step 5 will be used automatically.

**Create Private HuggingFace Repositories First**

Option A - Using HuggingFace UI (Easiest):
```
1. Go to https://huggingface.co/new
2. Repository name: semantic-rqvae (or your preferred name)
3. Type: Model
4. Visibility: Private
5. Click "Create repository"
6. Note your repo ID: your-username/semantic-rqvae
```


**Upload Your Models**

The `HF_TOKEN` environment variable you set in Step 5 will be used automatically.

```bash
# For RQ-VAE model - use Python script or notebook
python -c "
from src.rqvae.hub import upload_to_hub
import os
upload_to_hub(
    local_dir='models/rqvae_hub',
    repo_id='your-username/semantic-rqvae',
    token=os.getenv('HF_TOKEN')
)
"
```


### Step 10: Access Jupyter (Optional)

If you're using the Jupyter template:

1. Access RunPod's Jupyter via **Connect** → **HTTP Service [Port 8888]**
2. Open your notebook
3. Click **Kernel** → **Change Kernel** → **Python (semantic-id-recommender)**

### Step 11: Monitor Training

```bash
# Training progress is shown in real-time
# Watch GPU usage:
watch -n 1 nvidia-smi

# View training logs in tmux
tmux attach -t training
```

### Step 12: Stop Your Pod

**Important**: RunPod charges per minute while pod is running!

1. Go to RunPod dashboard
2. Click **Stop** or **Terminate** pod
3. **Stop** = Keeps disk, can restart later (~$0.10/GB/month storage)
4. **Terminate** = Deletes everything, download checkpoints first!

### RunPod Tips

- **Use tmux/screen** to keep training running if disconnected:
  ```bash
  # Start tmux session
  tmux new -s training

  # Run your training
  python -m scripts.train_llm --config configs/stage1_config.yaml --stage 1,2 --stage2-config configs/stage2_config.yaml

  # Detach: Ctrl+B then D
  # Reattach later: tmux attach -t training
  ```

- **Check GPU availability**:
  ```bash
  nvidia-smi
  python -c "import torch; print(torch.cuda.is_available())"
  ```

- **Estimate costs**:
  - RTX 4090: ~$0.40/hr × 3 hours (fine-tuning) = ~$1.20
  - A100 40GB: ~$1.50/hr × 3 hours = ~$4.50
  - Storage: 50GB × $0.10/GB/month if you stop (not terminate)

- **Save money**:
  - Use **Community Cloud** for cheaper rates (less reliable)
  - Use **Secure Cloud** for guaranteed availability
  - Terminate pod immediately after training completes

---

## Using a Pre-built Docker Image (Recommended)

To avoid running `uv sync` every time you start a pod, build and push a Docker image with dependencies pre-installed.

### Step 1: Create a Docker Hub Account

1. Sign up at https://hub.docker.com/signup (free tier is fine)
2. Note your username (e.g., `yourusername`)

### Step 2: Build and Push the Image

From your local machine (or any machine with Docker):

```bash
# Clone the repo if you haven't
git clone https://github.com/charleslow/semantic_id_recommender
cd semantic_id_recommender

# Login to Docker Hub
docker login

# Build the image (replace 'yourusername' with your Docker Hub username)
docker build -t yourusername/semantic-id-recommender:latest .

# Push to Docker Hub
docker push yourusername/semantic-id-recommender:latest
```


### Step 3: Launch RunPod with Your Image

1. Go to **Pods** → **Deploy**
2. Under **Container Image**, enter: `yourusername/semantic-id-recommender:latest`
   - Or `ghcr.io/yourusername/semantic-id-recommender:latest` if using GitHub
3. Set disk space (at least 50GB) and select GPU
4. Add environment variables (`HF_TOKEN`, `WANDB_API_KEY`) as described above
5. Click **Deploy**

### Step 4: Connect and Transfer Data

When you connect to the pod, the virtual environment is automatically activated:

```bash
# Connect via SSH
ssh root@${POD_IP} -p ${POD_PORT} -i ~/.ssh/id_ed25519

# You're ready to go! Just transfer your data:
# (from your local machine)
scp -P ${POD_PORT} -i ~/.ssh/id_ed25519 \
  data/* \
  root@${POD_IP}:/workspace/semantic_id_recommender/data/

# Then run training directly - no uv sync needed!
python -m scripts.train_rqvae --config configs/rqvae_config.yaml
```

### What's Excluded from the Image

The Docker image excludes (via `.dockerignore`):
- `data/` - Your data files (transfer separately)
- `checkpoints/` - Model weights
- `.venv/` - Recreated during build
- `notebooks/` - Jupyter notebooks
- `config.yaml` - Local configuration

---

## Local Setup

### 1. Clone and Install

```bash
# Clone the repository
git clone <your-repo-url>
cd semantic_id_recommender

# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync

# Activate the environment
source .venv/bin/activate
```

### 2. Configure Environment

```bash
# Login to HuggingFace
huggingface-cli login

# Login to Weights & Biases (optional)
wandb login

# Login to Modal
modal token new
```

### 3. Prepare Your Data

Your catalogue should be a JSON or JSONL file with items containing at minimum:
- `id`: Unique identifier
- `title` or `name`: Item name
- `description` (optional): Item description

Example format:
```json
{"id": "item_001", "title": "Wireless Mouse", "description": "Ergonomic wireless mouse with USB receiver"}
{"id": "item_002", "title": "Mechanical Keyboard", "description": "RGB mechanical keyboard with blue switches"}
```

Place your data in `data/catalogue.jsonl` (or update the config).

---

## Stage 1: Train RQ-VAE

The RQ-VAE learns semantic IDs for each item in your catalogue.

### Quick Start

```bash
# Using your config file
python -m scripts.train_rqvae --config configs/rqvae_config.yaml

# Or create dummy data for testing
python -m scripts.train_rqvae --config configs/rqvae_config.yaml --create-dummy --dummy-size 1000
```

### Configuration

Create or modify your config file (e.g., `configs/rqvae_config.yaml`):

```yaml
# Data configuration
catalogue_path: "data/catalogue.jsonl"
embeddings_cache_path: "data/embeddings.pt"
catalogue_fields:
  - "title"
  - "description"
catalogue_id_field: "id"
embedding_model: "sentence-transformers/all-MiniLM-L6-v2"

# Model architecture
embedding_dim: 384           # Match embedding model output
hidden_dim: 256
codebook_size: 256           # Codes per level (256^4 = 4B possible items)
num_quantizers: 4            # Number of levels in semantic ID

# Training hyperparameters
learning_rate: 1.0e-3
max_epochs: 500
batch_size: 512
train_split: 0.9

# W&B configuration
wandb_project: "semantic-id-recommender"
wandb_run_name: null         # Auto-generates as "rqvae_{codebook_size}x{num_quantizers}"
log_wandb_artifacts: true
artifact_name: "rqvae-model"

# Output paths
model_save_path: "models/rqvae_model.pt"
```

### What Happens

1. Loads catalogue items
2. Generates embeddings using sentence-transformers
3. Trains RQ-VAE to learn discrete codes
4. Saves model and logs to W&B (if configured)

### Expected Output

```
Loaded config from: notebooks/rqvae_config.yaml
Loading catalogue from data/catalogue.jsonl...
Dataset: 10000 items, 384-dim embeddings
Train/Val split: 9000/1000
Model: 4 quantizers x 256 codes = 4,294,967,296 possible IDs
Starting training...
Training complete!

=== Evaluation Results ===
Avg perplexity: 245.32 / 256
Avg usage: 95.8%
Unique IDs: 9987 / 10000
Collision rate: 0.13%
```

---

## Stage 2: Fine-tune LLM

Fine-tune a small LLM to generate semantic IDs from user queries.

### Quick Start

```bash
# Stage 1 only (embedding training - backbone frozen)
python -m scripts.train_llm --config configs/stage1_config.yaml --stage 1

# Stage 2 only (LoRA fine-tuning)
python -m scripts.train_llm --config configs/stage2_config.yaml --stage 2

# Both stages sequentially (recommended for full training)
python -m scripts.train_llm --config configs/stage1_config.yaml --stage 1,2 --stage2-config configs/stage2_config.yaml
```

### Stage 1 Configuration

Create `configs/stage1_config.yaml`:

```yaml
# RQ-VAE Model Source (load from W&B artifact)
wandb_rqvae_artifact: "rqvae-model:latest"

# Catalogue (must match RQ-VAE training)
catalogue_path: "data/catalogue.jsonl"
catalogue_id_field: "id"
embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
embeddings_cache_path: "data/embeddings.pt"

# Query templates for training data
query_templates:
  predict_semantic_id:
    - "{title}"
    - "Find: {title}"
    - "Search for {title} {description}"
  predict_attribute:
    - "What is the {field_name} for {semantic_id}?"

field_mapping:
  title: "title"
  description: "description"

num_examples_per_item: 5
predict_semantic_id_ratio: 0.8
val_split: 0.1

# Base LLM
base_model: "HuggingFaceTB/SmolLM2-135M-Instruct"
max_seq_length: 512
load_in_4bit: false

# Stage 1: Embedding training (backbone frozen)
stage: 1

# Training hyperparameters
learning_rate: 1.0e-3
batch_size: 32
num_train_epochs: 5
warmup_ratio: 0.03

# Output
output_dir: "checkpoints/llm_stage1"
semantic_ids_output_path: "data/semantic_ids.json"

# W&B configuration
wandb_project: "semantic-id-recommender"
wandb_run_name: "llm-stage1"
log_wandb_artifacts: true

# Test queries for evaluation
recommendation_test_queries:
  - "Find a wireless mouse"
  - "Search for mechanical keyboard"
```

### Stage 2 Configuration

Create `configs/stage2_config.yaml`:

```yaml
# RQ-VAE Model Source (same as stage 1)
wandb_rqvae_artifact: "rqvae-model:latest"

# Catalogue (must match)
catalogue_path: "data/catalogue.jsonl"
catalogue_id_field: "id"
embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
embeddings_cache_path: "data/embeddings.pt"

# Same query templates as stage 1
query_templates:
  predict_semantic_id:
    - "{title}"
    - "Find: {title}"
    - "Search for {title} {description}"

# Stage 2: LoRA fine-tuning
stage: 2
wandb_stage1_artifact: "llm-stage1:latest"  # Load stage 1 from W&B

# LoRA settings
lora_r: 512
lora_alpha: 512
lora_dropout: 0.05

# Training hyperparameters
learning_rate: 2.0e-4
batch_size: 32
num_train_epochs: 5

# Output
output_dir: "checkpoints/llm_stage2"

# W&B configuration
wandb_project: "semantic-id-recommender"
wandb_run_name: "llm-stage2"
log_wandb_artifacts: true
```

### What Happens

**Stage 1 (Embedding Training)**:
1. Loads RQ-VAE from W&B artifact
2. Creates semantic ID mapping for all items
3. Adds semantic ID tokens to LLM vocabulary
4. Freezes backbone, trains only embedding layers
5. Logs model to W&B artifact

**Stage 2 (LoRA Fine-tuning)**:
1. Loads stage 1 checkpoint
2. Applies LoRA adapters to all layers
3. Fine-tunes on semantic ID prediction task
4. Saves final model

### Expected Output

```
=== Starting LLM Training - Stage 1 ===
Base model: HuggingFaceTB/SmolLM2-135M-Instruct
Mode: Embedding training (backbone frozen)
RQ-VAE source: W&B artifact 'rqvae-model:latest'
...
Stage 1 Training Complete!

Cleared GPU memory

=== Starting LLM Training - Stage 2 ===
Mode: LoRA fine-tuning
LoRA rank: 512
...
Stage 2 Training Complete!

All Training Complete!
```

### Weights & Biases Logging

When wandb is enabled, the following metrics are tracked:

#### Run Configuration (logged at initialization)
| Config Key | Description |
|------------|-------------|
| `base_model` | The HuggingFace model being fine-tuned |
| `rqvae_codebook_size` | Number of codes per quantizer level |
| `rqvae_num_quantizers` | Number of levels in semantic IDs |
| `num_items` | Total items in catalogue |
| `num_train_examples` | Number of training examples generated |

#### Training Metrics (logged every N steps)
| Metric | Description |
|--------|-------------|
| `train/loss` | Cross-entropy loss on training batch |
| `train/learning_rate` | Current learning rate (with warmup/decay) |
| `train/epoch` | Current training epoch |
| `train/global_step` | Total optimization steps taken |
| `train/grad_norm` | Gradient norm (useful for debugging instability) |

#### Inference Results (logged at end)
A wandb Table with columns:
| Column | Description |
|--------|-------------|
| `query` | The test query string |
| `rank` | Position in beam search results (1-10) |
| `semantic_id` | Generated semantic ID (e.g., `<S0_5><S1_12><S2_3>`) |
| `score` | Log probability score from beam search |
| `category` | Item category (if valid match) |
| `title` | Item title (if valid match) |
| `valid` | Whether the semantic ID maps to a real item |

View your runs at: https://wandb.ai/your-username/semantic-id-recommender

---

## Stage 3: Deploy to Modal

Deploy the fine-tuned model as a serverless endpoint.

### Step 1: Upload Model to Modal Volume

```bash
python -m scripts.deploy \
  --upload-model checkpoints/llm \
  --catalogue data/catalogue.json \
  --semantic-ids data/semantic_ids.json
```

### Step 2: Deploy the App

```bash
python -m scripts.deploy --deploy
```

### Step 3: Test the Deployment

```bash
python -m scripts.deploy --test "wireless mouse"
```

### Expected Output

```
Uploading model to Modal volume...
  Uploaded: config.json
  Uploaded: model.safetensors
  Uploaded: tokenizer.json
Upload complete!

Deploying to Modal...
✓ Created semantic-id-recommender
✓ App deployed: https://your-username--semantic-id-recommender.modal.run
```

### Get Your API URL

After deployment, Modal will provide URLs:
- `https://your-username--semantic-id-recommender-recommend-api.modal.run`
- `https://your-username--semantic-id-recommender-health.modal.run`

---

## Stage 4: Run Frontend

### Local Frontend (with API)

```bash
python -m src.frontend.app \
  --api-url "https://your-username--semantic-id-recommender-recommend-api.modal.run" \
  --catalogue data/catalogue.json \
  --semantic-ids data/semantic_ids.json
```

### Local Frontend (with Local Model)

```bash
python -m src.frontend.app \
  --model-path checkpoints/llm \
  --catalogue data/catalogue.json \
  --semantic-ids data/semantic_ids.json
```

### Access the UI

Open http://localhost:7860 in your browser.

### Create Public Link

```bash
python -m src.frontend.app --api-url "..." --share
```

---

## Testing in Colab

For testing without local GPU, use the provided Colab notebook.

### Option 1: Quick Test

1. Open `notebooks/test_training.ipynb` in Google Colab
2. Select GPU runtime (Runtime → Change runtime type → T4 GPU)
3. Run all cells

### Option 2: Manual Test

```python
# In Colab, run:
!git clone <your-repo>
%cd semantic_id_recommender
!pip install -e ".[dev]"

# Test RQ-VAE
!python -m scripts.train_rqvae --config configs/rqvae_config.yaml --create-dummy --dummy-size 100

# Test LLM (requires GPU)
!python -m scripts.train_llm --config configs/stage1_config.yaml --stage 1
```

---

## Troubleshooting

### Out of Memory (LLM Fine-tuning)

Reduce batch size in your config:
```yaml
batch_size: 2
gradient_accumulation_steps: 8
```

Or use a smaller model:
```yaml
base_model: "HuggingFaceTB/SmolLM2-135M-Instruct"
```

### Slow Embedding Generation

Use GPU for embeddings - the script auto-detects GPU, but ensure CUDA is available.

### Modal Deployment Fails

```bash
# Check Modal status
modal status

# View logs
modal logs semantic-id-recommender

# Redeploy
modal deploy src/inference/modal_app.py
```

### Cold Start Too Slow

1. Increase `container_idle_timeout` in config
2. Use Modal's keep-warm feature:
```python
@app.cls(..., keep_warm=1)  # Keep 1 container warm
```

### HuggingFace Rate Limits

```bash
# Use token authentication
export HF_TOKEN="your-token"
```

### Training Interrupted / Connection Lost

If you used tmux, simply reattach:
```bash
tmux attach -t training
```

If not using tmux and training was interrupted, restart from the last checkpoint:
```bash
# For RQ-VAE: Training restarts from scratch (fast)
python -m scripts.train_rqvae --config configs/rqvae_config.yaml

# For LLM: If stage 1 completed, start from stage 2
python -m scripts.train_llm --config configs/stage2_config.yaml --stage 2
```

---

## Project Structure Reference

```
semantic_id_recommender/
├── configs/                    # Training configs (gitignored, user-specific)
│   ├── rqvae_config.yaml       # RQ-VAE config
│   ├── stage1_config.yaml      # LLM stage 1 config
│   └── stage2_config.yaml      # LLM stage 2 config
├── data/
│   ├── catalogue.jsonl         # Your item catalogue
│   ├── embeddings.pt           # Cached embeddings (generated)
│   └── semantic_ids.json       # Semantic ID mapping (generated)
├── checkpoints/
│   ├── llm_stage1/             # Stage 1 checkpoint
│   └── llm_stage2/             # Stage 2 checkpoint
├── models/
│   └── rqvae_model.pt          # RQ-VAE model
├── src/
│   ├── rqvae/                  # RQ-VAE implementation
│   ├── llm/                    # LLM fine-tuning
│   ├── inference/              # Modal deployment
│   └── frontend/               # Gradio UI
├── scripts/
│   ├── train_rqvae.py          # RQ-VAE training script
│   ├── train_llm.py            # LLM training script (stages 1 & 2)
│   └── deploy.py               # Modal deployment
└── notebooks/
    ├── train_rqvae.ipynb       # RQ-VAE training notebook
    └── train_llm.ipynb         # LLM training notebook
```

---

## CLI Reference

### train_rqvae

```bash
python -m scripts.train_rqvae --config <config.yaml> [--create-dummy] [--dummy-size N]

Options:
  --config         Path to YAML config file (required)
  --create-dummy   Create a dummy catalogue for testing
  --dummy-size     Size of dummy catalogue (default: 1000)
```

### train_llm

```bash
python -m scripts.train_llm --config <config.yaml> --stage <1|2|1,2> [--stage2-config <config.yaml>]

Options:
  --config         Path to YAML config file (required)
  --stage          Stage(s) to run: '1', '2', or '1,2' (default: 1)
  --stage2-config  Path to stage 2 config (required when --stage=1,2)

Examples:
  # Stage 1 only
  python -m scripts.train_llm --config stage1_config.yaml --stage 1

  # Stage 2 only
  python -m scripts.train_llm --config stage2_config.yaml --stage 2

  # Both stages
  python -m scripts.train_llm --config stage1_config.yaml --stage 1,2 --stage2-config stage2_config.yaml
```

---

## Next Steps

1. **Improve Recommendations**: Train on real user interaction data
2. **Multiple Recommendations**: Implement beam search for top-k results
3. **Constrained Generation**: Add grammar constraints for valid semantic IDs
4. **A/B Testing**: Compare against baseline recommenders
