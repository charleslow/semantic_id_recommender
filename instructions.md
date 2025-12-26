# Semantic ID Recommender - Setup & Usage Guide

This guide walks you through setting up and running the semantic ID recommender pipeline.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Running on RunPod](#running-on-runpod)
3. [Local Setup](#local-setup)
4. [Stage 1: Train RQ-VAE](#stage-1-train-rq-vae)
5. [Stage 2: Fine-tune LLM](#stage-2-fine-tune-llm)
6. [Stage 3: Deploy to Modal](#stage-3-deploy-to-modal)
7. [Stage 4: Run Frontend](#stage-4-run-frontend)
8. [Testing in Colab](#testing-in-colab)
9. [Troubleshooting](#troubleshooting)

---

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

### Hardware Requirements

| Stage | Minimum GPU | Recommended |
|-------|-------------|-------------|
| RQ-VAE Training | CPU (slow) / Any GPU | RTX 3060+ |
| LLM Fine-tuning | 16GB VRAM | 24GB VRAM (RTX 4090, A10G) |
| Inference | A10G (Modal) | A10G |

---

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

**Important**: These variables will be automatically injected into all your pods when they start.

### Step 3: Launch a Pod

1. Go to **Pods** → **GPU Instances**
2. Select a GPU based on your needs:
   - **RQ-VAE Training**: RTX 3060 or higher (~$0.20/hr)
   - **LLM Fine-tuning**: RTX 4090, A6000, or A100 (~$0.40-$2.00/hr)
3. Choose a template:
   - **RunPod PyTorch** (recommended)
   - Or **RunPod Pytorch + JupyterLab** for notebook access
4. Set disk space: at least **50 GB** for model downloads
5. **Advanced Options** (optional):
   - You can also add environment variables per-pod here if you prefer
   - Add `HF_TOKEN` and `WANDB_API_KEY` in the "Environment Variables" section
6. Click **Deploy**

### Step 4: Connect to Your Pod

Make sure to add your public SSH key to RunPod first (Settings → SSH Keys).

In your pod's **Connect** menu, use **SSH over exposed TCP** (not the proxy method):

```bash
# Get the connection command from RunPod dashboard
# It will look something like:
ssh root@<ip-address> -p <port> -i ~/.ssh/id_ed25519

# Save these for later use:
export POD_IP="<ip-address>"
export POD_PORT="22022"
ssh root@"$POD_IP" -p "$POD_PORT" -i ~/.ssh/id_ed25519
```

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


### Step 8: Run Training

#### Train RQ-VAE
```bash
# Using your data
python -m scripts.train_rqvae --catalogue data/catalogue.jsonl

# Or test with dummy data
python -m scripts.train_rqvae --create-dummy --dummy-size 1000
```

#### Fine-tune LLM
```bash
# Using default config
python -m scripts.finetune_llm

# Or with custom model
python -m scripts.finetune_llm --base-model "unsloth/Qwen3-4B"

# Push to HuggingFace Hub when done
python -m scripts.finetune_llm --push-to-hub --hub-repo "your-username/semantic-recommender"
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

# For fine-tuned LLM - push during training
# The HF_TOKEN environment variable will be used automatically
python -m scripts.finetune_llm \
  --push-to-hub \
  --hub-repo "your-username/semantic-recommender"
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
  python -m scripts.finetune_llm
  
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
# Using existing data
python -m scripts.train_rqvae --catalogue data/mcf_articles.jsonl

# Or create dummy data for testing
python -m scripts.train_rqvae --create-dummy --dummy-size 1000
```

### Configuration

Copy the sample config and customize:

```bash
cp config.sample.yaml config.yaml
```

Edit `config.yaml` to customize:

```yaml
rqvae:
  codebook_size: 256      # Codes per level (256^4 = 4B possible items)
  num_quantizers: 4       # Number of levels in semantic ID
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  max_epochs: 100
  batch_size: 256
```

### What Happens

1. Loads catalogue items
2. Generates embeddings using sentence-transformers
3. Trains RQ-VAE to learn discrete codes
4. Saves semantic ID mapping to `data/semantic_ids.json`

### Expected Output

```
Loaded 10000 items from data/catalogue.jsonl
Generating embeddings with sentence-transformers/all-MiniLM-L6-v2...
100%|████████████████████████████████| 10000/10000
Dataset size: 10000
Epoch 99: 100%|████████████████████████████████|
val/loss: 0.0234
Saved semantic IDs to data/semantic_ids.json
```

### Verify Results

```bash
# Check the semantic ID mapping
head -20 data/semantic_ids.json
```

---

## Stage 2: Fine-tune LLM

Fine-tune a small LLM to generate semantic IDs from user queries.

### Quick Start

```bash
python -m scripts.finetune_llm
```

### Configuration

Edit `config.yaml`:

```yaml
llm:
  base_model: "unsloth/Qwen3-4B"  # Or "unsloth/Ministral-3B-Instruct"
  lora_r: 16
  lora_alpha: 32
  max_epochs: 3
  batch_size: 4
  gradient_accumulation_steps: 4
```

### What Happens

1. Generates training data from catalogue + semantic IDs
2. Creates query → semantic ID pairs
3. Fine-tunes with QLoRA using Unsloth
4. Saves model to `checkpoints/llm/`

### Push to HuggingFace Hub

**First, create a private HuggingFace repository:**

```bash
# Option 1: Using the UI
# Go to https://huggingface.co/new and create a private Model repository

# Option 2: Using CLI
huggingface-cli login
huggingface-cli repo create semantic-recommender --type model --private
```

**Then push your fine-tuned model:**

```bash
python -m scripts.finetune_llm --push-to-hub --hub-repo "your-username/semantic-recommender"
```

### Expected Output

```
Preparing training data...
Generated 30000 training examples
Train: 27000, Val: 3000
Fine-tuning unsloth/Qwen3-4B...
Epoch 1/3: loss=2.34
Epoch 2/3: loss=0.89
Epoch 3/3: loss=0.45
Model saved to checkpoints/llm
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
!python -m scripts.train_rqvae --create-dummy --dummy-size 100

# Test LLM (requires GPU)
!python -m scripts.finetune_llm --base-model "unsloth/Qwen3-0.6B"
```

---

## Troubleshooting

### Out of Memory (LLM Fine-tuning)

```yaml
# Reduce batch size in config
llm:
  batch_size: 2
  gradient_accumulation_steps: 8
```

Or use a smaller model:
```bash
python -m scripts.finetune_llm --base-model "unsloth/Qwen3-0.6B"
```

### Slow Embedding Generation

```yaml
# Use GPU for embeddings
# The script auto-detects GPU, but ensure CUDA is available
```

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

---

## Project Structure Reference

```
semantic_id_recommender/
├── config.sample.yaml          # Sample configuration (copy to config.yaml)
├── config.yaml                 # Your local configuration (gitignored)
├── data/
│   ├── catalogue.json          # Your item catalogue
│   ├── embeddings.pt           # Cached embeddings (generated)
│   ├── semantic_ids.json       # Semantic ID mapping (generated)
│   ├── train.jsonl             # Training data (generated)
│   └── val.jsonl               # Validation data (generated)
├── checkpoints/
│   ├── rqvae/                  # RQ-VAE checkpoints
│   └── llm/                    # Fine-tuned LLM
├── src/
│   ├── config/                 # Config loader
│   ├── rqvae/                  # RQ-VAE implementation
│   ├── llm/                    # LLM fine-tuning
│   ├── inference/              # Modal deployment
│   └── frontend/               # Gradio UI
├── scripts/
│   ├── train_rqvae.py          # Stage 1
│   ├── finetune_llm.py         # Stage 2
│   └── deploy.py               # Stage 3
└── notebooks/
    └── test_training.ipynb     # Colab notebook
```

---

## Next Steps

1. **Improve Recommendations**: Train on real user interaction data
2. **Multiple Recommendations**: Implement beam search for top-k results
3. **Constrained Generation**: Add grammar constraints for valid semantic IDs
4. **A/B Testing**: Compare against baseline recommenders
