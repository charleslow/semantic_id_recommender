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

If you don't have local GPU access, you can use RunPod for training and fine-tuning.

### Step 1: Create a RunPod Account

1. Sign up at https://www.runpod.io/
2. Add credits to your account (pay-as-you-go)

### Step 2: Launch a Pod

1. Go to **Pods** → **GPU Instances**
2. Select a GPU based on your needs:
   - **RQ-VAE Training**: RTX 3060 or higher (~$0.20/hr)
   - **LLM Fine-tuning**: RTX 4090, A6000, or A100 (~$0.40-$2.00/hr)
3. Choose a template:
   - **RunPod PyTorch** (recommended)
   - Or **RunPod Pytorch + JupyterLab** for notebook access
4. Set disk space: at least **50 GB** for model downloads
5. Click **Deploy**

### Step 3: Connect to Your Pod

**Option A: Terminal (SSH)**
```bash
ssh root@<pod-id>.runpod.io -i ~/.ssh/id_ed25519
```
### Step 3.1: Starting Jupyter Notebook in RunPod

If you chose a non-JupyterLab template or want to start Jupyter manually:

**Option B: Install in your virtual environment (Recommended)**
```bash
# After setting up your project
cd semantic_id_recommender
source .venv/bin/activate
python -m jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

**Option C: Add jupyter to pyproject.toml (Best for team projects)**
```bash
# Add jupyter as a dev dependency
uv add --dev jupyterlab

# Then start it
source .venv/bin/activate
python -m jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

**Access JupyterLab**
1. Go to RunPod dashboard
2. Click **Connect** → **HTTP Service [Port 8888]**
3. Or manually go to: `https://<pod-id>-8888.proxy.runpod.net/`

**Tip: Keep Jupyter running with tmux**
```bash
# Start tmux session
tmux new -s jupyter

# Activate venv and start Jupyter
cd semantic_id_recommender
source .venv/bin/activate
python -m jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# Detach: Ctrl+B then D
# Now you can close your terminal and Jupyter keeps running
```

### Step 4: Setup on RunPod

```bash
git clone https://github.com/charleslow/semantic_id_recommender.git
cd semantic_id_recommender

# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$PATH"

uv sync
source .venv/bin/activate

huggingface-cli login  # Paste your HF token
wandb login            # Optional: paste your wandb token
```

### Step 5: Upload Your Data

**Option A: Using SCP**
```bash
# From your local machine
scp data/catalogue.jsonl root@<pod-id>.runpod.io:~/semantic_id_recommender/data/
```

**Option B: Using wget/curl**
```bash
# If data is hosted online
wget https://your-url.com/catalogue.jsonl -O data/catalogue.jsonl
```

**Option C: Using JupyterLab Upload**
1. Click the upload button in JupyterLab file browser
2. Navigate to `data/` folder and upload

### Step 6: Run Training

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

### Step 7: Download Results

**Option A: Using SCP**
```bash
# From your local machine
scp -r root@<pod-id>.runpod.io:~/semantic_id_recommender/checkpoints ./checkpoints
scp root@<pod-id>.runpod.io:~/semantic_id_recommender/data/semantic_ids.json ./data/
```

**Option B: Push to HuggingFace Hub**
```bash
# In RunPod terminal
python -m scripts.finetune_llm --push-to-hub --hub-repo "your-username/semantic-recommender"

# Later, download locally
git clone https://huggingface.co/your-username/semantic-recommender checkpoints/llm
```

**Option C: Using JupyterLab Download**
1. Right-click on files/folders in JupyterLab
2. Select **Download**

### Step 8: Monitor Training

**Using W&B (Recommended)**
```bash
# W&B will log metrics automatically
# View at: https://wandb.ai/your-username/semantic-id-recommender
```

**Using Terminal Output**
```bash
# Training progress is shown in real-time
# Watch GPU usage:
watch -n 1 nvidia-smi
```

### Step 9: Stop Your Pod

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

### Push to HuggingFace (Optional)

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
