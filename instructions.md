# Semantic ID Recommender - Setup & Usage Guide

This guide walks you through setting up and running the semantic ID recommender pipeline.

---

## 1. Prerequisites

### 1.1 Required Accounts

1. **HuggingFace Account** (free)
   - Sign up at https://huggingface.co/join
   - Create an access token at https://huggingface.co/settings/tokens
   - Token needs "Write" permission for pushing models
   - Create private repositories for your models:
     1. Go to https://huggingface.co/new
     2. Repository name: e.g., `semantic-rqvae` or `semantic-recommender`
     3. Type: **Model**
     4. Visibility: **Private** (recommended)
     5. Click "Create repository"
     6. Note your full repo ID: `your-username/semantic-rqvae`

2. **Weights & Biases Account** (free, optional)
   - Sign up at https://wandb.ai/authorize
   - Used for training logging

3. **Modal Account** (free tier available)
   - Sign up at https://modal.com/signup
   - $30 free credits for new accounts

---

## 2. Running on RunPod

This repo is designed for running on RunPod.

### 2.1 Create a RunPod Account

1. Sign up at https://www.runpod.io/
2. Add credits to your account (pay-as-you-go)

### 2.2 Configure Environment Variables (Secrets)

Before launching your pod, set up your API tokens as environment variables.

1. Go to **Settings** → **Environment Variables** in RunPod
2. Add the following environment variables:

| Variable Name   | Description                          | How to Get                                                                 |
|-----------------|--------------------------------------|----------------------------------------------------------------------------|
| `HF_TOKEN`      | HuggingFace API token                | https://huggingface.co/settings/tokens (needs "Write" permission)          |
| `WANDB_API_KEY` | Weights & Biases API key (optional)  | https://wandb.ai/authorize                                                 |

### 2.3 Connect to Your Pod

1. Add your public SSH key to RunPod first (**Settings** → **SSH Keys**)
2. In your pod's **Connect** menu, use **SSH over exposed TCP** (not the proxy method)
3. Note the connection details: `ssh root@<POD_IP> -p <POD_PORT> -i ~/.ssh/id_ed25519`

### 2.4 Connect Using VS Code Remote SSH (Recommended)

Using VS Code's Remote SSH extension provides a full IDE experience on your RunPod instance.

#### Install the Extension

1. Open VS Code
2. Go to **Extensions** (Ctrl+Shift+X)
3. Search for "Remote - SSH" and install the extension by Microsoft

#### Configure SSH Host

1. Open the Command Palette (Ctrl+Shift+P)
2. Type "Remote-SSH: Open SSH Configuration File" and select it
3. Choose your SSH config file (usually `~/.ssh/config`)
4. Add an entry for your RunPod pod:

```
Host runpod
    HostName <POD_IP>
    User root
    Port <POD_PORT>
    IdentityFile ~/.ssh/id_ed25519
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    PubkeyAuthentication yes
```

> **Note**: Replace `<POD_IP>` and `<POD_PORT>` with the values from your pod's **Connect** menu (SSH over exposed TCP).

#### WSL / Devcontainer Users

If you're running VS Code in WSL or a devcontainer, VS Code Remote SSH uses the **Windows** SSH config (`C:\Users\<username>\.ssh\config`), not the Linux one.

**Option 1: Use Windows SSH config (Recommended)**

Edit your Windows SSH config file at `C:\Users\<username>\.ssh\config` and add the RunPod entry there. Make sure your SSH key is also accessible from Windows.

```bash
# Copy SSH key from devcontainer/WSL to Windows
cp ~/.ssh/id_ed25519 /mnt/c/Users/<username>/.ssh/
cp ~/.ssh/id_ed25519.pub /mnt/c/Users/<username>/.ssh/
```

Then update the Windows SSH config at `C:\Users\<username>\.ssh\config`:

```
Host runpod
    HostName <POD_IP>
    User root
    Port <POD_PORT>
    IdentityFile C:\Users\<username>\.ssh\id_ed25519
    StrictHostKeyChecking no
    UserKnownHostsFile NUL
    PubkeyAuthentication yes
```


**Option 3: Configure VS Code to use WSL SSH**

In VS Code settings, add:
```json
"remote.SSH.path": "/usr/bin/ssh"
```
This tells VS Code to use the Linux SSH binary instead of Windows.

#### Connect to RunPod

1. Open the Command Palette (Ctrl+Shift+P)
2. Type "Remote-SSH: Connect to Host" and select it
3. Select `runpod` from the list
4. VS Code will open a new window connected to your pod
5. Open the folder `/workspace/semantic_id_recommender`

### 2.5 Setup the Environment

On the RunPod pod, clone the repository and install dependencies:

```bash
# Clone the code repository
cd /workspace
git clone https://github.com/charleslow/semantic_id_recommender
cd semantic_id_recommender

# Install uv and dependencies
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.bashrc
export PATH="$HOME/.local/bin:$PATH"
uv sync
source .venv/bin/activate

# Install Jupyter kernel
uv pip install ipykernel
python -m ipykernel install --user --name=semantic-id-recommender --display-name="Python (semantic-id-recommender)"
```

### 2.6 Transfer Data to RunPod

From your **local machine**, transfer your data files using scp:

```bash
# Transfer data files to RunPod using direct TCP connection
# Use the POD_IP and POD_PORT from Step 2.3
scp -P ${POD_PORT} -i ~/.ssh/id_ed25519 \
  data/* \
  root@${POD_IP}:/workspace/semantic_id_recommender/data/

scp -P ${POD_PORT} -i ~/.ssh/id_ed25519 \
  configs/* \
  root@${POD_IP}:/workspace/semantic_id_recommender/configs/

# Verify files transferred
ssh root@${POD_IP} -p ${POD_PORT} -i ~/.ssh/id_ed25519 \
  "ls -lh /workspace/semantic_id_recommender/data/"
```

> **Note**: Use the **SSH over exposed TCP** connection (direct IP and port) from Step 2.3.


#### Train RQ-VAE

```bash
# Using your config file
python -m scripts.train_rqvae --config configs/rqvae_config.yaml

# Or test with dummy data
python -m scripts.train_rqvae --config configs/rqvae_config.yaml --create-dummy --dummy-size 1000
```

#### Generate Semantic IDs from Existing RQ-VAE Model

If you've already trained an RQ-VAE model and want to generate semantic IDs for a new catalogue (or re-generate for the same catalogue), use eval mode:

```bash
# Load model from W&B artifact and generate semantic IDs
python -m scripts.train_rqvae \
    --config configs/rqvae_config.yaml \
    --eval \
    --wandb-artifact <your-username>/semantic-id-recommender/rqvae-model:latest
```

This will:
1. Download the RQ-VAE model from your W&B artifact
2. Load the catalogue specified in your config
3. Generate embeddings and semantic IDs for all items
4. Save `semantic_ids.jsonl` and `catalogue.jsonl` to the output directory
5. Log the data artifact back to W&B (if `log_wandb_artifacts: true` in config)

#### Fine-tune LLM

```bash
# Stage 1 only (embedding training)
python -m scripts.train_llm --config configs/stage1_config.yaml --stage 1

# Stage 2 only (LoRA fine-tuning)
python -m scripts.train_llm --config configs/stage2_config.yaml --stage 2

# Both stages sequentially (recommended)
python -m scripts.train_llm --config configs/stage1_config.yaml --stage 1,2 --stage2-config configs/stage2_config.yaml
```

---

## 4. Deploying to Modal

Modal provides serverless GPU deployment for the fine-tuned recommender model.

### 4.1 Prerequisites

1. **Modal Account** - Sign up at https://modal.com/signup ($30 free credits)
2. **Modal CLI** - Install and authenticate:
   ```bash
   pip install modal
   modal setup  # Opens browser for authentication
   ```

### 4.2 Download Artifacts from Weights & Biases

Download all necessary artifacts (model + data) using the download script:

```bash
# Login to W&B (if not already logged in)
wandb login

# Download all artifacts to outputs/
python -m scripts.download_artifacts --project <your-username>/semantic-id-recommender
```

This downloads to the `outputs/` directory:
- `outputs/llm/` - fine-tuned LLM model (stage 2)
- `outputs/data/semantic_ids.jsonl` - mappings between item IDs and semantic IDs
- `outputs/data/catalogue.jsonl` - catalogue items with their semantic IDs

You can also specify custom artifact versions:

```bash
python -m scripts.download_artifacts \
    --project <your-username>/semantic-id-recommender \
    --model-artifact llm-stage2:v3 \
    --data-artifact rqvae-model-data:v2 \
    --output-dir outputs
```

### 4.3 Upload Artifacts to Modal Volume

After downloading, upload your fine-tuned model and data files to a Modal volume:

```bash
# Upload model and mapping files (uses default paths from outputs/)
python -m scripts.upload_artifacts

# Or specify custom paths
python -m scripts.upload_artifacts \
    --model outputs/llm \
    --catalogue outputs/data/catalogue.jsonl \
    --semantic-ids outputs/data/semantic_ids.jsonl
```

This uploads:
- Your fine-tuned LLM model to `/model/semantic-recommender/`
- The catalogue JSONL to `/model/catalogue.jsonl`
- The semantic ID mappings to `/model/semantic_ids.jsonl`

### 4.5 Deploy the App

Deploy the serverless endpoint:

```bash
python -m scripts.deploy
```

This deploys the `semantic-id-recommender` app to Modal with:
- A10G GPU
- 5-minute idle timeout (scales to zero when not in use)
- Support for 10 concurrent requests

### 4.6 Test the Deployment

Test your deployed recommender:

```bash
# Test with a query
python -m scripts.deploy --test "wireless mouse"

# Or test against a specific API URL
python -m scripts.deploy --test "wireless mouse" --api-url "https://your-app-url.modal.run"
```

### 4.6 Full Deployment Workflow

Complete deployment workflow:

```bash
# 1. Download artifacts from W&B
python -m scripts.download_artifacts --project <your-username>/semantic-id-recommender

# 2. Upload to Modal volume
python -m scripts.upload_artifacts

# 3. Deploy and test
python -m scripts.deploy --test "wireless mouse"
```
