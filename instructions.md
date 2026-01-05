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

**Option 2: Copy SSH key into devcontainer**

```bash
# Inside your devcontainer, create .ssh directory
mkdir -p ~/.ssh
chmod 700 ~/.ssh

# Copy your key from Windows mount (adjust path as needed)
cp /mnt/c/Users/<username>/.ssh/id_ed25519 ~/.ssh/
cp /mnt/c/Users/<username>/.ssh/id_ed25519.pub ~/.ssh/
chmod 600 ~/.ssh/id_ed25519

# Create SSH config inside devcontainer
cat >> ~/.ssh/config << 'EOF'
Host runpod
    HostName <POD_IP>
    User root
    Port <POD_PORT>
    IdentityFile ~/.ssh/id_ed25519
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    PubkeyAuthentication yes
    IdentitiesOnly yes
EOF
chmod 600 ~/.ssh/config
```

Then connect from the devcontainer terminal: `ssh runpod`

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

#### Tips for Remote Development

- **Install extensions on remote**: Some extensions need to be installed on the remote machine. VS Code will prompt you to install them.
- **Terminal access**: Use VS Code's integrated terminal (Ctrl+`) to run commands directly on the pod
- **Port forwarding**: VS Code automatically forwards ports. If you run a Jupyter notebook or web server, it will be accessible locally.
- **Reconnecting**: If the connection drops, VS Code will attempt to reconnect automatically. You can also manually reconnect via the Command Palette.

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

### 2.7 Run Training with tmux

> **Important**: Always use `tmux` for long-running training jobs. This ensures training continues even if your SSH connection drops.

Install tmux if not available.

```bash
apt-get update && apt-get install -y tmux
```

Now we start a new tmux session or attach to an existing one:

```bash
# Start a new session
tmux new -s training
source .venv/bin/activate

# Attach to an existing session
tmux attach -t training 
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

| Action                                      | Command / Shortcut           |
|---------------------------------------------|------------------------------|
| Detach from tmux (training continues)       | `Ctrl+B`, then `D`           |
| Reattach to running session                 | `tmux attach -t training`    |
| List all sessions                           | `tmux ls`                    |
| Kill a session when done                    | `tmux kill-session -t training` |
| Create a new window in the same session     | `Ctrl+B`, then `C`           |
| Switch between windows                      | `Ctrl+B`, then window number (0, 1, 2...) |

### 2.8 Save Models to HuggingFace Hub

After training completes, push your models to HuggingFace Hub (required for deployment).

> **Note**: Your `HF_TOKEN` environment variable from Step 2.2 will be used automatically.

#### Create Private HuggingFace Repositories First

1. Go to https://huggingface.co/new
2. Repository name: `semantic-rqvae` (or your preferred name)
3. Type: **Model**
4. Visibility: **Private**
5. Click "Create repository"
6. Note your repo ID: `your-username/semantic-rqvae`

#### Upload Your Models

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

---

## 3. Using a Pre-built Docker Image (Recommended)

To avoid running `uv sync` every time you start a pod, build and push a Docker image with dependencies pre-installed.

### 3.1 Create a Docker Hub Account

1. Sign up at https://hub.docker.com/signup (free tier is fine)
2. Note your username (e.g., `yourusername`)

### 3.2 Build and Push the Image

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

### 3.3 Launch RunPod with Your Image

1. Go to **Pods** → **Deploy**
2. Under **Container Image**, enter: `yourusername/semantic-id-recommender:latest`
   - Or `ghcr.io/yourusername/semantic-id-recommender:latest` if using GitHub
3. Set disk space (at least 50GB) and select GPU
4. Add environment variables (`HF_TOKEN`, `WANDB_API_KEY`) as described in Step 2.2
5. Click **Deploy**
