"""
HuggingFace Hub utilities for uploading and downloading RQ-VAE models.

This module provides functions to save/load RQ-VAE models and semantic ID mappings
to/from the HuggingFace Hub for easy sharing and deployment.
"""

import json
from pathlib import Path
from typing import Optional

import torch
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

from .model import SemanticRQVAE, SemanticRQVAEConfig


def save_model_for_hub(
    model: SemanticRQVAE,
    save_dir: str | Path,
    semantic_ids_path: Optional[str | Path] = None,
    training_info: Optional[dict] = None,
) -> None:
    """
    Save RQ-VAE model and optional semantic ID mappings to a directory.

    This prepares the model for uploading to HuggingFace Hub.

    Args:
        model: Trained SemanticRQVAE model
        save_dir: Directory to save model files
        semantic_ids_path: Optional path to semantic_ids.json to include
        training_info: Optional dictionary with training metrics/info

    Files created:
        - rqvae_model.pt: Model checkpoint with config
        - config.json: Model configuration (human-readable)
        - semantic_ids.json: Item to semantic ID mappings (if provided)
        - README.md: Auto-generated model card
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save model checkpoint
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": {
            "embedding_dim": model.config.embedding_dim,
            "hidden_dim": model.config.hidden_dim,
            "codebook_size": model.config.codebook_size,
            "num_quantizers": model.config.num_quantizers,
            "commitment_weight": model.config.commitment_weight,
            "decay": model.config.decay,
            "threshold_ema_dead_code": model.config.threshold_ema_dead_code,
        },
    }

    if training_info:
        checkpoint["training_info"] = training_info

    torch.save(checkpoint, save_dir / "rqvae_model.pt")

    # Save config as JSON for easy inspection
    with open(save_dir / "config.json", "w") as f:
        json.dump(checkpoint["config"], f, indent=2)

    # Copy semantic IDs if provided
    if semantic_ids_path:
        semantic_ids_path = Path(semantic_ids_path)
        if semantic_ids_path.exists():
            import shutil
            shutil.copy(semantic_ids_path, save_dir / "semantic_ids.json")

    # Create README
    readme = f"""---
tags:
- semantic-ids
- recommendation
- rq-vae
- vector-quantization
library_name: pytorch
---

# RQ-VAE Semantic ID Model

This model was trained using Residual Vector Quantization (RQ-VAE) to learn semantic IDs for catalogue items.

## Model Details

- **Embedding Dimension**: {model.config.embedding_dim}
- **Hidden Dimension**: {model.config.hidden_dim}
- **Codebook Size**: {model.config.codebook_size} codes per level
- **Number of Quantizers**: {model.config.num_quantizers} levels
- **Total Semantic ID Space**: {model.config.codebook_size ** model.config.num_quantizers:,} unique IDs

## Usage

```python
import torch
from huggingface_hub import hf_hub_download

# Download model
model_path = hf_hub_download(
    repo_id="your-username/your-repo",
    filename="rqvae_model.pt"
)

# Load checkpoint
checkpoint = torch.load(model_path)

# Recreate model
from src.rqvae.model import SemanticRQVAE, SemanticRQVAEConfig

config = SemanticRQVAEConfig(**checkpoint["config"])
model = SemanticRQVAE(config)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Generate semantic IDs
with torch.no_grad():
    embeddings = ...  # Your item embeddings
    semantic_ids = model.get_semantic_ids(embeddings)
    semantic_strings = model.semantic_id_to_string(semantic_ids)
```

## Files

- `rqvae_model.pt`: Model checkpoint with weights and config
- `config.json`: Model configuration (human-readable)
- `semantic_ids.json`: Item ID to semantic ID mappings (if available)

## Training Info

"""

    if training_info:
        for key, value in training_info.items():
            readme += f"- **{key}**: {value}\n"

    readme += "\n## Citation\n\nIf you use this model, please cite the original RQ-VAE paper and the vector-quantize-pytorch library.\n"

    with open(save_dir / "README.md", "w") as f:
        f.write(readme)

    print(f"✓ Saved model files to {save_dir}")
    print(f"  - rqvae_model.pt")
    print(f"  - config.json")
    if semantic_ids_path and Path(semantic_ids_path).exists():
        print(f"  - semantic_ids.json")
    print(f"  - README.md")


def upload_to_hub(
    model_dir: str | Path,
    repo_id: str,
    token: Optional[str] = None,
    private: bool = False,
    commit_message: str = "Upload RQ-VAE model",
) -> str:
    """
    Upload model directory to HuggingFace Hub.

    Args:
        model_dir: Directory containing model files (from save_model_for_hub)
        repo_id: HuggingFace repo ID (username/repo-name)
        token: HuggingFace API token (or use HF_TOKEN env var)
        private: Whether to create a private repo
        commit_message: Commit message for the upload

    Returns:
        URL of the uploaded model repo

    Example:
        >>> upload_to_hub(
        ...     model_dir="models/rqvae_hub",
        ...     repo_id="myusername/semantic-rqvae",
        ...     token="hf_...",
        ... )
    """
    api = HfApi(token=token)
    model_dir = Path(model_dir)

    # Create repo if it doesn't exist
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
        print(f"✓ Repository ready: {repo_id}")
    except Exception as e:
        print(f"Note: {e}")

    # Upload all files in the directory
    api.upload_folder(
        folder_path=str(model_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
    )

    url = f"https://huggingface.co/{repo_id}"
    print(f"✓ Model uploaded successfully!")
    print(f"  View at: {url}")

    return url


def load_from_hub(
    repo_id: str,
    token: Optional[str] = None,
    revision: str = "main",
    cache_dir: Optional[str] = None,
) -> tuple[SemanticRQVAE, Optional[dict]]:
    """
    Load RQ-VAE model from HuggingFace Hub.

    Args:
        repo_id: HuggingFace repo ID (username/repo-name)
        token: HuggingFace API token (for private repos)
        revision: Git revision (branch, tag, or commit hash)
        cache_dir: Directory to cache downloaded files

    Returns:
        Tuple of (model, semantic_ids_mapping)
        - model: Loaded SemanticRQVAE model in eval mode
        - semantic_ids_mapping: Dict with semantic ID mappings (None if not available)

    Example:
        >>> model, mappings = load_from_hub("myusername/semantic-rqvae")
        >>> semantic_ids = model.get_semantic_ids(embeddings)
    """
    # Download model checkpoint
    model_path = hf_hub_download(
        repo_id=repo_id,
        filename="rqvae_model.pt",
        revision=revision,
        token=token,
        cache_dir=cache_dir,
    )

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location="cpu")

    # Recreate model from config
    config = SemanticRQVAEConfig(**checkpoint["config"])
    model = SemanticRQVAE(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"✓ Loaded model from {repo_id}")
    print(f"  Config: {checkpoint['config']}")
    if "training_info" in checkpoint:
        print(f"  Training info: {checkpoint['training_info']}")

    # Try to load semantic IDs if available
    semantic_ids = None
    try:
        semantic_ids_path = hf_hub_download(
            repo_id=repo_id,
            filename="semantic_ids.json",
            revision=revision,
            token=token,
            cache_dir=cache_dir,
        )
        with open(semantic_ids_path) as f:
            semantic_ids = json.load(f)
        print(f"  ✓ Loaded semantic ID mappings")
    except Exception:
        print(f"  Note: No semantic_ids.json found in repo")

    return model, semantic_ids


def download_model_files(
    repo_id: str,
    local_dir: str | Path,
    token: Optional[str] = None,
    revision: str = "main",
) -> Path:
    """
    Download all model files from HuggingFace Hub to a local directory.

    This is useful if you want to have all files locally instead of using the cache.

    Args:
        repo_id: HuggingFace repo ID (username/repo-name)
        local_dir: Local directory to download files to
        token: HuggingFace API token (for private repos)
        revision: Git revision (branch, tag, or commit hash)

    Returns:
        Path to the local directory

    Example:
        >>> download_model_files(
        ...     repo_id="myusername/semantic-rqvae",
        ...     local_dir="models/downloaded",
        ... )
    """
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    snapshot_path = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        token=token,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,  # Copy files instead of symlinking
    )

    print(f"✓ Downloaded model files to {local_dir}")
    print(f"  Files: {[f.name for f in local_dir.iterdir()]}")

    return Path(snapshot_path)
