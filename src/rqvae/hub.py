"""
Utilities for loading/saving RQ-VAE models from various sources.

This module provides functions to save/load RQ-VAE models and semantic ID mappings
to/from HuggingFace Hub, W&B artifacts, and local paths.
"""

import json
from pathlib import Path
from typing import Optional

import torch
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

from .model import SemanticRQVAE, SemanticRQVAEConfig


def load_from_path(model_path: str | Path) -> tuple[SemanticRQVAE, dict]:
    """
    Load RQ-VAE model from a local path.

    Args:
        model_path: Path to the model checkpoint file

    Returns:
        Tuple of (model, config_dict)
    """
    checkpoint = torch.load(model_path, map_location="cpu")
    config_dict = checkpoint["config"]

    config = SemanticRQVAEConfig(**config_dict)
    model = SemanticRQVAE(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded RQ-VAE from: {model_path}")
    print(f"  Config: {config_dict}")

    return model, config_dict


def load_from_wandb(
    artifact_name: str,
    project: str | None = None,
) -> tuple[SemanticRQVAE, dict]:
    """
    Load RQ-VAE model from a wandb artifact.

    Args:
        artifact_name: Artifact name with version (e.g., "rqvae-model:v3" or "rqvae-model:latest")
        project: W&B project name (optional, uses current run's project if None)

    Returns:
        Tuple of (model, config_dict)
    """
    import wandb

    # Build artifact path
    if project:
        artifact_path = f"{project}/{artifact_name}"
    else:
        artifact_path = artifact_name

    # Download artifact
    if wandb.run is not None:
        artifact = wandb.run.use_artifact(artifact_path, type="model")
    else:
        api = wandb.Api()
        artifact = api.artifact(artifact_path, type="model")

    artifact_dir = artifact.download()
    model_path = Path(artifact_dir) / "rqvae_model.pt"

    return load_from_path(model_path)


def save_model_for_hub(
    model: SemanticRQVAE,
    local_dir: str | Path,
    semantic_ids_path: Optional[str | Path] = None,
    training_info: Optional[dict] = None,
) -> None:
    """
    Save RQ-VAE model and optional semantic ID mappings to a local directory.

    This prepares the model for uploading to HuggingFace Hub.

    Args:
        model: Trained SemanticRQVAE model
        local_dir: Local directory to save model files
        semantic_ids_path: Optional path to semantic_ids.json to include
        training_info: Optional dictionary with training metrics/info

    Files created:
        - rqvae_model.pt: Model checkpoint with config
        - config.json: Model configuration (human-readable)
        - semantic_ids.json: Item to semantic ID mappings (if provided)

    Note: README.md should be created/edited directly on HuggingFace Hub
    """
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    # Save model checkpoint locally
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

    torch.save(checkpoint, local_dir / "rqvae_model.pt")

    # Save config as JSON for easy inspection
    with open(local_dir / "config.json", "w") as f:
        json.dump(checkpoint["config"], f, indent=2)

    # Copy semantic IDs if provided
    if semantic_ids_path:
        semantic_ids_path = Path(semantic_ids_path)
        if semantic_ids_path.exists():
            import shutil

            shutil.copy(semantic_ids_path, local_dir / "semantic_ids.json")

    print(f"✓ Saved model files to {local_dir}")
    print("  - rqvae_model.pt")
    print("  - config.json")
    if semantic_ids_path and Path(semantic_ids_path).exists():
        print("  - semantic_ids.json")
    print(
        "\nNote: Create/edit the README.md directly on HuggingFace Hub after uploading."
    )


def upload_to_hub(
    local_dir: str | Path,
    repo_id: str,
    token: Optional[str] = None,
    commit_message: str = "Upload RQ-VAE model",
) -> str:
    """
    Upload model directory to HuggingFace Hub.

    IMPORTANT: This function assumes a private repository has already been created.
    Create the repo first using the HuggingFace UI or CLI:

    Using HuggingFace UI:
        1. Go to https://huggingface.co/new
        2. Enter repository name (e.g., "semantic-rqvae")
        3. Select "Model" as the repository type
        4. Check "Make this repository private"
        5. Click "Create repository"

    Args:
        local_dir: Local directory containing model files (from save_model_for_hub)
        repo_id: HuggingFace repo ID (username/repo-name) - repo must already exist
        token: HuggingFace API token (or use HF_TOKEN env var)
        commit_message: Commit message for the upload

    Returns:
        URL of the uploaded model repo

    Example:
        >>> # First create the repo (see above)
        >>> upload_to_hub(
        ...     local_dir="models/rqvae_hub",
        ...     repo_id="myusername/semantic-rqvae",
        ...     token="hf_...",
        ... )
    """
    api = HfApi(token=token)
    local_dir = Path(local_dir)

    # Verify repo exists
    try:
        api.repo_info(repo_id=repo_id, repo_type="model", token=token)
        print(f"✓ Found existing repository: {repo_id}")
    except Exception as e:
        print(f"✗ Error: Repository '{repo_id}' not found or not accessible.")
        print("  Please create the private repository first:")
        raise ValueError(
            f"Repository {repo_id} must be created before uploading"
        ) from e

    # Upload all files in the directory
    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
    )

    url = f"https://huggingface.co/{repo_id}"
    print("✓ Model uploaded successfully!")
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
        print("  ✓ Loaded semantic ID mappings")
    except Exception:
        print("  Note: No semantic_ids.json found in repo")

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
