from .callbacks import RecommendationTestCallback, SemanticIDEvalCallback
from .data import DEFAULT_SYSTEM_PROMPT, SemanticIDDataset, prepare_training_data, format_as_messages, get_semantic_id_tokens
from .finetune import (
    FinetuneConfig,
    LLMTrainConfig,
    LLMTrainResult,
    add_semantic_tokens,
    create_semantic_id_mapping,
    finetune_model,
    freeze_backbone,
    train,
)
from .inference import load_finetuned_model, SemanticIDGenerator, extract_semantic_id

# Re-export from rqvae for backwards compatibility
from src.rqvae import load_from_path as load_rqvae_from_path
from src.rqvae import load_from_wandb as load_rqvae_from_wandb

__all__ = [
    # Data
    "DEFAULT_SYSTEM_PROMPT",
    "SemanticIDDataset",
    "prepare_training_data",
    "format_as_messages",
    "get_semantic_id_tokens",
    # Training - high-level API
    "train",
    "LLMTrainConfig",
    "LLMTrainResult",
    # Training - low-level API
    "finetune_model",
    "add_semantic_tokens",
    "freeze_backbone",
    "FinetuneConfig",
    # RQ-VAE loading (deprecated, use src.rqvae.load_from_wandb/load_from_path)
    "load_rqvae_from_wandb",
    "load_rqvae_from_path",
    "create_semantic_id_mapping",
    # Callbacks
    "SemanticIDEvalCallback",
    "RecommendationTestCallback",
    # Inference
    "load_finetuned_model",
    "SemanticIDGenerator",
    "extract_semantic_id",
]
