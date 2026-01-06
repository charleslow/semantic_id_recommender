"""
LLM fine-tuning and inference for semantic ID generation.

Note: Heavy imports (finetune, callbacks) are lazy-loaded to avoid
importing unsloth when only data/inference utilities are needed.
"""

# Light imports - no unsloth dependency
from .data import (
    DEFAULT_SYSTEM_PROMPT,
    SemanticIDDataset,
    format_as_messages,
    get_semantic_id_tokens,
    prepare_training_data,
)
from .inference import SemanticIDGenerator, extract_semantic_id, load_finetuned_model

__all__ = [
    # Data
    "DEFAULT_SYSTEM_PROMPT",
    "SemanticIDDataset",
    "prepare_training_data",
    "format_as_messages",
    "get_semantic_id_tokens",
    # Training - high-level API (lazy imports)
    "train",
    "LLMTrainConfig",
    "LLMTrainResult",
    # Training - low-level API (lazy imports)
    "finetune_model",
    "add_semantic_tokens",
    "freeze_backbone",
    "FinetuneConfig",
    # Callbacks (lazy imports)
    "SemanticIDEvalCallback",
    "RecommendationTestCallback",
    # Inference
    "load_finetuned_model",
    "SemanticIDGenerator",
    "extract_semantic_id",
]


def __getattr__(name: str):
    """Lazy import heavy dependencies (unsloth) only when needed."""
    if name in (
        "FinetuneConfig",
        "LLMTrainConfig",
        "LLMTrainResult",
        "add_semantic_tokens",
        "create_semantic_id_mapping",
        "finetune_model",
        "freeze_backbone",
        "train",
    ):
        from . import finetune

        return getattr(finetune, name)

    if name in ("RecommendationTestCallback", "SemanticIDEvalCallback"):
        from . import callbacks

        return getattr(callbacks, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
