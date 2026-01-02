"""
LLM fine-tuning script using Unsloth.

Fine-tunes a base model to generate semantic IDs from user queries.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import torch
from datasets import Dataset
from transformers import PreTrainedTokenizerBase
from trl import SFTConfig, SFTTrainer

from .callbacks import (
    RecommendationTestCallback,
    SemanticIDEvalCallback,
    WandbArtifactCallback,
)
from .data import DEFAULT_SYSTEM_PROMPT, get_semantic_id_tokens


@dataclass
class FinetuneConfig:
    """Configuration for LLM fine-tuning.

    For two-stage training:
        Stage 1: Set stage=1, base_model to the pretrained model (e.g., "unsloth/Qwen3-4B")
        Stage 2: Set stage=2, stage1_checkpoint to the stage 1 output directory
    """

    # Model settings
    base_model: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    stage1_checkpoint: str | None = None  # Path to stage 1 checkpoint for stage 2
    max_length: int = 512
    load_in_4bit: bool = True

    # Semantic ID settings
    num_quantizers: int = 4
    codebook_size: int = 256

    # LoRA settings
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # Training settings
    learning_rate: float = 2e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_train_epochs: int = 3
    max_steps: int = -1  # -1 for epoch-based training
    warmup_ratio: float = 0.03
    warmup_steps: int = 0  # If > 0, overrides warmup_ratio
    weight_decay: float = 0.01
    optim: str = "adamw_8bit"
    lr_scheduler_type: str = "linear"
    gradient_checkpointing: bool = True

    # Logging and saving
    output_dir: str = "checkpoints/llm"
    logging_steps: int = 10
    save_strategy: str = "epoch"
    save_steps: int = 500
    save_total_limit: int = 2
    eval_steps: int = 500

    # Stage training
    stage: Literal[1, 2] = 1

    # Misc
    seed: int = 42
    num_proc: int = 4
    dataloader_num_workers: int = 16
    report_to: str = "wandb"
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    # W&B artifact logging
    log_wandb_artifacts: bool = False
    wandb_artifact_name: str | None = None  # Auto-generated if None

    # Recommendation test callback settings
    recommendation_test_queries: list[str] = field(default_factory=list)
    semantic_id_to_item: dict[str, dict] | None = None


def add_semantic_tokens(
    tokenizer: PreTrainedTokenizerBase,
    num_quantizers: int = 4,
    codebook_size: int = 256,
) -> PreTrainedTokenizerBase:
    """
    Add semantic ID special tokens to tokenizer.

    Args:
        tokenizer: HuggingFace tokenizer
        num_quantizers: Number of RQ-VAE quantizers
        codebook_size: Size of each codebook

    Returns:
        Updated tokenizer
    """
    special_tokens = get_semantic_id_tokens(num_quantizers, codebook_size)
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    return tokenizer


def freeze_backbone(model, num_new_tokens: int = 0) -> list:
    """
    Freeze all model parameters except input/output embeddings.

    Used in stage 1 training to only train the new semantic ID token embeddings
    while keeping the pretrained weights frozen.

    Args:
        model: The language model to freeze
        num_new_tokens: Number of newly added tokens. If > 0, gradient hooks will
            be registered to zero out gradients for all tokens except the last
            `num_new_tokens` tokens, ensuring only new token embeddings are trained.

    Returns:
        List of gradient hook handles. These should be removed after training
        by calling handle.remove() for each handle.
    """
    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze embedding layers using model's built-in methods
    embedding_layer = model.get_input_embeddings()
    output_layer = model.get_output_embeddings()

    if embedding_layer is not None:
        for param in embedding_layer.parameters():
            param.requires_grad = True

    if output_layer is not None:
        for param in output_layer.parameters():
            param.requires_grad = True

    # Register gradient hooks to zero out gradients for original tokens
    hook_handles = []
    if num_new_tokens > 0:

        def make_gradient_mask_hook(num_new: int):
            """Create a hook that zeros gradients for all but the last num_new tokens."""

            def hook(grad):
                # grad shape: [vocab_size, hidden_dim]
                # Zero out gradients for original tokens (all but last num_new)
                mask = torch.zeros_like(grad)
                mask[-num_new:] = 1.0
                return grad * mask

            return hook

        if embedding_layer is not None:
            for param in embedding_layer.parameters():
                handle = param.register_hook(make_gradient_mask_hook(num_new_tokens))
                hook_handles.append(handle)

        if output_layer is not None and output_layer is not embedding_layer:
            # Only add hooks if output layer is different from input embedding
            # (some models tie weights)
            for param in output_layer.parameters():
                handle = param.register_hook(make_gradient_mask_hook(num_new_tokens))
                hook_handles.append(handle)

    return hook_handles


def create_formatting_func(
    tokenizer: PreTrainedTokenizerBase,
) -> callable:
    """Create a formatting function that uses the tokenizer's chat template."""

    def formatting_prompts_func(examples):
        """Format examples for training using the tokenizer's chat template."""
        messages_list = examples["messages"]

        # Handle both single example and batched examples
        # Single: {"messages": [{"role": ..., "content": ...}, ...]}
        # Batched: {"messages": [[{"role": ..., "content": ...}, ...], ...]}
        if messages_list and isinstance(messages_list[0], dict):
            # Single example - messages_list is the list of message dicts
            text = tokenizer.apply_chat_template(
                messages_list,
                tokenize=False,
                add_generation_prompt=False,
            )
            return [text]

        # Batched examples
        output_texts = []
        for messages in messages_list:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            output_texts.append(text)
        return output_texts

    return formatting_prompts_func


def finetune_model(
    train_dataset: Dataset,
    val_dataset: Dataset | None = None,
    config: FinetuneConfig | None = None,
):
    """
    Fine-tune LLM using Unsloth for semantic ID generation.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset (optional)
        config: Fine-tuning configuration. If None, uses default FinetuneConfig.

    Returns:
        Trained model and tokenizer
    """
    from unsloth import FastLanguageModel

    if config is None:
        config = FinetuneConfig()

    # Determine which model to load
    if config.stage == 2:
        if config.stage1_checkpoint is None:
            raise ValueError("stage1_checkpoint must be set for stage 2 training")
        model_to_load = config.stage1_checkpoint
    else:
        model_to_load = config.base_model

    # Load model with Unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_to_load,
        max_seq_length=config.max_length,
        dtype=None,  # Auto-detect
        load_in_4bit=config.load_in_4bit,
    )

    # Add semantic ID tokens (only for stage 1 or no stage)
    # For stage 2, tokens should already be in the tokenizer from stage 1 checkpoint
    if config.stage != 2:
        tokenizer = add_semantic_tokens(
            tokenizer, config.num_quantizers, config.codebook_size
        )
        model.resize_token_embeddings(len(tokenizer))

    # Add LoRA adapters (only for stage 2 or no stage)
    # Stage 1 trains embeddings directly without LoRA
    if config.stage != 1:
        model = FastLanguageModel.get_peft_model(
            model,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            bias="none",
            use_gradient_checkpointing="unsloth"
            if config.gradient_checkpointing
            else False,
            random_state=config.seed,
        )

    # Stage 1: Freeze all parameters except new token embeddings
    if config.stage == 1:
        # Calculate number of new semantic ID tokens:
        # 3 special tokens (REC, SEM_START, SEM_END) + num_quantizers * codebook_size
        num_new_tokens = 3 + config.num_quantizers * config.codebook_size
        freeze_backbone(model, num_new_tokens=num_new_tokens)

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    # Build SFTConfig
    sft_config = SFTConfig(
        output_dir=config.output_dir,
        # Dataset processing
        dataset_num_proc=config.num_proc,
        # Batch and accumulation
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        # Training duration
        max_steps=config.max_steps if config.max_steps > 0 else -1,
        num_train_epochs=config.num_train_epochs if config.max_steps <= 0 else 1,
        # Optimizer settings
        learning_rate=config.learning_rate,
        optim=config.optim,
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_steps=config.warmup_steps if config.warmup_steps > 0 else 0,
        warmup_ratio=config.warmup_ratio if config.warmup_steps <= 0 else 0.0,
        # Precision
        bf16=use_bf16,
        fp16=not use_bf16,
        # Logging
        logging_steps=config.logging_steps,
        report_to=config.report_to,
        # Saving
        save_strategy=config.save_strategy,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        # Evaluation (must match save_strategy when load_best_model_at_end is True)
        eval_strategy=config.save_strategy if val_dataset else "no",
        eval_steps=config.eval_steps if val_dataset else None,
        metric_for_best_model="eval_loss" if val_dataset else None,
        greater_is_better=False if val_dataset else None,
        load_best_model_at_end=bool(val_dataset),
        # Misc
        seed=config.seed,
        gradient_checkpointing=config.gradient_checkpointing,
        # Sequence length
        max_length=config.max_length,
        packing=False,
        # DataLoader
        dataloader_num_workers=config.dataloader_num_workers,
    )

    # Create formatting function with tokenizer's chat template
    formatting_func = create_formatting_func(tokenizer)

    # Create callbacks
    callbacks = []
    if val_dataset is not None:
        semantic_id_eval_callback = SemanticIDEvalCallback(
            val_dataset=val_dataset,
            tokenizer=tokenizer,
            num_quantizers=config.num_quantizers,
            system_prompt=config.system_prompt,
        )
        callbacks.append(semantic_id_eval_callback)

    if config.recommendation_test_queries and config.semantic_id_to_item:
        recommendation_test_callback = RecommendationTestCallback(
            test_queries=config.recommendation_test_queries,
            tokenizer=tokenizer,
            semantic_id_to_item=config.semantic_id_to_item,
            num_quantizers=config.num_quantizers,
            system_prompt=config.system_prompt,
        )
        callbacks.append(recommendation_test_callback)

    # Add artifact logging callback (logs at end of training while wandb run is active)
    if config.log_wandb_artifacts and config.report_to == "wandb":
        artifact_callback = WandbArtifactCallback(
            config=config,
            train_examples=len(train_dataset),
            val_examples=len(val_dataset) if val_dataset else 0,
        )
        callbacks.append(artifact_callback)

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=sft_config,
        formatting_func=formatting_func,
        callbacks=callbacks if callbacks else None,
    )

    # Train
    trainer.train()

    # Save
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    return model, tokenizer


@dataclass
class LLMTrainConfig:
    """
    Comprehensive configuration for end-to-end LLM training.

    This config captures all parameters needed to run the full training pipeline:
    - Loading RQ-VAE model from wandb artifacts
    - Creating semantic ID mappings
    - Preparing training data
    - Training stage 1 (embedding) and/or stage 2 (LoRA)
    - Logging artifacts to wandb

    Example:
        >>> config = LLMTrainConfig(
        ...     wandb_rqvae_artifact="rqvae-model:latest",
        ...     catalogue_path="data/catalogue.jsonl",
        ...     stage=1,
        ... )
        >>> result = train(config)
    """

    # === RQ-VAE Model Source ===
    # Option 1: Load from wandb artifact
    wandb_rqvae_artifact: str | None = (
        None  # e.g., "rqvae-model:v3" or "rqvae-model:latest"
    )
    wandb_rqvae_project: str | None = (
        None  # e.g., "my-project" (defaults to wandb_project)
    )
    # Option 2: Load from local path
    rqvae_model_path: str | None = None  # e.g., "models/rqvae_model.pt"

    # === Catalogue and Data ===
    catalogue_path: str = "data/catalogue.jsonl"
    catalogue_id_field: str = "id"  # Field name for item IDs
    embedding_model: str = "TaylorAI/gte-tiny"  # For generating embeddings
    embeddings_cache_path: str | None = None  # Cache path for embeddings
    embedding_batch_size: int = 32  # Batch size for embedding generation

    # Query templates for training data generation
    query_templates: dict[str, list[str]] | None = None
    field_mapping: dict[str, str] | None = (
        None  # Map template placeholders to catalogue fields
    )
    num_examples_per_item: int = 5
    predict_semantic_id_ratio: float = (
        0.8  # Ratio of semantic ID prediction vs attribute prediction
    )
    val_split: float = 0.1

    # === Base LLM ===
    base_model: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    max_length: int = 512
    load_in_4bit: bool = True

    # === Stage Training ===
    stage: Literal[1, 2] = 1
    # For stage 2: path to stage 1 checkpoint (local path or wandb artifact)
    stage1_checkpoint: str | None = None
    # For stage 2 from wandb: artifact name (e.g., "llm-stage1:latest")
    wandb_stage1_artifact: str | None = None

    # === LoRA Settings (Stage 2 only) ===
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # === Training Hyperparameters ===
    learning_rate: float = 2e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_train_epochs: int = 3
    max_steps: int = -1
    warmup_ratio: float = 0.03
    warmup_steps: int = 0
    weight_decay: float = 0.01
    optim: str = "adamw_8bit"
    lr_scheduler_type: str = "linear"
    gradient_checkpointing: bool = True

    # === Output ===
    output_dir: str = "checkpoints/llm"
    semantic_ids_output_path: str = "data/semantic_ids.json"

    # === Logging and Saving ===
    logging_steps: int = 10
    save_strategy: str = "epoch"
    save_steps: int = 500
    save_total_limit: int = 2
    eval_steps: int = 500

    # === W&B Configuration ===
    wandb_project: str | None = "semantic-id-recommender"
    wandb_run_name: str | None = None
    report_to: str = "wandb"
    log_wandb_artifacts: bool = False
    wandb_artifact_name: str | None = None

    # === Evaluation ===
    recommendation_test_queries: list[str] = field(default_factory=list)

    # === Misc ===
    seed: int = 42
    num_proc: int = 4
    dataloader_num_workers: int = 16
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    def to_finetune_config(self) -> FinetuneConfig:
        """Convert to FinetuneConfig for the finetune_model function."""
        return FinetuneConfig(
            base_model=self.base_model,
            stage1_checkpoint=self.stage1_checkpoint,
            max_length=self.max_length,
            load_in_4bit=self.load_in_4bit,
            num_quantizers=0,  # Will be set from RQ-VAE config
            codebook_size=0,  # Will be set from RQ-VAE config
            lora_r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            num_train_epochs=self.num_train_epochs,
            max_steps=self.max_steps,
            warmup_ratio=self.warmup_ratio,
            warmup_steps=self.warmup_steps,
            weight_decay=self.weight_decay,
            optim=self.optim,
            lr_scheduler_type=self.lr_scheduler_type,
            gradient_checkpointing=self.gradient_checkpointing,
            output_dir=self.output_dir,
            logging_steps=self.logging_steps,
            save_strategy=self.save_strategy,
            save_steps=self.save_steps,
            save_total_limit=self.save_total_limit,
            eval_steps=self.eval_steps,
            stage=self.stage,
            seed=self.seed,
            num_proc=self.num_proc,
            dataloader_num_workers=self.dataloader_num_workers,
            report_to=self.report_to,
            system_prompt=self.system_prompt,
            log_wandb_artifacts=self.log_wandb_artifacts,
            wandb_artifact_name=self.wandb_artifact_name,
            recommendation_test_queries=self.recommendation_test_queries,
        )


@dataclass
class LLMTrainResult:
    """Result of LLM training."""

    model: object  # The trained model
    tokenizer: object  # The tokenizer
    semantic_id_mapping: dict  # Mapping from item_id to semantic_id
    config: LLMTrainConfig
    metrics: dict | None = None


def load_rqvae_from_wandb(
    artifact_name: str,
    project: str | None = None,
    entity: str | None = None,
) -> tuple["SemanticRQVAE", dict]:  # noqa: F821
    """
    Load RQ-VAE model from a wandb artifact.

    Args:
        artifact_name: Artifact name with version (e.g., "rqvae-model:v3" or "rqvae-model:latest")
        project: W&B project name (optional, uses current run's project if None)
        entity: W&B entity/username (optional)

    Returns:
        Tuple of (model, config_dict)
    """
    import wandb

    from src.rqvae.model import SemanticRQVAE, SemanticRQVAEConfig

    # Build artifact path
    if project:
        if entity:
            artifact_path = f"{entity}/{project}/{artifact_name}"
        else:
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

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location="cpu")
    config_dict = checkpoint["config"]

    # Create model
    config = SemanticRQVAEConfig(**config_dict)
    model = SemanticRQVAE(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded RQ-VAE from artifact: {artifact_path}")
    print(f"  Config: {config_dict}")

    return model, config_dict


def load_rqvae_from_path(model_path: str) -> tuple["SemanticRQVAE", dict]:  # noqa: F821
    """
    Load RQ-VAE model from a local path.

    Args:
        model_path: Path to the model checkpoint file

    Returns:
        Tuple of (model, config_dict)
    """
    from src.rqvae.model import SemanticRQVAE, SemanticRQVAEConfig

    checkpoint = torch.load(model_path, map_location="cpu")
    config_dict = checkpoint["config"]

    config = SemanticRQVAEConfig(**config_dict)
    model = SemanticRQVAE(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded RQ-VAE from: {model_path}")
    print(f"  Config: {config_dict}")

    return model, config_dict


def create_semantic_id_mapping(
    rqvae_model: "SemanticRQVAE",  # noqa: F821
    catalogue_path: str,
    embedding_model: str,
    id_field: str = "id",
    embeddings_cache_path: str | None = None,
    output_path: str | None = None,
    embedding_batch_size: int = 32,
) -> dict:
    """
    Create semantic ID mapping for all items in a catalogue.

    Args:
        rqvae_model: Trained RQ-VAE model
        catalogue_path: Path to catalogue JSONL file
        embedding_model: Name of embedding model to use
        id_field: Field name for item IDs in catalogue
        embeddings_cache_path: Optional path to cache embeddings
        output_path: Optional path to save the mapping JSON
        embedding_batch_size: Batch size for embedding generation

    Returns:
        Dictionary with:
        - item_to_semantic: Dict mapping item_id -> {codes, semantic_id}
        - semantic_to_item: Dict mapping semantic_id -> item_id
        - config: RQ-VAE config info
    """
    from src.rqvae.dataset import ItemEmbeddingDataset

    # Load catalogue and generate embeddings
    dataset = ItemEmbeddingDataset.from_catalogue(
        catalogue_path=catalogue_path,
        embedding_model=embedding_model,
        id_field=id_field,
        cache_path=embeddings_cache_path,
        batch_size=embedding_batch_size,
    )

    # Generate semantic IDs
    device = next(rqvae_model.parameters()).device
    embeddings = dataset.embeddings.to(device)

    with torch.no_grad():
        indices = rqvae_model.get_semantic_ids(embeddings)
        semantic_strings = rqvae_model.semantic_id_to_string(indices)

    # Build mapping
    item_to_semantic = {}
    semantic_to_item = {}

    for i, item_id in enumerate(dataset.item_ids):
        item_id_str = str(item_id)
        sem_id = semantic_strings[i]
        item_to_semantic[item_id_str] = {
            "codes": indices[i].cpu().tolist(),
            "semantic_id": sem_id,
        }
        semantic_to_item[sem_id] = item_id_str

    mapping = {
        "item_to_semantic": item_to_semantic,
        "semantic_to_item": semantic_to_item,
        "config": {
            "num_quantizers": rqvae_model.config.num_quantizers,
            "codebook_size": rqvae_model.config.codebook_size,
        },
    }

    # Compute collision stats
    unique_ids = len(semantic_to_item)
    total_items = len(item_to_semantic)
    collision_rate = 1 - unique_ids / total_items if total_items > 0 else 0

    print("Created semantic ID mapping:")
    print(f"  Total items: {total_items}")
    print(f"  Unique IDs: {unique_ids}")
    print(f"  Collision rate: {collision_rate * 100:.2f}%")

    # Save if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(mapping, f, indent=2)
        print(f"  Saved to: {output_path}")

    return mapping


def _get_default_query_templates() -> dict[str, list[str]]:
    """Get default query templates for training data generation."""
    return {
        "predict_semantic_id": [
            "{title}",
            "Find: {title}",
            "Search for {title}",
            "Recommend: {title}",
            "Show me {title}",
        ],
        "predict_attribute": [
            "What is the {field_name} for {semantic_id}?",
            "Get {field_name} for {semantic_id}",
            "{semantic_id} - what is the {field_name}?",
        ],
    }


def train(config: LLMTrainConfig) -> LLMTrainResult:
    """
    End-to-end LLM training function.

    This function handles the complete training pipeline:
    1. Initialize W&B (if configured)
    2. Load RQ-VAE model (from wandb artifact or local path)
    3. Create semantic ID mapping for catalogue items
    4. Prepare training data
    5. Train the LLM (stage 1 or stage 2)
    6. Log artifacts to W&B (if configured)
    7. Clean up

    Args:
        config: LLMTrainConfig with all parameters

    Returns:
        LLMTrainResult with trained model, tokenizer, and mappings

    Example:
        >>> config = LLMTrainConfig(
        ...     wandb_rqvae_artifact="rqvae-model:latest",
        ...     catalogue_path="data/catalogue.jsonl",
        ...     stage=1,
        ...     num_train_epochs=3,
        ... )
        >>> result = train(config)
    """
    from .data import (
        format_as_messages,
        generate_training_examples,
        load_catalogue_with_semantic_ids,
    )

    wandb_run = None

    try:
        # === 1. Initialize W&B ===
        if config.wandb_project and config.report_to == "wandb":
            try:
                import wandb

                run_name = config.wandb_run_name or f"llm-stage{config.stage}"
                wandb_run = wandb.init(
                    project=config.wandb_project,
                    name=run_name,
                    config=asdict(config),
                )
                print(f"W&B initialized: {wandb_run.url}")
            except ImportError:
                print("wandb not installed, skipping W&B logging")
            except Exception as e:
                print(f"Failed to initialize W&B: {e}")

        # === 2. Load RQ-VAE Model ===
        print("\n=== Loading RQ-VAE Model ===")
        if config.wandb_rqvae_artifact:
            rqvae_project = config.wandb_rqvae_project or config.wandb_project
            rqvae_model, rqvae_config = load_rqvae_from_wandb(
                artifact_name=config.wandb_rqvae_artifact,
                project=rqvae_project,
            )
        elif config.rqvae_model_path:
            rqvae_model, rqvae_config = load_rqvae_from_path(config.rqvae_model_path)
        else:
            raise ValueError(
                "Must specify either wandb_rqvae_artifact or rqvae_model_path"
            )

        num_quantizers = rqvae_config["num_quantizers"]
        codebook_size = rqvae_config["codebook_size"]

        # Move model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rqvae_model = rqvae_model.to(device)

        # === 3. Create Semantic ID Mapping ===
        print("\n=== Creating Semantic ID Mapping ===")
        semantic_mapping = create_semantic_id_mapping(
            rqvae_model=rqvae_model,
            catalogue_path=config.catalogue_path,
            embedding_model=config.embedding_model,
            id_field=config.catalogue_id_field,
            embeddings_cache_path=config.embeddings_cache_path,
            output_path=config.semantic_ids_output_path,
            embedding_batch_size=config.embedding_batch_size,
        )

        # Free RQ-VAE model memory before LLM training
        del rqvae_model
        torch.cuda.empty_cache()
        print("Freed RQ-VAE model memory")

        # === 4. Prepare Training Data ===
        print("\n=== Preparing Training Data ===")

        # Load catalogue with semantic IDs
        items_dataset = load_catalogue_with_semantic_ids(
            catalogue_path=config.catalogue_path,
            semantic_ids_path=config.semantic_ids_output_path,
            strict=True,
            id_field=config.catalogue_id_field,
        )
        print(f"Loaded {len(items_dataset)} items with semantic IDs")

        # Use provided templates or defaults
        query_templates = config.query_templates or _get_default_query_templates()

        # Generate training examples
        examples = generate_training_examples(
            items_dataset,
            query_templates=query_templates,
            field_mapping=config.field_mapping,
            num_examples_per_item=config.num_examples_per_item,
            id_field=config.catalogue_id_field,
            predict_semantic_id_ratio=config.predict_semantic_id_ratio,
        )
        print(f"Generated {len(examples)} training examples")

        # Format as messages
        formatted = format_as_messages(examples, system_prompt=config.system_prompt)

        # Split into train/val
        formatted = formatted.shuffle(seed=config.seed)
        split = formatted.train_test_split(test_size=config.val_split, seed=config.seed)
        train_dataset = split["train"]
        val_dataset = split["test"]

        print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

        # === 5. Handle Stage 2 Checkpoint ===
        stage1_checkpoint = config.stage1_checkpoint
        if config.stage == 2 and config.wandb_stage1_artifact and not stage1_checkpoint:
            # Download stage 1 artifact
            import wandb

            # Build full artifact path (similar to load_rqvae_from_wandb)
            artifact_path = config.wandb_stage1_artifact
            if config.wandb_project and "/" not in artifact_path.split(":")[0]:
                # Add project prefix if not already fully qualified
                artifact_path = f"{config.wandb_project}/{artifact_path}"

            if wandb.run is not None:
                artifact = wandb.run.use_artifact(artifact_path, type="model")
            else:
                api = wandb.Api()
                artifact = api.artifact(artifact_path, type="model")
            stage1_checkpoint = artifact.download()
            print(f"Downloaded stage 1 checkpoint from: {artifact_path}")

        # === 6. Build Finetune Config ===
        finetune_config = config.to_finetune_config()
        finetune_config.num_quantizers = num_quantizers
        finetune_config.codebook_size = codebook_size
        finetune_config.stage1_checkpoint = stage1_checkpoint

        # Build semantic_id_to_item mapping for recommendation callback
        if config.recommendation_test_queries:
            # Load full catalogue items for metadata
            import json as json_module

            semantic_id_to_item = {}
            with open(config.catalogue_path) as f:
                for line in f:
                    item = json_module.loads(line)
                    item_id = str(item.get(config.catalogue_id_field, ""))
                    if item_id in semantic_mapping["item_to_semantic"]:
                        sem_id = semantic_mapping["item_to_semantic"][item_id][
                            "semantic_id"
                        ]
                        semantic_id_to_item[sem_id] = item

            finetune_config.semantic_id_to_item = semantic_id_to_item

        # === 7. Train ===
        print(f"\n=== Training Stage {config.stage} ===")
        model, tokenizer = finetune_model(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            config=finetune_config,
        )

        print("\nTraining complete!")

        return LLMTrainResult(
            model=model,
            tokenizer=tokenizer,
            semantic_id_mapping=semantic_mapping,
            config=config,
        )

    finally:
        # === 8. Clean up W&B ===
        if wandb_run:
            import wandb

            wandb.finish()
            print("W&B run finished")
