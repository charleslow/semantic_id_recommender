"""
LLM fine-tuning script using Unsloth.

Fine-tunes a base model to generate semantic IDs from user queries.
"""

from dataclasses import dataclass, field
from typing import Literal

import torch
from datasets import Dataset
from transformers import PreTrainedTokenizerBase
from trl import SFTConfig, SFTTrainer

from .callbacks import RecommendationTestCallback, SemanticIDEvalCallback
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
    max_seq_length: int = 512
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


def freeze_backbone(model) -> None:
    """
    Freeze all model parameters except input/output embeddings.

    Used in stage 1 training to only train the new semantic ID token embeddings
    while keeping the pretrained weights frozen.

    Args:
        model: The language model to freeze
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


def create_formatting_func(
    tokenizer: PreTrainedTokenizerBase,
) -> callable:
    """Create a formatting function that uses the tokenizer's chat template."""

    def formatting_prompts_func(examples):
        """Format examples for training using the tokenizer's chat template."""
        output_texts = []
        for messages in examples["messages"]:
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
        max_seq_length=config.max_seq_length,
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
            target_modules="all-linear",
            bias="none",
            use_gradient_checkpointing="unsloth"
            if config.gradient_checkpointing
            else False,
            random_state=config.seed,
        )

    # Stage 1: Freeze all parameters except input/output embeddings
    if config.stage == 1:
        freeze_backbone(model)

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
        # Evaluation
        eval_strategy="steps" if val_dataset else "no",
        eval_steps=config.eval_steps if val_dataset else None,
        metric_for_best_model="eval_loss" if val_dataset else None,
        greater_is_better=False if val_dataset else None,
        load_best_model_at_end=bool(val_dataset),
        # Misc
        seed=config.seed,
        gradient_checkpointing=config.gradient_checkpointing,
        # Sequence length
        max_seq_length=config.max_seq_length,
        packing=False,
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

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
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

    # Log model as W&B artifact
    if config.log_wandb_artifacts and config.report_to == "wandb":
        _log_wandb_artifact(config, train_dataset, val_dataset)

    return model, tokenizer


def _log_wandb_artifact(
    config: FinetuneConfig,
    train_dataset: Dataset,
    val_dataset: Dataset | None,
) -> None:
    """Log the trained model as a W&B artifact."""
    try:
        import wandb

        if wandb.run is None:
            return

        # Generate artifact name if not provided
        artifact_name = config.wandb_artifact_name
        if artifact_name is None:
            artifact_name = f"llm-stage{config.stage}"

        artifact = wandb.Artifact(
            name=artifact_name,
            type="model",
            description=f"LLM Stage {config.stage}: "
            + ("Embedding training" if config.stage == 1 else "LoRA fine-tuning"),
            metadata={
                "base_model": config.base_model,
                "stage": config.stage,
                "epochs": config.num_train_epochs,
                "learning_rate": config.learning_rate,
                "num_quantizers": config.num_quantizers,
                "codebook_size": config.codebook_size,
                "lora_r": config.lora_r if config.stage == 2 else None,
                "lora_alpha": config.lora_alpha if config.stage == 2 else None,
                "train_examples": len(train_dataset),
                "val_examples": len(val_dataset) if val_dataset else 0,
            },
        )
        artifact.add_dir(config.output_dir)

        aliases = ["latest"]
        if config.stage == 2:
            aliases.append("best")

        wandb.log_artifact(artifact, aliases=aliases)
    except ImportError:
        pass
