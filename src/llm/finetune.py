"""
LLM fine-tuning script using Unsloth.

Fine-tunes a base model to generate semantic IDs from user queries.
"""

from typing import Literal

import torch
from datasets import Dataset
from transformers import PreTrainedTokenizerBase, TrainingArguments
from trl import SFTTrainer

from .data import get_semantic_id_tokens


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
    base_model: str = "unsloth/Qwen3-4B",
    output_dir: str = "checkpoints/llm",
    num_quantizers: int = 4,
    codebook_size: int = 256,
    max_seq_length: int = 512,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    learning_rate: float = 2e-4,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    max_epochs: int = 3,
    warmup_ratio: float = 0.03,
    push_to_hub: bool = False,
    hub_repo: str | None = None,
    stage: Literal[1, 2] | None = None,
):
    """
    Fine-tune LLM using Unsloth for semantic ID generation.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset (optional)
        base_model: Base model name (Unsloth-compatible), or path to stage1 checkpoint for stage2
        output_dir: Directory to save checkpoints
        num_quantizers: Number of semantic ID quantizers
        codebook_size: Size of each codebook
        max_seq_length: Maximum sequence length
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        learning_rate: Learning rate
        batch_size: Batch size per device
        gradient_accumulation_steps: Gradient accumulation steps
        max_epochs: Number of training epochs
        warmup_ratio: Warmup ratio
        push_to_hub: Whether to push to HuggingFace Hub
        hub_repo: Hub repository name
        stage: Training stage (1 or 2). Stage 1 freezes all parameters except embeddings
               to train new tokens. Stage 2 unfreezes all parameters for full training.
               If None, trains without freezing (original behavior).

    Returns:
        Trained model and tokenizer
    """
    from unsloth import FastLanguageModel

    # Load model with Unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        dtype=None,  # Auto-detect
        load_in_4bit=True,  # QLoRA
    )

    # Add semantic ID tokens (only for stage 1 or no stage)
    # For stage 2, tokens should already be in the tokenizer from stage 1 checkpoint
    if stage != 2:
        tokenizer = add_semantic_tokens(tokenizer, num_quantizers, codebook_size)
        model.resize_token_embeddings(len(tokenizer))

    # Add LoRA adapters (only for stage 2 or no stage)
    # Stage 1 trains embeddings directly without LoRA
    if stage != 1:
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules="all-linear",
            bias="none",
            use_gradient_checkpointing="unsloth",  # Memory efficient
            random_state=42,
        )

    # Stage 1: Freeze all parameters except input/output embeddings
    if stage == 1:
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

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=max_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if val_dataset else "no",
        bf16=use_bf16,
        fp16=not use_bf16,
        optim="adamw_8bit",
        report_to="wandb",
        push_to_hub=push_to_hub,
        hub_model_id=hub_repo,
    )

    # Create formatting function with tokenizer's chat template
    formatting_func = create_formatting_func(tokenizer)

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        formatting_func=formatting_func,
        max_seq_length=max_seq_length,
        packing=False,
    )

    # Train
    trainer.train()

    # Save
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Push to hub if configured
    if push_to_hub and hub_repo:
        model.push_to_hub(hub_repo)
        tokenizer.push_to_hub(hub_repo)

    return model, tokenizer
