"""
LLM fine-tuning script using Unsloth.

Fine-tunes a base model to generate semantic IDs from user queries.
"""

from pathlib import Path

from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments

from .data import get_semantic_id_tokens


def add_semantic_tokens(tokenizer, num_quantizers: int = 4, codebook_size: int = 256):
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


def formatting_prompts_func(examples):
    """Format examples for training using chat template."""
    output_texts = []
    for messages in examples["messages"]:
        text = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                text += f"<|system|>\n{content}\n"
            elif role == "user":
                text += f"<|user|>\n{content}\n"
            elif role == "assistant":
                text += f"<|assistant|>\n{content}\n"
        output_texts.append(text)
    return output_texts


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
):
    """
    Fine-tune LLM using Unsloth for semantic ID generation.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset (optional)
        base_model: Base model name (Unsloth-compatible)
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

    # Add semantic ID tokens
    tokenizer = add_semantic_tokens(tokenizer, num_quantizers, codebook_size)
    model.resize_token_embeddings(len(tokenizer))

    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",  # Memory efficient
        random_state=42,
    )

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
        fp16=True,
        optim="adamw_8bit",
        report_to="wandb",
        push_to_hub=push_to_hub,
        hub_model_id=hub_repo,
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        formatting_func=formatting_prompts_func,
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


def load_finetuned_model(
    model_path: str,
    max_seq_length: int = 512,
    load_in_4bit: bool = True,
):
    """
    Load fine-tuned model for inference.

    Args:
        model_path: Path to saved model or HuggingFace repo
        max_seq_length: Maximum sequence length
        load_in_4bit: Whether to load in 4-bit

    Returns:
        Model and tokenizer
    """
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=load_in_4bit,
    )

    # Set to inference mode
    FastLanguageModel.for_inference(model)

    return model, tokenizer


def generate_semantic_id(
    model,
    tokenizer,
    query: str,
    max_new_tokens: int = 32,
    temperature: float = 0.1,
) -> str:
    """
    Generate semantic ID for a query.

    Args:
        model: Fine-tuned model
        tokenizer: Tokenizer
        query: User query
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        Generated semantic ID string
    """
    system_prompt = (
        "You are a recommendation system. Given a user query, "
        "output the semantic ID of the most relevant item. "
        "Respond only with the semantic ID tokens."
    )

    prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{query}\n<|assistant|>\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
        pad_token_id=tokenizer.pad_token_id,
    )

    # Decode and extract semantic ID
    generated = tokenizer.decode(outputs[0], skip_special_tokens=False)
    # Extract part after <|assistant|>
    if "<|assistant|>" in generated:
        semantic_id = generated.split("<|assistant|>")[-1].strip()
    else:
        semantic_id = generated[len(prompt):].strip()

    return semantic_id
