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


class SemanticIDGenerator:
    """Generator for semantic IDs with optional constrained decoding."""

    def __init__(
        self,
        model,
        tokenizer,
        num_quantizers: int = 4,
        codebook_size: int = 256,
        use_constrained_decoding: bool = True,
    ):
        """
        Initialize the generator.

        Args:
            model: Fine-tuned model
            tokenizer: Tokenizer with semantic ID tokens
            num_quantizers: Number of RQ-VAE quantizers
            codebook_size: Size of each codebook
            use_constrained_decoding: Whether to use constrained decoding
        """
        self.model = model
        self.tokenizer = tokenizer
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.use_constrained_decoding = use_constrained_decoding

        # Build semantic token ID lookup
        self.semantic_token_ids = [
            tokenizer.convert_tokens_to_ids(f"[SEM_{q}_{c}]")
            for q in range(num_quantizers)
            for c in range(codebook_size)
        ]
        self.semantic_token_ids_set = set(self.semantic_token_ids)
        self.eos_token_id = tokenizer.eos_token_id

        self.system_prompt = (
            "You are a recommendation system. Given a user query, "
            "output the semantic ID of the most relevant item. "
            "Respond only with the semantic ID tokens."
        )

    def _get_allowed_tokens_for_position(self, position: int) -> list[int]:
        """Return allowed token IDs for each position in the semantic ID."""
        if position < self.num_quantizers:
            # Only allow tokens from the correct quantizer level
            start_idx = position * self.codebook_size
            end_idx = start_idx + self.codebook_size
            return self.semantic_token_ids[start_idx:end_idx]
        else:
            # After all quantizers, only allow EOS
            return [self.eos_token_id]

    def _prefix_allowed_tokens_fn(self, batch_id: int, input_ids) -> list[int]:
        """Constrained decoding: only allow valid semantic ID tokens in sequence."""
        generated = input_ids.tolist()
        sem_count = sum(1 for tok_id in generated if tok_id in self.semantic_token_ids_set)
        return self._get_allowed_tokens_for_position(sem_count)

    def _extract_semantic_id(self, generated_text: str) -> str:
        """Extract semantic ID from generated text."""
        if "<|assistant|>" in generated_text:
            semantic_id = generated_text.split("<|assistant|>")[-1].strip()
        else:
            semantic_id = generated_text.strip()

        # Clean up EOS token if present
        if self.tokenizer.eos_token:
            semantic_id = semantic_id.replace(self.tokenizer.eos_token, "").strip()

        return semantic_id

    def generate(
        self,
        query: str,
        temperature: float = 0.1,
    ) -> str:
        """
        Generate semantic ID for a query.

        Args:
            query: User query
            temperature: Sampling temperature (ignored if constrained)

        Returns:
            Generated semantic ID string
        """
        prompt = f"<|system|>\n{self.system_prompt}\n<|user|>\n{query}\n<|assistant|>\n"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        if self.use_constrained_decoding:
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.num_quantizers + 1,
                do_sample=False,  # Greedy for constrained
                pad_token_id=self.tokenizer.pad_token_id,
                prefix_allowed_tokens_fn=self._prefix_allowed_tokens_fn,
            )
        else:
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=32,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        return self._extract_semantic_id(generated)

    def generate_beam(
        self,
        query: str,
        num_beams: int = 10,
        num_return_sequences: int = 10,
    ) -> list[tuple[str, float]]:
        """
        Generate multiple semantic IDs using beam search.

        Args:
            query: User query
            num_beams: Number of beams for beam search
            num_return_sequences: Number of sequences to return

        Returns:
            List of (semantic_id, score) tuples, sorted by score (highest first)
        """
        prompt = f"<|system|>\n{self.system_prompt}\n<|user|>\n{query}\n<|assistant|>\n"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.num_quantizers + 1,
            num_beams=num_beams,
            num_return_sequences=min(num_return_sequences, num_beams),
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            prefix_allowed_tokens_fn=self._prefix_allowed_tokens_fn if self.use_constrained_decoding else None,
            output_scores=True,
            return_dict_in_generate=True,
        )

        results = []
        sequences = outputs.sequences
        # Compute sequence scores (sum of log probs)
        scores = outputs.sequences_scores if hasattr(outputs, 'sequences_scores') else [0.0] * len(sequences)

        for seq, score in zip(sequences, scores):
            generated = self.tokenizer.decode(seq, skip_special_tokens=False)
            semantic_id = self._extract_semantic_id(generated)
            score_val = score.item() if hasattr(score, 'item') else float(score)
            results.append((semantic_id, score_val))

        # Sort by score (highest first) and deduplicate
        seen = set()
        unique_results = []
        for sem_id, score in sorted(results, key=lambda x: x[1], reverse=True):
            if sem_id not in seen:
                seen.add(sem_id)
                unique_results.append((sem_id, score))

        return unique_results

    def __call__(self, query: str, **kwargs) -> str:
        """Shorthand for generate()."""
        return self.generate(query, **kwargs)


def generate_semantic_id(
    model,
    tokenizer,
    query: str,
    max_new_tokens: int = 32,
    temperature: float = 0.1,
) -> str:
    """
    Generate semantic ID for a query (simple version without constrained decoding).

    For constrained decoding, use SemanticIDGenerator instead.

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
