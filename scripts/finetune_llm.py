"""
Fine-tuning script for LLM on semantic IDs.

Usage:
    python -m scripts.finetune_llm --config config.yaml
"""

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from src.llm.data import prepare_training_data
from src.llm.finetune import finetune_model


def main():
    parser = argparse.ArgumentParser(description="Fine-tune LLM for semantic ID generation")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--catalogue",
        type=str,
        help="Path to catalogue (overrides config)",
    )
    parser.add_argument(
        "--semantic-ids",
        type=str,
        help="Path to semantic IDs (overrides config)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory (overrides config)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        help="Base model (overrides config)",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push model to HuggingFace Hub",
    )
    parser.add_argument(
        "--hub-repo",
        type=str,
        help="HuggingFace Hub repo name",
    )
    parser.add_argument(
        "--skip-data-prep",
        action="store_true",
        help="Skip data preparation (use existing train/val files)",
    )
    args = parser.parse_args()

    # Load config
    config = OmegaConf.load(args.config)

    # Override with CLI args
    if args.catalogue:
        config.data.catalogue_path = args.catalogue
    if args.semantic_ids:
        config.data.semantic_ids_path = args.semantic_ids
    if args.output_dir:
        config.output.llm_checkpoint = args.output_dir
    if args.base_model:
        config.llm.base_model = args.base_model
    if args.hub_repo:
        config.output.hf_repo = args.hub_repo

    # Check prerequisites
    if not Path(config.data.semantic_ids_path).exists():
        print(f"Semantic IDs not found at {config.data.semantic_ids_path}")
        print("Run train_rqvae.py first to generate semantic IDs")
        return

    if not Path(config.data.catalogue_path).exists():
        print(f"Catalogue not found at {config.data.catalogue_path}")
        return

    # Prepare training data
    if not args.skip_data_prep:
        print("Preparing training data...")
        train_dataset, val_dataset = prepare_training_data(
            catalogue_path=config.data.catalogue_path,
            semantic_ids_path=config.data.semantic_ids_path,
            output_train_path=config.data.train_data_path,
            output_val_path=config.data.val_data_path,
            num_examples_per_item=3,
            val_split=0.1,
        )
    else:
        print("Loading existing training data...")
        from datasets import load_dataset
        train_dataset = load_dataset("json", data_files=config.data.train_data_path)["train"]
        val_dataset = load_dataset("json", data_files=config.data.val_data_path)["train"]

    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    # Fine-tune
    print(f"Fine-tuning {config.llm.base_model}...")
    model, tokenizer = finetune_model(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        base_model=config.llm.base_model,
        output_dir=config.output.llm_checkpoint,
        num_quantizers=config.llm.num_codebooks,
        codebook_size=config.llm.codebook_size,
        max_seq_length=config.llm.max_seq_length,
        lora_r=config.llm.lora_r,
        lora_alpha=config.llm.lora_alpha,
        lora_dropout=config.llm.lora_dropout,
        learning_rate=config.llm.learning_rate,
        batch_size=config.llm.batch_size,
        gradient_accumulation_steps=config.llm.gradient_accumulation_steps,
        max_epochs=config.llm.max_epochs,
        warmup_ratio=config.llm.warmup_ratio,
        push_to_hub=args.push_to_hub,
        hub_repo=config.output.hf_repo,
    )

    print(f"Model saved to {config.output.llm_checkpoint}")
    if args.push_to_hub and config.output.hf_repo:
        print(f"Model pushed to {config.output.hf_repo}")


if __name__ == "__main__":
    main()
