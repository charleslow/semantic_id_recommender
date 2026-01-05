"""
Training script for LLM fine-tuning on semantic IDs.

Usage:
    # Stage 1 only (embedding training)
    python -m scripts.train_llm --config configs/stage1_config.yaml --stage 1

    # Stage 2 only (LoRA fine-tuning)
    python -m scripts.train_llm --config configs/stage2_config.yaml --stage 2

    # Both stages sequentially
    python -m scripts.train_llm --config configs/stage1_config.yaml --stage 1,2 --stage2-config configs/stage2_config.yaml
"""

import unsloth  # noqa: F401 isort: skip - Must be imported before transformers

import argparse
import gc
from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.llm.finetune import LLMTrainConfig, train


def load_config(config_path: str) -> LLMTrainConfig:
    """Load and validate config from YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw_config = OmegaConf.load(path)
    config_dict = OmegaConf.to_container(raw_config, resolve=True)
    return LLMTrainConfig(**config_dict)


def run_stage(config: LLMTrainConfig, stage: int):
    """Run a single training stage."""
    print("\n" + "=" * 60)
    print(f"Starting LLM Training - Stage {stage}")
    print("=" * 60)
    print(f"Base model: {config.base_model}")
    print(f"Catalogue: {config.catalogue_path}")

    if stage == 1:
        print("Mode: Embedding training (backbone frozen)")
        if config.wandb_rqvae_artifact:
            print(f"RQ-VAE source: W&B artifact '{config.wandb_rqvae_artifact}'")
        else:
            print(f"RQ-VAE source: {config.rqvae_model_path}")
    else:
        print("Mode: LoRA fine-tuning")
        print(f"LoRA rank: {config.lora_r}")
        if config.wandb_stage1_artifact:
            print(f"Stage 1 source: W&B artifact '{config.wandb_stage1_artifact}'")
        else:
            print(f"Stage 1 source: {config.stage1_checkpoint}")

    print(f"Epochs: {config.num_train_epochs}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Batch size: {config.batch_size}")
    print(f"Output dir: {config.output_dir}")
    print(f"W&B project: {config.wandb_project}")
    print("=" * 60 + "\n")

    result = train(config)

    print("\n" + "=" * 60)
    print(f"Stage {stage} Training Complete!")
    print("=" * 60)
    print(f"Model saved to: {config.output_dir}")
    print(f"Semantic IDs saved to: {config.semantic_ids_output_path}")

    return result


def clear_gpu_memory():
    """Clear GPU memory between stages."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("\nCleared GPU memory")


def main():
    parser = argparse.ArgumentParser(
        description="Train LLM for semantic ID generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train stage 1 only
  python -m scripts.train_llm --config notebooks/stage1_config.yaml --stage 1

  # Train stage 2 only
  python -m scripts.train_llm --config notebooks/stage2_config.yaml --stage 2

  # Train both stages sequentially
  python -m scripts.train_llm --config notebooks/stage1_config.yaml --stage 1,2 --stage2-config notebooks/stage2_config.yaml
        """,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file for stage 1 (or the only stage)",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="1",
        help="Stage(s) to run: '1', '2', or '1,2' for both stages",
    )
    parser.add_argument(
        "--stage2-config",
        type=str,
        help="Path to config YAML file for stage 2 (required when --stage=1,2)",
    )
    args = parser.parse_args()

    # Parse stages
    stages = [int(s.strip()) for s in args.stage.split(",")]
    for stage in stages:
        if stage not in [1, 2]:
            parser.error(f"Invalid stage: {stage}. Must be 1 or 2.")

    # Validate stage 2 config if running both stages
    if len(stages) > 1 and 2 in stages:
        if not args.stage2_config:
            parser.error("--stage2-config is required when running --stage=1,2")

    # Load configs
    config1 = load_config(args.config)
    print(f"Loaded config from: {args.config}")

    config2 = None
    if args.stage2_config:
        config2 = load_config(args.stage2_config)
        print(f"Loaded stage 2 config from: {args.stage2_config}")

    # Run stages
    stage1_result = None

    for stage in stages:
        if stage == 1:
            # Ensure config is set to stage 1
            config1.stage = 1
            stage1_result = run_stage(config1, stage=1)

        elif stage == 2:
            # Clear GPU memory between stages
            if stage1_result is not None:
                del stage1_result
                clear_gpu_memory()

            # Use stage 2 config if provided, otherwise modify stage 1 config
            if config2 is not None:
                config2.stage = 2
                # If we just ran stage 1, use its output as stage 2 input
                if 1 in stages and stages.index(1) < stages.index(2):
                    # Override stage1_checkpoint if not using wandb artifact
                    if not config2.wandb_stage1_artifact:
                        config2.stage1_checkpoint = config1.output_dir
                        print(f"Using stage 1 output as stage 2 input: {config1.output_dir}")
                run_stage(config2, stage=2)
            else:
                # Modify config1 for stage 2
                config1.stage = 2
                if stage1_result is not None:
                    config1.stage1_checkpoint = config1.output_dir
                run_stage(config1, stage=2)

    print("\n" + "=" * 60)
    print("All Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
