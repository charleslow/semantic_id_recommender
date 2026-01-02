"""
Training script for RQ-VAE model.

Usage:
    python -m scripts.train_rqvae --config configs/rqvae_config.yaml
    python -m scripts.train_rqvae --config configs/rqvae_config.yaml --create-dummy --dummy-size 1000
"""

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from src.rqvae.dataset import create_dummy_catalogue
from src.rqvae.trainer import RqvaeTrainConfig, train


def main():
    parser = argparse.ArgumentParser(description="Train RQ-VAE for semantic IDs")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file (e.g., notebooks/rqvae_config.yaml)",
    )
    parser.add_argument(
        "--create-dummy",
        action="store_true",
        help="Create dummy catalogue for testing",
    )
    parser.add_argument(
        "--dummy-size",
        type=int,
        default=1000,
        help="Size of dummy catalogue",
    )
    args = parser.parse_args()

    # Load config from YAML
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        print("Please provide a valid config file path.")
        return

    raw_config = OmegaConf.load(config_path)
    config_dict = OmegaConf.to_container(raw_config, resolve=True)
    config = RqvaeTrainConfig(**config_dict)

    print(f"Loaded config from: {config_path}")

    # Create dummy catalogue if requested
    if args.create_dummy:
        catalogue_path = config.catalogue_path or "data/dummy_catalogue.jsonl"
        create_dummy_catalogue(args.dummy_size, catalogue_path)
        config.catalogue_path = catalogue_path
        print(f"Created dummy catalogue with {args.dummy_size} items at {catalogue_path}")

    # Check if catalogue exists
    if not config.catalogue_path or not Path(config.catalogue_path).exists():
        print(f"Catalogue not found at {config.catalogue_path}")
        print("Use --create-dummy to create a test catalogue")
        return

    # Run training
    print("\n" + "=" * 50)
    print("Starting RQ-VAE Training")
    print("=" * 50)
    print(f"Catalogue: {config.catalogue_path}")
    print(f"Embedding model: {config.embedding_model}")
    print(f"Codebook size: {config.codebook_size}")
    print(f"Num quantizers: {config.num_quantizers}")
    print(f"Max epochs: {config.max_epochs}")
    print(f"W&B project: {config.wandb_project}")
    print("=" * 50 + "\n")

    result = train(config)

    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)
    print(f"Model saved to: {config.model_save_path}")
    print(f"Unique semantic IDs: {result.metrics['unique_semantic_ids']}")
    print(f"Collision rate: {result.metrics['collision_rate'] * 100:.2f}%")
    print(f"Avg perplexity: {result.metrics['avg_perplexity']:.2f}")


if __name__ == "__main__":
    main()
