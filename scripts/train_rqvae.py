"""
Training script for RQ-VAE model.

Usage:
    python -m scripts.train_rqvae --config src/config/default.yaml
    python -m scripts.train_rqvae --catalogue data/catalogue.json
"""

import argparse
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, random_split

from src.rqvae.model import SemanticRQVAEConfig
from src.rqvae.trainer import RQVAETrainer
from src.rqvae.dataset import ItemEmbeddingDataset, create_dummy_catalogue


def main():
    parser = argparse.ArgumentParser(description="Train RQ-VAE for semantic IDs")
    parser.add_argument(
        "--config",
        type=str,
        default="src/config/default.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--catalogue",
        type=str,
        help="Path to catalogue (overrides config)",
    )
    parser.add_argument(
        "--embeddings",
        type=str,
        help="Path to pre-computed embeddings (overrides config)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory (overrides config)",
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

    # Load config
    config = OmegaConf.load(args.config)

    # Override with CLI args
    if args.catalogue:
        config.data.catalogue_path = args.catalogue
    if args.embeddings:
        config.data.embeddings_path = args.embeddings
    if args.output_dir:
        config.output.rqvae_checkpoint = args.output_dir

    # Create dummy catalogue if requested
    if args.create_dummy:
        create_dummy_catalogue(args.dummy_size, config.data.catalogue_path)

    # Check if catalogue exists
    if not Path(config.data.catalogue_path).exists():
        print(f"Catalogue not found at {config.data.catalogue_path}")
        print("Use --create-dummy to create a test catalogue")
        return

    # Load or create dataset
    if Path(config.data.embeddings_path).exists():
        print(f"Loading embeddings from {config.data.embeddings_path}")
        dataset = ItemEmbeddingDataset.from_embeddings_file(config.data.embeddings_path)
    else:
        print(f"Generating embeddings from {config.data.catalogue_path}")
        dataset = ItemEmbeddingDataset.from_catalogue(
            catalogue_path=config.data.catalogue_path,
            embedding_model=config.rqvae.embedding_model,
            cache_path=config.data.embeddings_path,
        )

    print(f"Dataset size: {len(dataset)}")

    # Split into train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.rqvae.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.rqvae.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Create model config
    model_config = SemanticRQVAEConfig(
        embedding_dim=config.rqvae.embedding_dim,
        hidden_dim=config.rqvae.hidden_dim,
        codebook_size=config.rqvae.codebook_size,
        num_quantizers=config.rqvae.num_quantizers,
        commitment_weight=config.rqvae.commitment_weight,
        decay=config.rqvae.decay,
    )

    # Create trainer module
    model = RQVAETrainer(
        config=model_config,
        learning_rate=config.rqvae.learning_rate,
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.output.rqvae_checkpoint,
        filename="rqvae-{epoch:02d}-{val_loss:.4f}",
        save_top_k=3,
        monitor="val/loss",
        mode="min",
    )

    early_stopping = EarlyStopping(
        monitor="val/loss",
        patience=10,
        mode="min",
    )

    # Logger
    logger = None
    if config.logging.wandb_project:
        logger = WandbLogger(
            project=config.logging.wandb_project,
            entity=config.logging.wandb_entity,
            name="rqvae-training",
        )

    # Create Lightning trainer
    trainer = L.Trainer(
        max_epochs=config.rqvae.max_epochs,
        callbacks=[checkpoint_callback, early_stopping],
        logger=logger,
        log_every_n_steps=config.logging.log_every_n_steps,
        accelerator="auto",
        devices=1,
        precision="16-mixed",
    )

    # Train
    trainer.fit(model, train_loader, val_loader)

    # Save semantic ID mapping
    print("Generating semantic ID mapping...")
    full_loader = DataLoader(
        dataset,
        batch_size=config.rqvae.batch_size,
        shuffle=False,
        num_workers=4,
    )

    model.save_semantic_id_mapping(
        dataloader=full_loader,
        item_ids=dataset.item_ids,
        output_path=config.data.semantic_ids_path,
    )

    print(f"Saved semantic IDs to {config.data.semantic_ids_path}")
    print(f"Checkpoints saved to {config.output.rqvae_checkpoint}")


if __name__ == "__main__":
    main()
