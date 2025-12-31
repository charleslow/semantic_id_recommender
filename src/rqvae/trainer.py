"""
PyTorch Lightning trainer for RQ-VAE model.
"""

from dataclasses import asdict, dataclass
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback
from torch.utils.data import DataLoader

from .model import SemanticRQVAE, SemanticRQVAEConfig


@dataclass
class RqvaeTrainConfig(SemanticRQVAEConfig):
    """Configuration for RQ-VAE training. Extends SemanticRQVAEConfig with training params."""

    # Data configuration
    catalogue_path: str | None = None
    embeddings_cache_path: str | None = None
    catalogue_fields: list[str] | None = None
    catalogue_id_field: str = "id"
    embedding_model: str = "TaylorAI/gte-tiny"

    # Training hyperparameters
    learning_rate: float = 1e-3
    max_epochs: int = 500
    batch_size: int = 512
    train_split: float = 0.9
    num_workers: int = 16

    # W&B configuration
    wandb_project: str | None = "semantic-id-recommender"
    wandb_run_name: str | None = None
    log_wandb_artifacts: bool = False
    artifact_name: str = "rqvae-model"

    # Output paths
    model_save_path: str = "models/rqvae_model.pt"

    def to_model_config(self) -> SemanticRQVAEConfig:
        """Convert to SemanticRQVAEConfig for model instantiation."""
        return SemanticRQVAEConfig(
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            codebook_size=self.codebook_size,
            num_quantizers=self.num_quantizers,
            commitment_weight=self.commitment_weight,
            decay=self.decay,
            threshold_ema_dead_code=self.threshold_ema_dead_code,
        )


class RQVAETrainer(L.LightningModule):
    """Lightning module for training SemanticRQVAE."""

    def __init__(
        self,
        config: SemanticRQVAEConfig,
        learning_rate: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["config"])
        self.config = config
        self.learning_rate = learning_rate
        self.model = SemanticRQVAE(config)

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def _shared_step(self, batch: torch.Tensor, stage: str) -> torch.Tensor:
        """
        Shared step logic for training and validation.

        Args:
            batch: Input embeddings
            stage: Either "train" or "val"

        Returns:
            Total loss (reconstruction + commitment)
        """
        embeddings = batch
        _, indices, recon_loss, commit_loss = self.model(embeddings)

        loss = recon_loss + commit_loss

        # Log losses
        self.log(f"{stage}/recon_loss", recon_loss, prog_bar=True)
        self.log(f"{stage}/commit_loss", commit_loss, prog_bar=stage == "train")
        self.log(f"{stage}/loss", loss, prog_bar=True)

        # Track codebook usage and perplexity
        codebook_stats = self.model.compute_codebook_stats(indices)
        self.log(
            f"{stage}/avg_perplexity",
            codebook_stats["avg_perplexity"],
            prog_bar=stage == "val",
        )
        self.log(f"{stage}/avg_codebook_usage", codebook_stats["avg_usage"])

        # Log per quantizer level stats
        for q_idx in range(self.config.num_quantizers):
            self.log(
                f"{stage}/perplexity_level_{q_idx}",
                codebook_stats["perplexity_per_level"][q_idx],
            )
            self.log(
                f"{stage}/usage_level_{q_idx}",
                codebook_stats["usage_per_level"][q_idx],
            )

        return loss

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:  # noqa: ARG002
        return self._shared_step(batch, "train")

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:  # noqa: ARG002
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=self.learning_rate * 0.1,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    def get_semantic_ids(self, dataloader: DataLoader) -> dict[int, list[int]]:
        """
        Extract semantic IDs for all items in the dataset.

        Args:
            dataloader: DataLoader with item embeddings

        Returns:
            Dictionary mapping item index to semantic ID codes
        """
        self.model.eval()
        semantic_ids = {}
        idx = 0

        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                indices = self.model.get_semantic_ids(batch)

                for i in range(indices.shape[0]):
                    semantic_ids[idx] = indices[i].cpu().tolist()
                    idx += 1

        return semantic_ids


class WandbArtifactCallback(Callback):
    """
    Lightning callback to log RQ-VAE model as W&B artifact at the end of training.
    """

    def __init__(
        self,
        train_config: RqvaeTrainConfig,
        embedding_model: str | None = None,
        extra_metadata: dict | None = None,
    ):
        """
        Initialize the callback.

        Args:
            train_config: Training configuration
            embedding_model: Name of the embedding model used
            extra_metadata: Additional metadata to include in the artifact
        """
        self.train_config = train_config
        self.embedding_model = embedding_model
        self.extra_metadata = extra_metadata or {}
        self._artifact_logged = False

    def on_fit_end(self, trainer: L.Trainer, pl_module: RQVAETrainer) -> None:
        """Log model as W&B artifact after training completes."""
        if self._artifact_logged or not self.train_config.log_wandb_artifacts:
            return

        try:
            import wandb
        except ImportError:
            return

        if wandb.run is None:
            return

        model = pl_module.model
        config = pl_module.config
        model_save_path = Path(self.train_config.model_save_path)

        # Ensure parent directory exists
        model_save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save the model checkpoint
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
        }
        torch.save(checkpoint, model_save_path)

        # Build artifact metadata
        metadata = asdict(self.train_config)
        if self.embedding_model:
            metadata["embedding_model"] = self.embedding_model
        metadata.update(self.extra_metadata)

        # Create and log artifact
        artifact = wandb.Artifact(
            name=self.train_config.artifact_name,
            type="model",
            description="RQ-VAE model for semantic ID generation",
            metadata=metadata,
        )
        artifact.add_file(str(model_save_path))
        wandb.log_artifact(artifact, aliases=["latest", "best"])
        self._artifact_logged = True


@dataclass
class TrainResult:
    """Result of training containing the trained model and metrics."""

    model: SemanticRQVAE
    config: SemanticRQVAEConfig
    metrics: dict
    semantic_ids: dict[str, str] | None = None


def train(config: RqvaeTrainConfig) -> TrainResult:
    """
    End-to-end training function for RQ-VAE.

    Handles the complete training lifecycle:
    1. Initialize W&B (if project provided)
    2. Load catalogue and generate/cache embeddings
    3. Split dataset into train/val
    4. Train the model with Lightning
    5. Evaluate and compute final metrics
    6. Log summary metrics to W&B
    7. Clean up W&B run

    Args:
        config: Training configuration containing all parameters

    Returns:
        TrainResult containing:
        - model: Trained SemanticRQVAE model
        - config: Model configuration
        - metrics: Dictionary of final evaluation metrics
        - semantic_ids: Mapping of item_id -> semantic_id string (if catalogue has IDs)

    Example:
        >>> from src.rqvae import train, RqvaeTrainConfig
        >>> config = RqvaeTrainConfig(
        ...     catalogue_path="data/catalogue.jsonl",
        ...     embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        ...     codebook_size=256,
        ...     num_quantizers=4,
        ...     max_epochs=100,
        ...     wandb_project="my-project",
        ... )
        >>> result = train(config)
        >>> print(f"Collision rate: {result.metrics['collision_rate']:.2%}")
    """
    import lightning as L
    from lightning.pytorch.loggers import WandbLogger
    from torch.utils.data import DataLoader, random_split

    from .dataset import ItemEmbeddingDataset

    if config.catalogue_path is None:
        raise ValueError("catalogue_path must be set in config")

    catalogue_path = Path(config.catalogue_path)

    # --- 1. Initialize W&B ---
    wandb_logger = None
    wandb_run = None

    if config.wandb_project:
        try:
            import wandb

            run_name = (
                config.wandb_run_name
                or f"rqvae_{config.codebook_size}x{config.num_quantizers}"
            )

            wandb_run = wandb.init(
                project=config.wandb_project,
                name=run_name,
                config={
                    "embedding_model": config.embedding_model,
                    "embedding_dim": config.embedding_dim,
                    "hidden_dim": config.hidden_dim,
                    "codebook_size": config.codebook_size,
                    "num_quantizers": config.num_quantizers,
                    "commitment_weight": config.commitment_weight,
                    "learning_rate": config.learning_rate,
                    "max_epochs": config.max_epochs,
                    "batch_size": config.batch_size,
                    "train_split": config.train_split,
                    "catalogue_path": str(catalogue_path),
                },
            )

            wandb_logger = WandbLogger(
                project=config.wandb_project,
                log_model=False,
            )

            print(f"W&B initialized: {wandb_run.url}")
        except ImportError:
            print("wandb not installed, skipping W&B logging")
        except Exception as e:
            print(f"Failed to initialize W&B: {e}")

    try:
        # --- 2. Load catalogue and generate embeddings ---
        print(f"Loading catalogue from {catalogue_path}...")
        dataset = ItemEmbeddingDataset.from_catalogue(
            catalogue_path=catalogue_path,
            embedding_model=config.embedding_model,
            fields=config.catalogue_fields,
            id_field=config.catalogue_id_field,
            cache_path=config.embeddings_cache_path,
            batch_size=32,
        )
        embeddings = dataset.embeddings

        # Update config embedding_dim if it doesn't match
        actual_embedding_dim = embeddings.shape[1]
        if config.embedding_dim != actual_embedding_dim:
            print(
                f"Updating embedding_dim from {config.embedding_dim} to {actual_embedding_dim}"
            )
            config.embedding_dim = actual_embedding_dim

        print(f"Dataset: {len(dataset)} items, {actual_embedding_dim}-dim embeddings")

        # --- 3. Split dataset ---
        train_size = int(config.train_split * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            shuffle=True,
            persistent_workers=config.num_workers > 0,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            shuffle=False,
            persistent_workers=config.num_workers > 0,
        )

        print(f"Train/Val split: {len(train_dataset)}/{len(val_dataset)}")

        # --- 4. Create model and train ---
        model_config = config.to_model_config()
        trainer_module = RQVAETrainer(
            config=model_config,
            learning_rate=config.learning_rate,
        )

        print(
            f"Model: {config.num_quantizers} quantizers x {config.codebook_size} codes "
            f"= {config.codebook_size**config.num_quantizers:,} possible IDs"
        )

        # Build callbacks
        callbacks = []
        if config.log_wandb_artifacts and wandb_run:
            callbacks.append(
                WandbArtifactCallback(
                    train_config=config,
                    embedding_model=config.embedding_model,
                )
            )

        # Create Lightning trainer
        trainer = L.Trainer(
            max_epochs=config.max_epochs,
            accelerator="auto",
            devices=1,
            enable_progress_bar=True,
            log_every_n_steps=5,
            logger=wandb_logger,
            callbacks=callbacks,
        )

        print("Starting training...")
        trainer.fit(trainer_module, train_loader, val_loader)
        print("Training complete!")

        # --- 5. Evaluate on full dataset ---
        model = trainer_module.model
        model.eval()
        device = next(model.parameters()).device

        with torch.no_grad():
            all_indices = model.get_semantic_ids(embeddings.to(device))
            stats = model.compute_codebook_stats(all_indices)

        # Compute collision rate
        semantic_id_strings = model.semantic_id_to_string(all_indices)
        unique_ids = len(set(semantic_id_strings))
        collision_rate = 1 - unique_ids / len(embeddings)

        # Build semantic ID mapping
        semantic_ids_map = {
            str(item_id): sem_id
            for item_id, sem_id in zip(dataset.item_ids, semantic_id_strings)
        }

        # Build metrics dict
        metrics = {
            "avg_perplexity": stats["avg_perplexity"].item(),
            "avg_usage": stats["avg_usage"].item(),
            "total_items": len(embeddings),
            "unique_semantic_ids": unique_ids,
            "collision_rate": collision_rate,
        }

        # Add per-level metrics
        for q in range(config.num_quantizers):
            metrics[f"level_{q}_perplexity"] = stats["perplexity_per_level"][q].item()
            metrics[f"level_{q}_usage"] = stats["usage_per_level"][q].item()

        print("\n=== Evaluation Results ===")
        print(
            f"Avg perplexity: {metrics['avg_perplexity']:.2f} / {config.codebook_size}"
        )
        print(f"Avg usage: {metrics['avg_usage'] * 100:.1f}%")
        print(f"Unique IDs: {unique_ids} / {len(embeddings)}")
        print(f"Collision rate: {collision_rate * 100:.2f}%")

        # --- 6. Log summary metrics to W&B ---
        if wandb_run:
            import wandb

            # Log final evaluation metrics
            wandb.log({f"eval/{k}": v for k, v in metrics.items()})

            # Set summary metrics (shown in W&B run overview)
            wandb.summary["final_avg_perplexity"] = metrics["avg_perplexity"]
            wandb.summary["final_avg_usage"] = metrics["avg_usage"]
            wandb.summary["final_collision_rate"] = metrics["collision_rate"]
            wandb.summary["final_unique_ids"] = metrics["unique_semantic_ids"]
            wandb.summary["total_items"] = metrics["total_items"]

            for q in range(config.num_quantizers):
                wandb.summary[f"final_level_{q}_perplexity"] = metrics[
                    f"level_{q}_perplexity"
                ]
                wandb.summary[f"final_level_{q}_usage"] = metrics[f"level_{q}_usage"]

            print("Summary metrics logged to W&B")

        # Save model locally
        model_save_path = Path(config.model_save_path)
        model_save_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "config": asdict(model_config),
        }
        torch.save(checkpoint, model_save_path)
        print(f"Model saved to {model_save_path}")

        return TrainResult(
            model=model,
            config=model_config,
            metrics=metrics,
            semantic_ids=semantic_ids_map,
        )

    finally:
        # --- 7. Clean up W&B ---
        if wandb_run:
            import wandb

            wandb.finish()
            print("W&B run finished")
