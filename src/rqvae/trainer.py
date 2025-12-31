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

    # Training hyperparameters
    learning_rate: float = 1e-3
    max_epochs: int = 500
    batch_size: int = 512
    train_split: float = 0.95

    # W&B artifact logging
    log_wandb_artifacts: bool = False
    artifact_name: str = "rqvae-model"

    # Output paths
    model_save_path: str = "models/rqvae_model.pt"
    semantic_ids_path: str = "data/semantic_ids.json"

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

    def save_semantic_id_mapping(
        self,
        dataloader: DataLoader,
        item_ids: list,
        output_path: str,
    ) -> None:
        """
        Save semantic ID mapping to JSON file.

        Args:
            dataloader: DataLoader with item embeddings
            item_ids: List of original item IDs (same order as dataloader)
            output_path: Path to save JSON file
        """
        import json

        semantic_ids = self.get_semantic_ids(dataloader)

        # Map original item IDs to semantic IDs
        mapping = {}
        for idx, item_id in enumerate(item_ids):
            if idx in semantic_ids:
                codes = semantic_ids[idx]
                # Create both directions of mapping
                semantic_str = self.model.semantic_id_to_string(torch.tensor([codes]))[
                    0
                ]
                mapping[str(item_id)] = {
                    "codes": codes,
                    "semantic_id": semantic_str,
                }

        # Also create reverse mapping (semantic_id -> item_id)
        reverse_mapping = {v["semantic_id"]: str(k) for k, v in mapping.items()}

        output = {
            "item_to_semantic": mapping,
            "semantic_to_item": reverse_mapping,
            "config": {
                "num_quantizers": self.config.num_quantizers,
                "codebook_size": self.config.codebook_size,
            },
        }

        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)


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
        semantic_ids_path = Path(self.train_config.semantic_ids_path)

        # Ensure parent directories exist
        model_save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save the model checkpoint
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
        }
        torch.save(checkpoint, model_save_path)

        # Build artifact metadata from train_config
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
        if semantic_ids_path.exists():
            artifact.add_file(str(semantic_ids_path))
        wandb.log_artifact(artifact, aliases=["latest", "best"])
        self._artifact_logged = True
