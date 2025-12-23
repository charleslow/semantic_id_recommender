"""
PyTorch Lightning trainer for RQ-VAE model.
"""

import lightning as L
import torch
from torch.utils.data import DataLoader

from .model import SemanticRQVAE, SemanticRQVAEConfig


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

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        embeddings = batch
        _, _, recon_loss, commit_loss = self.model(embeddings)

        loss = recon_loss + commit_loss

        self.log("train/recon_loss", recon_loss, prog_bar=True)
        self.log("train/commit_loss", commit_loss, prog_bar=True)
        self.log("train/loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        embeddings = batch
        _, indices, recon_loss, commit_loss = self.model(embeddings)

        loss = recon_loss + commit_loss

        # Log codebook utilization
        unique_codes = torch.unique(indices, dim=0).shape[0]
        total_items = indices.shape[0]
        utilization = unique_codes / total_items

        self.log("val/recon_loss", recon_loss, prog_bar=True)
        self.log("val/commit_loss", commit_loss)
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/codebook_utilization", utilization)

        return loss

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
                semantic_str = self.model.semantic_id_to_string(
                    torch.tensor([codes])
                )[0]
                mapping[str(item_id)] = {
                    "codes": codes,
                    "semantic_id": semantic_str,
                }

        # Also create reverse mapping (semantic_id -> item_id)
        reverse_mapping = {
            v["semantic_id"]: str(k) for k, v in mapping.items()
        }

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
