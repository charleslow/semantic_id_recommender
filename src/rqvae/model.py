"""
RQ-VAE model for learning semantic IDs from item embeddings.

Uses ResidualVQ from vector-quantize-pytorch to learn hierarchical
discrete representations (semantic IDs) for catalogue items.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from vector_quantize_pytorch import ResidualVQ


@dataclass
class SemanticRQVAEConfig:
    """Configuration for SemanticRQVAE model."""

    embedding_dim: int = 384  # Input embedding dimension
    hidden_dim: int = 512  # Hidden layer dimension
    codebook_size: int = 256  # Size of each codebook
    num_quantizers: int = 4  # Number of residual quantization levels
    commitment_weight: float = 0.25  # Commitment loss weight
    decay: float = 0.99  # EMA decay for codebook updates


class SemanticRQVAE(nn.Module):
    """
    RQ-VAE for learning semantic IDs from item embeddings.

    The model encodes item embeddings into a sequence of discrete codes
    (semantic IDs) using residual vector quantization. Each item gets
    assigned `num_quantizers` codes, one from each codebook level.

    Example:
        Item embedding -> [42, 156, 89, 3] (4 codes = semantic ID)
    """

    def __init__(self, config: SemanticRQVAEConfig):
        super().__init__()
        self.config = config

        # Encoder: project input embeddings to hidden dimension
        self.encoder = nn.Sequential(
            nn.Linear(config.embedding_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )

        # Residual Vector Quantizer
        self.rq = ResidualVQ(
            dim=config.hidden_dim,
            codebook_size=config.codebook_size,
            num_quantizers=config.num_quantizers,
            commitment_weight=config.commitment_weight,
            decay=config.decay,
            kmeans_init=True,  # Initialize codebooks with k-means
            threshold_ema_dead_code=2,  # Reset dead codes
        )

        # Decoder: reconstruct original embedding from quantized representation
        self.decoder = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.embedding_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input embeddings to hidden representation."""
        return self.encoder(x)

    def quantize(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize hidden representation using residual VQ.

        Args:
            z: Hidden representation [batch_size, hidden_dim]

        Returns:
            quantized: Quantized representation [batch_size, hidden_dim]
            indices: Codebook indices [batch_size, num_quantizers]
            commit_loss: Commitment loss scalar
        """
        # ResidualVQ expects [batch, seq_len, dim], add seq_len=1
        z = z.unsqueeze(1)
        quantized, indices, commit_loss = self.rq(z)
        # Remove seq_len dimension
        quantized = quantized.squeeze(1)
        indices = indices.squeeze(1)
        # Sum commitment losses across quantizers if tensor
        if commit_loss.dim() > 0:
            commit_loss = commit_loss.sum()
        return quantized, indices, commit_loss

    def decode(self, quantized: torch.Tensor) -> torch.Tensor:
        """Decode quantized representation back to embedding space."""
        return self.decoder(quantized)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass: encode, quantize, decode.

        Args:
            x: Input embeddings [batch_size, embedding_dim]

        Returns:
            reconstructed: Reconstructed embeddings [batch_size, embedding_dim]
            indices: Semantic ID codes [batch_size, num_quantizers]
            recon_loss: Reconstruction loss
            commit_loss: Commitment loss
        """
        # Encode
        z = self.encode(x)

        # Quantize
        quantized, indices, commit_loss = self.quantize(z)

        # Decode
        reconstructed = self.decode(quantized)

        # Reconstruction loss (MSE)
        recon_loss = nn.functional.mse_loss(reconstructed, x)

        return reconstructed, indices, recon_loss, commit_loss

    def get_semantic_ids(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get semantic IDs for input embeddings.

        Args:
            x: Input embeddings [batch_size, embedding_dim]

        Returns:
            indices: Semantic ID codes [batch_size, num_quantizers]
        """
        z = self.encode(x)
        _, indices, _ = self.quantize(z)
        return indices

    def semantic_id_to_string(self, indices: torch.Tensor) -> list[str]:
        """
        Convert semantic ID indices to string format for LLM training.

        Args:
            indices: Semantic ID codes [batch_size, num_quantizers]

        Returns:
            List of semantic ID strings like "[SEM_0_42][SEM_1_156][SEM_2_89][SEM_3_3]"
        """
        batch_size = indices.shape[0]
        results = []
        for i in range(batch_size):
            tokens = []
            for q_idx in range(self.config.num_quantizers):
                code = indices[i, q_idx].item()
                tokens.append(f"[SEM_{q_idx}_{code}]")
            results.append("".join(tokens))
        return results

    @staticmethod
    def string_to_semantic_id(
        s: str, num_quantizers: int = 4
    ) -> list[int] | None:
        """
        Parse semantic ID string back to indices.

        Args:
            s: Semantic ID string like "[SEM_0_42][SEM_1_156][SEM_2_89][SEM_3_3]"
            num_quantizers: Expected number of quantizers

        Returns:
            List of code indices, or None if parsing fails
        """
        import re

        pattern = r"\[SEM_(\d+)_(\d+)\]"
        matches = re.findall(pattern, s)

        if len(matches) != num_quantizers:
            return None

        codes = [None] * num_quantizers
        for q_idx, code in matches:
            q_idx, code = int(q_idx), int(code)
            if q_idx >= num_quantizers:
                return None
            codes[q_idx] = code

        if None in codes:
            return None

        return codes
