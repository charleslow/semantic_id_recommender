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
    decay: float = 0.999  # EMA decay for codebook updates
    threshold_ema_dead_code: int = 1  # Reset codes used fewer than this many times


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

        self.rq = ResidualVQ(
            dim=config.hidden_dim,
            codebook_size=config.codebook_size,
            num_quantizers=config.num_quantizers,
            commitment_weight=config.commitment_weight,
            decay=config.decay,
            kmeans_init=True,  # Initialize codebooks with k-means
            threshold_ema_dead_code=config.threshold_ema_dead_code,
            # Follow TIGER's approach of separate codebooks per level
            shared_codebook=False,
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
        commit_loss = commit_loss.sum()  # Sum over quantizer levels and batch
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
        z = self.encode(x)
        quantized, indices, commit_loss = self.quantize(z)
        reconstructed = self.decode(quantized)
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
        indices_list = indices.cpu().tolist()
        results = []
        for codes in indices_list:
            tokens = [f"[SEM_{q_idx}_{code}]" for q_idx, code in enumerate(codes)]
            results.append("".join(tokens))
        return results

    def string_to_semantic_id(self, s: str) -> list[int] | None:
        """
        Parse semantic ID string back to indices.

        Args:
            s: Semantic ID string like "[SEM_0_42][SEM_1_156][SEM_2_89][SEM_3_3]"

        Returns:
            List of code indices, or None if parsing fails
        """
        import re

        pattern = r"\[SEM_(\d+)_(\d+)\]"
        matches = re.findall(pattern, s)

        if len(matches) != self.config.num_quantizers:
            return None

        codes = [None] * self.config.num_quantizers
        for q_idx, code in matches:
            q_idx, code = int(q_idx), int(code)
            if q_idx >= self.config.num_quantizers:
                return None
            codes[q_idx] = code

        if None in codes:
            return None

        return codes

    def compute_codebook_stats(self, indices: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Compute codebook usage statistics and perplexity.

        Perplexity measures how uniformly the codebook is used:
        - Perplexity = exp(entropy) where entropy = -sum(p * log(p))
        - Max perplexity = codebook_size (all codes used equally)
        - Low perplexity indicates codebook collapse (few codes used)

        Args:
            indices: Codebook indices [batch_size, num_quantizers]

        Returns:
            Dictionary with:
            - perplexity_per_level: Perplexity for each quantizer level
            - avg_perplexity: Average perplexity across all levels
            - usage_per_level: Fraction of codes used per level
            - avg_usage: Average usage fraction across all levels
        """
        num_quantizers = indices.shape[1]
        codebook_size = self.config.codebook_size

        perplexities = []
        usage_fractions = []
        for q_idx in range(num_quantizers):
            level_indices = indices[:, q_idx]
            counts = torch.bincount(
                level_indices.view(-1), minlength=codebook_size
            ).float()
            probs = counts / counts.sum()
            nonzero_probs = probs[probs > 0]

            # Compute entropy: H = -sum(p * log(p))
            # It is standard to ignore zero probabilities in entropy calculation
            entropy = -(nonzero_probs * torch.log(nonzero_probs)).sum()
            # Perplexity = exp(entropy)
            perplexity = torch.exp(entropy)
            perplexities.append(perplexity)

            # Usage: fraction of codes that are used at least once in this batch
            codes_used = (counts > 0).sum().float()
            usage_fraction = codes_used / codebook_size
            usage_fractions.append(usage_fraction)

        perplexities = torch.stack(perplexities)
        usage_fractions = torch.stack(usage_fractions)

        # Detach stats from computation graph - these are metrics, not for backprop
        return {
            "perplexity_per_level": perplexities.detach(),
            "avg_perplexity": perplexities.mean().detach(),
            "usage_per_level": usage_fractions.detach(),
            "avg_usage": usage_fractions.mean().detach(),
        }
