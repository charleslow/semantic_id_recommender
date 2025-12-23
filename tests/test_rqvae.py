"""
Tests for RQ-VAE model.

Tests the core functionality of semantic ID learning.
"""

import pytest
import torch

from src.rqvae.model import SemanticRQVAE, SemanticRQVAEConfig


@pytest.fixture
def config():
    """Default test config with small dimensions for speed."""
    return SemanticRQVAEConfig(
        embedding_dim=64,
        hidden_dim=128,
        codebook_size=32,
        num_quantizers=4,
        commitment_weight=0.25,
        decay=0.99,
    )


@pytest.fixture
def model(config):
    """Create model instance."""
    return SemanticRQVAE(config)


@pytest.fixture
def sample_embeddings(config):
    """Generate sample embeddings."""
    batch_size = 16
    return torch.randn(batch_size, config.embedding_dim)


class TestSemanticRQVAE:
    """Tests for SemanticRQVAE model."""

    def test_forward_shape(self, model, sample_embeddings, config):
        """Test forward pass output shapes."""
        reconstructed, indices, recon_loss, commit_loss = model(sample_embeddings)

        batch_size = sample_embeddings.shape[0]

        assert reconstructed.shape == (batch_size, config.embedding_dim)
        assert indices.shape == (batch_size, config.num_quantizers)
        assert recon_loss.dim() == 0  # scalar
        assert commit_loss.dim() == 0  # scalar

    def test_indices_valid_range(self, model, sample_embeddings, config):
        """Test that generated indices are within valid range."""
        _, indices, _, _ = model(sample_embeddings)

        assert indices.min() >= 0
        assert indices.max() < config.codebook_size

    def test_get_semantic_ids(self, model, sample_embeddings, config):
        """Test semantic ID extraction."""
        indices = model.get_semantic_ids(sample_embeddings)

        batch_size = sample_embeddings.shape[0]
        assert indices.shape == (batch_size, config.num_quantizers)

    def test_semantic_id_to_string(self, model, sample_embeddings):
        """Test semantic ID string conversion."""
        indices = model.get_semantic_ids(sample_embeddings)
        strings = model.semantic_id_to_string(indices)

        assert len(strings) == sample_embeddings.shape[0]
        assert all("[SEM_" in s for s in strings)

        # Check format: [SEM_0_X][SEM_1_X][SEM_2_X][SEM_3_X]
        for s in strings:
            assert s.count("[SEM_") == model.config.num_quantizers

    def test_string_to_semantic_id_roundtrip(self, model, sample_embeddings, config):
        """Test roundtrip conversion string <-> indices."""
        indices = model.get_semantic_ids(sample_embeddings)
        strings = model.semantic_id_to_string(indices)

        for i, s in enumerate(strings):
            parsed = SemanticRQVAE.string_to_semantic_id(s, config.num_quantizers)
            assert parsed is not None
            assert parsed == indices[i].tolist()

    def test_string_to_semantic_id_invalid(self, config):
        """Test parsing of invalid semantic ID strings."""
        # Missing tokens
        assert SemanticRQVAE.string_to_semantic_id("[SEM_0_1]", config.num_quantizers) is None

        # Invalid format
        assert SemanticRQVAE.string_to_semantic_id("invalid", config.num_quantizers) is None

        # Wrong number of quantizers
        s = "[SEM_0_1][SEM_1_2]"
        assert SemanticRQVAE.string_to_semantic_id(s, 4) is None

    def test_encode_decode(self, model, sample_embeddings, config):
        """Test encode-decode path."""
        z = model.encode(sample_embeddings)
        assert z.shape == (sample_embeddings.shape[0], config.hidden_dim)

        quantized, _, _ = model.quantize(z)
        assert quantized.shape == z.shape

        reconstructed = model.decode(quantized)
        assert reconstructed.shape == sample_embeddings.shape

    def test_reconstruction_loss_decreases(self, model, sample_embeddings):
        """Test that training reduces reconstruction loss."""
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Get initial loss
        _, _, initial_loss, _ = model(sample_embeddings)
        initial_loss_val = initial_loss.item()

        # Train for a few steps
        for _ in range(50):
            optimizer.zero_grad()
            _, _, recon_loss, commit_loss = model(sample_embeddings)
            loss = recon_loss + commit_loss
            loss.backward()
            optimizer.step()

        # Get final loss
        _, _, final_loss, _ = model(sample_embeddings)
        final_loss_val = final_loss.item()

        assert final_loss_val < initial_loss_val

    def test_deterministic_indices(self, model, sample_embeddings):
        """Test that same input gives same indices (in eval mode)."""
        model.eval()
        with torch.no_grad():
            indices1 = model.get_semantic_ids(sample_embeddings)
            indices2 = model.get_semantic_ids(sample_embeddings)

        assert torch.equal(indices1, indices2)


class TestSemanticRQVAEConfig:
    """Tests for config dataclass."""

    def test_default_values(self):
        """Test default config values."""
        config = SemanticRQVAEConfig()

        assert config.embedding_dim == 384
        assert config.hidden_dim == 512
        assert config.codebook_size == 256
        assert config.num_quantizers == 4

    def test_custom_values(self):
        """Test custom config values."""
        config = SemanticRQVAEConfig(
            embedding_dim=128,
            codebook_size=64,
        )

        assert config.embedding_dim == 128
        assert config.codebook_size == 64


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
