"""
Tests for LLM fine-tuning module.

These tests verify:
- Tokenizer resizing with semantic ID tokens
- Training data preparation
- SFTTrainer integration with multiprocessing
- Freeze backbone functionality

Run on GPU (e.g., RunPod) with: pytest tests/test_llm.py -v
"""

import pytest
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.llm.data import (
    format_as_messages,
    generate_training_examples,
    get_semantic_id_tokens,
)
from src.llm.finetune import add_semantic_tokens, freeze_backbone


# Test config - small values for speed
NUM_QUANTIZERS = 2
CODEBOOK_SIZE = 16
MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"


@pytest.fixture
def tokenizer():
    """Load base tokenizer."""
    return AutoTokenizer.from_pretrained(MODEL_NAME)


@pytest.fixture
def model():
    """Load small model for CPU tests."""
    return AutoModelForCausalLM.from_pretrained(MODEL_NAME)


@pytest.fixture
def sample_items_dataset():
    """Create sample items dataset with semantic IDs."""
    return Dataset.from_list([
        {"id": "1", "title": "Blue Widget", "category": "widgets", "semantic_id": "[SEM_START][SEM_0_1][SEM_1_2][SEM_END]"},
        {"id": "2", "title": "Red Gadget", "category": "gadgets", "semantic_id": "[SEM_START][SEM_0_3][SEM_1_4][SEM_END]"},
        {"id": "3", "title": "Green Tool", "category": "tools", "semantic_id": "[SEM_START][SEM_0_5][SEM_1_6][SEM_END]"},
    ])


@pytest.fixture
def query_templates():
    """Query templates for training data generation."""
    return {
        "predict_semantic_id": [
            "Find: {title}",
            "Search for {title}",
        ],
        "predict_attribute": [
            "What is the {field_name} for {semantic_id}?",
        ],
    }


class TestSemanticTokens:
    """Tests for semantic ID token handling."""

    def test_get_semantic_id_tokens_count(self):
        """Test correct number of tokens generated."""
        tokens = get_semantic_id_tokens(NUM_QUANTIZERS, CODEBOOK_SIZE)
        # 3 special tokens + num_quantizers * codebook_size
        expected = 3 + NUM_QUANTIZERS * CODEBOOK_SIZE
        assert len(tokens) == expected

    def test_get_semantic_id_tokens_format(self):
        """Test token format is correct."""
        tokens = get_semantic_id_tokens(NUM_QUANTIZERS, CODEBOOK_SIZE)
        assert "[REC]" in tokens
        assert "[SEM_START]" in tokens
        assert "[SEM_END]" in tokens
        assert "[SEM_0_0]" in tokens
        assert f"[SEM_{NUM_QUANTIZERS - 1}_{CODEBOOK_SIZE - 1}]" in tokens

    def test_add_semantic_tokens_resizes_vocab(self, tokenizer):
        """Test that adding semantic tokens increases vocab size."""
        original_vocab_size = len(tokenizer)
        tokenizer = add_semantic_tokens(tokenizer, NUM_QUANTIZERS, CODEBOOK_SIZE)
        new_vocab_size = len(tokenizer)

        expected_new_tokens = 3 + NUM_QUANTIZERS * CODEBOOK_SIZE
        assert new_vocab_size == original_vocab_size + expected_new_tokens

    def test_semantic_tokens_are_tokenizable(self, tokenizer):
        """Test that semantic tokens encode to single token IDs."""
        tokenizer = add_semantic_tokens(tokenizer, NUM_QUANTIZERS, CODEBOOK_SIZE)

        # Each semantic token should encode to exactly one token
        test_tokens = ["[REC]", "[SEM_START]", "[SEM_END]", "[SEM_0_0]", "[SEM_1_5]"]
        for token in test_tokens:
            encoded = tokenizer.encode(token, add_special_tokens=False)
            assert len(encoded) == 1, f"Token {token} encoded to {len(encoded)} tokens"


class TestModelResizing:
    """Tests for model embedding resizing."""

    def test_model_resize_embeddings(self, model, tokenizer):
        """Test model embedding layer resizes correctly."""
        original_vocab_size = model.config.vocab_size
        tokenizer = add_semantic_tokens(tokenizer, NUM_QUANTIZERS, CODEBOOK_SIZE)
        model.resize_token_embeddings(len(tokenizer))

        expected_new_tokens = 3 + NUM_QUANTIZERS * CODEBOOK_SIZE
        assert model.config.vocab_size == original_vocab_size + expected_new_tokens

    def test_embedding_layer_shape(self, model, tokenizer):
        """Test input/output embedding shapes match new vocab size."""
        tokenizer = add_semantic_tokens(tokenizer, NUM_QUANTIZERS, CODEBOOK_SIZE)
        model.resize_token_embeddings(len(tokenizer))

        input_emb = model.get_input_embeddings()
        output_emb = model.get_output_embeddings()

        assert input_emb.weight.shape[0] == len(tokenizer)
        assert output_emb.weight.shape[0] == len(tokenizer)


class TestFreezeBackbone:
    """Tests for backbone freezing in stage 1 training."""

    def test_freeze_backbone_freezes_most_params(self, model):
        """Test that freeze_backbone freezes non-embedding parameters."""
        freeze_backbone(model)

        # Count frozen vs trainable parameters
        frozen = sum(1 for p in model.parameters() if not p.requires_grad)
        trainable = sum(1 for p in model.parameters() if p.requires_grad)

        # Most params should be frozen, only embeddings trainable
        assert frozen > trainable
        assert trainable > 0  # At least embeddings should be trainable

    def test_freeze_backbone_keeps_embeddings_trainable(self, model):
        """Test that embedding layers remain trainable."""
        freeze_backbone(model)

        input_emb = model.get_input_embeddings()
        output_emb = model.get_output_embeddings()

        assert all(p.requires_grad for p in input_emb.parameters())
        assert all(p.requires_grad for p in output_emb.parameters())

    def test_gradient_hooks_mask_original_tokens(self, model, tokenizer):
        """Test gradient hooks zero out gradients for original tokens."""
        tokenizer = add_semantic_tokens(tokenizer, NUM_QUANTIZERS, CODEBOOK_SIZE)
        original_vocab_size = model.config.vocab_size
        model.resize_token_embeddings(len(tokenizer))

        num_new_tokens = 3 + NUM_QUANTIZERS * CODEBOOK_SIZE
        hooks = freeze_backbone(model, num_new_tokens=num_new_tokens)

        assert len(hooks) > 0, "Expected gradient hooks to be registered"

        # Simulate a backward pass
        input_emb = model.get_input_embeddings()
        fake_loss = input_emb.weight.sum()
        fake_loss.backward()

        # Check gradient is zeroed for original tokens, non-zero for new
        grad = input_emb.weight.grad
        assert grad[:original_vocab_size].abs().sum() == 0, "Original token gradients should be zero"
        assert grad[-num_new_tokens:].abs().sum() > 0, "New token gradients should be non-zero"

        # Clean up hooks
        for h in hooks:
            h.remove()


class TestTrainingDataPreparation:
    """Tests for training data generation."""

    def test_generate_training_examples(self, sample_items_dataset, query_templates):
        """Test training examples are generated correctly."""
        examples = generate_training_examples(
            sample_items_dataset,
            query_templates=query_templates,
            num_examples_per_item=2,
            id_field="id",
        )

        assert len(examples) > 0
        assert "query" in examples.column_names
        assert "response" in examples.column_names
        assert "type" in examples.column_names

    def test_format_as_messages(self, sample_items_dataset, query_templates):
        """Test messages format is correct for SFTTrainer."""
        examples = generate_training_examples(
            sample_items_dataset,
            query_templates=query_templates,
            num_examples_per_item=2,
            id_field="id",
        )
        formatted = format_as_messages(examples)

        assert "messages" in formatted.column_names

        # Check message structure
        first_msg = formatted[0]["messages"]
        assert isinstance(first_msg, list)
        assert len(first_msg) >= 2  # At least system/user or user/assistant
        assert all("role" in m and "content" in m for m in first_msg)


class TestSFTTrainerIntegration:
    """Integration tests for SFTTrainer with multiprocessing."""

    @pytest.fixture
    def training_dataset(self):
        """Create minimal training dataset."""
        examples = []
        for i in range(20):
            examples.append({
                "messages": [
                    {"role": "user", "content": f"Find item {i}"},
                    {"role": "assistant", "content": f"[SEM_START][SEM_0_{i % 16}][SEM_1_{i % 16}][SEM_END]"},
                ]
            })
        return Dataset.from_list(examples)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
    def test_sft_trainer_with_unsloth_multiproc(self, training_dataset):
        """Test SFTTrainer works with Unsloth and num_proc > 1."""
        # Import here to control when Unsloth patches SFTTrainer
        from unsloth import FastLanguageModel
        from trl import SFTConfig, SFTTrainer

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_NAME,
            max_seq_length=128,
            dtype=None,
            load_in_4bit=True,
        )

        # Formatting function required by Unsloth for multiprocessing
        # Must return a list of strings, not a dict
        def formatting_func(examples):
            texts = []
            for messages in examples["messages"]:
                # Ensure messages is a proper list of dicts
                if isinstance(messages, dict):
                    messages = [messages]
                text = tokenizer.apply_chat_template(
                    list(messages), tokenize=False, add_generation_prompt=False
                )
                texts.append(text)
            return texts

        config = SFTConfig(
            output_dir="/tmp/test_sft_unsloth",
            max_steps=1,
            per_device_train_batch_size=2,
            report_to="none",
            dataset_num_proc=4,  # Test multiprocessing with Unsloth
            dataloader_num_workers=0,
            logging_steps=1,
            max_length=128,
            packing=False,
        )

        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=training_dataset,
            formatting_func=formatting_func,
            args=config,
        )

        # Should not raise
        trainer.train()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
