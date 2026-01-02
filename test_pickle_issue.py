"""
Minimal script to reproduce ConfigModuleInstance pickle error with Unsloth.

Run on a GPU machine (e.g., RunPod) with:
    pip install unsloth trl datasets
    python test_pickle_issue.py
"""

from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel


def test_with_unsloth(num_proc: int = 4):
    """Test SFTTrainer with Unsloth tokenizer and multiprocessing."""

    print(f"\n{'=' * 60}")
    print(f"Testing with Unsloth, dataset_num_proc={num_proc}")
    print("=" * 60)

    # Load model with Unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
        max_seq_length=512,
        dtype=None,
        load_in_4bit=True,
    )

    # Create simple dataset with messages
    data = {
        "messages": [
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ],
            [
                {"role": "user", "content": "How are you?"},
                {"role": "assistant", "content": "I am fine."},
            ],
        ]
        * 20  # 40 examples
    }
    dataset = Dataset.from_dict(data)

    config = SFTConfig(
        output_dir="/tmp/test_sft",
        max_steps=2,
        per_device_train_batch_size=2,
        report_to="none",
        dataset_num_proc=num_proc,  # This triggers multiprocessing
        dataloader_num_workers=0,
        logging_steps=1,
    )

    print("Creating SFTTrainer...")
    try:
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=dataset,
            args=config,
        )
        print("SFTTrainer created successfully!")

        print("Starting training...")
        trainer.train()
        print("Training completed successfully!")
        return True
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_with_transformers(num_proc: int = 4):
    """Test SFTTrainer with standard transformers tokenizer (should work)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n{'=' * 60}")
    print(f"Testing with standard transformers, dataset_num_proc={num_proc}")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")

    data = {
        "messages": [
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ],
            [
                {"role": "user", "content": "How are you?"},
                {"role": "assistant", "content": "I am fine."},
            ],
        ]
        * 20
    }
    dataset = Dataset.from_dict(data)

    config = SFTConfig(
        output_dir="/tmp/test_sft",
        max_steps=2,
        per_device_train_batch_size=2,
        report_to="none",
        dataset_num_proc=num_proc,
        dataloader_num_workers=0,
        logging_steps=1,
    )

    print("Creating SFTTrainer...")
    try:
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=dataset,
            args=config,
        )
        print("SFTTrainer created successfully!")

        print("Starting training...")
        trainer.train()
        print("Training completed successfully!")
        return True
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    import torch

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Test 1: Standard transformers with multiprocessing (should work)
    test_with_transformers(num_proc=4)

    # Test 2: Unsloth with num_proc=1 (should work)
    test_with_unsloth(num_proc=1)

    # Test 3: Unsloth with num_proc=4 (may fail with pickle error)
    test_with_unsloth(num_proc=4)
