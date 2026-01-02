"""
Minimal script to test SFTTrainer with Unsloth, matching finetune.py structure.

Run on a GPU machine (e.g., RunPod) with:
    pip install unsloth trl datasets
    python test_pickle_issue.py
"""

from unsloth import FastLanguageModel  # isort: skip

from datasets import Dataset
from trl import SFTConfig, SFTTrainer


def create_test_dataset() -> Dataset:
    """Create a test dataset matching the format from finetune.py."""
    # Simulate the format from format_as_messages() in src/llm/data.py
    examples = []
    queries = [
        (
            "Find: Blue Widget",
            "[SEM_START][SEM_0_1][SEM_1_2][SEM_2_3][SEM_3_4][SEM_END]",
        ),
        (
            "Search for Red Gadget",
            "[SEM_START][SEM_0_5][SEM_1_6][SEM_2_7][SEM_3_8][SEM_END]",
        ),
        (
            "Recommend: Green Tool",
            "[SEM_START][SEM_0_9][SEM_1_10][SEM_2_11][SEM_3_12][SEM_END]",
        ),
        (
            "Show me Yellow Device",
            "[SEM_START][SEM_0_13][SEM_1_14][SEM_2_15][SEM_3_16][SEM_END]",
        ),
    ]

    for query, response in queries * 10:  # 40 examples
        examples.append(
            {
                "messages": [
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": response},
                ]
            }
        )

    return Dataset.from_list(examples)


def test_with_unsloth(num_proc: int = 4):
    """Test SFTTrainer with Unsloth, matching finetune.py configuration."""

    print(f"\n{'=' * 60}")
    print(f"Testing with Unsloth, dataset_num_proc={num_proc}")
    print("=" * 60)

    # Load model with Unsloth (same as finetune.py)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
        max_seq_length=512,
        dtype=None,
        load_in_4bit=True,
    )

    dataset = create_test_dataset()

    # Match SFTConfig from finetune.py
    config = SFTConfig(
        output_dir="/tmp/test_sft",
        max_steps=2,
        per_device_train_batch_size=2,
        report_to="none",
        dataset_num_proc=num_proc,
        dataloader_num_workers=0,
        logging_steps=1,
        # Match finetune.py settings
        max_length=512,
        packing=False,
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

    dataset = create_test_dataset()

    config = SFTConfig(
        output_dir="/tmp/test_sft",
        max_steps=2,
        per_device_train_batch_size=2,
        report_to="none",
        dataset_num_proc=num_proc,
        dataloader_num_workers=0,
        logging_steps=1,
        max_length=512,
        packing=False,
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
