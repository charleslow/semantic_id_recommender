"""
Training data preparation for LLM fine-tuning.

Generates query -> semantic ID training pairs for cold-start scenario.
"""

import json
import random
from pathlib import Path

from datasets import Dataset

# Special tokens for semantic ID generation
REC_TOKEN = "[REC]"
SEM_START_TOKEN = "[SEM_START]"
SEM_END_TOKEN = "[SEM_END]"

# Default system prompt used across all LLM components
DEFAULT_SYSTEM_PROMPT = (
    "You are a recommendation system. You can:\n"
    "1. Given item attributes, output the semantic ID\n"
    "2. Given a semantic ID, output item attributes\n"
    "Respond only with the requested information."
)


def load_catalogue_with_semantic_ids(
    catalogue_path: str | Path,
    semantic_ids_path: str | Path,
    strict: bool = True,
    id_field: str = "id",
) -> Dataset:
    """
    Load catalogue items with their semantic IDs incrementally into a HuggingFace Dataset.

    Args:
        catalogue_path: Path to catalogue JSON/JSONL
        semantic_ids_path: Path to semantic ID mapping JSON
        strict: If True, raise error when items don't have semantic IDs. If False, skip missing items.
        id_field: Name of the field containing item IDs (default: "id")

    Returns:
        HuggingFace Dataset with semantic_id field added to each item

    Raises:
        ValueError: If strict=True and catalogue/semantic IDs don't match up
    """
    catalogue_path = Path(catalogue_path)

    if not catalogue_path.suffix == ".jsonl":
        raise NotImplementedError("Only JSONL catalogue format is supported currently.")

    with open(semantic_ids_path) as f:
        semantic_mapping = json.load(f)
    item_to_semantic = semantic_mapping["item_to_semantic"]

    def generate_items():
        """Generator that yields items with semantic IDs."""

        with open(catalogue_path) as f:
            for line in f:
                item = json.loads(line)
                item_id = str(item.get(id_field, ""))
                if item_id in item_to_semantic:
                    item["semantic_id"] = item_to_semantic[item_id]["semantic_id"]
                    yield item
                elif strict:
                    raise ValueError(
                        "Found catalogue items without semantic IDs. "
                        f"Item ID: {item_id}."
                    )

    return Dataset.from_generator(generate_items)


def generate_training_examples(
    items: Dataset,
    query_templates: dict[str, list[str]],
    field_mapping: dict[str, str] | None = None,
    num_examples_per_item: int = 5,
    id_field: str = "id",
    predict_semantic_id_ratio: float = 0.8,
) -> Dataset:
    """
    Generate training examples for LLM fine-tuning.

    Generates two types of examples:
    1. Predict semantic_id: Given item attributes, predict the semantic ID
    2. Predict attribute: Given semantic ID, predict an item attribute

    Args:
        items: Dataset of items with semantic_id field
        num_examples_per_item: Number of query variations per item
        query_templates: Dict with two keys:
                        - "predict_semantic_id": Templates for predicting semantic ID from attributes
                        - "predict_attribute": Templates for predicting attributes from semantic ID

            For example:

                query_templates = {
                    "predict_semantic_id": [
                        "What is the semantic ID of {title}?",
                        "Find the semantic ID for {title}",
                        "Semantic ID for: {title}",
                        "Recommend something like {title}. What is its semantic ID?",
                        "{title} - what's the semantic ID?",
                        "Item: {title}, {category}. Semantic ID?",
                    ],
                    "predict_attribute": [
                        "What is the {field_name} for semantic ID {semantic_id}?",
                        "For semantic ID {semantic_id}, what is the {field_name}?",
                        "Get {field_name} for {semantic_id}",
                        "{semantic_id} - what's the {field_name}?",
                    ],
                }

        field_mapping: Mapping from template placeholder names to actual item field names.
                      E.g., {"title": "name", "description": "desc"} if your items use "name" and "desc"
                      Default: {"title": "title", "description": "description", "category": "category"}
        id_field: Name of the field containing item IDs (default: "id")
        predict_semantic_id_ratio: Ratio of examples that predict semantic_id (default: 0.7)
                                   Remaining examples will predict attributes

    Returns:
        Dataset of training examples with 'query', 'response', and 'type' fields
    """
    # If no field_mapping, use field names as-is from dataset columns
    if field_mapping is None:
        field_mapping = {col: col for col in items.column_names}

    num_predict_semantic = int(num_examples_per_item * predict_semantic_id_ratio)
    num_predict_attr = num_examples_per_item - num_predict_semantic

    def generate_examples_batch(batch):
        """Generate multiple training examples for a batch of items."""
        all_queries = []
        all_responses = []
        all_types = []
        all_item_ids = []

        batch_size = len(batch["semantic_id"])
        for i in range(batch_size):
            semantic_id = batch["semantic_id"][i]
            if not semantic_id:
                continue

            # Get item text fields using field_mapping
            field_values = {}
            for placeholder, field_name in field_mapping.items():
                field_values[placeholder] = (
                    batch.get(field_name, [""] * batch_size)[i] or ""
                )

            item_id = str(batch.get(id_field, [""] * batch_size)[i] or "")

            # Generate "predict semantic_id" examples
            if num_predict_semantic > 0:
                templates_to_use = random.choices(
                    query_templates["predict_semantic_id"],
                    k=min(
                        num_predict_semantic,
                        len(query_templates["predict_semantic_id"]),
                    ),
                )
                for template in templates_to_use:
                    try:
                        query = template.format(**field_values)
                        if query.strip():
                            # Append [REC] token to query to signal semantic ID generation
                            all_queries.append(f"{query.strip()}{REC_TOKEN}")
                            all_responses.append(semantic_id)
                            all_types.append("predict_semantic_id")
                            all_item_ids.append(item_id)
                    except KeyError:
                        pass

            # Generate "predict attribute" examples
            if num_predict_attr > 0:
                available_fields = [
                    (placeholder, value)
                    for placeholder, value in field_values.items()
                    if value and str(value).strip()
                ]

                if available_fields:
                    for _ in range(num_predict_attr):
                        field_to_predict, field_value = random.choice(available_fields)
                        template = random.choice(query_templates["predict_attribute"])
                        query = template.format(
                            semantic_id=semantic_id, field_name=field_to_predict
                        )
                        if query.strip() and str(field_value).strip():
                            all_queries.append(query.strip())
                            all_responses.append(str(field_value).strip())
                            all_types.append("predict_attribute")
                            all_item_ids.append(item_id)

        return {
            "query": all_queries,
            "response": all_responses,
            "type": all_types,
            "item_id": all_item_ids,
        }

    return items.map(
        generate_examples_batch,
        batched=True,
        remove_columns=items.column_names,
        desc="Generating training examples",
    )


def format_as_messages(
    examples: Dataset,
    system_prompt: str | None = None,
) -> Dataset:
    """
    Format examples as message lists for chat-style fine-tuning.

    Args:
        examples: Dataset of training examples with query and response
        system_prompt: Optional system prompt. If None, uses a default prompt that adapts
                      to both predict_semantic_id and predict_attribute tasks.

    Returns:
        Dataset with 'messages' field containing role/content dicts
    """
    if system_prompt is None:
        system_prompt = DEFAULT_SYSTEM_PROMPT

    def add_messages(example):
        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": example["query"]},
                {"role": "assistant", "content": example["response"]},
            ]
        }

    return examples.map(add_messages)


def prepare_training_data(
    catalogue_path: str | Path,
    semantic_ids_path: str | Path,
    output_train_path: str | Path,
    output_val_path: str | Path,
    query_templates: dict[str, list[str]],
    *,
    num_examples_per_item: int = 5,
    val_split: float = 0.1,
    seed: int = 42,
    field_mapping: dict[str, str] | None = None,
    id_field: str = "id",
    strict: bool = True,
    predict_semantic_id_ratio: float = 0.8,
) -> tuple[Dataset, Dataset]:
    """
    Prepare training and validation datasets.

    Args:
        catalogue_path: Path to catalogue
        semantic_ids_path: Path to semantic ID mapping
        output_train_path: Path to save training JSONL
        output_val_path: Path to save validation JSONL
        num_examples_per_item: Examples per item
        val_split: Validation split ratio
        seed: Random seed
        query_templates: Custom query templates dict with "predict_semantic_id" and "predict_attribute" keys
        field_mapping: Mapping from template placeholder names to actual item field names
        id_field: Name of the field containing item IDs (default: "id")
        strict: If True, raise error when items don't have semantic IDs
        predict_semantic_id_ratio: Ratio of examples that predict semantic_id (default: 0.7)

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    random.seed(seed)

    # Load items with semantic IDs as a Dataset
    items_dataset = load_catalogue_with_semantic_ids(
        catalogue_path, semantic_ids_path, strict=strict, id_field=id_field
    )
    print(f"Loaded {len(items_dataset)} items with semantic IDs")

    # Generate examples
    examples = generate_training_examples(
        items_dataset,
        query_templates=query_templates,
        field_mapping=field_mapping,
        num_examples_per_item=num_examples_per_item,
        id_field=id_field,
        predict_semantic_id_ratio=predict_semantic_id_ratio,
    )
    print(f"Generated {len(examples)} training examples")

    # Print distribution of example types
    type_counts = {t: 0 for t in ["predict_semantic_id", "predict_attribute"]}
    for t in examples["type"]:
        if t in type_counts:
            type_counts[t] += 1
    print(f"  - {type_counts['predict_semantic_id']} predict semantic_id examples")
    print(f"  - {type_counts['predict_attribute']} predict attribute examples")

    # Format as messages
    formatted = format_as_messages(examples)

    # Shuffle and split
    formatted = formatted.shuffle(seed=seed)
    split = formatted.train_test_split(test_size=val_split, seed=seed)
    train_dataset = split["train"]
    val_dataset = split["test"]

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Save to JSONL
    output_train_path = Path(output_train_path)
    output_val_path = Path(output_val_path)
    output_train_path.parent.mkdir(parents=True, exist_ok=True)
    output_val_path.parent.mkdir(parents=True, exist_ok=True)

    train_dataset.to_json(output_train_path, orient="records", lines=True)
    val_dataset.to_json(output_val_path, orient="records", lines=True)

    return train_dataset, val_dataset


class SemanticIDDataset:
    """Dataset wrapper for semantic ID training."""

    def __init__(
        self,
        data_path: str | Path,
        tokenizer=None,
        max_length: int = 512,
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load data as HuggingFace Dataset
        self._dataset = Dataset.from_json(str(self.data_path))

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> dict:
        return self._dataset[idx]

    def to_hf_dataset(self) -> Dataset:
        """Return the HuggingFace Dataset."""
        return self._dataset


def get_semantic_id_tokens(
    num_quantizers: int = 4,
    codebook_size: int = 256,
) -> list[str]:
    """
    Get list of all semantic ID special tokens.

    Args:
        num_quantizers: Number of RQ-VAE quantizers
        codebook_size: Size of each codebook

    Returns:
        List of special tokens like ["[REC]", "[SEM_START]", "[SEM_END]", "[SEM_0_0]", ...]
    """
    tokens = [REC_TOKEN, SEM_START_TOKEN, SEM_END_TOKEN]
    for q in range(num_quantizers):
        for c in range(codebook_size):
            tokens.append(f"[SEM_{q}_{c}]")
    return tokens
