"""
Training data preparation for LLM fine-tuning.

Generates query -> semantic ID training pairs for cold-start scenario.
"""

import json
import random
from pathlib import Path

from datasets import Dataset
from tqdm import tqdm


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
    items: list[dict],
    query_templates: dict[str, list[str]],
    field_mapping: dict[str, str] | None = None,
    num_examples_per_item: int = 5,
    id_field: str = "id",
    predict_semantic_id_ratio: float = 0.8,
) -> list[dict]:
    """
    Generate training examples for LLM fine-tuning.

    Generates two types of examples:
    1. Predict semantic_id: Given item attributes, predict the semantic ID
    2. Predict attribute: Given semantic ID, predict an item attribute

    Args:
        items: List of items with semantic_id field
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
                        "What is the semantic ID for an item with title: {title}?",
                        "Recommend something like {title}. What is its semantic ID?",
                        "{title} - what's the semantic ID?",
                        "Get semantic ID: {description}",
                        "Item: {title}, {category}. Semantic ID?",
                    ],
                    "predict_attribute": [
                        "What is the {field_name} for semantic ID {semantic_id}?",
                        "For semantic ID {semantic_id}, what is the {field_name}?",
                        "Semantic ID {semantic_id}: {field_name}?",
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
        List of training examples with 'query', 'response', and 'type' fields
    """
    # If no field_mapping, use field names as-is
    if field_mapping is None:
        sample_item = items[0] if items else {}
        available_fields = sample_item.keys()
        field_mapping = {field: field for field in available_fields}

    examples = []
    for item in tqdm(items, desc="Generating training examples"):
        semantic_id = item.get("semantic_id")
        if not semantic_id:
            continue

        # Get item text fields using field_mapping
        field_values = {}
        for placeholder, field_name in field_mapping.items():
            field_values[placeholder] = item.get(field_name, "")

        # Determine number of each type of example
        num_predict_semantic = int(num_examples_per_item * predict_semantic_id_ratio)
        num_predict_attr = num_examples_per_item - num_predict_semantic

        # Generate "predict semantic_id" examples
        if num_predict_semantic > 0:
            templates_to_use = random.choices(
                query_templates["predict_semantic_id"],
                k=min(
                    num_predict_semantic, len(query_templates["predict_semantic_id"])
                ),
            )
            for template in templates_to_use:
                query = template.format(**field_values)
                if query.strip():
                    examples.append(
                        {
                            "query": query.strip(),
                            "response": semantic_id,
                            "type": "predict_semantic_id",
                            "item_id": item.get(id_field, ""),
                        }
                    )

        # Generate "predict attribute" examples
        if num_predict_attr > 0:
            # For each example, pick a random field to predict
            available_fields = [
                (placeholder, value)
                for placeholder, value in field_values.items()
                if value and value.strip()
            ]

            for _ in range(num_predict_attr):
                # Pick random field to predict
                field_to_predict, field_value = random.choice(available_fields)
                template = random.choice(query_templates["predict_attribute"])
                query = template.format(
                    semantic_id=semantic_id, field_name=field_to_predict
                )
                if query.strip() and field_value.strip():
                    examples.append(
                        {
                            "query": query.strip(),
                            "response": field_value.strip(),
                            "type": "predict_attribute",
                            "field_name": field_to_predict,
                            "item_id": item.get(id_field, ""),
                        }
                    )

    return examples


def format_as_messages(
    examples: list[dict],
    system_prompt: str | None = None,
) -> list[dict]:
    """
    Format examples as message lists for chat-style fine-tuning.

    Args:
        examples: Training examples with query and response
        system_prompt: Optional system prompt. If None, uses a default prompt that adapts
                      to both predict_semantic_id and predict_attribute tasks.

    Returns:
        List of examples with 'messages' field containing role/content dicts
    """
    if system_prompt is None:
        system_prompt = (
            "You are a recommendation system. You can:\n"
            "1. Given item attributes, output the semantic ID\n"
            "2. Given a semantic ID, output item attributes\n"
            "Respond only with the requested information."
        )

    formatted = []
    for ex in examples:
        formatted.append(
            {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": ex["query"]},
                    {"role": "assistant", "content": ex["response"]},
                ]
            }
        )

    return formatted


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
    # Convert to list for generate_training_examples (which needs random access for templates)
    items = list(items_dataset)
    print(f"Loaded {len(items)} items with semantic IDs")

    # Generate examples
    examples = generate_training_examples(
        items,
        num_examples_per_item,
        query_templates=query_templates,
        field_mapping=field_mapping,
        id_field=id_field,
        predict_semantic_id_ratio=predict_semantic_id_ratio,
    )
    print(f"Generated {len(examples)} training examples")

    # Print distribution of example types
    predict_sem_count = sum(
        1 for ex in examples if ex.get("type") == "predict_semantic_id"
    )
    predict_attr_count = sum(
        1 for ex in examples if ex.get("type") == "predict_attribute"
    )
    print(f"  - {predict_sem_count} predict semantic_id examples")
    print(f"  - {predict_attr_count} predict attribute examples")

    # Format as messages
    formatted = format_as_messages(examples)

    # Shuffle and split
    random.shuffle(formatted)
    split_idx = int(len(formatted) * (1 - val_split))
    train_examples = formatted[:split_idx]
    val_examples = formatted[split_idx:]

    print(f"Train: {len(train_examples)}, Val: {len(val_examples)}")

    # Save to JSONL
    output_train_path = Path(output_train_path)
    output_val_path = Path(output_val_path)
    output_train_path.parent.mkdir(parents=True, exist_ok=True)
    output_val_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_train_path, "w") as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + "\n")

    with open(output_val_path, "w") as f:
        for ex in val_examples:
            f.write(json.dumps(ex) + "\n")

    # Create HuggingFace datasets
    train_dataset = Dataset.from_list(train_examples)
    val_dataset = Dataset.from_list(val_examples)

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

        # Load data
        self.examples = []
        with open(self.data_path) as f:
            for line in f:
                self.examples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return self.examples[idx]

    def to_hf_dataset(self) -> Dataset:
        """Convert to HuggingFace Dataset."""
        return Dataset.from_list(self.examples)


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
        List of special tokens like ["[SEM_START]", "[SEM_END]", "[SEM_0_0]", "[SEM_0_1]", ...]
    """
    tokens = ["[SEM_START]", "[SEM_END]"]
    for q in range(num_quantizers):
        for c in range(codebook_size):
            tokens.append(f"[SEM_{q}_{c}]")
    return tokens
