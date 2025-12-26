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
) -> list[dict]:
    """
    Load catalogue items with their semantic IDs.

    Args:
        catalogue_path: Path to catalogue JSON/JSONL
        semantic_ids_path: Path to semantic ID mapping JSON

    Returns:
        List of items with semantic_id field added
    """
    # Load catalogue
    catalogue_path = Path(catalogue_path)
    if catalogue_path.suffix == ".jsonl":
        items = []
        with open(catalogue_path) as f:
            for line in f:
                items.append(json.loads(line))
    else:
        with open(catalogue_path) as f:
            data = json.load(f)
            items = data if isinstance(data, list) else data.get("items", [])

    # Load semantic IDs
    with open(semantic_ids_path) as f:
        semantic_mapping = json.load(f)

    item_to_semantic = semantic_mapping["item_to_semantic"]

    # Merge
    result = []
    for item in items:
        item_id = str(item.get("id", ""))
        if item_id in item_to_semantic:
            item["semantic_id"] = item_to_semantic[item_id]["semantic_id"]
            result.append(item)

    return result


def generate_training_examples(
    items: list[dict],
    num_examples_per_item: int = 3,
    query_templates: list[str] | None = None,
) -> list[dict]:
    """
    Generate training examples for LLM fine-tuning.

    For cold-start scenario, we create query -> semantic ID pairs using
    item metadata as the basis for queries.

    Args:
        items: List of items with semantic_id field
        num_examples_per_item: Number of query variations per item
        query_templates: Custom query templates (optional)

    Returns:
        List of training examples with 'query' and 'semantic_id' fields
    """
    if query_templates is None:
        query_templates = [
            "Find me a {title}",
            "I'm looking for {title}",
            "Recommend a {title}",
            "Show me {title}",
            "I want {title}",
            "Search for {title}",
            "{title}",
            "{description}",
            "I need something like {title}",
            "Find products similar to {title}",
        ]

    examples = []

    for item in tqdm(items, desc="Generating training examples"):
        semantic_id = item.get("semantic_id")
        if not semantic_id:
            continue

        # Get item text fields
        title = item.get("title", item.get("name", ""))
        description = item.get("description", "")
        category = item.get("category", "")

        # Generate query variations
        templates_to_use = random.sample(
            query_templates,
            min(num_examples_per_item, len(query_templates)),
        )

        for template in templates_to_use:
            try:
                query = template.format(
                    title=title,
                    description=description,
                    category=category,
                )
            except KeyError:
                query = template.format(title=title)

            examples.append({
                "query": query.strip(),
                "semantic_id": semantic_id,
                "item_id": item.get("id", ""),
            })

    return examples


def format_for_chat(
    examples: list[dict],
    system_prompt: str | None = None,
) -> list[dict]:
    """
    Format examples for chat-style fine-tuning.

    Args:
        examples: Training examples with query and semantic_id
        system_prompt: Optional system prompt

    Returns:
        List of chat-formatted examples
    """
    if system_prompt is None:
        system_prompt = (
            "You are a recommendation system. Given a user query, "
            "output the semantic ID of the most relevant item. "
            "Respond only with the semantic ID tokens."
        )

    formatted = []
    for ex in examples:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": ex["query"]},
            {"role": "assistant", "content": ex["semantic_id"]},
        ]
        formatted.append({
            "messages": messages,
            "item_id": ex.get("item_id", ""),
        })

    return formatted


def prepare_training_data(
    catalogue_path: str | Path,
    semantic_ids_path: str | Path,
    output_train_path: str | Path,
    output_val_path: str | Path,
    num_examples_per_item: int = 3,
    val_split: float = 0.1,
    seed: int = 42,
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

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    random.seed(seed)

    # Load items with semantic IDs
    items = load_catalogue_with_semantic_ids(catalogue_path, semantic_ids_path)
    print(f"Loaded {len(items)} items with semantic IDs")

    # Generate examples
    examples = generate_training_examples(items, num_examples_per_item)
    print(f"Generated {len(examples)} training examples")

    # Format for chat
    formatted = format_for_chat(examples)

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
