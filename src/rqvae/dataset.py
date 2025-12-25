"""
Dataset classes for RQ-VAE training.

Handles loading catalogue items and generating embeddings.
"""

import json
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset


class ItemEmbeddingDataset(Dataset):
    """
    Dataset of item embeddings for RQ-VAE training.

    Can either load pre-computed embeddings or generate them from text.
    """

    def __init__(
        self,
        embeddings: torch.Tensor | None = None,
        item_ids: list | None = None,
    ):
        """
        Initialize dataset with embeddings.

        Args:
            embeddings: Pre-computed embeddings [num_items, embedding_dim]
            item_ids: List of item IDs corresponding to embeddings
        """
        self.embeddings = embeddings
        self.item_ids = (
            item_ids or list(range(len(embeddings))) if embeddings is not None else []
        )

    def __len__(self) -> int:
        return len(self.embeddings) if self.embeddings is not None else 0

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.embeddings[idx]

    @classmethod
    def from_catalogue(
        cls,
        catalogue_path: str | Path,
        *,
        fields: list[str] | None = None,
        embedding_model: str = "TaylorAI/gte-tiny",
        id_field: str = "id",
        cache_path: str | Path | None = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> "ItemEmbeddingDataset":
        """
        Create dataset from catalogue file, generating embeddings.

        Args:
            catalogue_path: Path to JSON/JSONL file with items
            embedding_model: Sentence transformer model name
            fields: List of field names to include in the text representation.
                    Text will be formatted as "{field_name}: {field_value}\n..."
                    Defaults to ["title", "description"] if not specified.
            id_field: Field name for item ID
            cache_path: Optional path to cache embeddings
            device: Device for embedding generation

        Returns:
            ItemEmbeddingDataset with computed embeddings
        """
        # Check cache first
        if cache_path and Path(cache_path).exists():
            print(f"Loading cached embeddings from {cache_path}")
            data = torch.load(cache_path)
            return cls(embeddings=data["embeddings"], item_ids=data["item_ids"])

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

        print(f"Loaded {len(items)} items from {catalogue_path}")

        if fields is None:
            # Take all fields except the ID field
            sample_item = items[0]
            fields = [k for k in sample_item.keys() if k != id_field]

        # Extract texts and IDs
        texts = []
        item_ids = []
        for item in items:
            text_parts = [
                f"{field}: {item[field]}"
                for field in fields
                if (field in item) and item[field]
            ]
            text = "\n".join(text_parts)
            texts.append(text)
            item_ids.append(item.get(id_field, len(item_ids)))

        # Generate embeddings
        print(f"Generating embeddings with {embedding_model}...")

        model = SentenceTransformer(embedding_model, device=device)
        embeddings = model.encode(
            texts,
            show_progress_bar=True,
            convert_to_tensor=True,
            device=device,
        )

        # Move to CPU for storage
        embeddings = embeddings.cpu()

        # Cache embeddings
        if cache_path:
            print(f"Caching embeddings to {cache_path}")
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"embeddings": embeddings, "item_ids": item_ids}, cache_path)

        return cls(embeddings=embeddings, item_ids=item_ids)

    @classmethod
    def from_embeddings_file(cls, path: str | Path) -> "ItemEmbeddingDataset":
        """Load dataset from saved embeddings file."""
        data = torch.load(path)
        return cls(embeddings=data["embeddings"], item_ids=data["item_ids"])

    def save(self, path: str | Path) -> None:
        """Save embeddings to file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"embeddings": self.embeddings, "item_ids": self.item_ids}, path)


def create_dummy_catalogue(
    num_items: int = 1000,
    output_path: str | Path = "data/catalogue.json",
) -> None:
    """
    Create a dummy catalogue for testing.

    Args:
        num_items: Number of items to generate
        output_path: Path to save catalogue
    """
    import random

    categories = ["Electronics", "Books", "Clothing", "Home", "Sports", "Beauty"]
    adjectives = ["Premium", "Budget", "Professional", "Compact", "Luxury", "Essential"]
    nouns = ["Widget", "Gadget", "Device", "Tool", "Accessory", "Kit"]

    items = []
    for i in range(num_items):
        category = random.choice(categories)
        adj = random.choice(adjectives)
        noun = random.choice(nouns)

        items.append(
            {
                "id": f"item_{i:06d}",
                "title": f"{adj} {category} {noun}",
                "description": f"A high-quality {adj.lower()} {noun.lower()} for {category.lower()} enthusiasts.",
                "category": category,
                "price": round(random.uniform(10, 500), 2),
            }
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump({"items": items}, f, indent=2)

    print(f"Created dummy catalogue with {num_items} items at {output_path}")
