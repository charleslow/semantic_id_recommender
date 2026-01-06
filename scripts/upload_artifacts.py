"""
Upload artifacts to Modal volume for deployment.

Uploads the fine-tuned LLM model, catalogue, and semantic ID mappings
to a Modal volume for use by the serverless recommender.

Usage:
    python -m scripts.upload_artifacts \
        --model outputs/llm \
        --catalogue outputs/data/catalogue.jsonl \
        --semantic-ids outputs/data/semantic_ids.jsonl
"""

import argparse
from pathlib import Path

import modal

from src.inference.constants import (
    CATALOGUE_FILE,
    MODAL_MOUNT_PATH,
    MODAL_VOLUME_NAME,
    MODEL_DIR,
    SEMANTIC_IDS_FILE,
)


def upload_artifacts(
    model_path: str,
    catalogue_path: str,
    semantic_ids_path: str,
) -> None:
    """
    Upload model and data files to Modal volume.

    Args:
        model_path: Path to fine-tuned model directory
        catalogue_path: Path to catalogue JSONL file
        semantic_ids_path: Path to semantic IDs JSONL file

    Remote paths on Modal volume:
        - Model: /model/semantic-recommender/
        - Catalogue: /model/catalogue.jsonl
        - Semantic IDs: /model/semantic_ids.jsonl
    """
    volume = modal.Volume.from_name(MODAL_VOLUME_NAME, create_if_missing=True)

    model_dir = Path(model_path)
    catalogue_file = Path(catalogue_path)
    semantic_ids_file = Path(semantic_ids_path)

    # Validate paths
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_path}")
    if not catalogue_file.exists():
        raise FileNotFoundError(f"Catalogue file not found: {catalogue_path}")
    if not semantic_ids_file.exists():
        raise FileNotFoundError(f"Semantic IDs file not found: {semantic_ids_path}")

    print(f"📤 Uploading to Modal volume: {MODAL_VOLUME_NAME}")
    print(f"   Mount path: {MODAL_MOUNT_PATH}\n")

    uploaded_count = 0
    total_size = 0

    with volume.batch_upload() as batch:
        # Upload model files
        print(f"📁 Model: {model_path} → {MODAL_MOUNT_PATH}/{MODEL_DIR}/")
        for file in model_dir.rglob("*"):
            if file.is_file():
                rel_path = file.relative_to(model_dir)
                remote_path = f"/{MODEL_DIR}/{rel_path}"
                batch.put_file(str(file), remote_path)
                file_size = file.stat().st_size
                total_size += file_size
                uploaded_count += 1
                print(f"   • {rel_path} ({file_size / 1024 / 1024:.1f} MB)")

        # Upload catalogue
        print(f"\n📄 Catalogue: {catalogue_path} → {MODAL_MOUNT_PATH}/{CATALOGUE_FILE}")
        batch.put_file(str(catalogue_file), f"/{CATALOGUE_FILE}")
        total_size += catalogue_file.stat().st_size
        uploaded_count += 1

        # Upload semantic IDs
        print(
            f"📄 Semantic IDs: {semantic_ids_path} → {MODAL_MOUNT_PATH}/{SEMANTIC_IDS_FILE}"
        )
        batch.put_file(str(semantic_ids_file), f"/{SEMANTIC_IDS_FILE}")
        total_size += semantic_ids_file.stat().st_size
        uploaded_count += 1

    print("\n✅ Upload complete!")
    print(f"   Files: {uploaded_count}")
    print(f"   Total size: {total_size / 1024 / 1024:.1f} MB")
    print("\n🚀 Ready to deploy! Run:")
    print("   python -m scripts.deploy")


def main():
    parser = argparse.ArgumentParser(
        description="Upload artifacts to Modal volume for deployment"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="outputs/llm",
        help="Path to fine-tuned model directory (default: outputs/llm)",
    )
    parser.add_argument(
        "--catalogue",
        type=str,
        default="outputs/data/catalogue.jsonl",
        help="Path to catalogue JSONL file (default: outputs/data/catalogue.jsonl)",
    )
    parser.add_argument(
        "--semantic-ids",
        type=str,
        default="outputs/data/semantic_ids.jsonl",
        help="Path to semantic IDs JSONL file (default: outputs/data/semantic_ids.jsonl)",
    )
    args = parser.parse_args()

    upload_artifacts(
        model_path=args.model,
        catalogue_path=args.catalogue,
        semantic_ids_path=args.semantic_ids,
    )


if __name__ == "__main__":
    main()
