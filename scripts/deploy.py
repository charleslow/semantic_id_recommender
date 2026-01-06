"""
Deployment script for Modal.

Usage:
    python -m scripts.deploy --upload checkpoints/llm
    python -m scripts.deploy --deploy
    python -m scripts.deploy --test "wireless mouse"
"""

import argparse
import json
from pathlib import Path

import modal

from src.inference.constants import (
    CATALOGUE_FILE,
    MODAL_VOLUME_NAME,
    MODEL_DIR,
    SEMANTIC_IDS_FILE,
)


def upload_model_to_volume(
    model_path: str,
    catalogue_path: str,
    semantic_ids_path: str,
):
    """
    Upload model and data files to Modal volume.

    Args:
        model_path: Path to fine-tuned model
        catalogue_path: Path to catalogue JSON
        semantic_ids_path: Path to semantic IDs JSON
    """
    volume = modal.Volume.from_name(MODAL_VOLUME_NAME, create_if_missing=True)

    print("Uploading model to Modal volume...")

    with volume.batch_upload() as batch:
        # Upload model files
        model_dir = Path(model_path)
        for file in model_dir.rglob("*"):
            if file.is_file():
                rel_path = file.relative_to(model_dir)
                remote_path = f"/{MODEL_DIR}/{rel_path}"
                batch.put_file(str(file), remote_path)
                print(f"  Uploaded: {rel_path}")

        # Upload catalogue
        if Path(catalogue_path).exists():
            batch.put_file(catalogue_path, f"/{CATALOGUE_FILE}")
            print(f"  Uploaded: {CATALOGUE_FILE}")

        # Upload semantic IDs
        if Path(semantic_ids_path).exists():
            batch.put_file(semantic_ids_path, f"/{SEMANTIC_IDS_FILE}")
            print(f"  Uploaded: {SEMANTIC_IDS_FILE}")

    print("Upload complete!")


def deploy_app():
    """Deploy the Modal app."""
    from src.inference.modal_app import app

    print("Deploying to Modal...")
    modal.runner.deploy_app(app)
    print("Deployment complete!")


def test_deployment(query: str, api_url: str | None = None):
    """
    Test the deployed recommender.

    Args:
        query: Test query
        api_url: Optional API URL (if not provided, uses Modal remote call)
    """
    if api_url:
        import requests

        response = requests.post(
            api_url,
            json={"query": query, "num_recommendations": 5},
            headers={"Content-Type": "application/json"},
        )
        results = response.json()
    else:
        from src.inference.modal_app import Recommender, app

        with app.run():
            recommender = Recommender()
            results = recommender.recommend.remote(query, num_recommendations=5)

    print(f"Query: {query}")
    print(f"Results: {json.dumps(results, indent=2)}")


def main():
    parser = argparse.ArgumentParser(description="Deploy semantic ID recommender")
    parser.add_argument(
        "--upload",
        type=str,
        help="Upload model from this path to Modal volume",
    )
    parser.add_argument(
        "--catalogue",
        type=str,
        default="data/catalogue.jsonl",
        help="Path to catalogue JSONL",
    )
    parser.add_argument(
        "--semantic-ids",
        type=str,
        default="data/semantic_ids.jsonl",
        help="Path to semantic IDs JSONL",
    )
    parser.add_argument(
        "--deploy",
        action="store_true",
        help="Deploy the Modal app",
    )
    parser.add_argument(
        "--test",
        type=str,
        help="Test with this query",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        help="API URL for testing (optional)",
    )
    args = parser.parse_args()

    if args.upload:
        upload_model_to_volume(
            args.upload,
            args.catalogue,
            args.semantic_ids,
        )

    if args.deploy:
        deploy_app()

    if args.test:
        test_deployment(args.test, args.api_url)


if __name__ == "__main__":
    main()
