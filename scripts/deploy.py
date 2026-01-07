"""
Deployment script for Modal.

Usage:
    python -m scripts.deploy              # Deploy the app
    python -m scripts.deploy --test "wireless mouse"  # Test deployment
"""

import argparse
import json


def deploy_app():
    """Deploy the Modal app."""
    from src.inference.modal_app import app

    print("🚀 Deploying to Modal...")
    app.deploy()
    print("✅ Deployment complete!")


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
            json={"query": query},
            headers={"Content-Type": "application/json"},
        )
        results = response.json()
    else:
        import modal

        # Call the already-deployed app (no ephemeral instance)
        Recommender = modal.Cls.from_name("semantic-id-recommender", "Recommender")
        recommender = Recommender()
        results = recommender.recommend.remote(query)

    print(f"Query: {query}")
    print(f"Results: {json.dumps(results, indent=2)}")


def main():
    parser = argparse.ArgumentParser(description="Deploy semantic ID recommender")
    parser.add_argument(
        "--deploy",
        action="store_true",
        default=True,
        help="Deploy the Modal app (default: True)",
    )
    parser.add_argument(
        "--no-deploy",
        action="store_true",
        help="Skip deployment (useful when only testing)",
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

    if args.deploy and not args.no_deploy:
        deploy_app()

    if args.test:
        test_deployment(args.test, args.api_url)


if __name__ == "__main__":
    main()
