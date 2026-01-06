"""
Download W&B artifacts for Modal deployment.

Downloads the fine-tuned LLM and RQ-VAE data artifacts to the outputs/ directory.
"""

import argparse
from pathlib import Path


def download_artifacts(
    project: str,
    model_artifact: str = "llm-stage2:latest",
    data_artifact: str = "rqvae-model-data:latest",
    output_dir: str = "outputs",
) -> None:
    """
    Download model and data artifacts from W&B.

    Args:
        project: W&B project name (e.g., 'username/semantic-id-recommender')
        model_artifact: Model artifact name with version
        data_artifact: Data artifact name with version
        output_dir: Local directory to save artifacts
    """
    import wandb

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize W&B run for artifact download
    run = wandb.init(project=project.split("/")[-1], job_type="download")

    try:
        # Download LLM model artifact
        model_dir = output_path / "llm"
        print(f"\n📥 Downloading model artifact: {project}/{model_artifact}")
        model_artifact_ref = run.use_artifact(f"{project}/{model_artifact}")
        model_artifact_ref.download(root=str(model_dir))
        print(f"✅ Model saved to: {model_dir}")

        # Download RQ-VAE data artifact (catalogue + semantic IDs)
        data_dir = output_path / "data"
        print(f"\n📥 Downloading data artifact: {project}/{data_artifact}")
        data_artifact_ref = run.use_artifact(f"{project}/{data_artifact}")
        data_artifact_ref.download(root=str(data_dir))
        print(f"✅ Data saved to: {data_dir}")

        # List downloaded files
        print("\n📁 Downloaded files:")
        for subdir in [model_dir, data_dir]:
            if subdir.exists():
                for f in sorted(subdir.rglob("*")):
                    if f.is_file():
                        size_mb = f.stat().st_size / (1024 * 1024)
                        print(f"   {f.relative_to(output_path)} ({size_mb:.1f} MB)")

        print("\n🚀 Ready for Modal deployment! Run:")
        print("   python -m scripts.deploy \\")
        print(f"       --upload {model_dir} \\")
        print(f"       --catalogue {data_dir}/catalogue.jsonl \\")
        print(f"       --semantic-ids {data_dir}/semantic_ids.jsonl \\")
        print("       --deploy")

    finally:
        run.finish()


def main():
    parser = argparse.ArgumentParser(
        description="Download W&B artifacts for Modal deployment"
    )
    parser.add_argument(
        "--project",
        type=str,
        required=True,
        help="W&B project (e.g., 'username/semantic-id-recommender')",
    )
    parser.add_argument(
        "--model-artifact",
        type=str,
        default="llm-stage2:latest",
        help="Model artifact name (default: llm-stage2:latest)",
    )
    parser.add_argument(
        "--data-artifact",
        type=str,
        default="rqvae-model-data:latest",
        help="Data artifact name (default: rqvae-model-data:latest)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory (default: outputs/)",
    )
    args = parser.parse_args()

    download_artifacts(
        project=args.project,
        model_artifact=args.model_artifact,
        data_artifact=args.data_artifact,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
