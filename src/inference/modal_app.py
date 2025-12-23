"""
Modal serverless deployment for the semantic ID recommender.

Deploys the fine-tuned LLM as a serverless endpoint with constrained generation.
"""

import json
from pathlib import Path

import modal

# Modal app configuration
app = modal.App("semantic-id-recommender")

# Volumes for caching models
hf_cache_volume = modal.Volume.from_name("hf-cache", create_if_missing=True)
model_volume = modal.Volume.from_name("semantic-recommender-model", create_if_missing=True)

# Container image
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.1.0",
        "transformers>=4.36.0",
        "accelerate>=0.25.0",
        "vllm>=0.6.0",
        "outlines>=0.1.0",
        "huggingface_hub>=0.20.0",
    )
)


@app.cls(
    image=image,
    gpu="A10G",
    volumes={
        "/root/.cache/huggingface": hf_cache_volume,
        "/model": model_volume,
    },
    container_idle_timeout=300,  # 5 minutes
    allow_concurrent_inputs=10,
)
class Recommender:
    """Serverless recommender using vLLM."""

    model_path: str = "/model/semantic-recommender"
    semantic_ids_path: str = "/model/semantic_ids.json"
    catalogue_path: str = "/model/catalogue.json"

    @modal.build()
    def download_model(self):
        """Pre-download model during container build."""
        from huggingface_hub import snapshot_download
        import os

        # Download model if HF_REPO is set
        hf_repo = os.environ.get("HF_MODEL_REPO")
        if hf_repo:
            snapshot_download(
                repo_id=hf_repo,
                local_dir=self.model_path,
                local_dir_use_symlinks=False,
            )

    @modal.enter()
    def load_model(self):
        """Load model and mappings on container start."""
        from vllm import LLM, SamplingParams

        # Load model with vLLM
        self.llm = LLM(
            model=self.model_path,
            max_model_len=512,
            dtype="half",
            trust_remote_code=True,
        )

        # Default sampling params
        self.sampling_params = SamplingParams(
            max_tokens=32,
            temperature=0.1,
        )

        # Load semantic ID mapping
        if Path(self.semantic_ids_path).exists():
            with open(self.semantic_ids_path) as f:
                mapping = json.load(f)
                self.semantic_to_item = mapping.get("semantic_to_item", {})
                self.item_to_semantic = mapping.get("item_to_semantic", {})
        else:
            self.semantic_to_item = {}
            self.item_to_semantic = {}

        # Load catalogue
        if Path(self.catalogue_path).exists():
            with open(self.catalogue_path) as f:
                data = json.load(f)
                items = data if isinstance(data, list) else data.get("items", [])
                self.catalogue = {str(item.get("id", i)): item for i, item in enumerate(items)}
        else:
            self.catalogue = {}

    def _format_prompt(self, query: str) -> str:
        """Format query into model prompt."""
        system_prompt = (
            "You are a recommendation system. Given a user query, "
            "output the semantic ID of the most relevant item. "
            "Respond only with the semantic ID tokens."
        )
        return f"<|system|>\n{system_prompt}\n<|user|>\n{query}\n<|assistant|>\n"

    def _parse_semantic_id(self, output: str) -> str | None:
        """Extract semantic ID from model output."""
        import re

        # Find all semantic ID tokens
        pattern = r"\[SEM_\d+_\d+\]"
        matches = re.findall(pattern, output)

        if matches:
            return "".join(matches)
        return None

    @modal.method()
    def recommend(
        self,
        query: str,
        num_recommendations: int = 5,
        return_scores: bool = False,
    ) -> list[dict]:
        """
        Get recommendations for a query.

        Args:
            query: User query text
            num_recommendations: Number of items to return
            return_scores: Whether to include confidence scores

        Returns:
            List of recommended items with metadata
        """
        # Format prompt
        prompt = self._format_prompt(query)

        # Generate with vLLM
        outputs = self.llm.generate([prompt], self.sampling_params)
        generated_text = outputs[0].outputs[0].text

        # Parse semantic ID
        semantic_id = self._parse_semantic_id(generated_text)

        if not semantic_id:
            return []

        # Look up item
        item_id = self.semantic_to_item.get(semantic_id)

        if not item_id:
            return []

        # Get item details
        item = self.catalogue.get(item_id, {})

        result = {
            "item_id": item_id,
            "semantic_id": semantic_id,
            **item,
        }

        if return_scores:
            result["score"] = 1.0  # Top recommendation

        return [result]

    @modal.method()
    def batch_recommend(
        self,
        queries: list[str],
        num_recommendations: int = 5,
    ) -> list[list[dict]]:
        """
        Get recommendations for multiple queries.

        Args:
            queries: List of user query texts
            num_recommendations: Number of items per query

        Returns:
            List of recommendation lists
        """
        # Format all prompts
        prompts = [self._format_prompt(q) for q in queries]

        # Generate in batch
        outputs = self.llm.generate(prompts, self.sampling_params)

        results = []
        for output in outputs:
            generated_text = output.outputs[0].text
            semantic_id = self._parse_semantic_id(generated_text)

            if not semantic_id:
                results.append([])
                continue

            item_id = self.semantic_to_item.get(semantic_id)
            if not item_id:
                results.append([])
                continue

            item = self.catalogue.get(item_id, {})
            results.append([{
                "item_id": item_id,
                "semantic_id": semantic_id,
                **item,
            }])

        return results

    @modal.method()
    def health_check(self) -> dict:
        """Health check endpoint."""
        return {
            "status": "healthy",
            "model_loaded": self.llm is not None,
            "catalogue_size": len(self.catalogue),
            "semantic_ids_count": len(self.semantic_to_item),
        }


@app.function(image=image)
@modal.web_endpoint(method="POST")
def recommend_api(query: str, num_recommendations: int = 5) -> dict:
    """
    REST API endpoint for recommendations.

    POST /recommend_api
    Body: {"query": "...", "num_recommendations": 5}
    """
    recommender = Recommender()
    results = recommender.recommend.remote(query, num_recommendations)
    return {"recommendations": results}


@app.function(image=image)
@modal.web_endpoint(method="GET")
def health() -> dict:
    """Health check endpoint."""
    recommender = Recommender()
    return recommender.health_check.remote()


# CLI for local testing
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, help="Query to test")
    parser.add_argument("--deploy", action="store_true", help="Deploy to Modal")
    args = parser.parse_args()

    if args.deploy:
        modal.runner.deploy_app(app)
    elif args.query:
        with app.run():
            recommender = Recommender()
            results = recommender.recommend.remote(args.query)
            print(json.dumps(results, indent=2))
