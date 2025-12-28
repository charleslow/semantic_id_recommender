"""
Gradio frontend for the semantic ID recommender.
"""

import json
from pathlib import Path

import gradio as gr


def create_demo(
    api_url: str | None = None,
    local_model_path: str | None = None,
    catalogue_path: str | None = None,
    semantic_ids_path: str | None = None,
):
    """
    Create Gradio demo for the recommender.

    Can run in two modes:
    1. API mode: Calls Modal deployment
    2. Local mode: Loads model locally

    Args:
        api_url: Modal API endpoint URL (for API mode)
        local_model_path: Path to local model (for local mode)
        catalogue_path: Path to catalogue JSON
        semantic_ids_path: Path to semantic IDs mapping
    """
    # Load catalogue for display
    catalogue = {}
    if catalogue_path and Path(catalogue_path).exists():
        with open(catalogue_path) as f:
            data = json.load(f)
            items = data if isinstance(data, list) else data.get("items", [])
            catalogue = {str(item.get("id", i)): item for i, item in enumerate(items)}

    # Load semantic ID mapping
    semantic_to_item = {}
    if semantic_ids_path and Path(semantic_ids_path).exists():
        with open(semantic_ids_path) as f:
            mapping = json.load(f)
            semantic_to_item = mapping.get("semantic_to_item", {})

    # Local model and generator (lazy loaded)
    generator = None

    def load_local_generator():
        nonlocal generator
        if generator is None and local_model_path:
            from src.llm import load_finetuned_model, SemanticIDGenerator
            model, tokenizer = load_finetuned_model(local_model_path)
            generator = SemanticIDGenerator(model, tokenizer)
        return generator

    def recommend_api(query: str, num_results: int) -> list[dict]:
        """Call Modal API for recommendations."""
        import requests

        response = requests.post(
            api_url,
            json={"query": query, "num_recommendations": num_results},
            headers={"Content-Type": "application/json"},
        )

        if response.status_code == 200:
            return response.json().get("recommendations", [])
        return []

    def recommend_local(query: str, num_results: int) -> list[dict]:
        """Generate recommendations locally."""
        gen = load_local_generator()
        if gen is None:
            return []

        semantic_id = gen.generate(query)
        item_id = semantic_to_item.get(semantic_id)

        if not item_id:
            return []

        item = catalogue.get(item_id, {})
        return [{
            "item_id": item_id,
            "semantic_id": semantic_id,
            **item,
        }]

    def get_recommendations(query: str, num_results: int = 5) -> str:
        """Main recommendation function."""
        if not query.strip():
            return "Please enter a query."

        # Choose mode
        if api_url:
            results = recommend_api(query, num_results)
        elif local_model_path:
            results = recommend_local(query, num_results)
        else:
            return "No model configured. Set api_url or local_model_path."

        if not results:
            return "No recommendations found."

        # Format results
        output = []
        for i, item in enumerate(results, 1):
            item_text = f"**{i}. {item.get('title', 'Unknown')}**\n"
            if "description" in item:
                item_text += f"_{item['description']}_\n"
            if "category" in item:
                item_text += f"Category: {item['category']}\n"
            if "price" in item:
                item_text += f"Price: ${item['price']:.2f}\n"
            item_text += f"ID: `{item.get('item_id', 'N/A')}`\n"
            item_text += f"Semantic ID: `{item.get('semantic_id', 'N/A')}`"
            output.append(item_text)

        return "\n\n---\n\n".join(output)

    # Create Gradio interface
    with gr.Blocks(title="Semantic ID Recommender") as demo:
        gr.Markdown("# Semantic ID Recommender")
        gr.Markdown(
            "Enter a query to get product recommendations using semantic ID matching."
        )

        with gr.Row():
            with gr.Column(scale=3):
                query_input = gr.Textbox(
                    label="Search Query",
                    placeholder="e.g., wireless mouse with ergonomic design",
                    lines=2,
                )
            with gr.Column(scale=1):
                num_results = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=1,
                    label="Number of Results",
                )

        submit_btn = gr.Button("Get Recommendations", variant="primary")

        output = gr.Markdown(label="Recommendations")

        # Example queries
        gr.Examples(
            examples=[
                ["wireless keyboard"],
                ["premium headphones"],
                ["budget laptop stand"],
                ["ergonomic office chair"],
            ],
            inputs=query_input,
        )

        # Event handlers
        submit_btn.click(
            fn=get_recommendations,
            inputs=[query_input, num_results],
            outputs=output,
        )
        query_input.submit(
            fn=get_recommendations,
            inputs=[query_input, num_results],
            outputs=output,
        )

    return demo


def main():
    """Run the Gradio demo."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", type=str, help="Modal API endpoint URL")
    parser.add_argument("--model-path", type=str, help="Local model path")
    parser.add_argument("--catalogue", type=str, default="data/catalogue.json")
    parser.add_argument("--semantic-ids", type=str, default="data/semantic_ids.json")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create public link")
    args = parser.parse_args()

    demo = create_demo(
        api_url=args.api_url,
        local_model_path=args.model_path,
        catalogue_path=args.catalogue,
        semantic_ids_path=args.semantic_ids,
    )

    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
