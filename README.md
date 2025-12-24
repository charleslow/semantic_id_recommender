# Semantic ID Recommender

POC of training a semantic ID-based recommender for handling flexible queries.

## Overview

This project implements an LLM-based recommender using semantic IDs:

1. **RQ-VAE** learns compact semantic IDs for catalogue items
2. **Fine-tuned LLM** predicts semantic IDs from natural language queries
3. **Constrained generation** ensures valid recommendations

## Quick Start

```bash
# Install dependencies
uv sync
source .venv/bin/activate

# Train RQ-VAE (generates semantic IDs)
python -m scripts.train_rqvae --catalogue data/mcf_articles.jsonl

# Fine-tune LLM
python -m scripts.finetune_llm

# Deploy to Modal
python -m scripts.deploy --upload-model checkpoints/llm --deploy
```

## Documentation

- [instructions.md](instructions.md) - Full setup and usage guide
- [specs.md](specs.md) - Technical specification
- [notebooks/test_training.ipynb](notebooks/test_training.ipynb) - Colab test notebook

## Project Structure

```
├── src/
│   ├── rqvae/          # Semantic ID learning
│   ├── llm/            # LLM fine-tuning
│   ├── inference/      # Modal deployment
│   └── frontend/       # Gradio UI
├── scripts/            # Training & deployment scripts
├── notebooks/          # Colab notebooks
└── data/               # Catalogue & generated data
```

## Demonstrated on

MCF articles dataset.
