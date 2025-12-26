# Semantic ID Recommender

POC of training a semantic ID-based recommender for handling flexible queries.

## Overview

This project implements an LLM-based recommender using semantic IDs:

1. **RQ-VAE** learns compact semantic IDs for catalogue items
2. **Fine-tuned LLM** predicts semantic IDs from natural language queries
3. **Constrained generation** ensures valid recommendations

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

## Code

### src/rqvae/dataset.py

Contains a simple `ItemEmbeddingDataset` class that embeds each item from the catalogue (a `.jsonl` file) and caches the embeddings. Each item index contains one item and its embedding.
- `from_catalogue`: the main method to get items and embeddings from the path of a catalogue file
- Can probably optimize this when we get to larger item catalogues

### src/rqvae/model.py

Contains `SemanticRQVAE` which is a wrapper around `ResidualVQ` from `vector_quantize_pytorch`. Can probably experiment with the other variants from the library later.

At init:
- Set up simple `encoder` and `decoder`. `encoder` projects from item embedding dimension to codebook `hidden_dim`. `decoder` projects from `hidden_dim` back to item embedding dimension. 

The main method is `quantize`, which quantizes the encoder output `z` into discrete indices and its quantized embedding representation.
- `ResidualVQ` is designed for sequence modelling tasks where we have `seq_len` number of embeddings in a sequence. Since we only have one item, we artificially `unsqueeze` to account for sequence dimension and `squeeze` again.
- `quantized, indices, commit_loss = self.rq(z)`
    - This is the main quantization step from `ResidualVQ`
    - `quantized` is the quantized embedding used to compute reconstruction loss later
    - `commit_loss` is the commitment loss, which encourages the encoder output to be close to the codebook

In the `forward` pass:
- First we encode the original item embedding `x`
- Then we run `quantize`
- The quantized embedding `quantized` is decoded to get `reconstructed`
- Reconstruction loss is computed as `F.mse_loss(reconstructed, x)`
- These are returned as a tuple

Some utility functions:
- `get_semantic_ids(x)`: convert item embedding to semantic IDs
    - `x: (batch_size, embedding_dim)`
    - Returns semantic ID integer codes `indices: (batch_size, num_quantizers)`
- `semantic_id_to_string(indices)`: convert semantic ID integer codes to string for LLM usage
    - `indices: (batch_size, num_quantizers)`
    - Returns `results: list[str] (batch_size)` containing string representation of each item in batch, e.g. `[SEM_START][SEM_0_1][SEM_1_2][SEM_END]` etc.
- `string_to_semantic_id(s: str)`: parse semantic ID string back into integer codes
    - Uses regex-based parsing
    - Returns a `list[int]` of integer codes

`compute_codebook_stats` computes some statistics measuring the effectiveness of the residual quantization model. Meant for tracking progress during training.
- Input is `indices: (batch_size, num_quantizers)`
- There are two main statistics we want to compute:
    - Codebook usage: track the % of codes in each level used at least once in each batch
    - Perplexity: track `exp(entropy)` of the codes in each level to measure distribution
        - Entropy is `sum(-p log p)` over each code in the level, ranges from `0` to `log K` where `K` is the number of codes
        - Perplexity ranges from `1` (single point, bad) to `K` (uniformly distributed)

### src/rqvae/trainer.py

Contains the lightning module for training the `RQVAE` model:
- `training_step` / `validation_step`:
    - Runs the batch through the model
    - Computes `loss = recon_loss + commit_loss`
    - Uses `model.compute_codebook_stats` to get codebook stats and log them




