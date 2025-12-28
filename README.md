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

### src/llm/data.py

This file handles the preparation of training data for the LLM finetune to learn semantic IDs.

`load_catalogue_with_semantic_ids`: Loads the catalogue of items into a huggingface dataset.
- Also adds the string `semantic_id` as a field
- In `strict` mode, we raise an error if any item has no semantic ID

`generate_training_examples`: based on the catalogue of items, generates training examples in the form of query and response.
- There are two categories of training data:
    - `predict_semantic_id`: the task is to predict a `semantic_id` given the query. For example, "What is the semantic ID of {title}?"
    - `predict_attribute`: the task is to predict a field attribute given the `semantic_id` of an item. For example, "What is the {field_name} of {semantic_id}?"
- The user specifies these with a list of `query_templates`, which will be randomly selected for each training example
- We return `num_examples_per_item` of examples per item in the catalogue.
- Note that for the `predict_semantic_id` task, we append special `[REC]` token to the end of query to trigger recommendation behaviour

`format_as_messages`: a simple formatting function to get the `query` and `response` from the generated examples and put them into the `messages` format using `user` and `assistant`. Also adds a simple system prompt.

`prepare_training_data`: uses the functions above and generates the training and validation datasets.
- Calls `load_catalogue_with_semantic_ids` to load the item catalogue
- Calls `generate_training_data` to generate examples
- Calls `format_as_messages` to put into message format
- Shuffles and splits into train and val
- Writes into output jsonl files

`SemanticIDDataset`: a simple torch dataset to yield examples from the training and val examples defined above.
- Note that each item contains:
    - `messages`: the list of `system`, `user`, `assistant` messages for training
    - `query`
    - `response`
    - `type`: e.g. `predict_semantic_id`
    - `item_id`

### src/llm/finetune.py

`finetune_model`: the main method for finetuning. 






