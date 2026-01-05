"""
Trainer callbacks for LLM fine-tuning evaluation.
"""

import random
import re
from dataclasses import asdict
from pathlib import Path

import torch
import wandb
from datasets import Dataset
from transformers import PreTrainedTokenizerBase, TrainerCallback

from .data import DEFAULT_SYSTEM_PROMPT
from .inference import SemanticIDGenerator, extract_semantic_id


def _parse_semantic_id_tokens(semantic_id: str) -> list[str]:
    """
    Parse a semantic ID string into its component tokens.

    Args:
        semantic_id: String like "[SEM_START][SEM_0_5][SEM_1_10][SEM_END]"

    Returns:
        List of SEM tokens like ["[SEM_0_5]", "[SEM_1_10]"]
    """
    pattern = r"\[SEM_\d+_\d+\]"
    return re.findall(pattern, semantic_id)


def _compute_prefix_accuracy_target(
    predictions: list[str],
    targets: list[str],
    num_quantizers: int,
) -> dict[str, float]:
    """
    Compute accuracy at each prefix level by matching against target.

    Args:
        predictions: List of predicted semantic ID strings
        targets: List of target semantic ID strings
        num_quantizers: Number of quantizer levels

    Returns:
        Dict mapping "prefix_k_accuracy" to accuracy value for k=1..num_quantizers
    """
    prefix_correct = {k: 0 for k in range(1, num_quantizers + 1)}
    total = len(predictions)

    if total == 0:
        return {f"prefix_{k}_accuracy": 0.0 for k in range(1, num_quantizers + 1)}

    for pred, target in zip(predictions, targets):
        pred_tokens = _parse_semantic_id_tokens(pred)
        target_tokens = _parse_semantic_id_tokens(target)

        for k in range(1, num_quantizers + 1):
            pred_prefix = pred_tokens[:k]
            target_prefix = target_tokens[:k]
            if pred_prefix == target_prefix and len(pred_prefix) == k:
                prefix_correct[k] += 1

    return {
        f"prefix_{k}_accuracy": prefix_correct[k] / total
        for k in range(1, num_quantizers + 1)
    }


def _compute_prefix_accuracy_any(
    predictions: list[str],
    num_quantizers: int,
) -> dict[str, float]:
    """
    Compute accuracy at each prefix level by checking if tokens are valid.

    A token is valid if it matches the pattern [SEM_X_Y] where X is the
    quantizer level (0 to num_quantizers-1) and Y is any codebook index.

    Args:
        predictions: List of predicted semantic ID strings
        num_quantizers: Number of quantizer levels

    Returns:
        Dict mapping "prefix_k_accuracy" to accuracy value for k=1..num_quantizers
    """
    prefix_valid = {k: 0 for k in range(1, num_quantizers + 1)}
    total = len(predictions)

    if total == 0:
        return {f"prefix_{k}_accuracy": 0.0 for k in range(1, num_quantizers + 1)}

    for pred in predictions:
        pred_tokens = _parse_semantic_id_tokens(pred)

        for k in range(1, num_quantizers + 1):
            # Check if we have at least k tokens and all k tokens are valid
            if len(pred_tokens) >= k:
                # Check each token has the correct quantizer level
                all_valid = True
                for i in range(k):
                    token = pred_tokens[i]
                    # Extract quantizer level from token like [SEM_0_5]
                    match = re.match(r"\[SEM_(\d+)_\d+\]", token)
                    if not match or int(match.group(1)) != i:
                        all_valid = False
                        break
                if all_valid:
                    prefix_valid[k] += 1

    return {
        f"prefix_{k}_accuracy": prefix_valid[k] / total
        for k in range(1, num_quantizers + 1)
    }


def _is_valid_semantic_id(semantic_id: str, num_quantizers: int) -> bool:
    """Check if a semantic ID is valid (has correct format and token count)."""
    if not semantic_id.startswith("[SEM_START]") or not semantic_id.endswith(
        "[SEM_END]"
    ):
        return False
    tokens = _parse_semantic_id_tokens(semantic_id)
    return len(tokens) == num_quantizers


class SemanticIDEvalCallback(TrainerCallback):
    """
    Trainer callback to evaluate semantic ID generation accuracy at each prefix level.

    Only evaluates examples of type "predict_semantic_id".
    """

    def __init__(
        self,
        val_dataset: Dataset,
        tokenizer: PreTrainedTokenizerBase,
        num_quantizers: int = 4,
        max_eval_samples: int = 100,
        generation_max_new_tokens: int = 32,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ):
        """
        Initialize the callback.

        Args:
            val_dataset: Validation dataset with 'messages' and 'type' fields
            tokenizer: Tokenizer for generation
            num_quantizers: Number of RQ-VAE quantizers (prefix levels)
            max_eval_samples: Maximum samples to evaluate (for speed)
            generation_max_new_tokens: Max tokens to generate
            system_prompt: System prompt for generation
        """
        self.tokenizer = tokenizer
        self.num_quantizers = num_quantizers
        self.max_eval_samples = max_eval_samples
        self.generation_max_new_tokens = generation_max_new_tokens
        self.system_prompt = system_prompt

        # Filter for predict_semantic_id examples only
        if "type" in val_dataset.column_names:
            self.eval_dataset = val_dataset.filter(
                lambda x: x["type"] == "predict_semantic_id"
            )
        else:
            self.eval_dataset = val_dataset

        # Limit samples for speed
        if len(self.eval_dataset) > max_eval_samples:
            self.eval_dataset = self.eval_dataset.select(range(max_eval_samples))

    def _generate_predictions(self, model) -> tuple[list[str], list[str]]:
        """Generate predictions for the eval dataset."""
        predictions = []
        targets = []

        model.eval()
        device = next(model.parameters()).device

        for example in self.eval_dataset:
            messages = example["messages"]
            # Target is the assistant response
            target = messages[-1]["content"]
            targets.append(target)

            # Build prompt without the assistant response
            prompt_messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": messages[1]["content"]},
            ]
            prompt = self.tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = self.tokenizer(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.generation_max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
            pred = extract_semantic_id(generated, self.tokenizer.eos_token)
            predictions.append(pred)

        return predictions, targets

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        """Run semantic ID evaluation after each evaluation step."""
        if model is None:
            return

        print("\n--- Semantic ID Prefix Accuracy Evaluation ---")
        predictions, targets = self._generate_predictions(model)

        # Compute both types of prefix accuracy
        target_metrics = _compute_prefix_accuracy_target(
            predictions, targets, self.num_quantizers
        )
        any_metrics = _compute_prefix_accuracy_any(predictions, self.num_quantizers)

        # Add exact match accuracy
        exact_matches = sum(p == t for p, t in zip(predictions, targets))
        exact_match_accuracy = exact_matches / len(predictions) if predictions else 0.0

        print("  Target prefix accuracy (must match target):")
        for key, value in target_metrics.items():
            print(f"    {key}: {value:.4f}")
        print(f"    exact_match_accuracy: {exact_match_accuracy:.4f}")

        print("  Any prefix accuracy (any valid token):")
        for key, value in any_metrics.items():
            print(f"    {key}: {value:.4f}")

        print("----------------------------------------------\n")

        # Log metrics to wandb
        logged_metrics = {}
        for k, v in target_metrics.items():
            logged_metrics[f"eval_semantic_id_target/{k}"] = v
        logged_metrics["eval_semantic_id_target/exact_match_accuracy"] = (
            exact_match_accuracy
        )

        for k, v in any_metrics.items():
            logged_metrics[f"eval_semantic_id_any/{k}"] = v

        logged_metrics["eval_semantic_id/num_samples"] = len(predictions)

        if wandb.run is not None:
            wandb.log(logged_metrics, step=state.global_step)


class RecommendationTestCallback(TrainerCallback):
    """
    Trainer callback to test recommendation generation with example queries.

    Generates semantic IDs using beam search and displays the top result's metadata.
    """

    def __init__(
        self,
        test_queries: list[str],
        tokenizer: PreTrainedTokenizerBase,
        semantic_id_to_item: dict[str, dict],
        num_quantizers: int = 4,
        num_beams: int = 10,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        truncate_length: int = 120,
    ):
        """
        Initialize the callback.

        Args:
            test_queries: List of recommendation queries to test
            tokenizer: Tokenizer for generation
            semantic_id_to_item: Mapping from semantic ID string to item metadata dict
            num_quantizers: Number of RQ-VAE quantizers
            num_beams: Number of beams for beam search
            system_prompt: System prompt for generation
            truncate_length: Maximum length for displayed field values
        """
        self.test_queries = test_queries
        self.tokenizer = tokenizer
        self.semantic_id_to_item = semantic_id_to_item
        self.num_quantizers = num_quantizers
        self.num_beams = num_beams
        self.system_prompt = system_prompt
        self.truncate_length = truncate_length
        self._generator = None

    def _truncate(self, value: str) -> str:
        """Truncate a string to the configured length."""
        value = str(value)
        if len(value) > self.truncate_length:
            return value[: self.truncate_length - 3] + "..."
        return value

    def _get_generator(self, model) -> SemanticIDGenerator:
        """Get or create a SemanticIDGenerator for the model."""
        if self._generator is None or self._generator.model is not model:
            self._generator = SemanticIDGenerator(
                model=model,
                tokenizer=self.tokenizer,
                num_quantizers=self.num_quantizers,
                system_prompt=self.system_prompt,
                # Append [REC] because these are plain queries
                append_rec_token=True,
            )
        return self._generator

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        """Run recommendation tests after each evaluation step."""
        if model is None or not self.test_queries:
            return

        print("\n--- Recommendation Test Examples ---")
        model.eval()
        generator = self._get_generator(model)

        # Collect results for wandb logging
        wandb_table_data = []

        for query in self.test_queries:
            print(f"\nQuery: {query}")

            results = generator.generate_beam(
                query,
                num_beams=self.num_beams,
                num_return_sequences=self.num_beams,
            )

            # Display top 2 results with clear ranking (valid or not)
            print("  Results:")
            max_to_show = min(2, len(results))

            for rank, (sem_id, score) in enumerate(results[:max_to_show], 1):
                is_valid = _is_valid_semantic_id(sem_id, self.num_quantizers)
                item = self.semantic_id_to_item.get(sem_id) if is_valid else None

                # Build status string
                if not is_valid:
                    status = "INVALID"
                elif item:
                    status = "IN CATALOGUE"
                else:
                    status = "NOT IN CATALOGUE"

                print(f"  ├─ Rank {rank}: {sem_id} (score: {score:.4f}) [{status}]")

                # Collect for wandb table
                item_title = item.get("title", "") if item else ""
                wandb_table_data.append(
                    [query, rank, sem_id, score, status, item_title]
                )

                if item:
                    print(f"  │     title: {self._truncate(item_title)}")

        print("\n" + "-" * 46 + "\n")

        # Log results table to wandb
        if wandb.run is not None and wandb_table_data:
            print(f"Logging {len(wandb_table_data)} rows to wandb table")
            print(f"First row sample: {wandb_table_data[0]}")
            # Add step to each row, ensuring all values are basic Python types
            table_data_with_step = []
            for row in wandb_table_data:
                query, rank, sem_id, score, status, title = row
                table_data_with_step.append(
                    [
                        int(state.global_step),
                        str(query),
                        int(rank),
                        str(sem_id),
                        float(score),
                        str(status),
                        str(title),
                    ]
                )

            # Reset random state before logging to work around wandb bug
            # https://github.com/wandb/wandb/issues/11112
            random_state = random.getstate()
            random.seed()

            table = wandb.Table(
                columns=[
                    "step",
                    "query",
                    "rank",
                    "semantic_id",
                    "score",
                    "status",
                    "title",
                ],
                data=table_data_with_step,
            )
            print(f"Table has {len(table.data)} rows")
            wandb.run.log(
                {f"recommendation_results/step_{state.global_step}": table},
                step=state.global_step,
            )

            # Restore random state
            random.setstate(random_state)


class WandbArtifactCallback(TrainerCallback):
    """
    Trainer callback to log the trained model as a W&B artifact.

    This callback logs the artifact at the end of training, ensuring
    the W&B run is still active when the artifact is uploaded.
    """

    def __init__(
        self,
        config: "FinetuneConfig",  # noqa: F821
        train_examples: int = 0,
        val_examples: int = 0,
    ):
        self.config = config
        self.artifact_name = config.wandb_artifact_name or f"llm-stage{config.stage}"
        self.train_examples = train_examples
        self.val_examples = val_examples
        self._logged = False

    def _find_best_checkpoint(self, output_dir: Path) -> Path | None:
        """
        Find the best checkpoint based on trainer_state.json.

        Returns the path to the best checkpoint directory, or None if not found.
        """
        import json

        # Look for trainer_state.json in checkpoints to find best model
        checkpoints = sorted(output_dir.glob("checkpoint-*"))
        if not checkpoints:
            return None

        # Check the latest checkpoint's trainer_state.json for best_model_checkpoint
        for checkpoint in reversed(checkpoints):
            trainer_state_path = checkpoint / "trainer_state.json"
            if trainer_state_path.exists():
                with open(trainer_state_path) as f:
                    trainer_state = json.load(f)
                best_checkpoint_path = trainer_state.get("best_model_checkpoint")
                if best_checkpoint_path:
                    best_path = Path(best_checkpoint_path)
                    if best_path.exists():
                        return best_path
                break

        # Fallback to latest checkpoint
        return checkpoints[-1]

    def on_train_end(self, args, state, control, **kwargs):
        """Log the model artifact at the end of training."""
        if self._logged:
            return

        if wandb.run is None:
            print("Warning: wandb.run is None, skipping artifact logging")
            return

        try:
            config = self.config
            output_dir = Path(config.output_dir)
            print(f"\n=== Logging model artifact: {self.artifact_name} ===")
            print(f"  Run ID: {wandb.run.id}")
            print(f"  Output dir: {output_dir}")

            metadata = asdict(config)
            metadata["train_examples"] = self.train_examples
            metadata["val_examples"] = self.val_examples

            artifact = wandb.Artifact(
                name=self.artifact_name,
                type="model",
                description=f"LLM Stage {config.stage}: "
                + ("Embedding training" if config.stage == 1 else "LoRA fine-tuning"),
                metadata=metadata,
            )

            # Only add the final model files, not checkpoint subdirectories
            # This ensures the artifact has config.json at the root level
            # which is required by Unsloth/transformers to load the model
            files_added = 0
            for file_path in output_dir.iterdir():
                if file_path.is_file():
                    artifact.add_file(str(file_path))
                    files_added += 1

            if files_added == 0:
                # Fallback: if no files at root, find the best checkpoint
                # based on eval_loss from trainer_state.json
                best_checkpoint = self._find_best_checkpoint(output_dir)
                if best_checkpoint:
                    print(f"  No files at root, using best checkpoint: {best_checkpoint}")
                    for file_path in best_checkpoint.iterdir():
                        if file_path.is_file():
                            artifact.add_file(str(file_path), name=file_path.name)
                            files_added += 1

            print(f"  Added {files_added} files to artifact")

            aliases = ["latest"]
            if config.stage == 2:
                aliases.append("best")

            logged_artifact = wandb.run.log_artifact(artifact, aliases=aliases)
            logged_artifact.wait()
            self._logged = True
            print(
                f"Artifact {self.artifact_name} logged successfully with aliases: {aliases}"
            )
        except Exception as e:
            print(f"Error logging artifact: {e}")
