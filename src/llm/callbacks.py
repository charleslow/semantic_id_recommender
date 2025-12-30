"""
Trainer callbacks for LLM fine-tuning evaluation.
"""

import re

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
    if not semantic_id.startswith("[SEM_START]") or not semantic_id.endswith("[SEM_END]"):
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
        logged_metrics["eval_semantic_id_target/exact_match_accuracy"] = exact_match_accuracy

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

        # Track metrics for logging
        num_valid = 0
        num_in_catalogue = 0

        for query in self.test_queries:
            print(f"\nQuery: {query}")

            results = generator.generate_beam(
                query,
                num_beams=self.num_beams,
                num_return_sequences=self.num_beams,
            )

            # Find first valid semantic ID
            valid_result = None
            for sem_id, score in results:
                if _is_valid_semantic_id(sem_id, self.num_quantizers):
                    valid_result = (sem_id, score)
                    break

            if valid_result is None:
                print("  No valid semantic ID generated")
                if results:
                    print(f"  Top result (invalid): {results[0][0]}")
                continue

            num_valid += 1
            sem_id, score = valid_result
            print(f"  Semantic ID: {sem_id} (score: {score:.4f})")

            # Look up item metadata
            item = self.semantic_id_to_item.get(sem_id)
            if item is None:
                print("  Item not found in catalogue")
                continue

            num_in_catalogue += 1
            print("  Item metadata:")
            for key, value in item.items():
                if key != "semantic_id":
                    print(f"    {key}: {self._truncate(value)}")

        print("\n" + "-" * 46 + "\n")

        # Log metrics to wandb
        num_queries = len(self.test_queries)
        rec_metrics = {
            "eval_recommendation/valid_id_rate": num_valid / num_queries if num_queries else 0.0,
            "eval_recommendation/catalogue_hit_rate": num_in_catalogue / num_queries if num_queries else 0.0,
            "eval_recommendation/num_queries": num_queries,
        }

        if wandb.run is not None:
            wandb.log(rec_metrics, step=state.global_step)
