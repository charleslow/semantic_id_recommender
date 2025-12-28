"""
LLM inference for semantic ID generation.
"""

import re

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .data import DEFAULT_SYSTEM_PROMPT, REC_TOKEN


def extract_semantic_id(generated_text: str, eos_token: str | None = None) -> str:
    """
    Extract semantic ID from generated text.

    Args:
        generated_text: Raw generated text from the model
        eos_token: Optional EOS token to strip

    Returns:
        Extracted semantic ID string
    """
    if eos_token:
        generated_text = generated_text.replace(eos_token, "").strip()

    pattern = r"\[SEM_START\](?:\[SEM_\d+_\d+\])+\[SEM_END\]"
    match = re.search(pattern, generated_text)
    if match:
        return match.group(0)

    return generated_text.strip()


def load_finetuned_model(
    model_path: str,
    max_seq_length: int = 512,
    load_in_4bit: bool = True,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """
    Load fine-tuned model for inference.

    Args:
        model_path: Path to saved model or HuggingFace repo
        max_seq_length: Maximum sequence length
        load_in_4bit: Whether to load in 4-bit

    Returns:
        Model and tokenizer
    """
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=load_in_4bit,
    )

    # Set to inference mode
    FastLanguageModel.for_inference(model)

    return model, tokenizer


class SemanticIDGenerator:
    """Generator for semantic IDs."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        num_quantizers: int = 4,
        codebook_size: int = 256,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        append_rec_token: bool = True,
    ):
        """
        Initialize the generator.

        Args:
            model: Fine-tuned model
            tokenizer: Tokenizer with semantic ID tokens
            num_quantizers: Number of RQ-VAE quantizers
            codebook_size: Size of each codebook
            system_prompt: System prompt for generation
            append_rec_token: Whether to append [REC] token to queries.
                             Set to True for recommendation queries (default).
                             Set to False if query already contains [REC] token.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.system_prompt = system_prompt
        self.append_rec_token = append_rec_token

    def _build_prompt(self, query: str) -> str:
        """Build prompt using the tokenizer's chat template."""
        content = f"{query}{REC_TOKEN}" if self.append_rec_token else query
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": content},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def generate(
        self,
        query: str,
        temperature: float = 0.1,
    ) -> str:
        """
        Generate semantic ID for a query.

        Args:
            query: User query
            temperature: Sampling temperature

        Returns:
            Generated semantic ID string
        """
        prompt = self._build_prompt(query)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=32,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        return extract_semantic_id(generated, self.tokenizer.eos_token)

    def generate_beam(
        self,
        query: str,
        num_beams: int = 10,
        num_return_sequences: int = 10,
    ) -> list[tuple[str, float]]:
        """
        Generate multiple semantic IDs using beam search.

        Args:
            query: User query
            num_beams: Number of beams for beam search
            num_return_sequences: Number of sequences to return

        Returns:
            List of (semantic_id, score) tuples, sorted by score (highest first)
        """
        prompt = self._build_prompt(query)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.num_quantizers + 3,
            num_beams=num_beams,
            num_return_sequences=min(num_return_sequences, num_beams),
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            output_scores=True,
            return_dict_in_generate=True,
        )

        results = []
        sequences = outputs.sequences
        # Compute sequence scores (sum of log probs)
        scores = (
            outputs.sequences_scores
            if hasattr(outputs, "sequences_scores")
            else [0.0] * len(sequences)
        )

        for seq, score in zip(sequences, scores):
            generated = self.tokenizer.decode(seq, skip_special_tokens=False)
            semantic_id = extract_semantic_id(generated, self.tokenizer.eos_token)
            score_val = score.item() if hasattr(score, "item") else float(score)
            results.append((semantic_id, score_val))

        # Sort by score (highest first) and deduplicate
        seen = set()
        unique_results = []
        for sem_id, score in sorted(results, key=lambda x: x[1], reverse=True):
            if sem_id not in seen:
                seen.add(sem_id)
                unique_results.append((sem_id, score))

        return unique_results

    def __call__(self, query: str, **kwargs) -> str:
        """Shorthand for generate()."""
        return self.generate(query, **kwargs)
