"""Model loading and DistilGPT2 story continuation logic (Phase 2).

Separates Hugging Face / PyTorch inference logic from the Streamlit frontend.
DistilGPT2 weights are downloaded automatically on first use and cached locally.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_NAME = "distilgpt2"
DEFAULT_DEVICE = "cpu"

MIN_TEMPERATURE = 0.1
MAX_TEMPERATURE = 1.5
DEFAULT_TEMPERATURE = 0.8

DEFAULT_REPETITION_PENALTY = 1.15
DEFAULT_NO_REPEAT_NGRAM_SIZE = 3
DEFAULT_TOP_P = 0.92
MAX_MODEL_CONTEXT_TOKENS = 1024

LENGTH_PRESETS: dict[str, dict[str, Any]] = {
    "Short": {
        "label": "Short",
        "max_new_tokens": 50,
        "approx_words": "~50 tokens",
    },
    "Medium": {
        "label": "Medium",
        "max_new_tokens": 100,
        "approx_words": "~100 tokens",
    },
    "Long": {
        "label": "Long",
        "max_new_tokens": 180,
        "approx_words": "~180 tokens",
    },
}


def validate_story_prompt(prompt: str | None) -> tuple[bool, str]:
    """Validate that the user-provided story opening is a non-empty string.

    Args:
        prompt: Raw text input from the user.

    Returns:
        A tuple ``(is_valid, error_message)``. When valid, ``error_message`` is ``""``.
    """
    if prompt is None or not isinstance(prompt, str):
        return False, "Please enter a story opening or prompt before generating a continuation."

    if not prompt.strip():
        return False, "Your story opening cannot be empty. Please write a few words to begin."

    return True, ""


def validate_generation_params(length_option: str, temperature: float) -> tuple[str, float]:
    """Validate response length preset and creativity temperature parameters.

    Args:
        length_option: One of ``"Short"``, ``"Medium"``, or ``"Long"``.
        temperature: Creativity value in ``[0.1, 1.5]``.

    Returns:
        Normalized ``(length_option, temperature)`` tuple.

    Raises:
        ValueError: If ``length_option`` is unsupported or ``temperature`` is out of bounds.
    """
    if length_option not in LENGTH_PRESETS:
        valid_options = ", ".join(LENGTH_PRESETS.keys())
        raise ValueError(f"Unsupported response length '{length_option}'. Choose from: {valid_options}.")

    try:
        temp_value = float(temperature)
    except (TypeError, ValueError) as exc:
        raise ValueError("Creativity (temperature) must be a numeric value.") from exc

    if not (MIN_TEMPERATURE <= temp_value <= MAX_TEMPERATURE):
        raise ValueError(
            f"Creativity (temperature) must be between {MIN_TEMPERATURE} and {MAX_TEMPERATURE}."
        )

    return length_option, round(temp_value, 2)


def resolve_generation_config(
    length_option: str,
    temperature: float,
    *,
    max_new_tokens: int | None = None,
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
    no_repeat_ngram_size: int = DEFAULT_NO_REPEAT_NGRAM_SIZE,
    top_p: float = DEFAULT_TOP_P,
) -> dict[str, Any]:
    """Validate and resolve configurable decoding parameters for DistilGPT2."""
    normalized_length, normalized_temp = validate_generation_params(length_option, temperature)

    resolved_tokens = (
        LENGTH_PRESETS[normalized_length]["max_new_tokens"]
        if max_new_tokens is None
        else max_new_tokens
    )
    if not isinstance(resolved_tokens, int) or isinstance(resolved_tokens, bool) or resolved_tokens <= 0:
        raise ValueError("max_new_tokens must be a positive integer.")

    if repetition_penalty < 1.0:
        raise ValueError("repetition_penalty must be >= 1.0.")

    if not isinstance(no_repeat_ngram_size, int) or no_repeat_ngram_size < 0:
        raise ValueError("no_repeat_ngram_size must be a non-negative integer.")

    if not (0.0 < float(top_p) <= 1.0):
        raise ValueError("top_p must be in (0.0, 1.0].")

    return {
        "length_option": normalized_length,
        "temperature": normalized_temp,
        "max_new_tokens": resolved_tokens,
        "repetition_penalty": float(repetition_penalty),
        "no_repeat_ngram_size": no_repeat_ngram_size,
        "top_p": float(top_p),
    }


@lru_cache(maxsize=1)
def load_model(
    model_name: str = DEFAULT_MODEL_NAME,
    device: str = DEFAULT_DEVICE,
) -> tuple[Any, Any]:
    """Load and cache the pretrained causal LM tokenizer and model in eval mode.

    Downloads weights from Hugging Face on first run and reuses the local cache
    on subsequent runs. Cached per process to prevent reloading on UI reruns.

    Args:
        model_name: Hugging Face model identifier (defaults to ``"distilgpt2"``).
        device: Target PyTorch device (defaults to ``"cpu"``).

    Returns:
        Tuple of ``(tokenizer, model)``.

    Raises:
        RuntimeError: If loading the tokenizer or model fails.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
            model.config.pad_token_id = tokenizer.pad_token_id

        model.to(device)
        model.eval()
        return tokenizer, model
    except Exception as exc:
        raise RuntimeError(f"Failed to load model '{model_name}' on '{device}': {exc}") from exc


def clear_model_cache() -> None:
    """Clear the cached tokenizer and model instance."""
    load_model.cache_clear()


def tokenize_prompt(
    prompt: str,
    tokenizer: Any,
    max_input_tokens: int = MAX_MODEL_CONTEXT_TOKENS,
) -> dict[str, torch.Tensor]:
    """Tokenize a validated story opening into PyTorch tensors.

    Args:
        prompt: Cleaned non-empty story opening.
        tokenizer: Hugging Face tokenizer instance.
        max_input_tokens: Maximum context tokens retained for the prompt.

    Returns:
        Dictionary containing ``input_ids`` and ``attention_mask`` tensors.
    """
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
        padding=False,
    )
    if "attention_mask" not in encoded or encoded["attention_mask"] is None:
        encoded["attention_mask"] = torch.ones_like(encoded["input_ids"])
    return encoded


def join_prompt_and_continuation(prompt: str, continuation: str) -> str:
    """Combine user story opening and generated continuation with natural spacing."""
    if not continuation:
        return prompt
    if prompt.endswith((" ", "\n", "\t")):
        return f"{prompt}{continuation}"
    if continuation[0] in (",", ".", "!", "?", ";", ":", "'", '"', "”", ")"):
        return f"{prompt}{continuation}"
    return f"{prompt} {continuation}"


def generate_story_continuation(
    prompt: str,
    length_option: str = "Medium",
    temperature: float = DEFAULT_TEMPERATURE,
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    device: str = DEFAULT_DEVICE,
    max_new_tokens: int | None = None,
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
    no_repeat_ngram_size: int = DEFAULT_NO_REPEAT_NGRAM_SIZE,
    top_p: float = DEFAULT_TOP_P,
) -> dict[str, Any]:
    """Generate an AI story continuation using pretrained DistilGPT2 on CPU.

    Args:
        prompt: Story opening text provided by the user.
        length_option: Preset length (``"Short"``, ``"Medium"``, or ``"Long"``).
        temperature: Sampling creativity in ``[0.1, 1.5]``.
        model_name: Hugging Face model ID (default ``"distilgpt2"``).
        device: Inference device (default ``"cpu"``).
        max_new_tokens: Optional token limit override.
        repetition_penalty: Penalty >= 1.0 to discourage repetitive phrases.
        no_repeat_ngram_size: N-gram size blocked from repeating.
        top_p: Nucleus sampling probability threshold.

    Returns:
        Dictionary with separate ``prompt`` and ``continuation`` fields,
        ``full_story``, and generation metadata.

    Raises:
        ValueError: If the prompt is empty or generation parameters are invalid.
        RuntimeError: If model loading or inference fails.
    """
    is_valid, error_message = validate_story_prompt(prompt)
    if not is_valid:
        raise ValueError(error_message)

    config = resolve_generation_config(
        length_option=length_option,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        top_p=top_p,
    )
    cleaned_prompt = prompt.strip()

    tokenizer, model = load_model(model_name=model_name, device=device)

    try:
        max_input_tokens = max(1, MAX_MODEL_CONTEXT_TOKENS - config["max_new_tokens"])
        inputs = tokenize_prompt(cleaned_prompt, tokenizer, max_input_tokens=max_input_tokens)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        prompt_token_count = int(input_ids.shape[1])

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                min_new_tokens=min(15, config["max_new_tokens"]),
                max_new_tokens=config["max_new_tokens"],
                temperature=config["temperature"],
                do_sample=True,
                top_p=config["top_p"],
                repetition_penalty=config["repetition_penalty"],
                no_repeat_ngram_size=config["no_repeat_ngram_size"],
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        newly_generated_ids = output_ids[0][prompt_token_count:]
        continuation_text = tokenizer.decode(
            newly_generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()
        full_story = join_prompt_and_continuation(cleaned_prompt, continuation_text)

        return {
            "prompt": cleaned_prompt,
            "continuation": continuation_text,
            "full_story": full_story,
            "length_option": config["length_option"],
            "max_new_tokens": config["max_new_tokens"],
            "generated_token_count": int(len(newly_generated_ids)),
            "temperature": config["temperature"],
            "model_name": model_name,
            "device": device,
            "is_placeholder": False,
        }
    except Exception as exc:
        raise RuntimeError(f"Story generation failed: {exc}") from exc
